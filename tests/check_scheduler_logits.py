"""Optional GPU check; NOT a benchmark or a public greedy-sampling implementation.

Run once per policy with the same model/execution mode. --compare loads a trusted
local artifact from the other run, checks controlled argmax tokens and all retained
logit rows. Default max-num-seqs=1 isolates scheduling from batch-shape changes.
The optional max-num-seqs=2 case also probes BF16 shape sensitivity. CPU copies
intentionally synchronize; do not use these timings or silently widen tolerances.

Matched-history control (run in separate processes, using the same model):
  --scheduling-policy prefill_first --max-num-seqs 2 --reference-plan --output ref.pt
  --scheduling-policy interleave --max-num-seqs 2 --output actual.pt
      --compare ref.pt --require-matched-steps
Supply --model to both commands; add --execution eager to BOTH for the eager check.
The reference plan bypasses scheduling decisions but reuses allocator/postprocess
and model code. It validates this fixture, not all model numerics or workloads.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ReferencePlan:
    """Hand-written schedule for THIS fixture only; never calls Scheduler.schedule.

    It shares the original allocator/postprocess/model, so it is a scheduling
    differential check, not an independent oracle for attention or model math.
    Each entry specifies phase and (request, expected cached length, query length).
    """

    steps = (
        (True, (("A", 0, 128),)),
        (True, (("A", 128, 127),)),
        (False, (("A", 255, 1),)),
        (True, (("B", 0, 128),)),
        (False, (("A", 256, 1),)),
        (True, (("B", 128, 128),)),
        (False, (("A", 257, 1),)),
        (True, (("B", 256, 1),)),
        (False, (("A", 258, 1), ("B", 257, 1))),
        (False, (("A", 259, 1), ("B", 258, 1))),
        (False, (("A", 260, 1), ("B", 259, 1))),
        (False, (("A", 261, 1),)),
    )

    def __init__(self, scheduler, labels):
        self.scheduler, self.labels, self.index = scheduler, labels, 0

    def __call__(self):
        from nanovllm.engine.sequence import SequenceStatus

        if self.index >= len(self.steps):
            raise AssertionError("reference plan exhausted")
        prefill, specifications = self.steps[self.index]
        scheduler = self.scheduler
        manager = scheduler.block_manager
        active = {self.labels[s.seq_id]: s for s in (*scheduler.waiting, *scheduler.running)}
        selected = []
        for label, cached, length in specifications:
            seq = active[label]
            assert seq.num_cached_tokens == cached, (label, cached, seq.num_cached_tokens)
            if prefill:
                assert seq in scheduler.waiting
                if not seq.block_table:
                    assert manager.can_allocate(seq) == 0  # fixture has no shared prefix
                    manager.allocate(seq, 0)
                assert cached + length <= len(seq)
                if cached + length == len(seq):
                    scheduler.waiting.remove(seq)
                    scheduler.running.append(seq)
                    seq.status = SequenceStatus.RUNNING
            else:
                assert seq in scheduler.running and cached == len(seq) - 1
                assert manager.can_append(seq)  # fixture is not a preemption test
                manager.may_append(seq)
                seq.is_prefill = False
            seq.num_scheduled_tokens = length
            selected.append(seq)
        self.index += 1
        return selected, prefill


def describe_step(seqs, prefill, labels):
    return {"prefill": prefill, "requests": [
        {"id": labels[s.seq_id], "cached": s.num_cached_tokens,
         "scheduled": s.num_scheduled_tokens, "blocks": list(s.block_table),
         "input_ids": list(s.token_ids[s.num_cached_tokens:
                                      s.num_cached_tokens + s.num_scheduled_tokens])}
        for s in seqs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--scheduling-policy", choices=("prefill_first", "interleave"), required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), default="graph")
    parser.add_argument("--max-num-seqs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--reference-plan", action="store_true",
                        help="Use the fixed independent scheduling fixture (prefill_first, max-num-seqs=2 only)")
    parser.add_argument("--require-matched-steps", action="store_true",
                        help="Require identical execution descriptors before comparing logits")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    if args.reference_plan and (args.scheduling_policy != "prefill_first" or args.max_num_seqs != 2):
        parser.error("reference-plan requires prefill_first and max-num-seqs=2")
    if args.require_matched_steps and not args.compare:
        parser.error("require-matched-steps requires compare")

    import torch
    from nanovllm import LLM, SamplingParams
    from bench_baseline import check_idle, file_sha256

    engine = LLM(str(args.model.expanduser().resolve()), scheduling_policy=args.scheduling_policy,
                 tensor_parallel_size=1, enforce_eager=args.execution == "eager",
                 max_num_seqs=args.max_num_seqs, max_num_batched_tokens=128, max_model_len=512,
                 gpu_memory_utilization=args.gpu_memory_utilization)
    runner = engine.model_runner
    captured, rows, labels, row_shapes = {}, {}, {}, {}
    steps = []
    reference_plan = ReferencePlan(engine.scheduler, labels) if args.reference_plan else None
    if reference_plan is not None:
        engine.scheduler.schedule = reference_plan
    original_model = runner.run_model
    original_run = runner.run

    def capture_logits(*arguments):
        logits = original_model(*arguments)
        captured["logits"] = logits.detach().float().cpu()
        return logits

    def observe_run(seqs, is_prefill):
        steps.append(describe_step(seqs, is_prefill, labels))
        output = original_run(seqs, is_prefill)
        for seq, row in zip(seqs, captured["logits"]):
            if seq.num_cached_tokens + seq.num_scheduled_tokens >= len(seq):
                key = f"{labels[seq.seq_id]}:{seq.num_completion_tokens}"
                if key in rows:
                    raise AssertionError(f"duplicate logit position: {key}")
                rows[key] = row.clone()
                row_shapes[key] = {"prefill": is_prefill, "batch_size": len(seqs)}
        return output

    runner.run_model = capture_logits
    runner.run = observe_run
    # Test-only deterministic sampling isolates schedule order from global RNG.
    runner.sampler = lambda logits, temperatures: logits.argmax(dim=-1)
    engine.add_request([42 + i % 13 for i in range(255)], SamplingParams(max_tokens=8, ignore_eos=True))
    a = engine.scheduler.waiting[-1]
    labels[a.seq_id] = "A"
    while a.num_completion_tokens < 2:
        engine.step()
    engine.add_request([83 + i % 11 for i in range(257)], SamplingParams(max_tokens=4, ignore_eos=True))
    b = engine.scheduler.waiting[-1]
    labels[b.seq_id] = "B"
    for _ in range(100):
        if engine.is_finished():
            break
        engine.step()
    else:
        raise AssertionError("no progress within bounded steps")
    check_idle(engine)
    assert (a.num_completion_tokens, b.num_completion_tokens) == (8, 4)
    assert len(rows) == 12
    if reference_plan is not None:
        assert reference_plan.index == len(reference_plan.steps)
    result = {"policy": args.scheduling_policy, "execution": args.execution, "max_num_seqs": args.max_num_seqs,
              "model_config_sha256": file_sha256(args.model.expanduser() / "config.json"),
              "tokens": {"A": a.completion_token_ids, "B": b.completion_token_ids}, "logits": rows,
              "row_shapes": row_shapes, "steps": steps, "reference_plan": args.reference_plan}
    # Save even failed comparisons for diagnosis, not only successful examples.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"Saved trusted local comparison artifact: {args.output}")
    if args.compare:
        reference = torch.load(args.compare, map_location="cpu", weights_only=True)
        assert result["model_config_sha256"] == reference["model_config_sha256"]
        assert result["execution"] == reference["execution"]
        if args.require_matched_steps:
            assert result["max_num_seqs"] == reference["max_num_seqs"]
            assert steps == reference["steps"], "execution history differs (shape/input/KV mapping)"
        assert result["tokens"] == reference["tokens"], "controlled argmax outputs differ"
        assert rows.keys() == reference["logits"].keys()
        maximum_error = 0.0
        for key in rows:
            expected = reference["logits"][key]
            # Fixed BEFORE running the comparison; BF16 tolerance, not bitwise equivalence.
            torch.testing.assert_close(rows[key], expected, rtol=1e-2, atol=1e-1,
                                       msg=lambda message: f"logit row {key}: {message}")
            maximum_error = max(maximum_error, (rows[key] - expected).abs().max().item())
        print(f"PASS: 12 logit rows (rtol=0.01, atol=0.1), identical controlled tokens; max_abs_error={maximum_error}")


if __name__ == "__main__":
    main()
