"""Single-threaded open-loop arrival replay for the original nano-vllm engine.

Observe real appended tokens at step() return, not partial-prefill samples.
TTFT/ITL are CPU-observed engine timings, NOT client/network timings. Replay can
only admit between steps: report planned arrival, admission and admission lag.
Run with --only-request A for a paired short-request-only control.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from bench_baseline import (check_idle, command_output, file_sha256, make_prompts,
                            required_blocks, reset_prefix_cache, validate_model_files,
                            write_json)


def validate_trace(data):
    if not isinstance(data, list) or not data:
        raise ValueError("trace must be a non-empty JSON array")
    ids = set()
    for row in data:
        if not isinstance(row, dict) or set(row) != {"id", "arrival_s", "input_len", "output_len"}:
            raise ValueError("each request needs exactly id, arrival_s, input_len, output_len")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("request IDs must be non-empty unique strings")
        ids.add(row["id"])
        arrival = row["arrival_s"]
        if (type(arrival) not in (float, int) or not math.isfinite(arrival) or arrival < 0):
            raise ValueError("arrival_s must be finite and non-negative")
        for key in ("input_len", "output_len"):
            if type(row[key]) is not int or row[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
    # Stable ordering for equal arrival times; never move arrival based on progress.
    return sorted(data, key=lambda row: row["arrival_s"])


def build_workload(trace, valid_ids, seed, only_request=None):
    prompts = make_prompts(valid_ids, len(trace), max(r["input_len"] for r in trace), seed)
    full = [dict(row, prompt_token_ids=prompt[:row["input_len"]]) for row, prompt in zip(trace, prompts)]
    if only_request is None:
        return full
    selected = [row for row in full if row["id"] == only_request]
    if not selected:
        raise ValueError(f"unknown --only-request: {only_request}")
    # Filter AFTER generating prompts: control A uses exactly the mixed-run A input.
    return selected


def request_metrics(spec, admitted_s, token_times):
    if len(token_times) != spec["output_len"]:
        raise RuntimeError(f"{spec['id']}: expected {spec['output_len']} real output tokens, got {len(token_times)}")
    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    return {
        "request_id": spec["id"], "planned_arrival_s": spec["arrival_s"],
        "admitted_s": admitted_s, "admission_lag_s": admitted_s - spec["arrival_s"],
        "input_tokens": spec["input_len"], "output_tokens": len(token_times),
        "first_token_s": token_times[0], "last_token_s": token_times[-1],
        "ttft_from_planned_s": token_times[0] - spec["arrival_s"],
        "ttft_from_admission_s": token_times[0] - admitted_s,
        "completion_from_planned_s": token_times[-1] - spec["arrival_s"],
        "tpot_s": (token_times[-1] - token_times[0]) / len(gaps) if gaps else None,
        "itl_median_s": statistics.median(gaps) if gaps else None,
        "itl_max_s": max(gaps) if gaps else None,
    }


def replay(engine, workload, sampling_factory, *, clock=time.perf_counter,
           sleep=time.sleep, timeout_s=120):
    """Keep measurement outside the engine; original add_request/step are unchanged.

    add_request currently returns None. Capture its appended Sequence reference
    immediately, and retain it even when scheduler deque membership changes.
    Only an increase in num_completion_tokens produces an output event.
    """
    check_idle(engine)
    states = {}
    admissions, tokens, steps = [], [], []
    next_request = 0
    origin = clock()
    while next_request < len(workload) or not engine.is_finished():
        elapsed = clock() - origin
        if elapsed > timeout_s:
            raise TimeoutError("replay exceeded timeout between steps; an in-flight GPU call cannot be interrupted here")
        # Fixed cutoff for each iteration: replay is single-threaded, not a server.
        while next_request < len(workload) and workload[next_request]["arrival_s"] <= elapsed:
            spec = workload[next_request]
            generating = [key for key, state in states.items()
                          if state["seq"].num_completion_tokens > 0 and not state["seq"].is_finished]
            pending = [key for key, state in states.items() if not state["seq"].is_finished]
            before = len(engine.scheduler.waiting)
            engine.add_request(spec["prompt_token_ids"], sampling_factory(spec["output_len"]))
            if len(engine.scheduler.waiting) != before + 1:
                raise RuntimeError("add_request contract changed: expected one appended waiting Sequence")
            seq = engine.scheduler.waiting[-1]
            admitted_s = clock() - origin
            states[spec["id"]] = dict(spec=spec, seq=seq, admitted_s=admitted_s, times=[], observed=0)
            admissions.append({"request_id": spec["id"], "planned_arrival_s": spec["arrival_s"],
                               "admitted_s": admitted_s,
                               "admission_lag_s": admitted_s - spec["arrival_s"],
                               "pending_request_ids": pending, "generating_request_ids": generating})
            next_request += 1
        if engine.is_finished():
            if next_request < len(workload):
                sleep(max(0, min(workload[next_request]["arrival_s"] - (clock() - origin), 0.01)))
            continue

        waiting_before = len(engine.scheduler.waiting)
        running_before = len(engine.scheduler.running)
        start_s = clock() - origin
        _, signed_tokens = engine.step()
        end_s = clock() - origin
        if signed_tokens == 0:
            raise RuntimeError("unknown step phase: update observer if engine.step API changes")
        step_index = len(steps)
        emitted = 0
        for key, state in states.items():
            seq = state["seq"]
            count = seq.num_completion_tokens
            delta = count - state["observed"]
            if delta not in (0, 1):
                raise RuntimeError("observer expects at most one new token per request per step")
            if delta:
                state["times"].append(end_s)
                tokens.append({"request_id": key, "token_index": count,
                               "token_id": seq.last_token, "observed_s": end_s, "step": step_index})
                emitted += 1
            state["observed"] = count
        steps.append({"step": step_index, "phase": "prefill" if signed_tokens > 0 else "decode",
                      "start_s": start_s, "end_s": end_s, "wall_seconds": end_s - start_s,
                      "scheduled_input_tokens": abs(signed_tokens), "new_output_tokens": emitted,
                      "waiting_before": waiting_before, "running_before": running_before,
                      "waiting_after": len(engine.scheduler.waiting),
                      "running_after": len(engine.scheduler.running),
                      "free_blocks_after": len(engine.scheduler.block_manager.free_block_ids)})
    wall_seconds = clock() - origin
    if wall_seconds > timeout_s:
        raise TimeoutError("last step exceeded replay timeout")
    check_idle(engine)
    requests = [request_metrics(state["spec"], state["admitted_s"], state["times"])
                for state in states.values()]
    return {"wall_seconds": wall_seconds, "requests": requests, "admissions": admissions,
            "tokens": tokens, "steps": steps,
            "output_tokens_per_second": len(tokens) / wall_seconds}


def summarize_rounds(rounds):
    measured = [r for r in rounds if r["phase"] == "measure"]
    if not measured:
        raise ValueError("no measured rounds")
    request_ids = [r["request_id"] for r in measured[0]["requests"]]
    summary = {"measured_rounds": len(measured), "requests": {},
               "note": "Per-request medians across repeats of ONE trace; not population p95/p99. Throughput includes arrival gaps and observer overhead."}
    for key in request_ids:
        rows = [next(r for r in run["requests"] if r["request_id"] == key) for run in measured]
        summary["requests"][key] = {
            field + "_median": statistics.median(r[field] for r in rows) if rows[0][field] is not None else None
            for field in ("ttft_from_planned_s", "ttft_from_admission_s", "admission_lag_s", "tpot_s", "itl_max_s")
        }
        # The interruption experiment is valid only if A was still generating at B's admission.
        summary["requests"][key]["rounds_admitted_during_other_generation"] = sum(
            bool(next(a for a in run["admissions"] if a["request_id"] == key)["generating_request_ids"])
            for run in measured
        )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trace", type=Path, default=Path(__file__).parent / "benchmarks/traces/prefill_interrupt.json")
    parser.add_argument("--only-request", help="Run one request from the same full trace as a control")
    parser.add_argument("--execution", choices=("eager", "graph"), default="graph")
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4352)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    for name in ("max_num_seqs", "max_num_batched_tokens", "max_model_len", "warmup", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or not 0 <= args.seed < 2**63:
        parser.error("invalid memory utilization or seed")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout-s must be finite and positive")
    if args.execution == "graph" and not (args.max_num_seqs in (1, 2, 4, 8)
            or 16 <= args.max_num_seqs <= 512 and args.max_num_seqs % 16 == 0):
        parser.error("current Graph runner requires max_num_seqs=1/2/4/8 or a multiple of 16 <=512")
    args.model = args.model.expanduser().resolve()
    args.trace = args.trace.expanduser().resolve()
    if not (args.model / "config.json").is_file():
        parser.error("model must be a local directory with config.json")
    return args


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args, directory, metadata):
    import torch
    import triton
    import flash_attn
    import transformers
    import nanovllm
    from nanovllm import LLM, SamplingParams

    trace = validate_trace(json.loads(args.trace.read_text()))
    selected = [r for r in trace if args.only_request is None or r["id"] == args.only_request]
    if not selected:
        raise ValueError("--only-request not found in trace")
    if any(r["input_len"] + r["output_len"] > args.max_model_len for r in selected):
        raise ValueError("a request's input + output exceeds max-model-len")
    if max(r["arrival_s"] for r in selected) >= args.timeout_s:
        raise ValueError("arrival time must be less than timeout-s")
    validate_model_files(args.model)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    config = transformers.AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3" or args.max_model_len > config.max_position_embeddings:
        raise ValueError("require dense Qwen3 and max-model-len within its position limit")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    specials = set(tokenizer.all_special_ids)
    valid_ids = [i for i in tokenizer.get_vocab().values() if 0 <= i < config.vocab_size and i not in specials]
    workload = build_workload(trace, valid_ids, args.seed, args.only_request)
    write_json(directory / "trace.json", trace)
    write_json(directory / "workload.json", workload)
    metadata.update({
        "workload_sha256": file_sha256(directory / "workload.json"),
        "trace_sha256": file_sha256(directory / "trace.json"),
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
                     "flash_attn": flash_attn.__version__, "transformers": transformers.__version__},
        "nanovllm_import_path": nanovllm.__file__, "model_dtype": str(config.dtype),
        "model_metadata_sha256": {p.name: file_sha256(p) for p in (
            args.model / "config.json", args.model / "tokenizer.json", args.model / "tokenizer_config.json",
            args.model / "model.safetensors.index.json") if p.is_file()},
        "weight_file_sizes": {p.name: p.stat().st_size for p in sorted(args.model.glob("*.safetensors"))},
        "model_identity_note": "Use the same model snapshot; config hashes and file sizes do not fully identify weights.",
        "gpu": {"name": torch.cuda.get_device_name(0), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
    })
    write_json(directory / "metadata.json", metadata)
    torch.manual_seed(args.seed)
    print("Loading engine / warming up...", flush=True)
    start = time.perf_counter()
    engine = LLM(str(args.model), tensor_parallel_size=1, enforce_eager=args.execution == "eager",
                 max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                 max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    torch.cuda.synchronize()
    metadata["engine_init_seconds"] = time.perf_counter() - start
    manager = engine.scheduler.block_manager
    metadata["kv_pool"] = {"num_blocks": len(manager.blocks), "block_size": manager.block_size}
    write_json(directory / "metadata.json", metadata)
    needed = sum(required_blocks(1, r["input_len"], r["output_len"], manager.block_size) for r in workload)
    if needed > len(manager.blocks):
        raise ValueError(f"need {needed} blocks for conservative no-pressure replay, only {len(manager.blocks)} available")
    rounds = []
    for i in range(args.warmup + args.repeats):
        phase = "warmup" if i < args.warmup else "measure"
        number = i + 1 if phase == "warmup" else i - args.warmup + 1
        torch.cuda.synchronize()
        reset_prefix_cache(engine)
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        result = replay(engine, workload, lambda n: SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n),
                        timeout_s=args.timeout_s)
        result.update(phase=phase, round=number)
        rounds.append(result)
        round_dir = directory / f"{phase}-{number:02d}"
        round_dir.mkdir()
        write_json(round_dir / "timeline.json", result)
        for table in ("requests", "steps", "tokens"):
            save_csv(round_dir / f"{table}.csv", result[table])
        print(f"{phase} {number}: {result['wall_seconds']:.3f}s", flush=True)
        for row in result["requests"]:
            print(f"  {row['request_id']}: TTFT(planned)={row['ttft_from_planned_s']:.4f}s, "
                  f"admission_lag={row['admission_lag_s']:.4f}s, max_ITL={row['itl_max_s']}", flush=True)
        for admission in result["admissions"]:
            print(f"  admitted {admission['request_id']}; already generating: {admission['generating_request_ids']}", flush=True)
        if phase == "measure":
            write_json(directory / "summary.json", summarize_rounds(rounds))
    print(json.dumps(summarize_rounds(rounds), indent=2), flush=True)


def main(argv=None):
    args = parse_args(argv)
    directory = (args.output_dir or Path("results") / (
        "timeline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    metadata = {
        "status": "running", "schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": sys.version, "executable": sys.executable, "command": sys.argv,
        "git_commit": command_output(["git", "rev-parse", "HEAD"], root),
        "git_status": command_output(["git", "status", "--porcelain"], root),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "cache_policy": "cold metadata each round; compiled kernels and GPU KV tensor retained",
        "sampling": {"temperature": 0.6, "ignore_eos": True},
        "clock_semantics": "perf_counter relative to round start; fixed planned arrivals, actual admission only between steps; outputs observed at step return (upper bound on CPU availability)",
        "timing_scope": "step wall time includes schedule/runner/postprocess, NOT pure GPU; no extra per-step CUDA sync; original runner returns CPU token IDs; no text detokenization",
        "limitations": "synthetic workload, no server/network; observer overhead included; no p95/p99 inference from few requests; no KV pressure; warmup may not cover all arrival-dependent batch shapes; phase reporting uses original signed-token step API and must change with mixed batches",
    }
    for name in ("bench_timeline.py", "bench_baseline.py"):
        (directory / name).write_bytes((root / name).read_bytes())
    (directory / "tracked_changes.patch").write_text(command_output(["git", "diff", "HEAD", "--binary"], root))
    write_json(directory / "metadata.json", metadata)
    print(f"Results: {directory}", flush=True)
    try:
        run(args, directory, metadata)
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "metadata.json", metadata)
        raise
    metadata["status"] = "complete"
    write_json(directory / "metadata.json", metadata)
    print(f"Saved: {directory}", flush=True)


if __name__ == "__main__":
    main()
