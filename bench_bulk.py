"""Official-style 256-request offline throughput workload; no engine changes.

Default: separate prefill_first/mixed processes, identical workload, full KV,
decode CUDA Graph enabled. This is NOT an upstream nano-vLLM vs vLLM comparison.
Run --dry-run to inspect workload/configuration without importing torch.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from uuid import uuid4

from bench_baseline import (check_idle, command_output, file_sha256,
                            reset_prefix_cache, summarize, validate_model_files,
                            write_json)


ROOT = Path(__file__).resolve().parent


def make_workload(seed=0, num_requests=256):
    # Preserve upstream bench.py's draw order, including prompt token draws.
    # Drawing all lengths first would produce a DIFFERENT output workload.
    rng = random.Random(seed)
    prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(100, 1024))]
               for _ in range(num_requests)]
    lengths = [rng.randint(100, 1024) for _ in range(num_requests)]
    return {"prompt_token_ids": prompts, "output_lengths": lengths,
            "temperature": 0.6, "ignore_eos": True}


def workload_stats(workload):
    prompts, lengths = workload["prompt_token_ids"], workload["output_lengths"]
    return {"requests": len(prompts), "input_tokens": sum(map(len, prompts)),
            "output_tokens": sum(lengths),
            "max_total_length": max(len(p) + n for p, n in zip(prompts, lengths))}


def validate_outputs(outputs, expected):
    actual = [len(o["token_ids"]) for o in outputs]
    if actual != expected:
        raise RuntimeError(f"per-request output work mismatch: expected {expected}, got {actual}")
    return sum(actual)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--policy", choices=("both", "prefill_first", "mixed"), default="both")
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-blocks", type=int)
    parser.add_argument("--timeout-s", type=float, default=1800,
                        help="Per child process (init + all rounds), in --policy both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for key in ("num_requests", "warmup", "repeats", "max_num_seqs",
                "max_num_batched_tokens", "max_model_len"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.kv_cache_blocks is not None and args.kv_cache_blocks <= 0:
        parser.error("kv_cache_blocks must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout_s must be finite and positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu_memory_utilization must be between 0 and 1")
    # Match the current runner's capture bucket coverage.
    if not (args.max_num_seqs in (1, 2, 4, 8) or
            16 <= args.max_num_seqs <= 512 and args.max_num_seqs % 16 == 0):
        parser.error("Graph requires max_num_seqs=1/2/4/8 or a multiple of 16 <=512")
    if workload_stats(make_workload(args.seed, args.num_requests))["max_total_length"] > args.max_model_len:
        parser.error("a generated prompt plus output exceeds max_model_len")
    args.model = args.model.expanduser().resolve()
    if not args.dry_run and not (args.model / "config.json").is_file():
        parser.error("model must be an existing local model directory")
    return args


def child_command(args, policy, directory):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--model", str(args.model),
               "--policy", policy, "--output-dir", str(directory)]
    for key in ("num_requests", "seed", "warmup", "repeats", "max_num_seqs",
                "max_num_batched_tokens", "max_model_len", "gpu_memory_utilization", "kv_cache_blocks"):
        value = getattr(args, key)
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    return command


def compare_metadata(items):
    if len(items) != 2:
        raise RuntimeError("comparison requires exactly two completed policies")
    for field in ("gpu", "versions", "kv_pool", "workload_sha256", "model_dtype",
                  "model_metadata_sha256", "weight_file_sizes", "git_commit", "git_status"):
        if any(item[field] != items[0][field] for item in items[1:]):
            raise RuntimeError(f"comparison refused: {field} differs; retain results and inspect")
    for item, policy in zip(items, ("prefill_first", "mixed")):
        if item["status"] != "complete" or item["effective_policy"] != policy:
            raise RuntimeError("comparison refused: incomplete run or wrong policy")


def run_suite(args, directory, metadata):
    metadata["jobs"] = []
    for policy in ("prefill_first", "mixed"):
        command = child_command(args, policy, directory / policy)
        job = {"policy": policy, "command": command}
        metadata["jobs"].append(job)
        write_json(directory / "metadata.json", metadata)
        log_path = directory / f"{policy}.log"
        print(f"Running {policy}; live progress: tail -f {log_path}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout_s)
        job["returncode"] = result.returncode
        write_json(directory / "metadata.json", metadata)
        if result.returncode:
            raise RuntimeError(f"{policy} failed; see {log_path}; later runs skipped")
        summary = json.loads((directory / policy / "summary.json").read_text())
        print(f"{policy}: {summary['output_tokens_per_second_median']:.2f} output tok/s", flush=True)
    items = [json.loads((directory / p / "metadata.json").read_text())
             for p in ("prefill_first", "mixed")]
    compare_metadata(items)
    comparison = {p: json.loads((directory / p / "summary.json").read_text())
                  for p in ("prefill_first", "mixed")}
    comparison["mixed_throughput_change_percent"] = 100 * (
        comparison["mixed"]["output_tokens_per_second_median"] /
        comparison["prefill_first"]["output_tokens_per_second_median"] - 1)
    comparison["note"] = "Current repo policies, NOT unmodified upstream or production vLLM; no TTFT/ITL claim."
    write_json(directory / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


def run_gpu(args, directory, metadata):
    import torch
    import triton
    import flash_attn
    import transformers
    import nanovllm
    from nanovllm import LLM, SamplingParams

    validate_model_files(args.model)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the cloud nano-vllm environment")
    config = transformers.AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3" or config.vocab_size <= 10000:
        raise ValueError("requires dense Qwen3 with vocabulary covering token IDs 0..10000")
    if args.max_model_len > config.max_position_embeddings:
        raise ValueError("max_model_len exceeds model position limit")
    workload = make_workload(args.seed, args.num_requests)
    write_json(directory / "workload.json", workload)
    stats = workload_stats(workload)
    metadata.update({
        "workload": stats, "workload_sha256": file_sha256(directory / "workload.json"),
        "gpu": {"name": torch.cuda.get_device_name(0),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "capability": list(torch.cuda.get_device_capability(0))},
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "triton": triton.__version__, "flash_attn": flash_attn.__version__,
                     "transformers": transformers.__version__},
        "model_dtype": str(config.dtype), "nanovllm_import_path": nanovllm.__file__,
        "model_metadata_sha256": {p.name: file_sha256(p) for name in (
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json", "download_revision.txt")
            if (p := args.model / name).is_file()},
        "weight_file_sizes": {p.name: p.stat().st_size for p in sorted(args.model.glob("*.safetensors"))},
        "model_identity_note": "Metadata hashes and weight sizes are not full weight checksums; retain snapshot.",
    })
    write_json(directory / "metadata.json", metadata)
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    engine = LLM(str(args.model), enforce_eager=False, tensor_parallel_size=1,
                 max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                 max_num_batched_tokens=args.max_num_batched_tokens,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 scheduling_policy=args.policy, kv_allocation="full",
                 kv_cache_blocks=args.kv_cache_blocks)
    torch.cuda.synchronize()
    manager = engine.scheduler.block_manager
    metadata.update(engine_init_seconds=time.perf_counter() - start,
                    effective_policy=engine.scheduler.scheduling_policy,
                    kv_pool={"num_blocks": len(manager.blocks), "block_size": manager.block_size})
    write_json(directory / "metadata.json", metadata)
    # Unlike bench_baseline, a whole queued workload need not fit simultaneously.
    # Still reject a request that cannot finish even with exclusive use of the pool.
    if stats["max_total_length"] - 1 > len(manager.blocks) * manager.block_size:
        raise ValueError("a request cannot finish within this KV pool")
    sampling = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n)
                for n in workload["output_lengths"]]
    rows = []
    for index in range(args.warmup + args.repeats):
        phase = "warmup" if index < args.warmup else "measure"
        number = index + 1 if phase == "warmup" else index - args.warmup + 1
        torch.cuda.synchronize()
        reset_prefix_cache(engine)
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(workload["prompt_token_ids"], sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        count = validate_outputs(outputs, workload["output_lengths"])
        check_idle(engine)
        row = {"phase": phase, "round": number, "elapsed_seconds": elapsed,
               "requests": stats["requests"], "input_tokens": stats["input_tokens"],
               "output_tokens": count, "output_tokens_per_second": count / elapsed,
               "preemptions": engine.scheduler.preemptions,
               "recomputed_tokens": engine.scheduler.recomputed_tokens,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        rows.append(row)
        with (directory / "rounds.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        write_json(directory / f"{phase}-{number:02}-outputs.json", [o["token_ids"] for o in outputs])
        print(f"{phase} {number}: {elapsed:.3f}s, {count} output tokens, "
              f"{count / elapsed:.2f} tok/s, preemptions={row['preemptions']}", flush=True)
    write_json(directory / "summary.json", summarize(rows))
    # The engine registers exit with atexit; do not invoke it twice.


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps({"workload": workload_stats(make_workload(args.seed, args.num_requests)),
                          "arguments": {k: str(v) if isinstance(v, Path) else v
                                        for k, v in vars(args).items()},
                          "note": "No GPU imports or writes. Target model: Qwen3-0.6B; 256 is total requests, not active batch size."}, indent=2))
        return
    directory = (args.output_dir or ROOT / "results" / (
        "bulk-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "schema_version": 1,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "command": sys.argv, "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(),
                "git_commit": command_output(["git", "rev-parse", "HEAD"], ROOT),
                "git_status": command_output(["git", "status", "--porcelain"], ROOT),
                "nvidia_smi": command_output(["nvidia-smi"]),
                "cache_policy": "cold prefix metadata each round; weights/KV tensors/JIT remain resident",
                "timing_scope": "synchronized generate wall time incl enqueue, scheduling, CPU copies and detokenization; excludes init, cache reset, validation and file I/O",
                "limitations": "Upstream-style workload, NOT exact upstream protocol: full workload warmups/repeats, cold prefix metadata, synchronized perf_counter; current repo policies only. No vLLM comparison, online latency or maximum-throughput claim."}
    for name in ("bench_bulk.py", "bench_baseline.py", "bench.py"):
        (directory / name).write_bytes((ROOT / name).read_bytes())
    (directory / "tracked_changes.patch").write_text(command_output(["git", "diff", "HEAD", "--binary"], ROOT))
    write_json(directory / "metadata.json", metadata)
    print(f"Results: {directory}", flush=True)
    try:
        if args.policy == "both":
            run_suite(args, directory, metadata)
        else:
            run_gpu(args, directory, metadata)
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(directory / "metadata.json", metadata)
    print(f"Saved: {directory}", flush=True)


if __name__ == "__main__":
    main()
