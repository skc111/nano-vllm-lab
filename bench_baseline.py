"""Offline, fixed-work benchmark. Run --help without importing the GPU stack.

Every round starts with cold prefix-cache metadata, but reuses the loaded model,
KV tensor and compiled kernels. This measures generate() wall time, NOT TTFT/ITL.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--num-requests", type=int, default=2)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--scheduling-policy", choices=("prefill_first", "interleave", "mixed"), default="prefill_first")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="New directory; never overwrite a run")
    args = parser.parse_args(argv)
    for name in ("num_requests", "input_len", "output_len", "warmup", "repeats",
                 "max_num_batched_tokens", "max_model_len"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_num_seqs is None:
        args.max_num_seqs = args.num_requests
    if args.max_num_seqs <= 0 or args.num_requests > args.max_num_seqs:
        parser.error("require 0 < num_requests <= max_num_seqs for this baseline")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input_len + output_len must not exceed max_model_len")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu_memory_utilization must be between 0 and 1 (exclusive)")
    if not math.isfinite(args.temperature) or args.temperature <= 1e-10:
        parser.error("temperature must be finite and > 1e-10; engine has no greedy path")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    # Avoid the existing runner's incomplete bucket coverage (e.g. max_num_seqs=17).
    if args.execution == "graph" and not (
        args.max_num_seqs in (1, 2, 4, 8)
        or 16 <= args.max_num_seqs <= 512 and args.max_num_seqs % 16 == 0
    ):
        parser.error("current Graph runner requires max_num_seqs=1/2/4/8 or a multiple of 16 <=512")
    args.model = args.model.expanduser().resolve()
    if not (args.model / "config.json").is_file():
        parser.error(f"missing local model config: {args.model / 'config.json'}")
    return args


def make_prompts(valid_token_ids, num_requests, input_len, seed):
    """Use real tokenizer IDs; distinct first tokens prevent shared batch prefixes."""
    vocab = sorted(set(valid_token_ids))
    if input_len <= 0 or not 0 < num_requests <= len(vocab):
        raise ValueError("need positive input_len and enough distinct valid token IDs")
    rng = random.Random(seed)
    first_tokens = rng.sample(vocab, num_requests)
    return [[first] + rng.choices(vocab, k=input_len - 1) for first in first_tokens]


def validate_model_files(model):
    """The original loader can silently load nothing; reject missing downloads."""
    weights = list(model.glob("*.safetensors"))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError("model needs non-empty safetensors weights")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing = [name for name in shards if not (model / name).is_file()]
        if missing:
            raise ValueError(f"missing model shards: {sorted(missing)}")


def check_idle(engine):
    manager = engine.scheduler.block_manager
    if not engine.is_finished() or manager.used_block_ids:
        raise RuntimeError("benchmark requires an idle engine with no held KV blocks")
    if (any(block.ref_count for block in manager.blocks)
            or set(manager.free_block_ids) != set(range(len(manager.blocks)))
            or len(manager.free_block_ids) != len(manager.blocks)):
        raise RuntimeError("KV reference/free-list invariant failed after generation")


def reset_prefix_cache(engine):
    """Harness-only reset, outside timing. GPU pool/Graph addresses remain unchanged.

    Old GPU bytes need not be zeroed: no hash or block table can reference them.
    This intentionally depends on the original BlockManager constructor.
    """
    check_idle(engine)
    old = engine.scheduler.block_manager
    engine.scheduler.block_manager = type(old)(len(old.blocks), old.block_size)
    for name in ("preemptions", "evicted_cached_tokens", "recomputed_tokens"):
        if hasattr(engine.scheduler, name):
            setattr(engine.scheduler, name, 0)
    if hasattr(engine.scheduler, "_computed_high_water"):
        engine.scheduler._computed_high_water.clear()
    if hasattr(engine.scheduler, "_last_was_prefill"):
        engine.scheduler._last_was_prefill = False


def required_blocks(num_requests, input_len, output_len, block_size):
    # Conservative: reserve room for even the final sampled token (not yet cached).
    return num_requests * ((input_len + output_len + block_size - 1) // block_size)


def validate_outputs(outputs, num_requests, output_len):
    if len(outputs) != num_requests:
        raise RuntimeError(f"expected {num_requests} results, got {len(outputs)}")
    lengths = [len(output["token_ids"]) for output in outputs]
    if lengths != [output_len] * num_requests:
        raise RuntimeError(f"fixed output work violated: expected {output_len}, got {lengths}")
    return sum(lengths)


def summarize(rows):
    rows = [row for row in rows if row["phase"] == "measure"]
    if not rows:
        raise ValueError("no measured rounds")
    rates = [row["output_tokens_per_second"] for row in rows]
    times = [row["elapsed_seconds"] for row in rows]
    return {
        "measured_rounds": len(rows),
        "elapsed_seconds_median": statistics.median(times),
        "output_tokens_per_second_median": statistics.median(rates),
        "output_tokens_per_second_min": min(rates),
        "output_tokens_per_second_max": max(rates),
        "output_tokens_per_second_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "aggregate_output_tokens_per_second": sum(r["output_tokens"] for r in rows) / sum(times),
    }


def command_output(command, cwd=None):
    try:
        return subprocess.check_output(command, cwd=cwd, stderr=subprocess.STDOUT,
                                       timeout=15, text=True).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, output_dir, metadata):
    # Keep CLI help and CPU helper tests independent of torch/FlashAttention.
    import torch
    import triton
    import flash_attn
    import transformers
    import nanovllm
    from nanovllm import LLM, SamplingParams

    validate_model_files(args.model)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run GPU measurements in the activated GPU environment")
    config = transformers.AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError("this engine only implements dense Qwen3")
    if args.max_model_len > config.max_position_embeddings:
        raise ValueError("max_model_len exceeds the model's position limit")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    specials = set(tokenizer.all_special_ids)
    valid_ids = [i for i in tokenizer.get_vocab().values()
                 if 0 <= i < config.vocab_size and i not in specials]
    prompts = make_prompts(valid_ids, args.num_requests, args.input_len, args.seed)
    workload = {"prompt_token_ids": prompts, "output_len": args.output_len,
                "ignore_eos": True, "temperature": args.temperature,
                "description": "synthetic token IDs, no chat template; not a quality evaluation"}
    write_json(output_dir / "workload.json", workload)
    model_files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors.index.json", "download_revision.txt"):
        path = args.model / name
        if path.is_file():
            model_files[name] = file_sha256(path)
    metadata.update({
        "versions": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                     "triton": triton.__version__, "flash_attn": flash_attn.__version__,
                     "transformers": transformers.__version__},
        "gpu": {"name": torch.cuda.get_device_name(0),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "capability": list(torch.cuda.get_device_capability(0))},
        "model_dtype": str(config.dtype),
        "nanovllm_import_path": nanovllm.__file__,
        "model_metadata_sha256": model_files,
        "weight_file_sizes": {p.name: p.stat().st_size for p in sorted(args.model.glob("*.safetensors"))},
        "model_revision": ((args.model / "download_revision.txt").read_text().strip()
                           if (args.model / "download_revision.txt").is_file() else None),
        "model_identity_note": "Metadata hashes and weight sizes are NOT full weight checksums; preserve the model snapshot.",
        "workload_sha256": file_sha256(output_dir / "workload.json"),
    })
    write_json(output_dir / "metadata.json", metadata)
    torch.manual_seed(args.seed)
    print("Loading model / engine warmup...", flush=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    engine = LLM(str(args.model), tensor_parallel_size=1,
                 enforce_eager=args.execution == "eager",
                 max_num_seqs=args.max_num_seqs,
                 max_num_batched_tokens=args.max_num_batched_tokens,
                 max_model_len=args.max_model_len,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 scheduling_policy=args.scheduling_policy)
    torch.cuda.synchronize()
    metadata["engine_init_seconds"] = time.perf_counter() - start
    metadata["effective_scheduling_policy"] = engine.scheduler.scheduling_policy
    manager = engine.scheduler.block_manager
    metadata["kv_pool"] = {"num_blocks": len(manager.blocks), "block_size": manager.block_size}
    write_json(output_dir / "metadata.json", metadata)
    needed = required_blocks(args.num_requests, args.input_len, args.output_len, manager.block_size)
    if needed > len(manager.blocks):
        raise ValueError(f"baseline needs conservative capacity {needed} blocks, pool has {len(manager.blocks)}; "
                         "reduce requests/lengths. KV-pressure experiments are outside this first benchmark.")
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.output_len, ignore_eos=True)
    rows = []
    for index in range(args.warmup + args.repeats):
        phase = "warmup" if index < args.warmup else "measure"
        round_index = index + 1 if phase == "warmup" else index - args.warmup + 1
        torch.cuda.synchronize()
        reset_prefix_cache(engine)
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        output_tokens = validate_outputs(outputs, args.num_requests, args.output_len)
        check_idle(engine)
        row = {
            "phase": phase, "round": round_index, "elapsed_seconds": elapsed,
            "requests": args.num_requests, "input_tokens": args.num_requests * args.input_len,
            "output_tokens": output_tokens, "output_tokens_per_second": output_tokens / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        rows.append(row)
        # I/O and validation are outside the measured generate() interval.
        with (output_dir / "rounds.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output_dir / f"{phase}-{round_index:02d}-outputs.json",
                   [o["token_ids"] for o in outputs])
        print(f"{phase} {round_index}: {elapsed:.3f}s, {output_tokens} output tokens, "
              f"{row['output_tokens_per_second']:.2f} output tok/s", flush=True)
    write_json(output_dir / "summary.json", summarize(rows))
    print(json.dumps(summarize(rows), indent=2), flush=True)
    # LLMEngine already registers exit with atexit; calling exit again is not safe.


def main(argv=None):
    args = parse_args(argv)
    output_dir = (args.output_dir or Path("results") / (
        "baseline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results: {output_dir}", flush=True)
    root = Path(__file__).resolve().parent
    metadata = {
        "status": "running", "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "command": sys.argv, "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"], root),
        "git_status": command_output(["git", "status", "--porcelain"], root),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "cache_policy": "cold prefix-cache metadata before EACH round; model/KV tensor/JIT remain resident",
        "timing_scope": "synchronized generate() wall time including enqueue, scheduler, GPU work, result copy and detokenization; excludes init, cache reset and file I/O",
        "limitations": "offline simultaneous arrival, synthetic fixed lengths, single GPU; no TTFT/ITL, no online replay or profiler; warmup does not prove absence of all later compilation",
    }
    (output_dir / "benchmark_source.py").write_bytes(Path(__file__).read_bytes())
    (output_dir / "tracked_changes.patch").write_text(command_output(["git", "diff", "HEAD", "--binary"], root))
    write_json(output_dir / "metadata.json", metadata)
    try:
        run(args, output_dir, metadata)
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output_dir / "metadata.json", metadata)
        raise
    metadata["status"] = "complete"
    write_json(output_dir / "metadata.json", metadata)
    print(f"Saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
