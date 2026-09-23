"""CPU-only harness tests: python3 -m unittest discover -s tests -v"""

import contextlib
import io
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bench_baseline import (check_idle, main, make_prompts, parse_args, required_blocks,
                            reset_prefix_cache, summarize, validate_model_files,
                            validate_outputs, write_json)


class FakeManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks = [SimpleNamespace(ref_count=0) for _ in range(num_blocks)]
        self.used_block_ids = set()
        self.free_block_ids = deque(range(num_blocks))
        self.hash_to_block_id = {}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model = Path(self.temporary.name)
        (self.model / "config.json").write_text("{}")

    def parse(self, *args):
        return parse_args(["--model", str(self.model), *args])

    def assert_bad_args(self, *args):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse(*args)

    def engine(self):
        return SimpleNamespace(is_finished=lambda: True,
                               scheduler=SimpleNamespace(block_manager=FakeManager(4, 256)))

    def test_defaults(self):
        args = self.parse()
        self.assertEqual(args.max_num_seqs, 2)
        self.assertEqual(args.execution, "eager")

    def test_positive_work_and_warmup(self):
        for option in ("--num-requests", "--input-len", "--output-len", "--warmup",
                       "--repeats", "--max-num-seqs", "--max-num-batched-tokens"):
            with self.subTest(option=option):
                self.assert_bad_args(option, "0")

    def test_length_and_request_limits(self):
        self.assert_bad_args("--input-len", "1000", "--output-len", "25")
        self.assert_bad_args("--num-requests", "3", "--max-num-seqs", "2")
        self.parse("--input-len", "960", "--output-len", "64")

    def test_invalid_floats_and_seed(self):
        for option, values in (("--temperature", ("0", "nan", "inf")),
                               ("--gpu-memory-utilization", ("0", "1", "nan"))):
            for value in values:
                self.assert_bad_args(option, value)
        self.assert_bad_args("--seed", "-1")

    def test_graph_bucket_guard(self):
        for cap in (3, 9, 17, 513):
            self.assert_bad_args("--execution", "graph", "--max-num-seqs", str(cap))
        self.parse("--execution", "graph", "--max-num-seqs", "16")
        self.parse("--execution", "eager", "--max-num-seqs", "17")

    def test_missing_model(self):
        (self.model / "config.json").unlink()
        self.assert_bad_args()

    def test_missing_and_empty_weights(self):
        with self.assertRaises(ValueError):
            validate_model_files(self.model)
        (self.model / "model.safetensors").touch()
        with self.assertRaises(ValueError):
            validate_model_files(self.model)

    def test_sharded_download_completeness(self):
        (self.model / "part1.safetensors").write_bytes(b"fixture-not-real-weights")
        index = self.model / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}))
        with self.assertRaises(ValueError):
            validate_model_files(self.model)
        (self.model / "part2.safetensors").write_bytes(b"fixture-not-real-weights")
        validate_model_files(self.model)

    def test_reproducible_prompts_and_unique_prefixes(self):
        ids = [0, 5, 19, 42]  # non-contiguous tokenizer IDs
        prompts = make_prompts(ids, 3, 257, 7)
        self.assertEqual(prompts, make_prompts(reversed(ids), 3, 257, 7))
        self.assertNotEqual(prompts, make_prompts(ids, 3, 257, 8))
        self.assertEqual(len({p[0] for p in prompts}), 3)
        self.assertTrue(all(len(p) == 257 and set(p) <= set(ids) for p in prompts))
        self.assertEqual(len(make_prompts(ids, 1, 1, 0)[0]), 1)
        with self.assertRaises(ValueError):
            make_prompts(ids, 5, 128, 0)

    def test_capacity_boundaries(self):
        for total, expected in ((255, 1), (256, 1), (257, 2)):
            self.assertEqual(required_blocks(2, total - 1, 1, 256), expected * 2)

    def test_cache_reset(self):
        engine = self.engine()
        old = engine.scheduler.block_manager
        old.hash_to_block_id[123] = 0
        reset_prefix_cache(engine)
        new = engine.scheduler.block_manager
        self.assertIsNot(old, new)
        self.assertEqual(new.hash_to_block_id, {})
        self.assertEqual(len(new.blocks), 4)
        self.assertEqual(new.block_size, 256)

    def test_reset_refuses_active_engine(self):
        engine = self.engine()
        engine.is_finished = lambda: False
        with self.assertRaises(RuntimeError):
            reset_prefix_cache(engine)

    def test_detects_held_blocks_and_reference_leaks(self):
        engine = self.engine()
        engine.scheduler.block_manager.used_block_ids.add(1)
        with self.assertRaises(RuntimeError):
            check_idle(engine)
        engine = self.engine()
        engine.scheduler.block_manager.blocks[0].ref_count = 1
        with self.assertRaises(RuntimeError):
            check_idle(engine)

    def test_detects_duplicate_or_missing_free_blocks(self):
        for ids in ((0, 1, 2), (0, 1, 2, 2), (0, 1, 2, 3, 3)):
            engine = self.engine()
            engine.scheduler.block_manager.free_block_ids = deque(ids)
            with self.assertRaises(RuntimeError):
                check_idle(engine)

    def test_actual_output_count(self):
        self.assertEqual(validate_outputs([{"token_ids": [1, 2]}] * 2, 2, 2), 4)
        with self.assertRaises(RuntimeError):
            validate_outputs([], 2, 2)
        with self.assertRaises(RuntimeError):
            validate_outputs([{"token_ids": [1]}], 1, 2)

    def test_summary_excludes_warmup(self):
        rows = [dict(phase=phase, elapsed_seconds=seconds, output_tokens=100,
                     output_tokens_per_second=100 / seconds)
                for phase, seconds in (("warmup", 100), ("measure", 1), ("measure", 2))]
        summary = summarize(rows)
        self.assertEqual(summary["measured_rounds"], 2)
        self.assertEqual(summary["output_tokens_per_second_median"], 75)
        self.assertAlmostEqual(summary["aggregate_output_tokens_per_second"], 200 / 3)
        with self.assertRaises(ValueError):
            summarize(rows[:1])

    def test_json_output(self):
        path = self.model / "result.json"
        write_json(path, {"status": "running"})
        write_json(path, {"status": "complete"})
        self.assertEqual(json.loads(path.read_text()), {"status": "complete"})
        self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_failure_is_recorded(self):
        output = self.model / "failed-run"
        with patch("bench_baseline.command_output", return_value="test"), \
                patch("bench_baseline.run", side_effect=RuntimeError("test failure")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            main(["--model", str(self.model), "--output-dir", str(output)])
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertIn("test failure", metadata["error"])
        self.assertTrue((output / "benchmark_source.py").is_file())

    def test_existing_results_never_overwritten(self):
        output = self.model / "existing-run"
        output.mkdir()
        sentinel = output / "metadata.json"
        sentinel.write_text("keep me")
        with self.assertRaises(FileExistsError):
            main(["--model", str(self.model), "--output-dir", str(output)])
        self.assertEqual(sentinel.read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
