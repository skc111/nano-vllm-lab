"""CPU-only tests for official-style bulk workload and orchestration."""

import contextlib
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bench_bulk as bulk


class BulkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")

    def args(self, *extra):
        return bulk.parse_args(["--model", str(self.model), *extra])

    def metadata_pair(self):
        common = dict(gpu={"name": "fixture"}, versions={}, kv_pool={"num_blocks": 100},
                      workload_sha256="fixture", model_dtype="bfloat16",
                      model_metadata_sha256={}, weight_file_sizes={},
                      git_commit="fixture", git_status="", status="complete")
        return [dict(common, effective_policy=p) for p in ("prefill_first", "mixed")]

    def test_official_workload_totals_and_draw_order(self):
        workload = bulk.make_workload()
        self.assertEqual(bulk.workload_stats(workload), dict(
            requests=256, input_tokens=142827, output_tokens=133966, max_total_length=2011))
        rng = random.Random(0)
        prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(100, 1024))]
                   for _ in range(256)]
        lengths = [rng.randint(100, 1024) for _ in range(256)]
        self.assertEqual(workload["prompt_token_ids"], prompts)
        self.assertEqual(workload["output_lengths"], lengths)
        self.assertTrue(workload["ignore_eos"])

    def test_rng_is_local_and_seeded(self):
        state = random.getstate()
        self.assertEqual(bulk.make_workload(3, 2), bulk.make_workload(3, 2))
        self.assertNotEqual(bulk.make_workload(3, 2), bulk.make_workload(4, 2))
        self.assertEqual(random.getstate(), state)

    def test_output_lengths_checked_per_request_not_just_total(self):
        outputs = [{"token_ids": [1] * 2}, {"token_ids": [1] * 3}]
        self.assertEqual(bulk.validate_outputs(outputs, [2, 3]), 5)
        for expected in ([3, 2], [5], [2, 3, 0]):
            with self.assertRaises(RuntimeError):
                bulk.validate_outputs(outputs, expected)

    def test_defaults_and_queued_requests_may_exceed_active_limit(self):
        args = self.args()
        self.assertEqual((args.num_requests, args.max_num_seqs,
                          args.max_num_batched_tokens, args.max_model_len),
                         (256, 512, 16384, 4096))
        self.assertEqual(args.policy, "both")
        self.assertEqual(self.args("--max-num-seqs", "8").num_requests, 256)

    def test_invalid_arguments(self):
        cases = [("--warmup", "0"), ("--repeats", "0"), ("--num-requests", "0"),
                 ("--max-num-seqs", "17"), ("--max-num-seqs", "528"),
                 ("--max-num-batched-tokens", "0"), ("--max-model-len", "2000"),
                 ("--kv-cache-blocks", "0"), ("--seed", "-1"),
                 ("--timeout-s", "nan"), ("--timeout-s", "0"),
                 ("--gpu-memory-utilization", "nan"), ("--gpu-memory-utilization", "1")]
        for case in cases:
            with self.subTest(case=case), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*case)

    def test_dry_run_needs_no_model_or_gpu_and_writes_nothing(self):
        output = io.StringIO()
        directory = self.root / "not-created"
        with contextlib.redirect_stdout(output), patch.object(bulk, "run_gpu") as gpu:
            bulk.main(["--model", str(self.root / "missing"), "--dry-run",
                       "--output-dir", str(directory)])
        gpu.assert_not_called()
        self.assertFalse(directory.exists())
        self.assertEqual(json.loads(output.getvalue())["workload"]["output_tokens"], 133966)

    def test_children_share_parameters_without_recursing(self):
        args = self.args("--kv-cache-blocks", "100", "--max-num-seqs", "16")
        command = bulk.child_command(args, "mixed", self.root / "mixed")
        child = bulk.parse_args(command[3:])
        self.assertEqual(child.policy, "mixed")
        for name in ("model", "seed", "num_requests", "warmup", "repeats",
                     "max_num_seqs", "kv_cache_blocks", "max_num_batched_tokens"):
            self.assertEqual(getattr(child, name), getattr(args, name))

    def test_comparison_rejects_mismatched_pool_workload_or_status(self):
        pair = self.metadata_pair()
        bulk.compare_metadata(pair)
        for field in pair[1]:
            bad = copy.deepcopy(pair)
            bad[1][field] = "different"
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                bulk.compare_metadata(bad)
        with self.assertRaises(RuntimeError):
            bulk.compare_metadata(pair[:1])

    def fake_child(self, command, **kwargs):
        directory = Path(command[command.index("--output-dir") + 1])
        policy = command[command.index("--policy") + 1]
        directory.mkdir()
        index = int(policy == "mixed")
        bulk.write_json(directory / "metadata.json", self.metadata_pair()[index])
        bulk.write_json(directory / "summary.json", {"output_tokens_per_second_median": 100 + 10 * index})
        return SimpleNamespace(returncode=0)

    def test_suite_runs_separate_children_and_compares(self):
        with patch.object(bulk.subprocess, "run", side_effect=self.fake_child) as run, contextlib.redirect_stdout(io.StringIO()):
            metadata = {}
            bulk.run_suite(self.args(), self.root, metadata)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(j["returncode"] == 0 for j in metadata["jobs"]))
        comparison = json.loads((self.root / "comparison.json").read_text())
        self.assertAlmostEqual(comparison["mixed_throughput_change_percent"], 10)

    def test_failed_child_stops_suite(self):
        with patch.object(bulk.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            bulk.run_suite(self.args(), self.root, {})
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.root / "comparison.json").exists())

    def test_main_preserves_timeout_failure_metadata(self):
        directory = self.root / "result"
        with patch.object(bulk.platform, "platform", return_value="fixture"), patch.object(bulk, "command_output", return_value="fixture"), patch.object(
                bulk.subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 1)), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.TimeoutExpired):
            bulk.main(["--model", str(self.model), "--output-dir", str(directory)])
        metadata = json.loads((directory / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertIn("TimeoutExpired", metadata["error"])
        self.assertFalse((directory / "comparison.json").exists())

    def test_no_overwrite(self):
        marker = self.root / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            bulk.main(["--model", str(self.model), "--output-dir", str(self.root)])
        self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
