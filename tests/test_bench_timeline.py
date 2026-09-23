"""Deterministic CPU tests with a fake clock and a prefill-first fake engine."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from bench_timeline import (build_workload, parse_args, replay, request_metrics,
                            summarize_rounds, validate_trace)


class Clock:
    def __init__(self):
        self.value = 10.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class Engine:
    def __init__(self, clock):
        self.clock = clock
        self.scheduler = SimpleNamespace(
            waiting=deque(), running=deque(),
            block_manager=SimpleNamespace(blocks=[SimpleNamespace(ref_count=0)],
                                          used_block_ids=set(), free_block_ids=deque([0])))

    def is_finished(self):
        return not self.scheduler.waiting and not self.scheduler.running

    def add_request(self, prompt, output_len):
        self.scheduler.waiting.append(SimpleNamespace(
            remaining=len(prompt), num_completion_tokens=0, output_len=output_len,
            is_finished=False, last_token=0))

    def emit(self, seq):
        seq.num_completion_tokens += 1
        seq.last_token = 100 + seq.num_completion_tokens
        if seq.num_completion_tokens == seq.output_len:
            seq.is_finished = True
            self.scheduler.running.remove(seq)

    def step(self):
        if self.scheduler.waiting:
            seq = self.scheduler.waiting[0]
            seq.remaining -= 1
            self.clock.sleep(0.4)
            if seq.remaining == 0:
                self.scheduler.waiting.popleft()
                self.scheduler.running.append(seq)
                self.emit(seq)
            return [], 1
        running = list(self.scheduler.running)
        self.clock.sleep(0.1)
        for seq in running:
            self.emit(seq)
        return [], -len(running)


class TimelineTests(unittest.TestCase):
    def workload(self):
        return [dict(id="A", arrival_s=0, input_len=1, output_len=3, prompt_token_ids=[1]),
                dict(id="B", arrival_s=0.45, input_len=3, output_len=1, prompt_token_ids=[2, 3, 4])]

    def replay(self, workload=None, timeout=10):
        clock = Clock()
        return replay(Engine(clock), workload if workload is not None else self.workload(),
                      lambda n: n, clock=clock.now, sleep=clock.sleep, timeout_s=timeout)

    def test_delayed_admission_not_hidden(self):
        result = self.replay()
        b = result["requests"][1]
        self.assertAlmostEqual(b["planned_arrival_s"], 0.45)
        self.assertAlmostEqual(b["admitted_s"], 0.5)
        self.assertAlmostEqual(b["admission_lag_s"], 0.05)
        self.assertAlmostEqual(b["ttft_from_admission_s"], 1.2)
        self.assertAlmostEqual(b["ttft_from_planned_s"], 1.25)
        self.assertEqual(result["admissions"][1]["generating_request_ids"], ["A"])

    def test_partial_prefill_samples_not_counted(self):
        result = self.replay()
        self.assertEqual(len(result["tokens"]), 4)
        partial_steps = [s for s in result["steps"] if s["phase"] == "prefill" and not s["new_output_tokens"]]
        self.assertEqual(len(partial_steps), 2)
        self.assertEqual([r["output_tokens"] for r in result["requests"]], [3, 1])

    def test_explicit_mixed_counters_override_legacy_positive_sign(self):
        class CounterEngine(Engine):
            def step(self):
                result, signed = super().step()
                self.last_step_stats = ({'phase':'mixed','prefill_tokens':1,'decode_tokens':1}
                                        if signed > 0 else {'phase':'decode','prefill_tokens':0,'decode_tokens':-signed})
                return result, 2 if signed > 0 else signed
        clock=Clock()
        result=replay(CounterEngine(clock),self.workload(),lambda n:n,clock=clock.now,sleep=clock.sleep)
        mixed=[s for s in result['steps'] if s['phase']=='mixed']
        self.assertTrue(mixed)
        self.assertTrue(all(s['scheduled_input_tokens']==2 and s['prefill_tokens']==1 and s['decode_tokens']==1 for s in mixed))

    def test_bad_explicit_counters_fail_observer(self):
        clock=Clock();engine=Engine(clock)
        engine.last_step_stats={'phase':'mixed','prefill_tokens':999,'decode_tokens':1}
        with self.assertRaisesRegex(RuntimeError,'counters disagree'):
            replay(engine,self.workload(),lambda n:n,clock=clock.now,sleep=clock.sleep)

    def test_decode_interruption_is_visible(self):
        result = self.replay()
        a = result["requests"][0]
        self.assertAlmostEqual(a["first_token_s"], 0.4)
        self.assertAlmostEqual(a["last_token_s"], 1.8)
        self.assertAlmostEqual(a["itl_max_s"], 1.3)
        self.assertAlmostEqual(a["tpot_s"], 0.7)
        control = self.replay(self.workload()[:1])
        self.assertAlmostEqual(control["requests"][0]["itl_max_s"], 0.1)

    def test_single_token_has_no_itl_or_tpot(self):
        b = self.replay()["requests"][1]
        self.assertIsNone(b["tpot_s"])
        self.assertIsNone(b["itl_median_s"])
        self.assertIsNone(b["itl_max_s"])

    def test_idle_arrival_and_no_false_overlap(self):
        workload = self.workload()
        workload[0]["output_len"] = 1
        workload[1]["arrival_s"] = 2.0
        result = self.replay(workload)
        self.assertEqual(result["admissions"][1]["generating_request_ids"], [])
        self.assertAlmostEqual(result["requests"][1]["admission_lag_s"], 0)

    def test_simultaneous_arrival(self):
        workload = self.workload()
        workload[1]["arrival_s"] = 0
        result = self.replay(workload)
        self.assertEqual([a["request_id"] for a in result["admissions"]], ["A", "B"])
        self.assertEqual(result["admissions"][1]["pending_request_ids"], ["A"])
        self.assertEqual(result["admissions"][1]["generating_request_ids"], [])

    def test_invalid_output_count(self):
        with self.assertRaises(RuntimeError):
            request_metrics(self.workload()[0], 0, [0.4])

    def test_multiple_tokens_per_step_rejected(self):
        clock = Clock()
        engine = Engine(clock)
        old_emit = engine.emit
        def emit_twice(seq):
            old_emit(seq)
            old_emit(seq)
        engine.emit = emit_twice
        with self.assertRaisesRegex(RuntimeError, "at most one"):
            replay(engine, self.workload()[:1], lambda n: n, clock=clock.now, sleep=clock.sleep)

    def test_timeout(self):
        with self.assertRaises(TimeoutError):
            self.replay(timeout=0.3)

    def test_last_step_timeout(self):
        workload = self.workload()[:1]
        workload[0]["output_len"] = 1
        with self.assertRaises(TimeoutError):
            self.replay(workload, timeout=0.3)

    def test_requeued_sequence_keeps_output_history(self):
        clock = Clock()
        engine = Engine(clock)
        original_step = engine.step
        requeued = False
        def step():
            nonlocal requeued
            if not requeued and engine.scheduler.running:
                seq = engine.scheduler.running.popleft()
                seq.remaining = 2  # simulate recomputation, without erasing generated tokens
                engine.scheduler.waiting.appendleft(seq)
                requeued = True
            return original_step()
        engine.step = step
        result = replay(engine, self.workload()[:1], lambda n: n, clock=clock.now, sleep=clock.sleep)
        self.assertEqual([t["token_index"] for t in result["tokens"]], [1, 2, 3])
        self.assertEqual(len(result["tokens"]), 3)
        self.assertTrue(any(s["phase"] == "prefill" and s["new_output_tokens"] == 0 for s in result["steps"]))

    def test_summary_excludes_warmup(self):
        result = self.replay()
        warmup = copy.deepcopy(result)
        warmup.update(phase="warmup")
        warmup["requests"][0]["ttft_from_planned_s"] = 99
        result.update(phase="measure")
        summary = summarize_rounds([warmup, result])
        self.assertEqual(summary["measured_rounds"], 1)
        self.assertAlmostEqual(summary["requests"]["A"]["ttft_from_planned_s_median"], 0.4)
        self.assertEqual(summary["requests"]["B"]["rounds_admitted_during_other_generation"], 1)
        with self.assertRaises(ValueError):
            summarize_rounds([warmup])

    def test_trace_order_stable(self):
        trace = [{k: v for k, v in r.items() if k != "prompt_token_ids"} for r in self.workload()]
        self.assertEqual(validate_trace(list(reversed(trace))), trace)
        trace[1]["arrival_s"] = 0
        self.assertEqual(validate_trace(trace), trace)

    def test_invalid_trace(self):
        valid = dict(id="A", arrival_s=0, input_len=1, output_len=1)
        for bad in ([], {}, [valid, valid], [dict(valid, extra=True)],
                    [dict(valid, arrival_s=float("nan"))], [dict(valid, arrival_s=-1)],
                    [dict(valid, arrival_s=True)], [dict(valid, output_len=0)],
                    [dict(valid, input_len=1.2)], [dict(valid, id="")]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_trace(bad)

    def test_control_reuses_exact_prompt(self):
        trace = [{k: v for k, v in r.items() if k != "prompt_token_ids"} for r in self.workload()]
        full = build_workload(trace, range(20), 0)
        control = build_workload(trace, range(20), 0, "A")
        self.assertEqual(control, full[:1])
        self.assertNotEqual(full[0]["prompt_token_ids"][0], full[1]["prompt_token_ids"][0])
        with self.assertRaises(ValueError):
            build_workload(trace, range(20), 0, "unknown")

    def test_cli_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text("{}")
            self.assertEqual(parse_args(["--model", directory]).execution, "graph")
            self.assertEqual(parse_args(["--model", directory]).scheduling_policy, "prefill_first")
            self.assertEqual(parse_args(["--model", directory, "--scheduling-policy", "interleave"]).scheduling_policy, "interleave")
            for option, value in (("--warmup", "0"), ("--timeout-s", "nan"),
                                  ("--max-num-seqs", "17"), ("--gpu-memory-utilization", "1"),
                                  ("--scheduling-policy", "unknown")):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["--model", directory, option, value])


if __name__ == "__main__":
    unittest.main()
