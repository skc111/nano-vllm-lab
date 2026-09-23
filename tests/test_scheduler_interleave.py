"""Real scheduler/block-manager CPU tests. Run with the project venv (no GPU needed)."""

import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from nanovllm.config import Config
            from nanovllm.engine.scheduler import Scheduler
            from nanovllm.engine.sequence import Sequence
            from nanovllm.sampling_params import SamplingParams
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "triton", "transformers", "xxhash", "numpy", "flash_attn"}:
                raise unittest.SkipTest("activate project venv for real scheduler tests") from exc
            raise
        cls.Config, cls.Scheduler, cls.Sequence, cls.SamplingParams = Config, Scheduler, Sequence, SamplingParams

    def scheduler(self, policy="interleave", blocks=64, budget=128, max_seqs=2):
        return self.Scheduler(SimpleNamespace(
            max_num_seqs=max_seqs, max_num_batched_tokens=budget, eos=-1,
            kvcache_block_size=256, num_kvcache_blocks=blocks, scheduling_policy=policy))

    def seq(self, length, tag=1, output=16, ignore_eos=True):
        return self.Sequence([tag] * length, self.SamplingParams(max_tokens=output, ignore_eos=ignore_eos))

    def assert_blocks(self, scheduler):
        seqs = list(scheduler.waiting) + list(scheduler.running)
        self.assertEqual(len({s.seq_id for s in seqs}), len(seqs))
        owners = Counter(b for s in seqs for b in s.block_table)
        manager = scheduler.block_manager
        self.assertEqual(set(owners), manager.used_block_ids)
        self.assertEqual(set(manager.free_block_ids), set(range(len(manager.blocks))) - set(owners))
        self.assertEqual(len(manager.free_block_ids), len(set(manager.free_block_ids)))
        for block in manager.blocks:
            self.assertEqual(block.ref_count, owners[block.block_id])

    def tick(self, scheduler, token=None):
        seqs, prefill = scheduler.schedule()
        work = sum(s.num_scheduled_tokens for s in seqs)
        # Controlled fake sampling tests state transitions, not GPU/model math.
        ids = [500 + s.num_completion_tokens if token is None else token for s in seqs]
        scheduler.postprocess(seqs, ids, prefill)
        self.assert_blocks(scheduler)
        return seqs, prefill, work

    def drain(self, scheduler, limit=200):
        phases = []
        for _ in range(limit):
            if scheduler.is_finished():
                return phases
            phases.append(self.tick(scheduler)[1])
        self.fail("scheduler failed to finish within bounded steps")

    def prepared_pair(self, policy):
        scheduler = self.scheduler(policy)
        a = self.seq(1, output=32)
        scheduler.add(a)
        self.tick(scheduler)  # first token
        self.tick(scheduler)  # one decode, as in an already-running stream
        b = self.seq(300, tag=2, output=2)
        scheduler.add(b)
        return scheduler, a, b

    def test_default_and_invalid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("nanovllm.config.AutoConfig.from_pretrained",
                       return_value=SimpleNamespace(max_position_embeddings=4096)):
                self.assertEqual(self.Config(directory).scheduling_policy, "prefill_first")
                self.assertEqual(self.Config(directory, scheduling_policy="interleave").scheduling_policy, "interleave")
                with self.assertRaises(ValueError):
                    self.Config(directory, scheduling_policy="typo")

    def test_interleave_matches_hand_written_execution_plan(self):
        from check_scheduler_logits import ReferencePlan, describe_step

        def execute(manual):
            scheduler = self.scheduler("prefill_first" if manual else "interleave")
            labels, history = {}, []
            plan = ReferencePlan(scheduler, labels)
            if manual:
                scheduler.schedule = plan
            a = self.seq(255, output=8)
            scheduler.add(a)
            labels[a.seq_id] = "A"
            b = None
            for _ in range(30):
                if a.num_completion_tokens == 2 and b is None:
                    b = self.seq(257, tag=2, output=4)
                    labels[b.seq_id] = "B"
                    scheduler.add(b)
                if scheduler.is_finished():
                    break
                seqs, prefill = scheduler.schedule()
                history.append(describe_step(seqs, prefill, labels))
                scheduler.postprocess(seqs, [500 + s.num_completion_tokens for s in seqs], prefill)
                self.assert_blocks(scheduler)
            self.assertTrue(scheduler.is_finished())
            self.assertEqual((a.num_completion_tokens, b.num_completion_tokens), (8, 4))
            self.assertEqual(len(history), 12)
            if manual:
                self.assertEqual(plan.index, 12)
                # The independent fixture never evaluates the new policy flag.
                self.assertFalse(scheduler._last_was_prefill)
            return history

        self.assertEqual(execute(False), execute(True))

    def test_reference_plan_rejects_wrong_cached_position(self):
        from check_scheduler_logits import ReferencePlan

        scheduler = self.scheduler()
        a = self.seq(255, output=8)
        scheduler.add(a)
        a.num_cached_tokens = 1
        with self.assertRaises(AssertionError):
            ReferencePlan(scheduler, {a.seq_id: "A"})()

    def test_prefill_first_keeps_original_order(self):
        scheduler, a, b = self.prepared_pair("prefill_first")
        before = a.num_completion_tokens
        self.assertEqual([self.tick(scheduler)[1] for _ in range(3)], [True] * 3)
        self.assertEqual(a.num_completion_tokens, before)
        self.assertEqual(b.num_completion_tokens, 1)
        self.assertFalse(self.tick(scheduler)[1])
        self.drain(scheduler)

    def test_interleave_phase_order_and_partial_progress(self):
        scheduler, a, b = self.prepared_pair("interleave")
        before = a.num_completion_tokens
        phases = []
        for cached in (128, 256):
            phases.append(self.tick(scheduler)[1])
            self.assertEqual(b.num_cached_tokens, cached)
            self.assertEqual(b.num_completion_tokens, 0)
            phases.append(self.tick(scheduler)[1])
            self.assertEqual(b.num_cached_tokens, cached)
        phases.append(self.tick(scheduler)[1])
        self.assertEqual(phases, [True, False, True, False, True])
        self.assertEqual(a.num_completion_tokens, before + 2)
        self.assertEqual(b.num_completion_tokens, 1)
        self.drain(scheduler)

    def test_no_running_request_never_forces_empty_decode(self):
        scheduler = self.scheduler()
        seq = self.seq(300, output=1)
        scheduler.add(seq)
        self.assertEqual(self.drain(scheduler), [True, True, True])

    def test_pure_decode_and_finished_cleanup(self):
        scheduler = self.scheduler()
        seq = self.seq(1, output=4)
        scheduler.add(seq)
        self.assertEqual(self.drain(scheduler), [True, False, False, False])
        self.assertEqual(seq.completion_token_ids, [500, 501, 502, 503])
        self.assertEqual(seq.block_table, [])

    def test_finished_decode_does_not_block_partial_prefill(self):
        scheduler = self.scheduler()
        a = self.seq(1, output=2)
        scheduler.add(a)
        self.tick(scheduler)
        b = self.seq(300, tag=2, output=1)
        scheduler.add(b)
        self.assertFalse(self.tick(scheduler)[1])
        self.assertTrue(a.is_finished)
        self.assertEqual(self.drain(scheduler), [True, True, True])

    def test_memory_guard_allows_partial_prefill_to_finish(self):
        scheduler = self.scheduler(blocks=3)
        a = self.seq(255, output=4)
        scheduler.add(a)
        self.tick(scheduler)
        self.tick(scheduler)
        self.tick(scheduler)  # len(A)=257: needs a new block on next decode
        self.assertEqual(len(a), 257)
        b = self.seq(300, tag=2, output=1)
        scheduler.add(b)
        self.assertTrue(self.tick(scheduler)[1])  # B reserves both remaining blocks
        self.assertFalse(scheduler.block_manager.can_append(a))
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 0)
        self.assertTrue(self.tick(scheduler)[1])  # no unsafe forced decode/preemption
        self.assertTrue(self.tick(scheduler)[1])
        self.assertTrue(b.is_finished)
        self.assertIn(a, scheduler.running)
        self.drain(scheduler)

    def test_existing_preemption_and_reentry(self):
        scheduler = self.scheduler(blocks=2, budget=512)
        a, b = self.seq(256, output=2), self.seq(256, tag=2, output=2)
        scheduler.add(a)
        scheduler.add(b)
        self.tick(scheduler)
        self.assertFalse(self.tick(scheduler)[1])
        self.assertTrue(a.is_finished)
        self.assertIn(b, scheduler.waiting)
        self.assertEqual(b.num_cached_tokens, 0)
        self.assertEqual(b.block_table, [])
        self.drain(scheduler)
        self.assertEqual(b.completion_token_ids, [500, 501])

    def test_shared_prefix_reference_counts(self):
        scheduler = self.scheduler()
        a = self.seq(512, output=5)
        scheduler.add(a)
        while a.num_completion_tokens == 0:
            self.tick(scheduler)
        b = self.seq(512, output=3)
        scheduler.add(b)
        while not b.block_table:
            self.tick(scheduler)
        self.assertEqual(a.block_table[0], b.block_table[0])
        self.assertEqual(scheduler.block_manager.blocks[b.block_table[0]].ref_count, 2)
        self.assertEqual(b.num_cached_tokens, 384)  # shared 256 + this step's 128
        self.drain(scheduler)

    def test_255_256_257_boundaries(self):
        for policy in ("prefill_first", "interleave"):
            for length in (255, 256, 257):
                with self.subTest(policy=policy, length=length):
                    scheduler = self.scheduler(policy)
                    seq = self.seq(length, output=4)
                    scheduler.add(seq)
                    self.drain(scheduler)
                    self.assertEqual(len(seq), length + 4)
                    self.assertEqual(seq.block_table, [])

    def test_eos_still_stops(self):
        scheduler = self.scheduler()
        seq = self.seq(1, ignore_eos=False)
        scheduler.add(seq)
        self.tick(scheduler, token=-1)
        self.assertTrue(seq.is_finished)
        self.assertTrue(scheduler.is_finished())

    def test_repeated_batches_do_not_inherit_decode_turn(self):
        scheduler = self.scheduler()
        for tag in (1, 2, 3):
            scheduler.add(self.seq(300, tag=tag, output=1))
            self.assertEqual(self.drain(scheduler), [True, True, True])

    def test_prefill_and_decode_both_progress_with_waiting_requests(self):
        scheduler = self.scheduler(max_seqs=1)
        a = self.seq(1, output=16)
        scheduler.add(a)
        self.tick(scheduler)
        requests = [self.seq(257, tag=i, output=2) for i in range(2, 5)]
        for seq in requests:
            scheduler.add(seq)
        phases = self.drain(scheduler)
        self.assertIn(True, phases)
        self.assertIn(False, phases)
        self.assertTrue(all(seq.is_finished for seq in [a] + requests))


if __name__ == "__main__":
    unittest.main()
