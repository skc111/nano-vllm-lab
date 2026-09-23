"""Real allocator/scheduler tests with controlled tokens; no GPU execution."""
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_scheduler_interleave as reference


class OnDemandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.SchedulerTests.setUpClass()
        for name in ('Config', 'Scheduler', 'Sequence', 'SamplingParams'):
            setattr(cls, name, getattr(reference.SchedulerTests, name))

    seq = reference.SchedulerTests.seq
    assert_blocks = reference.SchedulerTests.assert_blocks
    tick = reference.SchedulerTests.tick
    drain = reference.SchedulerTests.drain

    def scheduler(self, blocks=32, budget=128, slots=2, allocation='on_demand'):
        return self.Scheduler(SimpleNamespace(
            max_num_seqs=slots, max_num_batched_tokens=budget, eos=-1,
            kvcache_block_size=256, num_kvcache_blocks=blocks,
            scheduling_policy='mixed', kv_allocation=allocation, observe_kv=True))

    def test_initial_chunk_uses_two_not_sixteen_blocks(self):
        for allocation, expected in [('full', 16), ('on_demand', 2)]:
            s = self.scheduler(budget=512, allocation=allocation)
            b = self.seq(4096, output=1); s.add(b)
            seqs, varlen = s.schedule()
            self.assertEqual(len(b.block_table), expected)
            self.assertEqual(s.kv_snapshot()['unwritten_blocks_before_forward'], expected)
            s.postprocess(seqs, [None], varlen)
            self.assertEqual(s.kv_snapshot()['unwritten_blocks_before_forward'], expected-2)
            self.assertEqual(b.num_cached_tokens, 512)
            self.drain(s)

    def test_block_boundaries_and_short_final_query(self):
        for length in (255, 256, 257, 511, 512, 513):
            s = self.scheduler(blocks=4)
            b = self.seq(length, output=4); s.add(b)
            for _ in range(30):
                if s.is_finished(): break
                seqs, varlen = s.schedule()
                for q in seqs:
                    self.assertEqual(len(q.block_table), (q.num_cached_tokens + q.num_scheduled_tokens + 255)//256)
                    self.assertLessEqual(sum(x.num_scheduled_tokens for x in seqs), 128)
                s.postprocess(seqs, [77 if q.num_cached_tokens+q.num_scheduled_tokens == len(q) else None for q in seqs], varlen)
                self.assert_blocks(s)
            self.assertTrue(s.is_finished())
            self.assertEqual(b.num_completion_tokens, 4)

    def test_failed_chunk_does_not_mutate_state(self):
        s = self.scheduler(blocks=2, budget=256)
        a = self.seq(257, output=1); s.add(a); self.tick(s)
        b = self.seq(257, tag=2, output=1)
        m = s.block_manager
        before = (list(m.free_block_ids), set(m.used_block_ids), dict(m.hash_to_block_id))
        self.assertEqual(m.allocate_chunk(b, 257), 0)
        self.assertEqual((list(m.free_block_ids), set(m.used_block_ids), dict(m.hash_to_block_id)), before)
        self.assertEqual(b.block_table, [])
        self.assertEqual(b.num_cached_tokens, 0)
        self.drain(s)

    def test_free_prefix_claimed_before_eviction_and_shared_prefix_refs(self):
        s = self.scheduler(blocks=4, budget=128)
        a = self.seq(513, output=1); s.add(a); self.drain(s)
        b = self.seq(513, output=5); s.add(b)
        seqs, varlen = s.schedule()
        self.assertEqual(b.num_cached_tokens, 512)
        shared = b.block_table[:2]
        s.postprocess(seqs, [42], varlen)
        c = self.seq(513, output=1); s.add(c)
        seqs, varlen = s.schedule()
        self.assertEqual(c.block_table[:2], shared)
        self.assertTrue(all(s.block_manager.blocks[x].ref_count == 2 for x in shared))
        s.postprocess(seqs, [42] * len(seqs), varlen)
        self.assertTrue(c.is_finished)
        self.assertTrue(all(s.block_manager.blocks[x].ref_count == 1 for x in shared))
        self.drain(s)

    def test_pressure_recovery_recomputes_and_finishes(self):
        s = self.scheduler(blocks=3, budget=128)
        a, b = self.seq(255, output=4), self.seq(511, tag=2, output=4)
        # Both have a valid individual final footprint of <=3 blocks.
        s.add(a); s.add(b)
        self.drain(s, limit=100)
        self.assertEqual((a.num_completion_tokens, b.num_completion_tokens), (4, 4))
        # A separate forced full-pool fixture ensures recovery is exercised,
        # independent of how the normal scheduler interleaves this workload.
        s = self.scheduler(blocks=3, budget=128)
        a, b = self.seq(255, output=300), self.seq(257, tag=2, output=3)
        s.add(a)
        while a.num_completion_tokens < 258: self.tick(s)
        self.assertEqual(a.num_cached_tokens, 512)
        s.add(b)
        # Give B the last physical block with a partial prefill.
        self.assertEqual(s.block_manager.allocate_chunk(b, 128), 128)
        b.num_scheduled_tokens = 128
        s.postprocess([b], [None], True)
        self.assertFalse(s.block_manager.free_block_ids)
        self.tick(s)  # B can fill the rest of its existing block before eviction
        seqs, varlen = s.schedule()
        self.assertEqual(seqs, [a])
        self.assertFalse(varlen)
        self.assertEqual(s.preemptions, 1)
        self.assertEqual(s.evicted_cached_tokens, 256)
        s.postprocess(seqs, [42], varlen)
        self.drain(s, limit=200)
        self.assertGreaterEqual(s.recomputed_tokens, 128)
        self.assertFalse(s._computed_high_water)

    def test_cloud_pressure_fixture_has_real_recovery(self):
        s = self.scheduler(blocks=3, budget=128)
        d = self.seq(511, tag=109, output=4); s.add(d)
        while not d.num_completion_tokens: self.tick(s)
        e = self.seq(257, tag=151, output=2); s.add(e)
        self.drain(s, limit=40)
        self.assertGreater(s.preemptions, 0)
        self.assertGreater(s.recomputed_tokens, 0)

    def test_roomy_pool_same_execution_plan_as_full(self):
        histories = []
        for allocation in ('full', 'on_demand'):
            s = self.scheduler(allocation=allocation, budget=128)
            a, b = self.seq(255, output=8), self.seq(513, tag=2, output=4)
            s.add(a); s.add(b)
            history = []
            while not s.is_finished():
                seqs, varlen = s.schedule()
                history.append([(q is a, q.num_cached_tokens, q.num_scheduled_tokens, q.is_prefill) for q in seqs])
                s.postprocess(seqs, [42] * len(seqs), varlen)
                self.assert_blocks(s)
            histories.append(history)
        self.assertEqual(*histories)

    def test_experiment_workload_finishes_at_both_pool_sizes(self):
        from bench_baseline import reset_prefix_cache
        for blocks in (20, 64):
            for allocation in ('full', 'on_demand'):
                s = self.scheduler(blocks=blocks, allocation=allocation, budget=512)
                a, b, c = self.seq(128, output=64), self.seq(4096, tag=2, output=32), self.seq(4096, tag=3, output=32)
                s.add(a); s.add(b); self.tick(s); s.add(c)
                self.drain(s, limit=300)
                self.assertEqual(sum(q.num_completion_tokens for q in (a,b,c)), 128)
                engine = SimpleNamespace(scheduler=s, is_finished=s.is_finished)
                reset_prefix_cache(engine)
                self.assertEqual(s.preemptions, 0)
                self.assertEqual(s.recomputed_tokens, 0)
                self.assertFalse(s._last_was_prefill)

    def test_impossible_request_rejected_without_occupancy(self):
        s = self.scheduler(blocks=1)
        with self.assertRaisesRegex(ValueError, 'cannot finish'):
            s.add(self.seq(255, output=3))
        self.assertTrue(s.is_finished())
        self.assertFalse(s.block_manager.used_block_ids)
        s.add(self.seq(255, output=2))  # last sampled token needs no KV
        self.drain(s)

    def test_random_small_pools_finish_and_preserve_references(self):
        rng = random.Random(17)
        for case in range(40):
            blocks = rng.choice([2, 3, 4, 5])
            s = self.scheduler(blocks=blocks, budget=rng.choice([1, 127, 256, 511]), slots=rng.choice([1, 2, 4]))
            requests = []
            for j in range(4):
                length = rng.randint(1, blocks*256-70)
                q = self.seq(length, tag=j+1, output=rng.randint(1, 64))
                requests.append(q); s.add(q)
            self.drain(s, limit=12000)
            self.assertTrue(all(q.is_finished and q.num_completion_tokens == q.max_tokens for q in requests), case)
            self.assertFalse(s.block_manager.used_block_ids)

    def test_config_rejects_unsupported_combinations(self):
        with tempfile.TemporaryDirectory() as directory, patch('nanovllm.config.AutoConfig.from_pretrained', return_value=SimpleNamespace(max_position_embeddings=4096)):
            for kwargs in [dict(kv_allocation='bad'), dict(kv_allocation='on_demand'),
                           dict(kv_allocation='on_demand', scheduling_policy='mixed', tensor_parallel_size=2),
                           dict(kv_cache_blocks=0)]:
                with self.assertRaises(ValueError): self.Config(directory, **kwargs)
            c = self.Config(directory, scheduling_policy='mixed', kv_allocation='on_demand', kv_cache_blocks=20)
            self.assertEqual(c.kv_cache_blocks, 20)

    def test_eos_and_length_validation(self):
        s = self.scheduler()
        s.max_model_len = 256
        with self.assertRaisesRegex(ValueError, 'max_model_len'):
            s.add(self.seq(255, output=2))
        a = self.seq(128, output=100, ignore_eos=False); s.add(a)
        self.tick(s, token=s.eos)
        self.assertTrue(a.is_finished)
        self.assertEqual(a.num_completion_tokens, 1)
        self.assertFalse(s._computed_high_water)


if __name__ == '__main__': unittest.main()
