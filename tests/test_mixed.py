"""CPU state and tensor-layout checks; no CUDA execution required."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
import test_scheduler_interleave as reference


class MixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.SchedulerTests.setUpClass()
        for name in ('Config', 'Scheduler', 'Sequence', 'SamplingParams'):
            setattr(cls, name, getattr(reference.SchedulerTests, name))

    scheduler = reference.SchedulerTests.scheduler
    seq = reference.SchedulerTests.seq
    assert_blocks = reference.SchedulerTests.assert_blocks
    tick = reference.SchedulerTests.tick
    drain = reference.SchedulerTests.drain

    def test_decode_and_chunk_share_token_budget(self):
        s = self.scheduler('mixed', budget=128)
        a = self.seq(1, output=8); s.add(a); self.tick(s)
        b = self.seq(300, tag=2, output=2); s.add(b)
        seqs, varlen = s.schedule()
        self.assertTrue(varlen)
        self.assertEqual(seqs, [a,b])
        self.assertEqual([x.num_scheduled_tokens for x in seqs], [1,127])
        self.assertEqual([x.is_prefill for x in seqs], [False,True])
        s.postprocess(seqs, [42,None], varlen)
        self.assertEqual((a.num_completion_tokens,b.num_completion_tokens,b.num_cached_tokens),(2,0,127))
        self.assert_blocks(s); self.drain(s)

    def test_multiple_prefills_can_fill_remainder(self):
        s = self.scheduler('mixed', budget=128,max_seqs=3)
        a,b,c = self.seq(20,output=1),self.seq(150,tag=2,output=1),self.seq(1,tag=3,output=1)
        for seq in (a,b,c): s.add(seq)
        seqs,_,work = self.tick(s)
        self.assertEqual(seqs,[a,b]);self.assertEqual(work,128)
        self.assertEqual(b.num_cached_tokens,108)
        self.drain(s)

    def test_prefill_progress_with_one_slot_or_one_token(self):
        for slots,budget in ((1,128),(2,1),(1,1)):
            with self.subTest(slots=slots,budget=budget):
                s = self.scheduler('mixed', budget=budget,max_seqs=slots)
                a=self.seq(1,output=20);s.add(a);self.tick(s)
                b=self.seq(3,tag=2,output=1);s.add(b)
                for _ in range(10):
                    if b.is_finished: break
                    self.tick(s)
                self.assertTrue(b.is_finished)
                self.assertGreater(a.num_completion_tokens,1)
                self.drain(s)

    def test_decode_rotation_under_request_cap(self):
        s=self.scheduler('mixed',budget=128,max_seqs=3)
        seqs=[self.seq(1,tag=i+1,output=10) for i in range(3)]
        for seq in seqs:s.add(seq)
        self.tick(s); s.max_num_seqs=1
        for _ in range(3): self.tick(s)
        self.assertEqual([x.num_completion_tokens for x in seqs],[2,2,2])
        self.drain(s)

    def test_memory_guard_and_preemption_keep_progress(self):
        for blocks in (2,3):
            s=self.scheduler('mixed',blocks=blocks,budget=128)
            a=self.seq(255,output=4);s.add(a)
            while a.num_completion_tokens<2:self.tick(s)
            b=self.seq(300,tag=2,output=2);s.add(b)
            self.drain(s)
            self.assertEqual((a.num_completion_tokens,b.num_completion_tokens),(4,2))

    def test_boundaries_eos_and_prefix_reuse(self):
        for length in (255,256,257):
            s=self.scheduler('mixed',budget=128)
            a=self.seq(length,output=1);s.add(a);self.drain(s)
            b=self.seq(length,output=3,ignore_eos=False);s.add(b)
            seqs,varlen=s.schedule()
            if length>256:self.assertGreater(b.num_cached_tokens,0)
            s.postprocess(seqs,[s.eos],varlen)
            while not b.is_finished:self.tick(s,token=s.eos)
            self.assertEqual(b.num_completion_tokens,1)
            self.assert_blocks(s)

    def test_partial_prefill_holding_blocks_not_stranded_by_decode(self):
        s=self.scheduler('mixed',blocks=3,budget=128,max_seqs=1)
        a=self.seq(255,output=5);s.add(a)
        while a.num_completion_tokens<2:self.tick(s)
        b=self.seq(300,tag=2,output=1);s.add(b)
        self.tick(s)  # reserved prefill occupies both remaining blocks
        self.assertEqual(b.num_cached_tokens,128)
        self.assertFalse(s.block_manager.can_append(a))
        self.drain(s)
        self.assertEqual((a.num_completion_tokens,b.num_completion_tokens),(5,1))

    def test_fp32_reference_uses_bottom_right_mask_and_gqa(self):
        import torch
        from check_mixed_gpu import dense_attention
        q=torch.zeros(2,4,2);k=torch.zeros(4,2,2)
        v=torch.arange(4,dtype=torch.float32)[:,None,None].expand(4,2,2)
        out=dense_attention(q,k,v,1.)
        torch.testing.assert_close(out[0],torch.ones(4,2))  # first query sees keys0..2
        torch.testing.assert_close(out[1],torch.full((4,2),1.5))

    def test_mixed_rejects_tp_before_model_load(self):
        with self.assertRaisesRegex(ValueError,'single GPU'):
            self.Config('not-a-directory',scheduling_policy='mixed',tensor_parallel_size=2)

    def test_live_prefix_shared_during_mixed_step(self):
        s=self.scheduler('mixed',budget=128)
        a=self.seq(257,output=8);s.add(a)
        while a.num_completion_tokens==0:self.tick(s)
        b=self.seq(257,output=2);s.add(b)
        seqs,varlen=s.schedule()
        self.assertTrue(varlen)
        self.assertEqual(seqs,[a,b])
        self.assertEqual(b.num_cached_tokens,256)
        self.assertEqual(a.block_table[0],b.block_table[0])
        s.postprocess(seqs,[5,6],varlen)
        self.assertEqual(s.block_manager.blocks[a.block_table[0]].ref_count,2)
        self.assert_blocks(s);self.drain(s)

    def test_output_alignment_is_checked(self):
        s=self.scheduler('mixed');a=self.seq(1);s.add(a)
        seqs,varlen=s.schedule()
        with self.assertRaises(ValueError):s.postprocess(seqs,[],varlen)

    def test_runner_mapping_and_selected_lm_head_rows(self):
        import torch
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import get_context
        a=self.seq(257);a.num_cached_tokens=256;a.num_scheduled_tokens=1;a.block_table=[2,7];a.is_prefill=False
        b=self.seq(300,tag=2);b.num_cached_tokens=128;b.num_scheduled_tokens=64;b.block_table=[3,9]
        c=self.seq(5,tag=3);c.num_scheduled_tokens=5;c.block_table=[10]
        runner=ModelRunner.__new__(ModelRunner)
        runner.block_size=256;runner.rank=0;runner.config=SimpleNamespace(scheduling_policy='mixed')
        original=torch.tensor
        def cpu_tensor(*args,**kwargs):
            kwargs.pop('pin_memory',None);kwargs['device']='cpu'
            return original(*args,**kwargs)
        seen=[]
        def model(ids,positions,varlen):
            ctx=get_context();seen.append(ctx)
            self.assertTrue(varlen)
            if ids.numel()==70:
                self.assertEqual(ctx.cu_seqlens_q.tolist(),[0,1,65,70])
                self.assertEqual(ctx.cu_seqlens_k.tolist(),[0,257,449,454])
                self.assertEqual(ctx.logits_indices.tolist(),[0,69])
                self.assertEqual(positions.tolist(),[256]+list(range(128,192))+list(range(5)))
                self.assertEqual(ctx.slot_mapping.tolist(),[7*256]+list(range(3*256+128,3*256+192))+list(range(10*256,10*256+5)))
                self.assertEqual(ctx.block_tables.tolist(),[[2,7],[3,9],[10,-1]])
            else:self.assertEqual(ctx.logits_indices.numel(),0)
            return torch.zeros((ctx.logits_indices.numel(),16))
        runner.run_model=Mock(side_effect=model)
        runner.sampler=Mock(return_value=original([4,6]))
        with patch('torch.tensor',side_effect=cpu_tensor),patch.object(torch.Tensor,'cuda',lambda t,**kw:t):
            self.assertEqual(runner.run([a,b,c],True),[4,None,6])
            self.assertEqual(runner.run([b],True),[None])
        self.assertEqual(runner.run_model.call_count,2)
        self.assertEqual(runner.sampler.call_count,1)
        self.assertIsNone(get_context().logits_indices)

    def test_lm_head_selects_only_requested_rows_including_empty(self):
        import torch
        import torch.nn as nn
        from nanovllm.layers.embed_head import ParallelLMHead
        from nanovllm.utils.context import get_context,reset_context
        head=ParallelLMHead.__new__(ParallelLMHead);nn.Module.__init__(head)
        head.tp_size=1;head.weight=nn.Parameter(torch.eye(3))
        x=torch.arange(12,dtype=torch.float32).reshape(4,3)
        try:
            get_context().logits_indices=torch.tensor([0,3])
            torch.testing.assert_close(head(x),x[[0,3]])
            get_context().logits_indices=torch.tensor([],dtype=torch.long)
            self.assertEqual(tuple(head(x).shape),(0,3))
        finally:reset_context()

    def test_engine_reports_mixed_phase_without_changing_legacy_return(self):
        from nanovllm.engine.llm_engine import LLMEngine
        s=self.scheduler('mixed',budget=128)
        a=self.seq(1,output=4);s.add(a);self.tick(s)
        b=self.seq(300,tag=2);s.add(b)
        e=LLMEngine.__new__(LLMEngine);e.scheduler=s
        e.model_runner=SimpleNamespace(call=lambda *args:[5,None])
        _,signed=e.step()
        self.assertEqual(signed,128)
        self.assertEqual(e.last_step_stats,{'phase':'mixed','prefill_tokens':127,'decode_tokens':1})
