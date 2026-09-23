"""Single-GPU mixed-path gate, NOT a benchmark or a full-model reference.

Checks each executed varlen attention against explicit FP32 causal attention
on the SAME Q/K/V, verifies KV writes and one model-body call per mixed step.
Pure decode retains Graph. All diagnostic synchronization is outside benchmarks.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def dense_attention(q, k, v, scale):
    """Independent bottom-right causal reference, including grouped query heads."""
    import torch
    groups = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(groups, dim=1)
    v = v.float().repeat_interleave(groups, dim=1)
    scores = torch.einsum('qhd,khd->hqk', q.float(), k) * scale
    query_positions = torch.arange(q.shape[0], device=q.device) + k.shape[0] - q.shape[0]
    allowed = torch.arange(k.shape[0], device=q.device)[None, :] <= query_positions[:, None]
    scores.masked_fill_(~allowed[None, :, :], float('-inf'))
    return torch.einsum('hqk,khd->qhd', scores.softmax(-1), v)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--gpu-memory-utilization', type=float, default=.85)
    p.add_argument('--execution', choices=('graph','eager'), default='graph')
    p.add_argument('--kv-allocation', choices=('full','on_demand'), default='full')
    p.add_argument('--kv-cache-blocks', type=int)
    p.add_argument('--check-kv-pressure', action='store_true')
    args = p.parse_args()
    if args.check_kv_pressure and (args.kv_allocation != 'on_demand' or args.kv_cache_blocks != 3):
        p.error('pressure fixture requires on_demand and exactly 3 KV blocks')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'scope':__doc__, 'arguments':vars(args).copy(),
              'attention_rtol':.025, 'attention_atol':.025, 'events':[]}
    report['arguments'] = {k:str(v) if isinstance(v,Path) else v for k,v in report['arguments'].items()}
    def save():
        (args.output_dir/'check.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import torch
        from nanovllm import LLM, SamplingParams
        from nanovllm.layers.attention import Attention
        from nanovllm.utils.context import get_context
        from bench_baseline import check_idle, command_output, file_sha256

        torch.backends.cuda.matmul.allow_tf32 = False  # FP32 reference, not TF32
        report['reference_allow_tf32'] = False
        report.update(git_commit=command_output(['git','rev-parse','HEAD']),
                      git_status=command_output(['git','status','--short']),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      model_config_sha256=file_sha256(args.model/'config.json'),
                      gpu=torch.cuda.get_device_name(), execution=args.execution)
        (args.output_dir/'tracked.patch').write_text(command_output(['git','diff','HEAD']))
        (args.output_dir/'check_mixed_gpu.py').write_bytes(Path(__file__).read_bytes())
        engine = LLM(str(args.model), scheduling_policy='mixed', tensor_parallel_size=1,
                     enforce_eager=args.execution=='eager', max_num_seqs=2,
                     max_num_batched_tokens=128, max_model_len=768 if args.check_kv_pressure else 512,
                     gpu_memory_utilization=args.gpu_memory_utilization,
                     kv_allocation=args.kv_allocation, kv_cache_blocks=args.kv_cache_blocks)
        runner = engine.model_runner
        report['kv_pool_blocks'] = len(engine.scheduler.block_manager.blocks)
        report['attention_checks'] = 0
        report['attention_max_abs_error'] = 0.
        body_calls = [0]
        runner.model.register_forward_pre_hook(lambda *unused: body_calls.__setitem__(0,body_calls[0]+1))

        def wrap_attention(module):
            original = module.forward
            def checked(q,k,v):
                actual = original(q,k,v)
                ctx = get_context()
                if not ctx.is_prefill:
                    return actual
                qs,ks = ctx.cu_seqlens_q.tolist(),ctx.cu_seqlens_k.tolist()
                if module.k_cache.numel():
                    slots = ctx.slot_mapping.long()
                    written_k = module.k_cache.view(-1,k.shape[1],k.shape[2])[slots]
                    written_v = module.v_cache.view(-1,v.shape[1],v.shape[2])[slots]
                    torch.testing.assert_close(written_k,k,rtol=0,atol=0)
                    torch.testing.assert_close(written_v,v,rtol=0,atol=0)
                for i in range(len(qs)-1):
                    length = ks[i+1]-ks[i]
                    if ctx.block_tables is None:
                        keys,values = k[ks[i]:ks[i+1]],v[ks[i]:ks[i+1]]
                    else:
                        block_size = module.k_cache.shape[1]
                        blocks = ctx.block_tables[i,:(length+block_size-1)//block_size].long()
                        keys = module.k_cache[blocks].flatten(0,1)[:length]
                        values = module.v_cache[blocks].flatten(0,1)[:length]
                    expected = dense_attention(q[qs[i]:qs[i+1]],keys,values,module.scale)
                    observed = actual[qs[i]:qs[i+1]].reshape_as(expected).float()
                    error = (observed-expected).abs().max().item()
                    report['attention_max_abs_error'] = max(report['attention_max_abs_error'],error)
                    torch.testing.assert_close(observed,expected,rtol=.025,atol=.025)
                    report['attention_checks'] += 1
                return actual
            module.forward = checked
        for module in runner.model.modules():
            if isinstance(module,Attention):wrap_attention(module)

        logits_rows = [None]
        original_model = runner.run_model
        def observe_model(*arguments):
            logits = original_model(*arguments)
            assert torch.isfinite(logits).all().item(), 'non-finite logits'
            logits_rows[0] = logits.shape[0]
            return logits
        runner.run_model = observe_model
        runner.sampler = lambda logits,temperatures: logits.argmax(-1)
        original_run = runner.run
        labels = {}
        def observe_run(seqs,varlen):
            before = body_calls[0]
            expected_outputs = [s.num_cached_tokens+s.num_scheduled_tokens == len(s) for s in seqs]
            phase = 'mixed' if any(s.is_prefill for s in seqs) and any(not s.is_prefill for s in seqs) else 'prefill' if varlen else 'decode'
            event = {'phase':phase,'q_lengths':[s.num_scheduled_tokens for s in seqs],
                     'cached':[s.num_cached_tokens for s in seqs], 'requests':[labels[s.seq_id] for s in seqs]}
            outputs = original_run(seqs,varlen)
            assert [x is not None for x in outputs] == expected_outputs
            assert logits_rows[0] == sum(expected_outputs)
            event['body_calls'] = body_calls[0]-before
            event['output_rows'] = logits_rows[0]
            if varlen or args.execution=='eager':assert event['body_calls']==1
            else:assert event['body_calls']==0  # captured Graph replay, not eager body
            report['events'].append(event)
            return outputs
        runner.run = observe_run
        def add(label,prompt,outputs):
            engine.add_request(prompt,SamplingParams(max_tokens=outputs,ignore_eos=True))
            seq=engine.scheduler.waiting[-1];labels[seq.seq_id]=label
            return seq
        a=add('A',[42+i%13 for i in range(255)],8)
        for _ in range(10):
            if a.num_completion_tokens>=2:break
            engine.step()
        assert a.num_completion_tokens==2
        bp=[83+i%11 for i in range(257)]
        b=add('B',bp,4)
        for _ in range(40):
            if engine.is_finished():break
            engine.step()
        check_idle(engine)
        assert (a.num_completion_tokens,b.num_completion_tokens)==(8,4)
        c=add('C',bp,1)  # reuse B's computed 256-token prefix after its completion
        engine.step()
        check_idle(engine)
        assert c.num_completion_tokens==1
        assert report['events'][-1]['cached']==[256]
        assert any(e['phase']=='mixed' for e in report['events'])
        assert any(e['output_rows']==0 for e in report['events'])
        assert any(e['phase']=='decode' for e in report['events'])
        if args.check_kv_pressure:
            from bench_baseline import reset_prefix_cache
            reset_prefix_cache(engine)
            d = add('D', [109+i%17 for i in range(511)], 4)
            for _ in range(10):
                if d.num_completion_tokens: break
                engine.step()
            assert d.num_completion_tokens == 1
            e = add('E', [151+i%19 for i in range(257)], 2)
            for _ in range(40):
                if engine.is_finished(): break
                engine.step()
            check_idle(engine)
            assert (d.num_completion_tokens,e.num_completion_tokens)==(4,2)
            assert engine.scheduler.preemptions > 0, 'pressure fixture did not exercise eviction'
            assert engine.scheduler.recomputed_tokens > 0, 'pressure fixture did not exercise recomputation'
            report['pressure'] = {'preemptions':engine.scheduler.preemptions,
                                  'evicted_cached_tokens':engine.scheduler.evicted_cached_tokens,
                                  'recomputed_tokens':engine.scheduler.recomputed_tokens}
        report['status']='complete'
        save()
        print(f"PASS: mixed one-forward, output alignment, prefix reuse, KV writes, {report['attention_checks']} FP32 attention comparisons; max abs error={report['attention_max_abs_error']}")
    except Exception as exc:
        report.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
