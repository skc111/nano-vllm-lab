"""CPU-only workload, metric and orchestration checks for the arrival suite."""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

from bench_timeline import request_metrics
from benchmarks.arrival_metrics import aggregate, compatible, outstanding, quantile, round_metrics, summarize_suite
import run_arrival_experiment as suite


def fixture(trace):
    """Synthetic timestamps for testing analysis, never performance evidence."""
    # Each request has its own nonoverlapping toy step sequence.
    steps=[];tokens=[];requests=[];admissions=[];clock=0.
    for spec in trace:
        admitted=max(clock,spec['arrival_s'])
        clock=admitted
        admissions.append({'request_id':spec['id'],'admitted_s':admitted})
        times=[]
        for i in range(spec['output_len']):
            start=clock;clock+=.01;times.append(clock)
            tokens.append({'request_id':spec['id'],'token_index':i+1,'token_id':42,'observed_s':clock,'step':len(steps)})
            steps.append({'step':len(steps),'phase':'prefill' if i==0 else 'decode',
                'start_s':start,'end_s':clock,'wall_seconds':clock-start,'new_output_tokens':1,
                'waiting_before':1,'waiting_after':0,'free_blocks_after':63,
                'allocated_blocks_before_forward':1,'unwritten_blocks_before_forward':1 if i==0 else 0,
                'preemptions_total':0,'evicted_cached_tokens_total':0,'recomputed_tokens_total':0})
        requests.append(request_metrics(spec,admitted,times))
    wall=clock+.001
    return {'phase':'measure','round':1,'wall_seconds':wall,'requests':requests,'admissions':admissions,
            'tokens':tokens,'steps':steps,'output_tokens_per_second':len(tokens)/wall,
            'kv_metrics':{'peak_allocated_blocks':1,'peak_unwritten_blocks':1,'preemptions':0,'evicted_cached_tokens':0,'recomputed_tokens':0}}


class ArrivalTests(unittest.TestCase):
    def trace(self):
        return [{'id':'A','arrival_s':0.,'input_len':128,'output_len':3},
                {'id':'B','arrival_s':.015,'input_len':4096,'output_len':1}]

    def test_periodic_mixed_and_short_control(self):
        a=suite.make_trace('mixed',32,1.)
        b=suite.make_trace('mixed',32,4.)
        self.assertEqual(sum(q['input_len']==4096 for q in a),8)
        self.assertEqual(sum(q['output_len'] for q in a),1792)
        self.assertEqual([q['arrival_s']/4 for q in a],[q['arrival_s'] for q in b])
        self.assertEqual([{k:v for k,v in q.items() if k!='arrival_s'} for q in a],
                         [{k:v for k,v in q.items() if k!='arrival_s'} for q in b])
        self.assertTrue(all(q['input_len']==128 for q in suite.make_trace('short',32,4)))

    def test_invalid_workloads_and_rates(self):
        for count,rate in [(7,1),(9,1),(8,0),(8,float('nan')),(8,float('inf'))]:
            with self.assertRaises(ValueError):suite.make_trace('mixed',count,rate)
        for rates in ([],[1,1],[1,2,3,4,5],[1,1.00000001]):
            with self.assertRaises(ValueError):suite.make_cases(8,rates)

    def test_quantile_definition(self):
        self.assertAlmostEqual(quantile([4,1,3,2],.95),3.85)
        self.assertEqual(quantile([9],.95),9)
        self.assertIsNone(quantile([],.5))
        with self.assertRaises(ValueError):quantile([float('nan')],.5)

    def test_backlog_includes_not_yet_admitted_requests_and_ties(self):
        result=outstanding([(0,2),(1,3)],4,1)
        self.assertEqual(result,{'peak':2,'time_mean':1.,'at_last_planned_arrival':2})
        self.assertEqual(outstanding([(0,1),(1,2)],2,1)['peak'],1)
        with self.assertRaises(ValueError):outstanding([(2,1)],3,1)

    def test_groups_single_token_and_no_pooled_repeat_percentile(self):
        r=round_metrics(fixture(self.trace()),self.trace())
        self.assertEqual(r['groups']['short']['requests'],1)
        self.assertEqual(r['groups']['long']['itl_samples'],0)
        self.assertIsNone(r['groups']['long']['token_itl_p95_ms'])
        self.assertAlmostEqual(r['groups']['all']['admission_lag_p95_ms'],14.25)
        # B arrives while A still runs: planned backlog 2, actual admission peak 1.
        self.assertEqual(r['planned_outstanding_peak'],2)
        self.assertEqual(r['admitted_outstanding_peak'],1)
        other=copy.deepcopy(r);other['wall_seconds']=r['wall_seconds']*2
        summary=aggregate([r,other])
        self.assertAlmostEqual(summary['scalars']['wall_seconds']['median'],r['wall_seconds']*1.5)
        self.assertEqual(summary['groups']['all']['requests']['median'],2)

    def test_bad_or_incomplete_raw_data_rejected(self):
        for mutation in ('drop','duplicate','time','kv','throughput'):
            r=fixture(self.trace())
            if mutation=='drop':r['tokens'].pop()
            if mutation=='duplicate':r['requests'].append(r['requests'][0])
            if mutation=='time':r['tokens'][0]['observed_s']+=.001
            if mutation=='kv':r['kv_metrics']['preemptions']=1
            if mutation=='throughput':r['output_tokens_per_second']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):round_metrics(r,self.trace())

    def test_plan_only_never_starts_a_process(self):
        with patch.object(suite.subprocess,'Popen') as start,contextlib.redirect_stdout(io.StringIO()):
            result=suite.main(['--plan-only'])
        start.assert_not_called()
        self.assertEqual(result['processes'],12)
        self.assertEqual(result['arrival_windows_floor_minutes'],15.5)

    def test_command_order_and_fixed_configuration(self):
        args=SimpleNamespace(kv_cache_blocks=64,gpu_memory_utilization=.85,warmup=2,repeats=3,timeout_s=180.)
        jobs=suite.jobs_for(Path('/tmp/results'),Path('/tmp/model'),suite.make_cases(32,[1,2,4]),args)
        self.assertEqual([n for n,c in jobs[:3]],['mixed-r1--prefill_first','mixed-r1--interleave','mixed-r1--mixed'])
        self.assertEqual(jobs[3][0],'mixed-r2--interleave')
        self.assertEqual(jobs[6][0],'mixed-r4--mixed')
        for _,cmd in jobs:
            self.assertEqual(cmd[cmd.index('--kv-allocation')+1],'full')
            self.assertEqual(cmd[cmd.index('--max-num-seqs')+1],'8')
            self.assertIn('--observe-kv',cmd)

    def test_first_failed_job_stops_and_keeps_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();(model/'config.json').write_text('{}')
            dest=root/'run';proc=MagicMock();proc.__enter__.return_value=proc
            proc.stdout=iter(['failure detail\n']);proc.wait.return_value=1
            with patch.object(suite.subprocess,'Popen',return_value=proc) as start,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):suite.main(['--model',str(model),'--output-dir',str(dest)])
            self.assertEqual(start.call_count,1)
            state=json.loads((dest/'suite.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertEqual((dest/'mixed-r1--prefill_first.log').read_text(),'failure detail\n')

    def test_complete_fake_suite_and_metadata_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'traces').mkdir()
            case={'name':'mixed-r1','workload':'mixed','rate':1.,'requests':8}
            trace=suite.make_trace('mixed',8,1.)
            arguments={'repeats':1,'warmup':2,'kv_cache_blocks':64,'gpu_memory_utilization':.85,'timeout_s':180.}
            (root/'suite.json').write_text(json.dumps({'cases':[case],'arguments':arguments}))
            (root/'traces/mixed-r1.json').write_text(json.dumps(trace))
            import hashlib
            paths=[]
            for policy in suite.POLICIES:
                path=root/case['name']/policy;path.mkdir(parents=True);paths.append(path)
                (path/'trace.json').write_text(json.dumps(trace))
                (path/'workload.json').write_text(json.dumps([dict(q,prompt_token_ids=[42]*q['input_len']) for q in trace]))
                meta={k:'same' for k in ('gpu','versions','kv_pool','model_dtype','model_metadata_sha256',
                                       'weight_file_sizes','git_commit','git_status','sampling','cache_policy','timing_scope')}
                meta.update(status='complete',effective_kv_allocation='full',effective_scheduling_policy=policy,
                    kv_pool={'num_blocks':64,'block_size':256},
                    arguments={**arguments,'scheduling_policy':policy,'output_dir':str(path),
                               'execution':'graph','kv_allocation':'full','max_num_seqs':8,
                               'max_num_batched_tokens':512,'max_model_len':4352,
                               'allow_kv_pressure':True,'observe_kv':True,'seed':0,'only_request':None})
                for f,k in [('trace.json','trace_sha256'),('workload.json','workload_sha256')]:
                    meta[k]=hashlib.sha256((path/f).read_bytes()).hexdigest()
                (path/'metadata.json').write_text(json.dumps(meta))
                (path/'measure-01').mkdir();(path/'measure-01/timeline.json').write_text(json.dumps(fixture(trace)))
                (root/(case['name']+'--'+policy+'.log')).write_text('log\n')
            report=summarize_suite(root)
            self.assertEqual(report['cases']['mixed-r1']['policies']['mixed']['summary']['measured_rounds'],1)
            ET.parse(root/'mixed-arrival-curves.svg')
            self.assertTrue((root/'comparison.csv').is_file())
            m=json.loads((paths[-1]/'metadata.json').read_text())
            wrong=copy.deepcopy(m);wrong['arguments']['execution']='eager'
            (paths[-1]/'metadata.json').write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'execution'):summarize_suite(root)
            m=json.loads((paths[-1]/'metadata.json').read_text());m['kv_pool']='different'
            m['arguments']['execution']='graph'
            (paths[-1]/'metadata.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'kv_pool'):summarize_suite(root)


if __name__=='__main__':unittest.main()
