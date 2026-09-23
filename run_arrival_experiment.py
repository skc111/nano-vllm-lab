"""Finite periodic arrivals: mixed at three rates plus a short-request control.

No engine changes, package installation, model downloads, or repeated GPU gate.
Use --plan-only first to see the minimum arrival-window time without using a GPU.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

from benchmarks.arrival_metrics import POLICIES, summarize_suite


def make_trace(kind, count, rate):
    if kind not in ('mixed','short') or count < 8 or count % 4:
        raise ValueError('workload must be mixed/short; requests must be a multiple of 4 >=8')
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('arrival rate must be finite and positive')
    return [{'id':f'R{i:04d}', 'arrival_s':i/rate,
             'input_len':4096 if kind=='mixed' and i%4==3 else 128,
             'output_len':32 if kind=='mixed' and i%4==3 else 64}
            for i in range(count)]


def make_cases(count, rates):
    if not rates or len(rates)>4 or len(set(rates))!=len(rates):
        raise ValueError('choose one to four distinct rates')
    cases=[]
    for rate in sorted(rates):
        make_trace('mixed',count,rate)  # validate before creating any output
        cases.append({'name':f'mixed-r{rate:g}','workload':'mixed','rate':rate,'requests':count})
    cases.append({'name':f'short-r{max(rates):g}','workload':'short','rate':max(rates),'requests':count})
    if len({case['name'] for case in cases})!=len(cases):
        raise ValueError('rates are too close to distinguish in result directory names')
    return cases


def jobs_for(directory, model, cases, args):
    jobs=[]
    for i,case in enumerate(cases):
        # Rotate process order across cases; repeats still happen within a process.
        order=POLICIES[i%3:]+POLICIES[:i%3]
        for policy in order:
            name=case['name']+'--'+policy
            command=[sys.executable,'-u','bench_timeline.py','--model',str(model),
                '--trace',str(directory/'traces'/(case['name']+'.json')),
                '--scheduling-policy',policy,'--kv-allocation','full',
                '--execution','graph','--max-num-seqs','8','--max-num-batched-tokens','512',
                '--max-model-len','4352','--kv-cache-blocks',str(args.kv_cache_blocks),
                '--gpu-memory-utilization',str(args.gpu_memory_utilization),
                '--allow-kv-pressure','--observe-kv','--seed','0',
                '--warmup',str(args.warmup),'--repeats',str(args.repeats),
                '--timeout-s',str(args.timeout_s),
                '--output-dir',str(directory/case['name']/policy)]
            jobs.append((name,command))
    return jobs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--requests',type=int,default=32)
    parser.add_argument('--rates',nargs='+',type=float,default=[1.,2.,4.])
    parser.add_argument('--kv-cache-blocks',type=int,default=64)
    parser.add_argument('--gpu-memory-utilization',type=float,default=.85)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout-s',type=float,default=180.)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args(argv)
    try: cases=make_cases(args.requests,args.rates)
    except ValueError as exc: parser.error(str(exc))
    if args.warmup<1 or args.repeats<1 or args.kv_cache_blocks<17:
        parser.error('warmup/repeats must be positive; need at least 17 KV blocks per long request')
    if not math.isfinite(args.timeout_s) or args.timeout_s<=max((args.requests-1)/c['rate'] for c in cases):
        parser.error('timeout must exceed the last planned arrival in every case')
    if not 0<args.gpu_memory_utilization<1: parser.error('invalid memory utilization')
    floor=3*(args.warmup+args.repeats)*sum((args.requests-1)/c['rate'] for c in cases)
    plan={'cases':cases,'processes':len(cases)*3,'rounds_per_process':args.warmup+args.repeats,
          'arrival_windows_floor_minutes':floor/60,
          'note':'A lower bound, NOT a runtime estimate: add model init, compilation, drain, logging and failures. No steady-state or p99 claim.'}
    print(json.dumps(plan,indent=2),flush=True)
    if args.plan_only:return plan
    if args.model is None:parser.error('--model is required unless --plan-only')
    model=args.model.expanduser().resolve()
    if not (model/'config.json').is_file():parser.error('model must be a downloaded local snapshot')
    repo=Path(__file__).resolve().parent
    directory=(args.output_dir or repo/'results'/('arrivals-'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'traces').mkdir()
    for case in cases:
        (directory/'traces'/(case['name']+'.json')).write_text(json.dumps(make_trace(case['workload'],args.requests,case['rate']),indent=2)+'\n')
    (directory/'source').mkdir()
    for f in ('run_arrival_experiment.py','benchmarks/arrival_metrics.py'):
        (directory/'source'/Path(f).name).write_bytes((repo/f).read_bytes())
    state={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
           'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
           'cases':cases,'plan':plan,'jobs':[],
           'limitations':'Finite periodic cohorts; same trace repeated, no independent workload seeds; no network/server. Mixed also skips discarded partial-prefill logits. Fixed full KV pool may constrain high load; report pressure, do not hide it.'}
    def save():(directory/'suite.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for name,command in jobs_for(directory,model,cases,args):
            print(f'\n=== {name} ===\nResults: {directory}',flush=True)
            job={'name':name,'command':command};state['jobs'].append(job);save()
            tail=deque(maxlen=35)
            with (directory/(name+'.log')).open('w') as log:
                with subprocess.Popen(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line);log.flush();tail.append(line)
                        if line.startswith(('Loading','warmup ','measure ','Saved:')) or any(w in line.lower() for w in ('warning','recompile','traceback')):
                            print(line,end='',flush=True)
                    job['returncode']=process.wait()
            save()
            if job['returncode']:
                print(''.join(tail),flush=True)
                raise RuntimeError(f'{name} failed; stopped before later jobs; full log retained')
        report=summarize_suite(directory)
        state['status']='complete';save()
        for name,case in report['cases'].items():
            for policy,row in case['policies'].items():
                s=row['summary'];m=lambda k:s['scalars'][k]['median']
                ttft=s['groups']['short']['ttft_planned_p95_ms']['median']
                gap=s['groups']['short']['request_max_itl_p95_ms']['median']
                print(f'{name:12} {policy:13} output={m("output_tokens_per_second"):.2f}tok/s short TTFT p95={ttft:.1f}ms request-max ITL p95={gap:.1f}ms backlog_peak={m("planned_outstanding_peak"):g} preemptions={m("preemptions"):g}')
        print(f'\nDONE: {directory}\nSend comparison.csv and suite.json; keep the complete directory and logs.')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
