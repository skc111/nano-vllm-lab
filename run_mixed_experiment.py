"""One cloud command: GPU correctness gate, then same-config scheduling A/B/C.

Stops on the first failure and keeps all logs. Does not install, download, or push.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--gpu-memory-utilization',type=float,default=.85)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parent
    model=args.model.expanduser().resolve()
    if not (model/'config.json').is_file():parser.error('model must be a downloaded local snapshot')
    if not 0<args.gpu_memory_utilization<1:parser.error('invalid memory utilization')
    directory=(args.output_dir or repo/'results'/('mixed-'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    common=['--model',str(model),'--gpu-memory-utilization',str(args.gpu_memory_utilization)]
    jobs=[('correctness',[sys.executable,'-u','tests/check_mixed_gpu.py',*common,'--output-dir',str(directory/'correctness')])]
    for policy in ('prefill_first','interleave','mixed'):
        jobs.append((policy,[sys.executable,'-u','bench_timeline.py',*common,
                            '--execution','graph','--scheduling-policy',policy,
                            '--warmup','2','--repeats','3','--output-dir',str(directory/policy)]))
    state={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
           'jobs':[], 'note':'One small trace. Mixed also skips partial-prefill sampling; no scheduling-only attribution.'}
    def save():(directory/'suite.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for name,command in jobs:
            print(f'\n=== {name} ===\nResults: {directory}',flush=True)
            job={'name':name,'command':command};state['jobs'].append(job);save()
            with (directory/f'{name}.log').open('w') as log:
                with subprocess.Popen(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                    for line in process.stdout:
                        print(line,end='',flush=True);log.write(line);log.flush()
                    job['returncode']=process.wait()
            save()
            if job['returncode']:
                raise RuntimeError(f'{name} failed; stopped before later experiments. See {name}.log')
        # A changed automatic KV pool must not silently become a capacity gain.
        metadata=[json.loads((directory/p/'metadata.json').read_text()) for p in ('prefill_first','interleave','mixed')]
        for field in ('gpu','versions','kv_pool','workload_sha256','trace_sha256','model_metadata_sha256','git_commit','git_status'):
            if any(m[field]!=metadata[0][field] for m in metadata[1:]):
                raise RuntimeError(f'A/B/C metadata differ: {field}; results retained, comparison needs review')
        for policy,m in zip(('prefill_first','interleave','mixed'),metadata):
            if m['status']!='complete' or m['effective_scheduling_policy']!=policy:
                raise RuntimeError(f'{policy}: effective policy/status mismatch')
        for i in (1,2,3):
            timeline=json.loads((directory/'mixed'/f'measure-{i:02}'/'timeline.json').read_text())
            if not any(s['phase']=='mixed' and s['prefill_tokens']>0 and s['decode_tokens']>0 for s in timeline['steps']):
                raise RuntimeError(f'measure-{i:02}: no real mixed step; cannot claim mixed-batch comparison')
        state['status']='complete';save()
        print(f'\nDONE: {directory}\nSend suite.json and the three summary.json files (keep the complete directory).')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
