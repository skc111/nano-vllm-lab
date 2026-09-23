"""Single cloud entry: on-demand GPU gate, then full/on-demand at two fixed pools.

This tests a deliberately limited KV pool, NOT the maximum capacity of a 4090.
Mixed scheduling, model, trace and output work are fixed within each pair.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys


def comparison(directory, blocks):
    """Validate paired results before reporting any performance numbers."""
    names = ('full', 'on_demand')
    meta = [json.loads((directory / n / 'metadata.json').read_text()) for n in names]
    for field in ('gpu', 'versions', 'kv_pool', 'workload_sha256', 'trace_sha256',
                  'model_metadata_sha256', 'weight_file_sizes', 'git_commit', 'git_status'):
        if meta[0][field] != meta[1][field]:
            raise RuntimeError(f'paired metadata mismatch: {field}')
    report = {}
    for n, m in zip(names, meta):
        if (m['status'] != 'complete' or m['effective_kv_allocation'] != n or
                m['effective_scheduling_policy'] != 'mixed' or m['kv_pool']['num_blocks'] != blocks):
            raise RuntimeError(f'{n}: incorrect effective configuration')
        args = {k:v for k,v in m['arguments'].items() if k not in ('kv_allocation', 'output_dir')}
        other = {k:v for k,v in meta[0]['arguments'].items() if k not in ('kv_allocation', 'output_dir')}
        if args != other: raise RuntimeError('paired benchmark arguments differ')
        rounds = [json.loads((directory/n/f'measure-{i:02}'/'timeline.json').read_text()) for i in (1,2,3)]
        for r in rounds:
            if sum(q['output_tokens'] for q in r['requests']) != 128:
                raise RuntimeError('output workload mismatch')
            if not r['steps'] or r['kv_metrics']['peak_allocated_blocks'] > blocks:
                raise RuntimeError('invalid pool occupancy')
        report[n] = {
            'wall_seconds_median': statistics.median(r['wall_seconds'] for r in rounds),
            'output_tokens_per_second_median': statistics.median(r['output_tokens_per_second'] for r in rounds),
            'kv_rounds': [r['kv_metrics'] for r in rounds],
            'request_summary': json.loads((directory/n/'summary.json').read_text())['requests'],
        }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--gpu-memory-utilization', type=float, default=.85)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    repo = Path(__file__).resolve().parent
    model = args.model.expanduser().resolve()
    if not (model/'config.json').is_file(): p.error('model must be a downloaded local snapshot')
    if not 0 < args.gpu_memory_utilization < 1: p.error('invalid memory utilization')
    directory = (args.output_dir or repo/'results'/('kv-'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'run_kv_experiment.py').write_bytes(Path(__file__).read_bytes())
    common = ['--model',str(model),'--gpu-memory-utilization',str(args.gpu_memory_utilization)]
    jobs = [('correctness', [sys.executable,'-u','tests/check_mixed_gpu.py',*common,
                            '--kv-allocation','on_demand','--kv-cache-blocks','3','--check-kv-pressure',
                            '--output-dir',str(directory/'correctness')])]
    for label, blocks in [('roomy',64),('pressure',20)]:
        for allocation in ('full','on_demand'):
            jobs.append((label+'-'+allocation, [sys.executable,'-u','bench_timeline.py',*common,
                '--trace',str(repo/'benchmarks/traces/kv_pressure.json'),
                '--scheduling-policy','mixed','--execution','graph',
                '--kv-allocation',allocation,'--kv-cache-blocks',str(blocks),
                '--allow-kv-pressure','--observe-kv','--warmup','2','--repeats','3',
                '--output-dir',str(directory/label/allocation)]))
    state = {'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
             'jobs':[], 'note':__doc__}
    def save(): (directory/'suite.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for name, command in jobs:
            print(f'\n=== {name} ===\nResults: {directory}',flush=True)
            job = {'name':name,'command':command}; state['jobs'].append(job); save()
            with (directory/(name+'.log')).open('w') as log:
                with subprocess.Popen(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                    for line in process.stdout:
                        print(line,end='',flush=True);log.write(line);log.flush()
                    job['returncode'] = process.wait()
            save()
            if job['returncode']: raise RuntimeError(f'{name} failed; stopped. See {name}.log')
            if name == 'correctness':
                gate = json.loads((directory/'correctness/check.json').read_text())
                if gate['status'] != 'complete' or gate['pressure']['preemptions'] <= 0 or gate['pressure']['recomputed_tokens'] <= 0:
                    raise RuntimeError('GPU gate did not verify pressure recovery')
        results = {label:comparison(directory/label, blocks) for label,blocks in [('roomy',64),('pressure',20)]}
        (directory/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
        state['status']='complete';save()
        print(json.dumps(results,indent=2))
        print(f'DONE: {directory}\nKeep the whole directory; send comparison.json and suite.json.')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise


if __name__ == '__main__': main()
