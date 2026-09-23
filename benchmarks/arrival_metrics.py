"""Analyze finite, open-loop arrival traces; no model/GPU imports.

All percentiles are descriptive sample statistics, not population tail guarantees.
"""
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics
import sys

from bench_timeline import request_metrics, validate_trace

POLICIES = ('prefill_first', 'interleave', 'mixed')


def quantile(values, q):
    """Sorted linear interpolation at (n-1)*q; empty -> None."""
    if not 0 <= q <= 1:
        raise ValueError('quantile must be between zero and one')
    values = sorted(values)
    if not values:
        return None
    if not all(math.isfinite(x) for x in values):
        raise ValueError('non-finite metric')
    pos = (len(values)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def outstanding(intervals, wall, cutoff):
    """Half-open intervals [arrival, completion); integrate across full round."""
    events = {}
    at_cutoff = 0
    for start, end in intervals:
        if not 0 <= start <= end <= wall:
            raise ValueError('invalid request lifetime')
        events[start] = events.get(start, 0)+1
        events[end] = events.get(end, 0)-1
        at_cutoff += start <= cutoff < end
    count = peak = 0
    area = previous = 0.
    for t, change in sorted(events.items()):
        area += count*(t-previous)
        count += change
        peak = max(peak, count)
        previous = t
    if count:
        raise ValueError('unfinished requests')
    return {'peak':peak, 'time_mean':area/wall, 'at_last_planned_arrival':at_cutoff}


def round_metrics(result, trace):
    trace = validate_trace(trace)
    wall = result['wall_seconds']
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError('invalid round wall time')
    requests = {q['request_id']:q for q in result['requests']}
    admissions = {q['request_id']:q for q in result['admissions']}
    ids = {q['id'] for q in trace}
    if len(requests) != len(result['requests']) or len(admissions) != len(result['admissions']) or set(requests) != ids or set(admissions) != ids:
        raise ValueError('missing/duplicate request or admission')
    tokens = {label:[] for label in ids}
    for token in result['tokens']:
        if token['request_id'] not in tokens:
            raise ValueError('unknown token request')
        tokens[token['request_id']].append(token)
    steps = result['steps']
    if not steps:
        raise ValueError('empty steps')
    by_step = [0]*len(steps)
    gaps = {}
    for spec in trace:
        label = spec['id']; ts = tokens[label]
        if [t['token_index'] for t in ts] != list(range(1,spec['output_len']+1)):
            raise ValueError('missing/duplicate output token')
        times = [t['observed_s'] for t in ts]
        if times != sorted(times) or times[-1] > wall:
            raise ValueError('invalid output timestamp')
        admitted = admissions[label]['admitted_s']
        if admitted < spec['arrival_s'] or admitted > times[0]:
            raise ValueError('invalid admission timestamp')
        if request_metrics(spec, admitted, times) != requests[label]:
            raise ValueError('request summary differs from raw tokens')
        gaps[label] = [b-a for a,b in zip(times,times[1:])]
        for t in ts:
            i = t['step']
            if not 0 <= i < len(steps) or t['observed_s'] != steps[i]['end_s']:
                raise ValueError('token time differs from step end')
            by_step[i] += 1
    for i,s in enumerate(steps):
        if s['step'] != i or s['new_output_tokens'] != by_step[i]:
            raise ValueError('step output count mismatch')
        if not 0 <= s['start_s'] <= s['end_s'] <= wall or (i and s['start_s'] < steps[i-1]['end_s']):
            raise ValueError('invalid step clock')
        if not math.isclose(s['wall_seconds'], s['end_s']-s['start_s'], abs_tol=1e-12):
            raise ValueError('step duration mismatch')
    output = sum(q['output_len'] for q in trace)
    if not math.isclose(output/wall,result['output_tokens_per_second']):
        raise ValueError('throughput mismatch')
    groups = {}
    for group in ('all','short','long'):
        labels = [q['id'] for q in trace if group=='all' or (q['input_len']<=128)==(group=='short')]
        if not labels:
            continue
        rows = [requests[label] for label in labels]
        group_gaps = [g for label in labels for g in gaps[label]]
        groups[group] = {'requests':len(labels), 'itl_samples':len(group_gaps)}
        for key, field in [('ttft_planned','ttft_from_planned_s'),('ttft_admitted','ttft_from_admission_s'),
                           ('admission_lag','admission_lag_s'),('request_latency','completion_from_planned_s'),
                           ('tpot','tpot_s'),('request_max_itl','itl_max_s')]:
            values = [r[field] for r in rows if r[field] is not None]
            for p in (50,95):
                value = quantile(values,p/100)
                groups[group][f'{key}_p{p}_ms'] = None if value is None else value*1000
        for p in (50,95):
            value = quantile(group_gaps,p/100)
            groups[group][f'token_itl_p{p}_ms'] = None if value is None else value*1000
    cutoff = max(q['arrival_s'] for q in trace)
    planned = outstanding([(r['planned_arrival_s'],r['last_token_s']) for r in requests.values()],wall,cutoff)
    admitted = outstanding([(r['admitted_s'],r['last_token_s']) for r in requests.values()],wall,cutoff)
    kv = result.get('kv_metrics',{})
    if not kv:
        raise ValueError('KV observations required for pressure attribution')
    expected_kv = {'peak_allocated_blocks':max(s['allocated_blocks_before_forward'] for s in steps),
                   'peak_unwritten_blocks':max(s['unwritten_blocks_before_forward'] for s in steps),
                   'preemptions':steps[-1]['preemptions_total'],
                   'evicted_cached_tokens':steps[-1]['evicted_cached_tokens_total'],
                   'recomputed_tokens':steps[-1]['recomputed_tokens_total']}
    if kv != expected_kv:
        raise ValueError('KV summary mismatch')
    return {'wall_seconds':wall, 'output_tokens':output,'output_tokens_per_second':output/wall,
            'completed_requests_per_second':len(trace)/wall,
            'arrival_span_seconds':cutoff,'drain_after_last_arrival_seconds':wall-cutoff,
            'planned_outstanding_peak':planned['peak'], 'planned_outstanding_time_mean':planned['time_mean'],
            'admitted_outstanding_peak':admitted['peak'],
            'unfinished_at_last_arrival':planned['at_last_planned_arrival'],
            'waiting_queue_peak':max(max(s['waiting_before'],s['waiting_after']) for s in steps),
            'min_free_blocks_after_step':min(s['free_blocks_after'] for s in steps),
            'mixed_steps':sum(s['phase']=='mixed' for s in steps), 'total_steps':len(steps),
            **kv,'groups':groups}


def spread(values):
    if all(v is None for v in values):
        return {'median':None,'min':None,'max':None}
    if any(v is None for v in values):
        raise ValueError('inconsistent metric availability across rounds')
    return {'median':statistics.median(values),'min':min(values),'max':max(values)}


def aggregate(rounds):
    if not rounds:
        raise ValueError('no measured rounds')
    result = {'measured_rounds':len(rounds), 'scalars':{}, 'groups':{}}
    for k in rounds[0]:
        if k != 'groups':result['scalars'][k]=spread([r[k] for r in rounds])
    for group,metrics in rounds[0]['groups'].items():
        result['groups'][group]={k:spread([r['groups'][group][k] for r in rounds]) for k in metrics}
    return result


def compatible(metadata):
    common = ('gpu','versions','kv_pool','model_dtype','model_metadata_sha256','weight_file_sizes',
              'git_commit','git_status','sampling','cache_policy','timing_scope')
    first=metadata[0]
    for m in metadata:
        if m['status']!='complete' or m['effective_kv_allocation']!='full':
            raise ValueError('incomplete run or wrong KV allocation')
        for k in common:
            if m[k]!=first[k]:raise ValueError(f'incompatible metadata: {k}')


def summarize_suite(directory):
    directory=Path(directory)
    state=json.loads((directory/'suite.json').read_text())
    report={'scope':'Finite periodic arrival traces, not a steady-state capacity/SLO measurement. No p99 claims.',
            'percentiles':'Linear interpolation within each round, then median/min/max across repeated identical traces. Do not pool repeats as independent requests.',
            'itl':'token_itl is token-weighted; request_max_itl gives each request one maximum gap.',
            'backlog':'planned_outstanding includes requests whose planned arrival passed while the single-thread replay was inside a step; admitted_outstanding does not.',
            'throughput':'Full cohort output tokens / wall time including arrival gaps and drain, not pure GPU speed.',
            'cases':{}}
    all_meta=[];flat=[]
    for case in state['cases']:
        name=case['name'];trace=json.loads((directory/'traces'/f'{name}.json').read_text())
        entries={};metas=[]
        for policy in POLICIES:
            path=directory/name/policy
            m=json.loads((path/'metadata.json').read_text());metas.append(m);all_meta.append(m)
            if m['effective_scheduling_policy']!=policy:raise ValueError('policy mismatch')
            for file,key in [('trace.json','trace_sha256'),('workload.json','workload_sha256')]:
                if hashlib.sha256((path/file).read_bytes()).hexdigest()!=m[key]:raise ValueError('workload hash mismatch')
            if json.loads((path/'trace.json').read_text())!=trace:raise ValueError('trace mismatch')
            expected = {'execution':'graph','kv_allocation':'full','max_num_seqs':8,
                        'max_num_batched_tokens':512,'max_model_len':4352,
                        'allow_kv_pressure':True,'observe_kv':True,'seed':0,
                        'scheduling_policy':policy,'only_request':None}
            for key in ('repeats','warmup','kv_cache_blocks','gpu_memory_utilization','timeout_s'):
                expected[key] = state['arguments'][key]
            for key,value in expected.items():
                if m['arguments'].get(key)!=value:raise ValueError(f'suite argument mismatch: {key}')
            if m['kv_pool']!={'num_blocks':expected['kv_cache_blocks'],'block_size':256}:
                raise ValueError('kv_pool differs from requested fixed pool')
            workload=json.loads((path/'workload.json').read_text())
            if [{k:v for k,v in q.items() if k!='prompt_token_ids'} for q in workload]!=trace:
                raise ValueError('workload specs differ from trace')
            if any(len(q['prompt_token_ids'])!=q['input_len'] for q in workload):
                raise ValueError('workload prompt length mismatch')
            rounds=[]
            for i in range(1,state['arguments']['repeats']+1):
                r=json.loads((path/f'measure-{i:02}'/'timeline.json').read_text())
                if r['phase']!='measure' or r['round']!=i:raise ValueError('wrong measurement round')
                rounds.append(round_metrics(r,trace))
            summary=aggregate(rounds)
            log=(directory/f'{name}--{policy}.log').read_text()
            warning_lines=[l for l in log.splitlines() if any(w in l.lower() for w in ('warning','recompile','traceback'))]
            entries[policy]={'summary':summary,'rounds':rounds,'warning_lines':warning_lines}
            for group,metrics in summary['groups'].items():
                flat.append({'case':name,'workload':case['workload'],'offered_rps':case['rate'],
                             'policy':policy,'group':group,
                             **{k:v['median'] for k,v in summary['scalars'].items()},
                             **{k:v['median'] for k,v in metrics.items()}})
        compatible(metas)
        for m in metas[1:]:
            for k in ('workload_sha256','trace_sha256'):
                if m[k]!=metas[0][k]:raise ValueError(f'paired {k} mismatch')
            args=lambda x:{k:v for k,v in x['arguments'].items() if k not in ('scheduling_policy','output_dir')}
            if args(m)!=args(metas[0]):raise ValueError('paired arguments mismatch')
        report['cases'][name]={**case,'policies':entries}
    compatible(all_meta)
    report['environment']={k:all_meta[0][k] for k in ('gpu','versions','kv_pool','git_commit','git_status')}
    (directory/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    with (directory/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=flat[0].keys());w.writeheader();w.writerows(flat)
    plot(report,directory/'mixed-arrival-curves.svg')
    return report


def plot(report,path):
    cases=sorted([c for c in report['cases'].values() if c['workload']=='mixed'],key=lambda c:c['rate'])
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1420" height="490" viewBox="0 0 1420 490">',
         '<rect width="1420" height="490" fill="white"/><g font-family="sans-serif" fill="#1d2939">']
    def text(x,y,t,size=13):svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}">{escape(t)}</text>')
    text(25,30,'Mixed arrivals: finite-cohort measurements',23)
    text(25,54,'Points = median of per-round statistics; bars = min/max across repeats (not confidence intervals).')
    colors={'prefill_first':'#c66a2c','interleave':'#287fb3','mixed':'#36845b'}
    metrics=[('Output token/s','scalars','output_tokens_per_second'),
             ('Short-request TTFT p95 (ms)','short','ttft_planned_p95_ms'),
             ('Short-request max ITL p95 (ms)','short','request_max_itl_p95_ms')]
    lo,hi=cases[0]['rate'],cases[-1]['rate']
    for panel,(title,group,key) in enumerate(metrics):
        left=65+panel*465;width=370
        x=lambda rate:left+(rate-lo)/(hi-lo)*width if hi>lo else left+width/2
        get=lambda c,p:c['policies'][p]['summary']['scalars'][key] if group=='scalars' else c['policies'][p]['summary']['groups'][group][key]
        top=max(get(c,p)['max'] for c in cases for p in POLICIES)*1.12 or 1
        y=lambda value:365-value/top*245
        text(left,94,title,15)
        for i in range(5):
            value=top*i/4;yy=y(value)
            svg.append(f'<path d="M{left},{yy:.2f} H{left+width}" stroke="#ddd"/>')
            text(left-45,yy+4,f'{value:.1f}',11)
        for c in cases:text(x(c['rate'])-10,389,str(c['rate']),12)
        text(left+105,412,'Offered requests/s')
        for policy,color in colors.items():
            points=[]
            for c in cases:
                v=get(c,policy);xx=x(c['rate']);yy=y(v['median']);points.append(f'{xx:.2f},{yy:.2f}')
                svg.append(f'<path d="M{xx:.2f},{y(v["min"]):.2f} V{y(v["max"]):.2f}" stroke="{color}" stroke-width="2"/>')
                svg.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="4" fill="{color}"/>')
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i,(policy,color) in enumerate(colors.items()):
        svg.append(f'<rect x="{40+i*220}" y="439" width="15" height="15" fill="{color}"/>')
        text(62+i*220,452,policy)
    text(25,478,'Sample p95 only; small long-request subsets. Inspect backlog, KV occupancy and logs before attributing changes.',12)
    svg.append('</g></svg>');path.write_text('\n'.join(svg))


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python -m benchmarks.arrival_metrics RESULTS_DIRECTORY')
    summarize_suite(Path(sys.argv[1]))
