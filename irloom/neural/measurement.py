"""Frozen program comparison: shared inputs, fresh baselines, balanced order."""
import json
import math
import random
import statistics
import time
from pathlib import Path

from irloom.compiler import ROOT, command
from .driver import CACHE, save
from .programs import RECIPES


def interleave(name, pilot_dir, search_path=None):
    pilot = json.loads((pilot_dir / 'report.json').read_text())
    config = pilot['config']
    search = json.loads(search_path.read_text()) if search_path else None
    if search and search['config'] != config:
        raise ValueError('frozen search/pilot protocol mismatch')
    output = CACHE / 'comparisons' / name
    output.mkdir(parents=True, exist_ok=False)
    logs = ROOT / 'logs/architecture' / name
    logs.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config['evaluation']['interleave_seed'])
    result = {'source_pilot':str(pilot_dir), 'source_search':str(search_path), 'config':config,
              'timing_scope':'independent C++ timer; includes native fork/exec/output; driver costs separate',
              'programs':{}, 'events':[]}
    start=time.perf_counter()
    for program in config['programs']:
        paths = {key:Path(path).parent for key,path in ((v,programs[program]) for v,programs in pilot['single_path'].items())}
        binaries={'single/'+v:path/'program' for v,path in paths.items()}
        reports={'single/'+v:path/'report.json' for v,path in paths.items()}
        if search:
            for variant,programs in search['searches'].items():
                path=Path(programs[program]['selected']['report'])
                binaries['beam/'+variant]=path.parent/'program'
                reports['beam/'+variant]=path
        binaries['native-clang-O3']=paths['default']/'reference'
        env=json.loads((paths['default']/'environment.json').read_text())
        cpu=min(env['affinity'])
        source=json.loads((paths['default']/'program-source.json').read_text())
        source_dir=(ROOT/'benchmark-data/sources/llvm-test-suite-18.1.3')
        if source.get('source_root') != 'suite':
            source_dir=source_dir/'MultiSource/Benchmarks'
        source_dir=(source_dir/source['path']).parent
        args=[x.replace('{source_dir}',str(source_dir)) for x in source['measure_args']]
        def run(binary,label,timed=False):
            index=len(result['events'])
            metric=logs/f'{index:05}.json'
            cmd=[binary,*args]
            if timed:
                cmd=[CACHE/'build/irloom-time',metric,*cmd]
            stdout=command(cmd,logs/f'{index:05}-{label}.log',result['events'],label,timeout=120,cpu=cpu)
            return stdout, json.loads(metric.read_text()) if timed else None
        expected,_=run(binaries['native-clang-O3'],'oracle')
        for label,binary in binaries.items():
            if run(binary,'warmup')[0] != expected:
                raise RuntimeError(f'{program}/{label}: frozen output mismatch')
        # Pair each shuffled order with its reversal, balancing serial drift.
        orders=[]
        for round_index in range(config['evaluation']['interleave_rounds']):
            if round_index % 2 == 0:
                order=list(binaries); rng.shuffle(order)
            else:
                order=list(reversed(orders[-1]))
            orders.append(order)
        samples={key:[] for key in binaries}
        for round_index,order in enumerate(orders):
            for label in order:
                stdout,metric=run(binaries[label],'measure',True)
                if stdout != expected:
                    raise RuntimeError(f'{program}/{label}: measured output mismatch')
                samples[label].append({**metric,'round':round_index,'order':order.index(label),
                                       'driver_wall_ms':result['events'][-1]['seconds']*1000})
        # Direct byte comparisons; no artifact hashes. Freeze first, never pick
        # a new candidate based on these independent measurements.
        artifact_groups=[]
        for label,path in binaries.items():
            content=path.read_bytes()
            group=next((g for g in artifact_groups if content==binaries[g[0]].read_bytes()),None)
            if group is None:
                artifact_groups.append([label])
            else:
                group.append(label)
        medians={key:statistics.median(s['latency_ms'] for s in row) for key,row in samples.items()}
        default_report=json.loads(reports['single/default'].read_text())
        differences={}
        for label,path in reports.items():
            report=json.loads(path.read_text())
            differences[label]={
                'equal_final_ir_to_default':Path(report['final_ir']).read_bytes()==Path(default_report['final_ir']).read_bytes(),
                'equal_object_to_default':(path.parent/'program.o').read_bytes()==(reports['single/default'].parent/'program.o').read_bytes(),
                'equal_executable_to_default':binaries[label].read_bytes()==binaries['single/default'].read_bytes()}
        ratios={label:[samples['single/default'][i]['latency_ms']/sample['latency_ms'] for i,sample in enumerate(row)] for label,row in samples.items()}
        result['programs'][program]={'split':config['splits'][program], 'family':RECIPES['programs'][program]['family'],
            'samples':samples,'orders':orders,'median_ms':medians,
            'paired_median_speedup':{key:statistics.median(row) for key,row in ratios.items()},
            'equal_executable_groups':artifact_groups,'binaries':{key:str(path) for key,path in binaries.items()},
            'artifact_comparison_to_default':differences,
            'reports':{key:str(path) for key,path in reports.items()}}
        save(output/'progress.json',result)
        print(json.dumps({'phase':'interleaved','program':program,'methods':len(binaries)}),flush=True)
    split_summary={}
    for split in ('train','validation','test'):
        programs=[p for p in result['programs'].values() if p['split']==split]
        labels=next(iter(programs))['paired_median_speedup'] if programs else {}
        split_summary[split]={label:{'geomean_speedup':math.exp(statistics.mean(math.log(p['paired_median_speedup'][label]) for p in programs)),
            'worst_speedup':min(p['paired_median_speedup'][label] for p in programs),
            'programs':len(programs),'families':len({p['family'] for p in programs})} for label in labels}
    result.update(split_summary=split_summary,wall_s=time.perf_counter()-start,
                  limitation='one seed, fixed CPU, uncontrolled frequency/load; paired rounds do not prove causality. Identical executable timing differences are noise. Stanford ancestry and one held-out family limit generalization.')
    save(output/'report.json',result)
    print(json.dumps({'interleaved_report':str(output/'report.json')}),flush=True)
    return result
