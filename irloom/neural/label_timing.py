"""Freeze label artifacts, then interleave their common root reference."""
import json
import random
from pathlib import Path
from .driver import CACHE,save


def retime(sessions,baseline,seed,rounds):
    if rounds%2:raise ValueError('paired label orders require an even round count')
    rng=random.Random(seed);samples={i:[] for i in range(len(sessions))};orders=[]
    expected=baseline.run([baseline.output/'reference',*[x.replace('{source_dir}',str(baseline.original_source.parent)) for x in baseline.recipe['measure_args']]],'labels-oracle')
    for turn in range(rounds):
        if turn%2==0:order=list(range(len(sessions)));rng.shuffle(order)
        else:order=list(reversed(orders[-1]))
        orders.append(order)
        for index in order:
            session=sessions[index]
            args=[x.replace('{source_dir}',str(session.original_source.parent)) for x in session.recipe['measure_args']]
            metric=session.logs/f'label-{turn}.json'
            actual=session.run([CACHE/'build/irloom-time',metric,session.output/'program',*args],'label-measure')
            if actual!=expected:raise RuntimeError('interleaved label output mismatch')
            samples[index].append(json.loads(metric.read_text()))
    for index,session in enumerate(sessions):
        path=session.output/'report.json';report=json.loads(path.read_text())
        report['initial_measurement']=report['measurement']
        report['measurement']={**report['measurement'],'samples':samples[index]}
        import statistics
        report['median_latency_ms']=statistics.median(s['latency_ms'] for s in samples[index])
        report['events']=session.events
        save(path,report)
    return {'seed':seed,'orders':orders,'reports':[str(s.output/'report.json') for s in sessions],
            'default_index':sessions.index(baseline),'rounds':rounds}
