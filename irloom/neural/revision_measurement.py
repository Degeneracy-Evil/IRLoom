"""Reuse the established frozen balanced timer for raw and guarded artifacts."""
import json
from .driver import CACHE, save
from .measurement import interleave


def measure(name):
    output=CACHE/'pilots'/name
    dataset=json.loads((output/'dataset.json').read_text())
    evaluation=json.loads((output/'evaluation.json').read_text())
    config=dataset['config']
    if evaluation['config']!=config:raise ValueError('evaluation protocol changed')
    adapted={**config,'evaluation':{'interleave_seed':config['measurement']['seed'],
                                   'interleave_rounds':config['measurement']['rounds']}}
    report_path=CACHE/'comparisons'/(name+'-interleaved')/'report.json'
    if report_path.exists():
        cached=json.loads(report_path.read_text())
        if cached['config']!=adapted:raise ValueError('cached measurement protocol changed')
        return cached
    frozen=output/'frozen'
    frozen.mkdir(exist_ok=True)
    paths={'default':dataset['defaults']}
    for method,programs in evaluation['methods'].items():
        for mode in ('raw','guarded'):
            paths[mode+'/'+method]={p:row[mode]['selected']['report'] for p,row in programs.items()}
    sources={program:json.loads((CACHE/'runs'/f'{name}-{program}-root'/'program-source.json').read_text())
             for program in config['programs']}
    save(frozen/'report.json',{'config':adapted,'single_path':paths,'program_sources':sources})
    return interleave(name+'-interleaved',frozen)
