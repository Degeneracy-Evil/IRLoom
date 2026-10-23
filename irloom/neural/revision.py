"""V1.2 phased entry; shared compiler, stage labels, frozen raw/guarded runs."""
import json
from pathlib import Path
import time

import torch

from .driver import CACHE, ROOT, save
from .guided import guided
from .interleaved import Interleaved
from .recurrent import Recurrent
from .stage_learning import collect, fit


def load_model(path, history=None):
    checkpoint=torch.load(path,weights_only=True)
    config=dict(checkpoint['config'])
    if history is not None: config['history']=history
    model=(Interleaved if checkpoint['model']=='A' else Recurrent)(**config)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model


def train(name, dataset, config):
    output=CACHE/'pilots'/name
    if (output/'training.json').exists():
        cached=json.loads((output/'training.json').read_text())
        if cached['config']!=config:raise ValueError('cached training protocol mismatch')
        return cached
    result={'config':config,'models':{}}
    for seed in config['seeds']:
        recipes=[(label,options,'stage-q-v1') for label,options in config['models'].items()]
        if seed==config['measurement']['policy_baseline_seed']:
            recipes += [('policy_'+label,config['models'][label],'policy-v1') for label in config['policy_baselines']]
        for label,options,version in recipes:
            key=f'{label}-s{seed}'; path=output/(key+'.pt'); record=output/(key+'.json')
            if path.exists() and record.exists():
                result['models'][key]=json.loads(record.read_text());continue
            if path.exists() or record.exists(): raise RuntimeError('incomplete training cache; use a fresh run name')
            torch.manual_seed(seed)
            kind='B' if label.startswith('B') or label=='policy_B' else 'A'
            model_config={'encoder_version':'structured-v2',**{k:v for k,v in options.items() if k!='loss_version'},'decision_version':version}
            model=(Interleaved if kind=='A' else Recurrent)(**model_config)
            objective={**config['training'],'loss_version':options.get('loss_version',config['training'].get('loss_version','softplus-v1'))}
            start=time.perf_counter();fitting=fit(model,dataset,objective)
            fitting.update(seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),
                           config=model_config,model=kind,seed=seed,objective=objective)
            torch.save({'model':kind,'config':model_config,'state_dict':model.state_dict()},path)
            save(record,fitting);result['models'][key]=fitting
            save(output/'training-progress.json',result)
            print(json.dumps({'phase':'stage-trained','model':key,'epoch':fitting['selected_epoch'],'seconds':fitting['seconds']}),flush=True)
    save(output/'training.json',result)
    return result


def evaluate(name, dataset, training, config):
    output=CACHE/'pilots'/name
    if (output/'evaluation.json').exists():
        cached=json.loads((output/'evaluation.json').read_text())
        if cached['config']!=config:raise ValueError('cached evaluation protocol mismatch')
        return cached
    start=time.perf_counter()
    result={'config':config,'methods':{},'model_loading_s':{}}
    variants=[(key,key,None) for key in training['models']]
    for seed in config['seeds']:
        variants += [(f'A_no_history-s{seed}',f'A-s{seed}',False),(f'B_reset-s{seed}',f'B-s{seed}',False)]
    for label,source,history in variants:
        load_start=time.perf_counter()
        model=load_model(output/(source+'.pt'),history)
        result['model_loading_s'][label]=time.perf_counter()-load_start
        result['methods'][label]={}
        for program in config['programs']:
            result['methods'][label][program]={}
            for mode in ('raw','guarded'):
                run_name=f'{name}-{label}-{program}-{mode}'
                path=CACHE/'runs'/(run_name+'-root')/'guided.json'
                if path.exists(): row=json.loads(path.read_text())
                else: row=guided(run_name,program,model,config['budget'],mode)
                result['methods'][label][program][mode]=row
                save(output/'evaluation-progress.json',result)
                print(json.dumps({'phase':'stage-evaluation','method':label,'program':program,'mode':mode,
                                  'fallback':row['fallback_to_default'],'terminals':row['actual_terminals']}),flush=True)
    result['wall_s']=time.perf_counter()-start
    save(output/'evaluation.json',result)
    return result


def run(args):
    torch.set_num_threads(1)
    config=json.loads((args.v12_config.resolve() if args.v12_config else ROOT/'experiments/architecture/pilot-v1-2.json').read_text())
    mode=args.v12
    if mode=='audit':
        from .decision_audit import audit
        audit(args.name);return
    if mode=='guard':
        if not args.weights or not args.program: raise ValueError('--v12 guard requires --program and --weights')
        result=guided(args.name,args.program,load_model(args.weights.resolve()),config['budget'])
        print(json.dumps({'selected':result['selected'],'default_paid':result['default_paid']}));return
    output=CACHE/'pilots'/args.name
    dataset_path=output/'dataset.json'
    if dataset_path.exists():
        dataset=json.loads(dataset_path.read_text())
        if dataset['config']!=config:raise ValueError('cached dataset protocol mismatch')
    elif mode in ('all','collect'):dataset=collect(args.name,config)
    else:raise FileNotFoundError('collect the named dataset first')
    if mode=='collect': return
    if mode in ('all','train'):training=train(args.name,dataset,config)
    else:training=json.loads((output/'training.json').read_text())
    if mode=='train':return
    if mode in ('all','evaluate'):evaluate(args.name,dataset,training,config)
    if mode in ('all','measure'):
        from .revision_measurement import measure
        measure(args.name)
        key='v13_report' if config.get('protocol')=='architecture-v1-3' else 'v12_report'
        print(json.dumps({key:str(CACHE/'comparisons'/(args.name+'-interleaved')/'report.json')}),flush=True)
