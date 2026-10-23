"""V1.3.1: fixed update budget and shared-root pair resolution on frozen data."""
import json
import math
from pathlib import Path
import time

import torch

from .driver import ROOT,CACHE,save
from .cost_objective import cost_loss,cost_parts
from .stage_learning import prepare_groups,forward_prefix,quality
from .revision import load_model
from .interleaved import Interleaved
from .recurrent import Recurrent


def data_audit(dataset,config):
    records=[]
    for split in ('train','validation'):
        old=prepare_groups(dataset,split,config)
        new=prepare_groups(dataset,split,{**config,'pair_version':'shared-root-v1'})
        for a,b in zip(old,new):
            details=[]
            new_by_pair={(i,j):(direction,gamma) for i,j,direction,gamma in b['pairs']}
            for i,j,direction in a['pairs']:
                y=torch.tensor([r['target'] for r in a['rows']],dtype=torch.float64)
                m=torch.tensor([r['margin'] for r in a['rows']],dtype=torch.float64)
                q=torch.zeros_like(y,requires_grad=True)
                interval,rank=cost_parts(q,y,m,[(i,j,direction)],'calibrated-margin-v1')
                gradient=torch.autograd.grad(rank,q)[0]
                reference=json.loads(Path(a['reference_report']).read_text())
                samples=[v['latency_ms'] for v in reference['measurement']['samples']]
                from .guided import tolerance
                root_delta=tolerance(samples,samples,config);base=a['root_reference_ms']
                details.append({'i':i,'j':j,'old_effective_gap':max(0,abs(float(y[i]-y[j]))-float(m[i]+m[j])),
                    'old_tie_interval_loss':float(interval.detach()),'old_tie_rank_loss':float(rank.detach()),'old_tie_gradient_l1':float(gradient.abs().sum()),
                    'sum_absolute_margins':float(m[i]+m[j]),'root_only_log_radius':math.log((base+root_delta)/max(base-root_delta,1e-9)),
                    'canonical_pair':new_by_pair.get((i,j))})
            for i,j,direction,gamma in b['pairs']:
                y=torch.tensor([r['target'] for r in b['rows']],dtype=torch.float64,requires_grad=True)
                m=torch.tensor([r['margin'] for r in b['rows']],dtype=torch.float64)
                loss=cost_loss(y,y.detach(),m,[(i,j,direction,gamma)],'shared-root-margin-v1')
                assert loss==0 and torch.autograd.grad(loss,y)[0].abs().sum()==0
                q=torch.zeros_like(y,requires_grad=True)
                loss=cost_loss(q,y.detach(),m,[(i,j,direction,gamma)],'shared-root-margin-v1')
                assert loss>0 and torch.autograd.grad(loss,q)[0].abs().sum()>0
            unique=[]
            for row in a['rows']:
                if not any(row['object']==content for content in unique):unique.append(row['object'])
            records.append({'program':a['program'],'stage':a['stage'],'prefix':a['prefix'],'split':split,
                'rows':len(a['rows']),'distinct_objects':len(unique),
                'old_strong':len(a['pairs']),'old_positive_gaps':sum(d['old_effective_gap']>0 for d in details),
                'canonical_strong':len(b['pairs']),'canonical_positive_gaps':sum(p[3]>0 for p in b['pairs']), 'pairs':details})
    # 100/97 ms: absolute intervals may overlap, but the confirmed pair must
    # remain directional after the common root cancels.
    from .pair_labels import canonicalize
    rows=[{'candidate_index':0,'target':0.0,'samples':[100.0]*6,'object':b'default'},
          {'candidate_index':1,'target':math.log(.97),'samples':[97.0]*6,'object':b'candidate'}]
    pairs,_=canonicalize(rows,100.0,config)
    y=torch.tensor([0.0,math.log(.97)],dtype=torch.float64)
    margins=torch.tensor([math.log(102/98)]*2,dtype=torch.float64)
    tied=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    old=cost_loss(tied,y,margins,[(1,0,-1)],'calibrated-margin-v1')
    assert old==0
    new=cost_loss(tied,y,margins,pairs,'shared-root-margin-v1')
    gradient=torch.autograd.grad(new,tied)[0]
    assert new>0 and gradient[1]>0 and gradient[0]<0
    true=y.clone().requires_grad_();truth=cost_loss(true,y,margins,pairs,'shared-root-margin-v1')
    assert truth==0 and torch.autograd.grad(truth,true)[0].abs().sum()==0
    neutral=torch.zeros(2,requires_grad=True)
    assert cost_loss(neutral,torch.zeros(2),torch.ones(2)*.05,[],'shared-root-margin-v1')==0
    return {'groups':records,'scalar_fixture_not_performance':{'times_ms':[100,97],'threshold_ms':2,
        'old_tied_loss':float(old.detach()),'new_tied_loss':float(new.detach()),'new_tied_gradient':gradient.tolist(),
        'gamma':pairs[0][3],'true_q_zero_loss_and_gradient':True},'note':'Canonical artifact targets, pair membership, sign and gamma use the same times; old measurements untouched.'}


def tensors(group):
    return torch.tensor([r['target'] for r in group['rows']]),torch.tensor([r['margin'] for r in group['rows']])


def step_loss(model,groups,version,backward=False):
    per_program={};regressions=[];rankings=[];scores_by_stage={}
    for group in groups:
        (scores,_,_),_=forward_prefix(model,group)
        if backward:scores.retain_grad();scores_by_stage.setdefault(group['stage'],[]).append(scores)
        predicted=torch.stack([-scores[r['candidate_index']] for r in group['rows']])
        y,m=tensors(group);reg,rank=cost_parts(predicted,y,m,group['pairs'],version)
        per_program.setdefault(group['program'],[]).append(reg+rank)
        regressions.append(reg);rankings.append(rank)
    loss=torch.stack([torch.stack(v).mean() for v in per_program.values()]).mean()
    parts={'interval':float(torch.stack(regressions).mean().detach()),'pair':float(torch.stack(rankings).mean().detach())}
    return loss,parts,scores_by_stage


def outputs(model,groups,config):
    rows=[]
    for group in groups:
        (scores,_,_),_=forward_prefix(model,group)
        selected=max(group['rows'],key=lambda row:float(scores[row['candidate_index']]))
        pair_correct=sum(float(scores[group['rows'][i]['candidate_index']])*direction < float(scores[group['rows'][j]['candidate_index']])*direction for i,j,direction,*rest in group['pairs'])
        # score=-q: slower i must have a lower score than faster j.
        rows.append({'program':group['program'],'stage':group['stage'],'prefix':group['prefix'],
            'candidate_indices':[r['candidate_index'] for r in group['rows']], 'predicted_q':[-float(scores[r['candidate_index']]) for r in group['rows']],
            'targets':[r['target'] for r in group['rows']],'margins':[r['margin'] for r in group['rows']],
            'selected_known_index':selected['candidate_index'],'selected_full_index':int(scores.argmax()),
            'strong_pairs':len(group['pairs']),'correct_pair_directions':pair_correct,'ranking_evaluable':bool(group['pairs'])})
    return {'metrics':quality(model,groups,config),'groups':rows}


def train_one(output,key,kind,model_config,dataset,objective,protocol,diagnostic=False):
    torch.manual_seed(protocol['seed'])
    model=(Interleaved if kind=='A' else Recurrent)(**model_config)
    train=prepare_groups(dataset,'train',objective);validation=prepare_groups(dataset,'validation',objective)
    common={**objective,'pair_version':'shared-root-v1'}
    common_train=prepare_groups(dataset,'train',common);common_validation=prepare_groups(dataset,'validation',common)
    if diagnostic:
        train=[g for g in train if g['program']==protocol['overfit']['program']][protocol['overfit']['group_ordinal']:][:1]
        common_train=train;validation=[];common_validation=[]
    optimizer=torch.optim.Adam(model.parameters(),lr=objective['learning_rate'])
    snapshots={};curves=[];start=time.perf_counter();earliest=None;best=math.inf
    def snapshot(step):
        model.eval()
        with torch.no_grad():
            result={'optimizer_steps':step,'train':outputs(model,train,objective),'validation':outputs(model,validation,objective) if validation else None,
                'canonical_train':outputs(model,common_train,common),'canonical_validation':outputs(model,common_validation,common) if validation else None}
        torch.save({'model':kind,'config':model_config,'optimizer_steps':step,'state_dict':model.state_dict()},output/f'{key}-step{step}.pt')
        save(output/f'{key}-step{step}.json',result);snapshots[str(step)]=result
        model.costs.clear()
    snapshot(0)
    for step in range(1,protocol['updates']+1):
        model.train();optimizer.zero_grad()
        loss,parts,stage=step_loss(model,train,objective['loss_version'],True)
        loss.backward()
        gradient={s:{'l1':sum(float(x.grad.abs().sum()) for x in xs),
                     'nonzero_outputs':sum(int((x.grad!=0).sum()) for x in xs)} for s,xs in stage.items()}
        optimizer.step();model.eval()
        with torch.no_grad():validation_metric=quality(model,validation,objective) if validation else None
        score=validation_metric['regret'] if validation else math.inf
        if validation and (score is None or score<best):
            best=math.inf if score is None else score;earliest=step
            torch.save({'model':kind,'config':model_config,'optimizer_steps':step,'state_dict':model.state_dict()},output/f'{key}-earliest-diagnostic.pt')
        curves.append({'optimizer_steps':step,'total':float(loss.detach()),**parts,'gradient':gradient,'validation':validation_metric})
        if step in protocol['snapshots'] or step==protocol['updates']:snapshot(step)
        model.costs.clear()
    # Main output is always the actual final weights; never load earliest here.
    torch.save({'model':kind,'config':model_config,'optimizer_steps':protocol['updates'],'state_dict':model.state_dict()},output/f'{key}.pt')
    result={'model':kind,'config':model_config,'objective':objective,'optimizer_steps':protocol['updates'],
        'selection':'fixed-final-v1','earliest_diagnostic_steps':earliest,'curves':curves,'snapshots':snapshots,
        'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters())}
    save(output/f'{key}.json',result)
    return result


def run(args):
    torch.set_num_threads(1)
    protocol=json.loads((args.training_config.resolve() if args.training_config else ROOT/'experiments/architecture/training-v1-3-1.json').read_text())
    dataset=json.loads((CACHE/protocol['dataset']).read_text())
    output=CACHE/'training-reviews'/args.name
    output.mkdir(parents=True,exist_ok=True)
    config_path=output/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text())!=protocol:raise ValueError('training review protocol mismatch')
    save(config_path,protocol)
    if args.training_review in ('audit','all'):
        path=output/'data-audit.json'
        if not path.exists():save(path,data_audit(dataset,dataset['config']['training']))
        print(json.dumps({'data_audit':str(path)}),flush=True)
    if args.training_review=='audit':return
    records={}
    for key,recipe in protocol['models'].items():
        path=output/(key+'.json');weights=output/(key+'.pt')
        objective={**dataset['config']['training'],'learning_rate':protocol['learning_rate'],
                   'pair_version':recipe['pair_version'],'loss_version':recipe['loss_version']}
        model_config={'hidden':128,'heads':4,'history':recipe['history'],'encoder_version':'hierarchical-v3','decision_version':'stage-q-v1'}
        if path.exists() and weights.exists():records[key]=json.loads(path.read_text())
        elif path.exists() or weights.exists():raise RuntimeError('partial training cache; use a new run name')
        else:
            records[key]=train_one(output,key,recipe['kind'],model_config,dataset,objective,protocol)
            print(json.dumps({'trained':key,'deployed_optimizer_steps':records[key]['optimizer_steps']}),flush=True)
    if not (output/'training.json').exists():save(output/'training.json',{'protocol':protocol,'models':records})
    diagnostic=protocol.get('overfit',{})
    if diagnostic.get('enabled_if_A_final_misses_training_preference'):
        groups=records['A']['snapshots'][str(protocol['updates'])]['canonical_train']['groups']
        group=[g for g in groups if g['program']==diagnostic['program']][diagnostic['group_ordinal']]
        missed=group['correct_pair_directions']<group['strong_pairs']
        path=output/'overfit-A.json'
        if missed and not path.exists():
            bounded={**protocol,'updates':diagnostic['updates'],'snapshots':diagnostic['snapshots']}
            train_one(output,'overfit-A','A',records['A']['config'],dataset,records['A']['objective'],bounded,True)
        save(output/'diagnostic-decision.json',{'trigger':'A final training strong-pair direction coverage < 100%',
            'triggered':missed,'group':group,'diagnostic_weights_never_deployed':True})
    if args.training_review=='all':
        from .training_replay import replay
        replay(output,dataset,protocol)
    print(json.dumps({'training_review':str(output)}),flush=True)
