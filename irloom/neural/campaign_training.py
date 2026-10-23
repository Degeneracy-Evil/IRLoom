"""Bounded development fits and four matched fixed-final main models."""
import json
import statistics
import time

import torch

from .driver import CACHE,save
from .interleaved import Interleaved
from .recurrent import Recurrent
from .stage_learning import prepare_groups,forward_prefix
from .training_review import outputs,tensors
from .cost_objective import cost_parts


def fit(output,key,kind,history,dataset,objective,seed,budget,scope='mixed'):
    path=output/(key+'.json');weights=output/(key+'.pt')
    if path.exists() and weights.exists():return json.loads(path.read_text())
    if path.exists() or weights.exists() or any(output.glob(key+'-step*.pt')):
        raise RuntimeError('partial fit; use a new campaign name: '+key)
    start=time.perf_counter();torch.manual_seed(seed)
    config={'hidden':128,'heads':4,'history':history,'encoder_version':'hierarchical-v3','decision_version':'stage-q-v1'}
    model=(Interleaved if kind=='A' else Recurrent)(**config)
    train=prepare_groups(dataset,'train',objective);validation=prepare_groups(dataset,'validation',objective)
    if scope=='single':train=[g for g in train if g['program']=='pc1' and g['stage']=='S0']
    optimizer=torch.optim.Adam(model.parameters(),lr=objective['learning_rate'])
    snapshots={};curves=[];programs=list(dict.fromkeys(g['program'] for g in train));visits={p:0 for p in programs}
    def snapshot(step):
        model.eval()
        with torch.no_grad():row={'optimizer_steps':step,'train':outputs(model,train,objective),'validation':outputs(model,validation,objective)}
        snapshots[str(step)]=row
        torch.save({'model':kind,'config':config,'optimizer_steps':step,'state_dict':model.state_dict()},output/f'{key}-step{step}.pt')
        save(output/f'{key}-step{step}.json',row);model.costs.clear()
    snapshot(0)
    for step in range(1,budget['updates']+1):
        if budget.get('groups_per_update'):
            batch=[]
            for j in range(min(budget['groups_per_update'],len(programs))):
                p=programs[((step-1)*budget['groups_per_update']+j)%len(programs)]
                groups=[g for g in train if g['program']==p]
                batch.append(groups[visits[p]%len(groups)]);visits[p]+=1
        else:batch=train
        optimizer.zero_grad();model.train();per_program={};group_rows=[];scores_by_stage={};head_gradients=[]
        for group in batch:
            (scores,_,_),_=forward_prefix(model,group);scores.retain_grad()
            scores_by_stage.setdefault(group['stage'],[]).append(scores)
            predicted=torch.stack([-scores[r['candidate_index']] for r in group['rows']]);y,m=tensors(group)
            interval,pair=cost_parts(predicted,y,m,group['pairs'],objective['loss_version']);loss_group=interval+pair
            per_program.setdefault(group['program'],[]).append(loss_group)
            group_rows.append({'program':group['program'],'stage':group['stage'],'prefix':group['prefix'],
                'interval':float(interval.detach()),'pair':float(pair.detach()),'strong_pairs':len(group['pairs'])})
            if not budget.get('groups_per_update') and step in (1,budget['updates']):
                gradients=torch.autograd.grad(loss_group,tuple(model.heads.q.parameters()),retain_graph=True)
                head_gradients.append(torch.cat([g.flatten() for g in gradients]).detach())
        loss=torch.stack([torch.stack(rows).mean() for rows in per_program.values()]).mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training objective; main checkpoint not published')
        for row in group_rows:row['objective_weight']=1/(len(per_program)*len(per_program[row['program']]))
        conflict=None
        if len(head_gradients)>1:
            a=head_gradients[0];b=torch.stack(head_gradients[1:]).mean(0)
            conflict={'q_head_cosine_pc1_S0_vs_other_groups':float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
                'pc1_q_head_l2':float(a.norm()),'others_q_head_l2':float(b.norm()),'interpretation':'unweighted diagnostic, not a causal explanation'}
        loss.backward()
        gradient={stage:{'l1':sum(float(s.grad.abs().sum()) for s in values),
            'nonzero_outputs':sum(int((s.grad!=0).sum()) for s in values)} for stage,values in scores_by_stage.items()}
        optimizer.step()
        curves.append({'optimizer_steps':step,'total':float(loss.detach()),'groups':group_rows,'gradient':gradient,'shared_gradient_diagnostic':conflict})
        if step in budget['snapshots']:snapshot(step)
        model.costs.clear()
    if str(budget['updates']) not in snapshots:snapshot(budget['updates'])
    torch.save({'model':kind,'config':config,'optimizer_steps':budget['updates'],'state_dict':model.state_dict()},weights)
    result={'model':kind,'config':config,'seed':seed,'scope':scope,'objective':objective,'budget':budget,
        'optimizer_steps':budget['updates'],'selection':'fixed-final','curves':curves,'snapshots':snapshots,
        'program_visits':visits,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters())}
    save(path,result);print(json.dumps({'campaign_fit':key,'updates':budget['updates'],'seconds':result['seconds']}),flush=True)
    return result


def diagnose(output,config):
    target=output/'development';target.mkdir(exist_ok=True)
    path=target/'diagnosis.json'
    if path.exists():return json.loads(path.read_text())
    plan=config['development'];dataset=json.loads((CACHE/plan['dataset']).read_text());records={}
    for kind in ('A','B'):
        for scope,rate in [('single',plan['learning_rates'][0])]+[('mixed',rate) for rate in plan['learning_rates']]:
            key=f'{kind}-{scope}-lr{rate}'
            if kind=='A' and scope=='single':
                old=json.loads((CACHE/plan['reuse_A_single']).read_text())
                assert old['optimizer_steps']==plan['updates'] and old['objective']['learning_rate']==rate
                records[key]={'reused':plan['reuse_A_single'],**old};continue
            objective={**config['objective'],'learning_rate':rate}
            records[key]=fit(target,key,kind,True,dataset,objective,plan['seed'],plan,scope)
    scores=[]
    for rate in plan['learning_rates']:
        values=[records[f'{kind}-mixed-lr{rate}']['snapshots'][str(plan['updates'])]['validation']['metrics'] for kind in ('A','B')]
        score=(statistics.mean(v['regret'] or 0 for v in values),statistics.mean(v['q_mae'] for v in values))
        scores.append((score,rate))
    selected=min(scores,key=lambda x:x[0])[1]
    result={'protocol':plan,'models':records,'selected_learning_rate':selected,'selection_scores':scores,
        'new_training_s':sum(v['seconds'] for v in records.values() if not v.get('reused')),
        'reused_training_s':sum(v['seconds'] for v in records.values() if v.get('reused')),
        'scope':'only existing train/validation; reused single A not counted as new training; no test access'}
    save(path,result);return result


def train(output,dataset,config,diagnosis):
    target=output/'models';target.mkdir(exist_ok=True);path=target/'training.json'
    if path.exists():return json.loads(path.read_text())
    objective={**config['objective'],'learning_rate':diagnosis['selected_learning_rate']};records={}
    for seed in config['training']['seeds']:
        for key,recipe in config['models'].items():
            name=f'{key}-s{seed}'
            records[name]=fit(target,name,recipe['kind'],recipe['history'],dataset,objective,seed,config['training'])
        if seed==config['training']['seeds'][0]:
            stability=target/'first-seed-stability.json'
            if not stability.exists():
                from .campaign_evaluation import matched
                replay_target=output/'evaluation';replay_target.mkdir(exist_ok=True)
                groups=[g for g in prepare_groups(dataset,'train',objective) if g['program']=='pc1']
                checks=[]
                for key in config['models']:
                    name=f'{key}-s{seed}'
                    for ordinal in (0,2):
                        if ordinal<len(groups):checks.append(matched(replay_target,name,target/(name+'.pt'),groups[ordinal]))
                save(stability,{'seed':seed,'matched_preflight':checks,'scope':'train-only compiler/finite-output preflight before second seed; no performance-based recipe changes'})
    result={'models':records,'objective':objective,'seconds':sum(v['seconds'] for v in records.values()),
        'selection':'fixed final common200 steps; G2 selects one common LR; second seed prelisted; test not used'}
    save(path,result);return result
