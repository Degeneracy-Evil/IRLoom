"""Matched real checkpoints and fixed-continuation stage cost supervision."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

import torch

from .driver import CACHE, Session, save
from .guided import complete, execute, tolerance
from .schema import Observation
from .training import comparison_groups, measured_loss


def collect(name, config):
    output=CACHE/'pilots'/name
    output.mkdir(parents=True,exist_ok=False)
    result={'config':config,'groups':[],'defaults':{},'trajectory_paths':{},'failures':[]}
    sessions=[]; start=time.perf_counter()
    result['label_timing']={}
    for program_index,program in enumerate(config['programs']):
        root=Session(name+'-'+program+'-root',program=program); sessions.append(root)
        root.observe('S0')
        baseline=root.fork(name+'-'+program+'-default'); sessions.append(baseline)
        report=complete(baseline,0)
        report['execution']={'policy':'default'};save(baseline.output/'report.json',report)
        path=str(baseline.output/'report.json')
        result['defaults'][program]=path; result['trajectory_paths'][program]=[path]
        split=config['splits'][program]
        if split not in ('train','validation'): continue
        measured_sessions=[baseline]
        structural=config['collection'].get('sampling_version')=='structure-strata-v1'
        prefixes=[('S0',root)]
        if structural:
            from .sampling import select
            choices,_=select(root.current.graph,config['collection']['s1_prefixes'],config['collection']['seed']+program_index*100)
            prefix_choices=[(f'choice-{index}',index) for index in choices]
        else:
            prefix_choices=[(policy,min(1,len(root.current.graph['candidates'])-1) if policy=='balanced' else 0)
                            for policy in config['collection']['prefix_policies']]
        for policy,index in prefix_choices:
            prefix=root.fork(name+'-'+program+'-prefix-'+policy); sessions.append(prefix)
            execute(prefix,prefix.manifest['stages'][0],index)
            prefix.observe('S1'); prefixes.append(('S1',prefix))
        for group_index,(stage,parent) in enumerate(prefixes):
            count=len(parent.current.graph['candidates'])
            if structural:
                indices,sampling=select(parent.current.graph,config['collection']['actions_per_prefix'],config['collection']['seed']+program_index*100+group_index+1)
            else:
                indices=sorted({0, min(1,count-1), count-1})[:config['collection']['actions_per_prefix']]
                sampling={'version':'legacy-first-last'}
            group={'program':program,'split':split,'stage':stage,'prefix':parent.branch,
                   'observations':parent.observations,'reference_report':path,'continuation':config['continuation'],'rows':[], 'sampling':sampling}
            for index in indices:
                child=parent.fork(f'{name}-{program}-g{group_index}-a{index}');sessions.append(child)
                try:
                    stage_index=int(stage[1:])
                    execute(child,child.manifest['stages'][stage_index],index)
                    terminal=complete(child,stage_index+1)
                    terminal['execution']={'policy':'matched-continuation'};save(child.output/'report.json',terminal)
                    if terminal['comparison_group']!=report['comparison_group']:
                        raise ValueError('matched checkpoint changed comparison group')
                    terminal_path=str(child.output/'report.json')
                    group['rows'].append({'candidate_index':index,'report':terminal_path})
                    result['trajectory_paths'][program].append(terminal_path)
                    measured_sessions.append(child)
                except RuntimeError as error:
                    result['failures'].append({'program':program,'prefix':parent.branch,'index':index,'error':str(error)})
            if not any(row['candidate_index']==0 for row in group['rows']):
                raise RuntimeError('matched prefix lacks a valid advance continuation')
            result['groups'].append(group)
            save(output/'progress.json',result)
        if structural:
            from .label_timing import retime
            if len(measured_sessions)>config['collection']['max_terminals_per_program']:
                raise RuntimeError('label terminal budget exceeded')
            result['label_timing'][program]=retime(measured_sessions,baseline,config['collection']['seed']+program_index,config['collection']['timing_rounds'])
        print(json.dumps({'phase':'matched-labels','program':program,'groups':len(prefixes)}),flush=True)
    events={e['log']:e for session in sessions for e in session.events}
    result.update(wall_s=time.perf_counter()-start,command_s=sum(e['seconds'] for e in events.values()),
                  events=list(events.values()),measurement_invocations=sum(e['stage'] in ('measure','label-measure') for e in events.values()))
    save(output/'dataset.json',result)
    return result


def prepare_groups(dataset, split, config):
    groups=[]
    for source in dataset['groups']:
        if source['split']!=split: continue
        reference=json.loads(Path(source['reference_report']).read_text())
        base=reference['median_latency_ms']; rows=[]
        for item in source['rows']:
            report=json.loads(Path(item['report']).read_text())
            if not report['measurement']['correctness'] or any(not t['success'] for t in report['transitions']):
                raise ValueError('invalid matched terminal label')
            samples=[x['latency_ms'] for x in report['measurement']['samples']]
            rows.append({**item,'latency':statistics.median(samples),'samples':samples,
                         'object':(Path(item['report']).parent/'program.o').read_bytes()})
        advance=next(r for r in rows if r['candidate_index']==0)
        pairs=[]; counts={'beneficial':0,'neutral':0,'harmful':0,'identical_pairs':0,'near_tie_pairs':0}
        for i,a in enumerate(rows):
            # Same emitted object receives the same target, irrespective of
            # launch noise. A root-identical object has normalized cost zero.
            cluster=[b['latency'] for b in rows if b['object']==a['object']]
            a['target']=math.log(statistics.median(cluster)/base)
            if a['object']==(Path(source['reference_report']).parent/'program.o').read_bytes(): a['target']=0.0
            a['margin']=math.log1p(tolerance(a['samples'],advance['samples'],config)/base)
            if config.get('interval_version')=='log-ratio-v2':
                reference_samples=[s['latency_ms'] for s in reference['measurement']['samples']]
                # Propagate frozen millisecond resolution through numerator
                # and denominator before constructing a symmetric log interval.
                center=base*math.exp(a['target'])
                delta=tolerance(a['samples'],a['samples'],config)
                delta0=tolerance(reference_samples,reference_samples,config)
                lower=math.log(max(center-delta,1e-9)/(base+delta0))
                upper=math.log((center+delta)/max(base-delta0,1e-9))
                a['margin']=max(a['target']-lower,upper-a['target'])
            difference=a['latency']-advance['latency']
            neutral=a['object']==advance['object'] or abs(difference)<=tolerance(a['samples'],advance['samples'],config)
            counts['neutral' if neutral else 'harmful' if difference>0 else 'beneficial']+=1
            for j,b in enumerate(rows[:i]):
                if a['object']==b['object']: counts['identical_pairs']+=1;continue
                if abs(a['latency']-b['latency'])<=tolerance(a['samples'],b['samples'],config):
                    counts['near_tie_pairs']+=1;continue
                pairs.append((i,j,math.copysign(1,a['latency']-b['latency'])))
        groups.append({**source,'observations':[Observation(**o) for o in source['observations']],
                       'rows':rows,'pairs':pairs,'counts':counts,'root_reference_ms':base})
    if not groups: raise ValueError('no groups in '+split)
    return groups


def forward_prefix(model, group):
    state, first=None,None
    for observation in group['observations']:
        outputs,state=model(observation,state)
        if first is None: first=state['h']
    return outputs,first


def quality(model, groups, config):
    metrics=[]; calibration=[]
    for group in groups:
        (scores,_,_),_=forward_prefix(model,group)
        rows=group['rows']; chosen=max(rows,key=lambda r:float(scores[r['candidate_index']]))
        best=min(rows,key=lambda r:r['latency'])
        if group['pairs']:
            gap=max(0,math.log(chosen['latency']/best['latency'])-math.log1p(tolerance(chosen['samples'],best['samples'],config)/best['latency']))
            metrics.append(gap)
        if model.heads.decision_version=='stage-q-v1':
            calibration.extend(abs(float(-scores[r['candidate_index']])-r['target']) for r in rows)
    return {'regret':statistics.mean(metrics) if metrics else None,'distinguishable_groups':len(metrics),
            'q_mae':statistics.mean(calibration) if calibration else None}


def fit(model, dataset, config):
    train=prepare_groups(dataset,'train',config); validation=prepare_groups(dataset,'validation',config)
    policy=model.heads.decision_version=='policy-v1'
    trajectory_groups=comparison_groups([p for program,paths in dataset['trajectory_paths'].items()
        if dataset['config']['splits'][program]=='train' for p in paths],config) if policy else None
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    best=math.inf; selected=None; curves=[]; best_state=None; cross_gradient=None
    for epoch in range(config['epochs']):
        model.train(); optimizer.zero_grad(); stage_scores={}; program_losses={}
        if policy:
            loss=measured_loss(model,trajectory_groups,config['resource_weight'])
        else:
            for group in train:
                (scores,_,_),first=forward_prefix(model,group)
                scores.retain_grad(); stage_scores.setdefault(group['stage'],[]).append(scores)
                predicted=torch.stack([-scores[r['candidate_index']] for r in group['rows']])
                targets=torch.tensor([r['target'] for r in group['rows']]); margins=torch.tensor([r['margin'] for r in group['rows']])
                from .cost_objective import cost_loss
                loss_group=cost_loss(predicted,targets,margins,group['pairs'],config.get('loss_version','softplus-v1'))
                program_losses.setdefault(group['program'],[]).append(loss_group)
                if epoch==0 and group['stage']=='S1' and model.history and cross_gradient is None:
                    gradient=torch.autograd.grad(loss_group,first,retain_graph=True,allow_unused=True)[0]
                    cross_gradient=0 if gradient is None else float(gradient.abs().sum())
            loss=torch.stack([torch.stack(rows).mean() for rows in program_losses.values()]).mean()
        loss.backward()
        gradients={stage:sum(float(scores.grad.abs().sum()) for scores in rows) for stage,rows in stage_scores.items()}
        optimizer.step(); model.eval()
        with torch.no_grad(): metric=quality(model,validation,config)
        curves.append({'epoch':epoch,'loss':float(loss.detach()),'validation':metric,'q_stage_gradient_l1':gradients})
        # No regression tie-breaker. With unresolved validation use fixed final
        # epoch; this rule was committed before labels, including test labels.
        score=metric['regret']
        if score is None or score<best:
            best=score if score is not None else math.inf;selected=epoch
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.costs.clear()
    model.load_state_dict(best_state)
    if not policy:
        assert all(any(c['q_stage_gradient_l1'].get(stage,0)>0 for c in curves) for stage in ('S0','S1'))
        if model.history: assert cross_gradient is not None and cross_gradient>0
    return {'selected_epoch':selected,'selection':'validation sampled-candidate regret; fixed final epoch if all unresolved',
            'curves':curves,'stage1_to_stage0_gradient_l1':cross_gradient,
            'coverage':{split:[{'program':g['program'],'stage':g['stage'],'prefix':g['prefix'],'labels':len(g['rows']),
                'strong_pairs':len(g['pairs']),**g['counts']} for g in groups] for split,groups in (('train',train),('validation',validation))}}
