"""Monte Carlo rollout cost regression along each model's own continuous state."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

from .driver import CACHE, save
from .data_recovery import read, RecoverySession, REVISION
from .schema import Observation
from .interleaved import Interleaved
from .recurrent import Recurrent
from .revision import load_model
from .guided import guided


def examples(dataset, split):
    result=[]
    for program in dataset['programs']:
        if program['split']!=split:continue
        for row in program['attempts']:
            if row['status']!='correct':
                assert 'target' not in row
                continue
            report=read(row['report'])
            assert all(c['status']=='pass' for c in report['measurement']['correctness'])
            assert len(report['transitions'])==3 and all(t['success'] for t in report['transitions'])
            observations=[Observation(**o) for o in report['observations'][:2]]
            assert [o.stage for o in observations]==['S0','S1']
            assert [len(o.history) for o in observations]==[0,1]
            assert all(observations[i].graph['candidates'][s['index']]['snapshot']==s['snapshot'] for i,s in enumerate(row['steps'][:2]))
            result.append({**row,'observations':observations,'indices':[s['index'] for s in row['steps'][:2]],'target':float(row['target'])})
    if not result:raise ValueError('no valid trajectory examples for '+split)
    return result


def selected_predictions(model, row):
    state=None; outputs=[]
    for observation,index in zip(row['observations'],row['indices']):
        (scores,_,_),state=model(observation,state)
        outputs.append(-scores[index])
    return torch.stack(outputs)


def gradient_audit(model, row, beta):
    model.train();model.zero_grad()
    (_,_,_),previous=model(row['observations'][0],None)
    (scores,_,_),_=model(row['observations'][1],previous)
    loss=F.smooth_l1_loss(-scores[row['indices'][1]],torch.tensor(row['target']),beta=beta)
    derivative=torch.autograd.grad(loss,previous['h'],allow_unused=True)[0]
    magnitude=float(derivative.abs().sum()) if derivative is not None else 0.0
    assert math.isfinite(magnitude)
    assert magnitude>0 if model.history else derivative is None
    model.costs.clear()
    return {'trajectory':row['id'],'S1_loss_to_S0_hidden_l1':magnitude,'history_enabled':model.history,
            'discrete_history_visible':True,'loss':float(loss.detach()),'selected_indices':row['indices']}


def metrics(model, rows, beta):
    model.eval();records=[]
    with torch.no_grad():
        for row in rows:
            prediction=selected_predictions(model,row)
            y=torch.tensor([row['target']]*2)
            records.append({'id':row['id'],'program':row['program'],'family':row['family'],
                            'target':row['target'],'predictions':prediction.tolist(),
                            'S0_abs_error':abs(float(prediction[0])-row['target']),
                            'S1_abs_error':abs(float(prediction[1])-row['target']),
                            'huber':float(F.smooth_l1_loss(prediction,y,beta=beta)),
                            'mae':float((prediction-y).abs().mean()),'mse':float(((prediction-y)**2).mean())})
    output={'trajectories':records}
    for unit in ('program','family'):
        groups=defaultdict(list)
        for row in records:groups[row[unit]].append(row)
        summaries={name:{key:statistics.mean(r[key] for r in values) for key in ('S0_abs_error','S1_abs_error','huber','mae','mse')} for name,values in groups.items()}
        output[unit+'_macro']={key:statistics.mean(r[key] for r in summaries.values()) for key in ('S0_abs_error','S1_abs_error','huber','mae','mse')}
        output[unit+'_macro']['rmse']=math.sqrt(output[unit+'_macro']['mse'])
        output['by_'+unit]=summaries
    model.costs.clear()
    return output


def train(output, config, dataset):
    torch.set_num_threads(1)
    target=output/'models';target.mkdir(exist_ok=True)
    path=target/'training.json'
    if path.exists():
        cached=read(path)
        if cached['config']!=config:raise ValueError('rollout training protocol mismatch')
        return cached
    training=config['training'];seed=training['seed'];beta=training['huber_beta']
    train_rows=examples(dataset,'train');validation=examples(dataset,'validation')
    joint=next(r for r in train_rows if all(s['kind']!='advance' for s in r['steps'][:2]))
    programs=list(dict.fromkeys(r['program'] for r in train_rows));rng=random.Random(seed)
    cycles={p:[i for i,r in enumerate(train_rows) if r['program']==p] for p in programs}
    for values in cycles.values():rng.shuffle(values)
    schedule=[[cycles[p][step%len(cycles[p])] for p in programs] for step in range(training['updates'])]
    result={'config':config,'schedule':[[train_rows[i]['id'] for i in step] for step in schedule],'models':{}}
    for name,recipe in config['models'].items():
        weights=target/(name+'.pt');record=target/(name+'.json')
        if weights.exists() and record.exists():
            result['models'][name]=read(record);continue
        if weights.exists() or record.exists() or list(target.glob(name+'-step*.pt')):
            raise RuntimeError('partial rollout fit preserved; use a fresh pilot name')
        torch.manual_seed(seed);started=time.perf_counter()
        model_config={'hidden':training['hidden'],'heads':training['heads'],'history':recipe['history'],
                      'encoder_version':'hierarchical-v3','decision_version':'rollout-q-v1'}
        model=(Interleaved if recipe['kind']=='A' else Recurrent)(**model_config)
        provenance={'compiler_revision':REVISION,'behavior_version':config['behavior']['version'],
                    'behavior':config['behavior'],'target':'log(T_rollout/T_root_default)','dataset':str(output/'dataset.json')}
        def checkpoint(step):
            return {'model':recipe['kind'],'config':model_config,'optimizer_steps':step,
                    'rollout_protocol':provenance,'state_dict':model.state_dict()}
        audit_before=gradient_audit(model,joint,beta)
        optimizer=torch.optim.Adam(model.parameters(),lr=training['learning_rate'])
        snapshots={};curves=[];visits=Counter()
        def snapshot(step):
            snapshots[str(step)]={'train':metrics(model,train_rows,beta),'validation':metrics(model,validation,beta)}
            torch.save(checkpoint(step),target/f'{name}-step{step}.pt')
            print(__import__('json').dumps({'rollout_snapshot':name,'step':step,
                  'train_mae':snapshots[str(step)]['train']['program_macro']['mae'],
                  'validation_mae':snapshots[str(step)]['validation']['program_macro']['mae']}),flush=True)
        snapshot(0)
        for step,indices in enumerate(schedule,1):
            optimizer.zero_grad();model.train();by_family=defaultdict(lambda:defaultdict(list));stage_gradients=[]
            for index in indices:
                row=train_rows[index];visits[row['id']]+=1
                prediction=selected_predictions(model,row);prediction.retain_grad()
                loss=F.smooth_l1_loss(prediction,torch.full_like(prediction,row['target']),beta=beta)
                by_family[row['family']][row['program']].append(loss);stage_gradients.append(prediction)
            loss=torch.stack([torch.stack([torch.stack(v).mean() for v in p.values()]).mean() for p in by_family.values()]).mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite rollout regression; no final checkpoint')
            loss.backward()
            gradients=[sum(float(p.grad[i].abs()) for p in stage_gradients) for i in range(2)]
            optimizer.step()
            curves.append({'step':step,'huber':float(loss.detach()),'selected_S0_S1_gradient_l1':gradients})
            model.costs.clear()
            if step in training['snapshots']:snapshot(step)
        audit_after=gradient_audit(model,joint,beta)
        torch.save(checkpoint(training['updates']),weights)
        fitted={'model':recipe['kind'],'config':model_config,'seed':seed,'updates':training['updates'],
                'selection':'fixed-final','objective':'chosen S0/S1 Huber against one terminal rollout sample; family/program balanced',
                'snapshots':snapshots,'curves':curves,'gradient_before':audit_before,'gradient_after':audit_after,
                'trajectory_exposures':sum(visits.values()),'selected_action_label_exposures':2*sum(visits.values()),
                'visits':dict(visits),'seconds':time.perf_counter()-started,
                'parameters':sum(p.numel() for p in model.parameters())}
        save(record,fitted);result['models'][name]=fitted
        save(target/'progress.json',result)
        print(__import__('json').dumps({'rollout_fit':name,'seconds':fitted['seconds'],'gradient':audit_after}),flush=True)
    result['seconds']=sum(v['seconds'] for v in result['models'].values())
    save(path,result)
    return result


def evaluate(output, config, dataset):
    torch.set_num_threads(1)
    target=output/'evaluation';target.mkdir(exist_ok=True);path=target/'evaluation.json'
    if path.exists():return read(path)
    rows=[]
    def factory(name,program):return RecoverySession(name,program=program,compiler_revision=REVISION)
    for name in config['models']:
        model=load_model(output/'models'/(name+'.pt'))
        for program in config['evaluation']['programs']:
            for mode in config['evaluation']['modes']:
                run=output.name+'-'+name+'-'+program+'-'+mode
                cached=CACHE/'runs'/(run+'-root')/'guided.json'
                try:
                    result=read(cached) if cached.exists() else guided(run,program,model,config['evaluation']['budget'],mode,session_factory=factory)
                    assert result['scorer']=='rollout-q-v1'
                    # All multi-parent ranking uses the latest common-root q.
                    for terminal in result['terminals']:
                        if terminal['role']!='proposal':continue
                        report=read(terminal['report']);S1=report['decisions'][1]
                        index=next(i for i,c in enumerate(report['observations'][1]['graph']['candidates']) if c==S1['selected'])
                        assert terminal['guidance']==S1['scores'][index]
                    rows.append({'model':name,'program':program,'mode':mode,'status':'correct','result':result})
                except RuntimeError as error:
                    rows.append({'model':name,'program':program,'mode':mode,'status':'failed','error':str(error),
                                 'failure_report':str(CACHE/'runs'/(run+'-root')/'guided-failure.json')})
                save(target/'progress.json',{'systems':rows})
                print(__import__('json').dumps({'rollout_compile':name,'program':program,'mode':mode,'status':rows[-1]['status']}),flush=True)
    result={'systems':rows,'scope':'Seen train pc1 and validation matrix only; four frozen final models; bounded raw/guarded, no test or extra tuning.'}
    save(path,result)
    return result


def integration(output, config, dataset):
    """Actual checkpoint, failure/stale observation and sample exclusion audit."""
    path=output/'integration.json'
    if path.exists():return read(path)
    torch.set_num_threads(1)
    training=read(output/'models/training.json')
    visits=[r['visits'] for r in training['models'].values()]
    assert all(v==visits[0] for v in visits)
    matched=[]
    for a,b in [('A','A_current_only'),('B','B_current_only')]:
        left=torch.load(output/'models'/(a+'-step0.pt'),weights_only=True)['state_dict']
        right=torch.load(output/'models'/(b+'-step0.pt'),weights_only=True)['state_dict']
        assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
        matched.append([a,b])
    old_path=CACHE/'campaigns/campaign-v1/models/A-s23.pt'
    old=load_model(old_path)
    assert old.heads.decision_version=='stage-q-v1'
    checkpoint=torch.load(old_path,weights_only=True)
    checkpoint['config']['decision_version']='rollout-q-v1'
    invalid=Path(__file__).resolve().parents[2]/'tmp/loomtrace-old-as-rollout.pt'
    torch.save(checkpoint,invalid)
    try:load_model(invalid)
    except ValueError:pass
    else:raise AssertionError('old all-advance checkpoint relabeled as rollout was accepted')
    valid=examples(dataset,'train')[0]
    root=RecoverySession(output.name+'-failure-audit-root',program=valid['program'],compiler_revision=REVISION)
    observation=root.observe('S0');before=root.ir.read_bytes()
    action=next(a for a in observation.graph['candidates'] if a['kind']!='advance')
    failed=root.fork(output.name+'-failure-audit-injected')
    remaining=failed.remaining
    transition=failed.apply(action,inject_failure=True)
    assert not transition.success and failed.ir.read_bytes()==before and failed.remaining==remaining-1
    stale=root.fork(output.name+'-failure-audit-stale')
    assert stale.apply(action).success
    changed=stale.ir.read_bytes();remaining=stale.remaining
    assert not stale.apply(action).success
    assert stale.ir.read_bytes()==changed and stale.remaining==remaining-1 and root.ir.read_bytes()==before
    failed_row={'status':'failed','run':str(failed.output),'transitions':failed.transitions,'history':failed.history}
    original=next(r for p in dataset['programs'] for r in p['attempts'] if r['id']==valid['id'])
    fixture={'programs':[{'split':'train','attempts':[original,failed_row]}]}
    assert len(examples(fixture,'train'))==1
    failed_row['target']=0.0
    try:examples(fixture,'train')
    except AssertionError:pass
    else:raise AssertionError('failed path carrying a numeric target was accepted')
    events=list({e['log']:e for s in (root,failed,stale) for e in s.events}.values())
    result={'matched_initialization_pairs':matched,'common_trajectory_schedule_and_visits':True,
            'old_stage_checkpoint_loads':True,'old_checkpoint_as_rollout_rejected':True,
            'injected_transform_failure_keeps_parent':True,'stale_observation_rejected':True,
            'failed_action_attempts_charged':2,'valid_private_transform_attempts':1,
            'failed_samples_without_target_excluded':True,'failed_sample_with_target_rejected':True,
            'events':events,'command_s':sum(e['seconds'] for e in events),'new_linked_terminals':0,
            'scope':'Private checkpoint copies and one injected/stale branch; no extra performance labels or model fitting.'}
    save(path,result)
    return result
