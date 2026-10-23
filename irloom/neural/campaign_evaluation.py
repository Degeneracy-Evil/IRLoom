"""Separate closed labels, full legal matched suffixes, and raw/guarded systems."""
import json
import math
import os
import random
from pathlib import Path
import statistics
import time

import torch

from irloom.compiler import command,CommandError
from .driver import ROOT,CACHE,Session,save
from .guided import guided,complete,execute
from .programs import RECIPES
from .revision import load_model
from .stage_learning import prepare_groups,forward_prefix
from .training_replay import action_key,check_observation


def _matched(output,key,weights,group):
    path=output/(key+f'-{group["program"]}-{group["stage"]}-matched.json')
    if path.exists():return json.loads(path.read_text())
    started=time.perf_counter();name=output.parent.name+'-'+path.stem;model=load_model(weights)
    with torch.no_grad():(expected,_,_),_=forward_prefix(model,group)
    root=Session(name+'-root',program=group['program']);root.observe('S0');child=root.fork(name+'-branch');state=None
    for position,observation in enumerate(group['observations']):
        current=child.current if position==0 else child.observe(observation.stage)
        check_observation(current,observation)
        with torch.no_grad():(scores,value,resources),state=model(current,state)
        if position+1<len(group['observations']):
            action=json.loads((CACHE/'runs'/group['prefix']/'trial-0-action.json').read_text())
            index=next(i for i,a in enumerate(current.graph['candidates']) if action_key(a)==action_key(action))
            execute(child,child.manifest['stages'][position],index)
    error=float((scores-expected).abs().max());assert error<1e-6
    index=int(scores.argmax());known=max(group['rows'],key=lambda r:float(scores[r['candidate_index']]))
    stage_index=int(group['stage'][1:]);execute(child,child.manifest['stages'][stage_index],index,scores.tolist(),float(value),resources.tolist())
    report=complete(child,stage_index+1)
    assert report['comparison_group']==json.loads(Path(group['reference_report']).read_text())['comparison_group']
    supported=next((r for r in group['rows'] if r['candidate_index']==index),None)
    if supported:assert (child.output/'program.o').read_bytes()==(Path(supported['report']).parent/'program.o').read_bytes()
    ledger={e['log']:e for s in (root,child) for e in s.events}
    result={'model':key,'program':group['program'],'split':group['split'],'stage':group['stage'],'prefix':group['prefix'],'status':'ok',
        'closed_index':known['candidate_index'],'full_index':index,'in_label_support':supported is not None,
        'predicted_q':-float(scores[index]),'scores':scores.tolist(),'prediction_max_error':error,
        'report':str(child.output/'report.json'),'correctness':report['measurement']['correctness'],
        'wall_s':time.perf_counter()-started,'command_s':sum(e['seconds'] for e in ledger.values()),
        'events':list(ledger.values()),'model_costs':model.costs,
        'scope':'full legal argmax actually executed, followed only by all-advance-v1; diagnostic never fed into training'}
    save(path,result);print(json.dumps({'full_proposal_replay':path.stem,'index':index,'known':supported is not None}),flush=True)
    return result


def matched(output,key,weights,group):
    path=output/(key+f'-{group["program"]}-{group["stage"]}-matched.json')
    if path.exists():return json.loads(path.read_text())
    start=time.perf_counter();affinity=os.sched_getaffinity(0)
    # A training-only taskset must not change the native matched-label CPU or
    # hardware group. Restore the original training affinity after replay.
    os.sched_setaffinity(0,set(group['observations'][0].hardware['affinity']))
    try:return _matched(output,key,weights,group)
    except (CommandError,RuntimeError) as error:
        row={'model':key,'program':group['program'],'stage':group['stage'],'prefix':group['prefix'],'status':'failed',
            'error':str(error),'wall_s':time.perf_counter()-start,'cost_note':'failed full-proposal suffix charged; not labeled negative or replaced by closed argmax'}
        save(path,row);return row
    finally:os.sched_setaffinity(0,affinity)


def final_measure(output,systems,matched_rows,catalog,plan):
    path=output/'measurement.json'
    if path.exists():return json.loads(path.read_text())
    start=time.perf_counter();records={};events=[];rng=random.Random(plan['seed'])
    matched_rows=[r for r in matched_rows if r['status']=='ok']
    all_programs=list(dict.fromkeys([r['program'] for r in systems]+[r['program'] for r in matched_rows]))
    for program in all_programs:
        probe=next(r for r in catalog['pool'] if r['program']==program)
        base=Path(probe['report']);items=[{'method':'default','report':str(base),'role':'default'}]
        items += [{'method':r['model']+'-'+r['mode'],'report':r['result']['selected']['report'],'role':'deployment'}
            for r in systems if r['program']==program and r['status']=='ok']
        items += [{'method':r['model']+'-'+r['stage']+'-matched','report':r['report'],'role':'matched','predicted_q':r['predicted_q']}
            for r in matched_rows if r['program']==program]
        distinct=[]
        for item in items:
            binary=Path(item['report']).parent/'program';data=binary.read_bytes()
            representative=next((j for j,x in enumerate(distinct) if (Path(x['report']).parent/'program').read_bytes()==data),None)
            if representative is None:item['artifact']=len(distinct);distinct.append(item)
            else:item['artifact']=representative
        logs=ROOT/'logs/architecture'/output.parent.name/'final-measure'/program;logs.mkdir(parents=True,exist_ok=True)
        # All current candidate programs use argument-only or no-input workloads;
        # preserve source_dir expansion for existing whole-program recipes.
        from .campaign_corpus import source_path
        args=[a.replace('{source_dir}',str(source_path(program).parent)) for a in RECIPES['programs'][program]['measure_args']]
        expected=command([base.parent/'reference',*args],logs/'oracle.log',events,'evaluation-oracle',cpu=0)
        samples={i:[] for i in range(len(distinct))};orders=[]
        for turn in range(plan['rounds']):
            if turn%2==0:order=list(range(len(distinct)));rng.shuffle(order)
            else:order=list(reversed(orders[-1]))
            orders.append(order)
            for i in order:
                metric=logs/f'{turn}-{i}.json';binary=Path(distinct[i]['report']).parent/'program'
                actual=command([CACHE/'build/irloom-time',metric,binary,*args],logs/f'{turn}-{i}.log',events,'evaluation-measure',cpu=0)
                if actual!=expected:raise RuntimeError('final interleaving oracle mismatch: '+program)
                samples[i].append(json.loads(metric.read_text()))
        base_ms=statistics.median(s['latency_ms'] for s in samples[0])
        for item in items:
            measured=statistics.median(s['latency_ms'] for s in samples[item['artifact']]);item['median_ms']=measured
            item['speedup']=base_ms/measured;item['same_binary_as_default']=item['artifact']==0
            item['paired_speedups']=[a['latency_ms']/b['latency_ms'] for a,b in zip(samples[0],samples[item['artifact']])]
        records[program]={'items':items,'distinct_executables':len(distinct),'orders':orders,'samples':samples,'baseline_ms':base_ms}
    result={'programs':records,'events':events,'wall_s':time.perf_counter()-start,'command_s':sum(e['seconds'] for e in events),
        'measurement_invocations':sum(e['stage']=='evaluation-measure' for e in events),'plan':plan}
    save(path,result);return result


def evaluate(output,config,dataset,training,catalog):
    target=output/'evaluation';target.mkdir(exist_ok=True);final=target/'evaluation.json'
    plan=json.loads((ROOT/'experiments/architecture/campaign-evaluation-v1.json').read_text());frozen=target/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=plan:raise ValueError('evaluation protocol mismatch')
    if not frozen.exists():save(frozen,plan)
    if final.exists():return json.loads(final.read_text())
    # All checkpoints already exist before entering any test inference.
    schedules=[];checks={}
    for key,record in training['models'].items():
        main=torch.load(output/'models'/(key+'.pt'),weights_only=True)
        last=torch.load(output/'models'/f'{key}-step{record["optimizer_steps"]}.pt',weights_only=True)
        assert main['optimizer_steps']==config['training']['updates']
        assert all(torch.equal(v,last['state_dict'][k]) for k,v in main['state_dict'].items())
        assert all(torch.isfinite(v).all() for v in main['state_dict'].values())
        assert [r['optimizer_steps'] for r in record['curves']]==list(range(1,main['optimizer_steps']+1))
        schedules.append([[(g['program'],g['prefix']) for g in row['groups']] for row in record['curves']])
        for step,snapshot in record['snapshots'].items():
            for split in ('train','validation'):
                assert all(math.isfinite(q) for g in snapshot[split]['groups'] for q in g['predicted_q'])
        checks[key]={'optimizer_steps':main['optimizer_steps'],'main_equals_last':True,'all_snapshots_finite':True,
            'group_exposures':sum(len(row['groups']) for row in record['curves'])}
    assert all(s==schedules[0] for s in schedules)
    for seed in config['training']['seeds']:
        for a,b in (('A','current_only'),('B','B_current_only')):
            left=torch.load(output/'models'/f'{a}-s{seed}-step0.pt',weights_only=True)['state_dict']
            right=torch.load(output/'models'/f'{b}-s{seed}-step0.pt',weights_only=True)['state_dict']
            assert all(torch.equal(v,right[k]) for k,v in left.items())
    acceptance=target/'training-acceptance.json'
    if not acceptance.exists():save(acceptance,{'models':checks,'matched_group_schedules':True,'matched_memory_pair_initializations':True,'test_inference_not_started':True})
    groups=prepare_groups(dataset,'train',training['objective'])+prepare_groups(dataset,'validation',training['objective'])
    matched_rows=[]
    for key in training['models']:
        for case in plan['matched_cases']:
            program_groups=[g for g in groups if g['program']==case['program']]
            for ordinal in case['group_ordinals']:
                if ordinal>=len(program_groups):continue
                matched_rows.append(matched(target,key,output/'models'/(key+'.pt'),program_groups[ordinal]))
    systems=[]
    for key,record in training['models'].items():
        programs=(plan['deployment_diagnostics'] if record['seed'] in plan['diagnostic_seeds'] else [])+(catalog['selected']['test'] if record['seed'] in plan['test_seeds'] else [])
        for program in programs:
            for mode in plan['modes']:
                path=target/f'{key}-{program}-{mode}.json'
                if path.exists():systems.append(json.loads(path.read_text()));continue
                started=time.perf_counter();name=output.name+'-'+path.stem
                try:
                    model=load_model(output/'models'/(key+'.pt'))
                    result=guided(name,program,model,config['budget'],mode)
                    row={'model':key,'seed':record['seed'],'program':program,'split':'test' if program in catalog['selected']['test'] else 'diagnostic',
                        'mode':mode,'status':'ok','result':result,'wall_s':time.perf_counter()-started}
                except (CommandError,RuntimeError) as error:
                    row={'model':key,'seed':record['seed'],'program':program,'mode':mode,'status':'failed','error':str(error),
                        'wall_s':time.perf_counter()-started,'cost_note':'failed whole call wall charged, not replaced by another method or candidate'}
                save(path,row);systems.append(row)
                print(json.dumps({'campaign_system':path.stem,'status':row['status']}),flush=True)
    measurement=final_measure(target,systems,matched_rows,catalog,plan['final_measurement'])
    result={'plan':plan,'matched':matched_rows,'systems':systems,'measurement':measurement,
        'system_wall_s':sum(r['wall_s'] for r in systems),'matched_wall_s':sum(r['wall_s'] for r in matched_rows),
        'note':'closed metrics remain in training snapshots; full proposals actually compiled; deployment suffixes not reused as early all-advance labels; no test feedback'}
    save(final,result);return result
