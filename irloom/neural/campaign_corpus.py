"""Ordered compatibility, matched-prefix labels, and one shared proposal round."""
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import torch

from irloom.compiler import CommandError,command
from .driver import ROOT,CACHE,Session,save
from .guided import complete,execute
from .programs import RECIPES,BASE
from .sampling import select
from .revision import load_model
from .schema import Observation
from .stage_learning import forward_prefix
from .label_timing import retime


def unique_name(base):
    name=base;index=0
    while (CACHE/'runs'/name).exists():index+=1;name=base+f'-attempt{index}'
    return name


def source_path(program):
    recipe=RECIPES['programs'][program]
    suite=ROOT/'benchmark-data/sources/llvm-test-suite-18.1.3'
    return (suite if recipe.get('source_root')=='suite' else suite/'MultiSource/Benchmarks')/recipe['path']


def summarize(graph):
    kinds=Counter(n['kind'] for n in graph['nodes'])
    return {'nodes':len(graph['nodes']),'edges':len(graph['edges']),'functions':kinds['function'],
        'regions':kinds['region'],'actions':dict(Counter(c['kind'] for c in graph['candidates']))}


def events(sessions):
    return list({e['log']:e for s in sessions for e in getattr(s,'events',[])}.values())


def probe(output,config):
    target=output/'compatibility';target.mkdir(exist_ok=True);path=target/'catalog.json'
    if path.exists():return json.loads(path.read_text())
    records=[];selected={k:[] for k in config['targets']}
    families={}
    for row in config['pool']:
        for identity in (row['family'],'source:'+row['source_group']):
            if identity in families and families[identity]!=row['split']:raise ValueError('family crosses splits: '+identity)
            families[identity]=row['split']
        program=row['program'];record_path=target/(program+'.json')
        if len(selected[row['split']])>=config['targets'][row['split']]:continue
        if record_path.exists():record=json.loads(record_path.read_text())
        else:
            start=time.perf_counter();sessions=[];source=source_path(program)
            record={**row,'recipe':RECIPES['programs'][program],'source':str(source),'source_lines':len(source.read_text(errors='replace').splitlines())}
            try:
                root=object.__new__(Session);sessions.append(root)
                root.__init__(unique_name(output.name+'-probe-'+program),program=program)
                root.observe('S0');default=root.fork(unique_name(output.name+'-probe-'+program+'-default'));sessions.append(default)
                report=complete(default,0)
                record['report']=str(default.output/'report.json')
                record['graphs']={o['stage']:summarize(o['graph']) for o in report['observations']}
                reachable=all(sum(record['graphs'][s]['actions'].values())>1 for s in ('S0','S1'))
                record['status']='supported' if reachable else 'single-stage-only'
                record['reason']='complete source and independent outputs pass; two action stages' if reachable else 'correct default, but S0/S1 does not both offer a non-advance action'
            except (CommandError,RuntimeError,ValueError) as error:
                record.update(status='failed',reason=str(error))
            record.update(events=events(sessions),wall_s=time.perf_counter()-start)
            record['command_s']=sum(e['seconds'] for e in record['events'])
            record['terminal_attempts']=int(any(e['stage']=='apply' for e in record['events']))
            save(record_path,record)
            print(json.dumps({'campaign_probe':program,'status':record['status'],'split':row['split']}),flush=True)
        records.append(record)
        if record['status']=='supported':selected[row['split']].append(program)
    app_path=target/'large-application.json'
    if app_path.exists():application=json.loads(app_path.read_text())
    else:
        app=config['large_application'];source=ROOT/'benchmark-data/sources/llvm-test-suite-18.1.3'/app['source']
        folder=target/'sqlite';folder.mkdir(exist_ok=True);logs=ROOT/'logs/architecture'/output.name/'sqlite-import';logs.mkdir(parents=True,exist_ok=True)
        ledger=[];start=time.perf_counter();ir=folder/'sqlite.mlir'
        application={**app,'source_lines':len(source.read_text(errors='replace').splitlines()),'source_bytes':source.stat().st_size}
        try:
            command([BASE/'build/bin/cgeist',source,'-S','-O0','-raise-scf-to-affine','-std=gnu89',
                '-resource-dir=/usr/lib/llvm-18/lib/clang/18',*['-D'+v for v in app['defines']],'-o',ir],logs/'import.log',ledger,'large-import',timeout=app['timeout_s'],cpu=0)
            application['ir_bytes']=ir.stat().st_size
            if ir.stat().st_size>app['max_ir_bytes']:raise RuntimeError('import exceeds frozen IR byte budget; no graph expansion')
            graph=folder/'graph.json'
            command([BASE/'build/bin/irloom-polygeist-bridge',ir,graph,'sqlite-0','sqlite','S0'],logs/'graph.log',ledger,'large-observe',timeout=app['timeout_s'],cpu=0)
            application.update(status='imported-library-only',graph=summarize(json.loads(graph.read_text())))
        except (CommandError,RuntimeError) as error:application.update(status='failed',reason=str(error))
        application.update(events=ledger,wall_s=time.perf_counter()-start,command_s=sum(e['seconds'] for e in ledger))
        save(app_path,application)
    result={'pool':records,'selected':selected,'targets':config['targets'],'large_application':application,
        'wall_s':sum(r['wall_s'] for r in records)+application['wall_s'],
        'command_s':sum(r['command_s'] for r in records)+application['command_s'],
        'selection':'prelisted priority, correctness, graph budget and two-stage reachability only; no speed filter'}
    save(path,result);return result


def collect(output,config,catalog):
    target=output/'corpus';target.mkdir(exist_ok=True);path=target/'dataset.json'
    if path.exists():return json.loads(path.read_text())
    c=config['collection'];ledger_path=target/'budget.json'
    if ledger_path.exists():ledger=json.loads(ledger_path.read_text())
    else:
        probe_attempts={r['program']:r['terminal_attempts'] for r in catalog['pool'] if r['split'] in ('train','validation')}
        ledger={'static':sum(probe_attempts.values()),'proposal':0,'by_program':probe_attempts}
        save(ledger_path,ledger)
    def charge(program,role):
        if ledger[role]>=c['static_terminal_limit' if role=='static' else 'proposal_terminal_limit']:return False
        if ledger['static']+ledger['proposal']>=c['terminal_limit'] or ledger['by_program'].get(program,0)>=c['max_per_program']:return False
        ledger[role]+=1;ledger['by_program'][program]=ledger['by_program'].get(program,0)+1;save(ledger_path,ledger);return True
    models={key:load_model(CACHE/path) for key,path in config['proposal']['weights'].items()}
    program_records=[]
    for program in catalog['selected']['train']+catalog['selected']['validation']:
        record_path=target/(program+'.json')
        if record_path.exists():program_records.append(json.loads(record_path.read_text()));continue
        start=time.perf_counter();sessions=[];failures=[];groups=[];proposals=[]
        entry=next(r for r in catalog['pool'] if r['program']==program);split=entry['split']
        root=Session(unique_name(output.name+'-labels-'+program+'-root'),program=program);sessions.append(root)
        root.observe('S0')
        old=json.loads(Path(entry['report']).read_text())
        assert root.current.graph['nodes']==old['observations'][0]['graph']['nodes']
        assert root.current.graph['edges']==old['observations'][0]['graph']['edges']
        baseline=root.fork(unique_name(output.name+'-labels-'+program+'-default'));sessions.append(baseline)
        for filename in ('program','program.o','reference'):shutil.copy2(Path(entry['report']).parent/filename,baseline.output/filename)
        save(baseline.output/'report.json',{**old,'default_origin_report':entry['report']})
        measured=[baseline];prefixes=[root]
        choices,prefix_sampling=select(root.current.graph,c['prefixes'],c['seed']+config['pool'].index(next(r for r in config['pool'] if r['program']==program))*100)
        for index in choices:
            child=root.fork(unique_name(output.name+f'-labels-{program}-prefix{index}'));sessions.append(child)
            try:
                execute(child,child.manifest['stages'][0],index);child.observe('S1');prefixes.append(child)
            except (CommandError,RuntimeError) as error:failures.append({'prefix_action':index,'error':str(error)})
        static_start=ledger['static'];proposal_start=ledger['proposal']
        # Default was already charged during compatibility; distribute the remaining
        # fixed per-program allowance across S0 and the actual S1 prefixes.
        remaining=max(0,c['static_per_program']-1)
        for ordinal,parent in enumerate(prefixes):
            available=len(parent.current.graph['candidates']);left=len(prefixes)-ordinal
            limit=min(c['small_pool'],available,max(1,(remaining+int(ordinal==0))//left))
            if available<=c['small_pool'] and available-int(ordinal==0)<=remaining-(left-1):limit=available
            indices,sampling=select(parent.current.graph,limit,c['seed']+ordinal)
            group={'program':program,'split':split,'family':entry['family'],'stage':parent.current.stage,'prefix':parent.branch,
                'observations':parent.observations,'reference_report':str(baseline.output/'report.json'),'continuation':'all-advance-v1','rows':[],'sampling':sampling}
            if ordinal==0:group['rows'].append({'candidate_index':0,'report':str(baseline.output/'report.json'),'alias':'root default'})
            def terminal(index,role):
                nonlocal remaining
                if not charge(program,role):return False
                if role=='static':remaining-=1
                child=parent.fork(unique_name(output.name+f'-labels-{program}-g{ordinal}-a{index}-{role}'));sessions.append(child)
                try:
                    stage_index=int(parent.current.stage[1:]);execute(child,child.manifest['stages'][stage_index],index)
                    report=complete(child,stage_index+1)
                    assert report['comparison_group']==old['comparison_group']
                    group['rows'].append({'candidate_index':index,'report':str(child.output/'report.json'),'sampling_role':role})
                    measured.append(child)
                except (CommandError,RuntimeError) as error:failures.append({'prefix':parent.branch,'candidate_index':index,'role':role,'error':str(error)})
                return True
            for index in indices:
                if ordinal==0 and index==0:continue
                if remaining<=0:break
                terminal(index,'static')
            if split=='train':
                predictions=[]
                for key,model in models.items():
                    observations={'observations':[Observation(**o) for o in group['observations']]}
                    with torch.no_grad():(scores,_,_),_=forward_prefix(model,observations)
                    index=int(scores.argmax());predictions.append(index)
                    proposals.append({'model':key,'stage':group['stage'],'prefix':parent.branch,'index':index,
                        'in_static_labels':any(r['candidate_index']==index for r in group['rows']),'scores':scores.tolist(),'costs':list(model.costs)})
                    model.costs.clear()
                for index in sorted(set(predictions)):
                    if not any(r['candidate_index']==index for r in group['rows']):terminal(index,'proposal')
            if not any(r['candidate_index']==0 for r in group['rows']):
                failures.append({'prefix':parent.branch,'error':'no valid advance suffix; whole group excluded'})
            else:groups.append(group)
        # Time each emitted object once, retaining every context/action alias.
        distinct=[];aliases=[]
        for session in measured:
            content=(session.output/'program.o').read_bytes()
            representative=next((s for s in distinct if (s.output/'program.o').read_bytes()==content),None)
            if representative:aliases.append((session,representative))
            else:distinct.append(session)
        timing=retime(distinct,baseline,c['seed'],c['timing_rounds'])
        for alias,representative in aliases:
            report=json.loads((alias.output/'report.json').read_text());reference=json.loads((representative.output/'report.json').read_text())
            report['initial_measurement']=report['measurement'];report['measurement']={**report['measurement'],'samples':reference['measurement']['samples']}
            report['median_latency_ms']=reference['median_latency_ms'];report['measurement_alias_report']=str(representative.output/'report.json')
            save(alias.output/'report.json',report)
        record={'program':program,'split':split,'groups':groups,'defaults':str(baseline.output/'report.json'),'failures':failures,
            'proposals':proposals,'prefix_sampling':prefix_sampling,'distinct_objects':len(distinct),'compiled_terminals':len(measured),
            'new_static_attempts':ledger['static']-static_start,'new_proposal_attempts':ledger['proposal']-proposal_start,
            'timing':timing,'events':events(sessions),'wall_s':time.perf_counter()-start}
        record['command_s']=sum(e['seconds'] for e in record['events']);save(record_path,record);program_records.append(record)
        print(json.dumps({'campaign_labels':program,'groups':len(groups),'compiled':len(measured),'distinct':len(distinct),'failures':len(failures)}),flush=True)
    groups=[g for r in program_records for g in r['groups']]
    result={'config':{'training':config['objective'],'splits':{r['program']:r['split'] for r in catalog['pool']}},'groups':groups,
        'defaults':{r['program']:r['defaults'] for r in program_records},'programs':program_records,'budget':ledger,
        'wall_s':sum(r['wall_s'] for r in program_records),'command_s':sum(r['command_s'] for r in program_records),
        'probe_costs':{'wall_s':catalog['wall_s'],'command_s':catalog['command_s']},
        'note':'No test labels. Same union of training-domain frozen proposals shared by every main model. Alias rows are contexts, not independent artifacts.'}
    save(path,result);return result


def audit(output,dataset,config,catalog):
    """Data/semantic contracts and coverage, without new executions or hashes."""
    from .stage_learning import prepare_groups
    from .cost_objective import cost_loss
    path=output/'corpus/audit.json'
    if path.exists():return json.loads(path.read_text())
    budget=dataset['budget'];limits=config['collection']
    assert budget['static']<=limits['static_terminal_limit'] and budget['proposal']<=limits['proposal_terminal_limit']
    assert budget['static']+budget['proposal']<=limits['terminal_limit']
    assert max(budget['by_program'].values())<=limits['max_per_program']
    rows=[]
    for split in ('train','validation'):
        for group in prepare_groups(dataset,split,config['objective']):
            assert group['program'] in catalog['selected'][split]
            if group['stage']=='S1':
                assert [o.stage for o in group['observations']]==['S0','S1']
                action=json.loads((CACHE/'runs'/group['prefix']/'trial-0-action.json').read_text())
                assert group['observations'][-1].history[-1]['kind']==action['kind']
            y=torch.tensor([r['target'] for r in group['rows']],requires_grad=True)
            margins=torch.tensor([r['margin'] for r in group['rows']])
            loss=cost_loss(y,y.detach(),margins,group['pairs'],'shared-root-margin-v1')
            assert loss==0 and torch.autograd.grad(loss,y)[0].abs().sum()==0
            unique=[]
            for row in group['rows']:
                if not any(row['object']==b for b in unique):unique.append(row['object'])
                report=json.loads(Path(row['report']).read_text())
                if report.get('measurement_alias_report'):
                    alias=Path(report['measurement_alias_report'])
                    assert row['object']==(alias.parent/'program.o').read_bytes()
                assert report['measurement']['correctness'] and all(c['status']=='pass' for c in report['measurement']['correctness'])
            rows.append({'program':group['program'],'split':split,'stage':group['stage'],'prefix':group['prefix'],
                'candidate_pool':len(group['observations'][-1].graph['candidates']),'labeled':len(group['rows']),
                'distinct_objects':len(unique),'strong_pairs':len(group['pairs']),'effects':group['counts'],
                'upstream_action':group['observations'][-1].history[-1]['kind'] if group['stage']=='S1' else None})
    result={'groups':rows,'budget':budget,'selected':catalog['selected'],
        'source_families':{split:sorted({r['family'] for r in catalog['pool'] if r['program'] in names}) for split,names in catalog['selected'].items()},
        'all_true_q_zero_loss_and_gradient':True,'matching_prefixes_and_correctness':True,
        'new_test_labels':0,'note':'programs, context rows, distinct objects and strong pairs are separate counts; all alias byte comparisons direct'}
    save(path,result);return result
