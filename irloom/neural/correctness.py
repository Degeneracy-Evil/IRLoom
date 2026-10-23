"""Fixed-trace compiler correctness audit; never asks a model to select a new fix."""
import json
from pathlib import Path
import time

from .driver import ROOT,CACHE,Session,save
from .guided import execute


def target_key(graph,target):
    parents={b:(a,port[0] if port else 0) for a,b,kind,*port in graph['edges'] if kind=='contains'}
    path=[];node=target
    while graph['nodes'][node]['kind']!='function' and node in parents:
        node,position=parents[node];path.append(position)
    owner=graph['nodes'][node].get('anchor','module')
    return {'owner':owner,'path':list(reversed(path)),'kind':graph['nodes'][target]['name']}


def locate(graph,descriptor,kind,parameters):
    candidates=[i for i,c in enumerate(graph['candidates']) if c['kind']==kind and c['parameters']==parameters
        and target_key(graph,c['target'])==descriptor]
    if len(candidates)!=1:raise RuntimeError('semantic action target is missing or ambiguous: '+str(descriptor))
    return candidates[0]


def inventory(output,config):
    path=output/'inventory.json'
    if path.exists():return json.loads(path.read_text())
    evaluation=json.loads((CACHE/config['campaign']/'evaluation/evaluation.json').read_text())
    records=[];classes=[];objects=[];llvm=[]
    for row in evaluation['systems']:
        leaves=([f'campaign-v1-{row["model"]}-{row["program"]}-{row["mode"]}-branch-2'] if row['status']=='failed'
            else [f['branch'] for f in row['result']['failures']])
        for leaf in leaves:
            here=CACHE/'runs'/leaf;chain=[]
            for stage in (2,1,0):
                request_path=here/f'trial-{stage}-action.json';action=json.loads(request_path.read_text());snapshot=action['snapshot']
                owner=leaf.split('-branch-')[0]+'-root' if snapshot=='root-0' else snapshot.rsplit('-',1)[0]
                source=CACHE/'runs'/owner;graph=json.loads((source/(snapshot+'.json')).read_text())
                assert graph['snapshot']==snapshot
                assert any(c['kind']==action['kind'] and c['target']==action['target'] and c['parameters']==action['parameters'] for c in graph['candidates'])
                chain.append({'stage':f'S{stage}','kind':action['kind'],'parameters':action['parameters'],
                    'descriptor':target_key(graph,action['target']),'request':str(request_path),
                    'snapshot_graph':str(source/(snapshot+'.json')),'input_ir':str(source/('imported.mlir' if stage==0 else f'after-S{stage-1}.mlir'))})
                here=source
            chain.reverse();semantic=[{k:a[k] for k in ('stage','kind','parameters','descriptor')} for a in chain]
            if semantic not in classes:classes.append(semantic)
            obj=(CACHE/'runs'/leaf/'program.o').read_bytes();ll=(CACHE/'runs'/leaf/'program.ll').read_bytes()
            if obj not in objects:objects.append(obj)
            if ll not in llvm:llvm.append(ll)
            records.append({'leaf':leaf,'program':row['program'],'model':row['model'],'mode':row['mode'],
                'class':classes.index(semantic),'actions':chain,'object_class':objects.index(obj),'llvm_class':llvm.index(ll)})
    result={'failures':records,'semantic_classes':classes,'distinct_objects':len(objects),'distinct_llvm_files':len(llvm),
        'note':'Exact bytes without hashes; two inline sites can trigger one reduction-cleanup mechanism. Descriptor paths are snapshot-relative structural ports, never reused numeric node IDs.'}
    save(path,result);return result


def session(name,program,revision='legacy'):
    return Session(name,program=program,compiler_revision=revision)


def finish(current,start):
    for i in range(start,len(current.manifest['stages'])):
        stage=current.manifest['stages'][i]
        if current.current is None or current.current.stage!=stage['id'] or current.ir!=current.observed_ir:current.observe(stage['id'])
        execute(current,stage,0)
    ll=current.output/'program.ll';obj=current.output/'program.o';exe=current.output/'program'
    current.run([current.translate,'--mlir-to-llvmir',current.ir,'-o',ll],'translate')
    current.run(['clang-18',*current.manifest['backend'],'-c',ll,'-o',obj],'codegen')
    current.run(['clang-18',obj,'-lm','-o',exe],'link')
    return exe


def check(current,exe,oracle,inputs):
    checks=[]
    for value in inputs:
        expected=current.run([oracle,str(value)],'correctness-oracle')
        actual=current.run([exe,str(value)],'correctness-candidate')
        checks.append({'args':[str(value)],'expected':expected,'actual':actual,'status':'pass' if expected==actual else 'wrong_output'})
    return checks


def isolate(output,config,listing):
    path=output/'isolation.json'
    if path.exists():return json.loads(path.read_text())
    records=[];start=time.perf_counter()
    for class_index in range(len(listing['semantic_classes'])):
        representative=next(r for r in listing['failures'] if r['class']==class_index)
        actions=representative['actions']
        for mode in ('default','inline_only','unroll_only','combined'):
            record_path=output/f'legacy-class{class_index}-{mode}.json'
            if record_path.exists():records.append(json.loads(record_path.read_text()));continue
            current=session(output.name+f'-legacy-c{class_index}-{mode}',representative['program']);current.observe('S0')
            assert current.current.graph['nodes']==json.loads(Path(actions[0]['snapshot_graph']).read_text())['nodes']
            index=locate(current.current.graph,actions[0]['descriptor'],actions[0]['kind'],actions[0]['parameters']) if mode in ('inline_only','combined') else 0
            execute(current,current.manifest['stages'][0],index);current.observe('S1')
            try:index=locate(current.current.graph,actions[1]['descriptor'],actions[1]['kind'],actions[1]['parameters']) if mode in ('unroll_only','combined') else 0
            except RuntimeError as error:
                row={'class':class_index,'mode':mode,'status':'not_applicable','reason':str(error)};save(record_path,row);records.append(row);continue
            execute(current,current.manifest['stages'][1],index);exe=finish(current,2)
            oracle=current.output/'reference'
            current.run(['clang-18',*current.manifest['backend'],'-std=gnu89',current.original_source,'-lm','-o',oracle],'reference-compile')
            checks=check(current,exe,oracle,config['inputs'])
            row={'class':class_index,'mode':mode,'status':'wrong_output' if any(c['status']=='wrong_output' for c in checks) else 'pass',
                'checks':checks,'run':str(current.output),'events':current.events,'command_s':sum(e['seconds'] for e in current.events)}
            save(record_path,row);records.append(row)
            print(json.dumps({'isolation_class':class_index,'mode':mode,'status':row['status']}),flush=True)
    result={'cases':records,'wall_s':time.perf_counter()-start,'scope':'Historical compiler and fixed semantic actions; correctness only, no model inference/performance timing.'}
    assert all(r['status']=='pass' for r in records if r['mode'] in ('default','inline_only'))
    assert all(r['status']=='wrong_output' for r in records if r['mode'] in ('unroll_only','combined'))
    result['expected_wrong_output_reproduced']=True
    save(path,result);return result


def minimal(output,config):
    path=output/'minimal.json'
    if path.exists():return json.loads(path.read_text())
    from .programs import BASE
    records=[]
    for revision in ('legacy',config['revision']):
        for mode in ('default','unroll'):
            name=output.name+f'-minimal-{revision}-{mode}';current=Session(name)
            current.ir=ROOT/'compiler/diagnostics/affine-cleanup.mlir'
            current.bridge=(BASE/'build/bin/irloom-polygeist-bridge' if revision=='legacy' else CACHE/'compilers'/revision/'bin/irloom-polygeist-bridge')
            current.opt=BASE/'build/bin/polygeist-opt';current.translate=BASE/'build/bin/mlir-translate'
            current.manifest=json.loads((ROOT/'compiler/pipelines/polygeist-v1.json').read_text())
            current.manifest['compiler_revision']=revision;save(current.output/'manifest.json',current.manifest)
            current.observe('S1')
            if mode=='unroll':
                for owner in ('same_init','distinct_init','bound_alias','constant_odd','constant_even'):
                    action=next(c for c in current.current.graph['candidates'] if c['kind']=='unroll2' and target_key(current.current.graph,c['target'])['owner']=='function:'+owner)
                    assert current.apply(action).success
                    current.observe('S1')
            current.advance(current.manifest['stages'][1])
            current.observe('S2');execute(current,current.manifest['stages'][2],0)
            ll=current.output/'program.ll';obj=current.output/'program.o';exe=current.output/'minimal'
            current.run([current.translate,'--mlir-to-llvmir',current.ir,'-o',ll],'translate')
            current.run(['clang-18',*current.manifest['backend'],'-c',ll,'-o',obj],'codegen')
            current.run(['clang++-18','-O3',ROOT/'compiler/diagnostics/affine-cleanup-runner.cpp',obj,'-o',exe],'link')
            checks=[]
            for n in config['inputs']:
                expected=f'{n*1002} {(n+1)*1002} {2*n} 3006 4008\n'
                actual=current.run([exe,str(n)],'minimal-check')
                checks.append({'n':n,'expected':expected,'actual':actual,'status':'pass' if actual==expected else 'wrong_output'})
            if revision!='legacy' or mode=='default':assert all(r['status']=='pass' for r in checks)
            else:assert any(r['status']=='wrong_output' for r in checks)
            records.append({'revision':revision,'mode':mode,'run':str(current.output),'checks':checks,'events':current.events})
    result={'cases':records,'scope':'Exact integer recurrence oracle, independent MLIR input, no frontend or model. Expected old failure is evidence, not a correctness pass.'}
    save(path,result);return result


def upstream_reproducer(output,config):
    path=output/'upstream-reproducer.json'
    if path.exists():return json.loads(path.read_text())
    from .programs import BASE
    current=Session(output.name+'-upstream-reproducer');source=ROOT/'compiler/diagnostics/affine-cleanup.mlir'
    after=current.output/'unrolled.mlir';ll=current.output/'program.ll';obj=current.output/'program.o';exe=current.output/'program'
    current.run([BASE/'build/bin/polygeist-opt',source,'--affine-loop-unroll=unroll-factor=2','--lower-affine',
        '--convert-polygeist-to-llvm','-o',after],'official-affine-unroll-pass')
    current.run([BASE/'build/bin/mlir-translate','--mlir-to-llvmir',after,'-o',ll],'translate')
    current.run(['clang-18','-O3','-march=native','-ffp-contract=off','-c',ll,'-o',obj],'codegen')
    current.run(['clang++-18','-O3',ROOT/'compiler/diagnostics/affine-cleanup-runner.cpp',obj,'-o',exe],'link')
    checks=[]
    for n in config['inputs']:
        actual=current.run([exe,str(n)],'upstream-check');expected=f'{n*1002} {(n+1)*1002} {2*n} 3006 4008\n'
        checks.append({'n':n,'actual':actual,'expected':expected,'status':'pass' if actual==expected else 'wrong_output'})
    assert any(c['status']=='wrong_output' for c in checks)
    result={'checks':checks,'events':current.events,'run':str(current.output),
        'scope':'Pinned official affine-loop-unroll pass, no IRLoom bridge/provenance attributes, no frontend/model; expected defect reproduced.'}
    save(path,result);return result


def program_checks(current,exe,oracle):
    checks=[];args_list=current.recipe['check_args']+[current.recipe['measure_args']]
    for args in args_list:
        args=[v.replace('{source_dir}',str(current.original_source.parent)) for v in args]
        if any(c['args']==args for c in checks):continue
        expected=current.run([oracle,*args],'oracle-check');actual=current.run([exe,*args],'candidate-check')
        checks.append({'args':args,'expected':expected,'actual':actual,'status':'pass' if actual==expected else 'wrong_output'})
    return checks


def fixed_replay(output,config,listing):
    path=output/'fixed-replay.json'
    if path.exists():return json.loads(path.read_text())
    records=[]
    for class_index in range(len(listing['semantic_classes'])):
        source=next(r for r in listing['failures'] if r['class']==class_index)
        for mode in ('default','inline_only','unroll_only','combined'):
            current=session(output.name+f'-fixed-c{class_index}-{mode}',source['program'],config['revision']);current.observe('S0')
            a,b=source['actions'][:2]
            index=locate(current.current.graph,a['descriptor'],a['kind'],a['parameters']) if mode in ('inline_only','combined') else 0
            execute(current,current.manifest['stages'][0],index);current.observe('S1')
            index=locate(current.current.graph,b['descriptor'],b['kind'],b['parameters']) if mode in ('unroll_only','combined') else 0
            execute(current,current.manifest['stages'][1],index);exe=finish(current,2)
            oracle=current.output/'reference'
            current.run(['clang-18',*current.manifest['backend'],'-std=gnu89',current.original_source,'-lm','-o',oracle],'reference-compile')
            checks=check(current,exe,oracle,config['inputs']);assert all(c['status']=='pass' for c in checks)
            records.append({'class':class_index,'mode':mode,'checks':checks,'run':str(current.output),'events':current.events})
    # Replay the original 32 concrete actions without rerunning model inference.
    evaluation=json.loads((CACHE/config['campaign']/'evaluation/evaluation.json').read_text())
    matched=[]
    for ordinal,row in enumerate(evaluation['matched']):
        old=json.loads(Path(row['report']).read_text());current=session(output.name+f'-matched-{ordinal}',row['program'],config['revision'])
        for stage_index,transition in enumerate(old['transitions']):
            stage=current.manifest['stages'][stage_index];obs=current.observe(stage['id']);a=transition['action']
            before=next(o for o in old['observations'] if o['snapshot']==a['snapshot'])
            index=0 if a['kind']=='advance' else locate(obs.graph,target_key(before['graph'],a['target']),a['kind'],a['parameters'])
            execute(current,stage,index)
        exe=finish(current,3);oracle=current.output/'reference'
        current.run(['clang-18',*current.manifest['backend'],'-std=gnu89',current.original_source,'-lm','-o',oracle],'reference-compile')
        checks=program_checks(current,exe,oracle);assert all(c['status']=='pass' for c in checks)
        matched.append({'model':row['model'],'program':row['program'],'stage':row['stage'],'checks':checks,'run':str(current.output),
            'object_equal_to_historical':(current.output/'program.o').read_bytes()==(Path(row['report']).parent/'program.o').read_bytes(),'events':current.events})
    result={'fixed_cases':records,'historical_32_matched_actions':matched,'scope':'No new labels or timing; revision-separated exact semantic actions, actual snapshots re-enumerated.'}
    save(path,result);return result


def frozen_models(output,config):
    path=output/'frozen-models.json'
    if path.exists():return json.loads(path.read_text())
    import shutil
    from .guided import guided
    from .revision import load_model
    import torch
    torch.set_num_threads(1)
    budget=json.loads((ROOT/'experiments/architecture/campaign-v1.json').read_text())['budget'];rows=[]
    factory=lambda name,program:session(name,program,config['revision'])
    for key in config['frozen_models']:
        model=load_model(CACHE/config['campaign']/'models'/(key+'.pt'))
        cached=CACHE/'runs'/(output.name+'-frozen-'+key+'-root')/'guided.json'
        result=json.loads(cached.read_text()) if cached.exists() else guided(output.name+'-frozen-'+key,'spectral_norm',model,budget,mode='raw',session_factory=factory)
        selected=Path(result['selected']['report']);report=json.loads(selected.read_text())
        # Extend the normal whole-program check beyond its historical two inputs.
        from irloom.compiler import command
        logs=ROOT/'logs/architecture'/output.name/(key+'-extended-final');logs.mkdir(parents=True,exist_ok=True);events=[];checks=[]
        for n in config['inputs']:
            expected=command([selected.parent/'reference',str(n)],logs/f'{n}-reference.log',events,'extended-oracle',cpu=0)
            actual=command([selected.parent/'program',str(n)],logs/f'{n}-candidate.log',events,'extended-candidate',cpu=0)
            assert actual==expected;checks.append({'n':n,'actual':actual,'expected':expected,'status':'pass'})
        rows.append({'model':key,'guided':result,'checks':checks,'events':events,'actions':[d['selected'] for d in report['decisions']]})
    class FaultSession(Session):
        def run(self,args,label):
            result=super().run(args,label)
            if label=='link' and not self.output.name.endswith('-default'):
                shutil.copy2(self.output/'program',self.output/'uninjected-program')
                super().run(['clang++-18',ROOT/'compiler/diagnostics/wrong-output.cpp','-o',self.output/'program'],'inject-wrong-artifact')
            return result
    def faulty(name,program):return FaultSession(name,program=program,compiler_revision=config['revision'])
    model=load_model(CACHE/config['campaign']/'models/B-s23.pt')
    corrupted=guided(output.name+'-guard-private-wrong-output','spectral_norm',model,budget,session_factory=faulty)
    assert corrupted['fallback_to_default'] and corrupted['confirmation_complete']
    assert any(f.get('category')=='wrong_output' for f in corrupted['failures'])
    failed=guided(output.name+'-guard-injected-failure','spectral_norm',model,budget,inject_failure=True,session_factory=factory)
    assert failed['fallback_to_default'] and failed['confirmation_complete'] and failed['failures']
    result={'original_frozen_raw':rows,'guard_wrong_artifact':corrupted,'guard_injected_failure':failed,
        'scope':'Original three weights unchanged; extended positive-input correctness only. Deliberate artifact corruption confined to proposal copies, never benchmark source or default.'}
    save(path,result);return result


def related_and_backend(output,config):
    path=output/'related-and-backend.json'
    if path.exists():return json.loads(path.read_text())
    dataset=json.loads((CACHE/config['campaign']/'corpus/dataset.json').read_text());records=[]
    for program in config['regression_programs']:
        group=next(g for g in dataset['groups'] if g['program']==program and g['stage']=='S1' and g['observations'][-1]['history'][-1]['kind']=='advance')
        chosen=next(r for r in group['rows'] if r['candidate_index']!=0)
        action=group['observations'][-1]['graph']['candidates'][chosen['candidate_index']]
        for mode in ('default','measured_unroll'):
            current=session(output.name+f'-related-{program}-{mode}',program,config['revision']);current.observe('S0')
            execute(current,current.manifest['stages'][0],0);current.observe('S1')
            index=locate(current.current.graph,target_key(group['observations'][-1]['graph'],action['target']),action['kind'],action['parameters']) if mode=='measured_unroll' else 0
            execute(current,current.manifest['stages'][1],index);exe=finish(current,2);oracle=current.output/'reference'
            current.run(['clang-18',*current.manifest['backend'],'-std=gnu89',*["-D"+x for x in current.recipe.get('defines',[])],current.original_source,'-lm','-o',oracle],'reference-compile')
            checks=program_checks(current,exe,oracle);assert all(c['status']=='pass' for c in checks)
            historical=chosen['report'] if mode=='measured_unroll' else dataset['defaults'][program]
            records.append({'program':program,'mode':mode,'checks':checks,'events':current.events,'run':str(current.output),
                'historical_report':historical,'object_equal':(current.output/'program.o').read_bytes()==(Path(historical).parent/'program.o').read_bytes()})
    # Existing malformed LLVM behaves wrongly even at O0; compare source O0 and
    # source O3 to exclude the oracle/normal backend optimisation as the trigger.
    old=json.loads((output/'legacy-class0-combined.json').read_text());folder=Path(old['run'])
    current=session(output.name+'-backend-diagnostic','spectral_norm');obj=current.output/'bad-O0.o';exe=current.output/'bad-O0';oracle=current.output/'oracle-O0'
    current.run(['clang-18','-O0','-march=native','-ffp-contract=off','-c',folder/'program.ll','-o',obj],'diagnostic-backend-O0')
    current.run(['clang-18',obj,'-lm','-o',exe],'link-O0')
    current.run(['clang-18','-O0','-march=native','-ffp-contract=off','-std=gnu89',current.original_source,'-lm','-o',oracle],'diagnostic-source-O0')
    checks=[]
    for n in config['inputs']:
        expected=current.run([folder/'reference',str(n)],'source-O3');source_O0=current.run([oracle,str(n)],'source-O0')
        actual=current.run([exe,str(n)],'malformed-LLVM-O0');assert expected==source_O0
        checks.append({'n':n,'oracle_O3':expected,'oracle_O0':source_O0,'malformed_LLVM_O0':actual,'status':'pass' if actual==expected else 'wrong_output'})
    assert any(c['status']=='wrong_output' for c in checks)
    result={'related_programs':records,'backend_diagnostic':{'checks':checks,'events':current.events,'run':str(current.output)},
        'scope':'Formal regression stays O3; O0 is evidence-only, not a proposed workaround.'}
    save(path,result);return result


def failure_reporting(output,config,listing):
    path=output/'failure-reporting.json'
    if path.exists():return json.loads(path.read_text())
    import torch
    from types import SimpleNamespace
    from .guided import guided
    actions=listing['failures'][0]['actions']
    class TraceRelay:
        # Diagnostic action relay, not a trained model or optimization method.
        heads=SimpleNamespace(decision_version='stage-q-v1')
        costs=[]
        def eval(self):return self
        def __call__(self,observation,state):
            a=actions[int(observation.stage[1:])]
            index=0 if a['kind']=='advance' else locate(observation.graph,a['descriptor'],a['kind'],a['parameters'])
            scores=torch.full((len(observation.graph['candidates']),),-1.0);scores[index]=1.0
            return (scores,torch.tensor(0.0),torch.zeros(2)),None
    budget=json.loads((ROOT/'experiments/architecture/campaign-v1.json').read_text())['budget']
    name=output.name+'-legacy-fixed-wrong-output'
    try:guided(name,'spectral_norm',TraceRelay(),budget,mode='raw')
    except RuntimeError:
        source=CACHE/'runs'/(name+'-root')/'guided-failure.json'
        result=json.loads(source.read_text())
        assert result['actual_terminals']==0 and result['attempts']==3
        assert result['wall_s']>0 and result['command_s']>0
        assert any(r.get('category')=='wrong_output' and r['actual']=='1.000000000\n' for r in result['failures'])
        save(path,{'diagnostic':'fixed historical actions, no neural selection','failure_report':str(source),'result':result})
        return result
    raise AssertionError('historical wrong-output trace unexpectedly accepted')


def run(args):
    config=json.loads((ROOT/'experiments/architecture/correctness-v1.json').read_text())
    output=CACHE/'correctness-v1'/args.name;output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('correctness protocol mismatch')
    if not path.exists():save(path,config)
    listing=inventory(output,config)
    if args.correctness_audit in ('isolate','all'):isolate(output,config,listing)
    if args.correctness_audit in ('minimal','all'):
        minimal(output,config);upstream_reproducer(output,config)
    if args.correctness_audit in ('regress','all'):
        fixed_replay(output,config,listing);frozen_models(output,config);related_and_backend(output,config);failure_reporting(output,config,listing)
    if args.correctness_audit in ('analyze','all'):
        from .correctness_analysis import analyze,review_exposed,impact_summary
        analysis=analyze(output,config);review=review_exposed(output,config,analysis)
        impact_summary(output,config,analysis,review)
    print(json.dumps({'correctness_audit':str(output)}),flush=True)
