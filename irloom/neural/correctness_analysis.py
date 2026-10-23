"""C2: read-only product coverage, mechanism exposure and equal-unit learning metrics."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics
import struct

from irloom.compiler import command
from .driver import ROOT,CACHE,save
from .correctness import target_key
from .stage_learning import prepare_groups
from .pair_labels import resolution


def elf_text(path):
    data=path.read_bytes()
    if data[:6]!=b'\x7fELF\x02\x01':raise ValueError('only recorded ELF64 little-endian artifacts are supported')
    offset=struct.unpack_from('<Q',data,40)[0];size,count,names=struct.unpack_from('<HHH',data,58)
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
    table=sections[names];strings=data[table[4]:table[4]+table[5]]
    for section in sections:
        name=strings[section[0]:].split(b'\0',1)[0]
        if name==b'.text':return data[section[4]:section[4]+section[5]]
    raise ValueError('recorded executable lacks .text')


def classes(values):
    representatives=[];indices=[]
    for value in values:
        if value not in representatives:representatives.append(value)
        indices.append(representatives.index(value))
    return indices,len(representatives)


def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def macro(rows,unit):
    fields=('direction_accuracy','regret','beneficial_selection','harmful_rejection','full_proposal_supported')
    grouped=defaultdict(list)
    for row in rows:grouped[row[unit]].append(row)
    by_unit={key:{field:mean([r[field] for r in values]) for field in fields} for key,values in grouped.items()}
    return {'units':len(by_unit),'ranking_units':sum(v['direction_accuracy'] is not None for v in by_unit.values()),
        'beneficial_units':sum(v['beneficial_selection'] is not None for v in by_unit.values()),
        'harmful_units':sum(v['harmful_rejection'] is not None for v in by_unit.values()),
        'metrics':{field:mean([v[field] for v in by_unit.values()]) for field in fields},'by_unit':by_unit}


def analyze(output,config):
    path=output/'candidate-analysis.json'
    if path.exists():return json.loads(path.read_text())
    dataset=json.loads((CACHE/config['campaign']/'corpus/dataset.json').read_text())
    objective=dataset['config']['training'];groups=prepare_groups(dataset,'train',objective)+prepare_groups(dataset,'validation',objective)
    training=json.loads((CACHE/config['campaign']/'models/training.json').read_text())
    logs=ROOT/'logs/architecture'/output.name/'candidate-analysis';logs.mkdir(parents=True,exist_ok=True)
    inspector=CACHE/'compilers'/config['revision']/'bin/inspect-loops'
    events=[];records=[];exposures=[];models={key:[] for key in training['models']}
    for ordinal,group in enumerate(groups):
        rows=group['rows'];advance=next(r for r in rows if r['candidate_index']==0)
        best=min(rows,key=lambda r:r['canonical_ms']);products=[]
        loops=[]
        if group['stage']=='S1':
            pre=CACHE/'runs'/group['prefix']/'after-S0.mlir';inspection=output/f'loops-{ordinal}.json'
            command([inspector,pre,inspection],logs/f'loops-{ordinal}.log',events,'read-only-loop-inspection',cpu=0)
            loops=json.loads(inspection.read_text())['loops']
        for row in rows:
            report=json.loads(Path(row['report']).read_text());native=Path(row['report']).parent;ir=Path(report['final_ir']).parent
            stage_index=int(group['stage'][1:]);high=ir/f'trial-{stage_index}.mlir'
            products.append({'mlir':high.read_bytes(),'llvm':(ir/'program.ll').read_bytes(),
                'object':row['object'],'executable':(native/'program').read_bytes(),'text':elf_text(native/'program')})
            exposure={'program':group['program'],'split':group['split'],'stage':group['stage'],'prefix':group['prefix'],
                'candidate_index':row['candidate_index'],'report':row['report'],'historical_correctness':'passed recorded full-output inputs',
                'label_review':'unaffected_by_cleanup_fix'}
            if group['stage']=='S1' and row['candidate_index']!=0:
                action=group['observations'][-1].graph['candidates'][row['candidate_index']]
                descriptor=target_key(group['observations'][-1].graph,action['target'])
                found=[l for l in loops if l['descriptor']==descriptor]
                if len(found)!=1:exposure.update(label_review='undetermined',descriptor=descriptor)
                elif found[0]['legacy_cleanup_risk']:
                    exposure.update(label_review='requires_revalidation_and_recollection',mechanism=found[0])
            exposures.append(exposure)
        artifact_classes={kind:classes([p[kind] for p in products]) for kind in products[0]}
        advance_position=rows.index(advance);best_position=rows.index(best)
        def effect(row):
            if row['object']==advance['object']:return 'neutral'
            delta=row['canonical_ms']-advance['canonical_ms']
            return 'neutral' if abs(delta)<=resolution(row,advance,objective) else 'harmful' if delta>0 else 'beneficial'
        benefits=[r['candidate_index'] for r in rows if effect(r)=='beneficial'];harmful=[r['candidate_index'] for r in rows if effect(r)=='harmful']
        statuses=[]
        for i,row in enumerate(rows):
            if products[i]['executable']==products[advance_position]['executable']:category='unchanged_executable'
            elif effect(row)=='beneficial':category='correct_measured_beneficial'
            elif effect(row)=='harmful':category='correct_measured_harmful'
            else:category='changed_no_resolvable_gain'
            statuses.append({'candidate_index':row['candidate_index'],'category':category,'effect':effect(row),
                'same_text_as_advance':products[i]['text']==products[advance_position]['text']})
        record={'program':group['program'],'family':group['family'],'split':group['split'],'stage':group['stage'],'prefix':group['prefix'],
            'candidate_pool':len(group['observations'][-1].graph['candidates']),'measured':len(rows),
            'classes':{k:n for k,(_,n) in artifact_classes.items()},'class_indices':{k:v for k,(v,_) in artifact_classes.items()},
            'measured_best':best['candidate_index'],'measured_best_speedup_vs_prefix_advance':advance['canonical_ms']/best['canonical_ms'],
            'beneficial_candidates':benefits,'harmful_candidates':harmful,'statuses':statuses,'strong_pairs':len(group['pairs']),
            'best_executable_same_as_advance':products[best_position]['executable']==products[advance_position]['executable']}
        records.append(record)
        for key,fit in training['models'].items():
            saved=fit['snapshots'][str(fit['optimizer_steps'])][group['split']]['groups']
            prediction=next(g for g in saved if g['program']==group['program'] and g['prefix']==group['prefix'] and g['stage']==group['stage'])
            chosen=next(r for r in rows if r['candidate_index']==prediction['selected_known_index'])
            regret=max(0,math.log(chosen['canonical_ms']/best['canonical_ms'])-math.log1p(resolution(chosen,best,objective)/best['canonical_ms']))
            models[key].append({'program':group['program'],'family':group['family'],'prefix_unit':group['program']+':'+group['stage']+':'+group['prefix'],
                'stage':group['stage'],'split':group['split'],'closed_index':chosen['candidate_index'],'full_index':prediction['selected_full_index'],
                'full_proposal_supported':prediction['selected_full_index'] in prediction['candidate_indices'],
                'direction_accuracy':prediction['correct_pair_directions']/len(group['pairs']) if group['pairs'] else None,
                'regret':regret if group['pairs'] else None,
                'beneficial_selection':chosen['candidate_index'] in benefits if benefits else None,
                'harmful_rejection':chosen['candidate_index'] not in harmful if harmful else None,
                'missed_measured_improvement':bool(benefits) and chosen['candidate_index'] not in benefits})
    scores={}
    for key,rows in models.items():
        scores[key]={split:{unit:macro([r for r in rows if r['split']==split],unit) for unit in ('prefix_unit','program','family')} for split in ('train','validation')}
    result={'groups':records,'label_review':exposures,'label_review_counts':dict(Counter(r['label_review'] for r in exposures)),
        'model_prefixes':models,'macro_scores':scores,'events':events,
        'scope':'Frozen legacy labels only; exact text/byte classes do not prove semantic equivalence. ELF .text equality excludes file metadata but not relocations/data. No new timing or proposal labels. Known best is not a full-pool oracle.'}
    save(path,result);return result


def review_exposed(output,config,analysis):
    path=output/'revalidated-labels.json'
    if path.exists():return json.loads(path.read_text())
    from .correctness import session,locate,finish,program_checks
    from .guided import execute
    records=[]
    exposed=[r for r in analysis['label_review'] if r['label_review']=='requires_revalidation_and_recollection']
    for ordinal,row in enumerate(exposed):
        old=json.loads(Path(row['report']).read_text());current=session(output.name+f'-exposed-{ordinal}',row['program'],config['revision'])
        for i,transition in enumerate(old['transitions']):
            stage=current.manifest['stages'][i];obs=current.observe(stage['id']);action=transition['action']
            before=next(o for o in old['observations'] if o['snapshot']==action['snapshot'])
            index=0 if action['kind']=='advance' else locate(obs.graph,target_key(before['graph'],action['target']),action['kind'],action['parameters'])
            execute(current,stage,index)
        exe=finish(current,3);oracle=current.output/'reference'
        current.run(['clang-18',*current.manifest['backend'],'-std=gnu89',*["-D"+x for x in current.recipe.get('defines',[])],current.original_source,'-lm','-o',oracle],'reference-compile')
        checks=program_checks(current,exe,oracle)
        same=exe.read_bytes()==(Path(row['report']).parent/'program').read_bytes()
        passed=all(c['status']=='pass' for c in checks)
        records.append({'program':row['program'],'prefix':row['prefix'],'candidate_index':row['candidate_index'],'old_report':row['report'],
            'new_run':str(current.output),'checks':checks,'same_executable':same,
            'review':'rechecked_same_executable' if passed and same else 'requires_new_performance_label' if passed else 'unresolved_wrong_output',
            'events':current.events})
    result={'candidates':records,'counts':dict(Counter(r['review'] for r in records)),
        'scope':'Only 13 structurally exposed old labels recompiled/checked, no timers or training. Different executable means old performance cannot be assigned to the repaired candidate. Historical files untouched.'}
    save(path,result);return result


def impact_summary(output,config,analysis,review):
    path=output/'impact-summary.json'
    if path.exists():return json.loads(path.read_text())
    dataset=json.loads((CACHE/config['campaign']/'corpus/dataset.json').read_text());groups=[]
    for split in ('train','validation'):
        groups+=prepare_groups(dataset,split,dataset['config']['training'])
    rows=[]
    for group in groups:
        bad={r['candidate_index'] for r in review['candidates'] if r['program']==group['program'] and r['prefix']==group['prefix'] and group['stage']=='S1'}
        touched=sum(group['rows'][i]['candidate_index'] in bad or group['rows'][j]['candidate_index'] in bad for i,j,*_ in group['pairs'])
        rows.append({'program':group['program'],'split':group['split'],'stage':group['stage'],'prefix':group['prefix'],
            'affected_candidates':sorted(bad),'strong_pairs':len(group['pairs']),'pairs_touching_changed_candidates':touched})
    text_changes=sum(elf_text(Path(r['new_run'])/'program')!=elf_text(Path(r['old_report']).parent/'program') for r in review['candidates'])
    result={'groups':rows,'changed_executable_texts':text_changes,
        'scope':'Legacy pair supervision touching a rewritten candidate is not evidence about the repaired candidate. No labels overwritten, no timing or retraining performed.'}
    save(path,result);return result
