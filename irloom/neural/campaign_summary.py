"""Small, auditable campaign summaries; raw trajectories remain in the cache."""
from collections import Counter
import json
import math
from pathlib import Path

from .driver import save


def compact_snapshot(snapshot):
    result={}
    for split in ('train','validation'):
        data=snapshot[split]
        if data is None:result[split]=None;continue
        stages={}
        for stage in ('S0','S1'):
            rows=[g for g in data['groups'] if g['stage']==stage]
            stages[stage]={'groups':len(rows),'strong_pairs':sum(g['strong_pairs'] for g in rows),
                'correct_directions':sum(g['correct_pair_directions'] for g in rows),
                'full_proposals_in_support':sum(g['selected_full_index'] in g['candidate_indices'] for g in rows),
                'choices':[{k:g[k] for k in ('program','prefix','selected_known_index','selected_full_index','strong_pairs','correct_pair_directions')} for g in rows]}
        result[split]={'metrics':data['metrics'],'stages':stages}
    return result


def summarize(output,config,catalog,dataset,diagnosis,training,evaluation):
    path=output/'summary.json'
    if path.exists():return json.loads(path.read_text())
    audit=json.loads((output/'corpus/audit.json').read_text())
    coverage={}
    for split in ('train','validation'):
        coverage[split]={}
        for stage in ('S0','S1'):
            gs=[g for g in audit['groups'] if g['split']==split and g['stage']==stage]
            coverage[split][stage]={'groups':len(gs),'context_rows':sum(g['labeled'] for g in gs),'strong_pairs':sum(g['strong_pairs'] for g in gs),
                'effects':dict(sum((Counter(g['effects']) for g in gs),Counter()))}
    models={}
    for key,r in training['models'].items():
        models[key]={'seed':r['seed'],'optimizer_steps':r['optimizer_steps'],'seconds':r['seconds'],'program_visits':r['program_visits'],
            'parameters':r['parameters'],'final':compact_snapshot(r['snapshots'][str(r['optimizer_steps'])]),
            'loss_first':r['curves'][0],'loss_last':r['curves'][-1],'raw_record':str(output/'models'/(key+'.json'))}
    development={}
    for key,r in diagnosis['models'].items():
        development[key]={'scope':r.get('scope','single'),'reused':r.get('reused'),'optimizer_steps':r['optimizer_steps'],'seconds':r['seconds'],
            'final':compact_snapshot(r['snapshots'][str(r['optimizer_steps'])]),'first_shared_gradient':r['curves'][0].get('shared_gradient_diagnostic'),
            'last_shared_gradient':r['curves'][-1].get('shared_gradient_diagnostic')}
    measurements=evaluation['measurement']['programs'];test_results={}
    for key in training['models']:
        test_results[key]={}
        for mode in ('raw','guarded'):
            rows=[r for r in evaluation['systems'] if r['model']==key and r['mode']==mode and r['program'] in catalog['selected']['test']]
            items={p:next((i for i in measurements[p]['items'] if i['method']==key+'-'+mode),None) for p in catalog['selected']['test']}
            complete=len(rows)==len(items) and all(r['status']=='ok' for r in rows)
            test_results[key][mode]={'complete':complete,'programs':items,
                'geomean_speedup':math.exp(sum(math.log(v['speedup']) for v in items.values())/len(items)) if complete and items else None,
                'fallbacks':sum(r.get('result',{}).get('fallback_to_default',False) for r in rows),
                'actual_terminals':sum(r.get('result',{}).get('actual_terminals',0) for r in rows),
                'wall_s':sum(r['wall_s'] for r in rows)}
    successful=[r for r in evaluation['matched'] if r['status']=='ok']
    result={'protocol':config,'corpus_audit':audit,'coverage':coverage,
        'development':{'selected_learning_rate':diagnosis['selected_learning_rate'],'selection_scores':diagnosis['selection_scores'],'models':development},
        'models':models,'matched':[{k:v for k,v in r.items() if k not in ('events','model_costs','scores')} for r in evaluation['matched']],
        'matched_successes':len(successful),'matched_outside_support':sum(not r['in_label_support'] for r in successful),
        'test_results':test_results,'system_failures':[{k:v for k,v in r.items() if k!='result'} for r in evaluation['systems'] if r['status']!='ok'],
        'costs':{'compatibility_wall_s':catalog['wall_s'],'label_wall_s':dataset['wall_s'],'new_development_training_s':diagnosis['new_training_s'],
            'reused_development_training_s':diagnosis['reused_training_s'],'main_training_s':training['seconds'],
            'matched_wall_s':evaluation['matched_wall_s'],'system_wall_s':evaluation['system_wall_s'],'final_measurement_wall_s':evaluation['measurement']['wall_s'],
            'final_measurement_invocations':evaluation['measurement']['measurement_invocations'],
            'note':'cumulative work, development/collection overlapped on CPU1/CPU0; not sum of serial elapsed wall. No model-cost subitems added twice.'},
        'raw_results':str(output),'limitations':'Three new heldout families, limited source distribution, all-advance teacher not Q*. S0 has no prior continuous state. Separate S1 observations and system guard benefit; no test feedback.'}
    save(path,result);return result
