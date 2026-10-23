"""Bounded known-label compiler replay, independent of search/performance selection."""
import json
from pathlib import Path
import time

import torch

from .driver import CACHE, Session, save
from .encoder import hardware_features
from .guided import complete, execute
from .revision import load_model
from .stage_learning import forward_prefix, prepare_groups


def action_key(action):
    return action['kind'],action['target'],action['parameters']


def check_observation(actual, expected):
    for field in ('nodes','edges','stage','feature_schema'):
        assert actual.graph[field]==expected.graph[field], 'replay graph changed: '+field
    assert [action_key(a) for a in actual.graph['candidates']]==[action_key(a) for a in expected.graph['candidates']]
    assert actual.history==expected.history
    assert hardware_features(actual.hardware)==hardware_features(expected.hardware)
    # Snapshot/branch names are fresh; structural correspondence is unchanged.
    assert actual.lineage['relations']==expected.lineage['relations']


def replay(output, dataset, protocol):
    records=[]
    for key,recipe in protocol['models'].items():
        config={**dataset['config']['training'],'pair_version':recipe['pair_version']}
        groups=prepare_groups(dataset,'train',config)+prepare_groups(dataset,'validation',config)
        for case in protocol['replay_cases']:
            group=[g for g in groups if g['program']==case['program']][case['group_ordinal']]
            name=f'{output.name}-{key}-{case["program"]}-g{case["group_ordinal"]}'
            path=output/(name+'-replay.json')
            if path.exists():
                records.append(json.loads(path.read_text()));continue
            start=time.perf_counter()
            checkpoint=torch.load(output/(key+'.pt'),weights_only=False)
            assert checkpoint['optimizer_steps']==protocol['updates']
            model=load_model(output/(key+'.pt'));model.eval()
            with torch.no_grad():expected_outputs,_=forward_prefix(model,group)
            expected_scores=expected_outputs[0]
            session=Session(name+'-root',program=case['program'])
            session.observe('S0')
            # All mutations happen in a private checkpoint branch.
            child=session.fork(name+'-branch')
            state=None
            for index,expected in enumerate(group['observations']):
                actual=child.current if index==0 else child.observe(expected.stage)
                check_observation(actual,expected)
                with torch.no_grad():(scores,value,resources),state=model(actual,state)
                if index+1<len(group['observations']):
                    action=json.loads((CACHE/'runs'/group['prefix']/'trial-0-action.json').read_text())
                    selected=next(i for i,a in enumerate(actual.graph['candidates']) if action_key(a)==action_key(action))
                    execute(child,child.manifest['stages'][index],selected)
            error=float((scores-expected_scores).abs().max())
            assert error<1e-6, 'fresh replay prediction differs from frozen matched prefix'
            selected=max(group['rows'],key=lambda r:float(scores[r['candidate_index']]))
            stage_index=int(group['stage'][1:])
            execute(child,child.manifest['stages'][stage_index],selected['candidate_index'],scores.tolist(),float(value),resources.tolist())
            report=complete(child,stage_index+1,policy='advance')
            identical=(child.output/'program.o').read_bytes()==(Path(selected['report']).parent/'program.o').read_bytes()
            assert identical, 'replayed object differs from matched label'
            assert report['comparison_group']==json.loads(Path(selected['report']).read_text())['comparison_group']
            events={e['log']:e for s in (session,child) for e in s.events}
            result={'model':key,'program':case['program'],'split':group['split'],'stage':group['stage'],'prefix':group['prefix'],
                'optimizer_steps':checkpoint['optimizer_steps'],'selected_known_index':selected['candidate_index'],
                'full_proposal_index':int(scores.argmax()),'scores':scores.tolist(),'matched_report':selected['report'],
                'report':str(child.output/'report.json'),'object_equal':identical,'prediction_max_error':error,
                'correctness':report['measurement']['correctness'],'latency_ms_diagnostic_only':report['median_latency_ms'],
                'wall_s':time.perf_counter()-start,'command_s':sum(e['seconds'] for e in events.values()),
                'measurement_invocations':sum(e['stage']=='measure' for e in events.values()),'events':list(events.values()),
                'model_costs':model.costs,'scope':'known candidate set, all-advance-v1 suffix; no new label or speedup claim'}
            save(path,result);records.append(result)
            print(json.dumps({'replayed':name,'selected_known_index':selected['candidate_index'],'object_equal':identical}),flush=True)
    path=output/'replay.json'
    if not path.exists():
        save(path,{'cases':records,'actual_terminals':len(records),'measurement_invocations':sum(r['measurement_invocations'] for r in records),
            'command_s':sum(r['command_s'] for r in records),'wall_s':sum(r['wall_s'] for r in records)})
    return records
