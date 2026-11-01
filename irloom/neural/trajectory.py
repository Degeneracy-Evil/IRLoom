"""Frozen visible-IR behavior sampling and actual sequential compiler rollouts."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import time

from .driver import ROOT, CACHE, save
from .data_recovery import RecoverySession, REVISION, BRIDGE, read, check_compiler, compile_terminal
from .correctness import target_key, locate
from .data_recovery import semantic_actions
from .guided import execute
from .hierarchy import structure
from .label_timing import retime
from .programs import RECIPES, BASE
from .encoder import hardware_features


def probabilities(graph, behavior):
    parents, _, callees, _ = structure(graph)
    strata = {}
    for i, action in enumerate(graph['candidates']):
        if action['kind'] == 'advance':
            continue
        node = action['target']; function = -1; depth = 0
        while node in parents:
            node = parents[node]
            if graph['nodes'][node]['name'] in ('affine.for', 'scf.for'):
                depth += 1
            if graph['nodes'][node]['kind'] == 'function':
                function = node
                break
        key = (action['kind'], function, depth, callees.get(action['target'], -1))
        strata.setdefault(key, []).append(i)
    p = [0.0] * len(graph['candidates'])
    assert graph['candidates'][0]['kind'] == 'advance'
    p[0] = behavior['advance_probability'] if strata else 1.0
    for indices in strata.values():
        for i in indices:
            p[i] = (1 - p[0]) / len(strata) / len(indices)
    assert abs(sum(p) - 1) < 1e-10
    return p, len(strata)


def draw(graph, behavior, rng):
    p, count = probabilities(graph, behavior)
    u = rng.random(); total = 0.0
    for index, probability in enumerate(p):
        total += probability
        if u < total:
            return index, probability, count
    raise AssertionError('probability mass missing')


def describe(observation, index):
    action = observation.graph['candidates'][index]
    return {'stage': observation.stage, 'kind': action['kind'], 'parameters': action['parameters'],
            'descriptor': target_key(observation.graph, action['target'])}


def visible(observation):
    """Every model-consumed input, excluding run labels and unconsumed timings."""
    graph = observation['graph']
    return {'stage': observation['stage'], 'pipeline': observation['pipeline'],
            'nodes': graph['nodes'], 'edges': graph['edges'],
            'candidates': [{k: v for k, v in a.items() if k != 'snapshot'} for a in graph['candidates']],
            'relations': observation['lineage']['relations'], 'history': observation['history'],
            'hardware_features': hardware_features(observation['hardware'])}


def ledger(sessions):
    return list({e['log']: e for s in sessions for e in s.events}.values())


def collect_program(output, config, entry, ordinal):
    path = output / (entry['program'] + '.json')
    if path.exists():
        return read(path)
    start = time.perf_counter(); program = entry['program']; sessions = []; attempts = []; measured = []
    root = RecoverySession(output.name + '-' + program + '-root', program=program, compiler_revision=REVISION)
    sessions.append(root); root.observe('S0')
    oracle = root.output / 'reference'
    root.run(['clang-18', *root.manifest['backend'], '-std=gnu89', '-Wno-implicit-int',
              *['-D' + x for x in root.recipe.get('defines', [])], root.original_source, '-lm', '-o', oracle], 'reference-compile')
    def fork(suffix):
        child = root.fork(output.name + '-' + program + '-' + suffix)
        check_compiler(child); sessions.append(child)
        return child
    def rollout(child, rng, steps=None):
        if steps is None:steps = []
        for stage_index, stage in enumerate(child.manifest['stages']):
            obs = child.current if stage_index == 0 else child.observe(stage['id'])
            if rng is None:
                index, probability, strata = 0, 1.0, 0
            else:
                index, probability, strata = draw(obs.graph, config['behavior'], rng)
            action = describe(obs, index)
            steps.append({**action, 'index': index, 'snapshot': obs.snapshot, 'probability': probability,
                          'candidate_count': len(obs.graph['candidates']), 'nonadvance_strata': strata})
            execute(child, stage, index)
        compile_terminal(child, steps, oracle)
        assert semantic_actions(read(child.output / 'report.json')) == [{k: step[k] for k in ('stage','kind','parameters','descriptor')} for step in steps]
        return steps
    baseline = fork('default')
    default_steps = rollout(baseline, None); measured.append(baseline)
    for i in range(config['collection']['trajectories_per_program']):
        seed = config['behavior']['seed'] + ordinal * config['behavior']['program_seed_stride'] + i * config['behavior']['trajectory_seed_stride']
        child = fork('t' + str(i)); row = {**entry, 'id': program + ':' + str(i), 'seed': seed,
                                          'behavior_version': config['behavior']['version'], 'run': str(child.output)}
        attempts.append(row)
        row['steps'] = []
        try:
            rollout(child, random.Random(seed), row['steps'])
            row.update(status='correct', report=str(child.output / 'report.json'))
            measured.append(child)
        except Exception as error:
            row.update(status='failed', error=str(error), transitions=child.transitions, history=child.history)
            # Failed action/semantic paths remain charged, with no synthetic y.
            save(child.output / 'trajectory-failure.json', row)
        save(output / (program + '-progress.json'), {'attempts': attempts, 'events': ledger(sessions)})
    distinct = []; aliases = []
    for child in measured:
        content = (child.output / 'program').read_bytes()
        representative = next((s for s in distinct if (s.output / 'program').read_bytes() == content), None)
        if representative is None:
            distinct.append(child)
        else:
            aliases.append((child, representative))
    timing = retime(distinct, baseline, config['collection']['timing_seed'] + ordinal, config['collection']['timing_rounds'])
    for alias, representative in aliases:
        report = read(alias.output / 'report.json'); reference = read(representative.output / 'report.json')
        report['measurement']['samples'] = reference['measurement']['samples']
        report['median_latency_ms'] = reference['median_latency_ms']
        report['measurement_alias_report'] = str(representative.output / 'report.json')
        save(alias.output / 'report.json', report)
    base = read(baseline.output / 'report.json')['median_latency_ms']
    for row in attempts:
        if row['status'] == 'correct':
            report = read(row['report'])
            row['target'] = math.log(report['median_latency_ms'] / base)
            row['default_report'] = str(baseline.output / 'report.json')
    events = ledger(sessions)
    result = {**entry, 'attempts': attempts, 'default': str(baseline.output / 'report.json'), 'default_steps': default_steps,
              'terminal_attempts': 1 + len(attempts), 'distinct_executables': len(distinct), 'alias_terminals': len(aliases),
              'timing': timing, 'environment': root.hardware, 'manifest': root.manifest, 'bridge': str(root.bridge),
              'sessions': [str(s.output) for s in sessions], 'events': events,
              'command_s': sum(e['seconds'] for e in events), 'wall_s': time.perf_counter() - start}
    save(path, result)
    print(json.dumps({'trajectory_program': program, 'correct': sum(r['status']=='correct' for r in attempts),
                      'distinct': len(distinct), 'terminal_attempts': result['terminal_attempts']}), flush=True)
    return result


def audit(dataset, config):
    rows = []; total = 0
    for record in dataset['programs']:
        total += record['terminal_attempts']
        for session in record['sessions']:
            assert read(Path(session) / 'manifest.json') == record['manifest']
        assert record['manifest']['compiler_revision'] == REVISION and record['bridge'] == str(BRIDGE)
        for event in record['events']:
            if event['stage'] in ('observe', 'apply'):
                assert event['command'][0] == str(BRIDGE)
        before = []; kinds = Counter(); prefixes = defaultdict(list)
        default = read(record['default']); base = default['median_latency_ms']
        for row in record['attempts']:
            if row['status'] != 'correct':
                assert 'target' not in row
                continue
            report = read(row['report']); rng = random.Random(row['seed'])
            assert all(c['status']=='pass' for c in report['measurement']['correctness'])
            assert len(report['observations']) == len(report['transitions']) == 3
            assert len(report['measurement']['samples']) == config['collection']['timing_rounds']
            assert row['target'] == math.log(report['median_latency_ms'] / base)
            for stage, (obs, transition, chosen) in enumerate(zip(report['observations'], report['transitions'], row['steps'])):
                index, probability, _ = draw(obs['graph'], config['behavior'], rng)
                assert index == chosen['index'] and probability == chosen['probability']
                assert transition['action'] == obs['graph']['candidates'][index] and transition['success']
                assert len(obs['history']) == stage and all(h['success'] for h in obs['history'])
            signature = tuple(json.dumps({k:s[k] for k in ('stage','kind','parameters','descriptor')},sort_keys=True) for s in row['steps'])
            before.append(signature); kinds[tuple(s['kind'] for s in row['steps'][:2])] += 1
            prefixes[signature[0]].append((row, report))
        for path in record['timing']['reports']:
            report = read(path); logs = ROOT / 'logs/architecture' / Path(path).parent.name
            assert report['measurement']['samples'] == [read(logs / f'label-{i}.json') for i in range(config['collection']['timing_rounds'])]
        candidate_sets=[]; prefix_rows=[]
        for key, values in prefixes.items():
            s1 = values[0][1]['observations'][1]['graph']
            candidates = [(c['kind'], c['parameters'], target_key(s1,c['target'])) for c in s1['candidates']]
            if candidates not in candidate_sets:
                candidate_sets.append(candidates)
            y = [v[0]['target'] for v in values]
            prefix_rows.append({'S0': json.loads(key), 'rollouts': len(values), 'S1_candidates': len(candidates),
                                'target_min': min(y), 'target_max': max(y), 'rollout_sample_stdev': statistics.stdev(y) if len(y)>1 else None})
        rows.append({'program':record['program'],'family':record['family'],'split':record['split'],
                     'attempts':len(record['attempts']),'correct':len(before),'failed':len(record['attempts'])-len(before),
                     'distinct_executables':record['distinct_executables'],'unique_action_trajectories':len(set(before)),
                     'action_combinations':{'/'.join(k):v for k,v in kinds.items()}, 'distinct_S1_candidate_sets':len(candidate_sets),
                     'prefixes':prefix_rows,'root_reference_ms':base})
    assert total <= config['collection']['terminal_limit']
    return {'programs':rows,'terminal_attempts':total,'native_measurements':sum(len(r['timing']['reports'])*config['collection']['timing_rounds'] for r in dataset['programs']),
            'valid_trajectories':sum(r['correct'] for r in rows),'distinct_executables':sum(r['distinct_executables'] for r in rows),
            'command_s':sum(r['command_s'] for r in dataset['programs']),'collection_wall_s':sum(r['wall_s'] for r in dataset['programs']),
            'scope':'Targets only for sampled S0/S1 actions; defaults excluded from training; no ranking/counterfactual labels or new test domain.'}


def replay(output, config, dataset):
    path=output/'replay.json'
    if path.exists():
        return read(path)
    row=next(r for p in dataset['programs'] if p['split']=='train' for r in p['attempts']
             if r['status']=='correct' and all(s['kind']!='advance' for s in r['steps'][:2]))
    original=read(row['report']); session=RecoverySession(output.name+'-replay', program=row['program'], compiler_revision=REVISION)
    for step in row['steps']:
        obs=session.observe(step['stage'])
        index=locate(obs.graph,step['descriptor'],step['kind'],step['parameters'])
        assert visible(session.observations[-1]) == visible(original['observations'][int(step['stage'][1:])])
        execute(session,session.manifest['stages'][int(step['stage'][1:])],index)
    compile_terminal(session,row['steps'],Path(row['report']).parent/'reference')
    assert (session.output/'program').read_bytes()==(Path(row['report']).parent/'program').read_bytes()
    result={'trajectory':row['id'],'report':str(session.output/'report.json'),'all_model_visible_observations_equal':True,
            'same_complete_executable':True,'correct':True,'events':session.events,
            'scope':'Fresh process; actual joint non-advance path; only run labels/unused environment fields differ; no native performance timing.'}
    save(path,result)
    return result


def run(args):
    config=read(ROOT/'experiments/architecture/loomtrace-v1.json')
    assert (1+config['collection']['trajectories_per_program'])*len(config['programs'])<=config['collection']['terminal_limit']
    output=CACHE/'trajectories'/args.name;output.mkdir(parents=True,exist_ok=True)
    contract={'config':config,'pipeline':read(ROOT/'compiler/pipelines/polygeist-v1.json'),
              'build':read(CACHE/'compilers'/REVISION/'build.json'),
              'recipes':{e['program']:RECIPES['programs'][e['program']] for e in config['programs']}}
    manifest=output/'manifest.json'
    if manifest.exists():
        if read(manifest)!=contract:raise ValueError('trajectory protocol/compiler mismatch; use new namespace')
    else:save(manifest,contract)
    mode=args.loomtrace
    if mode in ('collect','all'):
        path=output/'dataset.json'
        if not path.exists():
            programs=[collect_program(output,config,e,i) for i,e in enumerate(config['programs'])]
            save(path,{'schema':config['schema'],'config':config,'programs':programs})
        dataset=read(path)
        assert dataset['config']==config
        result=audit(dataset,config)
        if (output/'data-audit.json').exists():assert read(output/'data-audit.json')==result
        else:save(output/'data-audit.json',result)
    else:dataset=read(output/'dataset.json')
    if mode=='collect':
        print(json.dumps({'trajectory_data':str(output),'audit':result}),flush=True);return
    if mode=='replay':
        print(json.dumps({'replay':replay(output,config,dataset)}),flush=True);return
    if mode=='all' and not (output/'replay.json').exists():
        import subprocess
        log=ROOT/'logs/architecture'/(output.name+'-fresh-replay.log')
        with log.open('w') as stream:
            subprocess.run(['bash',str(ROOT/'scripts/architecture-python.sh'),'-m','irloom.neural.driver',
                            '--loomtrace','replay','--name',args.name],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    from .trajectory_learning import train, evaluate, integration
    if mode in ('train','all'):train(output,config,dataset)
    if mode in ('audit','all'):integration(output,config,dataset)
    if mode in ('evaluate','all'):evaluate(output,config,dataset)
    print(json.dumps({'loomtrace':str(output),'mode':mode}),flush=True)
