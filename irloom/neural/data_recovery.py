"""Bounded post-fix recovery: fixed paths, fresh batches, explicit inheritance."""
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import shutil
import statistics
import time

from .driver import ROOT, CACHE, Session, save
from .correctness import target_key, locate, finish, program_checks
from .guided import execute
from .label_timing import retime
from .programs import RECIPES
from .schema import TerminalMeasurement
from .stage_learning import prepare_groups
from .pair_labels import resolution

REVISION = 'cleanup-positional-v1'
BRIDGE = CACHE / 'compilers' / REVISION / 'bin/irloom-polygeist-bridge'


def read(path):
    return json.loads(Path(path).read_text())


def check_compiler(session):
    assert session.compiler_revision == REVISION and session.bridge == BRIDGE
    assert session.manifest['compiler_revision'] == REVISION
    expected = read(ROOT / 'compiler/pipelines/polygeist-v1.json')
    expected.update(id=expected['id'] + '-' + REVISION, compiler_revision=REVISION)
    assert session.manifest == expected


class RecoverySession(Session):
    def run(self, args, label):
        check_compiler(self)
        if label in ('observe', 'apply'):
            assert Path(args[0]) == BRIDGE
        return super().run(args, label)


def semantic_actions(report):
    actions = []
    for transition in report['transitions']:
        assert transition['success']
        action = transition['action']
        before = next(o for o in report['observations'] if o['snapshot'] == action['snapshot'])
        actions.append({'stage': before['stage'], 'kind': action['kind'], 'parameters': action['parameters'],
                        'descriptor': target_key(before['graph'], action['target'])})
    assert len(actions) == 3
    return actions


def perform(session, action):
    check_compiler(session)
    stage = session.manifest['stages'][int(action['stage'][1:])]
    if session.current is None or session.current.stage != stage['id'] or session.ir != session.observed_ir:
        session.observe(stage['id'])
    index = locate(session.current.graph, action['descriptor'], action['kind'], action['parameters'])
    execute(session, stage, index)


def plan(source, evidence):
    programs = {'puzzle': 36, 'des': 35}
    selected = []; inherited = []
    reviews = {(r['program'], r['stage'], r['prefix'], r['candidate_index']): r for r in evidence['label_review']}
    for group in source['groups']:
        assert group['continuation'] == 'all-advance-v1'
        for row in group['rows']:
            identity = (group['program'], group['stage'], group['prefix'], row['candidate_index'])
            audit = reviews[identity]
            item = {'program': group['program'], 'stage': group['stage'], 'prefix': group['prefix'],
                    'candidate_index': row['candidate_index'], 'old_report': row['report'], 'mechanism_review': audit['label_review']}
            if group['program'] in programs:
                report = read(row['report']); item['actions'] = semantic_actions(report)
                action = item['actions'][int(group['stage'][1:])]
                original = group['observations'][-1]['graph']['candidates'][row['candidate_index']]
                assert action['kind'] == original['kind'] and action['parameters'] == original['parameters']
                assert action['descriptor'] == target_key(group['observations'][-1]['graph'], original['target'])
                selected.append(item)
            else:
                assert audit['label_review'] == 'unaffected_by_cleanup_fix'
                inherited.append({**item, 'origin': 'C2-mechanism-audited historical measurement', 'compiler_revision': 'legacy',
                                  'reference_report': group['reference_report']})
    assert Counter(r['program'] for r in selected) == programs and len(inherited) == 202
    assert sum(r['mechanism_review'] == 'requires_revalidation_and_recollection' for r in selected) == 13
    defaults = {p: {'old_report': source['defaults'][p], 'actions': semantic_actions(read(source['defaults'][p]))} for p in programs}
    return {'paths': selected, 'defaults': defaults, 'inherited': inherited, 'requested_slots': 73,
            'note': '71 original rows plus two defaults; root-row aliases retained. No candidates added.'}


def compile_terminal(session, actions, oracle):
    for action in actions[len(session.transitions):]:
        perform(session, action)
    exe = finish(session, 3)
    shutil.copy2(oracle, session.output / 'reference')
    checks = program_checks(session, exe, session.output / 'reference')
    if not all(c['status'] == 'pass' for c in checks):
        save(session.output / 'wrong-output.json', checks)
        raise RuntimeError('recovered candidate failed full output; comparison group must not be used')
    # No five-shot per-candidate timing: all labels come from the common batch.
    session.report(TerminalMeasurement(checks, [], (session.output / 'program.o').stat().st_size, {}),
                   'post-fix fixed trajectory; correctness complete, batch timing pending')


def collect_program(output, config, source, listing, program):
    path = output / (program + '.json')
    if path.exists():
        return read(path)
    started = time.perf_counter(); sessions = []; terminals = []; mapped = []; groups = []
    def fork(parent, suffix):
        child = parent.fork(output.name + '-' + program + '-' + suffix)
        check_compiler(child); sessions.append(child)
        return child
    try:
        root = RecoverySession(output.name + '-' + program + '-root', program=program, compiler_revision=REVISION)
        sessions.append(root); root.observe('S0')
        oracle = root.output / 'reference'
        root.run(['clang-18', *root.manifest['backend'], '-std=gnu89', '-Wno-implicit-int',
                  *['-D' + x for x in root.recipe.get('defines', [])], root.original_source, '-lm', '-o', oracle], 'reference-compile')
        baseline = fork(root, 'default'); terminals.append(baseline)
        compile_terminal(baseline, listing['defaults'][program]['actions'], oracle)
        for ordinal, old in enumerate(g for g in source['groups'] if g['program'] == program):
            parent = root
            selected = [r for r in listing['paths'] if r['program'] == program and r['stage'] == old['stage'] and r['prefix'] == old['prefix']]
            if old['stage'] == 'S1':
                parent = fork(root, 'prefix' + str(ordinal))
                perform(parent, selected[0]['actions'][0]); parent.observe('S1')
            # Candidate identity is checked structurally in the current snapshot.
            def identities(graph):
                return [(a['kind'], a['parameters'], target_key(graph, a['target'])) for a in graph['candidates']]
            assert identities(parent.current.graph) == identities(old['observations'][-1]['graph'])
            group = {**copy.deepcopy(old), 'observations': copy.deepcopy(parent.observations), 'rows': [],
                     'reference_report': str(baseline.output / 'report.json'), 'artifact_identity': 'executable',
                     'measurement_origin': 'postfix-fresh-batch', 'compiler_revision': REVISION,
                     'measurement_batch': output.name + ':' + program}
            for item in selected:
                if old['stage'] == 'S0' and item['candidate_index'] == 0:
                    child = baseline
                    assert item['actions'] == listing['defaults'][program]['actions']
                else:
                    child = fork(parent, f'g{ordinal}-a{item["candidate_index"]}')
                    terminals.append(child)
                    compile_terminal(child, item['actions'], oracle)
                report = str(child.output / 'report.json')
                original = next(r for r in old['rows'] if r['candidate_index'] == item['candidate_index'])
                group['rows'].append({**original, 'report': report, 'old_report': item['old_report']})
                assert semantic_actions(read(report)) == item['actions']
                mapped.append({**item, 'new_report': report})
            groups.append(group)
        distinct = []; aliases = []
        for terminal in terminals:
            content = (terminal.output / 'program').read_bytes()
            representative = next((s for s in distinct if (s.output / 'program').read_bytes() == content), None)
            if representative is None:
                distinct.append(terminal)
            else:
                aliases.append((terminal, representative))
        # Finish every compile/check before entering the native timing phase.
        timing = retime(distinct, baseline, config['timing_seed'], config['timing_rounds'])
        for alias, representative in aliases:
            report = read(alias.output / 'report.json'); canonical = read(representative.output / 'report.json')
            report['measurement']['samples'] = canonical['measurement']['samples']
            report['median_latency_ms'] = canonical['median_latency_ms']
            report['measurement_alias_report'] = str(representative.output / 'report.json')
            save(alias.output / 'report.json', report)
        artifacts = output / 'artifacts' / program; artifacts.mkdir(parents=True)
        for index, terminal in enumerate(terminals):
            check_compiler(terminal)
            for filename in ('program', 'program.o'):
                shutil.copy2(terminal.output / filename, artifacts / f'{index}-{filename}')
        result = {'program': program, 'groups': groups, 'mapping': mapped, 'default': str(baseline.output / 'report.json'),
                  'sessions': [str(s.output) for s in sessions], 'terminals': [str(s.output) for s in terminals],
                  'terminal_attempts': len(terminals), 'distinct_executables': len(distinct), 'aliases': len(aliases),
                  'timing': timing, 'environment': root.hardware, 'manifest': root.manifest, 'bridge': str(root.bridge),
                  'events': list({e['log']: e for s in sessions for e in s.events}.values()),
                  'wall_s': time.perf_counter() - started, 'failures': []}
        result['command_s'] = sum(e['seconds'] for e in result['events'])
        save(path, result)
        print(json.dumps({'recovered': program, 'paths': len(mapped), 'distinct': len(distinct), 'wall_s': result['wall_s']}), flush=True)
        return result
    except Exception as error:
        ledger = list({e['log']: e for s in sessions for e in s.events}.values())
        save(output / (program + '-failure.json'), {'error': str(error), 'events': ledger, 'terminal_attempts': len(terminals),
                                                  'wall_s': time.perf_counter() - started, 'command_s': sum(e['seconds'] for e in ledger)})
        raise


def verify_cache(output, records):
    """Validate the actual tool invocations and saved bytes, not just revision text."""
    for record in records:
        assert record['bridge'] == str(BRIDGE)
        expected = read(ROOT / 'compiler/pipelines/polygeist-v1.json')
        expected.update(id=expected['id'] + '-' + REVISION, compiler_revision=REVISION)
        assert record['manifest'] == expected
        for directory in record['sessions']:
            manifest = read(Path(directory) / 'manifest.json')
            assert manifest == record['manifest'] and manifest['compiler_revision'] == REVISION
        for event in record['events']:
            if event['stage'] in ('observe', 'apply'):
                assert event['command'][0] == str(BRIDGE) and event['status'] == 'ok'
        timing = record['timing']
        assert timing['rounds'] == 6 and timing['seed'] == 71
        assert all(sorted(order) == list(range(record['distinct_executables'])) for order in timing['orders'])
        assert all(timing['orders'][i] == list(reversed(timing['orders'][i-1])) for i in (1, 3, 5))
        assert timing['reports'][timing['default_index']] == record['default']
        for report_path in timing['reports']:
            directory = Path(report_path).parent
            logs = ROOT / 'logs/architecture' / directory.name
            report = read(report_path)
            assert report['measurement']['samples'] == [read(logs / f'label-{i}.json') for i in range(6)]
        for index, directory in enumerate(record['terminals']):
            for filename in ('program', 'program.o'):
                assert (Path(directory) / filename).read_bytes() == (output / 'artifacts' / record['program'] / f'{index}-{filename}').read_bytes()
            report = read(Path(directory) / 'report.json')
            assert report['comparison_group']['pipeline'] == record['manifest']
            assert all(c['status'] == 'pass' for c in report['measurement']['correctness'])
            assert len(report['measurement']['samples']) == 6
            if report.get('measurement_alias_report'):
                representative = read(report['measurement_alias_report'])
                assert report['measurement']['samples'] == representative['measurement']['samples']
                assert (Path(directory) / 'program').read_bytes() == (Path(report['measurement_alias_report']).parent / 'program').read_bytes()
            assert report['median_latency_ms'] == statistics.median(s['latency_ms'] for s in report['measurement']['samples'])
        for row in record['mapping']:
            assert semantic_actions(read(row['new_report'])) == row['actions']


def label_summary(dataset):
    result = []
    for split in ('train', 'validation'):
        for group in prepare_groups(dataset, split, dataset['config']['training']):
            advance = next(r for r in group['rows'] if r['candidate_index'] == 0)
            best = min(group['rows'], key=lambda r: r['canonical_ms'])
            result.append({'program': group['program'], 'family': group['family'], 'split': split,
                           'stage': group['stage'], 'prefix': group['prefix'], 'counts': group['counts'],
                           'reference_report': group['reference_report'], 'root_reference_ms': group['root_reference_ms'],
                           'advance_ms': advance['canonical_ms'], 'best_index': best['candidate_index'],
                           'best_speedup_vs_advance': advance['canonical_ms'] / best['canonical_ms'],
                           'best_speedup_vs_root': group['root_reference_ms'] / best['canonical_ms'],
                           'pairs': [{'a': group['rows'][i]['candidate_index'], 'b': group['rows'][j]['candidate_index'], 'direction': d, 'gamma': g} for i, j, d, g in group['pairs']],
                           'rows': [{'candidate_index': r['candidate_index'], 'q': r['target'], 'margin': r['margin'],
                                     'canonical_ms': r['canonical_ms'], 'samples': r['samples'],
                                     'resolution_vs_advance': resolution(r, advance, dataset['config']['training']),
                                     'report': r['report']} for r in group['rows']]})
    return result


def differences(old, new, listing):
    old_groups = label_summary(old); new_groups = label_summary(new)
    changes = []
    for before in old_groups:
        after = next(g for g in new_groups if all(g[k] == before[k] for k in ('program', 'stage', 'prefix')))
        a = {(p['a'], p['b']): p['direction'] for p in before['pairs']}
        b = {(p['a'], p['b']): p['direction'] for p in after['pairs']}
        status = Counter('retained' if pair in b and b[pair] == direction else 'reversed' if pair in b else 'indistinguishable' for pair, direction in a.items())
        status['new'] = len(b.keys() - a.keys())
        changes.append({'program': before['program'], 'family': before['family'], 'split': before['split'],
                        'stage': before['stage'], 'prefix': before['prefix'], 'refreshed': before['program'] in ('puzzle', 'des'),
                        'before': before, 'after': after, 'pair_changes': dict(status)})
    exposed = []
    for record in listing['paths']:
        if record['mechanism_review'] != 'requires_revalidation_and_recollection':
            continue
        g = next(c for c in changes if all(c[k] == record[k] for k in ('program', 'stage', 'prefix')))
        a = next(r for r in g['before']['rows'] if r['candidate_index'] == record['candidate_index'])
        b = next(r for r in g['after']['rows'] if r['candidate_index'] == record['candidate_index'])
        exposed.append({**record, 'new_report': b['report'], 'old_q': a['q'], 'new_q': b['q'],
                        'executable_changed': (Path(a['report']).parent / 'program').read_bytes() != (Path(b['report']).parent / 'program').read_bytes()})
    macro = {}
    for split in ('train', 'validation'):
        for stage in ('S0', 'S1'):
            rows = [g for g in new_groups if g['split'] == split and g['stage'] == stage]
            metrics = {}
            for unit in ('prefix', 'program', 'family'):
                buckets = defaultdict(list)
                for g in rows:
                    key = g['program'] + ':' + g['prefix'] if unit == 'prefix' else g[unit]
                    buckets[key].append(g)
                metrics[unit] = {'units': len(buckets), 'by_unit': {k: {
                    'beneficial_fraction': statistics.mean(g['counts']['beneficial'] / len(g['rows']) for g in values),
                    'harmful_fraction': statistics.mean(g['counts']['harmful'] / len(g['rows']) for g in values),
                    'strong_pair_fraction': statistics.mean(len(g['pairs']) / max(1, math.comb(len(g['rows']), 2)) for g in values),
                    'prefixes_with_strong_pairs': sum(bool(g['pairs']) for g in values)} for k, values in buckets.items()}}
                metrics[unit]['mean'] = {field: statistics.mean(v[field] for v in metrics[unit]['by_unit'].values())
                                         for field in ('beneficial_fraction', 'harmful_fraction', 'strong_pair_fraction')}
            macro[split + ':' + stage] = metrics
    return {'groups': changes, 'exposed_13': exposed, 'macro_coverage': macro,
            'scope': 'Old and new within-batch comparisons; no frozen-model inference, tuning, new candidates or training.'}


def run(args):
    config = read(ROOT / 'experiments/architecture/postfix-recovery-v1.json')
    output = CACHE / 'recoveries' / args.name; output.mkdir(parents=True, exist_ok=True)
    source = read(CACHE / config['source']); evidence = read(CACHE / config['mechanism_audit'])
    listing = plan(source, evidence)
    build_path = CACHE / 'compilers' / REVISION / 'build.json'
    build = read(build_path)
    assert build['revision'] == REVISION and not build['old_loop_utils_archive_member_extracted']
    assert build['patch'] == (ROOT / 'compiler/diagnostics/affine-cleanup-positional.patch').read_text()
    tool_paths = [BRIDGE, CACHE / 'frontends/build/bin/cgeist', CACHE / 'frontends/build/bin/polygeist-opt',
                  CACHE / 'frontends/build/bin/mlir-translate', CACHE / 'build/irloom-time']
    protocol = {'config': config, 'plan': listing, 'build': build,
                'tool_files': {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in tool_paths},
                'recipes': {p: RECIPES['programs'][p] for p in config['programs']}}
    cached = output / 'protocol.json'
    if cached.exists():
        if read(cached) != protocol:
            raise ValueError('recovery protocol/tool/source action mismatch; use a new namespace')
    else:
        save(cached, protocol)
        shutil.copy2(CACHE / config['source'], output / 'source-dataset.json')
    assert (output / 'source-dataset.json').read_bytes() == (CACHE / config['source']).read_bytes()
    records = [collect_program(output, config, source, listing, p) for p in config['programs']]
    verify_cache(output, records)
    path = output / 'dataset.json'
    if not path.exists():
        fresh = {(g['program'], g['stage'], g['prefix']): g for r in records for g in r['groups']}
        groups = []
        for old in source['groups']:
            key = (old['program'], old['stage'], old['prefix'])
            if key in fresh:
                groups.append(fresh[key])
            else:
                groups.append({**copy.deepcopy(old), 'measurement_origin': 'C2-mechanism-audited historical measurement',
                               'compiler_revision': 'legacy', 'mechanism_evidence': config['mechanism_audit'],
                               'measurement_batch': 'campaign-v1:' + old['program']})
        dataset = {'protocol': config, 'config': copy.deepcopy(source['config']), 'groups': groups,
                   'defaults': {**source['defaults'], **{r['program']: r['default'] for r in records}},
                   'inheritance': listing['inherited'], 'source_dataset': config['source'],
                   'scope': '71 refreshed records, 202 mechanism-audited historical records; not an all-fixed-toolchain measurement campaign.'}
        save(path, dataset)
    dataset = read(path)
    assert sum(len(g['rows']) for g in dataset['groups']) == 273
    for group in dataset['groups']:
        if group['program'] in config['programs']:
            expected = next(g for r in records for g in r['groups'] if all(g[k] == group[k] for k in ('program', 'stage', 'prefix')))
            assert group == expected
            assert group['reference_report'] == dataset['defaults'][group['program']]
            assert group['measurement_origin'] == 'postfix-fresh-batch'
        else:
            original = next(g for g in source['groups'] if all(g[k] == group[k] for k in ('program', 'stage', 'prefix')))
            assert original['rows'] == group['rows'] and original['reference_report'] == group['reference_report']
            reference = read(group['reference_report'])
            for row in group['rows']:
                report = read(row['report'])
                assert report['comparison_group'] == reference['comparison_group']
                assert all(c['status'] == 'pass' for c in report['measurement']['correctness'])
    difference = differences(source, dataset, listing)
    if (output / 'differences.json').exists():
        assert read(output / 'differences.json') == difference
    else:
        save(output / 'differences.json', difference)
    if not (output / 'acceptance.json').exists():
        save(output / 'acceptance.json', {'revision': REVISION, 'requested_slots': 73, 'refreshed_records': 71,
             'inherited_records': 202, 'terminal_attempts': sum(r['terminal_attempts'] for r in records),
             'distinct_executables': sum(r['distinct_executables'] for r in records),
             'native_measurements': sum(len(r['timing']['reports']) * 6 for r in records),
             'wall_s': sum(r['wall_s'] for r in records), 'command_s': sum(r['command_s'] for r in records),
             'correctness': 'all original recipe inputs checked before timing and every timed output checked',
             'session_compiler_and_artifact_cache_audit': 'pass', 'new_candidates': 0, 'training_updates': 0,
             'frozen_model_calls': 0, 'test_performance_calls': 0})
    print(json.dumps({'recovery': str(output), 'acceptance': read(output / 'acceptance.json')}), flush=True)
