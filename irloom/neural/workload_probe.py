"""One bounded complete Susan compatibility probe, outside trajectory training."""
from collections import Counter
import argparse
import json
from pathlib import Path
import time

from .driver import ROOT, CACHE, save
from .data_recovery import RecoverySession, check_compiler
from .correctness import finish, target_key
from .guided import execute
from .programs import RECIPES


def graph_summary(graph):
    counts = Counter(n['kind'] for n in graph['nodes'])
    return {'nodes': len(graph['nodes']), 'edges': len(graph['edges']),
            'functions': counts['function'], 'regions': counts['region'],
            'actions': dict(Counter(a['kind'] for a in graph['candidates']))}


def run(name):
    output = CACHE / 'workload-probes' / name
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'benchmark-data/sources/llvm-test-suite-18.1.3/MultiSource/Benchmarks/MiBench/automotive-susan/susan.c'
    recipe = {'path': 'MiBench/automotive-susan/susan.c', 'check_args': [], 'measure_args': [],
              'scope': 'Bounded compatibility only, not train/validation/test or a performance label',
              'license': 'Source header and sibling LICENSE: Crown Copyright; research purposes only, header must remain, not for sale.'}
    protocol = {'schema': 'larger-workload-probe-v1', 'source': str(source), 'source_lines': len(source.read_text().splitlines()),
                'source_bytes': source.stat().st_size, 'source_url': 'https://github.com/llvm/llvm-test-suite/blob/72a946a42cbce06afef68b2326ff6c3869d09799/MultiSource/Benchmarks/MiBench/automotive-susan/susan.c',
                'suite_revision': '72a946a42cbce06afef68b2326ff6c3869d09799', 'compiler_revision': 'cleanup-positional-v1',
                'recipe': recipe, 'license_text': (source.parent / 'LICENSE').read_text(),
                'inputs': ['input_small.pgm', 'input_large.pgm'], 'operation': '-s', 'terminal_limit': 2,
                'timed_runs': 0, 'selection': 'Single complete >607-line multiroutine cached workload, chosen before execution; no speed filter.'}
    if (output / 'protocol.json').exists():
        assert json.loads((output / 'protocol.json').read_text()) == protocol
        if (output / 'result.json').exists():
            print(json.dumps(json.loads((output / 'result.json').read_text()))); return
        raise RuntimeError('partial probe preserved; choose a new name')
    save(output / 'protocol.json', protocol)
    RECIPES['programs']['susan_larger_probe'] = recipe
    started = time.perf_counter(); sessions = []; records = []
    result = {'protocol': protocol, 'status': 'started', 'terminals': records, 'graphs': {}, 'failures': []}
    try:
        root = object.__new__(RecoverySession); sessions.append(root)
        root.__init__(name + '-root', program='susan_larger_probe', compiler_revision='cleanup-positional-v1')
        root.observe('S0'); result['graphs']['S0'] = graph_summary(root.current.graph)
        oracle = root.output / 'reference'
        root.run(['clang-18', *root.manifest['backend'], '-std=gnu89', '-Wno-implicit-int', source, '-lm', '-o', oracle], 'reference-compile')
        for policy in ('default', 'first-joint'):
            terminal = root.fork(name + '-' + policy); sessions.append(terminal); selected = []
            for index, stage in enumerate(terminal.manifest['stages']):
                if index:
                    terminal.observe(stage['id'])
                summary = graph_summary(terminal.current.graph)
                result['graphs'][policy + ':' + stage['id']] = summary
                candidates = terminal.current.graph['candidates']
                choice = next((i for i, a in enumerate(candidates) if a['kind'] != 'advance'), 0) if policy == 'first-joint' and index < 2 else 0
                action = candidates[choice]
                selected.append({'stage': stage['id'], 'kind': action['kind'], 'parameters': action['parameters'],
                                 'descriptor': target_key(terminal.current.graph, action['target'])})
                execute(terminal, stage, choice)
            exe = finish(terminal, 3); checks = []
            for input_name in protocol['inputs']:
                image = source.parent / input_name
                expected = terminal.output / (input_name + '.reference-output.pgm')
                actual = terminal.output / (input_name + '.candidate-output.pgm')
                reference_stdout = terminal.run([oracle, image, expected, '-s'], 'image-oracle')
                candidate_stdout = terminal.run([exe, image, actual, '-s'], 'image-candidate')
                matches = reference_stdout == candidate_stdout and expected.read_bytes() == actual.read_bytes()
                checks.append({'input': str(image), 'args': [str(image), '<cache-output.pgm>', '-s'],
                               'status': 'pass' if matches else 'wrong_output', 'image_bytes': actual.stat().st_size,
                               'stdout_bytes': len(candidate_stdout), 'candidate_image': str(actual), 'oracle_image': str(expected)})
                if not matches:
                    raise RuntimeError('full PGM or stdout mismatch')
            check_compiler(terminal)
            records.append({'policy': policy, 'run': str(terminal.output), 'actions': selected, 'checks': checks,
                            'compiler_revision': terminal.compiler_revision, 'manifest': terminal.manifest, 'bridge': str(terminal.bridge)})
        result['status'] = 'correct-two-paths'
        result['same_complete_executable'] = (Path(records[0]['run']) / 'program').read_bytes() == (Path(records[1]['run']) / 'program').read_bytes()
        result['joint_path_nonadvance'] = all(a['kind'] != 'advance' for a in records[1]['actions'][:2])
    except Exception as error:
        result['status'] = 'unsupported'
        result['failures'].append({'error': str(error), 'type': type(error).__name__})
        # If the supported Session graph limit rejected an already emitted graph,
        # retain its actual size rather than truncating it or lifting the limit.
        for session in sessions:
            for path in session.output.glob('root-*.json'):
                graph = json.loads(path.read_text())
                if 'nodes' in graph:
                    result['graphs']['emitted:' + path.name] = graph_summary(graph)
    result['events'] = list({e['log']: e for s in sessions for e in getattr(s, 'events', [])}.values())
    result['command_s'] = sum(e['seconds'] for e in result['events'])
    result['wall_s'] = time.perf_counter() - started
    result['environment'] = sessions[0].hardware
    result['terminal_attempts'] = sum(e['stage'] == 'link' for e in result['events'])
    result['scope'] = 'One fixed complete source, at most default + first-joint path; full file/stdout checks, zero native timing, no labels or training.'
    save(output / 'result.json', result)
    print(json.dumps({'probe': str(output), 'status': result['status'], 'graphs': result['graphs'],
                      'failures': result['failures'], 'command_s': result['command_s'], 'wall_s': result['wall_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='loomtrace-v1-susan')
    run(parser.parse_args().name)
