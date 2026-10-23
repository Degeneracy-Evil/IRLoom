"""Read-only forensic audit of frozen V1.1 decisions and surviving artifacts."""
import json
from pathlib import Path
from .driver import CACHE, save


def audit(name):
    pilot=json.loads((CACHE/'pilots/pilot-v1-1-canonical/report.json').read_text())
    searches=json.loads((CACHE/'comparisons/beam-v1-1-canonical/report.json').read_text())
    output=CACHE/'comparisons'/name
    output.mkdir(parents=True,exist_ok=False)
    result={'source':'frozen V1.1 canonical reports; no retraining or new performance labels','programs':{}}
    for variant,programs in searches['searches'].items():
        result['programs'][variant]={}
        for program,search in programs.items():
            default=Path(pilot['single_path']['default'][program]).parent
            terminals=[]
            for terminal in search['terminals']:
                path=Path(terminal['report']); report=json.loads(path.read_text())
                rows=[]
                for obs,decision in zip(report['observations'],report['decisions']):
                    rank=decision['ranking'].index(0)+1
                    rows.append({'stage':obs['stage'],'candidates':len(obs['graph']['candidates']),
                        'ranking':decision['ranking'],'scores':decision['scores'],'advance_rank':rank,
                        'selected':decision['selected'], 'default_excluded_by_local_width':rank>search['budget']['beam_width']})
                terminals.append({'report':str(path),'stages':rows,
                    'equal_default':{file:(path.parent/file).read_bytes()==(default/file).read_bytes() for file in ('program.o','program')},
                    'equal_default_ir':Path(report['final_ir']).read_bytes()==Path(json.loads((default/'report.json').read_text())['final_ir']).read_bytes()})
            paths=[Path(t['report']).parent for t in search['terminals']]
            first=terminals[0]['stages'][0]
            raw_path=Path(pilot['single_path'][variant][program])
            raw=json.loads(raw_path.read_text())
            raw_trace=[{'stage':o['stage'],'candidate_count':len(o['graph']['candidates']),
                        'advance_rank':d['ranking'].index(0)+1,'ranking':d['ranking'],
                        'scores':d['scores'],'selected':d['selected']}
                       for o,d in zip(raw['observations'],raw['decisions'])]
            result['programs'][variant][program]={'terminals':terminals,'actual_terminals':len(terminals),
                'raw_proposal':{'report':str(raw_path),'stages':raw_trace,
                    'equal_default_executable':(raw_path.parent/'program').read_bytes()==(default/'program').read_bytes()},
                'terminal_cap':search['budget']['max_terminal_candidates'],
                'default_first_excluded_stage':'S0' if first['default_excluded_by_local_width'] else 'not determined from surviving prefixes alone',
                'two_terminal_executables_equal':len(paths)==2 and (paths[0]/'program').read_bytes()==(paths[1]/'program').read_bytes()}
    save(output/'audit.json',result)
    return result
