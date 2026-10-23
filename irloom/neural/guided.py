"""Shared finite search: explicit default seat, policy or stage-cost guidance."""
from dataclasses import asdict
import math
import statistics
import time

import torch

from irloom.compiler import CommandError
from .driver import Session, save
from .schema import Decision
from .programs import sample_existing


def execute(session, stage, index, scores=(), value=0.0, resources=()):
    observation = session.current
    action = observation.graph['candidates'][index]
    ranking = list(range(len(observation.graph['candidates'])))
    if len(scores):
        ranking = sorted(ranking, key=lambda i: -float(scores[i]))
    session.decisions.append(asdict(Decision(observation.snapshot, ranking, action, list(scores), value, list(resources))))
    transition=session.apply(action)
    if not transition.success:
        raise RuntimeError('registered action failed: '+transition.diagnostic)
    session.advance(stage)


def complete(session, start, policy='advance'):
    for stage in session.manifest['stages'][start:]:
        if session.current is None or session.current.stage != stage['id'] or session.observed_ir != session.ir:
            session.observe(stage['id'])
        index = min(1, len(session.current.graph['candidates'])-1) if policy == 'balanced' else 0
        execute(session, stage, index)
    return session.finalize()


def tolerance(a, b, config):
    ma, mb = statistics.median(a), statistics.median(b)
    mad = max(statistics.median(abs(x-ma) for x in a), statistics.median(abs(x-mb) for x in b))
    return max(config['near_tie_absolute_ms'], config['near_tie_relative']*max(ma,mb), config['uncertainty_mad_multiplier']*mad)


def guided(name, program, model, budget, mode='guarded', inject_failure=False):
    start = time.perf_counter()
    root = Session(name+'-root', program=program)
    root.deadline = start + budget['optimization_timeout_s']
    root.observe('S0')
    sessions, failures, traces = [root], [], []
    attempts, serial = 0, 0
    model.eval(); model.costs.clear()
    baseline, incumbent = None, None
    if mode == 'guarded':
        if budget['max_terminal_candidates'] < 2 or budget['max_action_attempts'] < 3:
            raise ValueError('guarded budget must pay for default and at least one proposal')
        baseline = root.fork(name+'-default'); sessions.append(baseline)
        attempts += 3
        report = complete(baseline, 0)
        incumbent = {'report':str(baseline.output/'report.json'), 'latency_ms':report['median_latency_ms'], 'role':'default'}
    active, terminals = [(0.0,root)], ([incumbent] if incumbent else [])
    is_q = getattr(model.heads, 'decision_version', 'policy-v1') == 'stage-q-v1'
    for stage_index,stage in enumerate(root.manifest['stages']):
        expanded=[]
        for parent_score,parent in active:
            try:
                observation = parent.current if stage_index == 0 else parent.observe(stage['id'])
                with torch.no_grad():
                    (scores,value,resources),parent.state=model(observation,parent.state)
                if not torch.isfinite(scores).all():
                    raise RuntimeError('nonfinite guidance')
                ranking=scores.argsort(descending=True,stable=True).tolist()
                row={'stage':stage['id'],'parent':parent.branch,'advance_rank':ranking.index(0)+1,
                     'ranking':ranking,'scores':scores.tolist(),'candidates':observation.graph['candidates'],
                     'scorer':'negative root-normalized log latency' if is_q else 'cumulative policy log probability'}
                traces.append(row)
                admitted=0
                width=1 if mode=='raw' else budget['beam_width']
                for index in ranking:
                    if attempts>=budget['max_action_attempts'] or time.perf_counter()>=root.deadline:
                        break
                    child=parent.fork(f'{name}-branch-{serial}'); serial+=1; sessions.append(child)
                    attempts+=1
                    if inject_failure=='deadline':
                        child.deadline=time.perf_counter()-1
                    elif inject_failure:
                        # Integration fault after the protected default is usable.
                        raise RuntimeError('injected proposal failure')
                    try:
                        execute(child,stage,index,scores.tolist(),value.item(),resources.tolist())
                    except (CommandError,RuntimeError) as error:
                        failures.append({'branch':child.branch,'error':str(error)}); continue
                    # S2 is observation-only and has no trained action-value label.
                    score=(float(scores[index]) if len(ranking)>1 else parent_score) if is_q else parent_score+float(scores.log_softmax(0)[index])
                    expanded.append((score,child)); admitted+=1
                    if admitted==width: break
            except (CommandError,RuntimeError) as error:
                failures.append({'branch':parent.branch,'error':str(error)})
        active=sorted(expanded,key=lambda item:item[0],reverse=True)[:(1 if mode=='raw' else budget['beam_width'])]
        if not active: break
    for score,candidate in active:
        if len(terminals)>=budget['max_terminal_candidates']: break
        try:
            report=candidate.finalize()
            terminals.append({'report':str(candidate.output/'report.json'),'latency_ms':report['median_latency_ms'],'role':'proposal','guidance':score})
        except (CommandError,RuntimeError) as error:
            failures.append({'branch':candidate.branch,'error':str(error)})
    if not terminals: raise RuntimeError('no correct terminal within the common budget')
    chosen=incumbent or min(terminals,key=lambda row:row['latency_ms'])
    def session_for(row):
        return next(s for s in sessions if str(s.output/'report.json')==row['report'])
    def samples(row):
        import json
        return [s['latency_ms'] for s in json.loads(open(row['report']).read())['measurement']['samples']]
    if mode=='guarded':
        for row in sorted(terminals,key=lambda r:r['latency_ms']):
            if row is incumbent: continue
            a,b=session_for(chosen),session_for(row)
            if (a.output/'program').read_bytes()==(b.output/'program').read_bytes(): continue
            if chosen['latency_ms']-row['latency_ms']>tolerance(samples(chosen),samples(row),budget):
                chosen=row
    confirmed=None
    try:
        selected_session=session_for(chosen)
        confirmed=sample_existing(selected_session,selected_session.output/'program',selected_session.output/'reference','confirm-')
    except (CommandError,RuntimeError) as error:
        failures.append({'confirmation':str(error)})
        if incumbent: chosen=incumbent
    events={e['log']:e for s in sessions for e in s.events}
    distinct=[]
    for row in terminals:
        binary=session_for(row).output/'program'
        if not any(binary.read_bytes()==p.read_bytes() for p in distinct): distinct.append(binary)
    result={'program':program,'mode':mode,'scorer':'stage-q-v1' if is_q else 'policy-v1', 'budget':budget,
            'selected':chosen,'terminals':terminals,'actual_terminals':len(terminals),'distinct_executables':len(distinct),
            'attempts':attempts,'default_paid':incumbent is not None,'fallback_to_default':chosen is incumbent,
            'traces':traces,'failures':failures,'confirmation':confirmed,
            'confirmation_complete':confirmed is not None,
            'wall_s':time.perf_counter()-start,'events':list(events.values()),'model_costs':list(model.costs),
            'command_s':sum(e['seconds'] for e in events.values()),
            'feature_s':sum(e['seconds'] for e in events.values() if e['stage']=='observe'),
            'deadline_overrun_s':max(0,time.perf_counter()-root.deadline),
            'measurement_invocations':sum(e['stage'] in ('measure','confirm-measure') for e in events.values())}
    save(root.output/'guided.json',result)
    return result
