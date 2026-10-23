"""Integration acceptance: real compiler forks, neural isolation and cache parity."""
from dataclasses import asdict
import time
import torch
from .driver import MANIFEST, Session, save
from .lineage import clone_state


def state_audit(model, name, context, size):
    root = Session(name + "-root", context, size, multi_function=True)
    obs = root.observe("S0")
    with torch.no_grad():
        _, root.state = model(obs)
    checkpoint = root.checkpoint()
    left_cp, right_cp = checkpoint.fork(), checkpoint.fork()
    original = right_cp.state["records"][0].clone()
    left_cp.state["records"][0].add_(1)
    assert torch.equal(right_cp.state["records"][0], original)
    assert torch.equal(checkpoint.state["records"][0], original)
    branches = [Session(name + "-" + side, context, size, multi_function=True) for side in ("left", "right")]
    for branch in branches:
        branch.restore(checkpoint.fork(), branch.output.name)
        branch.observations.append(asdict(branch.current))
    candidates = obs.graph["candidates"]
    assert len(candidates) > 1
    assert branches[0].apply(candidates[1]).success
    assert not branches[0].apply(candidates[1]).success
    assert branches[1].apply(candidates[0]).success
    assert branches[0].ir.read_bytes() != branches[1].ir.read_bytes()
    cache_costs = []
    for stage_index, stage in enumerate(MANIFEST["stages"]):
        if stage_index:
            pending = []
            # Interleave compiler branches. Each hidden state is tied to its
            # own transformed IR, not a shared mutable global model memory.
            for branch in branches:
                current = branch.observe(stage["id"])
                previous = clone_state(branch.state)
                with torch.no_grad():
                    model.incremental = False
                    full_start = time.perf_counter()
                    full, full_state = model(current, previous)
                    full_s = time.perf_counter() - full_start
                    model.incremental = True
                    inc_start = time.perf_counter()
                    incremental, inc_state = model(current, previous)
                    inc_s = time.perf_counter() - inc_start
                for a, b in zip(full, incremental):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(full_state["h"], inc_state["h"], rtol=1e-5, atol=1e-6)
                cache_costs.append({"branch": branch.branch, "stage": current.stage,
                                    "full_s": full_s, "incremental_s": inc_s, **model.costs[-1]})
                branch.state = inc_state
                selected = current.graph["candidates"][int(incremental[0].argmax())]
                assert branch.apply(selected).success
                # Re-evaluation from the same saved state after the other fork
                # must be independent of calls made on the shared nn.Module.
                pending.append((current, clone_state(previous), incremental))
            for current, previous, incremental in pending:
                with torch.no_grad():
                    replay, _ = model(current, previous)
                for a, b in zip(incremental, replay):
                    assert torch.equal(a, b)
        for branch in branches:
            branch.advance(stage)
    for branch in branches:
        branch.finalize()
    # Real checkpoint restore, then reapply S0 after both branches completed.
    replay = Session(name + "-restore", context, size, multi_function=True)
    replay.restore(checkpoint.fork(), "restored")
    draw = torch.rand(4)
    replay.restore(checkpoint.fork(), "restored")
    assert torch.equal(draw, torch.rand(4))
    assert replay.apply(candidates[1]).success
    assert replay.ir.read_bytes() == (branches[0].output / "trial-0.mlir").read_bytes()
    result = {"fork_tensor_isolation": True, "interleaved_branches_correct": True,
              "restored_action_replay_identical": True, "full_incremental_close": True,
              "costs": cache_costs,
              "limitation": "full graph extraction and sparse attention still run; only unchanged base embeddings are cached"}
    save(root.output / "state-audit.json", result)
    return result


def structure_audit(name):
    """Compiler-backed semantic fixtures; no separate unit-test framework."""
    import json
    from .encoder import Encoder, edges
    root = Session(name)
    torch.set_num_threads(1)
    torch.manual_seed(23)
    graphs = {}
    for opcode in ('subi', 'divsi', 'cmpi slt,'):
        for reverse in (False, True):
            args = '%b, %a' if reverse else '%a, %b'
            source = f'module {{ func.func @fixture(%a: i32, %b: i32) -> i32 {{ %x = arith.{opcode} {args} : i32\n'
            source += ' %y = arith.extui %x : i1 to i32\n return %y : i32 } }' if opcode.startswith('cmpi') else ' return %x : i32 } }'
            label = opcode.split()[0] + str(int(reverse))
            path, output = root.output / (label + '.mlir'), root.output / (label + '.json')
            path.write_text(source)
            root.run([root.bridge, path, output, label, 'fixture', 'S0'], 'structure')
            graphs[label] = json.loads(output.read_text())
            renamed = root.output / (label + '-renamed.mlir')
            renamed.write_text(source.replace('%a', '%left').replace('%b', '%right').replace('%x', '%answer'))
            root.run([root.bridge, renamed, output, label, 'fixture', 'S0'], 'structure-rename')
            assert graphs[label] == json.loads(output.read_text())
        assert not torch.equal(edges(graphs[opcode.split()[0] + '0'], 'structured-v2'), edges(graphs[opcode.split()[0] + '1'], 'structured-v2'))
    for index, mapping in enumerate(('d0, d1', 'd1, d0')):
        path, output = root.output / f'map{index}.mlir', root.output / f'map{index}.json'
        # affine.apply has one result; use an unused structured map attribute to
        # exercise canonical typed affine-map parsing with two ordered results.
        path.write_text('module attributes {fixture.map = affine_map<(d0, d1) -> (' + mapping + ')>} {}')
        root.run([root.bridge, path, output, f'map{index}', 'fixture', 'S0'], 'structure-map')
        graphs[f'map{index}'] = json.loads(output.read_text())
    encoder = Encoder(32, 'structured-v2')
    assert graphs['map0']['nodes'][0]['attributes'] != graphs['map1']['nodes'][0]['attributes']
    assert not torch.equal(encoder.base(graphs['map0']), encoder.base(graphs['map1']))
    # Real controlled fusion -> observe: ledger relations must enter Observation.
    before = root.observe('S0')
    assert root.apply(before.graph['candidates'][1]).success
    after = root.observe('S0')
    assert after.lineage['event'] == 'controlled composition'
    assert after.lineage['coverage']['operation']['known'] > 0
    stage = root.manifest['stages'][0]
    root.advance(stage)
    lowered = root.observe('S1')
    assert lowered.lineage['lowering_events'] == 1
    assert lowered.lineage['coverage']['operation']['known'] == 0
    result = {'ssa_rename_invariant': True, 'operand_ports_distinguish_sub_div_compare': True,
              'affine_order_distinguishable': True, 'controlled_observation': after.lineage,
              'lowered_observation': lowered.lineage, 'events': root.events}
    save(root.output / 'structure-audit.json', result)
    print(json.dumps({'structure_audit': str(root.output / 'structure-audit.json')}))
    return result
