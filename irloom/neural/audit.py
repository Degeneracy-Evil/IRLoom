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
