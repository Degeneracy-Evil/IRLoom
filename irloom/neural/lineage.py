"""Conservative external snapshot alignment; names are scoped stable anchors only."""
import copy
from dataclasses import dataclass
from pathlib import Path


def align(previous, current):
    if previous is None:
        return {"relations": [], "unknown": list(range(len(current["nodes"]))), "event": "initial"}
    old = {n["anchor"]: n["id"] for n in previous["nodes"] if n["anchor"]}
    relations = [{"old": [old[n["anchor"]]], "new": [n["id"]], "status": "known"}
                 for n in current["nodes"] if n["anchor"] in old]
    mapped = {i for r in relations for i in r["new"]}
    return {"relations": relations, "unknown": [n["id"] for n in current["nodes"] if n["id"] not in mapped],
            "event": "lowering: internal correspondence unknown; refresh"}


def clone_state(value):
    # Tensor.clone preserves autograd; deepcopy of non-leaf tensors is invalid.
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, dict):
        return {k: clone_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_state(v) for v in value]
    return copy.deepcopy(value)


@dataclass
class Checkpoint:
    ir: Path
    observation: object
    state: object
    history: list
    rng: object
    remaining: int

    def fork(self):
        return Checkpoint(self.ir, copy.deepcopy(self.observation), clone_state(self.state),
                          copy.deepcopy(self.history), clone_state(self.rng), self.remaining)
