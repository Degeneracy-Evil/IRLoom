"""Conservative external snapshot alignment; names are scoped stable anchors only."""
import copy
from dataclasses import dataclass
from pathlib import Path


def align(previous, current, events=()):
    if previous is None:
        result = {"relations": [], "unknown": list(range(len(current["nodes"]))), "event": "initial"}
        return statistics(result, current, events)
    old = {n["anchor"]: n["id"] for n in previous["nodes"] if n["anchor"]}
    relations = [{"old": [old[n["anchor"]]], "new": [n["id"]], "status": "known"}
                 for n in current["nodes"] if n["anchor"] in old]
    mapped = {i for r in relations for i in r["new"]}
    # Compose controlled rewrites only without intervening arbitrary lowering.
    if events and all(e.get("success") for e in events):
        mapping = {i: {i} for i in range(len(previous["nodes"]))}
        for event in events:
            next_mapping = {}
            for r in event["lineage"]["relations"]:
                origins = set().union(*(mapping.get(i, set()) for i in r["old"]))
                if origins:
                    for new in r["new"]:
                        next_mapping[new] = origins
            mapping = next_mapping
        relations = [r for r in relations if r["new"][0] not in mapping]
        relations += [{"old": sorted(old), "new": [new], "status": "known"} for new, old in mapping.items()]
    mapped = {i for r in relations for i in r["new"]}
    result = {"relations": relations, "unknown": [n["id"] for n in current["nodes"] if n["id"] not in mapped],
              "event": "controlled composition" if events and all(e.get("success") for e in events) else "anchor-only; lowering internals unknown"}
    return statistics(result, current, events)


def statistics(result, current, events):
    mapped = {i for r in result["relations"] for i in r["new"]}
    counts = {}
    for n in current["nodes"]:
        row = counts.setdefault(n["kind"], {"known": 0, "unknown": 0})
        row["known" if n["id"] in mapped else "unknown"] += 1
    result["coverage"] = counts
    result["ledger_relations"] = sum(len(e.get("lineage", {}).get("relations", [])) for e in events)
    result["lowering_events"] = sum(e.get("event") == "lowering" for e in events)
    return result


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
    observed_ir: Path
    observation: object
    state: object
    history: list
    rng: object
    remaining: int
    lineage: list
    lineage_cursor: int

    def fork(self):
        return Checkpoint(self.ir, self.observed_ir, copy.deepcopy(self.observation), clone_state(self.state),
                          copy.deepcopy(self.history), clone_state(self.rng), self.remaining, copy.deepcopy(self.lineage), self.lineage_cursor)
