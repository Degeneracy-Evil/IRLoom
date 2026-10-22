"""B: shared graph block and GRU, with module/function/region state records."""
import time
import torch
from torch import nn
from .encoder import Encoder, SparseBlock
from .heads import Heads

SUMMARY = {"module", "function", "region"}


def summary_owners(graph):
    parents = {}
    for source, target, kind in graph["edges"]:
        if kind in ("contains", "defines", "argument"):
            parents[target] = source
    result = []
    for node in graph["nodes"]:
        current, seen = node["id"], set()
        while graph["nodes"][current]["kind"] not in SUMMARY:
            if current in seen:
                raise ValueError("cyclic structural ownership")
            seen.add(current)
            current = parents.get(current, 0)
        result.append(current)
    return result


class Recurrent(nn.Module):
    def __init__(self, hidden=128, heads=4, history=True, incremental=False):
        super().__init__()
        self.hidden, self.history, self.incremental = hidden, history, incremental
        self.encoder = Encoder(hidden)
        self.block = SparseBlock(hidden, heads)
        self.update = nn.GRUCell(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.heads = Heads(hidden)
        self.costs = []

    def forward(self, observation, previous=None):
        start = time.perf_counter()
        graph = observation.graph
        previous = previous if self.history else None
        # Reuse only known, exactly unchanged encoder inputs. Unknown lowering
        # correspondence always refreshes. Cache is inference-only: no stale
        # parameter-dependent activations cross optimizer steps.
        reused = {}
        if self.incremental and not self.training and previous is not None:
            for relation in observation.lineage["relations"]:
                if relation["status"] != "known" or len(relation["old"]) != 1 or len(relation["new"]) != 1:
                    continue
                old, new = relation["old"][0], relation["new"][0]
                a, b = previous["graph"]["nodes"][old], graph["nodes"][new]
                keys = ("kind", "name", "type", "attributes", "features")
                if all(a.get(k) == b.get(k) for k in keys):
                    reused[new] = previous["base"][old]
        fresh = [i for i in range(len(graph["nodes"])) if i not in reused]
        if reused:
            base = torch.zeros(len(graph["nodes"]), self.hidden)
            if fresh:
                base[fresh] = self.encoder.base(graph, fresh)
            for new, value in reused.items():
                base[new] = value
        else:
            base = self.encoder.base(graph)
        encoded = time.perf_counter()
        summaries = [n["id"] for n in graph["nodes"] if n["kind"] in SUMMARY]
        inherited = torch.zeros_like(base)
        retained = set()
        if previous is not None:
            for relation in observation.lineage["relations"]:
                old = [i for i in relation["old"] if i in previous["records"]]
                new = [i for i in relation["new"] if i in summaries]
                if relation["status"] == "known" and old and new:
                    memory = torch.stack([previous["records"][i] for i in old]).mean(0)
                    inherited[new] = memory
                    retained.update(old)
        owners = summary_owners(graph)
        local = self.block(self.encoder.contextualize(base, observation) + inherited[owners], graph)
        written = torch.zeros_like(local)
        written[summaries] = self.update(local[summaries], inherited[summaries])
        h = self.norm(local + written[owners])
        # Store just the current live summaries. Deleted/unknown old regions
        # cannot be read on a later stage. h is for heads/gradient diagnostics;
        # recurrence exclusively reads records, never previous['h'].
        state = {"records": {i: h[i] for i in summaries}, "h": h, "base": base, "graph": graph}
        outputs = self.heads(h, graph["candidates"])
        self.costs.append({"encoding_s": encoded - start, "network_s": time.perf_counter() - encoded,
                           "mode": "incremental" if self.incremental and not self.training else "full",
                           "reused_nodes": len(reused), "fresh_nodes": len(fresh), "records": len(summaries),
                           "invalidated_records": len(previous["records"]) - len(retained) if previous else 0})
        return outputs, state
