"""A: dedicated sparse Transformer per boundary, differentiable stage alignment."""
import time
from torch import nn
from .encoder import Encoder, SparseBlock, aligned
from .heads import Heads


class Interleaved(nn.Module):
    def __init__(self, hidden=128, heads=4, history=True, encoder_version="legacy-v1", decision_version="policy-v1"):
        super().__init__()
        self.hidden, self.history = hidden, history
        self.encoder = Encoder(hidden, encoder_version)
        if encoder_version == "hierarchical-v3":
            from .hierarchy import Hierarchy
            self.hierarchy = Hierarchy(hidden, heads)
        block_version = "structured-v2" if encoder_version == "hierarchical-v3" else encoder_version
        self.blocks = nn.ModuleList([SparseBlock(hidden, heads, block_version) for _ in range(3)])
        self.heads = Heads(hidden, decision_version, encoder_version == "hierarchical-v3")
        self.costs = []

    def forward(self, observation, previous=None):
        start = time.perf_counter()
        x = self.encoder.contextualize(self.encoder.base(observation.graph), observation)
        summaries = {}
        if hasattr(self, "hierarchy"):
            x, summaries = self.hierarchy(x, observation.graph)
        encoded = time.perf_counter()
        x = x + aligned(previous if self.history else None, observation, self.hidden)
        h = self.blocks[int(observation.stage[1:])](x, observation.graph)
        contexts = None
        if hasattr(self, "hierarchy"):
            contexts, reading = self.hierarchy.read(h, observation.graph)
            summaries.update(reading)
        outputs = self.heads(h, observation.graph["candidates"], contexts)
        self.costs.append({**summaries, "full_encoding_s": encoded - start, "network_s": time.perf_counter() - encoded,
                           "read_state_nodes": sum(len(r["new"]) for r in observation.lineage["relations"] if r["status"] == "known") if previous is not None and self.history else 0})
        return outputs, {"h": h}
