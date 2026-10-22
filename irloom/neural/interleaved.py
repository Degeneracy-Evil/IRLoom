"""A: dedicated sparse Transformer per boundary, differentiable stage alignment."""
import time
from torch import nn
from .encoder import Encoder, SparseBlock, aligned
from .heads import Heads


class Interleaved(nn.Module):
    def __init__(self, hidden=128, heads=4, history=True):
        super().__init__()
        self.hidden, self.history = hidden, history
        self.encoder = Encoder(hidden)
        self.blocks = nn.ModuleList([SparseBlock(hidden, heads) for _ in range(3)])
        self.heads = Heads(hidden)
        self.costs = []

    def forward(self, observation, previous=None):
        start = time.perf_counter()
        x = self.encoder.contextualize(self.encoder.base(observation.graph), observation)
        encoded = time.perf_counter()
        x = x + aligned(previous if self.history else None, observation, self.hidden)
        h = self.blocks[int(observation.stage[1:])](x, observation.graph)
        outputs = self.heads(h, observation.graph["candidates"])
        self.costs.append({"full_encoding_s": encoded - start, "network_s": time.perf_counter() - encoded})
        return outputs, {"h": h}
