"""Identical target-conditioned action/value/resource heads for A and B."""
import torch
from torch import nn
from .encoder import ACTIONS


class Heads(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.actions = nn.Embedding(len(ACTIONS), hidden)
        self.parameter_embedding = nn.Linear(1, hidden)
        self.rank = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.value = nn.Linear(hidden, 1)
        self.resource = nn.Linear(hidden, 2)

    def forward(self, h, candidates):
        target = h[torch.tensor([c["target"] for c in candidates])]
        kind = self.actions(torch.tensor([ACTIONS[c["kind"]] for c in candidates]))
        parameters = self.parameter_embedding(torch.tensor([[c["parameters"].get("operand", 0) / 10] for c in candidates]))
        score = self.rank(torch.cat([target, h[0].expand_as(target), kind + parameters], dim=-1)).squeeze(-1)
        return score, self.value(h[0]).squeeze(), self.resource(h[0])
