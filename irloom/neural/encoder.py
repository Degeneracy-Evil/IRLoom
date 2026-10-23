"""Structured node encoding and genuinely edge-wise (O(E)) graph attention."""
import json
import math

import torch
from torch import nn

KINDS = {name: i for i, name in enumerate(("operation", "value", "block", "region", "function", "module"))}
ACTIONS = {"advance": 0, "fuse": 1, "unroll2": 2, "inline": 3}
RELATIONS = {name: i for i, name in enumerate(("uses", "defines", "contains", "argument", "cfg", "call", "conservative-memory", "unknown"))}


def hardware_features(hardware):
    fields = {x["field"].rstrip(":"): x["data"] for x in json.loads(hardware.get("cpu", '{"lscpu": []}'))["lscpu"]}
    flags = set(fields.get("Flags", "").split())
    values = [float("avx2" in flags), float("avx512f" in flags), float(bool(flags))]
    for level in (1, 2, 3):
        entries = [x for x in hardware.get("cpu_caches", []) if int(x["level"]) == level and x["type"] != "Instruction"]
        size = entries[0]["size"] if entries else ""
        scale = 1024 if size.endswith("K") else 1048576 if size.endswith("M") else 1
        values.extend([math.log1p(float(size.rstrip("KM")) * scale) / 24 if size else 0, float(bool(size))])
    return values


def numeric(node):
    features = list(node["features"][:8])
    features += [0] * (8 - len(features))
    return [math.copysign(math.log1p(abs(x)), x) / 10 if math.isfinite(x) else 0 for x in features]


class Encoder(nn.Module):
    def __init__(self, hidden, version="legacy-v1"):
        super().__init__()
        self.version = version
        if version not in ("legacy-v1", "structured-v2", "hierarchical-v3"):
            raise ValueError(version)
        self.names = nn.Embedding(256, hidden)
        self.kinds = nn.Embedding(len(KINDS), hidden)
        self.numbers = nn.Linear(8, hidden)
        self.context = nn.Linear(9 + 4, hidden)
        self.stages = nn.Embedding(3, hidden)

    def base(self, graph, indices=None):
        nodes = graph["nodes"] if indices is None else [graph["nodes"][i] for i in indices]
        # Byte vocabulary is fixed across programs/checkpoints; unknown operation
        # names still encode without growing a training-dependent vocabulary.
        names = []
        for n in nodes:
            attributes = n.get("attributes", "")
            if self.version == "legacy-v1":
                attributes = n.get("attributes_v1", attributes)
            names.append(list((n["name"] + ":" + n.get("type", "") + ":" + attributes).encode()))
        offsets, flat = [], []
        for name in names:
            offsets.append(len(flat)); flat.extend(name or [0])
        if self.version == "legacy-v1":
            text = nn.functional.embedding_bag(torch.tensor(flat), self.names.weight,
                                               torch.tensor(offsets), mode="mean")
        else:
            # Ordered bytes from canonical typed MLIR attributes. No vocabulary
            # learned from program identities and no silent sequence truncation.
            positions = torch.cat([torch.arange(len(name) or 1) for name in names]).float()
            weights = 1 + (positions * 0.37).sin() + (positions * 0.11).cos()
            text = nn.functional.embedding_bag(torch.tensor(flat), self.names.weight,
                       torch.tensor(offsets), mode="sum", per_sample_weights=weights)
            text = text / torch.tensor([len(name) or 1 for name in names])[:, None]
        return text + self.kinds(torch.tensor([KINDS[n["kind"]] for n in nodes])) + self.numbers(torch.tensor([numeric(n) for n in nodes]))

    def contextualize(self, base, observation):
        history = observation.history
        visible = [len(history) / 6, sum(x["success"] for x in history) / 6,
                   sum(x["kind"] == "fuse" for x in history) / 3,
                   sum(x["kind"] == "unroll2" for x in history) / 3]
        context = self.context(torch.tensor(hardware_features(observation.hardware) + visible))
        return base + context + self.stages.weight[int(observation.stage[1:])]


def edges(graph, version="legacy-v1"):
    if version == "structured-v2":
        # Keep parallel ports, relation direction and multiplicity. Artificial
        # reverse edges have distinct relation IDs from the original relation.
        records = []
        for edge in graph["edges"]:
            source, target, kind = edge[:3]
            position = edge[3] if len(edge) > 3 else 0
            relation = RELATIONS.get(kind, RELATIONS["unknown"])
            records.extend(((source, target, relation, position),
                            (target, source, relation + len(RELATIONS), position)))
        for i in range(len(graph["nodes"])):
            records.extend(((i, i, 16, 0), (0, i, 17, 0), (i, 0, 18, 0)))
        return torch.tensor(records).T
    pairs = {(i, i) for i in range(len(graph["nodes"]))}
    for edge in graph["edges"]:
        source, target = edge[:2]
        pairs.add((source, target)); pairs.add((target, source))
    # Module summary exchange is linear-sized, not all-pairs attention.
    for i in range(len(graph["nodes"])):
        pairs.add((0, i)); pairs.add((i, 0))
    return torch.tensor(sorted(pairs)).T


class SparseBlock(nn.Module):
    def __init__(self, hidden=128, heads=4, version="legacy-v1"):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.heads, self.width = heads, hidden // heads
        self.version = version
        if version == "structured-v2":
            self.relation_keys = nn.Embedding(19, hidden)
            self.relation_values = nn.Embedding(19, hidden)
            self.positions = nn.Linear(3, hidden, bias=False)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))

    def forward(self, x, graph):
        topology = edges(graph, self.version)
        src, dst = topology[:2]
        q, k, v = self.qkv(self.norm1(x)).reshape(len(x), 3, self.heads, self.width).unbind(1)
        keys, messages = k[src], v[src]
        if self.version == "structured-v2":
            relation, position = topology[2:]
            p = position.float()
            ports = self.positions(torch.stack((p / (1 + p), torch.log1p(p) / 10, (p * 0.37).sin()), -1))
            keys = keys + (self.relation_keys(relation) + ports).reshape(-1, self.heads, self.width)
            messages = messages + (self.relation_values(relation) + ports).reshape(-1, self.heads, self.width)
        logits = (q[dst] * keys).sum(-1) / math.sqrt(self.width)
        index = dst[:, None].expand(-1, self.heads)
        maximum = torch.full((len(x), self.heads), -torch.inf)
        maximum.scatter_reduce_(0, index, logits.detach(), reduce="amax", include_self=True)
        weight = (logits - maximum[dst]).exp()
        denominator = torch.zeros_like(maximum).scatter_add(0, index, weight)
        message = messages * (weight / denominator[dst]).unsqueeze(-1)
        pooled = torch.zeros_like(v).index_add(0, dst, message).reshape(len(x), -1)
        x = x + self.output(pooled)
        return x + self.ff(self.norm2(x))


def aligned(previous, observation, hidden):
    result = torch.zeros(len(observation.graph["nodes"]), hidden)
    if previous is None:
        return result
    for relation in observation.lineage["relations"]:
        if relation["status"] != "known":
            continue
        pooled = previous["h"][relation["old"]].mean(0)
        for new in relation["new"]:
            result[new] = pooled
    return result
