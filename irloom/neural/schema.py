"""Versioned records shared by both models and the compiler session."""
from dataclasses import dataclass, field


@dataclass
class CandidateAction:
    snapshot: str
    kind: str
    target: int
    parameters: dict = field(default_factory=dict)
    precondition: str = ""


@dataclass
class Observation:
    schema: int
    pipeline: str
    stage: str
    snapshot: str
    branch: str
    graph: dict
    hardware: dict
    lineage: dict
    history: list


@dataclass
class Decision:
    snapshot: str
    ranking: list
    selected: dict
    scores: list
    value: float = 0.0
    resources: list = field(default_factory=list)


@dataclass
class Transition:
    source: str
    destination: str
    success: bool
    action: dict
    lineage: dict
    seconds: float
    diagnostic: str = ""


@dataclass
class TerminalMeasurement:
    correctness: list
    samples: list
    code_bytes: int
    costs: dict
