from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Set

import torch
import torch.nn as nn


EVENT_TO_ACTIONS: Dict[str, Set[str]] = {
    "raise_hand": {"raise_hand", "raising_hand", "raise", "hand_raise"},
    "head_down": {"head_down", "doze", "sleep", "sleeping", "note", "writing", "write"},
    "discussion": {"chat", "chatting", "discussion", "group_discussion"},
    "respond_call": {"raise_hand", "listen", "chat"},
    "teacher_instruction": {"listen", "raise_hand", "read", "note", "teacher_interaction"},
    "teacher_interaction": {"teacher_interaction", "teacher_guidance", "teacher_instruction", "listen"},
    "stand": {"stand", "standing"},
    "chat": {"chat", "chatting", "group_discussion"},
    "group_discussion": {"group_discussion", "discussion", "chat", "chatting"},
    "phone": {"phone", "playing_phone", "cell phone", "turn_head"},
    "turn_head": {"turn_head", "turning_head", "distract", "phone"},
    "write": {"write", "writing", "note", "reading_writing", "head_down"},
    "listen": {"listen", "listening"},
    "read": {"read", "reading", "book", "note", "writing"},
    "tt": {"tt", "listen", "listening"},
    "dx": {"dx", "write", "writing", "note", "head_down", "doze"},
    "dk": {"dk", "read", "reading", "book"},
    "zt": {"zt", "turn_head", "turning_head", "distract"},
    "xt": {"xt", "group_discussion", "chat", "discussion"},
    "js": {"js", "raise_hand", "raising_hand", "raise"},
    "zl": {"zl", "stand", "standing"},
    "jz": {"jz", "teacher_interaction", "teacher_guidance", "teacher_instruction"},
    "unknown": set(),
}


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _text_similarity(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(a=a, b=b).ratio())


def action_match_score(event_type: str, query_text: str, action_label: str) -> float:
    event_key = (event_type or "unknown").strip().lower()
    action_key = (action_label or "").strip().lower()
    aliases = EVENT_TO_ACTIONS.get(event_key, set())
    if aliases and action_key in aliases:
        return 1.0
    if action_key == event_key:
        return 1.0
    best_alias = 0.0
    for alias in aliases:
        best_alias = max(best_alias, _text_similarity(alias, action_key))
    lexical = _text_similarity(query_text, action_key)
    return _clamp01(max(best_alias, lexical))


def build_feature_vector(
    *,
    event_type: str,
    query_text: str,
    action_label: str,
    overlap: float,
    action_confidence: float,
    uq_score: float,
) -> List[float]:
    text_score = action_match_score(event_type, query_text, action_label)
    overlap = _clamp01(overlap)
    action_confidence = _clamp01(action_confidence)
    uq_score = _clamp01(uq_score)
    stability_score = _clamp01(1.0 - uq_score)
    # Fixed 4-dim feature for compact verifier model.
    return [overlap, action_confidence, text_score, stability_score]


class VerifierMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> float:
    probs = probs.detach().cpu()
    labels = labels.detach().cpu()
    bins = torch.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = max(1, probs.numel())
    for i in range(num_bins):
        lo = bins[i].item()
        hi = bins[i + 1].item()
        mask = (probs >= lo) & (probs < hi if i < num_bins - 1 else probs <= hi)
        if mask.sum().item() == 0:
            continue
        conf = probs[mask].mean().item()
        acc = labels[mask].float().mean().item()
        ece += abs(conf - acc) * (mask.sum().item() / n)
    return float(ece)


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs = probs.detach().cpu()
    labels = labels.detach().cpu().float()
    return float(torch.mean((probs - labels) ** 2).item())


@dataclass
class VerifierRuntimeConfig:
    match_threshold: float = 0.60
    uncertain_threshold: float = 0.40
    uq_gate: float = 0.60
    temperature: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "match_threshold": float(self.match_threshold),
            "uncertain_threshold": float(self.uncertain_threshold),
            "uq_gate": float(self.uq_gate),
            "temperature": float(self.temperature),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "VerifierRuntimeConfig":
        cfg = cls()
        for key in ("match_threshold", "uncertain_threshold", "uq_gate", "temperature"):
            if key in data:
                setattr(cfg, key, float(data[key]))
        return cfg


def summarize_scores(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))

