"""Conservative semantic retrieval for the open classroom-action vocabulary.

The BGE encoder is deliberately local-only and CPU-only.  A missing or broken
encoder never causes a guessed synonym merge: the caller receives no semantic
matches and keeps a course-local provisional phrase instead.  This preserves
the distinction between *unavailable acceleration* and *semantic evidence*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .open_vocab import ActionDescriptor, normalize_phrase


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    """A candidate returned to the text-only synonym adjudicator."""

    action_id: str
    name_zh: str
    definition: str
    attributes: Mapping[str, Any]
    source: str

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name_zh": self.name_zh,
            "definition": self.definition,
            "attributes": dict(self.attributes),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class SemanticRetrieval:
    query: str
    candidates: tuple[SemanticCandidate, ...]
    scores: tuple[float, ...]
    status: str


class LocalBgeRetriever:
    """Lazy BAAI/bge-small-zh-v1.5 mean-pooling encoder.

    ``model_dir`` must be an already pinned local Hugging Face snapshot.  The
    class never invokes the network and only moves the encoder to CPU.
    """

    def __init__(self, model_dir: str | Path | None) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else None
        self._model: Any = None
        self._tokenizer: Any = None
        self._failure: str | None = None

    @property
    def available(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def status(self) -> str:
        if self.available:
            return "ready"
        if self._failure:
            return self._failure
        return "not_loaded"

    def _load(self) -> bool:
        if self.available:
            return True
        if self._failure is not None:
            return False
        if self.model_dir is None or not self.model_dir.is_dir():
            self._failure = "embedding_model_missing"
            return False
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self._model = AutoModel.from_pretrained(
                str(self.model_dir), local_files_only=True
            ).to("cpu").eval()
            # The explicit import keeps torch visible to static packagers and
            # makes it clear this runs inference under no_grad below.
            del torch
            return True
        except Exception as exc:  # model files/dependency errors are non-fatal
            self._model = None
            self._tokenizer = None
            self._failure = f"embedding_model_unavailable:{type(exc).__name__}"
            return False

    def encode(self, texts: Sequence[str]) -> np.ndarray | None:
        phrases = [normalize_phrase(text) for text in texts]
        if not phrases:
            return np.empty((0, 0), dtype=np.float32)
        if not self._load():
            return None
        try:
            import torch

            inputs = self._tokenizer(
                phrases, padding=True, truncation=True, max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self._model(**inputs)
                hidden = output.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            return vectors.detach().cpu().numpy().astype(np.float32, copy=False)
        except Exception as exc:
            self._failure = f"embedding_inference_failed:{type(exc).__name__}"
            return None

    def retrieve(
        self,
        phrase: str,
        candidates: Sequence[SemanticCandidate],
        *,
        threshold: float,
        top_k: int,
    ) -> SemanticRetrieval:
        query = normalize_phrase(phrase)
        if not query or not candidates:
            return SemanticRetrieval(query, (), (), "empty")
        vectors = self.encode([query, *[item.name_zh for item in candidates]])
        if vectors is None:
            return SemanticRetrieval(query, (), (), self.status)
        scores = vectors[1:] @ vectors[0]
        ranked = sorted(
            enumerate(scores.tolist()), key=lambda item: (-item[1], candidates[item[0]].action_id)
        )
        selected = [
            (candidates[index], float(score))
            for index, score in ranked[:max(0, top_k)]
            if float(score) >= threshold
        ]
        return SemanticRetrieval(
            query,
            tuple(item[0] for item in selected),
            tuple(round(item[1], 6) for item in selected),
            "ready",
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


Adjudicator = Callable[[str, Sequence[SemanticCandidate]], str | None]
BatchAdjudicator = Callable[
    [Sequence[tuple[str, Sequence[SemanticCandidate]]]], Mapping[str, str | None]
]


def canonicalize_discoveries(
    rows: Sequence[Mapping[str, Any]],
    *,
    registry_actions: Sequence[ActionDescriptor],
    retriever: LocalBgeRetriever,
    threshold: float,
    top_k: int,
    adjudicator: Adjudicator | None = None,
    batch_adjudicator: BatchAdjudicator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach a canonical phrase/action only after two order-inverted votes.

    Candidate attributes are passed to the adjudicator, which must reject
    incompatible posture, movement, hand, object or interaction evidence.  Matching IDs
    from both candidate orders are required.  Without an adjudicator, semantic
    retrieval remains audit-only and cannot merge phrases.
    """

    output = [dict(row) for row in rows]
    known = [
        SemanticCandidate(
            action_id=item.action_id,
            name_zh=item.name_zh,
            definition=item.definition,
            attributes=item.attributes,
            source="registry",
        )
        for item in registry_actions
    ]
    # One representative per original phrase ensures exactly two text Qwen
    # judgments for a proposed semantic merge, rather than a noisy vote per
    # video frame.
    representatives: dict[str, dict[str, Any]] = {}
    for row in output:
        phrase = normalize_phrase(str(row.get("raw_phrase") or ""))
        if phrase:
            representatives.setdefault(phrase, row)
    merged = 0
    unavailable = 0
    phrase_to_result: dict[str, tuple[str, str | None, dict[str, Any]]] = {}
    proposals: list[tuple[str, tuple[SemanticCandidate, ...]]] = []
    discovered: list[SemanticCandidate] = []
    for phrase in sorted(representatives):
        retrieval = retriever.retrieve(
            phrase, [*known, *discovered], threshold=threshold, top_k=top_k
        )
        if retrieval.status != "ready" and retrieval.status != "empty":
            unavailable += 1
        canonical_phrase = phrase
        canonical_action_id: str | None = None
        audit: dict[str, Any] = {
            "status": retrieval.status,
            "candidate_ids": [item.action_id for item in retrieval.candidates],
            "scores": list(retrieval.scores),
            "merged": False,
        }
        phrase_to_result[phrase] = (canonical_phrase, canonical_action_id, audit)
        if retrieval.candidates:
            proposals.append((phrase, retrieval.candidates))
        discovered.append(SemanticCandidate(
            action_id=f"phrase:{phrase}", name_zh=canonical_phrase,
            definition=f"可见的{canonical_phrase}", attributes={}, source="session",
        ))
    # One request covers all proposals and a second request reverses every
    # candidate list.  It keeps the Qwen request budget bounded by two while
    # retaining the required order-invariance check.
    if proposals and batch_adjudicator is not None:
        forward_votes = dict(batch_adjudicator(proposals))
        reverse_votes = dict(batch_adjudicator([
            (phrase, tuple(reversed(items))) for phrase, items in proposals
        ]))
    elif proposals and adjudicator is not None:
        forward_votes = {phrase: adjudicator(phrase, items) for phrase, items in proposals}
        reverse_votes = {
            phrase: adjudicator(phrase, tuple(reversed(items)))
            for phrase, items in proposals
        }
    else:
        forward_votes = {}
        reverse_votes = {}
    for phrase, items in proposals:
        forward = forward_votes.get(phrase)
        reverse = reverse_votes.get(phrase)
        canonical_phrase, canonical_action_id, audit = phrase_to_result[phrase]
        audit["forward_action_id"] = forward
        audit["reverse_action_id"] = reverse
        if forward and forward == reverse and any(item.action_id == forward for item in items):
            chosen = next(item for item in items if item.action_id == forward)
            canonical_phrase = chosen.name_zh
            canonical_action_id = chosen.action_id if chosen.source == "registry" else None
            audit["merged"] = True
            merged += 1
        phrase_to_result[phrase] = (canonical_phrase, canonical_action_id, audit)
    for row in output:
        phrase = normalize_phrase(str(row.get("raw_phrase") or ""))
        if not phrase:
            continue
        canonical_phrase, canonical_action_id, audit = phrase_to_result[phrase]
        row["canonical_phrase"] = canonical_phrase
        if canonical_action_id is not None:
            row["canonical_action_id"] = canonical_action_id
        row["semantic_merge"] = audit
    return output, {
        "status": "ready" if unavailable == 0 else "warning",
        "phrases": len(phrase_to_result),
        "merged": merged,
        "embedding_status": retriever.status,
    }


__all__ = [
    "LocalBgeRetriever", "SemanticCandidate", "SemanticRetrieval",
    "canonicalize_discoveries",
]
