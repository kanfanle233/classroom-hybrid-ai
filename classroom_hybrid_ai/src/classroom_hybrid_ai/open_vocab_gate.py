"""Optional open-vocabulary candidate gate contract.

X-CLIP and MobileViCLIP are intentionally not hard dependencies of the 8 GB
profile.  This module gives a future provider one narrow insertion point while
ensuring an enabled-but-unverified provider cannot silently change the review
queue.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class OpenVocabGateError(RuntimeError):
    """Raised when an optional gate is enabled without a verified provider."""


class OpenVocabGate(Protocol):
    provider: str

    def rank(
        self,
        candidates: Sequence[Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> list[Any]:
        """Return the same candidates in provider-ranked order."""

    def close(self) -> None:
        """Release provider resources."""


class DisabledOpenVocabGate:
    """No-op gate used by default; it cannot affect candidate selection."""

    provider = "none"

    def rank(
        self,
        candidates: Sequence[Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> list[Any]:
        del evidence
        return list(candidates)

    def close(self) -> None:
        return None


def build_open_vocab_gate(
    config: Any,
    *,
    provider_factory: Any = None,
) -> OpenVocabGate | DisabledOpenVocabGate:
    """Build the explicitly configured optional gate.

    A provider may be enabled only after an external same-evidence benchmark
    marks it verified.  The actual X-CLIP/MobileViCLIP adapters are injected by
    a caller so importing this package never downloads or imports either model.
    """

    options = config.get("open_vocabulary.open_vocab_gate", {})
    if not isinstance(options, Mapping) or not bool(options.get("enabled", False)):
        return DisabledOpenVocabGate()
    provider = str(options.get("provider", "none")).strip().lower()
    if provider not in {"xclip", "mobileviclip"}:
        raise OpenVocabGateError(f"unsupported_open_vocab_gate_provider:{provider}")
    if options.get("benchmark_verified") is not True:
        raise OpenVocabGateError("open_vocab_gate_requires_same_evidence_benchmark")
    if provider_factory is None:
        raise OpenVocabGateError(f"open_vocab_gate_provider_not_installed:{provider}")
    gate = provider_factory(provider, config)
    if not hasattr(gate, "rank") or not callable(gate.rank):
        raise OpenVocabGateError("open_vocab_gate_provider_missing_rank")
    return gate


__all__ = [
    "DisabledOpenVocabGate",
    "OpenVocabGate",
    "OpenVocabGateError",
    "build_open_vocab_gate",
]
