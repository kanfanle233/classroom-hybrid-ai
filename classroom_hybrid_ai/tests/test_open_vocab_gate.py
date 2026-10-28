from __future__ import annotations

import pytest
from pathlib import Path

from classroom_hybrid_ai.config import load_config
from classroom_hybrid_ai.open_vocab_gate import (
    DisabledOpenVocabGate,
    OpenVocabGateError,
    build_open_vocab_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_open_vocab_gate_is_disabled_without_changing_candidate_order() -> None:
    gate = build_open_vocab_gate(load_config(ROOT / "configs" / "default.yaml"))
    assert isinstance(gate, DisabledOpenVocabGate)
    candidates = ["late", "early"]
    assert gate.rank(candidates, []) == candidates


def test_open_vocab_gate_rejects_unverified_provider() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    payload = dict(config.payload)
    payload["open_vocabulary"] = dict(payload["open_vocabulary"])
    payload["open_vocabulary"]["open_vocab_gate"] = {
        "enabled": True,
        "provider": "xclip",
        "benchmark_verified": False,
    }
    from classroom_hybrid_ai.config import ProjectConfig

    errors = ProjectConfig(
        config.source_path, config.project_root, config.repository_root, payload
    ).validate_static()
    assert "open_vocab_gate_requires_same_evidence_benchmark" in errors

    # The builder also protects callers that construct a lightweight config
    # object without invoking the full static validator first.
    class MinimalConfig:
        def get(self, key, default=None):
            return {
                "open_vocabulary.open_vocab_gate": {
                    "enabled": True,
                    "provider": "xclip",
                    "benchmark_verified": False,
                }
            }.get(key, default)

    with pytest.raises(OpenVocabGateError, match="same_evidence_benchmark"):
        build_open_vocab_gate(MinimalConfig())
