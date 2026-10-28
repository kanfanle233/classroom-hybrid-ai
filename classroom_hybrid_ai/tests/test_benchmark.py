from __future__ import annotations

import json

from classroom_hybrid_ai.benchmark import score_predictions


def test_blind_score_keeps_unknown_outside_the_eight_action_classes() -> None:
    rows = [
        {"gold_label": "dx", "predicted_label": "dx", "raw_phrase": "书写"},
        {"gold_label": "dk", "predicted_label": "UNKNOWN", "raw_phrase": "低头"},
    ]
    report = score_predictions(rows)
    assert report["unknown_rate"] == 0.5
    assert set(report["per_class_f1"]) == {"dx", "dk", "tt", "zt", "js", "zl", "xt", "jz"}


def test_blind_score_rejects_unknown_gold_label() -> None:
    try:
        score_predictions([{"gold_label": "unknown", "predicted_label": "UNKNOWN"}])
    except ValueError as exc:
        assert "gold_label" in str(exc)
    else:
        raise AssertionError("must reject a ninth gold action")
