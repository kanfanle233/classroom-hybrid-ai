from classroom_hybrid_ai.evaluation import compare_candidate, evaluate_rows
from classroom_hybrid_ai.benchmark import EIGHT_LABELS
import pytest


def test_candidate_requires_real_gain() -> None:
    baseline = evaluate_rows([
        {"gold_label": "dx", "predicted_label": "dx", "schema_valid": True},
        {"gold_label": "dk", "predicted_label": "dk", "schema_valid": True},
    ])
    same = evaluate_rows([
        {"gold_label": "dx", "predicted_label": "dx", "schema_valid": True},
        {"gold_label": "dk", "predicted_label": "dk", "schema_valid": True},
    ])
    result = compare_candidate(baseline, same)
    assert not result["eligible"]
    assert "macro_f1_gain_below_gate" in result["blockers"]


def test_core8_unknown_and_missing_measurements() -> None:
    result = evaluate_rows([{"gold_label": "dx", "predicted_label": "UNKNOWN"}])
    assert tuple(result.per_class_f1) == EIGHT_LABELS
    assert result.unknown_rate == 1.0
    assert result.mean_seconds_per_event is None
    assert result.peak_vram_gib is None
    assert result.strict_json_rate is None
    assert result.oom_count is None
    with pytest.raises(ValueError, match="gold_label"):
        evaluate_rows([{"gold_label": "write", "predicted_label": "write"}])


def test_equivalent_performance_comparison_has_no_quality_gain_requirement() -> None:
    row = {"gold_label": "dx", "predicted_label": "dx", "schema_valid": True, "peak_vram_gib": 6.0, "oom": False}
    baseline = evaluate_rows([{**row, "elapsed_seconds": 10.0}])
    candidate = evaluate_rows([{**row, "elapsed_seconds": 7.0}])
    result = compare_candidate(baseline, candidate, comparison_kind="performance", outputs_equivalent=True)
    assert result["eligible"]
    assert result["macro_f1_gain_pp"] == 0
    assert result["speedup_fraction"] == pytest.approx(0.3)
    assert not compare_candidate(baseline, candidate, comparison_kind="performance")["eligible"]

