"""Candidate comparison on a shared, lesson-disjoint evidence set."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .benchmark import score_predictions


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    sample_count: int
    strict_json_rate: float | None
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    unknown_rate: float
    mean_seconds_per_event: float | None
    peak_vram_gib: float | None
    oom_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "strict_json_rate": self.strict_json_rate,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class_f1": self.per_class_f1,
            "unknown_rate": self.unknown_rate,
            "mean_seconds_per_event": self.mean_seconds_per_event,
            "peak_vram_gib": self.peak_vram_gib,
            "oom_count": self.oom_count,
        }


def evaluate_rows(rows: Iterable[dict[str, Any]]) -> CandidateMetrics:
    materialized = list(rows)
    if not materialized:
        raise ValueError("evaluation_rows_empty")
    normalized = [{**row, "predicted_label": "UNKNOWN" if row.get("predicted_label") in {None, "unknown", "UNKNOWN", ""} else row["predicted_label"]} for row in materialized]
    scored = score_predictions(normalized)

    def numbers(key: str) -> list[float]:
        result = []
        for row in materialized:
            value = row.get(key)
            if value is None:
                continue
            number = float(value)
            if math.isfinite(number) and number >= 0:
                result.append(number)
        return result

    elapsed = numbers("elapsed_seconds")
    peaks = numbers("peak_vram_gib")
    schema = [bool(row["schema_valid"]) for row in materialized if row.get("schema_valid") is not None]
    oom = [bool(row["oom"]) for row in materialized if row.get("oom") is not None]
    return CandidateMetrics(
        sample_count=len(materialized),
        strict_json_rate=sum(schema) / len(schema) if len(schema) == len(materialized) else None,
        accuracy=scored["overall_accuracy"],
        macro_f1=scored["macro_f1"],
        per_class_f1=scored["per_class_f1"],
        unknown_rate=scored["unknown_rate"],
        mean_seconds_per_event=sum(elapsed) / len(elapsed) if len(elapsed) == len(materialized) else None,
        peak_vram_gib=max(peaks) if len(peaks) == len(materialized) else None,
        oom_count=sum(oom) if len(oom) == len(materialized) else None,
    )


def compare_candidate(
    baseline: CandidateMetrics,
    candidate: CandidateMetrics,
    *,
    minimum_macro_f1_gain_pp: float = 3.0,
    maximum_any_class_drop_pp: float = 2.0,
    minimum_strict_json_rate: float = 0.99,
    maximum_peak_vram_gib: float = 7.4,
    comparison_kind: str = "quality",
    outputs_equivalent: bool | None = None,
    minimum_speedup_fraction: float = 0.20,
) -> dict[str, Any]:
    shared = sorted(set(baseline.per_class_f1) & set(candidate.per_class_f1))
    drops = {
        label: (baseline.per_class_f1[label] - candidate.per_class_f1[label]) * 100
        for label in shared
    }
    gain_pp = (candidate.macro_f1 - baseline.macro_f1) * 100
    blockers: list[str] = []
    if comparison_kind not in {"quality", "performance"}:
        raise ValueError("comparison_kind_must_be_quality_or_performance")
    if comparison_kind == "quality" and gain_pp < minimum_macro_f1_gain_pp:
        blockers.append("macro_f1_gain_below_gate")
    speedup = None
    if baseline.mean_seconds_per_event is not None and baseline.mean_seconds_per_event > 0 and candidate.mean_seconds_per_event is not None:
        speedup = 1.0 - candidate.mean_seconds_per_event / baseline.mean_seconds_per_event
    if comparison_kind == "performance":
        if outputs_equivalent is not True:
            blockers.append("output_equivalence_not_verified")
        if speedup is None:
            blockers.append("elapsed_seconds_not_measured")
        elif speedup < minimum_speedup_fraction:
            blockers.append("speedup_below_gate")
    if drops and max(drops.values()) > maximum_any_class_drop_pp:
        blockers.append("per_class_f1_drop_exceeds_gate")
    if candidate.strict_json_rate is None:
        blockers.append("strict_json_rate_not_measured")
    elif candidate.strict_json_rate < minimum_strict_json_rate:
        blockers.append("strict_json_rate_below_gate")
    if candidate.peak_vram_gib is None:
        blockers.append("peak_vram_not_measured")
    elif candidate.peak_vram_gib > maximum_peak_vram_gib:
        blockers.append("peak_vram_exceeds_gate")
    if candidate.oom_count is None:
        blockers.append("oom_status_not_measured")
    elif candidate.oom_count:
        blockers.append("oom_observed")
    return {
        "eligible": not blockers,
        "comparison_kind": comparison_kind,
        "speedup_fraction": speedup,
        "blockers": blockers,
        "macro_f1_gain_pp": gain_pp,
        "per_class_f1_drop_pp": drops,
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
    }
