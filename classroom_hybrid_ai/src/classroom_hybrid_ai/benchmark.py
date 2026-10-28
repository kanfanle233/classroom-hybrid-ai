"""Read-only frozen benchmark manifest and post-hoc eight-label scoring.

This module intentionally has no Qwen prompt and no inference code.  The
legacy labels are read only after a prediction has been emitted, solely to
report a blinded comparison.  They cannot affect discovery, vocabulary
formation, model prompts, thresholds, or tracking.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import canonical_sha256
from .storage import atomic_write_json, iter_jsonl


EIGHT_LABELS = ("dx", "dk", "tt", "zt", "js", "zl", "xt", "jz")


def freeze_manifest(source: Path, destination: Path) -> dict[str, Any]:
    """Freeze the existing 8,884-image local manifest by canonical hash."""

    rows = list(iter_jsonl(source))
    if len(rows) != 8884:
        raise ValueError(f"benchmark_requires_8884_images:got_{len(rows)}")
    required = {"sample_id", "image_path", "image_sha256", "annotation_sha256", "class_counts"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("benchmark_source_manifest_contract_invalid")
    normalized = [
        {
            "sample_id": str(row["sample_id"]),
            "image_path": str(row["image_path"]),
            "image_sha256": str(row["image_sha256"]),
            "annotation_sha256": str(row["annotation_sha256"]),
            "class_counts": dict(row["class_counts"]),
            "split": str(row.get("split", "")),
            "source_group": row.get("source_group"),
        }
        for row in rows
    ]
    payload = {
        "schema_version": "classroom_hybrid_ai.blind_benchmark_manifest.v1",
        "source_manifest": str(source),
        "sample_count": len(normalized),
        "labels": list(EIGHT_LABELS),
        "samples": normalized,
    }
    payload["sha256"] = canonical_sha256(payload)
    atomic_write_json(destination, payload)
    return payload


def score_predictions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score post-hoc rows; ``UNKNOWN`` is never counted as an eighth action."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("benchmark_predictions_empty")
    for row in materialized:
        if row.get("gold_label") not in EIGHT_LABELS:
            raise ValueError("benchmark_gold_label_invalid")
        if row.get("predicted_label") not in {*EIGHT_LABELS, "UNKNOWN", None}:
            raise ValueError("benchmark_predicted_label_invalid")
    per_class: dict[str, float] = {}
    for label in EIGHT_LABELS:
        tp = sum(row["gold_label"] == label and row.get("predicted_label") == label for row in materialized)
        fp = sum(row["gold_label"] != label and row.get("predicted_label") == label for row in materialized)
        fn = sum(row["gold_label"] == label and row.get("predicted_label") != label for row in materialized)
        per_class[label] = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    known = [row for row in materialized if row.get("predicted_label") in EIGHT_LABELS]
    correct = sum(row["gold_label"] == row.get("predicted_label") for row in materialized)
    correct_known = sum(row["gold_label"] == row.get("predicted_label") for row in known)
    return {
        "schema_version": "classroom_hybrid_ai.blind_benchmark_report.v1",
        "sample_count": len(materialized),
        "macro_f1": sum(per_class.values()) / len(per_class),
        "per_class_f1": per_class,
        "unknown_rate": (len(materialized) - len(known)) / len(materialized),
        "covered_sample_accuracy": None if not known else correct_known / len(known),
        "overall_accuracy": correct / len(materialized),
        "unavailable_metrics": {"synonym_label_fragmentation_rate": "not_applicable_to_fixed_eight_class_taxonomy"},
        "qwen_calls_per_hour": _mean_number(materialized, "qwen_calls_per_hour"),
        "course_action_flip_rate": _mean_number(materialized, "course_action_flip_rate"),
    }


def _mean_number(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None and math.isfinite(float(row[field]))]
    return None if not values else sum(values) / len(values)


__all__ = ["EIGHT_LABELS", "freeze_manifest", "score_predictions"]
