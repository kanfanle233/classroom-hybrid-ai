from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from classroom_hybrid_ai.hashing import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_frozen_benchmark.py"
SPEC = importlib.util.spec_from_file_location("score_frozen_benchmark_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_script)


def _manifest(tmp_path: Path) -> Path:
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps({
            "labels": [
                {"name": "dx", "x1": 0, "y1": 0, "x2": 10, "y2": 10},
                {"name": "tt", "x1": 20, "y1": 20, "x2": 30, "y2": 30},
            ]
        }),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "classroom_hybrid_ai.blind_benchmark_manifest.v1",
        "source_manifest": "synthetic",
        "sample_count": 1,
        "labels": ["dx", "dk", "tt", "zt", "js", "zl", "xt", "jz"],
        "samples": [{
            "sample_id": "sample-1",
            "image_path": "image.jpg",
            "annotation_path": "annotation.json",
            "image_sha256": "unused",
            "annotation_sha256": "unused",
            "class_counts": {"dx": 1, "tt": 1},
            "split": "test",
            "source_group": "lesson-1",
        }],
    }
    payload["sha256"] = canonical_sha256(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_frozen_join_reads_gold_from_annotation_and_materializes_missing_boxes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    rows, metadata = benchmark_script.materialize_prediction_rows(
        manifest,
        [{"sample_id": "sample-1", "box_index": 0, "predicted_label": "dx", "schema_valid": True}],
        split="test",
        repository_root=tmp_path,
    )
    assert [row["gold_label"] for row in rows] == ["dx", "tt"]
    assert [row["predicted_label"] for row in rows] == ["dx", "UNKNOWN"]
    assert metadata["expected_instances"] == 2
    assert metadata["submitted_instances"] == 1
    assert metadata["missing_predictions"] == 1
    assert metadata["coverage_rate"] == 0.5


def test_frozen_join_rejects_duplicate_or_out_of_range_boxes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    duplicate = [
        {"sample_id": "sample-1", "box_index": 0, "predicted_label": "dx"},
        {"sample_id": "sample-1", "box_index": 0, "predicted_label": "dx"},
    ]
    with pytest.raises(benchmark_script.FrozenBenchmarkError, match="duplicate"):
        benchmark_script.materialize_prediction_rows(
            manifest, duplicate, split="test", repository_root=tmp_path
        )
    with pytest.raises(benchmark_script.FrozenBenchmarkError, match="out_of_range"):
        benchmark_script.materialize_prediction_rows(
            manifest,
            [{"sample_id": "sample-1", "box_index": 2, "predicted_label": "dx"}],
            split="test",
            repository_root=tmp_path,
        )


def test_frozen_join_does_not_accept_a_prediction_from_another_split(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(benchmark_script.FrozenBenchmarkError, match="not_in_split"):
        benchmark_script.materialize_prediction_rows(
            manifest,
            [{"sample_id": "sample-not-test", "box_index": 0, "predicted_label": "dx"}],
            split="test",
            repository_root=tmp_path,
        )


def test_partial_report_separates_submitted_quality_from_missing_coverage(tmp_path: Path) -> None:
    report = benchmark_script.score_frozen_predictions(
        _manifest(tmp_path),
        [
            {"sample_id": "sample-1", "box_index": 0, "predicted_label": "dx", "schema_valid": True},
        ],
        split="test",
        repository_root=tmp_path,
    )
    assert report["benchmark"]["coverage_rate"] == 0.5
    assert report["submitted_only"]["sample_count"] == 1
    assert report["submitted_only"]["overall_accuracy"] == 1.0
    assert report["submitted_only"]["unknown_rate"] == 0.0
    assert report["submitted_only"]["strict_json_rate"] == 1.0
    assert report["submitted_only"]["confusion_matrix"]["dx"]["dx"] == 1
