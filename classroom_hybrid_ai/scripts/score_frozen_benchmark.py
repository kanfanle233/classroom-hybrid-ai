"""Score post-hoc predictions against the frozen classroom benchmark.

This command is deliberately outside the online package path.  It reads the
frozen manifest and annotation files only after a prediction JSONL has been
written; it is never imported by the YOLO/LocateAnything/Qwen pipeline and
cannot put the eight legacy labels into a prompt.

Prediction rows use one row per annotated student box::

    {"sample_id": "sample-...", "box_index": 0,
     "predicted_label": "UNKNOWN", "schema_valid": true}

``gold_label`` is taken from the hashed annotation, never trusted from the
prediction file.  Missing boxes are materialized as ``UNKNOWN`` so coverage
cannot be hidden by submitting only easy predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classroom_hybrid_ai.benchmark import EIGHT_LABELS, score_predictions  # noqa: E402
from classroom_hybrid_ai.hashing import canonical_sha256, sha256_file  # noqa: E402
from classroom_hybrid_ai.storage import read_jsonl  # noqa: E402


class FrozenBenchmarkError(ValueError):
    """Raised when a blind manifest or prediction contract is invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenBenchmarkError(f"json_read_failed:{path}:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise FrozenBenchmarkError(f"json_root_must_be_object:{path}")
    return value


def load_frozen_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and verify the immutable 8,884-image manifest."""

    payload = _load_json(path.resolve())
    if payload.get("schema_version") != "classroom_hybrid_ai.blind_benchmark_manifest.v1":
        raise FrozenBenchmarkError("blind_manifest_schema_invalid")
    samples = payload.get("samples")
    if not isinstance(samples, list) or payload.get("sample_count") != len(samples):
        raise FrozenBenchmarkError("blind_manifest_sample_count_invalid")
    if payload.get("labels") != list(EIGHT_LABELS):
        raise FrozenBenchmarkError("blind_manifest_label_order_invalid")
    recorded_hash = payload.get("sha256")
    without_hash = deepcopy(payload)
    without_hash.pop("sha256", None)
    if not isinstance(recorded_hash, str) or canonical_sha256(without_hash) != recorded_hash:
        raise FrozenBenchmarkError("blind_manifest_hash_mismatch")

    by_id: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise FrozenBenchmarkError("blind_manifest_sample_not_object")
        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id or sample_id in by_id:
            raise FrozenBenchmarkError("blind_manifest_sample_id_invalid_or_duplicate")
        class_counts = sample.get("class_counts")
        if not isinstance(class_counts, dict) or any(
            label not in EIGHT_LABELS
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for label, count in class_counts.items()
        ):
            raise FrozenBenchmarkError(f"blind_manifest_class_counts_invalid:{sample_id}")
        if not class_counts or sum(class_counts.values()) < 1:
            raise FrozenBenchmarkError(f"blind_manifest_empty_annotation:{sample_id}")
        by_id[sample_id] = sample
    return payload, by_id


def _resolve_repository_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _annotation_path(sample: Mapping[str, Any], repository_root: Path) -> Path:
    raw = sample.get("annotation_path")
    if not isinstance(raw, str) or not raw:
        image_path = sample.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise FrozenBenchmarkError(f"sample_image_path_missing:{sample.get('sample_id')}")
        raw = str(Path(image_path).with_suffix(".json"))
    return _resolve_repository_path(raw, repository_root)


def _annotation_labels(path: Path, sample_id: str) -> list[str]:
    payload = _load_json(path)
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise FrozenBenchmarkError(f"annotation_labels_missing:{sample_id}")
    output: list[str] = []
    for index, item in enumerate(labels):
        if not isinstance(item, dict) or item.get("name") not in EIGHT_LABELS:
            raise FrozenBenchmarkError(f"annotation_label_invalid:{sample_id}:{index}")
        output.append(str(item["name"]))
    return output


def _normalise_prediction_label(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    label = str(value).strip()
    if not label or label.upper() in {"UNKNOWN", "UNOBSERVED", "NONE"}:
        return "UNKNOWN"
    label = label.lower()
    if label not in EIGHT_LABELS:
        raise FrozenBenchmarkError(f"prediction_label_invalid:{value}")
    return label


def _verify_sample_files(
    sample: Mapping[str, Any], *, repository_root: Path, annotation_path: Path
) -> None:
    image_path = sample.get("image_path")
    expected_image = sample.get("image_sha256")
    expected_annotation = sample.get("annotation_sha256")
    if not isinstance(image_path, str) or not isinstance(expected_image, str):
        raise FrozenBenchmarkError(f"sample_image_hash_contract_invalid:{sample.get('sample_id')}")
    if not isinstance(expected_annotation, str):
        raise FrozenBenchmarkError(f"sample_annotation_hash_contract_invalid:{sample.get('sample_id')}")
    image = _resolve_repository_path(image_path, repository_root)
    if not image.is_file() or not annotation_path.is_file():
        raise FrozenBenchmarkError(f"benchmark_file_missing:{sample.get('sample_id')}")
    if sha256_file(image) != expected_image:
        raise FrozenBenchmarkError(f"image_hash_mismatch:{sample.get('sample_id')}")
    if sha256_file(annotation_path) != expected_annotation:
        raise FrozenBenchmarkError(f"annotation_hash_mismatch:{sample.get('sample_id')}")


def materialize_prediction_rows(
    manifest_path: Path,
    predictions: Iterable[Mapping[str, Any]],
    *,
    split: str = "test",
    repository_root: Path | None = None,
    verify_files: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join predictions to annotation boxes and materialize missing boxes.

    ``split=test`` is the safe default.  ``split=all`` is explicit so a caller
    cannot accidentally report training images as a held-out result.
    """

    manifest, samples = load_frozen_manifest(manifest_path)
    if split != "all" and split not in {"train", "val", "test", "quarantine"}:
        raise FrozenBenchmarkError(f"benchmark_split_invalid:{split}")
    repository_root = (repository_root or PROJECT_ROOT.parent).resolve()
    selected = {
        sample_id: sample
        for sample_id, sample in samples.items()
        if split == "all" or str(sample.get("split", "")) == split
    }
    if not selected:
        raise FrozenBenchmarkError(f"benchmark_split_empty:{split}")

    submitted: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in predictions:
        if not isinstance(raw, Mapping):
            raise FrozenBenchmarkError("prediction_row_not_object")
        sample_id = str(raw.get("sample_id", "")).strip()
        if sample_id not in selected:
            raise FrozenBenchmarkError(f"prediction_sample_not_in_split:{sample_id}")
        index = raw.get("box_index", raw.get("instance_index"))
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise FrozenBenchmarkError(f"prediction_box_index_invalid:{sample_id}")
        key = (sample_id, index)
        if key in submitted:
            raise FrozenBenchmarkError(f"prediction_duplicate_box:{sample_id}:{index}")
        submitted[key] = dict(raw)

    output: list[dict[str, Any]] = []
    missing = 0
    expected_instances = 0
    submitted_instances = 0
    labels_by_sample: dict[str, list[str]] = {}
    for sample_id, sample in selected.items():
        annotation_path = _annotation_path(sample, repository_root)
        if verify_files:
            _verify_sample_files(sample, repository_root=repository_root, annotation_path=annotation_path)
        labels = _annotation_labels(annotation_path, sample_id)
        labels_by_sample[sample_id] = labels
        expected_counts = Counter({str(key): int(value) for key, value in sample["class_counts"].items()})
        if Counter(labels) != expected_counts:
            raise FrozenBenchmarkError(f"annotation_class_counts_mismatch:{sample_id}")
        expected_instances += len(labels)
        for box_index, gold_label in enumerate(labels):
            raw = submitted.get((sample_id, box_index))
            if raw is None:
                missing += 1
                merged: dict[str, Any] = {
                    "sample_id": sample_id,
                    "box_index": box_index,
                    "predicted_label": "UNKNOWN",
                    "schema_valid": False,
                    "missing_prediction": True,
                }
            else:
                submitted_instances += 1
                merged = dict(raw)
                merged["sample_id"] = sample_id
                merged["box_index"] = box_index
                merged["predicted_label"] = _normalise_prediction_label(raw.get("predicted_label"))
                merged["schema_valid"] = bool(raw.get("schema_valid", False))
                merged["missing_prediction"] = False
            merged["gold_label"] = gold_label
            merged["split"] = str(sample.get("split", ""))
            merged["source_group"] = sample.get("source_group")
            output.append(merged)

    extra = [
        key for key in submitted
        if key[0] in selected and key[1] >= len(labels_by_sample[key[0]])
    ]
    if extra:
        sample_id, box_index = extra[0]
        raise FrozenBenchmarkError(f"prediction_box_index_out_of_range:{sample_id}:{box_index}")
    if any(key[0] not in selected for key in submitted):
        raise FrozenBenchmarkError("prediction_contains_unselected_sample")

    metadata = {
        "schema_version": "classroom_hybrid_ai.blind_benchmark_join.v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest["sha256"],
        "split": split,
        "image_count": len(selected),
        "expected_instances": expected_instances,
        "submitted_instances": submitted_instances,
        "missing_predictions": missing,
        "coverage_rate": submitted_instances / expected_instances if expected_instances else 0.0,
        "files_verified": verify_files,
    }
    return output, metadata


def score_frozen_predictions(
    manifest_path: Path,
    predictions: Iterable[Mapping[str, Any]],
    *,
    split: str = "test",
    repository_root: Path | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    rows, metadata = materialize_prediction_rows(
        manifest_path,
        predictions,
        split=split,
        repository_root=repository_root,
        verify_files=verify_files,
    )
    report = score_predictions(rows)
    report["benchmark"] = metadata
    report["strict_json_rate"] = sum(bool(row.get("schema_valid")) for row in rows) / len(rows)
    submitted = [row for row in rows if not row.get("missing_prediction")]
    if submitted:
        submitted_report = score_predictions(submitted)
        columns = [*EIGHT_LABELS, "UNKNOWN"]
        submitted_report.update({
            "scope": "submitted_predictions_only",
            "known_predictions": sum(row.get("predicted_label") in EIGHT_LABELS for row in submitted),
            "strict_json_rate": sum(bool(row.get("schema_valid")) for row in submitted) / len(submitted),
            "gold_support": {
                label: sum(row["gold_label"] == label for row in submitted)
                for label in EIGHT_LABELS
            },
            "predicted_counts": {
                label: sum(row.get("predicted_label") == label for row in submitted)
                for label in columns
            },
            "confusion_matrix": {
                gold: {
                    predicted: sum(
                        row["gold_label"] == gold
                        and (row.get("predicted_label") or "UNKNOWN") == predicted
                        for row in submitted
                    )
                    for predicted in columns
                }
                for gold in EIGHT_LABELS
            },
        })
        report["submitted_only"] = submitted_report
    else:
        report["submitted_only"] = {
            "scope": "submitted_predictions_only",
            "sample_count": 0,
            "reason": "no_submitted_predictions",
        }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score post-hoc predictions against the frozen classroom benchmark")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("test", "val", "train", "quarantine", "all"), default="test")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--verify-files", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = score_frozen_predictions(
        args.manifest.resolve(),
        read_jsonl(args.predictions.resolve()),
        split=args.split,
        repository_root=args.repository_root,
        verify_files=args.verify_files,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FrozenBenchmarkError",
    "load_frozen_manifest",
    "materialize_prediction_rows",
    "score_frozen_predictions",
]
