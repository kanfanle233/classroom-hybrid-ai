from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from classroom_hybrid_ai.paper_evaluation import audit_splits, evaluate_research_run
from classroom_hybrid_ai.profiles import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs" / "action_profiles.yaml"


def _add_frame_manifest(manifest_path: Path, *, empty_frame: int = 4) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [{"frame_index": frame, "pts_seconds": float(frame), "confirmed_empty": frame == empty_frame, "review_status": "complete", "reviewers": ["human-a", "human-b"]} for frame in range(empty_frame + 1)]
    manifest["annotated_frames_uri"] = "annotated_frames.v1.jsonl"
    manifest["annotated_frames_sha256"] = _jsonl(manifest_path.parent / "annotated_frames.v1.jsonl", rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, rows


def test_confirmed_empty_frame_counts_false_positive_without_inventing_unlisted_negatives(tmp_path: Path) -> None:
    run, gold, _ = _fixture(tmp_path)
    prediction_path = run / "track_observations.jsonl"
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    predictions.extend({"frame_index": frame, "pts_seconds": float(frame), "student_id": "hallucinated", "bbox_px": [0, 0, 10, 10], "confidence": .8} for frame in [4, 99])
    _jsonl(prediction_path, predictions)
    legacy = evaluate_research_run(run, gold, "core8_v1")
    assert legacy["detection"]["FP"] == 0
    _add_frame_manifest(gold)
    result = evaluate_research_run(run, gold, "core8_v1")
    assert result["evaluation_status"] == "evaluated"
    assert result["detection"]["FP"] == 1
    assert result["detection"]["TP"] == 4
    assert result["tracking"]["IDFP"] == 1
    assert result["prediction_source_contract"]["confirmed_empty_frame_count"] == 1
    assert result["prediction_source_contract"]["detections_scored"] == 5


@pytest.mark.parametrize("defect", ["hash", "pending", "one_reviewer", "empty_conflict", "missing_box_frame", "pts", "duplicate", "non_boolean"])
def test_annotated_frame_integrity_and_review_fail_closed(tmp_path: Path, defect: str) -> None:
    run, gold, _ = _fixture(tmp_path)
    manifest, rows = _add_frame_manifest(gold)
    if defect == "pending":
        rows[-1]["review_status"] = "pending"
    elif defect == "one_reviewer":
        rows[-1]["reviewers"] = ["same", "same"]
    elif defect == "empty_conflict":
        rows[0]["confirmed_empty"] = True
    elif defect == "missing_box_frame":
        rows = rows[1:]
    elif defect == "pts":
        rows[0]["pts_seconds"] = .5
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "non_boolean":
        rows[-1]["confirmed_empty"] = 1
    digest = _jsonl(gold.parent / "annotated_frames.v1.jsonl", rows)
    manifest["annotated_frames_sha256"] = "a" * 64 if defect == "hash" else digest
    gold.write_text(json.dumps(manifest), encoding="utf-8")
    result = evaluate_research_run(run, gold, "core8_v1")
    assert result["evaluation_status"] == "invalid_gold"
    assert "annotated_frames" in result["reason"] or "boxes_outside" in result["reason"]


def test_all_empty_reviewed_video_can_score_spurious_detections(tmp_path: Path) -> None:
    run, gold, _ = _fixture(tmp_path)
    manifest, rows = _add_frame_manifest(gold)
    for row in rows:
        row["confirmed_empty"] = True
    manifest["boxes_sha256"] = _jsonl(gold.parent / "boxes.v1.jsonl", [])
    manifest["actions_sha256"] = _jsonl(gold.parent / "actions.v1.jsonl", [])
    manifest["annotated_frames_sha256"] = _jsonl(gold.parent / "annotated_frames.v1.jsonl", rows)
    gold.write_text(json.dumps(manifest), encoding="utf-8")
    result = evaluate_research_run(run, gold, "core8_v1")
    assert result["evaluation_status"] == "evaluated"
    assert result["detection"]["TP"] == 0
    assert result["detection"]["FP"] == 4
    assert result["prediction_source_contract"]["confirmed_empty_frame_count"] == 5


def test_pending_double_review_cannot_be_scored_as_gold(tmp_path: Path) -> None:
    run, gold, manifest = _fixture(tmp_path)
    manifest["double_review_status"] = "pending"
    gold.write_text(json.dumps(manifest), encoding="utf-8")
    result = evaluate_research_run(run, gold, "core8_v1")
    assert result["evaluation_status"] == "invalid_gold"
    assert result["reason"] == "double_review_not_complete"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return _sha(path)


def _fixture(tmp_path: Path, *, confidence: bool = True, source_name: str = "video.bin") -> tuple[Path, Path, dict[str, object]]:
    profile = load_profile(PROFILE_PATH)
    source = tmp_path / source_name
    source.write_bytes(b"small deterministic source")
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    boxes = gold_dir / "boxes.v1.jsonl"
    box_rows = [
        {"frame_index": 0, "pts_seconds": 0.0, "gold_person_id": "g1", "bbox_px": [0, 0, 10, 10]},
        {"frame_index": 1, "pts_seconds": 1.0, "gold_person_id": "g1", "bbox_px": [0, 0, 10, 10]},
        {"frame_index": 2, "pts_seconds": 2.0, "gold_person_id": "g1", "bbox_px": [0, 0, 10, 10]},
        {"frame_index": 3, "pts_seconds": 3.0, "gold_person_id": "g1", "bbox_px": [0, 0, 10, 10]},
    ]
    boxes_sha = _jsonl(boxes, box_rows)
    actions = gold_dir / "actions.v1.jsonl"
    actions_sha = _jsonl(actions, [
        {"gold_person_id": "g1", "start_pts_seconds": 0.0, "end_pts_seconds": 2.0, "action_id": "a_dx"},
        {"gold_person_id": "g1", "start_pts_seconds": 2.0, "end_pts_seconds": 4.0, "action_id": "a_dk"},
    ])
    manifest = {
        "schema_version": "classroom_hybrid_ai.gold_manifest.v1",
        "dataset_id": "tiny_research_fixture",
        "split_id": "test_tiny",
        "split": "test",
        "source_video_sha256": _sha(source),
        "source_lesson_id": "lesson-1",
        "classroom_id": "room-1",
        "capture_session_id": "session-1",
        "taxonomy_profile_id": profile.profile_id,
        "taxonomy_contract_sha256": profile.sha256,
        "boxes_uri": "boxes.v1.jsonl",
        "boxes_sha256": boxes_sha,
        "actions_uri": "actions.v1.jsonl",
        "actions_sha256": actions_sha,
        "annotation_protocol": "two reviewers, frozen v1",
        "double_review_status": "complete",
        "disagreement_resolution": "adjudicated by reviewer 1",
        "license_status": "granted",
        "permitted_uses": ["evaluation"],
    }
    manifest_path = gold_dir / "gold_manifest.v1.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_manifest = {
        "source": {"path": str(source), "sha256": _sha(source)},
        "profile": {"profile_id": profile.profile_id, "contract_sha256": profile.sha256},
        "duration_seconds": 4.0,
        "frame_count": 4,
        "command": ["fixture"],
    }
    (run_dir / "run_manifest.v1.json").write_text(json.dumps(run_manifest, sort_keys=True), encoding="utf-8")
    pred_rows = []
    for frame in range(4):
        row: dict[str, object] = {"frame_index": frame, "pts_seconds": float(frame), "student_id": "p1", "bbox_px": [0, 0, 10, 10]}
        if confidence:
            row["confidence"] = 1.0 - frame * 0.1
        pred_rows.append(row)
    _jsonl(run_dir / "track_observations.jsonl", pred_rows)
    _jsonl(run_dir / "action_observations.jsonl", [
        {"student_id": "p1", "start_pts_seconds": 0.0, "end_pts_seconds": 1.0, "action_id": "a_dx"},
        {"student_id": "p1", "start_pts_seconds": 1.0, "end_pts_seconds": 2.0, "action_id": "a_dk"},
        {"student_id": "p1", "start_pts_seconds": 2.0, "end_pts_seconds": 4.0, "action_id": "UNKNOWN", "status": "gray"},
    ])
    return run_dir, manifest_path, manifest


def test_perfect_detection_and_core8_matrix(tmp_path: Path) -> None:
    run_dir, manifest, _ = _fixture(tmp_path)
    result = evaluate_research_run(run_dir, manifest, "core8_v1")
    assert result["evaluation_status"] == "evaluated"
    assert result["detection"]["precision"] == pytest.approx(1.0)
    assert result["detection"]["recall"] == pytest.approx(1.0)
    assert result["detection"]["AP50"] == pytest.approx(1.0)
    assert result["tracking"]["IDF1"] == pytest.approx(1.0)
    matrix = result["actions"]["confusion_matrix_seconds"]
    assert len(matrix) == 8 and all(len(row) == 9 for row in matrix)
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(1.0)
    assert matrix[1][8] == pytest.approx(2.0)
    assert result["actions"]["UNKNOWN_ratio"] == pytest.approx(0.5)
    assert result["temporal"]["flip_count"] == 1
    assert result["temporal"]["flip_rate_per_observed_student_minute"] == pytest.approx(30.0)
    assert result["temporal"]["segmental_F1"]["@0.10"] == pytest.approx(0.5)


def test_id_switch_and_hota_alpha_receipt(tmp_path: Path) -> None:
    run_dir, manifest, _ = _fixture(tmp_path)
    (run_dir / "track_observations.jsonl").write_text(
        "".join(json.dumps({"frame_index": frame, "pts_seconds": float(frame), "student_id": "p1" if frame < 2 else "p2", "bbox_px": [0, 0, 10, 10], "confidence": 1.0}) + "\n" for frame in range(4)),
        encoding="utf-8",
    )
    result = evaluate_research_run(run_dir, manifest, "core8_v1")
    tracking = result["tracking"]
    assert tracking["ID_switch_count"] == 1
    assert tracking["IDF1"] == pytest.approx(0.5)
    assert len(tracking["HOTA_by_alpha"]) == 19
    assert all(set(row) == {"alpha", "DetA", "AssA", "HOTA"} for row in tracking["HOTA_by_alpha"])
    assert tracking["HOTA"] == pytest.approx(2 ** -0.5)


def test_detection_without_real_confidence_is_partial_and_omits_ap(tmp_path: Path) -> None:
    run_dir, manifest, _ = _fixture(tmp_path, confidence=False)
    result = evaluate_research_run(run_dir, manifest, "core8_v1")
    assert result["detection"]["status"] == "partial"
    assert "AP50" not in result["detection"]
    assert "mAP50" not in result["detection"]


def test_gold_hash_license_uri_overlap_and_source_profile_are_rejected(tmp_path: Path) -> None:
    run_dir, manifest_path, manifest = _fixture(tmp_path)
    manifest["boxes_sha256"] = "0" * 64
    bad_hash = manifest_path.parent / "bad_hash.json"
    bad_hash.write_text(json.dumps(manifest), encoding="utf-8")
    result = evaluate_research_run(run_dir, bad_hash, "core8_v1")
    assert result["evaluation_status"] == "invalid_gold"
    assert "boxes_sha256" in result["reason"]

    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    original["license_status"] = "unclear"
    bad_license = manifest_path.parent / "bad_license.json"
    bad_license.write_text(json.dumps(original), encoding="utf-8")
    assert evaluate_research_run(run_dir, bad_license, "core8_v1")["evaluation_status"] == "invalid_gold"

    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    original["actions_uri"] = "../actions.v1.jsonl"
    bad_uri = manifest_path.parent / "bad_uri.json"
    bad_uri.write_text(json.dumps(original), encoding="utf-8")
    assert evaluate_research_run(run_dir, bad_uri, "core8_v1")["evaluation_status"] == "invalid_gold"

    actions = manifest_path.parent / "actions.v1.jsonl"
    actions.write_text("".join([
        json.dumps({"gold_person_id": "g1", "start_pts_seconds": 0.0, "end_pts_seconds": 3.0, "action_id": "a_dx"}),
        "\n",
        json.dumps({"gold_person_id": "g1", "start_pts_seconds": 2.0, "end_pts_seconds": 4.0, "action_id": "a_dk"}),
        "\n",
    ]), encoding="utf-8")
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    original["actions_sha256"] = _sha(actions)
    overlap = manifest_path.parent / "overlap.json"
    overlap.write_text(json.dumps(original), encoding="utf-8")
    assert evaluate_research_run(run_dir, overlap, "core8_v1")["evaluation_status"] == "invalid_gold"


def test_no_gold_is_explicit_and_split_leakage_is_detected(tmp_path: Path) -> None:
    run_dir, _, manifest = _fixture(tmp_path)
    result = evaluate_research_run(run_dir)
    assert result["evaluation_status"] == "not_evaluated"
    assert result["reason"] == "gold_annotations_not_provided"
    forbidden = {"precision", "recall", "mAP50", "IDF1", "HOTA", "macro_F1", "confusion_matrix", "segmental_F1"}
    assert forbidden.isdisjoint(set(json.dumps(result)))

    second = dict(manifest)
    second["split"] = "train"
    second["split_id"] = "train_tiny"
    second["source_video_sha256"] = "1" * 64
    second["source_lesson_id"] = manifest["source_lesson_id"]
    second["capture_session_id"] = manifest["capture_session_id"]
    second["classroom_id"] = manifest["classroom_id"]
    audit = audit_splits([manifest, second])
    assert audit["status"] == "invalid"
    assert any("split_leakage_lesson" in error for error in audit["errors"])
    second.update({"split": "validation", "split_id": "validation_tiny"})
    audit = audit_splits([manifest, second])
    assert audit["status"] == "invalid"
    assert any("split_leakage_lesson" in error for error in audit["errors"])


def test_no_gold_still_reports_system_peaks_across_all_stages(tmp_path: Path) -> None:
    run_dir, _, _ = _fixture(tmp_path)
    receipts = run_dir / "stage_receipts"
    receipts.mkdir()
    (receipts / "S2.json").write_text(json.dumps({"stats": {"peak_vram_gib": 3.0, "peak_ram_gib": 8.0}}), encoding="utf-8")
    (receipts / "S5.json").write_text(json.dumps({"stats": {"peak_allocated_gib": 6.4, "peak_reserved_gib": 7.2, "peak_ram_gib": 10.0}}), encoding="utf-8")
    result = evaluate_research_run(run_dir)
    assert result["evaluation_status"] == "not_evaluated"
    assert result["system"]["peak_vram"] == 7.2
    assert result["system"]["peak_ram"] == 10.0
    assert "total_wall_time_seconds" not in result["system"]
    assert "total_wall_time_seconds" in result["system"]["unavailable_metrics"]
    assert "actions" not in result


def test_main_evaluation_split_failure_cannot_produce_quality_metrics(tmp_path: Path) -> None:
    run_dir, gold_path, gold = _fixture(tmp_path)
    leaked = {**gold, "split": "train", "split_id": "training", "source_video_sha256": "1" * 64}
    result = evaluate_research_run(run_dir, gold_path, "core8_v1", split_manifests=[gold, leaked], require_split_audit=True)
    assert result["evaluation_status"] == "invalid_split"
    assert result["paper_result_eligible"] is False
    assert not {"detection", "tracking", "actions", "temporal"}.intersection(result)
    assert evaluate_research_run(run_dir, gold_path, "core8_v1", require_split_audit=True)["evaluation_status"] == "invalid_split"
    valid = evaluate_research_run(run_dir, gold_path, "core8_v1", split_manifests=[gold], require_split_audit=True)
    assert valid["paper_result_eligible"] is True


def test_model_allocator_bytes_are_distinct_from_post_request_and_physical_usage(tmp_path: Path) -> None:
    run_dir, _, _ = _fixture(tmp_path)
    directory = run_dir / "diagnostics" / "performance"
    directory.mkdir(parents=True)
    report = {"timings_seconds": {"stage.wall_seconds": 12.0}, "model_requests": [{"memory": {"allocated_bytes": 2 * 2**30, "reserved_bytes": 3 * 2**30, "request_peak_allocated_bytes": 5 * 2**30, "max_reserved_bytes": 6 * 2**30}}], "resources": {"sampled_peak_device_total_gpu_mib": 7680}}
    (directory / "S5.json").write_text(json.dumps(report), encoding="utf-8")
    system = evaluate_research_run(run_dir)["system"]
    assert system["peak_vram"] == 6.0
    assert system["peak_allocated_gib"] == 5.0
    assert system["maximum_post_request_occupancy_gib"] == {"allocated": 2.0, "reserved": 3.0}
    assert system["sampled_peak_physical_vram_gib"] == 7.5
    assert system["stage_wall_time_seconds"]["S5"] == 12.0


def test_compact_system_collection_does_not_read_jsonl_or_s7_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from classroom_hybrid_ai.paper_evaluation import collect_system_metrics
    run_dir, _, _ = _fixture(tmp_path)
    (run_dir / "stage_events.v1.jsonl").write_text('{"stage_id":"S0"}\n', encoding="utf-8")
    (run_dir / "metrics.v1.json").write_text('{"system":{"peak_vram":99}}', encoding="utf-8")
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        if path.suffix == ".jsonl" or path.name in {"metrics.v1.json", "S7.json", "manifest.json"}:
            raise AssertionError(f"unexpected_system_summary_read:{path}")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    system = collect_system_metrics(run_dir, video_info={"duration_seconds": 4, "frame_count": 4}, stage_ids=tuple(f"S{index}" for index in range(7)))
    assert "peak_vram" not in system
    assert system["unavailable_metrics"]["stage_wall_time_seconds.S7"] == "stage_not_yet_committed"


def test_hota_assignment_runs_once_per_frame(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import classroom_hybrid_ai.paper_evaluation as scoring
    run_dir, gold_path, _ = _fixture(tmp_path)
    meta, _ = scoring._run_metadata(run_dir)
    profile = scoring._normalise_profile("core8_v1")
    gold = scoring._validate_gold(run_dir, gold_path, profile, meta)
    predictions = scoring._load_predictions(run_dir, set())
    original = scoring._hungarian_max
    assignments = []
    def counted(weights):
        assignments.append(weights)
        return original(weights)
    monkeypatch.setattr(scoring, "_hungarian_max", counted)
    result = scoring._hota_metrics(gold, predictions)
    assert len(assignments) == 4
    assert result["HOTA"] == pytest.approx(1.0)
    assert len(result["HOTA_by_alpha"]) == 19


def test_evaluation_is_byte_stable_for_same_inputs(tmp_path: Path) -> None:
    run_dir, manifest, _ = _fixture(tmp_path)
    first = json.dumps(evaluate_research_run(run_dir, manifest, "core8_v1"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    second = json.dumps(evaluate_research_run(run_dir, manifest, "core8_v1"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert first == second


def test_prediction_source_priority_does_not_double_count_historical_copies(tmp_path: Path) -> None:
    run_dir, manifest, _ = _fixture(tmp_path)
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir()
    # These are intentional historical duplicates.  The evaluator must select
    # one source per role instead of rglob-ing both copies.
    (artifact_dir / "track_observations.jsonl").write_text((run_dir / "track_observations.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    (artifact_dir / "student_actions.v1.jsonl").write_text((run_dir / "action_observations.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    result = evaluate_research_run(run_dir, manifest, "core8_v1")
    assert result["detection"]["TP"] == 4
    assert result["detection"]["FP"] == 0
    assert result["observed_run_summary"]["predicted_detection_count"] == 4
    assert result["prediction_source_contract"]["mode"] == "split_explicit_priority"
    assert set(result["prediction_source_contract"]["roles"]) == {"detections", "actions"}


def test_unannotated_sparse_frames_are_not_treated_as_empty_gold(tmp_path: Path) -> None:
    run_dir, manifest, _ = _fixture(tmp_path)
    with (run_dir / "track_observations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "frame_index": 99,
            "pts_seconds": 99.0,
            "student_id": "outside-gold",
            "bbox_px": [50, 50, 60, 60],
            "confidence": 0.9,
        }) + "\n")
    result = evaluate_research_run(run_dir, manifest, "core8_v1")
    assert result["detection"]["FP"] == 0
    assert result["prediction_source_contract"]["detections_total"] == 5
    assert result["prediction_source_contract"]["detections_scored"] == 4
    assert result["prediction_source_contract"]["evaluation_frame_count"] == 4


def test_fragment_count_uses_sparse_gold_sequence_not_frame_number_gap(tmp_path: Path) -> None:
    run_dir, manifest_path, _ = _fixture(tmp_path)
    boxes = manifest_path.parent / "boxes.v1.jsonl"
    sparse = [
        {"frame_index": 0, "pts_seconds": 0.0, "gold_person_id": "g1", "bbox_px": [0, 0, 10, 10]},
        {"frame_index": 10, "pts_seconds": 1.0, "gold_person_id": "g1", "bbox_px": [0, 0, 10, 10]},
        {"frame_index": 20, "pts_seconds": 2.0, "gold_person_id": "g1", "bbox_px": [0, 0, 10, 10]},
    ]
    boxes_sha = _jsonl(boxes, sparse)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["boxes_sha256"] = boxes_sha
    sparse_manifest = manifest_path.parent / "sparse_manifest.json"
    sparse_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "track_observations.jsonl").write_text(
        "".join(json.dumps({"frame_index": frame, "pts_seconds": frame / 10.0, "student_id": "p1", "bbox_px": [0, 0, 10, 10], "confidence": 1.0}) + "\n" for frame in (0, 10, 20)),
        encoding="utf-8",
    )
    full = evaluate_research_run(run_dir, sparse_manifest, "core8_v1")
    assert full["tracking"]["fragment_count"] == 0
    (run_dir / "track_observations.jsonl").write_text(
        "".join(json.dumps({"frame_index": frame, "pts_seconds": frame / 10.0, "student_id": "p1", "bbox_px": [0, 0, 10, 10], "confidence": 1.0}) + "\n" for frame in (0, 20)),
        encoding="utf-8",
    )
    recovered = evaluate_research_run(run_dir, sparse_manifest, "core8_v1")
    assert recovered["tracking"]["fragment_count"] == 1


def test_identity_crossing_uses_global_alignment_hota(tmp_path: Path) -> None:
    run_dir, manifest_path, _ = _fixture(tmp_path)
    boxes = manifest_path.parent / "boxes.v1.jsonl"
    boxes_sha = _jsonl(boxes, [
        {"frame_index": 0, "pts_seconds": 0.0, "gold_person_id": "g1", "bbox_px": [0, 0, 10, 10]},
        {"frame_index": 0, "pts_seconds": 0.0, "gold_person_id": "g2", "bbox_px": [20, 0, 30, 10]},
        {"frame_index": 1, "pts_seconds": 1.0, "gold_person_id": "g1", "bbox_px": [8, 0, 18, 10]},
        {"frame_index": 1, "pts_seconds": 1.0, "gold_person_id": "g2", "bbox_px": [12, 0, 22, 10]},
    ])
    actions = manifest_path.parent / "actions.v1.jsonl"
    actions_sha = _jsonl(actions, [
        {"gold_person_id": "g1", "start_pts_seconds": 0.0, "end_pts_seconds": 2.0, "action_id": "a_dx"},
        {"gold_person_id": "g2", "start_pts_seconds": 0.0, "end_pts_seconds": 2.0, "action_id": "a_dk"},
    ])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["boxes_sha256"], manifest["actions_sha256"] = boxes_sha, actions_sha
    crossing_manifest = manifest_path.parent / "crossing_manifest.json"
    crossing_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "track_observations.jsonl").write_text("\n".join([
        json.dumps({"frame_index": 0, "pts_seconds": 0.0, "student_id": "p1", "bbox_px": [0, 0, 10, 10], "confidence": 1.0}),
        json.dumps({"frame_index": 0, "pts_seconds": 0.0, "student_id": "p2", "bbox_px": [20, 0, 30, 10], "confidence": 1.0}),
        json.dumps({"frame_index": 1, "pts_seconds": 1.0, "student_id": "p1", "bbox_px": [12, 0, 22, 10], "confidence": 1.0}),
        json.dumps({"frame_index": 1, "pts_seconds": 1.0, "student_id": "p2", "bbox_px": [8, 0, 18, 10], "confidence": 1.0}),
    ]) + "\n", encoding="utf-8")
    result = evaluate_research_run(run_dir, crossing_manifest, "core8_v1")
    assert result["tracking"]["IDF1"] == pytest.approx(0.5)
    assert result["tracking"]["ID_switch_count"] == 2
    # The full-score Hungarian intentionally keeps the identity-consistent
    # overlap at low alpha and switches to the exact crossing at higher alpha;
    # this is the TrackEval global-alignment behavior, not raw-IoU matching.
    assert result["tracking"]["HOTA"] == pytest.approx(0.6140350877192986)
    assert result["tracking"]["reference"].endswith("trackeval/metrics/hota.py")


def test_gray_ratio_is_unavailable_without_recorded_video_duration(tmp_path: Path) -> None:
    run_dir, manifest, _ = _fixture(tmp_path)
    run_manifest = json.loads((run_dir / "run_manifest.v1.json").read_text(encoding="utf-8"))
    run_manifest.pop("duration_seconds", None)
    (run_dir / "run_manifest.v1.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    temporal = evaluate_research_run(run_dir, manifest, "core8_v1")["temporal"]
    assert "gray_time_ratio" not in temporal
    assert "unknown_student_seconds" not in temporal
    assert "complete_student_video_seconds" not in temporal
    assert temporal["unavailable_metrics"]["gray_time_ratio"] == "video_duration_not_recorded"
    assert temporal["unavailable_metrics"]["unknown_student_seconds"] == "video_duration_not_recorded"
    assert temporal["unavailable_metrics"]["complete_student_video_seconds"] == "video_duration_not_recorded"
