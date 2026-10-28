from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from classroom_hybrid_ai.quality_diagnostics import diagnose_unknown, prepare_annotation_package


def _rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _response(root: Path, iteration: int, event: str, **overrides) -> None:
    path = root / "responses" / "stability" / f"{iteration:04d}" / f"{event}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidate_event_id": event, "schema_valid": True, "decision": "uncertain", "behavior": "UNKNOWN", "rationale": "证据不足", **overrides}), encoding="utf-8")


def test_deduplication_does_not_invent_visual_causes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for index in range(100):
        _response(run, index, f"event-{index % 30}")
    report = diagnose_unknown(run, output_path=tmp_path / "result.json")
    assert report["summary"]["attempts"] == 100
    assert report["summary"]["unique_candidates"] == 30
    assert report["summary"]["diagnostic_reason_candidate_counts"] == {"model_abstention": 30}
    assert report["summary"]["model_claimed_reason_candidate_counts"] == {}
    assert all(row["visual_cause_status"] == "unverified" for row in report["candidates"])
    assert report["evaluation_status"] == "not_evaluated"
    assert not {"accuracy", "macro_F1"}.intersection(report)


def test_model_claim_human_review_and_geometries_are_separate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _response(run, 0, "e1", rationale="手部遮挡")
    image = tmp_path / "subject.jpg"
    Image.new("RGB", (10, 8)).save(image)
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    _rows(run / "evidence/index.jsonl", [{"event_id": "e1", "evidence": [{"path": str(image), "sha256": sha, "bbox_xyxy": [0, 0, 10, 8], "kind": "subject_crop"}], "qwen_composite": {"path": str(image), "sha256": "a" * 64}}])
    reviews = tmp_path / "review.jsonl"
    _rows(reviews, [{"candidate_event_id": "e1", "reason_categories": ["missing_critical_parts"], "reviewer_id": "r1", "evidence_note": "inspected supplied image"}])
    report = diagnose_unknown(run, output_path=tmp_path / "result.json", review_path=reviews)
    row = report["candidates"][0]
    assert row["diagnostic_reasons"] == ["model_abstention"]
    assert row["model_claimed_reasons"] == ["missing_critical_parts", "target_small_or_occluded"]
    assert row["human_reviewed_reasons"] == ["missing_critical_parts"]
    assert row["evidence"][0]["image"]["integrity"] == "matched"
    assert row["actual_composite"]["integrity"] == "mismatch_or_missing"


def test_budget_identity_and_parse_failure_preserve_distinct_reasons(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _rows(run / "artifacts/candidates.unselected.jsonl", [{"event_id": "budget", "canonical_reason": "budget_exhausted"}, {"event_id": "identity", "metadata": {"id_uncertain": True}}, {"event_id": "not_called", "canonical_reason": "insufficient_evidence"}])
    _response(run, 0, "parse", schema_valid=False)
    report = diagnose_unknown(run, output_path=tmp_path / "result.json")
    reasons = {row["candidate_event_id"]: row["diagnostic_reasons"] for row in report["candidates"]}
    assert reasons == {"budget": ["budget_uncovered"], "identity": ["identity_uncertain"], "not_called": [], "parse": ["parse_failure"]}


def test_malformed_response_retained_and_output_not_overwritten(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _response(run, 0, "bad")
    (run / "responses/stability/0000/bad.json").write_text("{bad", encoding="utf-8")
    output = tmp_path / "result.json"
    report = diagnose_unknown(run, output_path=output)
    assert len(report["read_errors"]) == 1
    assert report["candidates"][0]["diagnostic_reasons"] == ["parse_failure"]
    with pytest.raises(FileExistsError):
        diagnose_unknown(run, output_path=output)


def test_normal_s5_uses_only_active_cache_and_keeps_transport_failure_separate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    active = run / "artifacts/s5_active_response_set.v1.json"
    active.parent.mkdir(parents=True)
    active.write_text(json.dumps({"response_prefix": "responses/classification/current"}))
    for directory in ["current", "stale"]:
        path = run / "responses/classification" / directory
        path.mkdir(parents=True)
        (path / "e1.json").write_text(json.dumps({"candidate_event_id": "e1", "status": "error", "error": "connection_failed"}))
    _rows(run / "reviews.classification.jsonl", [{"candidate_event_id": "e1", "schema_valid": False, "reviewer": "pipeline_error", "behavior": None}])
    result = diagnose_unknown(run, output_path=tmp_path / "result.json")
    assert result["summary"]["attempts"] == 1
    assert result["candidates"][0]["diagnostic_reasons"] == []
    assert len(result["candidates"][0]["request_failures"]) == 1
    assert "/stale/" not in json.dumps(result).replace("\\\\", "/")


def test_inference_budget_response_is_not_parse_failure(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _response(run, 0, "budget", schema_valid=False, reviewer="pipeline_budget", reason="qwen_call_budget_exhausted")
    report = diagnose_unknown(run, output_path=tmp_path / "result.json")
    assert report["candidates"][0]["diagnostic_reasons"] == ["budget_uncovered"]


@pytest.mark.parametrize("overrides", [{"reviewer_id": ""}, {"evidence_note": ""}, {"reason_categories": ["guessed_sleep"]}, {"candidate_event_id": "absent"}])
def test_human_review_requires_traceable_specific_evidence(tmp_path: Path, overrides: dict) -> None:
    run = tmp_path / "run"
    _response(run, 0, "e1")
    reviews = tmp_path / "review.jsonl"
    _rows(reviews, [{"candidate_event_id": "e1", "reason_categories": ["target_small_or_occluded"], "reviewer_id": "r1", "evidence_note": "checked frame", **overrides}])
    with pytest.raises(ValueError):
        diagnose_unknown(run, output_path=tmp_path / "result.json", review_path=reviews)


def test_annotation_templates_never_copy_predictions_or_invent_source_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import classroom_hybrid_ai.quality_diagnostics as diagnostic
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(diagnostic, "_video_frames", lambda _: [{"frame_index": index, "pts_seconds": index * .4, "pts_status": "valid", "width": 64, "height": 32} for index in range(8)])
    output = tmp_path / "package"
    result = prepare_annotation_package(video, output_dir=output)
    assert result["status"] == "pending_human_annotation"
    assert result["classroom_id"] is None
    assert result["double_review_status"] == "pending"
    assert result["source_video_sha256"] == hashlib.sha256(b"fixture").hexdigest()
    for name in ("boxes.v1.jsonl", "actions.v1.jsonl", "stable_identities.v1.jsonl"):
        assert (output / name).read_bytes() == b""
    reviewed = [json.loads(line) for line in (output / "annotated_frames.template.jsonl").read_text().splitlines()]
    assert {row["frame_index"] for row in reviewed} == {0, 3, 5, 7}
    assert all(row["confirmed_empty"] is None and row["review_status"] == "pending" for row in reviewed)
    assert not (output / "gold_manifest.v1.json").exists()
    with pytest.raises(FileExistsError):
        prepare_annotation_package(video, output_dir=output)


def test_annotation_rejects_wrong_run_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import classroom_hybrid_ai.quality_diagnostics as diagnostic
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(diagnostic, "_video_frames", lambda _: [{"frame_index": 0, "pts_seconds": 0, "pts_status": "valid"}])
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.v1.json").write_text(json.dumps({"source": {"sha256": "a" * 64}}))
    output = tmp_path / "package"
    with pytest.raises(ValueError, match="source_sha256_mismatch"):
        prepare_annotation_package(video, output_dir=output, run_dir=run)
    assert not output.exists()


def test_real_frame_ledger_retains_vfr_pts(tmp_path: Path) -> None:
    import av
    import numpy as np
    from fractions import Fraction
    from classroom_hybrid_ai.quality_diagnostics import _video_frames
    video = tmp_path / "vfr.mkv"
    with av.open(str(video), "w") as output:
        stream = output.add_stream("ffv1", rate=10)
        stream.width, stream.height = 16, 16
        stream.pix_fmt = "bgr0"
        for pts in [0, 100, 400, 1200]:
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    rows = _video_frames(video)
    assert [row["frame_index"] for row in rows] == [0, 1, 2, 3]
    assert [row["pts_seconds"] for row in rows] == pytest.approx([0, .1, .4, 1.2])
    assert all(row["pts_status"] == "valid" for row in rows)
