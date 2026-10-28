from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import classroom_hybrid_ai.outputs_v3 as outputs_v3
from classroom_hybrid_ai.outputs_v3 import (
    build_overlay_rows,
    build_v3_outputs,
    normalize_intervals,
    write_overlay_chunks,
    write_stage_preview,
)
from classroom_hybrid_ai.profiles import load_profile


ROOT = Path(__file__).resolve().parents[1]


def test_s0_preview_never_loads_downstream_rows(tmp_path, monkeypatch):
    def unexpected(path):
        raise AssertionError(f"unexpected downstream read: {path}")

    monkeypatch.setattr(outputs_v3, "_read_rows", unexpected)
    write_stage_preview(tmp_path, stage_id="S0", info={"duration_seconds": 1})
    assert (tmp_path / "previews/S0/preview.json").is_file()


def test_s4_preview_loads_only_evidence_dependencies(tmp_path, monkeypatch):
    reads = []

    def read(path):
        reads.append(path.relative_to(tmp_path).as_posix())
        return []

    monkeypatch.setattr(outputs_v3, "_read_rows", read)
    write_stage_preview(tmp_path, stage_id="S4", info={"duration_seconds": 1})
    assert sorted(reads) == ["artifacts/qwen_crop_queue.jsonl", "evidence/index.jsonl"]


def test_nvenc_missing_executable_falls_back_without_second_decode(tmp_path):
    source = tmp_path / "source.avi"
    _make_video(source)
    result = outputs_v3._render_videos_once(source, {"annotated_final": tmp_path / "final.mp4"}, [], info=_info(source), encoder_backend="nvenc", ffmpeg_path=tmp_path / "missing-ffmpeg.exe")
    receipt = result["annotated_final"]
    assert receipt["frames"] == 15
    assert receipt["encoder"]["effective"] == "opencv"
    assert receipt["encoder"]["fallback_reason"] == "ffmpeg_not_found"
    assert receipt["encoder"]["source_decode_passes"] == 1


def test_nvenc_runtime_failure_replays_overlay_stream(tmp_path, monkeypatch):
    source = tmp_path / "source.avi"
    _make_video(source)
    overlay = {"frame_index": 0, "student_id": "student-a", "bbox_xyxy": [5, 5, 20, 30]}
    monkeypatch.setattr(outputs_v3, "_probe_nvenc", lambda *args: {"status": "passed"})
    monkeypatch.setattr(outputs_v3, "_ffmpeg_executable", lambda value: "fake-ffmpeg")
    calls = []
    implementation = outputs_v3._render_videos_once_impl

    def render(source, paths, rows, **kwargs):
        rows = list(rows)
        calls.append(rows)
        if kwargs.get("ffmpeg"):
            raise RuntimeError("simulated encoder failure")
        return implementation(source, paths, rows, **kwargs)

    monkeypatch.setattr(outputs_v3, "_render_videos_once_impl", render)
    result = outputs_v3._render_videos_once(source, {"annotated_final": tmp_path / "final.mp4"}, iter([overlay]), info=_info(source), encoder_backend="nvenc")
    assert calls == [[overlay], [overlay]]
    assert result["annotated_final"]["encoder"]["source_decode_passes"] == 2
    assert not list(tmp_path.glob("encoder-retry-*"))


def test_audio_mux_truncation_preserves_original_render(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / "source.avi"
    rendered = tmp_path / "render.avi"
    _make_video(source)
    _make_video(rendered)
    original_bytes = rendered.read_bytes()
    monkeypatch.setattr(outputs_v3, "_ffmpeg_executable", lambda _: "fake-ffmpeg")

    def mux(command, **kwargs):
        target = Path(command[-1])
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
        for _ in range(3):
            writer.write(np.zeros((64, 64, 3), np.uint8))
        writer.release()
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(outputs_v3.subprocess, "run", mux)
    result = outputs_v3._preserve_audio(source, rendered, validate_streams=True)
    assert result["preserved"] is False
    assert "audio_mux_changed_video_timing" in result["reason"]
    assert rendered.read_bytes() == original_bytes
    assert not rendered.with_name("render.audio.mp4").exists()


def _make_video(path: Path, *, frames: int = 15, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 64))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((64, 64, 3), index * 7 % 255, dtype=np.uint8))
    writer.release()


def _write_observations(run_dir: Path, rows: list[dict[str, object]]) -> None:
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "track_observations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _info(video: Path, *, frames: int = 15, fps: float = 10.0, duration: float = 1.5) -> dict[str, object]:
    return {"path": str(video), "fps": fps, "frame_count": frames, "duration_seconds": duration, "width": 64, "height": 64}


def test_overlay_interval_lookup_and_two_second_hold_boundary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_observations(run_dir, [
        {"frame_index": 0, "pts_seconds": 0.0, "student_id": "student-001", "detection": {"bbox": [5, 5, 25, 50]}},
        {"frame_index": 30, "pts_seconds": 3.0, "student_id": "student-001", "detection": {"bbox": [15, 5, 35, 50]}},
    ])
    intervals = [{"student_id": "student-001", "start_pts_seconds": 0.0, "end_pts_seconds": 5.0, "action_id": "unknown", "status": "unobserved"}]
    rows = build_overlay_rows(run_dir, _info(tmp_path / "unused.mp4", frames=51, duration=5.0), intervals)
    synthetic = [row for row in rows if not row["bbox_observed"]]
    assert synthetic
    assert max(float(row["bbox_age_seconds"]) for row in synthetic) <= 2.0
    assert {int(row["frame_index"]) for row in synthetic} >= {1, 20}
    assert 21 not in {int(row["frame_index"]) for row in synthetic}
    assert any(int(row["frame_index"]) == 30 and row["bbox_observed"] for row in rows)


def test_sparse_multi_student_overlay_is_frame_sorted_and_expires_per_student(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_observations(run_dir, [
        {"frame_index": 0, "pts_seconds": 0.0, "student_id": "student-a", "detection": {"bbox": [5, 5, 25, 50]}},
        {"frame_index": 0, "pts_seconds": 0.0, "student_id": "student-b", "detection": {"bbox": [30, 5, 50, 50]}},
        {"frame_index": 5, "pts_seconds": 0.5, "student_id": "student-b", "detection": {"bbox": [31, 5, 51, 50]}},
        {"frame_index": 10, "pts_seconds": 1.0, "student_id": "student-a", "detection": {"bbox": [6, 5, 26, 50]}},
    ])
    intervals = [
        {"student_id": "student-a", "start_pts_seconds": 0.0, "end_pts_seconds": 5.0, "action_id": "unknown", "status": "unobserved"},
        {"student_id": "student-b", "start_pts_seconds": 0.0, "end_pts_seconds": 5.0, "action_id": "unknown", "status": "unobserved"},
    ]
    rows = build_overlay_rows(run_dir, _info(tmp_path / "unused.mp4", frames=50, duration=5.0), intervals)
    assert [int(row["frame_index"]) for row in rows] == sorted(int(row["frame_index"]) for row in rows)
    by_student = {student: [row for row in rows if row["student_id"] == student] for student in ("student-a", "student-b")}
    assert max(float(row["bbox_age_seconds"]) for row in by_student["student-a"] if not row["bbox_observed"]) <= 2.0
    assert max(float(row["bbox_age_seconds"]) for row in by_student["student-b"] if not row["bbox_observed"]) <= 2.0
    assert not any(int(row["frame_index"]) == 31 and row["student_id"] == "student-a" and not row["bbox_observed"] for row in rows)
    assert not any(int(row["frame_index"]) == 26 and row["student_id"] == "student-b" and not row["bbox_observed"] for row in rows)


def test_confirmed_qwen_interval_requires_candidate_event_trace(tmp_path: Path) -> None:
    profile = load_profile(ROOT / "configs" / "action_profiles.yaml")
    action = profile.actions[0]
    artifact = tmp_path / "run" / "action_intervals.jsonl"
    artifact.parent.mkdir(parents=True)
    base = {
        "student_id": "student-001",
        "start_pts_seconds": 0.0,
        "end_pts_seconds": 1.0,
        "action_id": action["action_id"],
        "status": "confirmed",
        "source": "qwen_classification",
    }
    artifact.write_text(json.dumps(base, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="confirmed_qwen_interval_missing_evidence_event_id"):
        normalize_intervals(tmp_path / "run", profile, 1.0, [{"student_id": "student-001"}])

    artifact.write_text(json.dumps({**base, "evidence_event_ids": ["candidate-001"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    intervals = normalize_intervals(tmp_path / "run", profile, 1.0, [{"student_id": "student-001"}])
    assert intervals[0]["evidence_event_ids"] == ["candidate-001"]
    assert intervals[0]["evidence_event_id"] == "candidate-001"


def test_build_outputs_decodes_once_and_receipts_have_distinct_video_semantics(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "sample.avi"
    _make_video(video)
    run_dir = tmp_path / "run"
    profile = load_profile(ROOT / "configs" / "action_profiles.yaml")
    calls: list[str] = []
    progress: list[tuple[int, int]] = []
    original_capture = outputs_v3.cv2.VideoCapture

    def counted_capture(*args, **kwargs):
        calls.append(str(args[0]) if args else "")
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(outputs_v3.cv2, "VideoCapture", counted_capture)
    result = build_v3_outputs(
        run_dir,
        source=video,
        info=_info(video),
        profile=profile,
        config_digest="config",
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )
    assert calls == [str(video.resolve())]
    assert progress[0] == (0, 15)
    assert progress[-1] == (15, 15)
    assert [row[0] for row in progress] == sorted(row[0] for row in progress)
    receipts = result["videos"]
    assert receipts["annotated_final"]["frames"] == 15
    assert receipts["tracking_preview"]["frames"] == receipts["action_preview"]["frames"] == 15
    assert receipts["tracking_preview"]["semantic"]["kind"] == "tracking_preview"
    assert receipts["action_preview"]["semantic"]["kind"] == "action_preview"
    assert receipts["annotated_final"]["semantic"]["kind"] == "annotated_final"
    assert receipts["tracking_preview"]["semantic"] != receipts["action_preview"]["semantic"]
    assert all("clip_ranges" in receipt and "sha256" in receipt and "bytes" in receipt for receipt in receipts.values())


def test_overlay_chunks_stream_at_sixty_seconds_and_frontend_uris_are_relative(tmp_path: Path) -> None:
    rows = [
        {"pts_seconds": 0.0, "frame_index": 0, "action_code": "js", "action_name_zh": "举手"},
        {"pts_seconds": 59.9, "frame_index": 1},
        {"pts_seconds": 60.0, "frame_index": 2},
    ]
    uris = write_overlay_chunks(tmp_path / "run", rows)
    assert uris == ["frontend/overlay_chunks/000000.json", "frontend/overlay_chunks/000001.json"]
    first = json.loads((tmp_path / "run" / uris[0]).read_text(encoding="utf-8"))
    assert first["rows"][0]["frame_index"] == 0
    assert first["rows"][0]["action_name_en"] == "HAND_RAISED"
    assert "action_name_zh" not in first["rows"][0]
    assert (tmp_path / "run" / uris[0]).read_text(encoding="utf-8").isascii()
    assert json.loads((tmp_path / "run" / uris[1]).read_text(encoding="utf-8"))["rows"][0]["frame_index"] == 2


def test_no_gold_metrics_are_explicitly_not_evaluated_and_have_no_none_or_forbidden_scores(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "sample.avi"
    _make_video(video, frames=3, fps=10.0)
    run_dir = tmp_path / "run"
    profile = load_profile(ROOT / "configs" / "action_profiles.yaml")
    monkeypatch.setattr(outputs_v3.shutil, "which", lambda _: None)
    result = build_v3_outputs(run_dir, source=video, info=_info(video, frames=3, duration=0.3), profile=profile, config_digest="config")
    metrics = json.loads((run_dir / "metrics.v1.json").read_text(encoding="utf-8"))
    assert metrics["evaluation_status"] == "not_evaluated"
    assert metrics["reason"] == "gold_annotations_not_provided"
    assert metrics["system"]["unavailable_metrics"]["stage_wall_time_seconds.S7"] == "stage_not_yet_committed"
    assert "total_wall_time_seconds" not in metrics["system"]

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)
        else:
            yield value

    flattened = list(walk(metrics))
    assert None not in flattened
    forbidden = {"precision", "recall", "mAP50", "IDF1", "HOTA", "macro_F1", "confusion_matrix", "segmental_F1@0.1", "segmental_F1@0.25", "segmental_F1@0.5"}
    assert forbidden.isdisjoint(set(flattened))
    assert result["warnings"]
    assert all(receipt["audio_status"] == "unavailable" for receipt in result["videos"].values())
    frontend = json.loads((run_dir / "frontend" / "index.v1.json").read_text(encoding="utf-8"))
    assert all(not Path(uri).is_absolute() for uri in frontend["video_uris"].values())
    assert all(not Path(uri).is_absolute() for uri in frontend["overlay_chunks"])
    frontend_text = (run_dir / "frontend" / "index.v1.json").read_text(encoding="utf-8") + (run_dir / "frontend" / "artifact_index.v1.json").read_text(encoding="utf-8")
    assert "diagnostics/" not in frontend_text
    assert frontend["display_language"] == "english_ascii"
    assert all(
        path.read_text(encoding="utf-8").isascii()
        for path in (run_dir / "frontend").rglob("*.json*")
    )
    for stage_id in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        preview = json.loads((run_dir / "previews" / stage_id / "preview.json").read_text(encoding="utf-8"))
        assert preview["semantic"]
        if preview["preview_status"] == "unavailable":
            assert preview["reason"]


def test_s1_preview_uses_only_the_active_schedule_after_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    active = run_dir / "artifacts" / "locate_screenshots" / "active" / "frame.jpg"
    stale = run_dir / "artifacts" / "locate_screenshots" / "stale" / "frame.jpg"
    active.parent.mkdir(parents=True)
    stale.parent.mkdir(parents=True)
    assert cv2.imwrite(str(active), np.full((64, 64, 3), 80, dtype=np.uint8))
    assert cv2.imwrite(str(stale), np.full((64, 64, 3), 220, dtype=np.uint8))
    schedule = run_dir / "artifacts" / "screenshot_schedule.v1.jsonl"
    schedule.write_text(
        json.dumps({"frame_index": 0, "pts_seconds": 0.0, "path": str(active)}) + "\n",
        encoding="utf-8",
    )

    write_stage_preview(
        run_dir,
        "S1",
        info=_info(tmp_path / "unused.avi", frames=1, duration=0.1),
    )

    preview = json.loads(
        (run_dir / "previews" / "S1" / "preview.json").read_text(encoding="utf-8")
    )
    assert preview["counts"]["screenshot_count"] == 1
    assert preview["uris"] == ["previews/S1/semantic_contact_12.jpg"]


def test_each_stage_owns_preview_before_s7_and_s7_preserves_upstream_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "sample.avi"
    _make_video(video, frames=3, fps=10.0)
    info = _info(video, frames=3, duration=0.3)
    profile = load_profile(ROOT / "configs" / "action_profiles.yaml")
    run_dir = tmp_path / "run"
    screenshot = run_dir / "artifacts" / "locate_screenshots" / "frame.jpg"
    screenshot.parent.mkdir(parents=True)
    assert cv2.imwrite(str(screenshot), np.full((64, 64, 3), 160, dtype=np.uint8))
    (run_dir / "run_manifest.v1.json").write_text(json.dumps({
        "source": info,
        "profile": profile.to_dict(),
    }, ensure_ascii=False), encoding="utf-8")

    s0_dir = run_dir / "previews" / "S0"
    s0_dir.mkdir(parents=True)
    for name in ("first", "middle", "last"):
        assert cv2.imwrite(str(s0_dir / f"input_{name}.jpg"), np.full((64, 64, 3), 80, dtype=np.uint8))
    write_stage_preview(run_dir, "S0", info=info)
    write_stage_preview(run_dir, "S1", info=info)

    (run_dir / "artifacts" / "locate_raw_results.jsonl").write_text(json.dumps({
        "frame_index": 0,
        "pts_seconds": 0.0,
        "path": str(screenshot),
        "detections": [{"detection_index": 1, "label": "student", "bbox": [5, 5, 30, 55]}],
    }) + "\n", encoding="utf-8")
    write_stage_preview(run_dir, "S2", info=info)

    (run_dir / "artifacts" / "locate_student_boxes.jsonl").write_text(json.dumps({
        "frame_index": 0,
        "pts_seconds": 0.0,
        "student_id": "student-001",
        "locate_detection_index": 1,
        "bbox": [5, 5, 30, 55],
        "annotated_path": str(screenshot),
    }) + "\n", encoding="utf-8")
    _write_observations(run_dir, [{
        "frame_index": 0,
        "pts_seconds": 0.0,
        "student_id": "student-001",
        "detection": {"bbox": [5, 5, 30, 55]},
    }])
    write_stage_preview(run_dir, "S3", info=info)

    (run_dir / "artifacts" / "qwen_crop_queue.jsonl").write_text(json.dumps({
        "candidate_event_id": "event-1",
        "student_id": "student-001",
        "status": "ready",
        "composite_path": str(screenshot),
    }) + "\n", encoding="utf-8")
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / "index.jsonl").write_text(json.dumps({
        "event_id": "event-1",
        "qwen_composite": {"path": str(screenshot)},
    }) + "\n", encoding="utf-8")
    write_stage_preview(run_dir, "S4", info=info)

    (run_dir / "artifacts" / "candidates.selected.jsonl").write_text(json.dumps({
        "event_id": "event-1",
        "track_id": "student-001",
        "trigger_pts_seconds": 0.0,
    }) + "\n", encoding="utf-8")
    (run_dir / "reviews.classification.jsonl").write_text(json.dumps({
        "candidate_event_id": "event-1",
        "behavior": "a_dx",
        "decision": "confirmed",
        "observability": "clear",
    }) + "\n", encoding="utf-8")
    write_stage_preview(run_dir, "S5", info=info)
    s5 = json.loads((run_dir / "previews" / "S5" / "preview.json").read_text(encoding="utf-8"))
    assert "HEAD_DOWN_WRITING" in s5["action_labels"]
    assert s5["action_colors"]["HEAD_DOWN_WRITING"] == "#4E79A7"

    (run_dir / "events.jsonl").write_text(json.dumps({
        "event_id": "event-1",
        "student_id": "student-001",
        "trigger_pts_seconds": 0.0,
    }) + "\n", encoding="utf-8")
    (run_dir / "action_intervals.jsonl").write_text(json.dumps({
        "student_id": "student-001",
        "start_pts_seconds": 0.0,
        "end_pts_seconds": 0.3,
        "action_id": "a_dx",
        "action_name_zh": "低头写字",
        "color": "#4E79A7",
        "status": "observed",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "students.jsonl").write_text(json.dumps({
        "student_id": "student-001",
        "first_pts_seconds": 0.0,
        "last_pts_seconds": 0.3,
    }) + "\n", encoding="utf-8")
    (run_dir / "action_observations.jsonl").write_text(json.dumps({
        "student_id": "student-001",
        "pts_seconds": 0.0,
        "action_id": "a_dx",
        "action_name_zh": "低头写字",
        "color": "#4E79A7",
        "status": "observed",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "taxonomy_profile.v1.json").write_text(
        json.dumps(outputs_v3.taxonomy_payload(profile), ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "timeline.v2.json").write_text(json.dumps({
        "video": info,
        "intervals": [],
        "taxonomy_profile_id": profile.profile_id,
        "taxonomy_contract_sha256": profile.sha256,
    }), encoding="utf-8")
    write_stage_preview(run_dir, "S6", info=info)

    upstream = {
        stage_id: (run_dir / "previews" / stage_id / "preview.json").read_bytes()
        for stage_id in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
    }
    assert all(
        json.loads(value)["preview_status"] == "available"
        for value in upstream.values()
    )
    monkeypatch.setattr(outputs_v3.shutil, "which", lambda _: None)
    build_v3_outputs(
        run_dir,
        source=video,
        info=info,
        profile=profile,
        config_digest="config",
        reuse_stage_artifacts=True,
    )
    for stage_id, expected in upstream.items():
        assert (run_dir / "previews" / stage_id / "preview.json").read_bytes() == expected


def test_video_overlay_uses_ascii_action_labels_and_hides_unobserved_action_text(
    monkeypatch,
) -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    labels: list[str] = []
    original = outputs_v3.cv2.putText

    def record_label(target, text, *args, **kwargs):
        labels.append(str(text))
        return original(target, text, *args, **kwargs)

    monkeypatch.setattr(outputs_v3.cv2, "putText", record_label)
    confirmed = {
        "bbox_px": [8, 24, 80, 72],
        "bbox_observed": True,
        "interval_status": "confirmed",
        "student_id": "student-007",
        "action_code": "js",
        "action_name_zh": "举手",
        "color": "#E15759",
    }
    outputs_v3._draw_overlay(image, confirmed, semantic="action_preview")
    assert labels == ["HAND_RAISED"]
    assert labels[0].isascii()

    labels.clear()
    outputs_v3._draw_overlay(
        image,
        {**confirmed, "interval_status": "unobserved"},
        semantic="action_preview",
    )
    assert labels == []

    outputs_v3._draw_overlay(
        image,
        {**confirmed, "interval_status": "unobserved"},
        semantic="annotated_final",
    )
    assert labels == ["S007"]


def test_s7_metrics_include_upstream_performance_before_artifact_hashing(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    video = tmp_path / "sample.avi"
    _make_video(video, frames=3, fps=10.0)
    run_dir = tmp_path / "run"
    diagnostics = run_dir / "diagnostics" / "performance"
    diagnostics.mkdir(parents=True)
    for stage, calls, peak in (("S2", {"locate.actual_calls": 2}, 3), ("S5", {"qwen.actual_calls": 4}, 6), ("S7", {}, 99)):
        (diagnostics / f"{stage}.json").write_text(json.dumps({
            "timings_seconds": {"stage.wall_seconds": 12.0}, "counts": calls,
            "model_requests": [{"memory": {"max_reserved_bytes": peak * 2**30}, "counts": {"input_tokens": 30, "output_tokens": 4}}],
        }), encoding="utf-8")
    # A stale prior attempt must not supply a fictitious completed-run time.
    (run_dir / "run_state.v1.json").write_text(json.dumps({"status": "completed", "total_wall_time_seconds": 999, "stages": {"S7": {"peak_vram_gib": 99}}}), encoding="utf-8")
    profile = load_profile(ROOT / "configs" / "action_profiles.yaml")
    monkeypatch.setattr(outputs_v3.shutil, "which", lambda _: None)
    build_v3_outputs(run_dir, source=video, info=_info(video, frames=3, duration=0.3), profile=profile, config_digest="config")
    path = run_dir / "metrics.v1.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    system = metrics["system"]
    assert metrics["evaluation_status"] == "not_evaluated"
    assert system["locate_actual_requests"] == 2
    assert system["qwen_actual_calls"] == 4
    assert system["peak_reserved_gib"] == 6
    assert system["stage_performance"]["S5"]["model_requests"][0]["counts"]["output_tokens"] == 4
    assert "S7" not in system["stage_performance"]
    assert "S7" not in system["stage_wall_time_seconds"]
    assert "total_wall_time_seconds" not in system
    assert system["rendered_video_count"] == 3
    assert system["unavailable_metrics"]["total_wall_time_seconds"] == "run_not_completed_when_metrics_committed"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for artifact_file in (run_dir / "manifest.json", run_dir / "frontend" / "artifact_index.v1.json"):
        manifest = json.loads(artifact_file.read_text(encoding="utf-8"))
        assert next(row["sha256"] for row in manifest["artifacts"] if row["uri"] == "metrics.v1.json") == digest
