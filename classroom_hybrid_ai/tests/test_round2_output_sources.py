"""Detector provenance must survive association, candidates and S7 rendering."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from classroom_hybrid_ai.contracts import Detection, TrackObservation
from classroom_hybrid_ai.hashing import canonical_sha256
from classroom_hybrid_ai.locate_tracking import candidates_from_locate_observations
from classroom_hybrid_ai.pipeline import fair_candidate_selection
from classroom_hybrid_ai.storage import atomic_write_jsonl
import classroom_hybrid_ai.outputs_v3 as outputs


def observation(source="yolo", *, generic=True, index=7):
    return {"frame_index": 0, "pts_seconds": 0., "track_id": "student-001",
        "detection": {"label": "student", "bbox": [5, 10, 25, 50], "source": source,
            "attributes": {"detector_detection_index": index, "association_score": .85,
                           "student_confirmed": True}},
        "metadata": {"localization_contract": "classroom_hybrid_ai.localization.v1"} if generic else {}}


def overlay_rows(tmp_path, row, auxiliary=None):
    atomic_write_jsonl(tmp_path / "artifacts/track_observations.jsonl", [row])
    if auxiliary is not None:
        atomic_write_jsonl(tmp_path / "artifacts/locate_student_boxes.jsonl", [
            {"frame_index": 0, "student_id": "student-001", **auxiliary}])
    info = {"fps": 2., "width": 100, "height": 100, "frame_count": 4, "duration_seconds": 2.}
    return list(outputs.iter_overlay_rows(tmp_path, info, []))


def test_yolo_provenance_survives_legacy_cursor_and_held_frames(tmp_path):
    row = observation()
    # The old field can remain a compatibility alias without proving source.
    row["detection"]["attributes"]["locate_detection_index"] = 99
    overlays = overlay_rows(tmp_path, row, {"source": "yolo", "locate_detection_index": 99})
    assert overlays[0]["bbox_source"] == "yolo"
    assert all(item["detector_source"] == "yolo" and item["detection_index"] == 7 for item in overlays)
    assert all(item["locate_detection_index"] is None for item in overlays)
    assert all(item["bbox_source"] == "held" and item["bbox_observed"] is False for item in overlays[1:])
    assert [item["frame_index"] for item in overlays] == [0, 1, 2, 3]
    assert all(item["bbox_px"] == [5., 10., 25., 50.] for item in overlays)


def test_observation_provider_overrides_conflicting_compatibility_box(tmp_path):
    overlays = overlay_rows(tmp_path, observation(), {"source": "locate_anything", "locate_detection_index": 99})
    assert overlays[0]["detector_source"] == "yolo" and overlays[0]["bbox_source"] == "yolo"
    assert overlays[0]["detection_index"] == 7 and overlays[0]["locate_detection_index"] is None


def test_versioned_missing_source_does_not_infer_locate_from_compatibility_index(tmp_path):
    row = observation()
    row["detection"].pop("source")
    overlays = overlay_rows(tmp_path, row, {"locate_detection_index": 99})
    assert overlays[0]["bbox_source"] == overlays[0]["detector_source"] == "unknown"
    assert overlays[0]["locate_detection_index"] is None


def test_generic_locate_retains_neutral_detection_index(tmp_path):
    overlays = overlay_rows(tmp_path, observation("locate_anything"), {"source": "locate_anything", "locate_detection_index": 7})
    assert overlays[0]["bbox_source"] == "locate" and overlays[0]["detector_source"] == "locate_anything"
    assert overlays[0]["detection_index"] == 7 and overlays[0]["locate_detection_index"] == 7


def test_old_locate_payload_bytes_match_pre_fix_snapshot():
    row = {"frame_index": 0, "pts_seconds": 0., "track_id": "student-001",
        "detection": {"bbox": [5, 10, 25, 50], "source": "locate_anything",
                      "attributes": {"locate_detection_index": 7, "association_score": .85}}}
    result = outputs._overlay_payload(row=row, student_id="student-001", frame=0, pts=0.,
        bbox=[5, 10, 25, 50], locate={"locate_detection_index": 7, "association_score": .85},
        width=100, height=100, action={"action_id": "a_js", "action_code": "js", "action_name_zh": "举手",
                                     "status": "confirmed", "color": "#E15759"},
        bbox_source="locate", bbox_observed=True, bbox_age_seconds=0., bbox_interpolation="none")
    # Captured before the S7 provider fix; includes every serialized field.
    assert canonical_sha256(result) == "ddc1a4bfb278c4d828157737f5a9f0d8592e97fb3568abf3973f45d086d7bebe"


@pytest.mark.parametrize("provider,expected", [("yolo", "S001 Y7"), ("other", "S001 D7"),
    ("locate_anything", "S001 L99"), (None, "S001 L99")])
def test_tracking_labels_use_actual_provider(monkeypatch, provider, expected):
    labels = []
    original = outputs.cv2.putText
    def capture(image, text, *args, **kwargs):
        labels.append(text)
        return original(image, text, *args, **kwargs)
    monkeypatch.setattr(outputs.cv2, "putText", capture)
    row = {"bbox_px": [5, 20, 50, 60], "student_id": "student-001", "locate_detection_index": 99,
           "detection_index": 7, "bbox_observed": True, "interval_status": "confirmed"}
    if provider is not None:
        row["detector_source"] = provider
    outputs._draw_overlay(np.zeros((80, 120, 3), dtype=np.uint8), row, semantic="tracking_preview")
    assert labels == [expected]


@pytest.mark.parametrize("semantic", ["action_preview", "annotated_final"])
def test_source_correction_does_not_change_action_rendering_pixels(semantic):
    row = {"bbox_px": [5, 20, 50, 60], "student_id": "student-001", "locate_detection_index": 7,
           "bbox_observed": True, "interval_status": "confirmed", "action_code": "js", "color": "#E15759"}
    old = np.zeros((80, 120, 3), dtype=np.uint8)
    new = old.copy()
    outputs._draw_overlay(old, row, semantic=semantic)
    outputs._draw_overlay(new, {**row, "locate_detection_index": None, "detector_source": "yolo", "detection_index": 7}, semantic=semantic)
    assert np.array_equal(old, new)


def model_observations(source="locate_anything"):
    return [TrackObservation(track_id="student-001", frame_index=index, pts_seconds=float(index),
        detection=Detection(label="student", bbox=(5, 10, 25, 50), source=source)) for index in (0, 1)]


def test_old_locate_candidate_bytes_match_pre_fix_snapshot():
    candidates = candidates_from_locate_observations(model_observations())
    assert canonical_sha256([item.to_dict() for item in candidates]) == "6b9a9da7010129b5353a6fd0ac6b9ef49aea836ca432c6d2e7bde11ac7358af4"


def test_generic_candidate_source_is_truthful_without_changing_ids_or_selection():
    old = candidates_from_locate_observations(model_observations())
    actual = candidates_from_locate_observations(model_observations("yolo"))
    assert [item.source for item in actual] == ["yolo", "yolo"]
    assert [item.event_id for item in actual] == [item.event_id for item in old]
    assert [item.event_type for item in actual] == [item.event_type for item in old]
    assert all(item.metadata["trigger"] == "every_localization_screenshot" for item in actual)
    assert [item.metadata["selection_features"] for item in actual] == [item.metadata["selection_features"] for item in old]
    selected_old, _ = fair_candidate_selection(old, limit=1, bucket_seconds=300)
    selected_actual, _ = fair_candidate_selection(actual, limit=1, bucket_seconds=300)
    assert [item.event_id for item in selected_actual] == [item.event_id for item in selected_old]


@pytest.mark.parametrize("generic", [False, True])
def test_tracking_video_receipt_reports_sources_without_changing_legacy_semantics(tmp_path, generic):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 2., (100, 100))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((100, 100, 3), index * 30, np.uint8))
    writer.release()
    overlay = {"frame_index": 0, "bbox_px": [5, 20, 50, 70], "student_id": "student-001", "locate_detection_index": 7}
    if generic:
        overlay.update({"detector_source": "yolo", "detection_index": 7, "locate_detection_index": None})
    output = outputs._render_videos_once_impl(source, {"tracking_preview": tmp_path / "tracking.mp4"}, [overlay],
        info={"fps": 2., "frame_count": 3, "duration_seconds": 1.5, "width": 100, "height": 100})
    semantic = output["tracking_preview"]["semantic"]
    assert output["tracking_preview"]["frames"] == 3
    if generic:
        assert semantic["detector_sources"] == ["yolo"] and "Y=YOLO" in semantic["description"]
        assert semantic["labels"] == ["student_id", "detector_source", "detection_index"]
    else:
        assert semantic == {"kind": "tracking_preview",
            "description": "Up to 10 seconds from the beginning, middle, and end; shows student IDs, track IDs, and Locate indices only.",
            "labels": ["student_id", "locate_detection_index"], "clip_policy": "first_middle_last_10_seconds_deduplicated"}


def test_unversioned_locate_missing_old_index_retains_original_bbox_source(tmp_path):
    overlays = overlay_rows(tmp_path, observation("locate_anything", generic=False))
    assert overlays[0]["bbox_source"] == "tracked"
    assert "detector_source" not in overlays[0] and "detection_index" not in overlays[0]


def test_empty_yolo_delivery_describes_generic_sources_without_claiming_locate(tmp_path):
    source = tmp_path / "empty.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 2., (100, 100))
    assert writer.isOpened()
    writer.write(np.zeros((100, 100, 3), np.uint8))
    writer.release()
    result = outputs._render_videos_once_impl(source, {"tracking_preview": tmp_path / "tracking.mp4"}, [],
        info={"fps": 2., "frame_count": 1, "duration_seconds": .5, "width": 100, "height": 100,
              "localization_route": "yolo_only"})
    semantic = result["tracking_preview"]["semantic"]
    assert semantic["detector_sources"] == []
    assert "actual detector indices" in semantic["description"] and "Locate indices only" not in semantic["description"]
