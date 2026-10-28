from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from classroom_hybrid_ai.locate_tracking import (
    associate_locate_rows,
    candidates_from_locate_observations,
    materialize_locate_screenshots,
    render_locate_student_annotations,
)


def _video(path: Path, frames: int = 10) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (80, 60))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((60, 80, 3), index * 10, dtype=np.uint8))
    writer.release()


def test_materialize_locate_screenshots_obeys_frame_stride(tmp_path: Path) -> None:
    video = tmp_path / "sample.avi"
    _video(video)
    rows, stage = materialize_locate_screenshots(
        video, tmp_path / "screens", frame_stride=3
    )
    assert [row["frame_index"] for row in rows] == [0, 3, 6, 9]
    assert all(Path(row["path"]).is_file() for row in rows)
    assert stage["screenshots"] == 4
    assert stage["truncated"] is False


def test_locate_boxes_receive_stable_ids_and_qwen_order(tmp_path: Path) -> None:
    rows = [
        {
            "status": "ok", "path": str(tmp_path / "f0.jpg"),
            "frame_index": 0, "pts_seconds": 0.0,
            "detections": [
                {"label": "student", "bbox": [10, 10, 30, 40], "detection_index": 8},
                {"label": "student", "bbox": [50, 10, 70, 40], "detection_index": 2},
            ],
        },
        {
            "status": "ok", "path": str(tmp_path / "f1.jpg"),
            "frame_index": 10, "pts_seconds": 1.0,
            # Reverse the model's output order; IDs must follow geometry.
            "detections": [
                {"label": "student", "bbox": [51, 10, 71, 40], "detection_index": 1},
                {"label": "student", "bbox": [11, 10, 31, 40], "detection_index": 9},
            ],
        },
    ]
    observations, enriched, flat, stage = associate_locate_rows(
        rows,
        max_gap_seconds=2.0,
        minimum_iou=0.1,
        maximum_center_distance=0.75,
        minimum_match_score=0.3,
        minimum_match_margin=0.1,
    )
    assert [(row.frame_index, row.track_id) for row in observations] == [
        (0, "student-001"), (0, "student-002"),
        (10, "student-001"), (10, "student-002"),
    ]
    assert [row["student_id"] for row in flat] == [
        "student-001", "student-002", "student-001", "student-002"
    ]
    assert all(item.get("student_id") for row in enriched for item in row["detections"])
    assert stage["source"] == "locate_anything"
    assert stage["students"] == 2
    assert stage["reliable_students"] == 2
    assert stage["ambiguous_associations"] == 0

    candidates = candidates_from_locate_observations(observations)
    assert [(row.trigger_frame_index, row.track_id) for row in candidates] == [
        (0, "student-001"), (0, "student-002"),
        (10, "student-001"), (10, "student-002"),
    ]
    assert all(row.source == "locate_anything" for row in candidates)


def test_crowded_ambiguous_matches_keep_ids_and_mark_qwen_uncertain(tmp_path: Path) -> None:
    rows = [
        {
            "status": "ok", "path": str(tmp_path / "f0.jpg"),
            "frame_index": 0, "pts_seconds": 0.0,
            "detections": [
                {"label": "student", "bbox": [0, 0, 10, 20], "detection_index": 1},
                {"label": "student", "bbox": [12, 0, 22, 20], "detection_index": 2},
            ],
        },
        {
            "status": "ok", "path": str(tmp_path / "f1.jpg"),
            "frame_index": 10, "pts_seconds": 1.0,
            "detections": [
                {"label": "student", "bbox": [6, 0, 16, 20], "detection_index": 1},
                {"label": "student", "bbox": [6, 0, 16, 20], "detection_index": 2},
            ],
        },
    ]
    observations, _, _, stage = associate_locate_rows(
        rows, max_gap_seconds=2.0, minimum_iou=0.1,
        maximum_center_distance=0.75, minimum_match_score=0.3,
        minimum_match_margin=0.1,
    )

    assert {row.track_id for row in observations} == {"student-001", "student-002"}
    second = [row for row in observations if row.frame_index == 10]
    assert any(bool(row.detection.attributes["association_ambiguous"]) for row in second)
    candidates = candidates_from_locate_observations(second)
    assert any(bool(row.metadata["id_uncertain"]) for row in candidates)
    assert stage["students"] == 2
    assert stage["reliable_students"] == 0
    assert stage["uncertain_students"] == 2
    assert stage["ambiguous_associations"] >= 1


def test_long_gap_reacquires_persistent_seat_id() -> None:
    rows = [
        {
            "status": "ok", "path": "f0.jpg", "frame_index": 0, "pts_seconds": 0.0,
            "detections": [{"label": "student", "bbox": [10, 10, 30, 40]}],
        },
        {
            "status": "ok", "path": "f1.jpg", "frame_index": 100, "pts_seconds": 10.0,
            "detections": [{"label": "student", "bbox": [11, 10, 31, 40]}],
        },
    ]
    observations, _, flat, stage = associate_locate_rows(
        rows, max_gap_seconds=2.0, minimum_iou=0.1,
        maximum_center_distance=0.75, minimum_match_score=0.3,
        minimum_match_margin=0.1,
    )

    assert [row.track_id for row in observations] == ["student-001", "student-001"]
    assert flat[1]["association_method"] == "seat_anchor_reacquisition"
    assert flat[1]["association_gap_seconds"] == 10.0
    assert stage["students"] == 1
    assert stage["reliable_students"] == 1


def test_render_locate_annotations_makes_ids_visible(tmp_path: Path) -> None:
    source = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(source), np.full((60, 80, 3), 230, dtype=np.uint8))
    rows = [{
        "status": "ok",
        "path": str(source),
        "frame_index": 7,
        "pts_seconds": 0.7,
        "detections": [{
            "label": "student",
            "bbox": [10, 10, 40, 50],
            "student_id": "student-003",
            "locate_detection_index": 2,
        }],
    }]
    rendered, stage = render_locate_student_annotations(rows, tmp_path / "annotated")
    output = Path(rendered[0]["annotated_path"])
    assert output.is_file()
    assert rendered[0]["annotation_status"] == "ok"
    assert stage["rendered_student_boxes"] == 1
    annotated = cv2.imread(str(output))
    original = cv2.imread(str(source))
    assert annotated is not None and original is not None
    assert np.any(annotated != original)
