from __future__ import annotations

from classroom_hybrid_ai.contracts import CandidateEvent
from classroom_hybrid_ai.evidence import (
    ObservationIndex,
    _observation_box,
    build_event_evidence,
    build_temporal_composite,
    expand_box,
    filter_observations_for_candidates,
    temporal_sample_times,
)


def test_expand_box_clamps_to_image() -> None:
    assert expand_box([5, 10, 25, 50], 100, 80, 0.5) == (0, 0, 35, 70)


def test_temporal_sample_uses_requested_count() -> None:
    assert temporal_sample_times(5.0, 1.0, 1.0, 3) == [4.0, 5.0, 6.0]


def test_observation_index_selects_nearest_pts() -> None:
    index = ObservationIndex([
        {"track_id": "7", "frame_index": 10, "pts_seconds": 1.0, "bbox_xyxy": [0, 0, 10, 10]},
        {"track_id": "7", "frame_index": 20, "pts_seconds": 2.0, "bbox_xyxy": [1, 1, 11, 11]},
        {"track_id": "8", "frame_index": 20, "pts_seconds": 2.0, "bbox_xyxy": [20, 20, 30, 30]},
    ])
    assert index.nearest("7", 1.8)["pts_seconds"] == 2.0
    assert {row["track_id"] for row in index.at_frame(20)} == {"7", "8"}


def test_canonical_nested_detection_box_is_supported() -> None:
    assert _observation_box({"detection": {"bbox": [1, 2, 3, 4]}}) == [1, 2, 3, 4]


def test_filter_observations_keeps_only_event_track_window() -> None:
    candidate = CandidateEvent(
        event_id="e1",
        track_id="s0:7",
        event_type="head_down",
        trigger_frame_index=30,
        trigger_pts_seconds=3.0,
    )
    rows = [
        {"track_id": "s0:7", "pts_seconds": 2.0},
        {"track_id": "s0:7", "pts_seconds": 3.1},
        {"track_id": "s0:7", "pts_seconds": 5.0},
        {"track_id": "s0:8", "pts_seconds": 3.0},
    ]
    selected = filter_observations_for_candidates(
        rows, [candidate], before_seconds=1.0, after_seconds=1.0
    )
    assert [row["pts_seconds"] for row in selected] == [2.0, 3.1]


def test_filter_observations_can_retain_full_context_window() -> None:
    candidate = CandidateEvent(
        event_id="e1",
        track_id="s0:7",
        event_type="head_down",
        trigger_frame_index=30,
        trigger_pts_seconds=3.0,
    )
    rows = [
        {"track_id": "s0:7", "pts_seconds": 3.0},
        {"track_id": "s0:8", "pts_seconds": 3.0},
        {"track_id": "s0:9", "pts_seconds": 7.0},
    ]
    selected = filter_observations_for_candidates(
        rows, [candidate], before_seconds=1.0, after_seconds=1.0,
        include_all_tracks_in_windows=True,
    )
    assert {row["track_id"] for row in selected} == {"s0:7", "s0:8"}


def test_temporal_composite_is_one_local_image(tmp_path) -> None:
    import cv2
    import numpy as np

    evidence = []
    for index in range(3):
        path = tmp_path / f"subject_{index}.jpg"
        assert cv2.imwrite(str(path), np.full((20, 30, 3), index * 50, dtype=np.uint8))
        evidence.append({"kind": "subject_crop", "path": str(path), "frame_index": index})
    receipt = build_temporal_composite({"evidence": evidence}, output_path=tmp_path / "composite.jpg")
    assert receipt is not None
    assert (tmp_path / "composite.jpg").is_file()
    assert receipt["shape"] == [480, 416]


def test_locate_student_box_drives_qwen_subject_crop_and_context(tmp_path) -> None:
    import cv2
    import numpy as np

    video = tmp_path / "one.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (100, 80)
    )
    writer.write(np.full((80, 100, 3), 120, dtype=np.uint8))
    writer.release()
    candidate = CandidateEvent(
        event_id="e-locate",
        track_id="track-7",
        event_type="review",
        trigger_frame_index=0,
        trigger_pts_seconds=0.0,
    )
    progress: list[tuple[int, int]] = []
    receipts = build_event_evidence(
        video_path=video,
        candidates=[candidate],
        observations=[{
            "track_id": "track-7", "student_id": "student-3",
            "frame_index": 0, "pts_seconds": 0.0, "bbox_xyxy": [10, 10, 50, 70],
        }],
        locate_rows=[{
            "status": "ok", "frame_index": 0, "pts_seconds": 0.0,
            "detections": [{
                "label": "student", "bbox": [12, 12, 48, 68], "detection_index": 4,
            }],
        }],
        output_dir=tmp_path / "evidence",
        before_seconds=0.0,
        after_seconds=0.0,
        frames_per_event=1,
        context_ratio=0.0,
        maximum_crop_side=100,
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )
    items = receipts[0]["evidence"]
    subject = next(item for item in items if item["kind"] == "subject_crop")
    context = next(item for item in items if item["kind"] == "annotated_context")
    assert subject["localization_source"] == "locate_anything"
    assert subject["locate_detection_index"] == 4
    assert subject["bbox_xyxy"] == [12, 12, 48, 68]
    assert cv2.imread(context["path"]) is not None
    assert progress == [(0, 1), (1, 1)]


def test_composite_reuses_repeated_context_read_and_resize(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import classroom_hybrid_ai.evidence as evidence_module

    image = tmp_path / "single.jpg"
    assert cv2.imwrite(str(image), np.full((30, 40, 3), 120, np.uint8))
    reads = []
    original = cv2.imread

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(evidence_module.cv2, "imread", read)
    receipt = build_temporal_composite({"evidence": [{"kind": "subject_crop", "path": str(image), "frame_index": 0}]}, output_path=tmp_path / "composite.jpg")
    assert receipt is not None
    assert reads == [str(image)]


def test_threaded_evidence_keeps_receipt_order_and_exact_jpeg_bytes(tmp_path):
    import cv2
    import numpy as np
    from pathlib import Path

    source = tmp_path / "events.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (100, 80))
    for index in range(40):
        writer.write(np.full((80, 100, 3), index * 5, np.uint8))
    writer.release()
    candidates = [{"event_id": f"e-{track}", "track_id": track, "trigger_pts_seconds": 2.0} for track in ("a", "b")]
    observations = [{"track_id": track, "frame_index": index, "pts_seconds": index / 10, "bbox_xyxy": box} for index in range(40) for track, box in (("a", [2, 2, 45, 75]), ("b", [50, 2, 95, 75]))]
    arguments = dict(video_path=source, candidates=candidates, observations=observations, before_seconds=1, after_seconds=1, frames_per_event=3, context_ratio=0.1, maximum_crop_side=80)
    sequential = build_event_evidence(**arguments, output_dir=tmp_path / "seq", preprocess_workers=1)
    progress = []
    threaded = build_event_evidence(**arguments, output_dir=tmp_path / "parallel", preprocess_workers=4, preprocess_queue_size=2, progress_callback=lambda done, total: progress.append((done, total)))
    assert [row["event_id"] for row in threaded] == [row["event_id"] for row in sequential]
    for actual, expected in zip(threaded, sequential):
        assert actual["target_highlighted"] == expected["target_highlighted"]
        assert len(actual["evidence"]) == len(expected["evidence"])
        for item, old in zip(actual["evidence"], expected["evidence"]):
            assert {k: v for k, v in item.items() if k != "path"} == {k: v for k, v in old.items() if k != "path"}
            assert Path(item["path"]).read_bytes() == Path(old["path"]).read_bytes()
    assert progress == [(index, 6) for index in range(7)]
