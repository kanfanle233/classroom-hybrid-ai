from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from urllib.error import URLError

import cv2
import numpy as np
import pytest

from classroom_hybrid_ai.config import ProjectConfig, load_config
from classroom_hybrid_ai.pipeline import HybridPipeline, PipelineError, _RoundRobinTracks, _box_overlap, _classification_conflict_groups, _classification_prompt, _custom_anomaly_pass, _discovery_prompt, _fair_candidates, _interaction_signature, _locate_anchor_frames, _locate_pass, _merge_and_select_candidates, _merge_candidate_events, _qwen_budget, _qwen_pass, _resolve_classification_conflicts, _semantic_batch_adjudicator, _student_catalog, _target_locate_evidence, _track_pass, _validate_classification_response, _validate_discovery_response
from classroom_hybrid_ai.storage import RunStore, iter_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_effective_qwen_prompts_are_not_mojibake() -> None:
    discovery = _discovery_prompt()
    classification = _classification_prompt([], "digest")
    assert "concrete Chinese verb-object phrase" in discovery
    assert "Frozen vocabulary=[]" in classification
    assert chr(0xFFFD) not in discovery + classification


def test_discovery_rejects_frozen_eight_label_codes() -> None:
    response = {
        "candidate_event_id": "event-1", "decision": "confirmed",
        "reviewed_event_type": "review", "behavior": "dx",
        "rationale": "visible", "evidence_frame_indices": [0, 1],
        "self_reported_confidence": None, "reviewer": "qwen3.5",
        "schema_valid": True, "secondary_behaviors": [], "posture": "sitting", "movement_state": "still",
        "head_direction": "down", "hand_action": "writing", "visible_objects": [],
        "interaction_target": None, "observability": "clear",
    }
    with pytest.raises(PipelineError, match="legacy_label_forbidden"):
        _validate_discovery_response(response, (0, 1))


class FakeTracker:
    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    def track(self, frame, persist=True):
        return {
            "detections": [{
                "label": "person",
                "bbox": [8, 8, 40, 55],
                "score": 0.9,
                "class_id": 0,
                "track_id": "1",
                "source": "fake_yolo",
                "attributes": {},
            }]
        }

    def close(self) -> None:
        self.closed = True


def _write_video(path: Path, frames: int = 8) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((64, 64, 3), index * 5, dtype=np.uint8)
        writer.write(image)
    writer.release()


def _temp_config(tmp_path: Path) -> ProjectConfig:
    original = load_config(ROOT / "configs" / "default.yaml")
    payload = deepcopy(original.payload)
    payload["paths"]["output_root"] = str(tmp_path / "output")
    payload["locate_anything"]["enabled"] = False
    payload["pipeline"]["primary_detector"] = "yolo"
    return ProjectConfig(
        source_path=original.source_path,
        project_root=original.project_root,
        repository_root=original.repository_root,
        payload=payload,
    )


def test_overlap_associates_tiny_object_inside_person() -> None:
    assert _box_overlap([0, 0, 100, 100], [40, 40, 50, 50]) == 1.0
    assert _box_overlap([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_qwen_locate_metadata_contains_only_current_student_and_is_bounded() -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent

    candidate = CandidateEvent("event", "student-002", "review", 10, 1.0)
    rows = []
    for frame_index, pts in enumerate((0.0, 0.5, 1.0, 1.5, 2.0)):
        rows.append({
            "frame_index": frame_index,
            "pts_seconds": pts,
            "detections": [
                {
                    "student_id": "student-001", "bbox": [0, 0, 10, 10],
                    "locate_detection_index": 1, "large_attributes": {"ignored": "x" * 1000},
                },
                {
                    "student_id": "student-002", "bbox": [10, 10, 20, 20],
                    "locate_detection_index": 2, "large_attributes": {"ignored": "x" * 1000},
                },
            ],
        })
    compact = _target_locate_evidence(candidate, rows)
    assert len(compact) == 3
    assert all(row["student_id"] == "student-002" for row in compact)
    assert all(set(row) == {
        "frame_index", "pts_seconds", "student_id", "locate_detection_index", "bbox"
    } for row in compact)
    assert compact[0]["pts_seconds"] == 1.0


def test_locate_materializes_exact_unique_qwen_anchor_frames(tmp_path: Path) -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent

    video = tmp_path / "anchors.avi"
    _write_video(video, frames=12)
    config = _temp_config(tmp_path)
    store = RunStore(tmp_path / "runs", "anchors", "test-anchors")
    candidates = [
        CandidateEvent("e1", "s0:1", "audit", 2, 0.2),
        CandidateEvent("e2", "s0:2", "audit", 2, 0.2),
        CandidateEvent("e3", "s0:1", "audit", 8, 0.8),
    ]
    rows, stage = _locate_anchor_frames(config, video, candidates, store)
    assert len(rows) == 2
    assert {row["frame_index"] for row in rows} == {2, 8}
    assert all(Path(row["path"]).is_file() for row in rows)
    assert all(row["anchor_source"] == "qwen_candidate" for row in rows)
    assert stage["requested_candidates"] == 3
    assert stage["unique_candidate_frames"] == 2
    assert stage["materialized_anchor_frames"] == 2


def test_interaction_signature_accepts_optional_hand_object_rows() -> None:
    from classroom_hybrid_ai.contracts import Detection, TrackObservation

    observation = TrackObservation(
        "student-1",
        0,
        0.0,
        Detection(
            "person",
            (0, 0, 100, 100),
            attributes={"objects": [{"label": "book", "bbox": [40, 40, 50, 50]}]},
        ),
    )
    signature = _interaction_signature(observation, [observation])
    assert signature["object:book:0"] == 1.0


def test_round_robin_changes_selected_track() -> None:
    from classroom_hybrid_ai.contracts import Detection, TrackObservation

    rows = [
        TrackObservation(str(index), 0, 0.0, Detection("person", (0, 0, 1, 1)))
        for index in range(3)
    ]
    selector = _RoundRobinTracks()
    assert [row.track_id for row in selector.select(rows, 1)] == ["0"]
    assert [row.track_id for row in selector.select(rows, 1)] == ["1"]


def test_fixed_eight_qwen_budget_uses_one_classification_call_per_candidate() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    total, discovery, classification = _qwen_budget(60.0 * 1000, config)
    assert total == 240
    assert discovery == 0
    assert classification == total


def test_open_vocabulary_budget_still_reserves_discovery_and_adjudication() -> None:
    original = load_config(ROOT / "configs" / "default.yaml")
    payload = deepcopy(original.payload)
    payload["qwen"]["action_mode"] = "open_vocabulary"
    payload["open_vocabulary"]["enabled"] = True
    payload["open_vocabulary"]["discovery_pass"] = True
    config = ProjectConfig(
        original.source_path, original.project_root, original.repository_root, payload
    )
    total, discovery, classification = _qwen_budget(60.0 * 1000, config)
    assert discovery >= 12
    assert discovery + classification == total - 2


def test_semantic_schema_failure_keeps_phrases_provisional() -> None:
    class BrokenSemanticQwen:
        def complete(self, *_args, **_kwargs):
            return {"unexpected": []}

    from classroom_hybrid_ai.semantic import SemanticCandidate

    candidate = SemanticCandidate(
        action_id="a_known", name_zh="低头看书", definition="可见的低头看书",
        attributes={}, source="registry",
    )
    decide = _semantic_batch_adjudicator(BrokenSemanticQwen())
    assert decide([("阅读资料", (candidate,))]) == {"阅读资料": None}


def test_student_catalog_stitches_short_same_epoch_tracklets(tmp_path: Path) -> None:
    path = tmp_path / "tracks.jsonl"
    path.write_text("\n".join([
        '{"track_id":"s0:1","frame_index":0,"pts_seconds":0.0,"detection":{"label":"person","bbox":[0,0,20,40],"source":"yolo"},"metadata":{"hsv_histogram":[1.0]}}',
        '{"track_id":"s0:2","frame_index":10,"pts_seconds":1.5,"detection":{"label":"person","bbox":[1,0,21,40],"source":"yolo"},"metadata":{"hsv_histogram":[1.0]}}',
        '{"track_id":"s1:3","frame_index":20,"pts_seconds":2.0,"detection":{"label":"person","bbox":[1,0,21,40],"source":"yolo"},"metadata":{"hsv_histogram":[1.0]}}',
    ]) + "\n", encoding="utf-8")
    mapping, rows, _ = _student_catalog(path)
    assert mapping["s0:1"] == mapping["s0:2"]
    assert mapping["s0:2"] != mapping["s1:3"]
    assert mapping["s0:1"].startswith("camera-default-segment-0-student-")
    assert rows[0]["identity_scope"] == "same_camera_continuous_segment"


def test_student_catalog_preserves_authoritative_locate_student_ids(tmp_path: Path) -> None:
    path = tmp_path / "locate-tracks.jsonl"
    path.write_text("\n".join([
        '{"track_id":"student-001","frame_index":0,"pts_seconds":0.0,"detection":{"label":"student","bbox":[0,0,20,40],"source":"locate_anything"}}',
        '{"track_id":"student-001","frame_index":10,"pts_seconds":1.0,"detection":{"label":"student","bbox":[1,0,21,40],"source":"locate_anything"}}',
        '{"track_id":"student-002","frame_index":0,"pts_seconds":0.0,"detection":{"label":"student","bbox":[30,0,50,40],"source":"locate_anything"}}',
    ]) + "\n", encoding="utf-8")
    mapping, rows, _ = _student_catalog(path)
    assert mapping == {"student-001": "student-001", "student-002": "student-002"}
    assert {row["student_id"] for row in rows} == {"student-001", "student-002"}
    assert all(row["identity_scope"] == "locate_assigned_same_camera_video_run" for row in rows)
    assert all(row["stitch_policy"]["applied"] is False for row in rows)


def test_new_track_emits_one_candidate_only_after_one_second(tmp_path: Path) -> None:
    video = tmp_path / "stable-track.avi"
    _write_video(video, frames=13)
    config = _temp_config(tmp_path)
    store = RunStore(tmp_path / "runs", "stable-track", "test-stable-track")
    candidates, _, _ = _track_pass(
        config, video, store, max_frames=None, tracker_factory=FakeTracker,
    )
    stable = [item for item in candidates if item.event_type == "stable_new_track_review"]
    assert len(stable) == 1
    assert stable[0].trigger_pts_seconds >= 1.0


def test_forced_audit_is_emitted_after_configured_long_segment(tmp_path: Path) -> None:
    video = tmp_path / "forced-audit.avi"
    _write_video(video, frames=13)
    original = _temp_config(tmp_path)
    payload = deepcopy(original.payload)
    payload["triggering"]["forced_audit_seconds"] = 1.0
    config = ProjectConfig(original.source_path, original.project_root, original.repository_root, payload)
    candidates, _, stage = _track_pass(
        config, video, RunStore(tmp_path / "runs", "forced-audit", "test-forced-audit"),
        max_frames=None, tracker_factory=FakeTracker,
    )
    assert any(item.event_type == "forced_120s_audit" for item in candidates)
    assert stage["forced_audit_candidates"] >= 1


def test_scene_change_emits_boundary_review_candidate(tmp_path: Path) -> None:
    video = tmp_path / "scene-boundary.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    assert writer.isOpened()
    for value in [0] * 4 + [255] * 4:
        writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    writer.release()
    original = _temp_config(tmp_path)
    payload = deepcopy(original.payload)
    payload["video"]["scene_change_threshold"] = 0.05
    config = ProjectConfig(original.source_path, original.project_root, original.repository_root, payload)
    candidates, _, stage = _track_pass(
        config, video, RunStore(tmp_path / "runs", "scene-boundary", "test-scene-boundary"),
        max_frames=None, tracker_factory=FakeTracker,
    )
    assert any(item.event_type == "scene_boundary_review" for item in candidates)
    assert stage["scene_boundary_candidates"] >= 1


def test_keypoint_motion_uses_half_second_window_and_k_of_m(tmp_path: Path) -> None:
    video = tmp_path / "motion-track.avi"
    _write_video(video, frames=10)
    config = _temp_config(tmp_path)

    class MotionTracker:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def track(self, _frame, persist=True):
            keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
            keypoints[0] = [10.0 + self.calls * 1.5, 20.0, 0.9]
            self.calls += 1
            return {"detections": [{
                "label": "person", "bbox": [0, 0, 50, 60], "score": 0.9,
                "class_id": 0, "track_id": "1", "source": "fake_yolo",
                "attributes": {"pose": {"keypoints": keypoints, "scores": [point[2] for point in keypoints]}},
            }]}

        def close(self) -> None:
            return None

    store = RunStore(tmp_path / "runs", "motion-track", "test-motion-track")
    candidates, _scenes, _stage = _track_pass(
        config, video, store, max_frames=None, tracker_factory=MotionTracker,
    )
    assert any(item.event_type == "keypoint_motion" for item in candidates)
    rows = list(iter_jsonl(store.run_dir / "artifacts/track_observations.jsonl"))
    assert any(row["anomaly_scores"].get("keypoint_motion", 0.0) >= 0.72 for row in rows)


def test_fair_candidate_wrapper_preserves_anomaly_priority() -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent

    ordinary = CandidateEvent(
        event_id="ordinary", track_id="s0:1", event_type="periodic_review",
        trigger_frame_index=0, trigger_pts_seconds=0.0, score=0.99,
    )
    anomaly = CandidateEvent(
        event_id="anomaly", track_id="s0:2", event_type="custom_anomaly",
        trigger_frame_index=900, trigger_pts_seconds=90.0, score=0.2,
        source="custom_anomaly", metadata={"priority": 100},
    )
    selected = _fair_candidates([ordinary, anomaly], limit=1, bucket_seconds=300.0)
    assert [item.event_id for item in selected] == ["anomaly"]


def test_nearby_cross_source_triggers_merge_with_priority_and_evidence() -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent

    ordinary = CandidateEvent(
        event_id="ordinary-near", track_id="s0:1", event_type="keypoint_motion",
        trigger_frame_index=10, trigger_pts_seconds=1.0, score=0.9,
        evidence_frame_indices=(8, 10), source="yolo26_pose",
    )
    boundary = CandidateEvent(
        event_id="boundary-near", track_id="s0:1", event_type="scene_boundary_review",
        trigger_frame_index=15, trigger_pts_seconds=1.5, score=None,
        evidence_frame_indices=(15,), source="periodic_audit", metadata={"priority": 50},
    )
    merged, groups = _merge_candidate_events([ordinary, boundary], window_seconds=1.0)
    assert groups == 1
    assert len(merged) == 1
    assert merged[0].event_id == "boundary-near"
    assert merged[0].metadata["merged_event_types"] == ["keypoint_motion", "scene_boundary_review"]
    assert merged[0].evidence_frame_indices == (8, 10, 15)


def test_fair_queue_consumes_merged_candidates_not_raw_triggers() -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent

    raw = [
        CandidateEvent(
            event_id="early-motion", track_id="s0:1", event_type="keypoint_motion",
            trigger_frame_index=0, trigger_pts_seconds=0.0, evidence_frame_indices=(0,),
        ),
        CandidateEvent(
            event_id="late-boundary", track_id="s0:1", event_type="scene_boundary_review",
            trigger_frame_index=5, trigger_pts_seconds=0.5, evidence_frame_indices=(5,),
            metadata={"priority": 50},
        ),
        CandidateEvent(
            event_id="other-student", track_id="s0:2", event_type="periodic_review",
            trigger_frame_index=600, trigger_pts_seconds=60.0, evidence_frame_indices=(600,),
        ),
    ]
    merged, selected, groups = _merge_and_select_candidates(
        raw, merge_window_seconds=1.0, limit=2, bucket_seconds=300.0,
    )
    assert groups == 1
    assert len(merged) == 2
    assert [item.event_id for item in selected] == ["late-boundary", "other-student"]


def test_qwen_oom_opens_stage_circuit_and_grays_remaining_candidates(tmp_path: Path) -> None:
    import cv2
    from classroom_hybrid_ai.contracts import CandidateEvent

    config = _temp_config(tmp_path)
    store = RunStore(tmp_path / "runs", "qwen-oom", "test-qwen-oom")
    evidence = []
    for event_id, track_id in (("e1", "s0:1"), ("e2", "s0:2")):
        event_dir = store.evidence_dir / event_id
        event_dir.mkdir(parents=True)
        items = []
        for ordinal in range(3):
            image_path = event_dir / f"subject_{ordinal}.jpg"
            assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
            items.append({"kind": "subject_crop", "path": str(image_path), "frame_index": ordinal})
        evidence.append({"event_id": event_id, "track_id": track_id, "status": "ok", "evidence": items})

    class OomQwen:
        def __init__(self) -> None:
            self.calls = 0

        def warmup(self) -> None:
            return None

        def review(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("CUDA out of memory")

        def close(self) -> None:
            return None

    adapter = OomQwen()
    candidates = [
        CandidateEvent(event_id="e1", track_id="s0:1", event_type="review", trigger_frame_index=0, trigger_pts_seconds=0.0),
        CandidateEvent(event_id="e2", track_id="s0:2", event_type="review", trigger_frame_index=1, trigger_pts_seconds=0.1),
    ]
    reviews, stage = _qwen_pass(
        config, candidates, evidence, store, skip=False, adapter=adapter, release_adapter=True,
    )
    assert adapter.calls == 1
    assert stage["oom_circuit_open"] is True
    assert [item.rationale for item in reviews] == ["RuntimeError:CUDA out of memory", "qwen_oom_circuit_open"]


def test_qwen_load_failure_circuits_semantic_and_classification_once(tmp_path: Path) -> None:
    video = tmp_path / "qwen-load-failure.avi"
    _write_video(video)
    config = _temp_config(tmp_path)

    class LoadFailQwen:
        warmup_calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False

        def warmup(self) -> None:
            type(self).warmup_calls += 1
            raise RuntimeError("synthetic_qwen_load_failure")

        def close(self) -> None:
            self.closed = True

    result = HybridPipeline(
        config,
        tracker_factory=FakeTracker,
        locate_factory=lambda **_kwargs: None,
        qwen_factory=LoadFailQwen,
    ).run(
        video,
        max_frames=5,
        skip_locate=True,
        require_preflight=False,
    )
    assert LoadFailQwen.warmup_calls == 1
    assert result.stages["qwen_discovery"]["status"] == "skipped"
    assert result.stages["qwen_classification"]["load_circuit_open"] is True
    assert result.stages["semantic_canonicalization"]["reason"] == "fixed_eight_taxonomy"


def test_qwen_schema_failure_gets_one_fresh_stateless_retry(tmp_path: Path) -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent

    config = _temp_config(tmp_path)
    store = RunStore(tmp_path / "runs", "qwen-retry", "test-qwen-retry")
    event_dir = store.evidence_dir / "e1"
    event_dir.mkdir(parents=True)
    items = []
    for ordinal in range(3):
        image_path = event_dir / f"subject_{ordinal}.jpg"
        assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
        items.append({"kind": "subject_crop", "path": str(image_path), "frame_index": ordinal})
    evidence = [{"event_id": "e1", "track_id": "s0:1", "status": "ok", "evidence": items}]

    class RetryQwen:
        def __init__(self) -> None:
            self.calls = 0

        def review(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"candidate_event_id": "e1"}
            return {
                "candidate_event_id": "e1", "decision": "confirmed",
                "reviewed_event_type": "review", "behavior": "低头写字",
                "rationale": "visible", "evidence_frame_indices": [0, 1, 2],
                "self_reported_confidence": None, "reviewer": "qwen3.5",
                "schema_valid": True, "secondary_behaviors": [], "posture": "sitting", "movement_state": "still",
                "head_direction": "down", "hand_action": "writing", "visible_objects": ["paper"],
                "interaction_target": None, "observability": "clear",
            }

        def close(self) -> None:
            return None

    candidate = CandidateEvent(
        event_id="e1", track_id="s0:1", event_type="review",
        trigger_frame_index=1, trigger_pts_seconds=0.1,
    )
    adapter = RetryQwen()
    reviews, stage = _qwen_pass(
        config, [candidate], evidence, store, skip=False, adapter=adapter,
        response_contract="discovery", release_adapter=True,
    )
    assert adapter.calls == 2
    assert stage["fresh_schema_retries"] == 1
    assert reviews[0].schema_valid is True
    assert reviews[0].metadata["fresh_retry_attempts"] == 1


def test_compact_classification_response_gets_request_fields_attached(tmp_path: Path) -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent

    config = _temp_config(tmp_path)
    store = RunStore(tmp_path / "runs", "qwen-compact", "test-qwen-compact")
    event_dir = store.evidence_dir / "e1"
    event_dir.mkdir(parents=True)
    items = []
    for ordinal in range(3):
        image_path = event_dir / f"subject_{ordinal}.jpg"
        assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
        items.append({"kind": "subject_crop", "path": str(image_path), "frame_index": ordinal})
    evidence = [{"event_id": "e1", "track_id": "s0:1", "status": "ok", "evidence": items}]

    class CompactQwen:
        def review(self, *_args, **kwargs):
            assert kwargs["compact_response"] is True
            return {
                "decision": "confirmed", "behavior": "a_known", "rationale": "可见",
                "evidence_frame_indices": [0, 1, 2], "observability": "clear",
                "reviewer": "qwen3.5", "schema_valid": True,
            }

        def close(self) -> None:
            return None

    candidate = CandidateEvent(
        event_id="e1", track_id="s0:1", event_type="review",
        trigger_frame_index=1, trigger_pts_seconds=0.1,
    )
    reviews, stage = _qwen_pass(
        config, [candidate], evidence, store, skip=False, adapter=CompactQwen(),
        response_contract="classification", expected_vocabulary_sha256="digest",
        allowed_action_ids={"a_known"}, release_adapter=True,
    )
    assert stage["compact_transport_responses"] == 1
    assert reviews[0].decision == "confirmed"
    assert json.loads((store.run_dir / "responses" / "e1.json").read_text(encoding="utf-8"))["vocabulary_sha256"] == "digest"


def test_qwen_progress_is_reported_only_after_each_response_is_durable(tmp_path: Path) -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent

    config = _temp_config(tmp_path)
    store = RunStore(tmp_path / "runs", "qwen-progress", "test-qwen-progress")
    evidence = []
    candidates = []
    for ordinal, event_id in enumerate(("e1", "e2")):
        event_dir = store.evidence_dir / event_id
        event_dir.mkdir(parents=True)
        items = []
        for frame_index in range(3):
            image_path = event_dir / f"subject_{frame_index}.jpg"
            assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
            items.append({"kind": "subject_crop", "path": str(image_path), "frame_index": frame_index})
        evidence.append({"event_id": event_id, "track_id": f"s0:{ordinal + 1}", "status": "ok", "evidence": items})
        candidates.append(CandidateEvent(
            event_id=event_id,
            track_id=f"s0:{ordinal + 1}",
            event_type="review",
            trigger_frame_index=ordinal,
            trigger_pts_seconds=float(ordinal),
        ))

    class CompactQwen:
        def review(self, event_id, *_args, **_kwargs):
            return {
                "decision": "uncertain", "behavior": "UNKNOWN", "rationale": "证据不足",
                "evidence_frame_indices": [0, 1, 2], "observability": "none",
                "reviewer": "qwen3.5", "schema_valid": True,
            }

        def close(self) -> None:
            return None

    progress = []

    def capture(done: int, total: int) -> None:
        expected_event = candidates[done - 1].event_id
        assert (store.run_dir / "responses" / f"{expected_event}.json").is_file()
        progress.append((done, total))

    reviews, _stage = _qwen_pass(
        config,
        candidates,
        evidence,
        store,
        skip=False,
        adapter=CompactQwen(),
        response_contract="classification",
        expected_vocabulary_sha256="digest",
        allowed_action_ids={"a_known"},
        release_adapter=True,
        resolve_conflicts=False,
        progress_callback=capture,
    )
    assert len(reviews) == 2
    assert progress == [(1, 2), (2, 2)]


def test_qwen_pass_stops_at_request_boundary_and_keeps_durable_responses(tmp_path: Path) -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent

    config = _temp_config(tmp_path)
    store = RunStore(tmp_path / "runs", "qwen-stop", "test-qwen-stop")
    evidence = []
    candidates = []
    for ordinal, event_id in enumerate(("e1", "e2")):
        event_dir = store.evidence_dir / event_id
        event_dir.mkdir(parents=True)
        items = []
        for frame_index in range(3):
            image_path = event_dir / f"subject_{frame_index}.jpg"
            assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
            items.append({"kind": "subject_crop", "path": str(image_path), "frame_index": frame_index})
        evidence.append({"event_id": event_id, "track_id": f"s0:{ordinal + 1}", "status": "ok", "evidence": items})
        candidates.append(CandidateEvent(
            event_id=event_id,
            track_id=f"s0:{ordinal + 1}",
            event_type="review",
            trigger_frame_index=ordinal,
            trigger_pts_seconds=float(ordinal),
        ))

    class StopQwen:
        calls = 0

        def review(self, *_args, **_kwargs):
            type(self).calls += 1
            return {
                "decision": "uncertain", "behavior": "UNKNOWN", "rationale": "证据不足",
                "evidence_frame_indices": [0, 1, 2], "observability": "none",
                "reviewer": "qwen3.5", "schema_valid": True,
            }

        def close(self) -> None:
            return None

    reviews, stats = _qwen_pass(
        config,
        candidates,
        evidence,
        store,
        skip=False,
        adapter=StopQwen(),
        response_contract="classification",
        expected_vocabulary_sha256="digest",
        allowed_action_ids={"a_known"},
        release_adapter=True,
        resolve_conflicts=False,
        stop_check=lambda: StopQwen.calls >= 1,
    )
    assert StopQwen.calls == 1
    assert stats["interruption_reason"] == "stop_requested"
    assert stats["executed_requests"] == 1
    assert len(reviews) == 1
    assert (store.run_dir / "responses" / "e1.json").is_file()
    assert not (store.run_dir / "responses" / "e2.json").exists()


def test_run_s5_stop_file_releases_model_and_requires_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from classroom_hybrid_ai.contracts import CandidateEvent
    from classroom_hybrid_ai.research_backend import ResearchBackend
    import classroom_hybrid_ai.research_stages as stages_module

    config = _temp_config(tmp_path)
    config.payload.setdefault("qwen", {})["stop_file"] = str(tmp_path / "qwen.stop")
    profile = ResearchBackend(config).profile()
    store = RunStore(tmp_path / "stage_runs", "sample", "config", unique_run=True)
    candidate = CandidateEvent(
        event_id="event-1", track_id="student-1", event_type="review",
        trigger_frame_index=0, trigger_pts_seconds=0.0,
    )
    store.write_jsonl("artifacts/candidates.selected.jsonl", [candidate])
    store.write_jsonl("artifacts/qwen_crop_queue.jsonl", [{"candidate_event_id": candidate.event_id, "status": "ready"}])
    store.write_jsonl("artifacts/locate_results.jsonl", [])
    store.write_jsonl("evidence/index.jsonl", [{"event_id": candidate.event_id}])
    (tmp_path / "qwen.stop").write_text("stop", encoding="utf-8")
    spec = stages_module.ResearchStageSpec(
        config=config, store=store, source=tmp_path / "unused.avi", video={}, profile=profile,
        mode="production_adaptive", max_frames=None, skip_qwen=False,
        allow_incomplete_preflight=False, experiment_digest="experiment", input_sha256="input",
        config_sha256="config", code_sha256="code", locate_model_sha256="locate", qwen_model_sha256="qwen",
    )
    adapter = SimpleNamespace(pid=4321, exitcode=0)
    captured: dict = {}
    release_calls: list = []

    def fake_qwen_pass(*_args, **kwargs):
        captured["stop_check"] = kwargs.get("stop_check")
        return [], {"status": "ok", "interruption_reason": "stop_requested", "executed_requests": 0}

    def fake_release(model):
        release_calls.append(model)
        return {"released": True, "close_error": None, "cuda_after_release": {}}

    monkeypatch.setattr(stages_module, "_make_qwen_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(stages_module, "_qwen_pass", fake_qwen_pass)
    monkeypatch.setattr(
        stages_module, "_profile_vocabulary",
        lambda _spec: (SimpleNamespace(sha256="registry"), [], "vocabulary"),
    )
    monkeypatch.setattr(
        stages_module, "GpuLease",
        lambda _path: SimpleNamespace(acquire=lambda: None, release=lambda: None),
    )
    monkeypatch.setattr(stages_module, "release_model", fake_release)
    executor = stages_module.ResearchStageExecutor(spec)
    progress: list = []
    executor.set_progress_reporter(
        lambda stage, done, total, message: progress.append((stage, done, total, message))
    )

    with pytest.raises(stages_module.StageExecutionError, match="stopped_by_user"):
        executor._run_s5(input_fingerprint_sha256="fingerprint", partial_outputs_reusable=False)

    assert release_calls == [adapter]
    assert captured["stop_check"] is not None
    assert captured["stop_check"]() is True
    receipt = json.loads(
        (store.run_dir / "diagnostics" / "qwen_worker_receipt.v1.json").read_text(encoding="utf-8")
    )
    assert receipt["released"] is True
    assert receipt["worker_pid"] == 4321
    assert not (store.run_dir / "stage_receipts" / "S5.json").exists()
    messages = [message for _stage, _done, _total, message in progress]
    assert any("正在释放 Qwen 模型" in message for message in messages)
    assert any("Qwen 模型已释放" in message for message in messages)


def test_classification_conflict_gets_one_independent_recheck(tmp_path: Path) -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent, ReviewDecision

    config = _temp_config(tmp_path)
    store = RunStore(tmp_path / "runs", "qwen-conflict", "test-qwen-conflict")
    evidence = []
    for event_id in ("e1", "e2"):
        event_dir = store.evidence_dir / event_id
        event_dir.mkdir(parents=True)
        items = []
        for ordinal in range(3):
            image_path = event_dir / f"subject_{ordinal}.jpg"
            assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
            items.append({"kind": "subject_crop", "path": str(image_path), "frame_index": ordinal})
        context_path = event_dir / "context.jpg"
        assert cv2.imwrite(str(context_path), np.zeros((16, 16, 3), dtype=np.uint8))
        items.append({"kind": "annotated_context", "path": str(context_path), "frame_index": 1})
        evidence.append({"event_id": event_id, "track_id": "s0:1", "status": "ok", "evidence": items})

    candidates = [
        CandidateEvent("e1", "s0:1", "review", 1, 0.0, evidence_frame_indices=(0, 1, 2)),
        CandidateEvent("e2", "s0:1", "review", 2, 0.5, evidence_frame_indices=(0, 1, 2)),
    ]
    reviews = [
        ReviewDecision("e1", "confirmed", "review", "a_write", "visible", (0, 1), schema_valid=True),
        ReviewDecision("e2", "confirmed", "review", "a_read", "visible", (0, 1), schema_valid=True),
    ]

    class RecheckQwen:
        calls = 0

        def review(self, *_args, **kwargs):
            type(self).calls += 1
            assert kwargs["compact_response"] is True
            return {
                "decision": "confirmed", "behavior": "a_write", "rationale": "可见",
                "evidence_frame_indices": [0, 1, 2], "observability": "clear",
                "reviewer": "qwen3.5", "schema_valid": True,
            }

    resolved, stats = _resolve_classification_conflicts(
        config,
        candidates,
        evidence,
        reviews,
        store,
        adapter=RecheckQwen(),
        prompt="classification",
        expected_vocabulary_sha256="digest",
        allowed_action_ids={"a_write", "a_read"},
        locate_rows=(),
        call_budget=__import__("classroom_hybrid_ai.pipeline", fromlist=["_QwenCallBudget"])._QwenCallBudget(1),
    )
    assert stats == {"conflict_groups": 1, "conflict_rechecks": 1, "conflict_unresolved": 0}
    assert RecheckQwen.calls == 1
    assert {review.behavior for review in resolved if review.decision == "confirmed"} == {"a_write"}
    assert any(review.decision == "uncertain" and "conflict" in review.rationale for review in resolved)


def test_recycled_raw_id_after_two_seconds_becomes_new_tracklet(tmp_path: Path) -> None:
    video = tmp_path / "recycled-track.avi"
    _write_video(video, frames=35)
    config = _temp_config(tmp_path)

    class RecycledIdTracker:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def track(self, _frame, persist=True):
            # The same raw ByteTrack ID reappears after a 3-second absence.
            call_index = getattr(self, "calls", 0)
            self.calls = call_index + 1
            if call_index not in {0, 30}:
                return {"detections": []}
            return {"detections": [{
                "label": "person", "bbox": [8, 8, 40, 55], "score": 0.9,
                "class_id": 0, "track_id": "1", "source": "fake_yolo", "attributes": {},
            }]}

        def close(self) -> None:
            return None

    store = RunStore(tmp_path / "runs", "recycled-track", "test-recycled-track")
    candidates, _, _ = _track_pass(config, video, store, max_frames=None, tracker_factory=RecycledIdTracker)
    rows = list(iter_jsonl(store.run_dir / "artifacts/track_observations.jsonl"))
    assert [row["track_id"] for row in rows] == ["s0:1~0", "s0:1~1"]
    assert any(item.event_type == "track_reconnect_review" for item in candidates)


def test_custom_anomaly_sparse_hits_are_reset_by_explicit_zero_and_gap(tmp_path: Path) -> None:
    video = tmp_path / "sparse-anomaly.avi"
    _write_video(video, frames=35)
    config = _temp_config(tmp_path)
    payload = deepcopy(config.payload)
    checkpoint = tmp_path / "custom-anomaly.pt"
    checkpoint.write_bytes(b"local-test-checkpoint")
    payload["yolo"]["anomaly_model"] = str(checkpoint)
    payload["yolo"]["anomaly_class_map"] = {0: "classroom_disruption"}
    config = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    store = RunStore(tmp_path / "runs", "sparse-anomaly", "test-sparse-anomaly")
    _candidates, _scenes, tracking_stage = _track_pass(
        config, video, store, max_frames=None, tracker_factory=FakeTracker,
    )

    class SparseAnomaly:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def predict(self, _frame):
            index = self.calls
            self.calls += 1
            if index in {0, 2, 4, 6}:
                return {"detections": [{"label": "classroom_disruption", "bbox": [8, 8, 40, 55], "score": 0.99}]}
            return {"detections": []}

        def close(self) -> None:
            return None

    anomaly_candidates, stage = _custom_anomaly_pass(
        config,
        video,
        store.run_dir / "artifacts/track_observations.jsonl",
        store,
        last_frame_index=int(tracking_stage["last_frame_index"]),
        candidate_limit=500,
        adapter_factory=SparseAnomaly,
    )
    assert anomaly_candidates == []
    assert stage["detections"] == 4


def test_locate_timeout_still_requires_remote_unload(tmp_path: Path, monkeypatch) -> None:
    config = _temp_config(tmp_path)
    payload = deepcopy(config.payload)
    payload["locate_anything"]["enabled"] = True
    payload["locate_anything"]["required"] = False
    config = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    image_path = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
    calls = []

    class TimeoutLocate:
        def __init__(self, **_kwargs) -> None:
            pass

        def detect(self, *_args, **_kwargs):
            raise TimeoutError("first_request_timeout")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "classroom_hybrid_ai.pipeline._post_json",
        lambda url, timeout: calls.append((url, timeout)) or {"status": "unloaded", "loaded": False},
    )
    store = RunStore(tmp_path / "runs", "locate", "timeout-unload")
    result = _locate_pass(
        config, [{"path": str(image_path), "frame_index": 0, "pts_seconds": 0.0}],
        store, skip=False, adapter_factory=TimeoutLocate,
    )
    assert result["status"] == "warning"
    assert len(calls) == 1


def test_locate_missing_wsl_service_is_optional_warning_without_unload(tmp_path: Path, monkeypatch) -> None:
    config = _temp_config(tmp_path)
    payload = deepcopy(config.payload)
    payload["locate_anything"]["enabled"] = True
    payload["locate_anything"]["required"] = False
    config = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    image_path = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
    unload_calls = []

    class MissingWslLocate:
        def __init__(self, **_kwargs) -> None:
            self.closed = False

        def health(self):
            raise URLError(ConnectionRefusedError(10061, "No connection could be made"))

        def detect(self, *_args, **_kwargs):
            raise AssertionError("detect must not run when the health endpoint is unavailable")

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "classroom_hybrid_ai.pipeline._post_json",
        lambda *_args, **_kwargs: unload_calls.append(True),
    )
    result = _locate_pass(
        config, [{"path": str(image_path), "frame_index": 0, "pts_seconds": 0.0}],
        RunStore(tmp_path / "runs", "locate", "missing-wsl"),
        skip=False, adapter_factory=MissingWslLocate,
    )
    assert result["status"] == "warning"
    assert result["reason"] == "service_unavailable"
    assert result["requests"] == 0
    assert result["remote_unload"] is None
    assert unload_calls == []


def test_locate_health_timeout_is_service_unavailable_warning(tmp_path: Path, monkeypatch) -> None:
    config = _temp_config(tmp_path)
    payload = deepcopy(config.payload)
    payload["locate_anything"]["enabled"] = True
    payload["locate_anything"]["required"] = False
    config = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    image_path = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))

    class TimedOutLocate:
        def __init__(self, **_kwargs) -> None:
            pass

        def health(self):
            raise URLError(TimeoutError("health timed out"))

        def detect(self, *_args, **_kwargs):
            raise AssertionError("detect must not run after a failed health probe")

        def close(self) -> None:
            return None

    monkeypatch.setattr("classroom_hybrid_ai.pipeline._post_json", lambda *_args, **_kwargs: pytest.fail("no unload"))
    result = _locate_pass(
        config,
        [{"path": str(image_path), "frame_index": 0, "pts_seconds": 0.0}],
        RunStore(tmp_path / "runs", "locate", "health-timeout"),
        skip=False,
        adapter_factory=TimedOutLocate,
    )
    assert result["status"] == "warning"
    assert result["reason"] == "service_unavailable"
    assert result["requests"] == 0
    assert result["remote_unload"] is None


def test_locate_queries_students_first_and_assigns_per_image_indices(tmp_path: Path) -> None:
    config = _temp_config(tmp_path)
    payload = deepcopy(config.payload)
    payload["locate_anything"].update({
        "enabled": True,
        "backend": "local",
        "prompts": ["student", "book", "teacher"],
    })
    config = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    image_path = tmp_path / "scene-students.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
    calls: list[list[str]] = []

    class RecordingLocate:
        def __init__(self, **_kwargs) -> None:
            pass

        def detect(self, _image, labels, **_kwargs):
            calls.append(list(labels))
            label = labels[0]
            return {
                "raw_answer": f"<ref>{label}</ref><box><0><0><500><500></box>",
                "detections": [{"label": label, "bbox": [0, 0, 8, 8]}],
            }

        def close(self) -> None:
            return None

    store = RunStore(tmp_path / "runs", "locate", "student-first")
    progress: list[tuple[int, int]] = []
    result = _locate_pass(
        config,
        [{"path": str(image_path), "frame_index": 0, "pts_seconds": 0.0}],
        store,
        skip=False,
        adapter_factory=RecordingLocate,
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    rows = list(iter_jsonl(store.run_dir / "artifacts/locate_results.jsonl"))
    assert calls == [["student"], ["book", "teacher"]]
    assert result["requests"] == 2
    assert [item["detection_index"] for item in rows[0]["detections"]] == [1, 2]
    assert rows[0]["metadata"]["mode"] == "student_first_spatial_localization"
    assert progress == [(1, 1)]


def test_locate_unload_without_ack_stops_pipeline_before_qwen(tmp_path: Path, monkeypatch) -> None:
    config = _temp_config(tmp_path)
    payload = deepcopy(config.payload)
    payload["locate_anything"]["enabled"] = True
    config = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    image_path = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))

    class SuccessfulLocate:
        def __init__(self, **_kwargs) -> None:
            pass

        def detect(self, *_args, **_kwargs):
            return {"detections": []}

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "classroom_hybrid_ai.pipeline._post_json",
        lambda _url, _timeout: {"status": "ok", "loaded": True},
    )
    with pytest.raises(PipelineError, match="locate_remote_unload_failed"):
        _locate_pass(
            config, [{"path": str(image_path), "frame_index": 0, "pts_seconds": 0.0}],
            RunStore(tmp_path / "runs", "locate", "missing-ack"),
            skip=False, adapter_factory=SuccessfulLocate,
        )


def test_locate_audit_write_failure_still_attempts_remote_unload(tmp_path: Path, monkeypatch) -> None:
    config = _temp_config(tmp_path)
    payload = deepcopy(config.payload)
    payload["locate_anything"]["enabled"] = True
    config = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    image_path = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
    calls: list[bool] = []

    class SuccessfulLocate:
        def __init__(self, **_kwargs) -> None:
            pass

        def detect(self, *_args, **_kwargs):
            return {"detections": []}

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "classroom_hybrid_ai.pipeline._post_json",
        lambda *_args, **_kwargs: calls.append(True) or {"status": "unloaded", "loaded": False},
    )
    store = RunStore(tmp_path / "runs", "locate", "write-failure")
    original_write = store.write_jsonl

    def fail_write(*_args, **_kwargs):
        raise OSError("synthetic_audit_write_failure")

    store.write_jsonl = fail_write  # type: ignore[method-assign]
    with pytest.raises(OSError, match="synthetic_audit_write_failure"):
        _locate_pass(
            config, [{"path": str(image_path), "frame_index": 0, "pts_seconds": 0.0}],
            store, skip=False, adapter_factory=SuccessfulLocate,
        )
    assert calls == [True]
    store.write_jsonl = original_write  # type: ignore[method-assign]


def test_classification_cannot_emit_action_outside_frozen_vocabulary() -> None:
    response = {
        "candidate_event_id": "event-1", "decision": "confirmed",
        "reviewed_event_type": "review", "behavior": "foreign-action",
        "rationale": "visible", "evidence_frame_indices": [3, 4],
        "self_reported_confidence": None, "vocabulary_sha256": "abc",
        "observability": "clear", "reviewer": "qwen3.5", "schema_valid": True,
    }
    with pytest.raises(PipelineError, match="frozen_vocabulary"):
        _validate_classification_response(response, (3, 4), allowed_action_ids={"a_known"})
    response["behavior"] = "UNKNOWN"
    _validate_classification_response(response, (3, 4), allowed_action_ids={"a_known"})


def test_small_pipeline_run_uses_streamed_observations(tmp_path: Path) -> None:
    video = tmp_path / "tiny.avi"
    _write_video(video)
    result = HybridPipeline(_temp_config(tmp_path), tracker_factory=FakeTracker).run(
        video,
        max_frames=5,
        skip_locate=True,
        skip_qwen=True,
        require_preflight=False,
    )
    run_dir = Path(result.run_dir)
    assert result.counts["observations"] == 5
    assert result.counts["candidates"] == 1
    assert result.counts["uncertain"] == 1
    assert (run_dir / "artifacts" / "track_observations.jsonl").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "students.jsonl").is_file()
    assert (run_dir / "session_vocabulary.v1.json").is_file()
    assert (run_dir / "timeline.v1.json").is_file()
    intervals = (run_dir / "action_intervals.jsonl").read_text(encoding="utf-8")
    assert '"action_id":"unobserved"' in intervals
    timeline = __import__("json").loads((run_dir / "timeline.v1.json").read_text(encoding="utf-8"))
    assert timeline["duration_seconds"] == pytest.approx(0.8)
    assert any(float(row["end_pts_seconds"]) == pytest.approx(0.8) for row in map(__import__("json").loads, intervals.splitlines()))


def test_locate_first_pipeline_saves_ids_and_orders_stateless_crop_queue(tmp_path: Path) -> None:
    video = tmp_path / "locate-first.avi"
    _write_video(video, frames=5)
    original = _temp_config(tmp_path)
    payload = deepcopy(original.payload)
    payload["pipeline"]["primary_detector"] = "locate_anything"
    payload["locate_anything"].update({
        "enabled": True,
        "required": True,
        "backend": "local",
        "prompts": ["student"],
    })
    payload["locate_first"]["screenshot_frame_stride"] = 2
    config = ProjectConfig(
        original.source_path, original.project_root, original.repository_root, payload
    )

    class FakeLocate:
        def __init__(self, **_kwargs) -> None:
            self.closed = False

        def detect(self, *_args, **_kwargs):
            return {
                "raw_answer": "two students",
                "detections": [
                    {"label": "student", "bbox": [8, 8, 28, 55], "detection_index": 1},
                    {"label": "student", "bbox": [34, 8, 58, 55], "detection_index": 2},
                ],
            }

        def close(self) -> None:
            self.closed = True

    def yolo_must_not_load(*_args, **_kwargs):
        raise AssertionError("YOLO must not load in locate-first mode")

    result = HybridPipeline(
        config,
        tracker_factory=yolo_must_not_load,
        anomaly_factory=yolo_must_not_load,
        locate_factory=FakeLocate,
    ).run(
        video,
        max_frames=5,
        skip_qwen=True,
        require_preflight=False,
    )
    run_dir = Path(result.run_dir)
    boxes = list(iter_jsonl(run_dir / "artifacts" / "locate_student_boxes.jsonl"))
    queue = list(iter_jsonl(run_dir / "artifacts" / "qwen_crop_queue.jsonl"))
    assert [(row["frame_index"], row["student_id"]) for row in boxes] == [
        (0, "student-001"), (0, "student-002"),
        (2, "student-001"), (2, "student-002"),
        (4, "student-001"), (4, "student-002"),
    ]
    assert [(row["frame_index"], row["student_id"]) for row in queue] == [
        (0, "student-001"), (0, "student-002"),
        (2, "student-001"), (2, "student-002"),
        (4, "student-001"), (4, "student-002"),
    ]
    assert all(row["request_is_stateless"] for row in queue)
    assert all(row["messages_in_request"] == 1 for row in queue)
    assert all(row["history_messages"] == 0 for row in queue)
    assert all(row["subject_crop_paths"] for row in queue)
    assert all(row["previous_student_context_included"] is False for row in queue)
    assert all(Path(row["annotated_path"]).is_file() for row in boxes)
    assert result.stages["yolo_tracking"]["yolo_model_loaded"] is False
    assert result.stages["student_crop_router"]["one_student_per_qwen_request"] is True
    assert result.stages["locate_visual_audit"]["rendered_screenshots"] == 3


def test_pipeline_surfaces_missing_wsl_as_optional_warning(tmp_path: Path) -> None:
    video = tmp_path / "missing-wsl.avi"
    _write_video(video)
    original = _temp_config(tmp_path)
    payload = deepcopy(original.payload)
    payload["locate_anything"]["enabled"] = True
    payload["locate_anything"]["required"] = False
    config = ProjectConfig(original.source_path, original.project_root, original.repository_root, payload)

    class MissingWsl:
        def __init__(self, **_kwargs) -> None:
            pass

        def health(self):
            raise URLError(ConnectionRefusedError(10061, "No connection could be made"))

        def close(self) -> None:
            return None

    result = HybridPipeline(
        config, tracker_factory=FakeTracker, locate_factory=MissingWsl,
    ).run(video, max_frames=5, skip_qwen=True, require_preflight=False)
    assert result.status == "warning"
    assert result.warnings == ("locate_anything_service_unavailable",)
    assert result.stages["locate_anything"]["requests"] == 0
    assert result.stages["locate_anything"]["remote_unload"] is None


def test_resume_reuses_yolo_checkpoint_without_retracking(tmp_path: Path) -> None:
    video = tmp_path / "resumable.avi"
    _write_video(video)
    config = _temp_config(tmp_path)
    first = HybridPipeline(config, tracker_factory=FakeTracker).run(
        video, max_frames=5, skip_locate=True, skip_qwen=True, require_preflight=False,
    )

    class FailingTracker:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("resume must not construct a tracker")

    second = HybridPipeline(config, tracker_factory=FailingTracker).run(
        video, max_frames=5, skip_locate=True, skip_qwen=True, resume=True,
        require_preflight=False,
    )
    assert second.run_dir == first.run_dir
    assert second.stages["resume"]["status"] == "resumed"
