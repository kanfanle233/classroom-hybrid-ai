from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from classroom_hybrid_ai.adapters.yolo import YoloTrackerAdapter
from classroom_hybrid_ai.hashing import sha256_file
from classroom_hybrid_ai.localization import (
    LocalizationError, associate_localization_rows, build_difficulty_queue,
    merge_localization_rows, run_yolo_screenshots, render_localization_annotations,
)
from classroom_hybrid_ai.locate_tracking import associate_locate_rows
from classroom_hybrid_ai.yolo_worker import YoloProcessClient, YoloWorkerError


def person(x=0.0, score=0.9, source="yolo", label="person", **extra):
    return {"label": label, "bbox": [x, 0, x + 30, 60], "score": score,
            "source": source, "class_id": 0, **extra}


def frame(index, detections=None, *, pts=None, source="yolo", **extra):
    return {"frame_index": index, "pts_seconds": float(index if pts is None else pts),
            "path": f"frame_{index}.jpg", "status": "ok", "source": source,
            "detections": [person()] if detections is None else detections, **extra}


def fixtures(tmp_path, count=3):
    model = tmp_path / "yolo26n-pose.pt"
    model.write_bytes(b"local-pinned-fake-checkpoint")
    scenes = []
    for index in range(count):
        path = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (32, 64), (index * 30, 0, 0)).save(path)
        scenes.append({**frame(index), "path": str(path)})
    return model, scenes


class FakeDetector:
    def __init__(self, *, fail_at=None, bad=None, **kwargs):
        self.calls = []
        self.closed = False
        self.fail_at = fail_at
        self.bad = bad
        self.last_performance = {}

    def predict_batch(self, images, **kwargs):
        assert not self.closed
        self.calls.append((list(images), kwargs))
        if self.fail_at == len(self.calls):
            raise RuntimeError("injected_request_failure")
        return [{"detections": [self.bad or person()]} for _ in images]

    def close(self):
        self.closed = True


def runner(tmp_path, *, count=3, detector=None, **overrides):
    model, scenes = fixtures(tmp_path, count)
    detector = detector or FakeDetector()
    kwargs = {"model_path": model, "cache_dir": tmp_path / "cache", "code_fingerprint": "source-sha",
              "adapter_factory": lambda **kw: detector, **overrides}
    return model, scenes, detector, kwargs


def test_screenshot_predict_is_per_image_and_never_calls_tracker(tmp_path, monkeypatch):
    model_path = tmp_path / "pose.pt"
    model_path.write_bytes(b"stub")
    calls = []
    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    monkeypatch.setenv("YOLO_OFFLINE", "false")
    settings = tmp_path / "workspace-settings"

    class Model:
        def predict(self, images, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=[[i, 0, i + 1, 2]],
                conf=[.9], cls=[0], id=[99]), keypoints=None, names={0: "person"}) for i in range(len(images))]
        def track(self, *a, **k):
            raise AssertionError("dense tracker must not be invoked")

    def factory(_):
        assert os.environ["YOLO_AUTOINSTALL"] == "false"
        assert os.environ["YOLO_OFFLINE"] == "true"
        assert Path(os.environ["YOLO_CONFIG_DIR"]) == settings
        return Model()

    adapter = YoloTrackerAdapter(model_path, config_dir=settings, model_factory=factory,
                                 default_kwargs={"tracker": "bad", "persist": True})
    outputs = adapter.predict_batch(["a", "b"], imgsz=960, conf=.2)
    assert len(outputs) == 2
    assert outputs[0]["detections"][0]["bbox"] != outputs[1]["detections"][0]["bbox"]
    assert outputs[0]["detections"][0]["raw_track_id"] == "99"
    assert "track_id" not in outputs[0]["detections"][0]
    assert "persist" not in calls[0] and "tracker" not in calls[0]
    assert calls[0]["stream"] is False


def test_per_image_cache_resume_and_true_close_before_return(tmp_path):
    _, scenes, detector, kwargs = runner(tmp_path)
    rows, stage = run_yolo_screenshots(scenes, **kwargs)
    assert detector.closed and stage["inference_calls"] == 3
    assert all(r["source"] == "yolo" and not r["detections"][0]["student_confirmed"] for r in rows)
    assert (rows[0]["width"], rows[0]["height"]) == (32, 64)
    kwargs["adapter_factory"] = lambda **kw: pytest.fail("cached run must not create model")
    cached, cached_stage = run_yolo_screenshots(scenes, **kwargs)
    assert cached_stage["inference_calls"] == 0 and cached_stage["cache_hits"] == 3
    assert [r["detections"] for r in rows] == [r["detections"] for r in cached]


def test_successful_requests_survive_interruption(tmp_path):
    _, scenes, first, kwargs = runner(tmp_path, detector=FakeDetector(fail_at=2))
    with pytest.raises(RuntimeError, match="injected_request_failure"):
        run_yolo_screenshots(scenes, **kwargs)
    assert first.closed
    second = FakeDetector()
    kwargs["adapter_factory"] = lambda **kw: second
    _, stage = run_yolo_screenshots(scenes, **kwargs)
    assert stage["cache_hits"] == 1 and len(second.calls) == 2 and second.closed
    assert json.loads((tmp_path / "cache" / "last_failure.v1.json").read_text())["successful_screenshots"] == 1


@pytest.mark.parametrize("change", ["image", "model", "code", "parameters", "tampered_result"])
def test_cache_invalidates_changed_inputs_and_corruption(tmp_path, change):
    model, scenes, _, kwargs = runner(tmp_path, count=1)
    run_yolo_screenshots(scenes, **kwargs)
    if change == "image":
        Image.new("RGB", (32, 64), "blue").save(scenes[0]["path"])
    elif change == "model":
        model.write_bytes(b"different-weights")
    elif change == "code":
        kwargs["code_fingerprint"] = "different-source"
    elif change == "parameters":
        kwargs["conf"] = .21
    else:
        path = next(path for path in (tmp_path / "cache").glob("*.json") if len(path.stem) == 64)
        row = json.loads(path.read_text())
        row["result"]["detections"] = []
        path.write_text(json.dumps(row))
    detector = FakeDetector()
    kwargs["adapter_factory"] = lambda **kw: detector
    _, stage = run_yolo_screenshots(scenes, **kwargs)
    assert stage["cache_hits"] == 0 and len(detector.calls) == 1


def test_declared_model_sha_is_verified(tmp_path):
    _, scenes, detector, kwargs = runner(tmp_path, count=1)
    with pytest.raises(LocalizationError, match="model_sha_mismatch"):
        run_yolo_screenshots(scenes, model_sha256="0" * 64, **kwargs)
    assert not detector.calls


@pytest.mark.parametrize("batch_size,expected", [(1, 5), (2, 3), (4, 2)])
def test_batching_preserves_every_screenshot(tmp_path, batch_size, expected):
    _, scenes, detector, kwargs = runner(tmp_path, count=5)
    rows, stage = run_yolo_screenshots(scenes, batch_size=batch_size, **kwargs)
    assert len(detector.calls) == expected and stage["inferred_screenshots"] == 5
    assert [row["frame_index"] for row in rows] == list(range(5))
    assert all(len(images) == params["batch"] for images, params in detector.calls)


@pytest.mark.parametrize("bad", [person(score=float("nan")), person(score=None), person(bbox=[0, 0, 0, 5])])
def test_invalid_model_result_cannot_be_saved_as_empty_success(tmp_path, bad):
    _, scenes, detector, kwargs = runner(tmp_path, count=1, detector=FakeDetector(bad=bad))
    with pytest.raises(LocalizationError):
        run_yolo_screenshots(scenes, **kwargs)
    assert detector.closed
    assert not any(len(path.stem) == 64 for path in (tmp_path / "cache").glob("*.json"))


def test_valid_empty_model_results_are_success_and_cached(tmp_path):
    _, scenes, detector, kwargs = runner(tmp_path, count=1)
    detector.predict_batch = lambda images, **kw: [{"detections": []} for _ in images]
    rows, stage = run_yolo_screenshots(scenes, **kwargs)
    assert rows[0]["status"] == "ok" and rows[0]["detections"] == []
    assert stage["status"] == "ok" and stage["detections"] == 0


def test_difficulty_anchors_and_scene_changes_are_deterministic():
    rows = [frame(i, pts=t) for i, t in enumerate([0, 10, 29, 30, 40, 61])]
    rows += [frame(6, pts=62, scene_id="new")]
    queue, audit = build_difficulty_queue(rows, association_kwargs={"max_gap_seconds": 100})
    assert [row["frame_index"] for row in queue] == [0, 3, 5, 6]
    assert "periodic_audit_anchor" in queue[1]["difficulty_reasons"]
    assert "scene_first_screenshot" in queue[-1]["difficulty_reasons"]
    assert audit["truncated"] is False


def test_difficulty_low_confidence_empty_and_broken_tracks_are_not_truncated():
    rows = [frame(0), frame(1, [person(score=.27)]), frame(2, []), frame(3), frame(4, [person(0), person(100)])]
    queue, audit = build_difficulty_queue(rows)
    assert len(queue) == 5
    reasons = {row["frame_index"]: row["reasons"] for row in audit["rows"]}
    assert "low_person_confidence" in reasons[1]
    assert "zero_person_detections" in reasons[2] and "broken_provisional_track" in reasons[2]
    assert "unmatched_previous_screenshot" in reasons[3]
    assert "new_provisional_track" in reasons[4]


def test_difficulty_ambiguous_people_and_long_gaps_trigger_locate():
    rows = [frame(0), frame(1, [person(0), person(0)]), frame(2, pts=10)]
    _, audit = build_difficulty_queue(rows)
    assert "association_ambiguous" in audit["rows"][1]["reasons"]
    assert "broken_provisional_track" in audit["rows"][2]["reasons"]


def test_duplicate_screenshot_has_one_queue_entry_and_conflicts_fail():
    row = frame(0, [person(score=.1)])
    queue, _ = build_difficulty_queue([row, dict(row)])
    assert len(queue) == 1
    with pytest.raises(LocalizationError, match="conflicting_localization_frame"):
        build_difficulty_queue([row, {**row, "path": "other.jpg"}])


def test_hard_frame_uses_complete_locate_result_with_real_source():
    yolo = [frame(0, [person(0), person(100)])]
    locate = [frame(0, [person(0, source="locate_anything", label="student")], source="locate_anything")]
    result = merge_localization_rows(yolo, locate, yolo, route="yolo_locate_hybrid")
    assert len(result[0]["detections"]) == 1
    assert result[0]["source"] == result[0]["detections"][0]["source"] == "locate_anything"
    assert merge_localization_rows(yolo, [frame(0, [], source="locate_anything")], yolo, route="yolo_locate_hybrid")[0]["detections"] == []


@pytest.mark.parametrize("located", [[], [frame(0, status="failed")], [frame(0, path="wrong.jpg")]])
def test_missing_failed_or_wrong_locate_response_cannot_fall_back_to_yolo(located):
    with pytest.raises(LocalizationError):
        merge_localization_rows([frame(0)], located, [frame(0)], route="yolo_locate_hybrid")


def test_hybrid_confirms_student_but_keeps_new_teacher_unverified():
    rows = [frame(0, [person(source="locate_anything", label="student")], source="locate_anything")]
    rows += [frame(i, [person(), person(200, raw_track_id="student-001")]) for i in [1, 2]]
    observations, enriched, boxes, stage = associate_localization_rows(rows, route="yolo_locate_hybrid")
    assert len(observations) == 3 and stage["students"] == 1
    assert [o.detection.source for o in observations] == ["locate_anything", "yolo", "yolo"]
    assert observations[1].detection.attributes["student_confirmation"] == "unambiguous_inheritance"
    teachers = [d for row in enriched for d in row["detections"] if not d.get("student_confirmed")]
    assert len(teachers) == 2 and all("student_id" not in d and d["label"] == "person" for d in teachers)
    assert len(boxes) == 5 and stage["unverified_person_observations"] == 2


def test_empty_frame_breaks_inheritance_until_fresh_locate_confirmation():
    rows = [frame(0, [person(source="locate_anything")], source="locate_anything"), frame(1, []), frame(2),
            frame(3, [person(source="locate_anything")], source="locate_anything"), frame(4)]
    observations, enriched, _, _ = associate_localization_rows(rows, route="yolo_locate_hybrid")
    assert [o.frame_index for o in observations] == [0, 3, 4]
    assert enriched[2]["detections"][0]["student_confirmation"] == "unverified_person"


def test_ambiguous_matching_never_inherits_student_confirmation():
    rows = [frame(0, [person(source="locate_anything")], source="locate_anything"), frame(1, [person(), person()])]
    observations, enriched, _, _ = associate_localization_rows(rows, route="yolo_locate_hybrid")
    assert [o.frame_index for o in observations] == [0]
    assert all(not d["student_confirmed"] for d in enriched[1]["detections"])


def test_yolo_only_person_predictions_are_explicitly_diagnostic():
    observations, enriched, boxes, stage = associate_localization_rows([frame(0), frame(1)], route="yolo_only")
    assert len(observations) == 2 and stage["students"] == 0 and stage["people"] == 1
    assert all(o.detection.label == "person" and o.detection.source == "yolo" and o.metadata["id_uncertain"] for o in observations)
    assert all("student_id" not in box for box in boxes)
    assert all("student_id" not in row["detections"][0] for row in enriched)
    assert all("locate_frame_detection_id" not in o.metadata for o in observations)


def test_legacy_route_remains_byte_contract_equivalent():
    kwargs = dict(max_gap_seconds=2, minimum_iou=.1, maximum_center_distance=.75, minimum_match_score=.3, minimum_match_margin=.1)
    rows = [frame(0, [person(source="locate_anything")]), frame(1, [])]
    assert associate_localization_rows(rows, route="locate_only", **kwargs) == associate_locate_rows(rows, **kwargs)


def test_zero_people_is_valid_association_output():
    observations, rows, boxes, stage = associate_localization_rows([frame(0, [])], route="yolo_locate_hybrid")
    assert not observations and not boxes and stage["status"] == "ok" and stage["students"] == 0


class FakeProcess:
    def __init__(self, *, exit_after_join=0):
        self.alive = True
        self.exitcode = None
        self.exit_after_join = exit_after_join
        self.terminated = False
    def is_alive(self):
        return self.alive
    def join(self, timeout):
        self.alive = False
        self.exitcode = self.exit_after_join
    def terminate(self):
        self.terminated = True
        self.exit_after_join = 15


class FakePipe:
    def __init__(self, response, poll=True):
        self.response, self.ready = response, poll
        self.closed = False
    def send(self, value):
        pass
    def poll(self, timeout):
        return self.ready
    def recv(self):
        return self.response
    def close(self):
        self.closed = True


@pytest.mark.parametrize("response,exitcode,ready", [({"ok": False}, 0, True), ({"ok": 1}, 0, True), ({"ok": True}, 1, True), ({"ok": True}, False, True), ({"ok": True}, 0, False)])
def test_worker_close_requires_ack_and_integer_zero_exit(response, exitcode, ready):
    client = YoloProcessClient({})
    process, pipe = FakeProcess(exit_after_join=exitcode), FakePipe(response, ready)
    client._process, client._connection = process, pipe
    with pytest.raises(YoloWorkerError):
        client.close()
    assert pipe.closed and client.receipt["status"] == "failed"
    client.close()


def test_worker_close_success_receipt():
    client = YoloProcessClient({})
    process, pipe = FakeProcess(), FakePipe({"ok": True})
    client._process, client._connection = process, pipe
    client.close()
    assert pipe.closed and not process.terminated
    assert client.receipt["close_acknowledged"] is True and client.receipt["exitcode"] == 0


def test_real_s1_scene_change_anchor_starts_new_audit_scene():
    rows = [frame(0), frame(1), frame(2, anchor_source="cpu_scene_change"), frame(3)]
    queue, audit = build_difficulty_queue(rows)
    assert [row["frame_index"] for row in queue] == [0, 2]
    assert audit["rows"][2]["scene_id"] != audit["rows"][1]["scene_id"]
    assert "scene_first_screenshot" in queue[1]["difficulty_reasons"]


def test_difficulty_queue_does_not_pollute_locate_receipts_with_yolo_fields():
    yolo = frame(0, model_sha256="yolo_sha", effective_parameters={"conf": .2}, metadata={"provider": "yolo"}, cache_hit=True)
    queue, _ = build_difficulty_queue([yolo])
    assert not {"model_sha256", "source", "effective_parameters", "metadata", "cache_hit", "detections"}.intersection(queue[0])
    locate = {**queue[0], "status": "ok", "detections": [person(source="locate_anything")]}
    merged = merge_localization_rows([yolo], [locate], queue, route="yolo_locate_hybrid")
    assert merged[0]["source"] == "locate_anything" and "model_sha256" not in merged[0]


def test_merge_rejects_forged_yolo_as_locate_response():
    with pytest.raises(LocalizationError, match="wrong_source"):
        merge_localization_rows([frame(0)], [frame(0)], [frame(0)], route="yolo_locate_hybrid")


def test_generic_annotation_shows_actual_source_and_unverified_person(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (640, 240), "white").save(path)
    row = frame(0, [person(source="locate_anything", student_id="student-001", person_id="student-001", student_confirmed=True),
                    person(200, person_id="student-002", student_confirmed=False)])
    row["path"] = str(path)
    rendered, stats = render_localization_annotations([row], tmp_path / "annotations")
    assert stats["rendered_student_boxes"] == stats["rendered_unverified_person_boxes"] == 1
    assert "Locate" in rendered[0]["annotation_labels"][0]
    assert "YOLO" in rendered[0]["annotation_labels"][1] and "unverified person" in rendered[0]["annotation_labels"][1]
    assert Path(rendered[0]["annotated_path"]).is_file()


def test_dense_annotation_suppresses_overlapping_label_plates_and_keeps_every_box(tmp_path):
    path = tmp_path / "dense.jpg"
    Image.new("RGB", (480, 320), "white").save(path)
    detections = [
        person(x=20 + index * 2, source="yolo", student_id=f"student-{index:02d}",
               person_id=f"student-{index:02d}", student_confirmed=True) if index % 3 else
        person(x=20 + index * 2, person_id=f"person-{index:02d}", student_confirmed=False)
        for index in range(14)
    ]
    row = frame(0, detections)
    row["path"] = str(path)
    rendered, stats = render_localization_annotations([row], tmp_path / "annotations")
    labels = rendered[0]["annotation_labels"]
    assert stats["rendered_student_boxes"] + stats["rendered_unverified_person_boxes"] == len(detections)
    assert len(labels) == len(detections)
    assert all("YOLO" in label for label in labels)
    assert stats["suppressed_label_count"] >= 1
    assert Path(rendered[0]["annotated_path"]).is_file()
