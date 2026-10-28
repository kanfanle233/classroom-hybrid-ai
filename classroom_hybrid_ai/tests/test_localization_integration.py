from __future__ import annotations

import json
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from PIL import Image

from classroom_hybrid_ai.config import ProjectConfig, load_config
from classroom_hybrid_ai.hashing import sha256_file
from classroom_hybrid_ai.outputs_v3 import _student_rows, write_stage_preview
from classroom_hybrid_ai.profiles import load_profile
from classroom_hybrid_ai.research_stages import (
    ResearchStageExecutor, ResearchStageSpec, StageExecutionError, StageOutcome, model_manifest_fingerprints,
)
from classroom_hybrid_ai.storage import RunStore, iter_jsonl
import classroom_hybrid_ai.research_stages as stages


ROOT = Path(__file__).resolve().parents[1]


def detector_person(source="yolo", x=5):
    return {"label": "person" if source == "yolo" else "student", "source": source,
            "bbox": [x, 5, x + 25, 55], "score": .9, "class_id": 0, "detection_index": 1}


def setup_executor(tmp_path, route="yolo_only", key="yolo26_pose"):
    base = load_config(ROOT / "configs/default.yaml")
    payload = deepcopy(base.payload)
    model = tmp_path / "weights.pt"
    model.write_bytes(b"small-fake-local-weights")
    manifest = tmp_path / "models.yaml"
    manifest.write_text(yaml.safe_dump({"models": {key: {
        "path": str(model), "sha256": sha256_file(model), "revision": "locked"}}}), encoding="utf-8")
    payload["pipeline"]["localization_route"] = route
    payload["localization"]["model_manifest_key"] = key
    payload["yolo"]["tracker_model"] = str(model)
    payload["paths"]["model_manifest"] = str(manifest)
    config = ProjectConfig(base.source_path, base.project_root, tmp_path, payload)
    store = RunStore(tmp_path / "runs", "sample", "config", unique_run=True)
    source = tmp_path / "unused.mp4"
    source.write_bytes(b"unused-for-localization-only-test")
    spec = ResearchStageSpec(config=config, store=store, source=source,
        video={"duration_seconds": 2, "frame_count": 2, "width": 240, "height": 64, "fps": 1},
        profile=load_profile(ROOT / "configs/action_profiles.yaml"), mode="production_adaptive",
        max_frames=None, skip_qwen=False, allow_incomplete_preflight=False,
        experiment_digest="experiment", input_sha256="input", config_sha256="config", code_sha256="code",
        locate_model_sha256="locate-metadata-fingerprint", qwen_model_sha256="qwen-metadata-fingerprint",
        yolo_model_sha256="DO-NOT-USE-AS-WEIGHT-SHA")
    executor = ResearchStageExecutor(spec)
    rows = []
    for index in range(2):
        image = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (240, 64), "white").save(image)
        rows.append({"frame_index": index, "pts_seconds": float(index), "path": str(image),
                     "status": "ok", "source": "yolo", "detections": [detector_person()]})
    store.write_jsonl("artifacts/screenshot_schedule.v1.jsonl", [
        {k: v for k, v in row.items() if k not in {"source", "detections"}} for row in rows])
    return executor, rows, manifest


class FakeYolo:
    def __init__(self):
        self.closed = False
        self.calls = 0
    def predict_batch(self, images, **kwargs):
        self.calls += 1
        return [{"detections": [detector_person()]} for _ in images]
    def close(self):
        self.closed = True


@pytest.mark.parametrize("manifest_key", ["yolo26_pose", "yolo26_medium_detection"])
def test_s2_uses_selected_manifest_weight_sha_and_yolo_prediction_route(tmp_path, monkeypatch, manifest_key):
    executor, _, _ = setup_executor(tmp_path, key=manifest_key)
    fake = FakeYolo()
    executor.yolo_factory = lambda **kwargs: fake
    monkeypatch.setattr(stages, "GpuLease", lambda _: nullcontext())
    outcome = executor._run_s2(input_fingerprint_sha256="code-fingerprint")
    assert fake.closed and fake.calls == 2
    assert outcome.stats["yolo"]["model_sha256"] == sha256_file(executor.spec.config.model_path("yolo.tracker_model"))
    assert outcome.stats["yolo"]["model_manifest_key"] == manifest_key
    preview = json.loads((executor.spec.run_dir / "previews/S2/preview.json").read_text())
    assert preview["counts"]["yolo_detection_count"] == 2
    assert preview["counts"]["locate_detection_count"] == 0
    assert preview["source_artifact"] == "artifacts/localization_raw_results.v1.jsonl"


def test_s2_rejects_wrong_weight_even_when_metadata_fingerprint_exists(tmp_path, monkeypatch):
    executor, _, manifest = setup_executor(tmp_path)
    payload = yaml.safe_load(manifest.read_text())
    payload["models"]["yolo26_pose"]["sha256"] = "0" * 64
    manifest.write_text(yaml.safe_dump(payload))
    monkeypatch.setattr(stages, "GpuLease", lambda _: nullcontext())
    executor.yolo_factory = lambda **kw: pytest.fail("wrong weights must fail before model initialization")
    with pytest.raises(RuntimeError, match="model_sha_mismatch"):
        executor._run_s2(input_fingerprint_sha256="code-fingerprint")


def test_model_fingerprint_reads_configured_manifest_entry(tmp_path):
    executor, _, manifest = setup_executor(tmp_path, key="medium")
    first = model_manifest_fingerprints(executor.spec.config)
    payload = yaml.safe_load(manifest.read_text())
    payload["models"]["medium"]["revision"] = "second-locked-revision"
    manifest.write_text(yaml.safe_dump(payload))
    second = model_manifest_fingerprints(executor.spec.config)
    assert first["yolo_model_sha256"] != second["yolo_model_sha256"]
    assert first["locate_model_sha256"] == second["locate_model_sha256"]


def test_hybrid_s2_closes_yolo_before_locate_and_preserves_provider_counts(tmp_path, monkeypatch):
    executor, rows, _ = setup_executor(tmp_path, route="yolo_locate_hybrid")
    fake = FakeYolo()
    executor.yolo_factory = lambda **kwargs: fake
    monkeypatch.setattr(stages, "GpuLease", lambda _: nullcontext())
    calls = []

    def locate(**kwargs):
        assert fake.closed
        calls.append(kwargs["scenes"])
        assert all("source" not in scene and "model_sha256" not in scene for scene in kwargs["scenes"])
        executor.spec.store.write_jsonl("artifacts/locate_raw_results.jsonl", [
            {**scene, "status": "ok", "detections": [detector_person("locate_anything")]} for scene in kwargs["scenes"]])
        return StageOutcome(stats={"requests": len(kwargs["scenes"])})

    monkeypatch.setattr(executor, "_run_locate_s2", locate)
    executor._run_s2(input_fingerprint_sha256="code-fingerprint")
    assert len(calls) == 1 and len(calls[0]) == 1
    preview = json.loads((executor.spec.run_dir / "previews/S2/preview.json").read_text())
    assert preview["counts"]["detections_by_source"] == {"locate_anything": 1, "yolo": 1}


def test_yolo_diagnostic_people_never_enter_student_registry_or_candidates(tmp_path):
    executor, rows, _ = setup_executor(tmp_path)
    executor.spec.store.write_jsonl("artifacts/localization_raw_results.v1.jsonl", rows)
    outcome = executor._run_s3()
    root = executor.spec.run_dir
    assert outcome.stats["students"] == 0
    for uri in ("artifacts/track_observations.jsonl", "artifacts/student_catalog.s3.jsonl", "artifacts/locate_student_boxes.jsonl"):
        assert list(iter_jsonl(root / uri)) == []
    assert len(list(iter_jsonl(root / "artifacts/localization_boxes.v1.jsonl"))) == 2
    assert len(list(iter_jsonl(root / "artifacts/localization_track_observations.v1.jsonl"))) == 2
    assert _student_rows(root, 2) == []
    preview = json.loads((root / "previews/S3/preview.json").read_text())
    assert preview["preview_status"] == "available" and preview["counts"]["student_ids"] == []
    assert preview["counts"]["unverified_person_boxes"] == 2
    annotated = list(iter_jsonl(root / "artifacts/localization_results.v1.jsonl"))
    assert "YOLO" in annotated[0]["annotation_labels"][0] and "unverified" in annotated[0]["annotation_labels"][0]
    executor._run_s4()
    assert list(iter_jsonl(root / "artifacts/candidates.all.jsonl")) == []
    assert list(iter_jsonl(root / "artifacts/qwen_crop_queue.jsonl")) == []


def test_hybrid_student_catalog_contains_confirmed_students_only(tmp_path):
    executor, rows, _ = setup_executor(tmp_path, route="yolo_locate_hybrid")
    rows[0].update({"source": "locate_anything", "detections": [detector_person("locate_anything")]})
    rows[1]["detections"].append(detector_person(x=150))
    executor.spec.store.write_jsonl("artifacts/localization_raw_results.v1.jsonl", rows)
    outcome = executor._run_s3()
    assert outcome.stats["students"] == 1
    assert len(list(iter_jsonl(executor.spec.run_dir / "artifacts/localization_boxes.v1.jsonl"))) == 3
    assert len(list(iter_jsonl(executor.spec.run_dir / "artifacts/locate_student_boxes.jsonl"))) == 2
    assert len(_student_rows(executor.spec.run_dir, 2)) == 1


def test_generic_empty_student_boundary_does_not_resurrect_stale_delivery(tmp_path):
    executor, rows, _ = setup_executor(tmp_path)
    executor.spec.store.write_jsonl("artifacts/localization_raw_results.v1.jsonl", rows)
    executor.spec.store.write_jsonl("students.jsonl", [{"student_id": "stale-student", "pts_seconds": 0}])
    executor._run_s3()
    assert _student_rows(executor.spec.run_dir, 2) == []


def test_locate_only_s2_replaces_old_generic_raw_preview_boundary(tmp_path, monkeypatch):
    executor, rows, _ = setup_executor(tmp_path, route="locate_only")
    executor.spec.store.write_jsonl("artifacts/localization_raw_results.v1.jsonl", rows)
    located = {**rows[0], "source": "locate_anything", "detections": [detector_person("locate_anything")]}

    def locate(**kwargs):
        assert kwargs["commit_preview"] is False
        executor.spec.store.write_jsonl("artifacts/locate_raw_results.jsonl", [located])
        return StageOutcome()

    monkeypatch.setattr(executor, "_run_locate_s2", locate)
    outcome = executor._run_s2(input_fingerprint_sha256="new-locate-only")
    assert "artifacts/localization_raw_results.v1.jsonl" in outcome.artifact_uris
    preview = json.loads((executor.spec.run_dir / "previews/S2/preview.json").read_text())
    assert preview["counts"]["locate_detection_count"] == 1 and preview["counts"]["yolo_detection_count"] == 0


def test_s2_raw_preview_never_uses_stale_associated_boxes(tmp_path):
    executor, rows, _ = setup_executor(tmp_path)
    executor.spec.store.write_jsonl("artifacts/localization_raw_results.v1.jsonl", rows)
    executor.spec.store.write_jsonl("artifacts/locate_student_boxes.jsonl", [
        {"student_id": "stale", "locate_detection_index": 99, "annotated_path": rows[0]["path"]}])
    write_stage_preview(executor.spec.run_dir, "S2", info=executor.spec.video)
    preview = json.loads((executor.spec.run_dir / "previews/S2/preview.json").read_text())
    assert preview["counts"]["student_ids"] == []
    assert preview["counts"]["locate_detection_indices"] == []
    assert preview["counts"]["yolo_detection_count"] == 2
