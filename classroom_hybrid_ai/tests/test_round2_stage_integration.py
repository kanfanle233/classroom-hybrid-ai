"""Real disk stages and real video encoding, with only model inference replaced."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from classroom_hybrid_ai.config import ProjectConfig, load_config
from classroom_hybrid_ai.research_backend import ResearchBackend
from classroom_hybrid_ai.research_stages import ResearchStageExecutor
from classroom_hybrid_ai.stage_runtime import STAGE_IDS, verify_stage_receipt


ROOT = Path(__file__).resolve().parents[1]


def _video(path: Path, count: int = 16) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    for index in range(count):
        writer.write(np.full((64, 64, 3), index * 7 % 255, dtype=np.uint8))
    writer.release()
    return path


def _config(tmp_path: Path, route: str) -> ProjectConfig:
    base = load_config(ROOT / "configs/default.yaml")
    payload = deepcopy(base.payload)
    payload["paths"]["output_root"] = str(tmp_path / "outputs")
    payload["paths"]["action_registry_root"] = str(tmp_path / "registry")
    payload["locate_anything"]["backend"] = "local"
    payload["locate_anything"]["prompts"] = ["student"]
    payload["runtime"]["device"] = "cpu"
    payload.setdefault("pipeline", {})["localization_route"] = route
    model = tmp_path / "yolo26n-pose.pt"
    model.write_bytes(b"mock-model-used-only-by-injected-cpu-detector")
    payload["yolo"]["tracker_model"] = str(model)
    manifest = yaml.safe_load(base.model_manifest_path.read_text(encoding="utf-8"))
    manifest["models"]["yolo26_pose"] = {"path": str(model), "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    manifest_path = tmp_path / "model_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    payload["paths"]["model_manifest"] = str(manifest_path)
    # Each test owns its own lease and registry; no production GPU lock is touched.
    return ProjectConfig(base.source_path, base.project_root, tmp_path, payload)


def _backend(config: ProjectConfig, control: dict) -> ResearchBackend:
    class EmptyLocate:
        def __init__(self, *args, **kwargs):
            assert not control.get("yolo_active", False), "YOLO must exit before Locate is created"
            control["locate_instances"] = control.get("locate_instances", 0) + 1

        def detect(self, image, labels, *, image_size):
            control.setdefault("locate_calls", []).append(str(image))
            if control.get("fail_locate_once") and not control.get("failed") and len(control["locate_calls"]) == 2:
                control["failed"] = True
                raise RuntimeError("injected_second_locate_interruption")
            return {"raw_answer": "no students", "detections": []}

        def close(self):
            control["locate_closed"] = control.get("locate_closed", 0) + 1

    class EmptyYolo:
        def __init__(self, *args, **kwargs):
            assert not control.get("yolo_active", False)
            control["yolo_active"] = True
            control["yolo_instances"] = control.get("yolo_instances", 0) + 1
            self.last_performance = {}

        def predict_batch(self, images, **kwargs):
            control.setdefault("yolo_calls", []).extend(str(image) for image in images)
            return [{"detections": []} for _ in images]

        def close(self):
            control["yolo_active"] = False
            control["yolo_closed"] = control.get("yolo_closed", 0) + 1

    class LocateSession:
        def __init__(self, config, run_dir):
            self.receipt = {"status": "not_applicable", "backend": "local", "owned_process": False, "pid": None, "exitcode": None}

        def __enter__(self):
            assert not control.get("yolo_active", False)
            return self

        def __exit__(self, *args):
            return False

    def forbid_qwen(*args, **kwargs):
        pytest.fail("Qwen must never be created without a ready candidate")

    def factory(spec):
        control["run_dir"] = spec.run_dir
        return ResearchStageExecutor(spec, locate_factory=EmptyLocate, yolo_factory=EmptyYolo,
            qwen_factory=forbid_qwen, locate_session_factory=LocateSession,
            preflight_runner=lambda *a, **kw: {"status": "ok", "warnings": [], "failed_checks": []})

    return ResearchBackend(config, stage_executor_factory=factory)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_empty_complete(result, *, frame_count: int) -> Path:
    root = Path(result.run_dir)
    assert result.status in {"ok", "warning"}
    if result.status == "warning":
        assert result.warnings and all("audio_not_preserved:" in warning for warning in result.warnings)
    assert result.counts["students"] == 0
    assert result.counts["candidates"] == 0
    for relative in ["artifacts/student_catalog.s3.jsonl", "artifacts/track_observations.jsonl",
                     "artifacts/candidates.selected.jsonl", "artifacts/qwen_crop_queue.jsonl",
                     "reviews.classification.jsonl", "students.jsonl", "action_intervals.jsonl"]:
        assert _rows(root / relative) == [], relative
    for stage_id in STAGE_IDS:
        valid, reasons, receipt = verify_stage_receipt(root, stage_id)
        assert valid, (stage_id, reasons)
        assert receipt["status"] in {"completed", "warning"}
    s5 = json.loads((root / "artifacts/s5_active_response_set.v1.json").read_text(encoding="utf-8"))
    assert s5["model_started"] is False
    assert s5["selected_candidates"] == 0
    assert json.loads((root / "metrics.v1.json").read_text(encoding="utf-8"))["evaluation_status"] == "not_evaluated"
    for name in ("tracking_preview.mp4", "action_preview.mp4", "annotated_final.mp4"):
        cap = cv2.VideoCapture(str(root / "videos" / name))
        assert cap.isOpened(), name
        decoded = 0
        pts = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.shape == (64, 64, 3)
            pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            decoded += 1
        cap.release()
        assert decoded == frame_count, (name, decoded)
        assert all(right > left for left, right in zip(pts, pts[1:]))
    return root


@pytest.mark.parametrize("route", ["locate_only", "yolo_only", "yolo_locate_hybrid"])
def test_real_empty_stages_finish_without_qwen_and_encode_full_video(tmp_path: Path, route: str) -> None:
    control = {}
    backend = _backend(_config(tmp_path, route), control)
    result = backend.run(_video(tmp_path / "empty.avi"))
    root = _assert_empty_complete(result, frame_count=16)
    if route != "yolo_only":
        assert control["locate_calls"]
        assert control["locate_closed"] == control["locate_instances"]
    else:
        assert not control.get("locate_calls")
    if route != "locate_only":
        assert control["yolo_calls"] and not control["yolo_active"]
        assert all(row["status"] == "ok" and row["detections"] == [] for row in _rows(root / "artifacts/localization_raw_results.v1.jsonl"))


def test_hybrid_s2_resume_keeps_successful_yolo_and_locate_requests(tmp_path: Path) -> None:
    control = {"fail_locate_once": True}
    backend = _backend(_config(tmp_path, "yolo_locate_hybrid"), control)
    with pytest.raises(RuntimeError, match="injected_second_locate_interruption"):
        backend.run(_video(tmp_path / "empty.avi", count=32))
    root = Path(control["run_dir"])
    before_yolo_calls = list(control["yolo_calls"])
    assert len(before_yolo_calls) >= 2
    locate_cache = list((root / "responses/locate").glob("*.json"))
    assert len(locate_cache) == 1
    digest_before = hashlib.sha256(locate_cache[0].read_bytes()).hexdigest()
    first_success = control["locate_calls"][0]
    result = backend.resume(root)
    _assert_empty_complete(result, frame_count=32)
    assert control["yolo_calls"] == before_yolo_calls
    assert control["yolo_instances"] == 1 and control["yolo_closed"] == 1
    assert Counter(control["locate_calls"])[first_success] == 1
    assert hashlib.sha256(locate_cache[0].read_bytes()).hexdigest() == digest_before
    stage = json.loads((root / "diagnostics/yolo_stage.v1.json").read_text(encoding="utf-8"))
    assert stage["inference_calls"] == 0 and stage["cache_hits"] == len(before_yolo_calls)
    state = json.loads((root / "run_state.v1.json").read_text(encoding="utf-8"))
    assert state["status"] in {"completed", "warning"}
