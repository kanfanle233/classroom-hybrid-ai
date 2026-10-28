from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from classroom_hybrid_ai.config import ProjectConfig, load_config
from classroom_hybrid_ai.research_backend import ResearchBackend
from classroom_hybrid_ai.research_stages import ResearchStageExecutor, ResearchStageSpec, StageOutcome
from classroom_hybrid_ai.stage_runtime import STAGE_IDS, verify_stage_receipt
from classroom_hybrid_ai.storage import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]


def _video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    for index in range(8):
        writer.write(np.full((64, 64, 3), index * 12, dtype=np.uint8))
    writer.release()
    return path


def _config(tmp_path: Path) -> ProjectConfig:
    base = load_config(ROOT / "configs" / "default.yaml")
    payload = deepcopy(base.payload)
    payload["paths"]["output_root"] = str((tmp_path / "output").resolve())
    return ProjectConfig(base.source_path, base.project_root, base.repository_root, payload)


class _FakeStages:
    def __init__(self, spec: ResearchStageSpec, control: dict[str, Any]) -> None:
        self.spec = spec
        self.control = control

    def execute(
        self,
        stage_id: str,
        *,
        input_fingerprint_sha256: str,
        partial_outputs_reusable: bool = False,
    ) -> StageOutcome:
        self.control.setdefault("run_dirs", []).append(str(self.spec.run_dir))
        self.control.setdefault("calls", {}).setdefault(stage_id, 0)
        self.control["calls"][stage_id] += 1
        if stage_id == "S5":
            response_dir = self.spec.run_dir / "responses" / "classification" / input_fingerprint_sha256[:16]
            response_dir.mkdir(parents=True, exist_ok=True)
            for event_id in ("event-1", "event-2", "event-3"):
                response = response_dir / f"{event_id}.json"
                if partial_outputs_reusable and response.is_file():
                    self.control.setdefault("qwen_reused", []).append(event_id)
                    continue
                self.control["qwen_calls"] = int(self.control.get("qwen_calls", 0)) + 1
                atomic_write_json(response, {"candidate_event_id": event_id, "status": "ok"})
                if self.control.get("fail_s5_once") and not self.control.get("failed_s5"):
                    self.control["failed_s5"] = True
                    raise RuntimeError("synthetic_s5_interrupt")
        artifact = self.spec.run_dir / "artifacts" / "fake" / f"{stage_id}.json"
        atomic_write_json(artifact, {
            "stage_id": stage_id,
            "input_fingerprint_sha256": input_fingerprint_sha256,
            "call": self.control["calls"][stage_id],
        })
        if stage_id == "S2" and self.control.get("fail_s2_once") and not self.control.get("failed_s2"):
            self.control["failed_s2"] = True
            raise RuntimeError("synthetic_s2_interrupt")
        artifacts = [f"artifacts/fake/{stage_id}.json"]
        if stage_id == "S0":
            artifacts.append("run_manifest.v1.json")
        if stage_id == "S5":
            artifacts.append(f"responses/classification/{input_fingerprint_sha256[:16]}")
        return StageOutcome(
            artifact_uris=tuple(artifacts),
            stats={"fake": True, "model_started": False},
            unload_ack=(
                not bool(self.control.get("missing_s2_unload"))
                if stage_id == "S2" else None
            ),
        )


def _backend(config: ProjectConfig, control: dict[str, Any]) -> ResearchBackend:
    return ResearchBackend(config, stage_executor_factory=lambda spec: _FakeStages(spec, control))


def test_formal_run_ids_are_unique_timestamp_plus_digest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    first = _backend(config, {}).run(video)
    second = _backend(config, {}).run(video)
    first_id = Path(first.run_dir).name
    second_id = Path(second.run_dir).name
    assert first_id != second_id
    assert re.fullmatch(r"\d{8}T\d{12}Z-[0-9a-f]{12}", first_id)
    assert re.fullmatch(r"\d{8}T\d{12}Z-[0-9a-f]{12}", second_id)
    receipt = json.loads((Path(first.run_dir) / "stage_receipts" / "S0.json").read_text(encoding="utf-8"))
    hash_metrics = receipt["stats"]["commit_hash"]
    assert hash_metrics["counts"]["hash.bytes_read"] > 0
    assert hash_metrics["timings_seconds"]["commit.artifact_hash_seconds"] >= hash_metrics["timings_seconds"]["hash.seconds"]
    assert verify_stage_receipt(first.run_dir, "S0")[0]


def test_backend_binds_incremental_stage_progress_reporter(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {}

    class ReportingStages(_FakeStages):
        def set_progress_reporter(self, reporter) -> None:
            self.reporter = reporter

        def execute(self, stage_id: str, **kwargs: Any) -> StageOutcome:
            if stage_id == "S5":
                self.reporter("S5", 1, 3, "S5 Qwen 响应已落盘 1/3")
            outcome = super().execute(stage_id, **kwargs)
            if stage_id != "S5":
                return outcome
            return StageOutcome(
                artifact_uris=outcome.artifact_uris,
                stats=outcome.stats,
                processed_units=3,
                total_units=3,
            )

    result = ResearchBackend(
        config,
        stage_executor_factory=lambda spec: ReportingStages(spec, control),
    ).run(video)
    events = [
        json.loads(line)
        for line in (Path(result.run_dir) / "stage_events.v1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    incremental = [
        row for row in events
        if row.get("stage_id") == "S5" and row.get("message") == "S5 Qwen 响应已落盘 1/3"
    ]
    assert len(incremental) == 1
    assert incremental[0]["processed_units"] == 1
    assert incremental[0]["total_units"] == 3


def test_s2_interruption_resumes_at_s2_without_repeating_s0_s1(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {"fail_s2_once": True}
    backend = _backend(config, control)
    with pytest.raises(RuntimeError, match="synthetic_s2_interrupt"):
        backend.run(video)
    run_dir = Path(control["run_dirs"][0])
    result = backend.resume(run_dir)
    assert result.status == "ok"
    assert control["calls"]["S0"] == 1
    assert control["calls"]["S1"] == 1
    assert control["calls"]["S2"] == 2
    assert all(control["calls"][stage_id] == 1 for stage_id in STAGE_IDS[3:])


def test_s5_interruption_reuses_successful_request_and_runs_remaining_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {"fail_s5_once": True}
    backend = _backend(config, control)
    with pytest.raises(RuntimeError, match="synthetic_s5_interrupt"):
        backend.run(video)
    run_dir = Path(control["run_dirs"][0])
    failed_state = json.loads((run_dir / "run_state.v1.json").read_text(encoding="utf-8"))
    assert failed_state["status"] == "failed"
    assert float(failed_state["overall_progress"]) < 100.0
    backend.resume(run_dir)
    assert control["calls"]["S5"] == 2
    assert control["qwen_calls"] == 3
    assert control["qwen_reused"] == ["event-1"]
    assert all(control["calls"][stage_id] == 1 for stage_id in STAGE_IDS[:5])


def test_tampered_stage_artifact_invalidates_that_stage_and_downstream(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {}
    backend = _backend(config, control)
    result = backend.run(video)
    run_dir = Path(result.run_dir)
    (run_dir / "artifacts" / "fake" / "S3.json").write_text("tampered\n", encoding="utf-8")
    backend.resume(run_dir)
    assert all(control["calls"][stage_id] == 1 for stage_id in STAGE_IDS[:3])
    assert all(control["calls"][stage_id] == 2 for stage_id in STAGE_IDS[3:])
    for stage_id in STAGE_IDS:
        ok, reasons, _ = verify_stage_receipt(run_dir, stage_id)
        assert ok, (stage_id, reasons)


def test_qwen_model_fingerprint_change_invalidates_from_s5(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {}
    backend = _backend(config, control)
    result = backend.run(video)
    run_dir = Path(result.run_dir)

    # Preserve the actual Locate fingerprint from the completed run so only S5
    # changes.  The value is stored in run_state metadata by the orchestrator.
    state = json.loads((run_dir / "run_state.v1.json").read_text(encoding="utf-8"))
    locate_sha = state["metadata"]["locate_model_sha256"]
    monkeypatch.setattr(
        "classroom_hybrid_ai.research_backend.model_manifest_fingerprints",
        lambda _config: {"locate_model_sha256": locate_sha, "qwen_model_sha256": "qwen-changed"},
    )
    backend.resume(run_dir)
    assert all(control["calls"][stage_id] == 1 for stage_id in STAGE_IDS[:5])
    assert all(control["calls"][stage_id] == 2 for stage_id in STAGE_IDS[5:])


def test_qwen_config_change_invalidates_only_s5_and_downstream(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {}
    result = _backend(config, control).run(video)
    payload = deepcopy(config.payload)
    payload["qwen"]["prompt_version"] = "compact_v1"
    changed = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    _backend(changed, control).resume(Path(result.run_dir))
    assert all(control["calls"][stage] == 1 for stage in STAGE_IDS[:5])
    assert all(control["calls"][stage] == 2 for stage in STAGE_IDS[5:])


def test_hash_cache_is_commit_scoped_and_audit_rereads_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from classroom_hybrid_ai.hashing import hash_file_cache
    from classroom_hybrid_ai.stage_runtime import StageLedger, artifact_refs
    ledger = StageLedger(tmp_path, run_id="run", video_id="video")
    artifact = tmp_path / "output.json"
    artifact.write_text('{"value":1}', encoding="utf-8")
    reads = []
    original_open = Path.open
    def counted_open(path, mode="r", *args, **kwargs):
        if path == artifact and mode == "rb":
            reads.append(path)
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", counted_open)
    with hash_file_cache():
        refs = artifact_refs(tmp_path, ["output.json"])
        ledger.receipt("S0", {"status": "completed", "artifacts": [ref.to_dict() for ref in refs]})
        assert verify_stage_receipt(tmp_path, "S0", reuse_commit_hashes=True)[0]
        assert len(reads) == 1
        assert verify_stage_receipt(tmp_path, "S0")[0]
        assert len(reads) == 2
    assert verify_stage_receipt(tmp_path, "S0")[0]
    assert len(reads) == 3
    ok, reasons, _ = verify_stage_receipt(tmp_path, "S0", expected_fingerprint_version="v2")
    assert not ok
    assert "stage_dependency_fingerprint_missing_or_obsolete" in reasons
    ledger.close()


def test_locate_model_fingerprint_change_invalidates_from_s2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {}
    backend = _backend(config, control)
    result = backend.run(video)
    run_dir = Path(result.run_dir)
    state = json.loads((run_dir / "run_state.v1.json").read_text(encoding="utf-8"))
    qwen_sha = state["metadata"]["qwen_model_sha256"]
    monkeypatch.setattr(
        "classroom_hybrid_ai.research_backend.model_manifest_fingerprints",
        lambda _config: {"locate_model_sha256": "locate-changed", "qwen_model_sha256": qwen_sha},
    )
    backend.resume(run_dir)
    assert all(control["calls"][stage_id] == 1 for stage_id in STAGE_IDS[:2])
    assert all(control["calls"][stage_id] == 2 for stage_id in STAGE_IDS[2:])


def test_input_changes_s0_and_evidence_config_changes_s4(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {}
    result = _backend(config, control).run(video)
    run_dir = Path(result.run_dir)
    with video.open("ab") as handle:
        handle.write(b"changed-input-digest")
    _backend(config, control).resume(run_dir)
    assert all(control["calls"][stage_id] == 2 for stage_id in STAGE_IDS)

    changed_payload = deepcopy(config.payload)
    changed_payload["evidence"]["jpeg_quality"] = int(changed_payload["evidence"]["jpeg_quality"]) - 1
    changed_config = ProjectConfig(config.source_path, config.project_root, config.repository_root, changed_payload)
    _backend(changed_config, control).resume(run_dir)
    assert all(control["calls"][stage_id] == 2 for stage_id in STAGE_IDS[:4])
    assert all(control["calls"][stage_id] == 3 for stage_id in STAGE_IDS[4:])


def test_profile_contract_change_invalidates_from_s0(tmp_path: Path) -> None:
    config = _config(tmp_path)
    profile_copy = tmp_path / "action_profiles.yaml"
    profile_copy.write_text((ROOT / "configs" / "action_profiles.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    payload = deepcopy(config.payload)
    payload["profiles"]["action_profiles"] = str(profile_copy.resolve())
    config = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {}
    backend = _backend(config, control)
    result = backend.run(video)
    run_dir = Path(result.run_dir)
    text = profile_copy.read_text(encoding="utf-8")
    profile_copy.write_text(text.replace("学生低头，手部可见地持续接触纸张或作业书写。", "学生低头并可见地持续接触纸张书写。", 1), encoding="utf-8")
    backend.resume(run_dir)
    assert all(control["calls"][stage_id] == 2 for stage_id in STAGE_IDS)


def test_s2_cannot_commit_without_explicit_unload_ack(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {"missing_s2_unload": True}
    with pytest.raises(RuntimeError, match="stage_s2_locate_unload_not_acknowledged"):
        _backend(config, control).run(video)
    run_dir = Path(control["run_dirs"][0])
    state = json.loads((run_dir / "run_state.v1.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert float(state["overall_progress"]) < 100.0
    assert not (run_dir / "stage_receipts" / "S2.json").exists()


@pytest.mark.parametrize("error_field", ["close_error", "cuda_teardown_error"])
def test_s5_cannot_commit_teardown_error_even_with_claimed_ack_and_zero_exit(
    tmp_path: Path, error_field: str,
) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {}
    release = {"worker_isolated": True, "worker_exitcode": 0, "close_error": None, "cuda_after_release": {}}
    if error_field == "close_error":
        release["close_error"] = "synthetic_child_close_failure"
    else:
        release["cuda_after_release"] = {"teardown_error": "synthetic_parent_cuda_cleanup_failure"}

    class TeardownErrorStages(_FakeStages):
        def execute(self, stage_id: str, **kwargs: Any) -> StageOutcome:
            outcome = super().execute(stage_id, **kwargs)
            if stage_id != "S5":
                return outcome
            # The backend must reject this contradictory producer receipt,
            # even if a stage or legacy adapter claims a successful unload.
            return StageOutcome(
                artifact_uris=outcome.artifact_uris,
                stats={"model_started": True, "release": release},
                unload_ack=True,
                exit_code=0,
            )

    backend = ResearchBackend(config, stage_executor_factory=lambda spec: TeardownErrorStages(spec, control))
    with pytest.raises(RuntimeError, match="stage_s5_qwen_teardown_failed"):
        backend.run(video)
    run_dir = Path(control["run_dirs"][0])
    state = json.loads((run_dir / "run_state.v1.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert not (run_dir / "stage_receipts" / "S5.json").exists()
    assert control["calls"].get("S6", 0) == 0


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("error_field", [None, "close_error", "cuda_teardown_error"])
def test_real_s5_unload_ack_requires_clean_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated: bool, error_field: str | None,
) -> None:
    from classroom_hybrid_ai.contracts import CandidateEvent, ReviewDecision
    from classroom_hybrid_ai.storage import RunStore
    import classroom_hybrid_ai.research_stages as stages_module

    config = _config(tmp_path)
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
    spec = ResearchStageSpec(
        config=config, store=store, source=tmp_path / "unused.avi", video={}, profile=profile,
        mode="production_adaptive", max_frames=None, skip_qwen=False,
        allow_incomplete_preflight=False, experiment_digest="experiment", input_sha256="input",
        config_sha256="config", code_sha256="code", locate_model_sha256="locate", qwen_model_sha256="qwen",
    )
    adapter = type("QwenProcessClient" if isolated else "DirectQwenAdapter", (), {"pid": 4321, "exitcode": 0})()
    release = {"close_error": None, "cuda_after_release": {}}
    if error_field == "close_error":
        release["close_error"] = "synthetic_close_failure"
    elif error_field == "cuda_teardown_error":
        release["cuda_after_release"] = {"teardown_error": "synthetic_cuda_cleanup_failure"}
    review = ReviewDecision(
        candidate_event_id=candidate.event_id, decision="uncertain", reviewed_event_type=candidate.event_type,
        behavior="UNKNOWN", rationale="stub", schema_valid=True,
    )
    monkeypatch.setattr(stages_module, "_make_qwen_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(stages_module, "_qwen_pass", lambda *_args, **_kwargs: ([review], {"status": "ok"}))
    monkeypatch.setattr(stages_module, "_profile_vocabulary", lambda _spec: (SimpleNamespace(sha256="registry"), [], "vocabulary"))
    monkeypatch.setattr(stages_module, "GpuLease", lambda _path: SimpleNamespace(acquire=lambda: None, release=lambda: None))
    monkeypatch.setattr(stages_module, "release_model", lambda _adapter: dict(release))
    executor = ResearchStageExecutor(spec)
    monkeypatch.setattr(executor, "_commit_preview", lambda _stage: None)

    outcome = executor._run_s5(input_fingerprint_sha256="fingerprint", partial_outputs_reusable=False)

    assert outcome.stats["model_started"] is True
    assert outcome.exit_code == 0
    assert outcome.unload_ack is (error_field is None)
    assert outcome.stats["release"]["close_error"] == release["close_error"]
    assert outcome.stats["release"]["cuda_after_release"] == release["cuda_after_release"]


def test_complete_receipts_do_not_mask_failed_run_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = _video(tmp_path / "sample.avi")
    control: dict[str, Any] = {}
    backend = _backend(config, control)
    result = backend.run(video)
    run_dir = Path(result.run_dir)
    state_path = run_dir / "run_state.v1.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"status": "failed", "overall_progress": 100.0, "research_validity": "invalid"})
    atomic_write_json(state_path, state)

    resumed = backend.resume(run_dir)

    repaired = json.loads(state_path.read_text(encoding="utf-8"))
    assert resumed.status == "ok"
    assert repaired["status"] == "completed"
    assert repaired["research_validity"] == "valid"
    assert all(control["calls"][stage_id] == 1 for stage_id in STAGE_IDS[:-1])
    assert control["calls"]["S7"] == 2


class _FakeLocate:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def detect(self, _image: str, _labels: list[str], *, image_size: tuple[int, int]) -> dict[str, Any]:
        width, height = image_size
        return {
            "raw_answer": "student",
            "detections": [{
                "label": "student",
                "bbox": [width * 0.2, height * 0.1, width * 0.7, height * 0.9],
                "confidence": 0.95,
                "source": "fake_locate",
                "attributes": {},
            }],
        }

    def close(self) -> None:
        pass


class _FakeLocateSession:
    def __init__(self, _config: ProjectConfig, _run_dir: Path) -> None:
        self.receipt = {
            "status": "not_applicable",
            "backend": "local",
            "owned_process": False,
            "pid": None,
            "exitcode": None,
        }

    def __enter__(self) -> "_FakeLocateSession":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False


class _FakeQwen:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls = 0

    def warmup(self) -> None:
        pass

    def review(
        self,
        candidate_event_id: str,
        _evidence: dict[str, Any],
        *,
        event_type: str,
        evidence_frame_indices: list[int],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.calls += 1
        return {
            "candidate_event_id": candidate_event_id,
            "decision": "confirmed",
            "reviewed_event_type": event_type,
            "behavior": "a_tt",
            "rationale": "抬头可见",
            "evidence_frame_indices": list(evidence_frame_indices),
            "self_reported_confidence": None,
            "reviewer": "fake_qwen",
            "schema_valid": True,
            "observability": "clear",
        }

    def close(self) -> None:
        pass


def test_default_disk_stages_complete_with_model_test_doubles_and_receipts_stay_valid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = deepcopy(config.payload)
    payload["locate_anything"]["backend"] = "local"
    config = ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)
    video = _video(tmp_path / "real-stages.avi")

    def factory(spec: ResearchStageSpec) -> ResearchStageExecutor:
        return ResearchStageExecutor(
            spec,
            locate_factory=_FakeLocate,
            qwen_factory=_FakeQwen,
            locate_session_factory=_FakeLocateSession,
            preflight_runner=lambda *_args, **_kwargs: {"status": "ok", "warnings": [], "failed_checks": []},
        )

    result = ResearchBackend(config, stage_executor_factory=factory).run(video)
    run_dir = Path(result.run_dir)
    assert result.status in {"ok", "warning"}
    if result.status == "warning":
        assert result.warnings
        assert all("audio_not_preserved:" in item for item in result.warnings)
    assert result.counts["students"] == 1
    assert result.counts["candidates"] >= 1
    assert (run_dir / "videos" / "annotated_final.mp4").is_file()
    assert (run_dir / "timeline.v2.json").is_file()
    students = [
        json.loads(line)
        for line in (run_dir / "students.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["student_id"] for row in students] == ["student-001"]
    assert students[0]["identity_scope"] == "locate_assigned_same_camera_video_run"
    state = json.loads((run_dir / "run_state.v1.json").read_text(encoding="utf-8"))
    assert state["research_validity"] == "valid"
    assert state["previews"] == [
        f"previews/{stage_id}/preview.json" for stage_id in STAGE_IDS
    ]
    for stage_id in STAGE_IDS:
        preview = json.loads(
            (run_dir / "previews" / stage_id / "preview.json").read_text(encoding="utf-8")
        )
        assert preview["semantic"]
        assert preview["preview_status"] == "available"
    for stage_id in STAGE_IDS:
        ok, reasons, receipt = verify_stage_receipt(run_dir, stage_id)
        assert ok, (stage_id, reasons)
        assert receipt is not None and receipt["status"] in {"completed", "warning"}

    def forbidden_factory(_spec: ResearchStageSpec) -> Any:
        raise AssertionError("a complete verified run must not construct a stage executor")

    resumed = ResearchBackend(config, stage_executor_factory=forbidden_factory).resume(run_dir)
    assert resumed.run_dir == result.run_dir
