"""Formal v3.1 research entry point with verified S0-S7 recovery boundaries."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import ProjectConfig
from .hashing import canonical_sha256, project_config_digest, project_config_payload, sha256_file, stage_dependency_fingerprints, hash_file_cache
from .pipeline import PipelineResult
from .performance import PerformanceRecorder, measured, recording
from .profiles import ActionProfile, load_profile
from .research_stages import (
    ResearchStageExecutor,
    ResearchStageSpec,
    StageExecutionError,
    StageOutcome,
    model_manifest_fingerprints,
)
from .research_policy import localization_route, budget_policy, qwen_selection_budget, effective_configuration_report
from .stage_runtime import (
    AdaptiveScreenshotScheduler,
    STAGE_DEFINITIONS,
    STAGE_IDS,
    StageLedger,
    artifact_refs,
    hash_tree,
    read_stage_receipt,
    verify_stage_receipt,
)
from .storage import RunStore, atomic_write_json, iter_jsonl
from .video import probe_video


class ResearchBackendError(RuntimeError):
    pass


def _warning_affects_research_validity(value: Any) -> bool:
    """Audio delivery failures do not invalidate otherwise valid visuals."""

    return "audio_not_preserved:" not in str(value)


def _clone_config(config: ProjectConfig, payload: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(config.source_path, config.project_root, config.repository_root, payload)


def _effective_config(config: ProjectConfig, *, profile: ActionProfile, mode: str) -> ProjectConfig:
    payload = deepcopy(config.payload)
    payload.setdefault("research", {}).update({"profile": profile.profile_id, "mode": mode})
    route = localization_route(config)
    payload.setdefault("pipeline", {}).update({"localization_route": route,
        "primary_detector": {"locate_only": "locate_anything", "yolo_only": "yolo", "yolo_locate_hybrid": "hybrid"}[route]})
    payload.setdefault("locate_anything", {}).update({"enabled": route != "yolo_only", "required": route != "yolo_only"})
    payload["research_budget"] = budget_policy(config)
    payload.setdefault("open_vocabulary", {})["max_hold_seconds"] = min(
        2.0,
        float(payload.get("open_vocabulary", {}).get("max_hold_seconds", 2.0)),
    )
    return _clone_config(config, payload)


def _experiment_digest(
    config: ProjectConfig,
    source: Path,
    profile: ActionProfile,
    *,
    mode: str,
    max_frames: int | None,
    skip_qwen: bool,
) -> str:
    stat = source.stat()
    return canonical_sha256({
        "config": project_config_payload(config),
        "source": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "profile_sha256": profile.sha256,
        "mode": mode,
        "max_frames": max_frames,
        "skip_qwen": skip_qwen,
    })


def _stage_input_fingerprint(
    stage_id: str,
    *,
    global_fingerprints: Mapping[str, str],
    upstream_receipt_sha256: str | None,
) -> str:
    payload: dict[str, Any] = {
        "schema_version": "classroom_hybrid_ai.stage_input.v2",
        "stage_id": stage_id,
        "code_sha256": global_fingerprints[f"stage_code:{stage_id}"],
        "config_sha256": global_fingerprints[f"stage_config:{stage_id}"],
    }
    if stage_id == "S0":
        payload.update({
            "input_sha256": global_fingerprints["input_sha256"],
            "profile_sha256": global_fingerprints["profile_sha256"],
            "execution_options_sha256": global_fingerprints.get("execution_options_sha256"),
        })
    else:
        payload["upstream_receipt_sha256"] = upstream_receipt_sha256
    if stage_id == "S2":
        payload["locate_model_sha256"] = global_fingerprints["locate_model_sha256"]
        if global_fingerprints.get("yolo_model_sha256"):
            payload["yolo_model_sha256"] = global_fingerprints["yolo_model_sha256"]
    if stage_id == "S5":
        payload["qwen_model_sha256"] = global_fingerprints["qwen_model_sha256"]
        payload["profile_sha256"] = global_fingerprints["profile_sha256"]
    return canonical_sha256(payload)


def _count_rows(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path)) if path.is_file() else 0


def _code_fingerprint(config: ProjectConfig) -> str:
    """Hash every executable/configuration surface used by the packaged runner."""

    project = config.project_root.resolve()
    repository = config.repository_root.resolve()
    payload: dict[str, Any] = {
        # Hash the importable package, not setuptools' generated sibling
        # ``src/*.egg-info`` metadata.  An in-place wheel build refreshes that
        # directory even when no executable source changed; including it made
        # a completed run spuriously invalidate immediately after packaging.
        "src": hash_tree(
            project / "src" / "classroom_hybrid_ai",
            relative_to=project / "src",
        ),
        "scripts": hash_tree(project / "scripts"),
    }
    for path in (
        project / "pyproject.toml",
        repository / "requirements.txt",
        repository / "uv.lock",
        repository / "poetry.lock",
    ):
        if path.is_file():
            try:
                key = path.relative_to(repository).as_posix()
            except ValueError:
                key = str(path)
            payload[f"file:{key}"] = sha256_file(path)
    return canonical_sha256(payload)


class ResearchBackend:
    """The only formal run/resume API; HybridPipeline supplies internal helpers."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        stage_executor_factory: Callable[[ResearchStageSpec], Any] = ResearchStageExecutor,
    ) -> None:
        self.config = config
        self.stage_executor_factory = stage_executor_factory

    def profile(self, profile_id: str | None = None) -> ActionProfile:
        selected = profile_id or str(self.config.get("research.profile", "core8_v1"))
        return load_profile(self.config.action_profiles_path, selected)

    def plan(
        self,
        video_path: str | Path,
        *,
        profile_id: str | None = None,
        mode: str | None = None,
        average_tracks: int = 30,
    ) -> dict[str, Any]:
        info = probe_video(video_path)
        selected_mode = mode or str(self.config.get("research.mode", "production_adaptive"))
        scheduler = AdaptiveScreenshotScheduler(
            fps=info.fps,
            frame_count=info.frame_count,
            duration_seconds=info.duration_seconds,
            mode=selected_mode,
            locate_cap=int(self.config.get("locate_first.locate_cap", 1500)),
            reserved_change_events=int(self.config.get("locate_first.reserved_change_events", 300)),
            max_screenshots=(
                int(self.config.get("locate_first.exhaustive_max_screenshots", 300))
                if selected_mode == "exhaustive_validation" else None
            ),
        )
        rows = scheduler.schedule()
        profile = self.profile(profile_id)
        effective = _effective_config(self.config, profile=profile, mode=selected_mode)
        route = localization_route(effective)
        if type(average_tracks) is not int or average_tracks < 0:
            raise ValueError("average_tracks_must_be_nonnegative_integer")
        assumed_candidates = len(rows) * average_tracks
        qwen_cap = qwen_selection_budget(effective, info.duration_seconds, average_tracks, assumed_candidates)
        return {
            "schema_version": "classroom_hybrid_ai.plan.v3",
            "video": info.to_dict(),
            "profile": profile.to_dict(),
            "mode": selected_mode,
            "primary_detector": effective.get("pipeline.primary_detector"),
            "localization_route": route,
            "locate_required": route != "yolo_only",
            "effective_policy": effective_configuration_report(effective),
            "scheduler": scheduler.to_dict(),
            "estimated_screenshots": len(rows),
            "estimated_qwen_requests_lower_bound": 0,
            "estimated_qwen_requests_upper_bound": 0 if route == "yolo_only" else min(assumed_candidates, qwen_cap),
            "estimated_workload_assumptions": {"average_tracks": average_tracks,
                "change_anchors_not_yet_scanned": True, "ready_candidates_unknown_until_S4": route != "yolo_only",
                "student_confirmation": "unavailable_in_yolo_only" if route == "yolo_only" else "Locate_confirmation_required"},
            "qwen_request_policy": budget_policy(effective),
            "qwen_every_screenshot": budget_policy(effective)["selection_mode"] == "all_candidates",
            "qwen_context": {
                "messages_per_request": 1,
                "history_messages": 0,
                "cross_request_kv_cache_reused": False,
                "input_images": 1,
            },
            "gpu_order": (["YOLO", "worker_exit_ack"] if route != "locate_only" else []) +
                         (["LocateAnything", "unload_ack"] if route != "yolo_only" else []) + ["Qwen3.5-9B child process"],
            "simultaneous_gpu_models": 1,
            "stages": [
                {"stage_id": stage_id, "name": name, "weight_percent": weight}
                for stage_id, name, weight in STAGE_DEFINITIONS
            ],
        }

    def _build_spec(
        self,
        *,
        store: RunStore,
        source: Path,
        profile: ActionProfile,
        mode: str,
        max_frames: int | None,
        skip_qwen: bool,
        allow_incomplete_preflight: bool,
        effective_config: ProjectConfig,
        experiment_digest: str,
    ) -> ResearchStageSpec:
        info = probe_video(source)
        fingerprints = model_manifest_fingerprints(effective_config)
        return ResearchStageSpec(
            config=effective_config,
            store=store,
            source=source,
            video=info.to_dict(),
            profile=profile,
            mode=mode,
            max_frames=max_frames,
            skip_qwen=skip_qwen,
            allow_incomplete_preflight=allow_incomplete_preflight,
            experiment_digest=experiment_digest,
            input_sha256=sha256_file(source),
            config_sha256=project_config_digest(effective_config),
            code_sha256=_code_fingerprint(effective_config),
            locate_model_sha256=fingerprints["locate_model_sha256"],
            qwen_model_sha256=fingerprints["qwen_model_sha256"],
            yolo_model_sha256=fingerprints.get("yolo_model_sha256", ""),
        )

    @staticmethod
    def _fingerprints(spec: ResearchStageSpec) -> dict[str, str]:
        stages = stage_dependency_fingerprints(spec.config)
        return {
            "input_sha256": spec.input_sha256,
            "config_sha256": spec.config_sha256,
            "code_sha256": spec.code_sha256,
            "profile_sha256": spec.profile.sha256,
            "locate_model_sha256": spec.locate_model_sha256,
            "qwen_model_sha256": spec.qwen_model_sha256,
            "yolo_model_sha256": spec.yolo_model_sha256 if localization_route(spec.config) != "locate_only" else "",
            "stage_fingerprint_version": "v2",
            "execution_options_sha256": canonical_sha256({
                "mode": spec.mode, "max_frames": spec.max_frames,
                "allow_incomplete_preflight": spec.allow_incomplete_preflight,
            }),
            **{f"stage_code:{stage}": values["code_sha256"] for stage, values in stages.items()},
            **{f"stage_config:{stage}": canonical_sha256({
                "config_sha256": values["config_sha256"],
                **({"skip_qwen": spec.skip_qwen} if stage == "S5" else {}),
            }) for stage, values in stages.items()},
        }

    def _scan_receipts(self, spec: ResearchStageSpec) -> dict[str, Any]:
        upstream_sha: str | None = None
        valid: list[str] = []
        fingerprints = self._fingerprints(spec)
        for stage_id in STAGE_IDS:
            expected = _stage_input_fingerprint(
                stage_id,
                global_fingerprints=fingerprints,
                upstream_receipt_sha256=upstream_sha,
            )
            ok, reasons, receipt = verify_stage_receipt(
                spec.run_dir,
                stage_id,
                expected_input_fingerprint=expected,
                expected_fingerprint_version="v2",
            )
            if not ok or receipt is None:
                return {
                    "status": "resume_required",
                    "resume_from": stage_id,
                    "valid_stages": valid,
                    "invalidated_by": list(reasons),
                    "input_fingerprint_sha256": expected,
                    "upstream_receipt_sha256": upstream_sha,
                }
            valid.append(stage_id)
            upstream_sha = str(receipt["sha256"])
        return {
            "status": "complete",
            "resume_from": None,
            "valid_stages": valid,
            "invalidated_by": [],
            "upstream_receipt_sha256": upstream_sha,
        }

    def _result(self, spec: ResearchStageSpec, *, warning: bool = False) -> PipelineResult:
        events_path = spec.run_dir / "events.jsonl"
        event_rows = list(iter_jsonl(events_path)) if events_path.is_file() else []
        counts = {
            "observations": _count_rows(spec.run_dir / "artifacts" / "track_observations.jsonl"),
            "candidates": _count_rows(spec.run_dir / "artifacts" / "candidates.all.jsonl"),
            "confirmed": sum(str(row.get("status")) == "confirmed" for row in event_rows),
            "rejected": sum(str(row.get("status")) == "rejected" for row in event_rows),
            "uncertain": sum(str(row.get("status")) == "uncertain" for row in event_rows),
            "students": _count_rows(spec.run_dir / "students.jsonl"),
            "timeline_intervals": _count_rows(spec.run_dir / "action_intervals.jsonl"),
        }
        stages: dict[str, Any] = {}
        warnings: list[str] = []
        for stage_id in STAGE_IDS:
            receipt = read_stage_receipt(spec.run_dir, stage_id)
            if receipt is None:
                continue
            stages[stage_id] = receipt.get("stats", {})
            warnings.extend(str(item) for item in receipt.get("warnings", ()))
        return PipelineResult(
            status="warning" if warning or warnings else "ok",
            run_dir=str(spec.run_dir),
            video=dict(spec.video),
            counts=counts,
            stages=stages,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _execute(self, spec: ResearchStageSpec, *, resume: bool) -> PipelineResult:
        fingerprints = self._fingerprints(spec)
        state_path = spec.run_dir / "run_state.v1.json"
        if resume:
            if not state_path.is_file():
                raise ResearchBackendError("resume_run_state_missing")
            scan = self._scan_receipts(spec)
            if scan["status"] == "complete":
                state_payload = json.loads(state_path.read_text(encoding="utf-8"))
                state_status = str(state_payload.get("status", "unknown"))
                if state_status in {"completed", "warning"}:
                    return self._result(spec, warning=state_status == "warning")
                s6_receipt = read_stage_receipt(spec.run_dir, "S6")
                if s6_receipt is None:
                    raise ResearchBackendError("complete_receipt_scan_missing_s6")
                upstream_s6 = str(s6_receipt["sha256"])
                scan = {
                    "status": "resume_required",
                    "resume_from": "S7",
                    "valid_stages": list(STAGE_IDS[:-1]),
                    "invalidated_by": [f"run_state_status:{state_status}"],
                    "input_fingerprint_sha256": _stage_input_fingerprint(
                        "S7",
                        global_fingerprints=fingerprints,
                        upstream_receipt_sha256=upstream_s6,
                    ),
                    "upstream_receipt_sha256": upstream_s6,
                }
            start_stage = str(scan["resume_from"])
            upstream_sha = scan.get("upstream_receipt_sha256")
            ledger = StageLedger.open_existing(spec.run_dir)
            ledger.prepare_resume(
                start_stage,
                invalidated_by=tuple(scan["invalidated_by"]),
                metadata=fingerprints,
            )
            atomic_write_json(spec.run_dir / "artifacts" / "recovery_plan.v1.json", scan)
        else:
            spec.store.write_json("run_manifest.v1.json", spec.manifest_payload())
            ledger = StageLedger(
                spec.run_dir,
                run_id=spec.store.run_id,
                video_id=spec.store.video_id,
                total_units=int(spec.video.get("frame_count", 0)) or None,
                metadata={**fingerprints, "mode": spec.mode},
            )
            start_stage = "S0"
            upstream_sha = None
        executor = self.stage_executor_factory(spec)
        attach_progress_reporter = getattr(executor, "set_progress_reporter", None)
        if callable(attach_progress_reporter):
            attach_progress_reporter(
                lambda stage_id, processed, total, message: ledger.progress(
                    stage_id,
                    processed,
                    total,
                    message=message,
                )
            )
        start_index = STAGE_IDS.index(start_stage)
        accumulated_warnings: list[str] = []
        degraded = bool(spec.skip_qwen or spec.allow_incomplete_preflight)
        current_stage = start_stage
        try:
            for stage_id in STAGE_IDS[start_index:]:
                current_stage = stage_id
                expected_input = _stage_input_fingerprint(
                    stage_id,
                    global_fingerprints=fingerprints,
                    upstream_receipt_sha256=upstream_sha,
                )
                attempt_components: dict[str, Any] = {
                    "stage_id": stage_id,
                    "code_sha256": fingerprints[f"stage_code:{stage_id}"],
                    "config_sha256": fingerprints[f"stage_config:{stage_id}"],
                    "upstream_receipt_sha256": upstream_sha,
                }
                if stage_id == "S2":
                    attempt_components["locate_model_sha256"] = fingerprints["locate_model_sha256"]
                if stage_id == "S5":
                    attempt_components["qwen_model_sha256"] = fingerprints["qwen_model_sha256"]
                    attempt_components["profile_sha256"] = fingerprints["profile_sha256"]
                attempt = ledger.begin_attempt(
                    stage_id,
                    expected_input,
                    components=attempt_components,
                )
                ledger.begin(
                    stage_id,
                    total_units=None,
                    message=f"执行独立磁盘阶段 {stage_id}",
                    metadata={
                        "input_fingerprint_sha256": expected_input,
                        "partial_outputs_reusable": attempt["partial_outputs_reusable"],
                        "fingerprint_mismatch": list(attempt["fingerprint_mismatch"]),
                    },
                )
                outcome: StageOutcome = executor.execute(
                    stage_id,
                    input_fingerprint_sha256=expected_input,
                    partial_outputs_reusable=bool(attempt["partial_outputs_reusable"]),
                )
                if outcome.status in {"failed", "blocked"} or outcome.errors:
                    raise StageExecutionError(
                        f"stage_{stage_id.lower()}_{outcome.status}:" + ",".join(outcome.errors)
                    )
                if stage_id == "S2" and outcome.unload_ack is not True:
                    raise StageExecutionError("stage_s2_locate_unload_not_acknowledged")
                lifecycle = outcome.stats.get("lifecycle", {})
                if not isinstance(lifecycle, Mapping):
                    lifecycle = {}
                if stage_id == "S2" and lifecycle.get("owned_process"):
                    if outcome.exit_code != 0:
                        raise StageExecutionError(
                            f"stage_s2_locate_worker_exit_failed:{outcome.exit_code}"
                        )
                if stage_id == "S5" and bool(outcome.stats.get("model_started")):
                    release = outcome.stats.get("release", {})
                    if isinstance(release, Mapping):
                        cuda_release = release.get("cuda_after_release", {})
                        teardown_error = release.get("close_error") or (
                            cuda_release.get("teardown_error") if isinstance(cuda_release, Mapping) else None
                        )
                        if teardown_error:
                            raise StageExecutionError(f"stage_s5_qwen_teardown_failed:{teardown_error}")
                    if outcome.unload_ack is not True:
                        raise StageExecutionError("stage_s5_qwen_unload_not_acknowledged")
                    if isinstance(release, Mapping) and release.get("worker_isolated"):
                        if outcome.exit_code != 0:
                            raise StageExecutionError(
                                f"stage_s5_qwen_worker_exit_failed:{outcome.exit_code}"
                            )
                # The producer has stopped writing; one transaction may reuse
                # its hashes during commit verification. Recovery uses fresh reads.
                with hash_file_cache():
                    commit_hash_recorder = PerformanceRecorder(f"{stage_id}.commit_hash")
                    with recording(commit_hash_recorder), measured("commit.artifact_hash_seconds"):
                        references = artifact_refs(spec.run_dir, outcome.artifact_uris)
                    stage_preview = f"previews/{stage_id}/preview.json"
                    preview_uris = (
                        (stage_preview,)
                        if (spec.run_dir / stage_preview).is_file()
                        else ()
                    )
                    receipt = ledger.receipt(stage_id, {
                        "status": outcome.status,
                        "fingerprint_version": "v2",
                        "dependencies": {
                            "code_sha256": fingerprints[f"stage_code:{stage_id}"],
                            "config_sha256": fingerprints[f"stage_config:{stage_id}"],
                        },
                        "input_fingerprint_sha256": expected_input,
                        "upstream_receipt_sha256": upstream_sha,
                        "artifacts": [reference.to_dict() for reference in references],
                        "preview_uris": list(preview_uris),
                        # The observations live in the receipt, which is not
                        # one of its own artifacts. Do not rewrite an already
                        # hashed stage diagnostic to attach commit telemetry.
                        "stats": {**dict(outcome.stats), "commit_hash": commit_hash_recorder.payload()},
                        "warnings": list(outcome.warnings),
                        "errors": list(outcome.errors),
                        "model_loaded": outcome.model_loaded,
                        "model_pid": outcome.model_pid,
                        "unload_ack": outcome.unload_ack,
                        "exit_code": outcome.exit_code,
                    })
                    verified, reasons, _ = verify_stage_receipt(
                        spec.run_dir,
                        stage_id,
                        expected_input_fingerprint=expected_input,
                        expected_fingerprint_version="v2",
                        reuse_commit_hashes=True,
                    )
                if not verified:
                    raise StageExecutionError(
                        f"stage_receipt_verification_failed:{stage_id}:" + ",".join(reasons)
                    )
                processed = max(0, int(outcome.processed_units))
                total = outcome.total_units
                ledger.progress(
                    stage_id,
                    processed,
                    total,
                    message=f"{stage_id} 产物已原子写入并校验哈希",
                    preview_uris=preview_uris,
                    artifact_uris=tuple(reference.uri for reference in references),
                )
                ledger.finish(
                    stage_id,
                    status=outcome.status,
                    preview_uris=preview_uris,
                    artifact_uris=tuple(reference.uri for reference in references),
                    warnings=outcome.warnings,
                    model_loaded=outcome.model_loaded,
                    model_pid=outcome.model_pid,
                    unload_ack=outcome.unload_ack,
                    exit_code=outcome.exit_code,
                    metadata={"receipt_sha256": receipt["sha256"]},
                )
                accumulated_warnings.extend(outcome.warnings)
                if outcome.status == "warning":
                    if not outcome.warnings:
                        accumulated_warnings.append(f"{stage_id.lower()}_warning")
                    validity_impact = str(
                        outcome.stats.get("research_validity_impact", "blocking")
                    )
                    if validity_impact != "none" or any(
                        _warning_affects_research_validity(item)
                        for item in outcome.warnings
                    ):
                        degraded = True
                upstream_sha = str(receipt["sha256"])
            research_blocking_warnings = [
                item for item in accumulated_warnings
                if _warning_affects_research_validity(item)
            ]
            degraded = degraded or bool(research_blocking_warnings)
            research_validity = "pending" if degraded else "valid"
            ledger.complete_run(
                status="warning" if accumulated_warnings else "completed",
                research_validity=research_validity,
                warnings=tuple(dict.fromkeys(accumulated_warnings)),
                metadata={
                    "last_receipt_sha256": upstream_sha,
                    "research_blocking_warnings": research_blocking_warnings,
                    "delivery_only_warnings": [
                        item for item in accumulated_warnings
                        if not _warning_affects_research_validity(item)
                    ],
                },
            )
            return self._result(spec, warning=bool(accumulated_warnings))
        except BaseException as exc:
            error = f"{type(exc).__name__}:{exc}"[:2000]
            try:
                stage_state = ledger.state.stages.get(current_stage)
                if stage_state is not None and stage_state.status == "running":
                    ledger.finish(current_stage, status="failed", errors=(error,))
                ledger.complete_run(
                    status="failed",
                    research_validity="invalid",
                    errors=(error, "resume_from_first_unverified_stage"),
                )
            finally:
                raise

    def run(
        self,
        video_path: str | Path,
        *,
        profile_id: str | None = None,
        mode: str | None = None,
        max_frames: int | None = None,
        allow_incomplete_preflight: bool = False,
        skip_qwen: bool = False,
        watch: bool = False,
    ) -> PipelineResult:
        del watch
        source = Path(video_path).resolve()
        selected_profile = self.profile(profile_id)
        selected_mode = mode or str(self.config.get("research.mode", "production_adaptive"))
        effective_config = _effective_config(self.config, profile=selected_profile, mode=selected_mode)
        digest = _experiment_digest(
            effective_config,
            source,
            selected_profile,
            mode=selected_mode,
            max_frames=max_frames,
            skip_qwen=skip_qwen,
        )
        store = RunStore(effective_config.output_root, source.stem, digest, unique_run=True)
        spec = self._build_spec(
            store=store,
            source=source,
            profile=selected_profile,
            mode=selected_mode,
            max_frames=max_frames,
            skip_qwen=skip_qwen,
            allow_incomplete_preflight=allow_incomplete_preflight,
            effective_config=effective_config,
            experiment_digest=digest,
        )
        return self._execute(spec, resume=False)

    def resume(
        self,
        run_dir: str | Path,
        *,
        allow_incomplete_preflight: bool = False,
    ) -> PipelineResult:
        store = RunStore.open(run_dir)
        manifest_path = store.run_dir / "run_manifest.v1.json"
        if not manifest_path.is_file():
            raise ResearchBackendError("resume_run_manifest_missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_payload = manifest.get("source") if isinstance(manifest.get("source"), Mapping) else {}
        source_value = source_payload.get("path")
        if not source_value:
            raise ResearchBackendError("resume_source_path_missing")
        source = Path(str(source_value)).resolve()
        profile_payload = manifest.get("profile") if isinstance(manifest.get("profile"), Mapping) else {}
        profile_id = str(profile_payload.get("profile_id", ""))
        selected_profile = self.profile(profile_id)
        selected_mode = str(manifest.get("mode") or self.config.get("research.mode", "production_adaptive"))
        effective_config = _effective_config(self.config, profile=selected_profile, mode=selected_mode)
        max_frames = manifest.get("max_frames")
        skip_qwen = bool(manifest.get("skip_qwen", False))
        digest = _experiment_digest(
            effective_config,
            source,
            selected_profile,
            mode=selected_mode,
            max_frames=max_frames,
            skip_qwen=skip_qwen,
        )
        spec = self._build_spec(
            store=store,
            source=source,
            profile=selected_profile,
            mode=selected_mode,
            max_frames=max_frames,
            skip_qwen=skip_qwen,
            allow_incomplete_preflight=(
                allow_incomplete_preflight or bool(manifest.get("development_degraded", False))
            ),
            effective_config=effective_config,
            experiment_digest=digest,
        )
        return self._execute(spec, resume=True)


def status_payload(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    state = json.loads((root / "run_state.v1.json").read_text(encoding="utf-8"))
    last_event: dict[str, Any] | None = None
    count = 0
    events_path = root / "stage_events.v1.jsonl"
    if events_path.is_file():
        for row in iter_jsonl(events_path):
            last_event = row
            count += 1
    return {"run_dir": str(root), "state": state, "last_event": last_event, "event_count": count}


__all__ = ["ResearchBackend", "ResearchBackendError", "status_payload"]
