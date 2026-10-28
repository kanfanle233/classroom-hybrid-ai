"""Independent disk-backed S0-S7 implementations for the v3.1 backend.

Each method deliberately reloads its upstream artifacts from ``run_dir``.  No
model output is passed between methods in Python memory, which makes the stage
receipt the real recovery boundary instead of a progress annotation around one
monolithic inference call.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .adapters import LocateAnythingAdapter, Qwen35Adapter
from .config import ProjectConfig
from .contracts import CandidateEvent, ReviewDecision, TrackObservation
from .evidence import (
    build_event_evidence,
    build_temporal_composite,
    extract_frames_at_times,
    filter_observations_for_candidates,
)
from .fusion import StrictFusionPolicy
from .hashing import canonical_sha256, sha256_file
from .locate_service import LocateServiceError, LocateServiceSession
from .locate_tracking import (
    associate_locate_rows,
    candidates_from_locate_observations,
    materialize_scheduled_screenshots,
    render_locate_student_annotations,
    student_id_sort_key,
)
from .open_vocab import session_vocabulary_digest
from .outputs_v3 import (
    build_action_observations,
    build_v3_outputs,
    normalize_intervals,
    taxonomy_payload,
    write_stage_preview,
)
from .pipeline import (
    PipelineError,
    _QwenCallBudget,
    _classification_prompt,
    fair_candidate_selection,
    _locate_pass,
    _make_qwen_adapter,
    _prepare_session_vocabulary,
    _qwen_pass,
    _target_highlighted,
    _validate_classification_response,
    _write_open_vocab_outputs,
)
from .preflight import run_preflight
from .performance import PerformanceRecorder, measured, recording
from .profiles import ActionProfile
from .research_policy import localization_route, budget_policy, qwen_selection_budget, effective_configuration_report
from .runtime import GpuLease, release_model
from .stage_runtime import AdaptiveScreenshotScheduler, runtime_fingerprint, utc_now
from .storage import RunStore, atomic_write_json, atomic_write_jsonl, iter_jsonl
from .video import detect_cpu_change_anchors, scan_and_materialize_screenshots


class StageExecutionError(RuntimeError):
    """A stage could not produce a research-valid disk boundary."""


@dataclass(frozen=True, slots=True)
class StageOutcome:
    status: str = "completed"
    artifact_uris: tuple[str, ...] = ()
    processed_units: int = 1
    total_units: int | None = 1
    stats: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    model_loaded: bool = False
    model_pid: int | None = None
    unload_ack: bool | None = None
    exit_code: int | None = None


@dataclass(frozen=True, slots=True)
class ResearchStageSpec:
    config: ProjectConfig
    store: RunStore
    source: Path
    video: Mapping[str, Any]
    profile: ActionProfile
    mode: str
    max_frames: int | None
    skip_qwen: bool
    allow_incomplete_preflight: bool
    experiment_digest: str
    input_sha256: str
    config_sha256: str
    code_sha256: str
    locate_model_sha256: str
    qwen_model_sha256: str
    yolo_model_sha256: str = ""

    @property
    def run_dir(self) -> Path:
        return self.store.run_dir

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "classroom_hybrid_ai.run_manifest.v1",
            "run_id": self.store.run_id,
            "video_id": self.store.video_id,
            "created_at": utc_now(),
            "source": {
                **dict(self.video),
                "path": str(self.source),
                "sha256": self.input_sha256,
                "size_bytes": self.source.stat().st_size,
            },
            "profile": self.profile.to_dict(),
            "profile_sha256": self.profile.sha256,
            "mode": self.mode,
            "max_frames": self.max_frames,
            "skip_qwen": self.skip_qwen,
            "development_degraded": self.allow_incomplete_preflight,
            "experiment_digest": self.experiment_digest,
            "effective_policy": effective_configuration_report(self.config),
            "fingerprints": {
                "input_sha256": self.input_sha256,
                "config_sha256": self.config_sha256,
                "code_sha256": self.code_sha256,
                "profile_sha256": self.profile.sha256,
                "locate_model_sha256": self.locate_model_sha256,
                "qwen_model_sha256": self.qwen_model_sha256,
                "yolo_model_sha256": self.yolo_model_sha256,
            },
            "runtime": runtime_fingerprint(),
        }


def model_manifest_fingerprints(config: ProjectConfig) -> dict[str, str]:
    """Use pinned model receipts without rereading multi-gigabyte weights."""

    path = config.model_manifest_path
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    models = payload.get("models", {}) if isinstance(payload, Mapping) else {}

    def one(model_id: str, configured_path: Path | None) -> str:
        raw = models.get(model_id, {}) if isinstance(models, Mapping) else {}
        row = dict(raw) if isinstance(raw, Mapping) else {}
        package_value = row.get("package_manifest")
        package_path = config.resolve(package_value) if package_value else None
        package_sha = sha256_file(package_path) if package_path is not None and package_path.is_file() else None
        weight_files: list[dict[str, Any]] = []
        if configured_path is not None and configured_path.is_file():
            stat = configured_path.stat()
            weight_files.append({
                "path": configured_path.name,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
        elif configured_path is not None and configured_path.is_dir():
            suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
            for weight in sorted(
                item for item in configured_path.rglob("*")
                if item.is_file() and item.suffix.lower() in suffixes
            ):
                stat = weight.stat()
                weight_files.append({
                    "path": weight.relative_to(configured_path).as_posix(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })
        return canonical_sha256({
            "model_id": model_id,
            "repository": row.get("repository"),
            "revision": row.get("revision"),
            "declared_sha256": row.get("sha256"),
            "weights_bytes": row.get("weights_bytes"),
            "package_manifest_sha256": package_sha,
            "configured_path": None if configured_path is None else str(configured_path.resolve()),
            "weight_files": weight_files,
        })

    return {
        "locate_model_sha256": one("locate_anything", config.model_path("locate_anything.model_dir")),
        "qwen_model_sha256": one("qwen35_9b_bnb4", config.model_path("qwen.model_dir")),
        "yolo_model_sha256": one(str(config.get("localization.model_manifest_key", "yolo26_pose")),
                                 config.model_path("yolo.tracker_model")),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _write_contact_sheet(rows: Sequence[Mapping[str, Any]], output: Path) -> bool:
    if not rows:
        return False
    selected = [rows[min(len(rows) - 1, round(index * (len(rows) - 1) / 11))] for index in range(12)]
    tiles: list[np.ndarray] = []
    for row in selected:
        image = cv2.imread(str(row.get("path", "")))
        if image is None:
            continue
        tile = cv2.resize(image, (240, 135), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            f"f={int(row.get('frame_index', 0))} t={float(row.get('pts_seconds', 0.0)):.1f}s",
            (4, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return False
    while len(tiles) < 12:
        tiles.append(tiles[-1].copy())
    sheet = cv2.vconcat([cv2.hconcat(tiles[row:row + 4]) for row in range(0, 12, 4)])
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]))


def _profile_vocabulary(spec: ResearchStageSpec) -> tuple[Any, tuple[Any, ...], str]:
    spec.store.write_jsonl("artifacts/open_vocabulary_discoveries.jsonl", [])
    registry, vocabulary, _ = _prepare_session_vocabulary(
        spec.config,
        spec.store,
        discovery_reviews=(),
        candidates=(),
        action_profile=spec.profile,
    )
    digest = session_vocabulary_digest(vocabulary, registry_sha256=registry.sha256)
    return registry, tuple(vocabulary), digest


class ResearchStageExecutor:
    """Default real stage implementation; factories remain injectable for tests."""

    def __init__(
        self,
        spec: ResearchStageSpec,
        *,
        locate_factory: Callable[..., Any] = LocateAnythingAdapter,
        qwen_factory: Callable[..., Any] = Qwen35Adapter,
        locate_session_factory: Callable[..., Any] = LocateServiceSession,
        preflight_runner: Callable[..., Mapping[str, Any]] = run_preflight,
        yolo_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.spec = spec
        self.locate_factory = locate_factory
        self.yolo_factory = yolo_factory
        self.qwen_factory = qwen_factory
        self.locate_session_factory = locate_session_factory
        self.preflight_runner = preflight_runner
        self._progress_reporter: Callable[[str, int, int | None, str], None] | None = None

    def set_progress_reporter(
        self,
        reporter: Callable[[str, int, int | None, str], None] | None,
    ) -> None:
        """Attach the run ledger without making stage implementations own it."""

        self._progress_reporter = reporter

    def _report_progress(
        self,
        stage_id: str,
        processed_units: int,
        total_units: int | None,
        message: str,
    ) -> None:
        if self._progress_reporter is not None:
            self._progress_reporter(stage_id, processed_units, total_units, message)

    def _commit_preview(self, stage_id: str) -> tuple[str, ...]:
        uris = tuple(write_stage_preview(
            self.spec.run_dir,
            stage_id,
            info=self.spec.video,
            preserve_existing_stage_previews=False,
        ))
        preview = self.spec.run_dir / "previews" / stage_id / "preview.json"
        if not preview.is_file():
            raise StageExecutionError(f"stage_preview_missing:{stage_id}")
        return uris

    def execute(
        self,
        stage_id: str,
        *,
        input_fingerprint_sha256: str,
        partial_outputs_reusable: bool = False,
    ) -> StageOutcome:
        method = getattr(self, f"_run_{stage_id.lower()}", None)
        if not callable(method):
            raise KeyError(f"unknown_stage:{stage_id}")
        recorder = PerformanceRecorder(stage_id)
        diagnostic_uri = f"diagnostics/performance/{stage_id}.json"
        try:
            with recording(recorder), measured("stage.wall_seconds"):
                outcome = method(
                    input_fingerprint_sha256=input_fingerprint_sha256,
                    partial_outputs_reusable=partial_outputs_reusable,
                )
        finally:
            atomic_write_json(self.spec.run_dir / diagnostic_uri, {
                **recorder.payload(), "input_fingerprint_sha256": input_fingerprint_sha256,
            })
        return replace(outcome, artifact_uris=(*outcome.artifact_uris, diagnostic_uri))

    def _run_s0(self, **_: Any) -> StageOutcome:
        preflight = dict(self.preflight_runner(
            self.spec.config,
            verify_hashes=False,
        ))
        self.spec.store.write_json("preflight.json", preflight)
        self.spec.store.write_json("run_manifest.v1.json", self.spec.manifest_payload())
        if preflight.get("status") != "ok" and not self.spec.allow_incomplete_preflight:
            raise StageExecutionError(
                "preflight_failed:" + ",".join(map(str, preflight.get("failed_checks", ())))
            )
        preview_dir = self.spec.run_dir / "previews" / "S0"
        duration = float(self.spec.video.get("duration_seconds", 0.0) or 0.0)
        fps = float(self.spec.video.get("fps", 1.0) or 1.0)
        targets = (0.0, duration / 2.0, max(0.0, duration - 1.0 / fps))
        packets = extract_frames_at_times(
            self.spec.source, targets,
            decode_strategy=str(self.spec.config.get("video.preview_decode_strategy", "sequential")),
        )
        preview_uris: list[str] = []
        labels = ("first", "middle", "last")
        for label, target in zip(labels, targets):
            packet = packets.get(target)
            if packet is None:
                continue
            frame_index, pts, image = packet
            path = preview_dir / f"input_{label}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise StageExecutionError(f"input_preview_write_failed:{path}")
            preview_uris.append(path.relative_to(self.spec.run_dir).as_posix())
        atomic_write_json(preview_dir / "preview.json", {
            "schema_version": "classroom_hybrid_ai.preview.v1",
            "stage_id": "S0",
            "uris": preview_uris,
        })
        preview_uris = list(self._commit_preview("S0"))
        warnings = tuple(str(item) for item in preflight.get("warnings", ()))
        if preflight.get("status") != "ok":
            warnings += ("development_preflight_incomplete",)
        return StageOutcome(
            status="warning" if warnings else "completed",
            artifact_uris=("run_manifest.v1.json", "preflight.json", "previews/S0"),
            stats={"preflight_status": preflight.get("status"), "preview_count": len(preview_uris)},
            warnings=warnings,
        )

    def _run_s1(self, *, input_fingerprint_sha256: str, **_: Any) -> StageOutcome:
        manifest = _read_json(self.spec.run_dir / "run_manifest.v1.json")
        video = manifest["source"]
        source_frame_units = max(1, int(video["frame_count"]))
        adaptive = str(manifest["mode"]) == "production_adaptive"
        shared_scan = adaptive and bool(self.spec.config.get("video.shared_screenshot_scan", False))
        s1_total_units = source_frame_units * (2 if adaptive and not shared_scan else 1)
        self._report_progress(
            "S1",
            0,
            s1_total_units,
            f"S1 视频扫描与截图落盘 0/{s1_total_units}",
        )
        scheduler = AdaptiveScreenshotScheduler(
            fps=float(video["fps"]),
            frame_count=int(video["frame_count"]),
            duration_seconds=float(video["duration_seconds"]),
            mode=str(manifest["mode"]),
            locate_cap=int(self.spec.config.get("locate_first.locate_cap", 1500)),
            reserved_change_events=int(self.spec.config.get("locate_first.reserved_change_events", 300)),
            max_screenshots=(
                int(self.spec.config.get("locate_first.exhaustive_max_screenshots", 300))
                if manifest["mode"] == "exhaustive_validation" else None
            ),
        )
        change_anchors: list[dict[str, Any]] = []
        if adaptive and not shared_scan:
            def change_progress(processed: int, total: int | None) -> None:
                denominator = max(1, int(total or source_frame_units))
                normalized = min(
                    source_frame_units,
                    int(round(min(processed, denominator) / denominator * source_frame_units)),
                )
                self._report_progress(
                    "S1",
                    normalized,
                    s1_total_units,
                    f"S1 CPU 画面变化扫描 {min(processed, denominator)}/{denominator}",
                )

            change_anchors = detect_cpu_change_anchors(
                self.spec.source,
                fps=float(video["fps"]),
                frame_count=int(video["frame_count"]),
                duration_seconds=float(video["duration_seconds"]),
                threshold=float(self.spec.config.get("locate_first.change_anchor_threshold", 0.08)),
                sample_stride=int(self.spec.config.get("locate_first.change_anchor_sample_stride", 10)),
                minimum_gap_seconds=float(self.spec.config.get("locate_first.change_anchor_min_gap_seconds", 0.5)),
                maximum_anchors=min(
                    300,
                    int(self.spec.config.get("locate_first.reserved_change_events", 300)),
                ),
                progress_callback=change_progress,
            )
        requested = scheduler.schedule(change_anchors=change_anchors)
        screenshot_uri = f"artifacts/locate_screenshots/{input_fingerprint_sha256[:16]}"

        def screenshot_progress(processed: int, total: int) -> None:
            denominator = max(1, int(total))
            phase_offset = source_frame_units if adaptive and not shared_scan else 0
            normalized = phase_offset + min(
                source_frame_units,
                int(round(min(processed, denominator) / denominator * source_frame_units)),
            )
            self._report_progress(
                "S1",
                normalized,
                s1_total_units,
                f"S1 调度截图已落盘 {min(processed, denominator)}/{denominator}",
            )

        if shared_scan:
            rows, sampling, change_anchors = scan_and_materialize_screenshots(
                self.spec.source, self.spec.run_dir / screenshot_uri,
                scheduler=scheduler,
                maximum_cache_bytes=int(self.spec.config.get("video.screenshot_cache_bytes", 2 * 1024**3)),
                max_frames=self.spec.max_frames,
                jpeg_quality=int(self.spec.config.get("locate_first.jpeg_quality", 95)),
                change_options={
                    "threshold": float(self.spec.config.get("locate_first.change_anchor_threshold", 0.08)),
                    "sample_stride": int(self.spec.config.get("locate_first.change_anchor_sample_stride", 10)),
                    "minimum_gap_seconds": float(self.spec.config.get("locate_first.change_anchor_min_gap_seconds", 0.5)),
                    "maximum_anchors": min(300, int(self.spec.config.get("locate_first.reserved_change_events", 300))),
                },
                progress_callback=screenshot_progress,
            )
        else:
            rows, sampling = materialize_scheduled_screenshots(
                self.spec.source,
                self.spec.run_dir / screenshot_uri,
                requested,
                max_frames=self.spec.max_frames,
                jpeg_quality=int(self.spec.config.get("locate_first.jpeg_quality", 95)),
                progress_callback=screenshot_progress,
            )
        self.spec.store.write_jsonl("artifacts/screenshot_schedule.v1.jsonl", rows)
        self.spec.store.write_jsonl("artifacts/change_anchors.v1.jsonl", change_anchors)
        self.spec.store.write_json("artifacts/screenshot_schedule.v1.json", {
            "schema_version": "classroom_hybrid_ai.screenshot_schedule.v1",
            "mode": self.spec.mode,
            "scheduler": scheduler.to_dict(),
            "sampling": sampling,
            "change_anchor_policy": {
                "detector": "cpu_global_frame_difference" if str(manifest["mode"]) == "production_adaptive" else "not_applicable",
                "requested": len(change_anchors),
                "reserved": scheduler.reserved_change_events,
                "upper_bound": min(300, scheduler.reserved_change_events),
                "used": sum(
                    1 for row in rows
                    if str(row.get("anchor_source", "")).startswith("cpu_")
                    or "change" in str(row.get("anchor_source", ""))
                ),
            },
            "rows": rows,
        })
        contact_uri = "previews/S1/contact_sheet_12.jpg"
        contact_written = _write_contact_sheet(rows, self.spec.run_dir / contact_uri)
        atomic_write_json(self.spec.run_dir / "previews" / "S1" / "preview.json", {
            "schema_version": "classroom_hybrid_ai.preview.v1",
            "stage_id": "S1",
            "uris": [contact_uri] if contact_written else [],
        })
        self._commit_preview("S1")
        artifacts = [
            "artifacts/screenshot_schedule.v1.json",
            "artifacts/screenshot_schedule.v1.jsonl",
            "artifacts/change_anchors.v1.jsonl",
            screenshot_uri,
            "previews/S1",
        ]
        return StageOutcome(
            artifact_uris=tuple(artifacts),
            processed_units=s1_total_units,
            total_units=s1_total_units,
            stats={
                "scheduler": scheduler.to_dict(),
                "sampling": sampling,
                "change_anchor_detector": "cpu_global_frame_difference" if str(manifest["mode"]) == "production_adaptive" else "not_applicable",
                "change_anchors_requested": len(change_anchors),
                "change_anchors_used": sum(
                    1 for row in rows
                    if str(row.get("anchor_source", "")).startswith("cpu_")
                    or "change" in str(row.get("anchor_source", ""))
                ),
                "change_anchor_upper_bound": min(300, scheduler.reserved_change_events),
                "locate_screenshot_cap": scheduler.cap,
                "progress_unit": "decoded_frame_equivalent_work_units",
                "progress_total_units": s1_total_units,
            },
            warnings=("scheduled_frames_missing",) if sampling.get("missing_frame_indices") else (),
        )

    def _run_s2(self, *, input_fingerprint_sha256: str = "", partial_outputs_reusable: bool = False, **_: Any) -> StageOutcome:
        route = localization_route(self.spec.config)
        if route == "locate_only":
            outcome = self._run_locate_s2(input_fingerprint_sha256=input_fingerprint_sha256,
                partial_outputs_reusable=partial_outputs_reusable, commit_preview=False)
            raw_rows = list(iter_jsonl(self.spec.run_dir / "artifacts/locate_raw_results.jsonl"))
            self.spec.store.write_jsonl("artifacts/localization_raw_results.v1.jsonl", [
                {**row, "localization_route": route, "source": "locate_anything"} for row in raw_rows])
            self.spec.store.write_json("artifacts/localization_routing.v1.json", {
                "route": route, "actual_locate_frames": len(raw_rows)})
            self._commit_preview("S2")
            return replace(outcome, artifact_uris=(*outcome.artifact_uris,
                "artifacts/localization_raw_results.v1.jsonl", "artifacts/localization_routing.v1.json"))
        from .localization import run_yolo_screenshots, build_difficulty_queue, merge_localization_rows
        scenes = list(iter_jsonl(self.spec.run_dir / "artifacts/screenshot_schedule.v1.jsonl"))
        if not scenes:
            raise StageExecutionError("s2_schedule_empty")
        model = self.spec.config.model_path("yolo.tracker_model")
        if model is None:
            raise StageExecutionError("yolo_model_missing")
        manifest_path = self.spec.config.model_manifest_path
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        models = manifest.get("models", {}) if isinstance(manifest, Mapping) else {}
        manifest_key = self.spec.config.get("localization.model_manifest_key", "yolo26_pose")
        if not isinstance(manifest_key, str) or not manifest_key.strip():
            raise StageExecutionError("yolo_model_manifest_key_missing_or_invalid")
        lock = models.get(manifest_key, {}) if isinstance(models, Mapping) else {}
        expected_sha = lock.get("sha256") if isinstance(lock, Mapping) else None
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in expected_sha)):
            raise StageExecutionError("yolo_locked_weight_sha256_missing_or_invalid")
        locked_path = self.spec.config.resolve(lock.get("path"))
        if locked_path is None or locked_path != model.resolve():
            raise StageExecutionError("yolo_configured_model_differs_from_locked_path")
        lease_path = self.spec.config.resolve(".runtime/classroom_hybrid_ai/gpu.lock")
        with GpuLease(lease_path):
            yolo_rows, yolo_stage = run_yolo_screenshots(scenes, model_path=model,
                model_sha256=expected_sha.lower(),
                cache_dir=self.spec.run_dir / "responses/yolo",
                code_fingerprint=input_fingerprint_sha256,
                imgsz=int(self.spec.config.get("yolo.image_size", 960)),
                conf=float(self.spec.config.get("yolo.confidence", .20)),
                batch_size=int(self.spec.config.get("localization.batch_size", 1)),
                device=self.spec.config.get("runtime.device", "cuda:0"),
                config_dir=self.spec.config.resolve(self.spec.config.get("runtime.yolo_config_dir", ".runtime/ultralytics")),
                reuse_responses=partial_outputs_reusable,
                adapter_factory=self.yolo_factory,
                progress_callback=lambda done, total: self._report_progress("S2", done, total, f"S2 YOLO {done}/{total}"))
        yolo_stage["model_manifest_key"] = manifest_key
        queue, routing = build_difficulty_queue(yolo_rows,
            audit_interval_seconds=float(self.spec.config.get("localization.audit_interval_seconds", 30)),
            low_confidence=float(self.spec.config.get("localization.low_confidence", .28)),
            association_kwargs=self._association_parameters())
        if route == "yolo_only":
            queue = []
        self.spec.store.write_jsonl("artifacts/yolo_raw_results.v1.jsonl", yolo_rows)
        self.spec.store.write_jsonl("artifacts/localization_difficulty_queue.v1.jsonl", queue)
        self.spec.store.write_json("diagnostics/yolo_stage.v1.json", yolo_stage)
        if queue:
            locate_outcome = self._run_locate_s2(input_fingerprint_sha256=input_fingerprint_sha256,
                partial_outputs_reusable=partial_outputs_reusable, scenes=queue, commit_preview=False)
            locate_rows = list(iter_jsonl(self.spec.run_dir / "artifacts/locate_raw_results.jsonl"))
            locate_stage = dict(locate_outcome.stats)
        else:
            locate_rows = []
            locate_stage = {"status": "ok", "requests": 0, "reason": "no_locate_requests_for_route"}
            self.spec.store.write_jsonl("artifacts/locate_raw_results.jsonl", [])
            self.spec.store.write_json("diagnostics/locate_stage.v1.json", locate_stage)
            self.spec.store.write_json("diagnostics/locate_service_receipt.v1.json", {"owned_process": False, "reason": "not_started"})
        rows = merge_localization_rows(yolo_rows, locate_rows, queue, route=route)
        self.spec.store.write_jsonl("artifacts/localization_raw_results.v1.jsonl", rows)
        self.spec.store.write_json("artifacts/localization_routing.v1.json", {"route": route, **routing,
            "actual_locate_frames": len(queue), "yolo": yolo_stage, "locate": locate_stage})
        self._commit_preview("S2")
        return StageOutcome(artifact_uris=("artifacts/localization_raw_results.v1.jsonl",
            "artifacts/yolo_raw_results.v1.jsonl", "artifacts/locate_raw_results.jsonl",
            "artifacts/localization_difficulty_queue.v1.jsonl", "artifacts/localization_routing.v1.json",
            "diagnostics/yolo_stage.v1.json", "diagnostics/locate_stage.v1.json",
            "diagnostics/locate_service_receipt.v1.json", "previews/S2"),
            stats={"route": route, "yolo": yolo_stage, "locate": locate_stage,
                "requests": locate_stage.get("requests", 0), "detections": sum(len(row.get("detections", [])) for row in rows),
                "lifecycle": {"owned_process": False, "sequential_model_teardown_verified": True}},
            processed_units=len(scenes), total_units=len(scenes), unload_ack=True, exit_code=0)

    def _association_parameters(self) -> dict[str, Any]:
        return {key: float(self.spec.config.get(f"locate_first.{key}", value)) for key, value in {
            "max_gap_seconds": 2.0, "minimum_iou": .10, "maximum_center_distance": .75,
            "minimum_match_score": .30, "minimum_match_margin": .10}.items()}

    def _run_locate_s2(self, *, input_fingerprint_sha256: str = "", partial_outputs_reusable: bool = False,
                       scenes: list[dict[str, Any]] | None = None, commit_preview: bool = True) -> StageOutcome:
        if scenes is None:
            scenes = list(iter_jsonl(self.spec.run_dir / "artifacts" / "screenshot_schedule.v1.jsonl"))
        if not scenes:
            raise StageExecutionError("s2_schedule_empty")
        self._report_progress(
            "S2",
            0,
            len(scenes),
            f"S2 Locate 截图已处理 0/{len(scenes)}",
        )
        lease_path = self.spec.config.resolve(".runtime/classroom_hybrid_ai/gpu.lock")
        assert lease_path is not None
        service = self.locate_session_factory(self.spec.config, self.spec.run_dir)
        stage: dict[str, Any]
        try:
            with GpuLease(lease_path):
                with service:
                    stage = _locate_pass(
                        self.spec.config,
                        scenes,
                        self.spec.store,
                        skip=False,
                        adapter_factory=self.locate_factory,
                        request_fingerprint=input_fingerprint_sha256 or None,
                        reuse_responses=partial_outputs_reusable,
                        progress_callback=lambda processed, total: self._report_progress(
                            "S2",
                            processed,
                            total,
                            f"S2 Locate 截图已处理 {processed}/{total}",
                        ),
                    )
        except LocateServiceError as exc:
            raise StageExecutionError(f"locate_service_prepare_failed:{type(exc).__name__}:{exc}") from exc
        service_receipt = dict(service.receipt)
        # Preserve the detector-owned boundary.  S3 adds persistent IDs to a
        # separate associated artifact and must never mutate S2's raw output.
        raw_rows = list(iter_jsonl(self.spec.run_dir / "artifacts" / "locate_results.jsonl"))
        self.spec.store.write_jsonl("artifacts/locate_raw_results.jsonl", raw_rows)
        atomic_write_json(self.spec.run_dir / "diagnostics" / "locate_service_receipt.v1.json", service_receipt)
        atomic_write_json(self.spec.run_dir / "diagnostics" / "locate_stage.v1.json", stage)
        detections = int(stage.get("detections", 0))
        if (stage.get("status") != "ok" or len(raw_rows) != len(scenes)
                or any(row.get("status") != "ok" for row in raw_rows)):
            raise StageExecutionError("locate_stage_output_failed_or_incomplete")
        backend = str(self.spec.config.get("locate_anything.backend", "http"))
        remote = stage.get("remote_unload")
        teardown = stage.get("teardown") if isinstance(stage.get("teardown"), Mapping) else {}
        unload_ack = (
            isinstance(remote, Mapping)
            and remote.get("status") in {"ok", "unloaded"}
            and remote.get("loaded") is False
        ) if backend == "http" else teardown.get("close_error") in {None, ""}
        if not unload_ack:
            raise StageExecutionError("locate_unload_ack_missing")
        exit_code = service_receipt.get("exitcode")
        if service_receipt.get("owned_process") and exit_code != 0:
            raise StageExecutionError(f"locate_worker_exit_failed:{exit_code}")
        lifecycle = {
            "backend": backend,
            "remote_unload_acknowledged": bool(unload_ack) if backend == "http" else None,
            "local_adapter_close_acknowledged": bool(unload_ack) if backend != "http" else None,
            "owned_process": bool(service_receipt.get("owned_process")),
            "worker_exitcode": exit_code,
            "model_started": True,
        }
        if commit_preview:
            self._commit_preview("S2")
        return StageOutcome(
            artifact_uris=(
                "artifacts/locate_raw_results.jsonl",
                "diagnostics/locate_service_receipt.v1.json",
                "diagnostics/locate_stage.v1.json",
                "previews/S2",
            ),
            processed_units=len(scenes),
            total_units=len(scenes),
            stats={**stage, "lifecycle": lifecycle},
            model_loaded=False,
            model_pid=service_receipt.get("pid"),
            unload_ack=unload_ack,
            exit_code=exit_code,
        )

    def _run_s3(self, **_: Any) -> StageOutcome:
        route = localization_route(self.spec.config)
        generic = route != "locate_only"
        raw_rows = list(iter_jsonl(self.spec.run_dir / "artifacts" / (
            "localization_raw_results.v1.jsonl" if generic else "locate_raw_results.jsonl")))
        if generic:
            from .localization import associate_localization_rows, render_localization_annotations
            observations, locate_rows, boxes, stage = associate_localization_rows(raw_rows, route=route,
                **self._association_parameters())
        else:
            observations, locate_rows, boxes, stage = associate_locate_rows(raw_rows, **self._association_parameters())
        # Valid empty detections are a complete observation, not a failed model.
        (self.spec.run_dir / "artifacts/locate_annotated").mkdir(parents=True, exist_ok=True)
        render_annotations = render_localization_annotations if generic else render_locate_student_annotations
        locate_rows, visual = render_annotations(
            locate_rows,
            self.spec.run_dir / "artifacts" / "locate_annotated",
            jpeg_quality=int(self.spec.config.get("locate_first.jpeg_quality", 95)),
        )
        annotated_by_frame = {
            int(row["frame_index"]): str(row["annotated_path"])
            for row in locate_rows if row.get("annotated_path")
        }
        boxes = [{**row, "annotated_path": annotated_by_frame.get(int(row["frame_index"]))} for row in boxes]
        # Keep every detected person available for detection/identity audit.
        # Only confirmed student observations cross the production S3 boundary
        # into the student registry, fair candidate selection and Qwen.
        self.spec.store.write_jsonl("artifacts/localization_boxes.v1.jsonl", boxes)
        self.spec.store.write_jsonl("artifacts/localization_track_observations.v1.jsonl", observations)
        diagnostic_observations = len(observations)
        if generic:
            observations = [row for row in observations if row.metadata.get("student_confirmed") is True]
            boxes = [row for row in boxes if row.get("student_confirmed") is True]
        self.spec.store.write_jsonl("artifacts/locate_results.jsonl", locate_rows)
        self.spec.store.write_jsonl("artifacts/localization_results.v1.jsonl", locate_rows)
        self.spec.store.write_jsonl("artifacts/locate_student_boxes.jsonl", boxes)
        self.spec.store.write_jsonl("artifacts/track_observations.jsonl", observations)
        students: dict[str, dict[str, Any]] = {}
        uncertain_student_ids = set(stage.get("uncertain_student_ids", ()))
        for observation in observations:
            row = students.setdefault(observation.track_id, {
                "student_id": observation.track_id,
                "first_pts_seconds": observation.pts_seconds,
                "last_pts_seconds": observation.pts_seconds,
                "observations": 0,
                "reliable_observations": 0,
                "ambiguous_observations": 0,
            })
            row["first_pts_seconds"] = min(float(row["first_pts_seconds"]), observation.pts_seconds)
            row["last_pts_seconds"] = max(float(row["last_pts_seconds"]), observation.pts_seconds)
            row["observations"] = int(row["observations"]) + 1
            ambiguous = bool(
                observation.metadata.get("id_uncertain")
                or observation.detection.attributes.get("association_ambiguous")
            )
            key = "ambiguous_observations" if ambiguous else "reliable_observations"
            row[key] = int(row[key]) + 1
        for row in students.values():
            row["identity_status"] = (
                "uncertain"
                if str(row["student_id"]) in uncertain_student_ids
                else "reliable"
            )
        ordered_students = [students[key] for key in sorted(students, key=student_id_sort_key)]
        self.spec.store.write_jsonl("artifacts/student_catalog.s3.jsonl", ordered_students)
        audit = {
            **dict(stage),
            "visual": visual,
            "identity_source": "LocateAnything+custom_CPU_association" if not generic else f"{route}+custom_CPU_association",
            "localization_route": route,
            "real_bytetrack_used": False,
            "students": len(ordered_students),
            "production_student_observations": len(observations),
            "diagnostic_track_observations": diagnostic_observations,
        }
        self.spec.store.write_json("artifacts/association_audit.v1.json", audit)
        self._commit_preview("S3")
        return StageOutcome(
            artifact_uris=(
                "artifacts/locate_results.jsonl",
                "artifacts/localization_results.v1.jsonl",
                "artifacts/localization_boxes.v1.jsonl",
                "artifacts/localization_track_observations.v1.jsonl",
                "artifacts/locate_student_boxes.jsonl",
                "artifacts/track_observations.jsonl",
                "artifacts/locate_annotated",
                "artifacts/association_audit.v1.json",
                "artifacts/student_catalog.s3.jsonl",
                "previews/S3",
            ),
            processed_units=len(observations),
            total_units=len(observations),
            stats=audit,
        )

    def _run_s4(self, **_: Any) -> StageOutcome:
        observations = [
            TrackObservation.from_dict(row)
            for row in iter_jsonl(self.spec.run_dir / "artifacts" / "track_observations.jsonl")
        ]
        candidates = candidates_from_locate_observations(observations)
        effective_students = len({
            candidate.track_id
            for candidate in candidates
            if not bool(candidate.metadata.get("id_uncertain"))
        })
        minutes = max(0.0, float(self.spec.video.get("duration_seconds", 0.0)) / 60.0)
        configured_budget = qwen_selection_budget(self.spec.config, minutes * 60.0, effective_students, len(candidates))
        selected, selection_audit = fair_candidate_selection(
            candidates,
            limit=min(len(candidates), configured_budget),
            bucket_seconds=float(self.spec.config.get("open_vocabulary.candidate_bucket_seconds", 300.0)),
        )
        selected.sort(key=lambda item: (
            item.trigger_frame_index,
            student_id_sort_key(item.track_id),
            item.event_id,
        ))
        self.spec.store.write_jsonl("artifacts/candidates.all.jsonl", candidates)
        evidence_cfg = self.spec.config.section("evidence")
        filtered = filter_observations_for_candidates(
            (row.to_dict() for row in observations),
            selected,
            before_seconds=float(evidence_cfg["before_seconds"]),
            after_seconds=float(evidence_cfg["after_seconds"]),
            include_all_tracks_in_windows=True,
        )
        locate_rows = list(iter_jsonl(self.spec.run_dir / "artifacts" / "locate_results.jsonl"))
        evidence_sample_units = 0

        def evidence_progress(processed: int, total: int) -> None:
            nonlocal evidence_sample_units
            evidence_sample_units = max(0, int(total))
            combined_total = max(1, evidence_sample_units + len(selected))
            self._report_progress(
                "S4",
                min(int(processed), evidence_sample_units),
                combined_total,
                f"S4 时序证据已提取 {min(int(processed), evidence_sample_units)}/{evidence_sample_units}",
            )

        evidence = build_event_evidence(
            video_path=self.spec.source,
            candidates=selected,
            observations=filtered,
            locate_rows=locate_rows,
            output_dir=self.spec.store.evidence_dir,
            before_seconds=float(evidence_cfg["before_seconds"]),
            after_seconds=float(evidence_cfg["after_seconds"]),
            frames_per_event=int(evidence_cfg["frames_per_event"]),
            context_ratio=float(evidence_cfg["crop_context_ratio"]),
            maximum_crop_side=int(evidence_cfg["maximum_crop_side"]),
            maximum_context_side=int(evidence_cfg.get("maximum_context_side", 768)),
            jpeg_quality=int(evidence_cfg["jpeg_quality"]),
            decode_strategy=str(self.spec.config.get("video.evidence_decode_strategy", "sequential")),
            preprocess_workers=int(self.spec.config.get("performance.preprocess_workers", 2)),
            preprocess_queue_size=int(self.spec.config.get("performance.preprocess_queue_size", 2)),
            progress_callback=evidence_progress,
        )
        by_event = {str(row.get("event_id")): row for row in evidence}
        queue: list[dict[str, Any]] = []
        for ordinal, candidate in enumerate(selected, 1):
            packet = by_event.get(candidate.event_id, {})
            composite = build_temporal_composite(
                packet,
                output_path=self.spec.store.evidence_dir / candidate.event_id / "temporal_composite.jpg",
                jpeg_quality=int(evidence_cfg["jpeg_quality"]),
            )
            if composite is not None:
                packet["qwen_composite"] = composite
            queue.append({
                "request_ordinal": ordinal,
                "candidate_event_id": candidate.event_id,
                "student_id": candidate.track_id,
                "frame_index": candidate.trigger_frame_index,
                "pts_seconds": candidate.trigger_pts_seconds,
                "status": "ready" if composite is not None else "insufficient_evidence",
                "composite_path": None if composite is None else composite["path"],
                "evidence_frame_indices": [] if composite is None else composite["frame_indices"],
                "target_highlighted": bool(packet.get("target_highlighted", False)),
                "request_is_stateless": True,
                "messages_in_request": 1,
                "history_messages": 0,
            })
            combined_total = max(1, evidence_sample_units + len(selected))
            self._report_progress(
                "S4",
                evidence_sample_units + ordinal,
                combined_total,
                f"S4 Qwen 时序拼图已落盘 {ordinal}/{len(selected)}",
            )
        self.spec.store.write_jsonl("evidence/index.jsonl", evidence)
        self.spec.store.write_jsonl("artifacts/qwen_crop_queue.jsonl", queue)
        ready_ids = {
            str(row.get("candidate_event_id"))
            for row in queue
            if row.get("status") == "ready"
        }
        review_selected = [candidate for candidate in selected if candidate.event_id in ready_ids]
        # A selected candidate without a valid composite is an auditable
        # budget miss, never a Qwen request. Move it to the unselected side of
        # the audit contract before S5 begins.
        finalized_audit: list[dict[str, Any]] = []
        for row in selection_audit:
            event_id = str(row.get("event_id", ""))
            if row.get("selection_status") == "selected" and event_id not in ready_ids:
                finalized_audit.append({
                    **row,
                    "selection_status": "unselected",
                    "selection_reason": "insufficient_evidence_no_qwen_request",
                    "canonical_reason": "insufficient_evidence",
                })
            else:
                finalized_audit.append(row)
        self.spec.store.write_jsonl(
            "artifacts/candidates.selected.jsonl",
            [row for row in finalized_audit if row.get("selection_status") == "selected"],
        )
        self.spec.store.write_jsonl(
            "artifacts/candidates.unselected.jsonl",
            [row for row in finalized_audit if row.get("selection_status") == "unselected"],
        )
        self.spec.store.write_jsonl("artifacts/candidates.selection_audit.v1.jsonl", finalized_audit)
        budget = {
            "schema_version": "classroom_hybrid_ai.qwen_budget.v1",
            "formula": "configured_fair_budget" if budget_policy(self.spec.config)["selection_mode"] == "fair" else "all_valid_candidates",
            "policy": budget_policy(self.spec.config),
            "effective_students": effective_students,
            "video_minutes": minutes,
            "request_cap": configured_budget,
            "all_candidates": len(candidates),
            "fair_selected_candidates": len(selected),
            "selected_candidates": len(review_selected),
            "unselected_candidates": len(candidates) - len(review_selected),
            "insufficient_evidence_candidates": len(selected) - len(review_selected),
            "selection_audit": "artifacts/candidates.selection_audit.v1.jsonl",
            "canonical_reason_counts": {
                reason: sum(str(row.get("canonical_reason", "")) == reason for row in finalized_audit)
                for reason in (
                    "budget_exhausted",
                    "insufficient_evidence",
                    "id_uncertain",
                )
            },
            "selected_reason_counts": {
                reason: sum(
                    reason in str(row.get("selection_reason", ""))
                    for row in finalized_audit
                    if row.get("selection_status") == "selected"
                )
                for reason in (
                    "selected_anomaly_coverage",
                    "selected_action_change_coverage",
                    "selected_new_id_coverage",
                    "selected_edge_coverage",
                    "selected_latter_half_coverage",
                    "selected_student_fairness_quota",
                )
            },
        }
        self.spec.store.write_json("artifacts/qwen_budget.v1.json", budget)
        self._commit_preview("S4")
        s4_total_units = max(1, evidence_sample_units + len(selected))
        return StageOutcome(
            artifact_uris=(
                "artifacts/candidates.all.jsonl",
                "artifacts/candidates.selected.jsonl",
                "artifacts/candidates.unselected.jsonl",
                "artifacts/candidates.selection_audit.v1.jsonl",
                "artifacts/qwen_crop_queue.jsonl",
                "artifacts/qwen_budget.v1.json",
                "evidence",
                "previews/S4",
            ),
            processed_units=s4_total_units,
            total_units=s4_total_units,
            stats={
                **budget,
                "ready_evidence": sum(row["status"] == "ready" for row in queue),
                "evidence_sample_units": evidence_sample_units,
                "composite_units": len(selected),
                "progress_unit": "evidence_samples_plus_temporal_composites",
            },
        )

    def _load_reusable_review(
        self,
        path: Path,
        candidate: CandidateEvent,
        packet: Mapping[str, Any],
        vocabulary_sha256: str,
    ) -> ReviewDecision | None:
        if not path.is_file():
            return None
        try:
            raw = _read_json(path)
            if raw.get("candidate_event_id") != candidate.event_id:
                return None
            if raw.get("reviewed_event_type") != candidate.event_type:
                return None
            frames = packet.get("qwen_composite", {}).get("frame_indices", ()) if isinstance(packet.get("qwen_composite"), Mapping) else ()
            normalized = _validate_classification_response(
                raw,
                [int(value) for value in frames],
                allowed_action_ids=set(self.spec.profile.action_ids),
                action_profile=self.spec.profile,
                target_highlighted=_target_highlighted(packet),
            )
            if normalized.get("vocabulary_sha256") != vocabulary_sha256:
                return None
            review = ReviewDecision.from_dict(normalized)
            return review if review.schema_valid else None
        except Exception:
            return None

    def _run_s5(
        self,
        *,
        input_fingerprint_sha256: str,
        partial_outputs_reusable: bool,
        **_: Any,
    ) -> StageOutcome:
        selected_rows = list(
            iter_jsonl(self.spec.run_dir / "artifacts" / "candidates.selected.jsonl")
        )
        queue_rows = list(
            iter_jsonl(self.spec.run_dir / "artifacts" / "qwen_crop_queue.jsonl")
        )
        ready_ids = {
            str(row.get("candidate_event_id"))
            for row in queue_rows
            if row.get("status") == "ready"
        }
        # S4's queue is the execution boundary. A stale or hand-edited
        # selected file cannot turn an insufficient-evidence row into a Qwen
        # request; only queue rows explicitly marked ``ready`` are consumed.
        candidates = [
            CandidateEvent.from_dict(row)
            for row in selected_rows
            if str(row.get("event_id", "")) in ready_ids
        ]
        filtered_non_ready = len(selected_rows) - len(candidates)
        evidence = list(iter_jsonl(self.spec.run_dir / "evidence" / "index.jsonl"))
        locate_rows = list(iter_jsonl(self.spec.run_dir / "artifacts" / "locate_results.jsonl"))
        registry, vocabulary, vocabulary_sha256 = _profile_vocabulary(self.spec)
        evidence_by_event = {str(row.get("event_id")): row for row in evidence}
        response_prefix = f"responses/classification/{input_fingerprint_sha256[:16]}"
        response_dir = self.spec.run_dir / response_prefix
        response_dir.mkdir(parents=True, exist_ok=True)
        reused: dict[str, ReviewDecision] = {}
        if partial_outputs_reusable:
            for candidate in candidates:
                review = self._load_reusable_review(
                    response_dir / f"{candidate.event_id}.json",
                    candidate,
                    evidence_by_event.get(candidate.event_id, {}),
                    vocabulary_sha256,
                )
                if review is not None:
                    reused[candidate.event_id] = review
        pending = [candidate for candidate in candidates if candidate.event_id not in reused]
        self._report_progress(
            "S5",
            len(reused),
            len(candidates),
            f"S5 已恢复 {len(reused)}/{len(candidates)} 个成功响应",
        )
        adapter: Any | None = None
        gpu_lease: GpuLease | None = None
        release_receipt: dict[str, Any] = {"released": False, "reason": "not_started"}
        pending_reviews: list[ReviewDecision] = []
        qwen_stats: dict[str, Any] = {
            "status": "ok" if not pending else "pending",
            "reviews": len(reused),
            "actual_calls": 0,
            "resumed_all_successful": bool(candidates and not pending),
        }
        try:
            if pending:
                if not self.spec.skip_qwen and bool(self.spec.config.get("qwen.enabled", True)):
                    lease_path = self.spec.config.resolve(".runtime/classroom_hybrid_ai/gpu.lock")
                    assert lease_path is not None
                    gpu_lease = GpuLease(lease_path)
                    gpu_lease.acquire()
                    adapter = _make_qwen_adapter(self.spec.config, adapter_factory=self.qwen_factory)
                remaining_budget = max(0, len(candidates) - len(reused))
                call_budget = _QwenCallBudget(remaining_budget)
                pending_reviews, qwen_stats = _qwen_pass(
                    self.spec.config,
                    pending,
                    evidence,
                    self.spec.store,
                    skip=self.spec.skip_qwen,
                    prompt=_classification_prompt(
                        vocabulary,
                        vocabulary_sha256,
                        action_profile=self.spec.profile,
                        prompt_version=str(self.spec.config.get("qwen.prompt_version", "legacy_v1")),
                    ),
                    response_prefix=response_prefix,
                    expected_vocabulary_sha256=vocabulary_sha256,
                    allowed_action_ids=set(self.spec.profile.action_ids),
                    action_profile=self.spec.profile,
                    locate_rows=locate_rows,
                    response_contract="classification",
                    adapter=adapter,
                    call_budget=call_budget,
                    release_adapter=False,
                    resolve_conflicts=False,
                    review_output="diagnostics/qwen_pending_reviews.jsonl",
                    adapter_factory=self.qwen_factory,
                    progress_callback=lambda processed, _pending_total: self._report_progress(
                        "S5",
                        len(reused) + processed,
                        len(candidates),
                        f"S5 Qwen 响应已落盘 {len(reused) + processed}/{len(candidates)}",
                    ),
                )
                qwen_stats = {**qwen_stats, "actual_calls": call_budget.used}
        finally:
            worker_pid = getattr(adapter, "pid", None) if adapter is not None else None
            if adapter is not None:
                release_receipt = release_model(adapter)
                release_receipt.update({
                    "worker_pid": worker_pid,
                    "worker_exitcode": getattr(adapter, "exitcode", None),
                    "worker_isolated": adapter.__class__.__name__ == "QwenProcessClient",
                })
            if gpu_lease is not None:
                gpu_lease.release()
            atomic_write_json(
                self.spec.run_dir / "diagnostics" / "qwen_worker_receipt.v1.json",
                release_receipt,
            )
        by_event = {**reused, **{review.candidate_event_id: review for review in pending_reviews}}
        ordered_reviews = [by_event[candidate.event_id] for candidate in candidates if candidate.event_id in by_event]
        if len(ordered_reviews) != len(candidates):
            raise StageExecutionError("qwen_review_count_mismatch")
        pending_ids = {review.candidate_event_id for review in pending_reviews}
        try:
            execution_epoch = int(_read_json(self.spec.run_dir / "run_state.v1.json").get("epoch", 0))
        except (OSError, TypeError, ValueError):
            execution_epoch = 0
        execution_audit = [
            {
                "candidate_event_id": candidate.event_id,
                "student_id": candidate.track_id,
                "status": "stable_cached" if candidate.event_id in reused else "executed",
                "canonical_reason": "stable_cached" if candidate.event_id in reused else "qwen_request_executed",
                "response_path": f"{response_prefix}/{candidate.event_id}.json",
                "epoch": execution_epoch,
            }
            for candidate in candidates
        ]
        # ``pending_ids`` is retained in the audit calculation to make a
        # resumed-vs-new request distinction explicit without relying on row
        # order. It also guards future adapters that may return a response
        # under a different candidate ordering.
        for row in execution_audit:
            row["new_request"] = row["candidate_event_id"] in pending_ids
        self.spec.store.write_jsonl("artifacts/qwen_execution_audit.v1.jsonl", execution_audit)
        self.spec.store.write_jsonl("reviews.classification.jsonl", ordered_reviews)
        vocabulary_payload = {
            "schema_version": "classroom_hybrid_ai.session_vocabulary.v1",
            "profile_id": self.spec.profile.profile_id,
            "profile_sha256": self.spec.profile.sha256,
            "registry_sha256": registry.sha256,
            "actions": [item.to_dict() for item in vocabulary],
            "sha256": vocabulary_sha256,
        }
        self.spec.store.write_json("session_vocabulary.v1.json", vocabulary_payload)
        active = {
            "schema_version": "classroom_hybrid_ai.qwen_response_set.v1",
            "input_fingerprint_sha256": input_fingerprint_sha256,
            "response_prefix": response_prefix,
            "selected_candidates": len(candidates),
            "ready_queue_candidates": len(ready_ids),
            "filtered_non_ready_candidates": filtered_non_ready,
            "reused_successful_responses": len(reused),
            "new_reviews": len(pending_reviews),
            "profile_id": self.spec.profile.profile_id,
            "profile_sha256": self.spec.profile.sha256,
            "vocabulary_sha256": vocabulary_sha256,
            "qwen": qwen_stats,
            "release": release_receipt,
            "model_started": adapter is not None,
            "stable_cached_candidates": len(reused),
            "new_request_candidates": len(pending_reviews),
            "execution_audit": "artifacts/qwen_execution_audit.v1.jsonl",
        }
        self.spec.store.write_json("artifacts/s5_active_response_set.v1.json", active)
        self._commit_preview("S5")
        warnings: tuple[str, ...] = ()
        status = "completed"
        if self.spec.skip_qwen or qwen_stats.get("status") in {"warning", "skipped"}:
            status = "warning"
            warnings = ("qwen_skipped_or_incomplete",)
        exit_code = release_receipt.get("worker_exitcode")
        cuda_release = release_receipt.get("cuda_after_release", {})
        cleanup_ok = not release_receipt.get("close_error") and not (
            isinstance(cuda_release, Mapping) and cuda_release.get("teardown_error")
        )
        unload_ack = None if adapter is None else (
            cleanup_ok and (exit_code == 0 if release_receipt.get("worker_isolated") else True)
        )
        return StageOutcome(
            status=status,
            artifact_uris=(
                "reviews.classification.jsonl",
                "session_vocabulary.v1.json",
                "artifacts/s5_active_response_set.v1.json",
                "diagnostics/qwen_worker_receipt.v1.json",
                "artifacts/qwen_execution_audit.v1.jsonl",
                response_prefix,
                "previews/S5",
            ),
            processed_units=len(candidates),
            total_units=len(candidates),
            stats=active,
            warnings=warnings,
            model_loaded=False,
            model_pid=release_receipt.get("worker_pid"),
            unload_ack=unload_ack,
            exit_code=exit_code,
        )

    def _run_s6(self, **_: Any) -> StageOutcome:
        candidates = [
            CandidateEvent.from_dict(row)
            for row in iter_jsonl(self.spec.run_dir / "artifacts" / "candidates.all.jsonl")
        ]
        selected_reviews = [
            ReviewDecision.from_dict(row)
            for row in iter_jsonl(self.spec.run_dir / "reviews.classification.jsonl")
        ]
        review_by_id = {review.candidate_event_id: review for review in selected_reviews}
        for candidate in candidates:
            if candidate.event_id in review_by_id:
                continue
            review_by_id[candidate.event_id] = ReviewDecision(
                candidate_event_id=candidate.event_id,
                decision="uncertain",
                reviewed_event_type=candidate.event_type,
                behavior="UNKNOWN",
                rationale="qwen_candidate_unreviewed",
                evidence_frame_indices=(),
                reviewer="pipeline_budget",
                schema_valid=False,
                metadata={"unreviewed": True, "reason": "qwen_candidate_unreviewed"},
            )
        reviews = [review_by_id[candidate.event_id] for candidate in candidates]
        registry, vocabulary, _ = _profile_vocabulary(self.spec)
        stage = _write_open_vocab_outputs(
            self.spec.config,
            self.spec.store,
            discovery_reviews=(),
            classification_reviews=reviews,
            candidates=candidates,
            registry=registry,
            vocabulary=vocabulary,
            duration_seconds=float(self.spec.video.get("duration_seconds", 0.0)),
            video=self.spec.video,
            write_session_vocabulary=False,
        )
        policy = StrictFusionPolicy(min_evidence_frames=1)
        final_events = [policy.fuse(candidate, review_by_id.get(candidate.event_id)) for candidate in candidates]
        self.spec.store.write_jsonl("candidates.jsonl", candidates)
        self.spec.store.write_jsonl("events.jsonl", final_events)
        students = list(iter_jsonl(self.spec.run_dir / "students.jsonl"))
        action_observations = build_action_observations(self.spec.run_dir, self.spec.profile)
        self.spec.store.write_jsonl("action_observations.jsonl", action_observations)
        duration = float(self.spec.video.get("duration_seconds", 0.0))
        intervals = normalize_intervals(self.spec.run_dir, self.spec.profile, duration, students)
        self.spec.store.write_jsonl("action_intervals.jsonl", intervals)
        self.spec.store.write_json("taxonomy_profile.v1.json", taxonomy_payload(self.spec.profile))
        timeline = {
            "schema_version": "classroom_hybrid_ai.timeline.v2",
            "video": dict(self.spec.video),
            "student_rows": [row["student_id"] for row in students],
            "start_pts_seconds": 0.0,
            "end_pts_seconds": duration,
            "duration_seconds": duration,
            "taxonomy_profile_id": self.spec.profile.profile_id,
            "taxonomy_contract_sha256": self.spec.profile.sha256,
            "intervals": intervals,
            "coverage": {"complete": True, "gray_gaps_included": True},
        }
        self.spec.store.write_json("timeline.v2.json", timeline)
        self._commit_preview("S6")
        return StageOutcome(
            artifact_uris=(
                "candidates.jsonl",
                "events.jsonl",
                "students.jsonl",
                "action_observations.jsonl",
                "action_intervals.jsonl",
                "session_vocabulary.v1.json",
                "taxonomy_profile.v1.json",
                "timeline.v1.json",
                "timeline.v2.json",
                "previews/S6",
            ),
            processed_units=len(candidates),
            total_units=len(candidates),
            stats={**stage, "final_events": len(final_events)},
        )

    def _run_s7(self, **_: Any) -> StageOutcome:
        expected_frames = max(1, int(self.spec.video.get("frame_count", 0) or 1))
        self._report_progress(
            "S7",
            0,
            expected_frames,
            f"S7 单次解码渲染 0/{expected_frames} 帧",
        )
        outputs = build_v3_outputs(
            self.spec.run_dir,
            source=self.spec.source,
            info=self.spec.video,
            profile=self.spec.profile,
            config_digest=self.spec.config_sha256,
            manifest_extra={
                "runtime": runtime_fingerprint(),
                "code_sha256": self.spec.code_sha256,
                "experiment_digest": self.spec.experiment_digest,
                "models": {
                    "locate_anything": {"fingerprint_sha256": self.spec.locate_model_sha256},
                    "qwen3.5_9b": {"fingerprint_sha256": self.spec.qwen_model_sha256},
                },
                "command": f"classroom-hybrid run {self.spec.source}",
                "profile": self.spec.profile.to_dict(),
                "profile_sha256": self.spec.profile.sha256,
            },
            reuse_stage_artifacts=True,
            encoder_backend=str(self.spec.config.get("video.encoder_backend", "opencv")),
            ffmpeg_path=self.spec.config.resolve(self.spec.config.get("video.ffmpeg_path")),
            progress_callback=lambda processed, total: self._report_progress(
                "S7",
                processed,
                total,
                f"S7 单次解码渲染 {processed}/{total} 帧",
            ),
        )
        artifacts = (
            "manifest.json",
            "timeline.v2.json",
            "frontend/index.v1.json",
            "frontend/artifact_index.v1.json",
            "metrics.v1.json",
            "report.html",
            "previews",
            "videos/tracking_preview.mp4",
            "videos/action_preview.mp4",
            "videos/annotated_final.mp4",
        )
        warnings = tuple(str(item) for item in outputs.get("warnings", ()))
        video_results = outputs.get("videos", {})
        annotated = video_results.get("annotated_final", {}) if isinstance(video_results, Mapping) else {}
        rendered_frames = max(1, int(annotated.get("frames", expected_frames)))
        return StageOutcome(
            status="warning" if warnings else "completed",
            artifact_uris=artifacts,
            stats={
                **outputs,
                "progress_unit": "source_frames_rendered_once",
                "research_validity_impact": (
                    "none" if warnings and all("audio_not_preserved:" in item for item in warnings)
                    else "blocking" if warnings else "none"
                ),
            },
            warnings=warnings,
            processed_units=rendered_frames,
            total_units=rendered_frames,
        )


__all__ = [
    "ResearchStageExecutor",
    "ResearchStageSpec",
    "StageExecutionError",
    "StageOutcome",
    "model_manifest_fingerprints",
]
