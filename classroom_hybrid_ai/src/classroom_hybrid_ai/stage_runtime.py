"""Durable v3 stage state, adaptive screenshot scheduling, and recovery helpers.

This module deliberately has no model imports.  It is safe for the read-only
``status``/``watch`` commands and for a crash-recovery process to import.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ArtifactRef, FrontendIndex, RunState, StageEvent, StageState
from .hashing import canonical_sha256, sha256_file, uncached_hashes
from contextlib import nullcontext
from .storage import atomic_write_json, atomic_write_jsonl, iter_jsonl


STAGE_DEFINITIONS: tuple[tuple[str, str, float], ...] = (
    ("S0", "输入与预检", 5.0),
    ("S1", "截图调度", 10.0),
    ("S2", "Locate 检测", 20.0),
    ("S3", "学生 ID 跟踪", 15.0),
    ("S4", "裁框与候选", 10.0),
    ("S5", "Qwen 分类", 25.0),
    ("S6", "时序合并", 7.0),
    ("S7", "渲染与审计", 8.0),
)
STAGE_IDS = tuple(item[0] for item in STAGE_DEFINITIONS)
STAGE_WEIGHTS = {item[0]: item[2] for item in STAGE_DEFINITIONS}
STAGE_NAMES = {item[0]: item[1] for item in STAGE_DEFINITIONS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_total(value: int | None) -> int | None:
    if value is None:
        return None
    return max(0, int(value))


def weighted_progress(stages: Mapping[str, StageState | Mapping[str, Any]]) -> float | None:
    """Return the fixed-weight progress in percent, or ``None`` if unknown."""

    total = 0.0
    for stage_id, _name, weight in STAGE_DEFINITIONS:
        raw = stages.get(stage_id)
        if raw is None:
            continue
        stage = raw if isinstance(raw, StageState) else StageState.from_dict(raw)
        if stage.total_units is None and stage.status in {"running", "completed"}:
            return None
        stage_progress = stage.progress
        total += weight * stage_progress
    return min(100.0, max(0.0, total))


def stage_progress(processed_units: int, total_units: int | None) -> float | None:
    if total_units is None or total_units <= 0:
        return None
    return min(1.0, max(0.0, float(processed_units) / float(total_units)))


class StageLedger:
    """Atomic state/event writer with monotonic progress and stage receipts."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        video_id: str,
        total_units: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "stage_receipts").mkdir(exist_ok=True)
        (self.run_dir / "previews").mkdir(exist_ok=True)
        self.state_path = self.run_dir / "run_state.v1.json"
        self.events_path = self.run_dir / "stage_events.v1.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_handle = self.events_path.open("a", encoding="utf-8", newline="\n")
        self.run_id = str(run_id)
        self.video_id = str(video_id)
        self.started_at = utc_now()
        self._last_progress = 0.0
        self._epoch = 0
        self._event_seq = 0
        self._last_event_progress = 0.0
        self._last_event_monotonic = 0.0
        self._last_state_progress = 0.0
        self._last_state_monotonic = 0.0
        self._last_state_status = "running"
        self._stages: dict[str, StageState] = {
            stage_id: StageState(
                stage_id=stage_id,
                name=name,
                weight=weight,
                total_units=total_units,
                epoch=self._epoch,
            )
            for stage_id, name, weight in STAGE_DEFINITIONS
        }
        self._state = RunState(
            run_id=self.run_id,
            video_id=self.video_id,
            status="running",
            current_stage="S0",
            overall_progress=0.0,
            current_stage_progress=0.0,
            total_units=_safe_total(total_units),
            started_at=self.started_at,
            updated_at=self.started_at,
            stages=self._stages,
            research_validity="pending",
            epoch=self._epoch,
            metadata=dict(metadata or {}),
        )
        self._write_state(force=True)

    @classmethod
    def open_existing(cls, run_dir: str | Path) -> "StageLedger":
        root = Path(run_dir).resolve()
        payload = json.loads((root / "run_state.v1.json").read_text(encoding="utf-8"))
        state = RunState.from_dict(payload)
        self = cls.__new__(cls)
        self.run_dir = root
        self.state_path = root / "run_state.v1.json"
        self.events_path = root / "stage_events.v1.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_handle = self.events_path.open("a", encoding="utf-8", newline="\n")
        self.run_id = state.run_id
        self.video_id = state.video_id
        self.started_at = state.started_at
        self._state = state
        self._stages = dict(state.stages)
        self._epoch = int(getattr(state, "epoch", 0) or state.metadata.get("epoch", 0) or 0)
        self._last_progress = float(state.overall_progress or 0.0)
        self._event_seq = 0
        self._last_event_progress = 0.0
        self._last_event_monotonic = time.monotonic()
        self._last_state_progress = float(state.overall_progress or 0.0)
        self._last_state_monotonic = time.monotonic()
        self._last_state_status = state.status
        self._restore_event_cursor()
        return self

    def _restore_event_cursor(self) -> None:
        """Recover the append cursor from the durable log after a restart."""

        if not self.events_path.is_file():
            return
        last_epoch = self._epoch
        last_progress = self._last_event_progress
        partial_offset: int | None = None
        # A one-time linear scan at process start is intentional. Every later
        # append is O(1); no update ever rereads the complete history.
        try:
            with self.events_path.open("rb") as handle:
                for fallback_seq, raw_line in enumerate(handle, 1):
                    line_start = handle.tell() - len(raw_line)
                    if not raw_line.endswith(b"\n"):
                        partial_offset = line_start
                        break
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        # A crash can leave a partial final line. It is not a
                        # committed event and is ignored for cursor recovery.
                        continue
                    if not isinstance(value, Mapping):
                        continue
                    try:
                        sequence = int(value.get("seq", fallback_seq))
                    except (TypeError, ValueError):
                        sequence = fallback_seq
                    self._event_seq = max(self._event_seq, sequence)
                    try:
                        last_epoch = max(last_epoch, int(value.get("epoch", 0)))
                    except (TypeError, ValueError):
                        pass
                    try:
                        last_progress = float(value.get("progress", last_progress))
                    except (TypeError, ValueError):
                        pass
        except OSError:
            return
        self._epoch = last_epoch
        self._last_event_progress = min(1.0, max(0.0, last_progress))
        if partial_offset is not None:
            # A crash can leave a final unterminated JSON record. It was never
            # a committed event; discard that tail before the next append so
            # the first post-restart event remains valid JSONL.
            try:
                self._event_handle.flush()
                self._event_handle.close()
                with self.events_path.open("r+b") as handle:
                    handle.truncate(partial_offset)
                self._event_handle = self.events_path.open("a", encoding="utf-8", newline="\n")
            except OSError:
                # If truncation is unavailable, keep the append cursor safe;
                # the malformed tail remains auditable and seq still advances.
                self._event_handle = self.events_path.open("a", encoding="utf-8", newline="\n")

    def prepare_resume(
        self,
        stage_id: str,
        *,
        invalidated_by: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Reset the selected stage and its downstream states for a new attempt."""

        if stage_id not in STAGE_IDS:
            raise KeyError(f"unknown_stage:{stage_id}")
        start = STAGE_IDS.index(stage_id)
        now = utc_now()
        self._epoch += 1
        for current_id in STAGE_IDS[start:]:
            previous = self._stages[current_id]
            self._stages[current_id] = replace(
                previous,
                status="pending",
                progress=0.0,
                processed_units=0,
                started_at=None,
                updated_at=now,
                ended_at=None,
                preview_uris=(),
                artifact_uris=(),
                warnings=(),
                errors=(),
                model_loaded=False,
                model_pid=None,
                unload_ack=None,
                exit_code=None,
                epoch=self._epoch,
                metadata={"resume_reset": True, "invalidated_by": list(invalidated_by)},
            )
        baseline = weighted_progress(self._stages)
        self._last_progress = float(baseline or 0.0)
        self._rebuild_state(
            current_stage=stage_id,
            status="running",
            research_validity="pending",
            processed_units=0,
            warnings=(),
            errors=(),
            model_loaded=False,
            model_pid=None,
            unload_ack=None,
            exit_code=None,
            metadata={
                "resume_from": stage_id,
                "resume_invalidated_by": list(invalidated_by),
                "epoch": self._epoch,
                **dict(metadata or {}),
            },
            force_write=True,
        )

    def begin_attempt(self, stage_id: str, input_fingerprint_sha256: str) -> dict[str, Any]:
        """Persist the fingerprint before work so matching partial outputs are reusable."""

        path = self.run_dir / "diagnostics" / "stage_attempts" / f"{stage_id}.json"
        previous: dict[str, Any] | None = None
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                previous = value if isinstance(value, dict) else None
            except Exception:
                previous = None
        attempt = {
            "schema_version": "classroom_hybrid_ai.stage_attempt.v1",
            "stage_id": stage_id,
            "started_at": utc_now(),
            "input_fingerprint_sha256": input_fingerprint_sha256,
            "partial_outputs_reusable": bool(
                previous
                and previous.get("input_fingerprint_sha256") == input_fingerprint_sha256
            ),
        }
        atomic_write_json(path, attempt)
        return attempt

    @property
    def state(self) -> RunState:
        return self._state

    def _write_state(self, *, force: bool = False) -> None:
        progress = float(self._state.overall_progress or 0.0)
        status_changed = self._state.status != self._last_state_status
        now = time.monotonic()
        if not force and not status_changed and progress - self._last_state_progress < 1.0 and now - self._last_state_monotonic < 1.0:
            return
        atomic_write_json(self.state_path, self._state.to_dict())
        self._last_state_progress = progress
        self._last_state_monotonic = now
        self._last_state_status = self._state.status

    def _estimate_eta(self, *, overall_progress: float | None, status: str) -> float | None:
        """Estimate remaining wall time from persisted run progress.

        ``overall_progress`` is expressed as percentage points in ``RunState``.
        Unknown totals and a zero-progress run intentionally keep ETA unset;
        a successful terminal state reports zero so the read-only watcher does
        not show stale work remaining after completion.
        """

        if status in {"completed", "warning", "skipped"}:
            return 0.0
        if status != "running" or overall_progress is None or overall_progress <= 0.0:
            return None
        try:
            started = datetime.fromisoformat(str(self._state.started_at).replace("Z", "+00:00"))
            elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
        remaining = max(0.0, 100.0 - float(overall_progress))
        return max(0.0, elapsed * remaining / float(overall_progress))

    def close(self) -> None:
        handle = getattr(self, "_event_handle", None)
        if handle is not None and not handle.closed:
            handle.flush()
            handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _append_event(self, event: StageEvent) -> None:
        """Append one event in O(1) time; this object is the sole writer."""

        if self._event_handle.closed:
            self._event_handle = self.events_path.open("a", encoding="utf-8", newline="\n")
        self._event_handle.write(
            json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._event_handle.flush()
        self._event_seq = event.seq
        self._last_event_progress = event.progress
        self._last_event_monotonic = time.monotonic()

    def _event_allowed(self, *, status: str, progress: float) -> bool:
        """Apply the one-second/one-percent progress event throttle."""

        if status in {"started", "completed", "warning", "failed", "skipped", "blocked"}:
            return True
        now = time.monotonic()
        if progress - self._last_event_progress >= 0.01:
            return True
        return now - self._last_event_monotonic >= 1.0

    def _rebuild_state(self, *, current_stage: str | None = None, status: str | None = None, research_validity: str | None = None, processed_units: int | None = None, total_units: int | None | object = ... , warnings: Iterable[str] | None = None, errors: Iterable[str] | None = None, previews: Iterable[str] | None = None, artifacts: Iterable[ArtifactRef | Mapping[str, Any]] | None = None, model_loaded: bool | None = None, model_pid: int | None | object = ..., unload_ack: bool | None | object = ..., exit_code: int | None | object = ..., metadata: Mapping[str, Any] | None = None, force_write: bool = False) -> None:
        effective_status = status or self._state.status
        overall = weighted_progress(self._stages)
        if overall is not None:
            overall = max(self._last_progress, overall)
            if effective_status in {"failed", "blocked"} and overall >= 100.0:
                overall = 99.999
            self._last_progress = overall
        stage_id = current_stage or self._state.current_stage
        current = self._stages.get(stage_id)
        eta_seconds = self._estimate_eta(overall_progress=overall, status=effective_status)
        aggregate_previews = tuple(dict.fromkeys(
            uri
            for stage in self._stages.values()
            for uri in stage.preview_uris
        ))
        self._state = RunState(
            run_id=self.run_id,
            video_id=self.video_id,
            status=effective_status,
            current_stage=stage_id,
            overall_progress=overall,
            current_stage_progress=None if current is None else round(current.progress * 100.0, 6),
            processed_units=self._state.processed_units if processed_units is None else max(0, int(processed_units)),
            total_units=self._state.total_units if total_units is ... else _safe_total(total_units),
            started_at=self._state.started_at,
            updated_at=utc_now(),
            eta_seconds=eta_seconds,
            stages=self._stages,
            previews=aggregate_previews if previews is None else tuple(previews),
            artifacts=self._state.artifacts if artifacts is None else tuple(artifacts),
            warnings=self._state.warnings if warnings is None else tuple(warnings),
            errors=self._state.errors if errors is None else tuple(errors),
            model_loaded=self._state.model_loaded if model_loaded is None else bool(model_loaded),
            model_pid=self._state.model_pid if model_pid is ... else model_pid,
            unload_ack=self._state.unload_ack if unload_ack is ... else unload_ack,
            exit_code=self._state.exit_code if exit_code is ... else exit_code,
            research_validity=research_validity or self._state.research_validity,
            epoch=self._epoch,
            metadata={**self._state.metadata, **dict(metadata or {})},
        )
        self._write_state(force=force_write or effective_status in {"completed", "warning", "failed", "skipped", "blocked"})

    def update(
        self,
        stage_id: str,
        *,
        processed_units: int = 0,
        total_units: int | None = None,
        status: str = "progress",
        message: str = "",
        preview_uris: Sequence[str] = (),
        artifact_uris: Sequence[str] = (),
        warnings: Sequence[str] = (),
        errors: Sequence[str] = (),
        model_loaded: bool | None = None,
        model_pid: int | None | object = ...,
        unload_ack: bool | None | object = ...,
        exit_code: int | None | object = ...,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageState:
        if stage_id not in self._stages:
            raise KeyError(f"unknown_stage:{stage_id}")
        previous = self._stages[stage_id]
        total = previous.total_units if total_units is None else _safe_total(total_units)
        total_changed = total != previous.total_units
        processed = max(previous.processed_units, int(processed_units))
        fraction = stage_progress(processed, total)
        fraction = previous.progress if fraction is None else max(previous.progress, fraction)
        state_status = "running" if status in {"started", "progress"} else status
        if state_status == "completed":
            fraction = 1.0
            if total is not None:
                processed = max(processed, total)
        stage = replace(
            previous,
            status=state_status,
            progress=fraction,
            processed_units=processed,
            total_units=total,
            started_at=previous.started_at or utc_now(),
            updated_at=utc_now(),
            ended_at=utc_now() if state_status in {"completed", "failed", "warning", "skipped", "blocked"} else previous.ended_at,
            preview_uris=tuple(preview_uris) or previous.preview_uris,
            artifact_uris=tuple(artifact_uris) or previous.artifact_uris,
            warnings=tuple(warnings) or previous.warnings,
            errors=tuple(errors) or previous.errors,
            model_loaded=previous.model_loaded if model_loaded is None else bool(model_loaded),
            model_pid=previous.model_pid if model_pid is ... else model_pid,
            unload_ack=previous.unload_ack if unload_ack is ... else unload_ack,
            exit_code=previous.exit_code if exit_code is ... else exit_code,
            epoch=self._epoch,
            metadata={**previous.metadata, **dict(metadata or {})},
        )
        self._stages[stage_id] = stage
        overall_fraction = weighted_progress(self._stages)
        event_progress = fraction if overall_fraction is None else min(1.0, max(0.0, overall_fraction / 100.0))
        # ``StageEvent.progress`` is the run-level monotonic value.  The
        # stage-local fraction remains available in metadata for renderers.
        event_type = "started" if status == "started" else status
        if total_changed or self._event_allowed(status=event_type, progress=event_progress):
            sequence = self._event_seq + 1
            self._append_event(StageEvent(
                event_id=canonical_sha256({
                    "run_id": self.run_id,
                    "stage_id": stage_id,
                    "epoch": self._epoch,
                    "seq": sequence,
                    "type": event_type,
                    "processed": processed,
                })[:32],
                stage_id=stage_id,
                event_type=event_type,
                timestamp=stage.updated_at or utc_now(),
                seq=sequence,
                epoch=self._epoch,
                progress=event_progress,
                processed_units=processed,
                total_units=total,
                status=state_status,
                message=message,
                preview_uris=tuple(preview_uris),
                artifact_uris=tuple(artifact_uris),
                warnings=tuple(warnings),
                errors=tuple(errors),
                metadata={"stage_progress": fraction, "epoch": self._epoch, **dict(metadata or {})},
            ))
        self._rebuild_state(current_stage=stage_id, processed_units=processed, model_loaded=model_loaded, model_pid=model_pid, unload_ack=unload_ack, exit_code=exit_code, force_write=status == "started" or total_changed)
        return stage

    def begin(self, stage_id: str, *, total_units: int | None = None, message: str = "", metadata: Mapping[str, Any] | None = None) -> StageState:
        return self.update(stage_id, total_units=total_units, status="started", message=message, metadata=metadata)

    def progress(self, stage_id: str, processed_units: int, total_units: int | None = None, *, message: str = "", preview_uris: Sequence[str] = (), artifact_uris: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> StageState:
        return self.update(stage_id, processed_units=processed_units, total_units=total_units, status="progress", message=message, preview_uris=preview_uris, artifact_uris=artifact_uris, metadata=metadata)

    def finish(self, stage_id: str, *, status: str = "completed", message: str = "", preview_uris: Sequence[str] = (), artifact_uris: Sequence[str] = (), warnings: Sequence[str] = (), errors: Sequence[str] = (), model_loaded: bool | None = None, model_pid: int | None | object = ..., unload_ack: bool | None | object = ..., exit_code: int | None | object = ..., metadata: Mapping[str, Any] | None = None) -> StageState:
        previous = self._stages[stage_id]
        successful_terminal = status in {"completed", "warning", "skipped"}
        processed = (
            previous.total_units or previous.processed_units
            if successful_terminal
            else previous.processed_units
        )
        total = previous.total_units or max(1, processed)
        return self.update(stage_id, processed_units=processed, total_units=total, status=status, message=message, preview_uris=preview_uris, artifact_uris=artifact_uris, warnings=warnings, errors=errors, model_loaded=model_loaded, model_pid=model_pid, unload_ack=unload_ack, exit_code=exit_code, metadata=metadata)

    def receipt(self, stage_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = {"schema_version": "classroom_hybrid_ai.stage_receipt.v1", "stage_id": stage_id, "created_at": utc_now(), **dict(payload)}
        value["sha256"] = canonical_sha256({key: item for key, item in value.items() if key != "sha256"})
        path = self.run_dir / "stage_receipts" / f"{stage_id}.json"
        if path.is_file():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
                previous_sha = str(previous.get("sha256", "unknown"))[:12] if isinstance(previous, Mapping) else "unknown"
                stamp = utc_now().replace(":", "").replace("-", "")
                atomic_write_json(
                    self.run_dir / "stage_receipts" / "history" / f"{stage_id}.{stamp}.{previous_sha}.json",
                    previous,
                )
            except Exception:
                # A corrupt prior receipt is already an invalidation reason;
                # failure to archive it must not block writing the repaired one.
                pass
        atomic_write_json(path, value)
        return value

    def complete_run(self, *, status: str = "completed", research_validity: str = "pending", warnings: Sequence[str] = (), errors: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> RunState:
        self._rebuild_state(status=status, research_validity=research_validity, warnings=warnings, errors=errors, metadata=metadata)
        return self._state


class AdaptiveScreenshotScheduler:
    """Deterministic scheduler for the production and validation profiles."""

    def __init__(self, *, fps: float, frame_count: int, duration_seconds: float, mode: str = "production_adaptive", max_screenshots: int | None = None, locate_cap: int = 1500, reserved_change_events: int = 300) -> None:
        if fps <= 0 or frame_count <= 0 or duration_seconds <= 0:
            raise ValueError("video_metadata_invalid")
        if mode not in {"production_adaptive", "exhaustive_validation"}:
            raise ValueError("screenshot_mode_invalid")
        self.fps = float(fps)
        self.frame_count = int(frame_count)
        self.duration_seconds = float(duration_seconds)
        self.mode = mode
        self.max_screenshots = max_screenshots
        # Keep the contract visible on the scheduler itself as well as in the
        # final row count. A caller cannot bypass the production Locate cap by
        # supplying a larger ``max_screenshots`` or ``locate_cap`` value.
        self.locate_cap = min(1500, max(1, int(locate_cap)))
        # The change-anchor budget is a hard contract, not a hint supplied by
        # a caller. It can never consume more than 300 Locate screenshots.
        self.reserved_change_events = min(
            300,
            max(0, int(reserved_change_events)),
            self.locate_cap,
        )

    @property
    def base_interval_seconds(self) -> float:
        if self.mode == "exhaustive_validation":
            return 10.0 / self.fps
        return max(1.0, self.duration_seconds / 1200.0)

    @property
    def cap(self) -> int:
        if self.mode == "exhaustive_validation":
            return min(300, max(1, int(self.max_screenshots or 300)))
        configured = self.locate_cap if self.max_screenshots is None else int(self.max_screenshots)
        return min(1500, max(1, configured))

    def schedule(
        self,
        *,
        extra_times: Iterable[float] = (),
        change_anchors: Iterable[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Build a bounded schedule from base PTS and CPU change anchors.

        ``extra_times`` remains a compatibility input for callers that only
        have timestamps. Production S1 supplies ``change_anchors`` from a CPU
        frame-difference pass, and those anchors are selected with temporal
        non-maximum suppression so late changes cannot be discarded by a
        first-N truncation.
        """

        if self.mode == "exhaustive_validation":
            stride = 10
            indices = list(range(0, self.frame_count, stride))[: self.cap]
            return [{"frame_index": index, "pts_seconds": index / self.fps, "anchor_source": "fixed_10_frames"} for index in indices]
        raw_anchors: list[dict[str, Any]] = []
        for value in change_anchors:
            try:
                pts = float(value.get("pts_seconds", value.get("time_seconds", 0.0)))
            except (TypeError, ValueError, AttributeError):
                continue
            row = dict(value)
            row["pts_seconds"] = pts
            raw_anchors.append(row)
        for value in extra_times:
            try:
                raw_anchors.append({"pts_seconds": float(value), "anchor_source": "cpu_change_anchor"})
            except (TypeError, ValueError):
                continue
        end_time = max(0.0, self.duration_seconds - 1.0 / self.fps)
        normalized_anchors: list[dict[str, Any]] = []
        for row in raw_anchors:
            value = min(max(0.0, float(row["pts_seconds"])), end_time)
            normalized = {
                **row,
                "pts_seconds": round(value, 6),
                "anchor_source": str(row.get("anchor_source", "cpu_change_anchor")),
            }
            normalized["frame_index"] = min(
                self.frame_count - 1,
                max(0, int(round(value * self.fps))),
            )
            normalized_anchors.append(normalized)
        # Prefer high-scoring changes, then use temporal NMS. A second pass
        # fills empty front/middle/back thirds when the detector supplied
        # enough candidates, preserving long-video coverage without static
        # first-N selection.
        normalized_anchors.sort(
            key=lambda row: (
                -float(row.get("change_score", row.get("score", 0.0)) or 0.0),
                float(row["pts_seconds"]),
                int(row["frame_index"]),
            )
        )
        selected_anchors: list[dict[str, Any]] = []
        min_anchor_gap = max(1.0 / self.fps, self.base_interval_seconds * 0.25)
        for row in normalized_anchors:
            if len(selected_anchors) >= self.reserved_change_events:
                break
            if any(abs(float(row["pts_seconds"]) - float(item["pts_seconds"])) < min_anchor_gap for item in selected_anchors):
                continue
            selected_anchors.append(row)
        if selected_anchors and self.reserved_change_events >= 3:
            # Explicitly protect the three broad temporal regions. This is
            # only a tie-breaker for the change budget; base anchors already
            # cover the complete video at the configured interval.
            regions = (
                (0.0, self.duration_seconds / 3.0),
                (self.duration_seconds / 3.0, 2.0 * self.duration_seconds / 3.0),
                (2.0 * self.duration_seconds / 3.0, self.duration_seconds + 1.0 / self.fps),
            )
            def region_index(row: Mapping[str, Any]) -> int:
                pts = float(row["pts_seconds"])
                for index, (lower, upper) in enumerate(regions):
                    if lower <= pts < upper:
                        return index
                return len(regions) - 1

            for target_region, (lower, upper) in enumerate(regions):
                if any(region_index(item) == target_region for item in selected_anchors):
                    continue
                region = [
                    row for row in normalized_anchors
                    if lower <= float(row["pts_seconds"]) < upper
                    and all(abs(float(row["pts_seconds"]) - float(item["pts_seconds"])) >= min_anchor_gap for item in selected_anchors)
                ]
                if not region:
                    continue
                replacement = region[0]
                if len(selected_anchors) < self.reserved_change_events:
                    selected_anchors.append(replacement)
                    continue
                # Replace the weakest row outside the protected region. This
                # keeps front/middle/back coverage even when all early peaks
                # have identical scores.
                region_counts = {
                    index: sum(region_index(item) == index for item in selected_anchors)
                    for index in range(len(regions))
                }
                removable = [
                    item for item in selected_anchors
                    if region_index(item) != target_region
                    and region_counts[region_index(item)] > 1
                ]
                if not removable:
                    continue
                drop = min(
                    removable,
                    key=lambda row: (
                        float(row.get("change_score", row.get("score", 0.0)) or 0.0),
                        -float(row["pts_seconds"]),
                    ),
                )
                selected_anchors.remove(drop)
                selected_anchors.append(replacement)
        selected_anchors.sort(key=lambda row: (float(row["pts_seconds"]), int(row["frame_index"])))
        times: list[float] = []
        current = 0.0
        mandatory_count = 3 if self.cap >= 3 else 1
        base_slots = max(1, self.cap - max(0, self.reserved_change_events) - mandatory_count)
        while current < self.duration_seconds and len(times) < base_slots:
            times.append(current)
            current += self.base_interval_seconds
        if not times:
            times = [0.0]
        times.append(end_time)
        base_rows = [{
            "frame_index": min(self.frame_count - 1, max(0, int(round(value * self.fps)))),
            "pts_seconds": round(value, 6),
            "anchor_source": "adaptive_base",
        } for value in times]
        by_frame: dict[int, dict[str, Any]] = {int(row["frame_index"]): row for row in base_rows}
        if self.cap >= 3:
            mandatory = (
                (0, "mandatory_start"),
                (self.frame_count // 2, "mandatory_middle"),
                (self.frame_count - 1, "mandatory_end"),
            )
        else:
            mandatory = ((0, "mandatory_start"),)
        for frame_index, reason in mandatory:
            frame_index = min(self.frame_count - 1, max(0, int(frame_index)))
            row = by_frame.get(frame_index)
            if row is None:
                row = {
                    "frame_index": frame_index,
                    "pts_seconds": round(frame_index / self.fps, 6),
                    "anchor_source": "adaptive_base",
                }
                by_frame[frame_index] = row
            row.update({"mandatory_anchor": True, "anchor_reason": reason})
        for row in selected_anchors:
            frame_index = int(row["frame_index"])
            existing = by_frame.get(frame_index)
            if existing is None:
                by_frame[frame_index] = row
                continue
            # A detected change wins over a coincident periodic anchor, while
            # retaining both labels for an auditable schedule.
            sources = list(existing.get("anchor_sources", (existing.get("anchor_source", "adaptive_base"),)))
            if row.get("anchor_source") not in sources:
                sources.append(row["anchor_source"])
            existing.update({"anchor_source": row["anchor_source"], "anchor_sources": sources})
            for key in ("change_score", "anchor_reason", "detector"):
                if key in row:
                    existing[key] = row[key]
        rows = sorted(by_frame.values(), key=lambda row: (int(row["frame_index"]), float(row["pts_seconds"])))
        # The periodic reserve can be smaller than one frame for very short
        # clips. Keep the hard cap while retaining all selected change rows.
        if len(rows) > self.cap:
            mandatory_rows = [row for row in rows if row.get("mandatory_anchor")]
            change_rows = [
                row for row in rows
                if row not in mandatory_rows
                and (str(row.get("anchor_source", "")).startswith("cpu_") or "change" in str(row.get("anchor_source", "")))
            ]
            base_rows = [row for row in rows if row not in mandatory_rows and row not in change_rows]
            keep = [*mandatory_rows[: self.cap], *change_rows[: max(0, self.cap - len(mandatory_rows))]]
            remaining = max(0, self.cap - len(keep))
            if remaining and base_rows:
                if remaining >= len(base_rows):
                    keep.extend(base_rows)
                elif remaining == 1:
                    keep.append(base_rows[len(base_rows) // 2])
                else:
                    keep.extend(
                        base_rows[round(index * (len(base_rows) - 1) / (remaining - 1))]
                        for index in range(remaining)
                    )
            rows = sorted(keep, key=lambda row: (int(row["frame_index"]), float(row["pts_seconds"])))[: self.cap]
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "base_interval_seconds": self.base_interval_seconds,
            "cap": self.cap,
            "locate_cap": self.locate_cap,
            "reserved_change_events": self.reserved_change_events,
            "change_anchor_cap": self.reserved_change_events,
            "mandatory_anchors": ["start", "middle", "end"] if self.cap >= 3 and self.mode == "production_adaptive" else ["start"],
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
        }


def runtime_fingerprint() -> dict[str, Any]:
    """Best-effort environment details for the reproducibility manifest."""
    result: dict[str, Any] = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "os": os.name}
    try:
        import torch  # type: ignore
        result["torch"] = getattr(torch, "__version__", "unknown")
        result["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}:{exc}"
    try:
        import transformers  # type: ignore
        result["transformers"] = getattr(transformers, "__version__", "unknown")
    except Exception as exc:
        result["transformers_error"] = f"{type(exc).__name__}:{exc}"
    return result


def hash_tree(root: str | Path, *, relative_to: str | Path | None = None) -> dict[str, str]:
    root_path = Path(root).resolve()
    anchor = Path(relative_to).resolve() if relative_to is not None else root_path
    result: dict[str, str] = {}
    if root_path.is_file():
        return {root_path.name: sha256_file(root_path)}
    if not root_path.is_dir():
        return result
    for path in sorted(item for item in root_path.rglob("*") if item.is_file() and ".runtime" not in item.parts and "__pycache__" not in item.parts):
        result[path.relative_to(anchor).as_posix()] = sha256_file(path)
    return result


def artifact_ref(run_dir: str | Path, uri: str, *, kind: str = "artifact") -> ArtifactRef:
    """Hash one relative file or directory after its producer has committed it."""

    root = Path(run_dir).resolve()
    normalized = str(uri).replace("\\", "/").strip("/")
    # Reuse the public contract's traversal/absolute-path validation before
    # resolving the target on disk.
    ArtifactRef(uri=normalized, kind=kind)
    target = (root / normalized).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact_outside_run_dir:{normalized}") from exc
    if target.is_file():
        return ArtifactRef(
            uri=normalized,
            kind=kind,
            sha256=sha256_file(target),
            bytes=target.stat().st_size,
            metadata={"node_type": "file"},
        )
    if target.is_dir():
        files = hash_tree(target)
        total_bytes = sum(item.stat().st_size for item in target.rglob("*") if item.is_file())
        return ArtifactRef(
            uri=normalized,
            kind=kind,
            sha256=canonical_sha256(files),
            bytes=total_bytes,
            metadata={"node_type": "directory", "file_count": len(files)},
        )
    raise FileNotFoundError(f"stage_artifact_missing:{normalized}")


def artifact_refs(run_dir: str | Path, uris: Iterable[str]) -> tuple[ArtifactRef, ...]:
    """Return verified, de-duplicated artifact references in input order."""

    seen: set[str] = set()
    result: list[ArtifactRef] = []
    for uri in uris:
        normalized = str(uri).replace("\\", "/").strip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(artifact_ref(run_dir, normalized))
    return tuple(result)


def read_stage_receipt(run_dir: str | Path, stage_id: str) -> dict[str, Any] | None:
    path = Path(run_dir).resolve() / "stage_receipts" / f"{stage_id}.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"stage_receipt_not_object:{stage_id}")
    return value


def verify_stage_receipt(
    run_dir: str | Path,
    stage_id: str,
    *,
    expected_input_fingerprint: str | None = None,
    expected_fingerprint_version: str | None = None,
    reuse_commit_hashes: bool = False,
) -> tuple[bool, tuple[str, ...], dict[str, Any] | None]:
    """Validate receipt envelope, stage input identity and every artifact hash."""

    try:
        receipt = read_stage_receipt(run_dir, stage_id)
    except Exception as exc:
        return False, (f"receipt_read_error:{type(exc).__name__}:{exc}",), None
    if receipt is None:
        return False, ("receipt_missing",), None
    reasons: list[str] = []
    if receipt.get("stage_id") != stage_id:
        reasons.append("stage_id_mismatch")
    if receipt.get("status") not in {"completed", "warning"}:
        reasons.append(f"receipt_status_not_reusable:{receipt.get('status')}")
    expected_receipt_sha = canonical_sha256({key: item for key, item in receipt.items() if key != "sha256"})
    if receipt.get("sha256") != expected_receipt_sha:
        reasons.append("receipt_sha256_mismatch")
    if expected_input_fingerprint is not None and receipt.get("input_fingerprint_sha256") != expected_input_fingerprint:
        reasons.append("input_fingerprint_mismatch")
    if expected_fingerprint_version is not None and receipt.get("fingerprint_version") != expected_fingerprint_version:
        reasons.append("stage_dependency_fingerprint_missing_or_obsolete")
    rows = receipt.get("artifacts", ())
    if not isinstance(rows, list):
        reasons.append("artifact_refs_invalid")
        rows = ()
    elif not rows:
        reasons.append("artifact_refs_empty")
    for raw in rows:
        try:
            reference = ArtifactRef.from_dict(raw)
            with nullcontext() if reuse_commit_hashes else uncached_hashes():
                current = artifact_ref(run_dir, reference.uri, kind=reference.kind)
            if current.sha256 != reference.sha256:
                reasons.append(f"artifact_sha256_mismatch:{reference.uri}")
            if current.bytes != reference.bytes:
                reasons.append(f"artifact_bytes_mismatch:{reference.uri}")
        except Exception as exc:
            uri = raw.get("uri", "unknown") if isinstance(raw, Mapping) else "unknown"
            reasons.append(f"artifact_verify_error:{uri}:{type(exc).__name__}:{exc}")
    return not reasons, tuple(reasons), receipt


def load_state(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir).resolve() / "run_state.v1.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def recovery_plan(run_dir: str | Path, *, current_input_hash: str | None = None, current_config_hash: str | None = None, current_code_hash: str | None = None) -> dict[str, Any]:
    state = load_state(run_dir)
    metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
    invalid_reasons: list[str] = []
    for key, value in (("input_sha256", current_input_hash), ("config_sha256", current_config_hash), ("code_sha256", current_code_hash)):
        previous = metadata.get(key)
        if value is not None and previous and str(previous) != str(value):
            invalid_reasons.append(key)
    stages = state.get("stages") if isinstance(state.get("stages"), Mapping) else {}
    last_completed = None
    for stage_id, _name, _weight in STAGE_DEFINITIONS:
        if isinstance(stages.get(stage_id), Mapping) and stages[stage_id].get("status") == "completed":
            last_completed = stage_id
        else:
            break
    if invalid_reasons:
        first = STAGE_IDS[0]
        return {"status": "invalidated", "resume_from": first, "last_completed": last_completed, "invalidated_by": invalid_reasons}
    next_index = 0 if last_completed is None else min(len(STAGE_IDS) - 1, STAGE_IDS.index(last_completed) + 1)
    return {"status": "ready", "resume_from": STAGE_IDS[next_index], "last_completed": last_completed, "invalidated_by": []}


__all__ = [
    "AdaptiveScreenshotScheduler", "STAGE_DEFINITIONS", "STAGE_IDS", "STAGE_NAMES", "STAGE_WEIGHTS", "StageLedger", "artifact_ref", "artifact_refs", "hash_tree", "load_state", "read_stage_receipt", "recovery_plan", "runtime_fingerprint", "stage_progress", "utc_now", "verify_stage_receipt", "weighted_progress",
]
