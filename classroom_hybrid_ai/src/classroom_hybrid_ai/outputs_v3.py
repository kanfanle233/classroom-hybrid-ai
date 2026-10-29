"""Portable v3 artifacts: timeline, overlays, previews, and final rendering.

The S7 output writer is deliberately independent from the inference stages.  It
reads their durable JSONL artifacts, builds bounded overlay chunks, and renders
all three video semantics in one sequential decode of the source video.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from html import escape as _html_escape
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np

from .hashing import canonical_sha256, sha256_file
from .performance import increment, record_timing
from .profiles import ActionProfile
from .storage import JsonlStreamWriter, atomic_write_json, atomic_write_jsonl, iter_jsonl


GRAY = "#B8B8B8"
UNKNOWN_ID = "unknown"
OVERLAY_CHUNK_SECONDS = 60.0
MAX_BOX_HOLD_SECONDS = 2.0
# This is a frozen contract value.  The render function accepts the old
# parameter for source compatibility, but never lets callers change it.
PROPAGATED_ALPHA = 0.55
# Portable English labels used by renderers and JSON display fields.  The
# canonical Chinese taxonomy remains in ``action_name_zh`` for traceability.
ASCII_ACTION_LABELS = {
    "dx": "HEAD_DOWN_WRITING",
    "dk": "HEAD_DOWN_READING",
    "tt": "LOOKING_UP_LISTENING",
    "zt": "TURNING_HEAD",
    "js": "HAND_RAISED",
    "zl": "STANDING",
    "xt": "PEER_DISCUSSION",
    "jz": "TEACHER_GUIDANCE",
    "bs": "BOARD_WRITING",
    "zs": "PRESENTING",
    "hd": "ANSWERING_QUESTION",
    "unknown": "UNKNOWN",
}


def _english_action_name(action_code: Any, fallback: Any = None) -> str:
    """Return an ASCII display action without changing taxonomy semantics."""

    code = str(action_code or "").strip().lower().removeprefix("a_")
    mapped = ASCII_ACTION_LABELS.get(code)
    if mapped:
        return mapped
    value = str(fallback or "UNKNOWN").encode("ascii", "ignore").decode("ascii").strip()
    return value or "UNKNOWN"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.is_file() else []


def taxonomy_payload(profile: ActionProfile) -> dict[str, Any]:
    payload = profile.to_dict()
    payload["contract_sha256"] = canonical_sha256({key: value for key, value in payload.items() if key != "contract_sha256"})
    return payload


def _student_rows(run_dir: Path, duration: float) -> list[dict[str, Any]]:
    # Locate-first is the identity source.  The legacy stitching output may
    # contain historical track-epoch aliases, so prefer the flat Locate box
    # artifact when it exists and keep its project ``student_id`` verbatim.
    by_id: dict[str, dict[str, Any]] = {}

    def ingest(rows: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for row in rows:
            count += 1
            if row.get("student_confirmed") is False:
                continue
            student_id = str(row.get("student_id") or row.get("track_id") or "").strip()
            if not student_id:
                continue
            current = by_id.setdefault(student_id, {"student_id": student_id, "first_pts_seconds": None, "last_pts_seconds": None, "observations": 0})
            pts = row.get("pts_seconds", row.get("first_pts_seconds"))
            if isinstance(pts, (int, float)):
                current["first_pts_seconds"] = pts if current["first_pts_seconds"] is None else min(current["first_pts_seconds"], pts)
                current["last_pts_seconds"] = pts if current["last_pts_seconds"] is None else max(current["last_pts_seconds"], pts)
            current["observations"] = int(current.get("observations", 0)) + 1
            for key in ("identity_scope", "bbox", "seat_anchor"):
                if key in row and key not in current:
                    current[key] = row[key]
        return count

    localization_path = run_dir / "artifacts/localization_boxes.v1.jsonl"
    if localization_path.is_file():
        # S3's complete generic audit is authoritative even when every person
        # is unverified. Never revive students from a previous S7 attempt.
        ingest(iter_jsonl(localization_path))
        return [by_id[key] for key in sorted(by_id)]
    locate_path = run_dir / "artifacts/locate_student_boxes.jsonl"
    student_path = run_dir / "students.jsonl"
    locate_count = ingest(iter_jsonl(locate_path)) if locate_path.is_file() else 0
    if not locate_count:
        by_id.clear()
        student_count = ingest(iter_jsonl(student_path)) if student_path.is_file() else 0
    else:
        student_count = locate_count
    track_path = run_dir / "artifacts/track_observations.jsonl"
    if not by_id and not student_count and track_path.is_file():
        ingest(iter_jsonl(track_path))
    return [by_id[key] for key in sorted(by_id)]


def _action_lookup(profile: ActionProfile) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        UNKNOWN_ID: {
            "action_id": UNKNOWN_ID,
            "action_code": "UNKNOWN",
            "action_name_zh": "证据不足",
            "action_name_en": "UNKNOWN",
            "display_label": "UNKNOWN",
            "color": GRAY,
        }
    }
    for action in profile.actions:
        result[str(action["action_id"])] = {
            "action_id": action["action_id"],
            "action_code": action["code"],
            "action_name_zh": action["name_zh"],
            "action_name_en": _english_action_name(action["code"], action["name_en"]),
            "display_label": _english_action_name(action["code"], action["name_en"]),
            "color": action["color"],
        }
        result[str(action["code"])] = result[str(action["action_id"])]
    return result


def _evidence_event_ids(raw: Mapping[str, Any]) -> list[str]:
    """Recover durable Qwen candidate IDs from current and legacy rows."""

    values = raw.get("evidence_event_ids")
    if values is None:
        values = raw.get("evidence_event_id")
    if values is None and isinstance(raw.get("metadata"), Mapping):
        values = raw["metadata"].get("candidate_event_id")
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, (list, tuple)):
        return []
    return list(dict.fromkeys(
        str(value).strip() for value in values if isinstance(value, str) and value.strip()
    ))


def normalize_intervals(run_dir: Path, profile: ActionProfile, duration: float, students: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    existing = _read_rows(run_dir / "action_intervals.jsonl")
    lookup = _action_lookup(profile)
    by_student: dict[str, list[dict[str, Any]]] = {}
    for raw in existing:
        student_id = str(raw.get("student_id", "")).strip()
        if not student_id:
            continue
        action_id = str(raw.get("action_id") or raw.get("primary_action_id") or UNKNOWN_ID)
        action = lookup.get(action_id, lookup[UNKNOWN_ID])
        start = max(0.0, float(raw.get("start_pts_seconds", raw.get("start", 0.0)) or 0.0))
        end = min(duration, float(raw.get("end_pts_seconds", raw.get("end", duration)) or duration))
        if end < start:
            continue
        status = str(raw.get("status", "unobserved"))
        if action_id == UNKNOWN_ID or status in {"unobserved", "uncertain", "gray"}:
            action = lookup[UNKNOWN_ID]
            status = "unobserved"
        else:
            status = "confirmed" if status == "confirmed" else "propagated"
        evidence_event_ids = _evidence_event_ids(raw)
        if status == "confirmed" and str(raw.get("source", "")) == "qwen_classification" and not evidence_event_ids:
            raise ValueError("confirmed_qwen_interval_missing_evidence_event_id")
        by_student.setdefault(student_id, []).append({
            "student_id": student_id, "start_pts_seconds": start, "end_pts_seconds": end,
            "action_id": action["action_id"], "action_code": action["action_code"],
            "action_name_zh": action["action_name_zh"],
            "action_name_en": action["action_name_en"],
            "display_label": action["display_label"],
            "color": action["color"],
            "status": status, "source": raw.get("source", "qwen"),
            "taxonomy_profile_id": profile.profile_id,
            "taxonomy_contract_sha256": profile.sha256,
            "unobserved_reason": raw.get("unobserved_reason") if status == "unobserved" else None,
            "evidence_event_ids": evidence_event_ids,
            "evidence_event_id": evidence_event_ids[0] if len(evidence_event_ids) == 1 else None,
        })
    result: list[dict[str, Any]] = []
    for student in students:
        student_id = str(student["student_id"])
        rows = sorted(by_student.get(student_id, []), key=lambda item: (item["start_pts_seconds"], item["end_pts_seconds"]))
        cursor = 0.0
        for row in rows:
            start = max(cursor, float(row["start_pts_seconds"]))
            end = max(start, float(row["end_pts_seconds"]))
            if start > cursor:
                result.append(_gray_interval(student_id, cursor, start, "no_qwen_evidence"))
            if end > start:
                result.append({**row, "start_pts_seconds": start, "end_pts_seconds": end})
            cursor = max(cursor, end)
        if cursor < duration:
            result.append(_gray_interval(student_id, cursor, duration, "insufficient_or_missing_evidence"))
    # Enforce no overlap after source artifacts are merged.
    compact: list[dict[str, Any]] = []
    for row in sorted(result, key=lambda item: (str(item["student_id"]), float(item["start_pts_seconds"]), float(item["end_pts_seconds"]))):
        if compact and compact[-1]["student_id"] == row["student_id"] and row["start_pts_seconds"] < compact[-1]["end_pts_seconds"]:
            row = {**row, "start_pts_seconds": compact[-1]["end_pts_seconds"]}
        if row["end_pts_seconds"] > row["start_pts_seconds"]:
            compact.append(row)
    return compact


def _gray_interval(student_id: str, start: float, end: float, reason: str) -> dict[str, Any]:
    return {
        "student_id": student_id,
        "start_pts_seconds": float(start),
        "end_pts_seconds": float(end),
        "action_id": UNKNOWN_ID,
        "action_code": "UNKNOWN",
        "action_name_zh": "证据不足",
        "action_name_en": "UNKNOWN",
        "display_label": "UNKNOWN",
        "color": GRAY,
        "status": "unobserved",
        "source": "temporal_merge",
        "unobserved_reason": reason,
        "evidence_event_id": None,
    }


def build_action_observations(run_dir: Path, profile: ActionProfile) -> list[dict[str, Any]]:
    rows = _read_rows(run_dir / "action_observations.jsonl")
    lookup = _action_lookup(profile)
    result: list[dict[str, Any]] = []
    for row in rows:
        action = lookup.get(str(row.get("primary_action_id", "")), lookup[UNKNOWN_ID])
        if str(row.get("observability", "none")) == "none":
            action = lookup[UNKNOWN_ID]
        evidence_event_ids = _evidence_event_ids(row)
        result.append({
            **row,
            "action_id": action["action_id"],
            "action_code": action["action_code"],
            "action_name_zh": action["action_name_zh"],
            "action_name_en": action["action_name_en"],
            "display_label": action["display_label"],
            "color": action["color"],
            "qwen_self_reported_confidence_audit_only": row.get("self_reported_confidence"),
            "taxonomy_profile_id": profile.profile_id,
            "taxonomy_contract_sha256": profile.sha256,
            "evidence_event_ids": evidence_event_ids,
            "evidence_event_id": evidence_event_ids[0] if len(evidence_event_ids) == 1 else None,
        })
    return result


def _intervals_by_student(intervals: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Index intervals once so each observation advances a single cursor.

    The previous implementation called ``next`` over every interval for every
    observation.  This index makes overlay lookup O(N + M) for each student
    stream (with M intervals and N observations), rather than O(N*M).
    """

    by_student: dict[str, list[Mapping[str, Any]]] = {}
    for item in intervals:
        student_id = str(item.get("student_id", "")).strip()
        if student_id:
            by_student.setdefault(student_id, []).append(item)
    for values in by_student.values():
        values.sort(key=lambda item: (float(item.get("start_pts_seconds", 0.0)), float(item.get("end_pts_seconds", 0.0))))
    return by_student


def _action_at(
    values: Sequence[Mapping[str, Any]],
    cursor: int,
    pts: float,
    student_id: str,
) -> tuple[Mapping[str, Any], int]:
    while cursor < len(values) and float(values[cursor].get("end_pts_seconds", 0.0)) <= pts:
        cursor += 1
    if cursor < len(values):
        current = values[cursor]
        start = float(current.get("start_pts_seconds", 0.0))
        end = float(current.get("end_pts_seconds", 0.0))
        if start <= pts < end:
            return current, cursor
    return _gray_interval(student_id, pts, pts, "no_interval"), cursor


def _bbox_from_observation(row: Mapping[str, Any]) -> list[float] | None:
    detection = row.get("detection") if isinstance(row.get("detection"), Mapping) else row
    bbox = detection.get("bbox") if isinstance(detection, Mapping) else None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _observation_detector_source(
    row: Mapping[str, Any], associated_box: Mapping[str, Any] | None,
) -> tuple[str | None, bool]:
    """Use explicit provenance; infer old Locate receipts only when unversioned."""
    detection = row.get("detection") if isinstance(row.get("detection"), Mapping) else row
    attributes = detection.get("attributes", {}) if isinstance(detection.get("attributes"), Mapping) else {}
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), Mapping) else {}
    source = detection.get("source") or row.get("detector_source") or row.get("source")
    if not source and associated_box is not None:
        source = associated_box.get("source") or associated_box.get("detector_source")
    generic = bool(metadata.get("localization_contract")) or source not in (None, "", "locate", "locate_anything")
    if source:
        return "locate_anything" if source == "locate" else str(source), generic
    if generic:
        return "unknown", True
    # Old Locate-only artifacts predate a detector-source field. Their
    # compatibility fallback must never override an explicit YOLO provider.
    if (associated_box is not None or attributes.get("locate_detection_index") is not None
            or row.get("locate_detection_index") is not None):
        return "locate_anything", False
    return None, False


def _generic_detection_index(row: Mapping[str, Any], associated_box: Mapping[str, Any] | None) -> Any:
    detection = row.get("detection") if isinstance(row.get("detection"), Mapping) else row
    attributes = detection.get("attributes", {}) if isinstance(detection.get("attributes"), Mapping) else {}
    sources = (detection, attributes, row, associated_box or {})
    for keys in (("detector_detection_index", "detection_index"), ("locate_detection_index",)):
        for source in sources:
            for key in keys:
                if source.get(key) is not None:
                    return source[key]
    return None


def _overlay_payload(
    *,
    row: Mapping[str, Any],
    student_id: str,
    frame: int,
    pts: float,
    bbox: Sequence[float],
    locate: Mapping[str, Any] | None,
    width: float,
    height: float,
    action: Mapping[str, Any],
    bbox_source: str,
    bbox_observed: bool,
    bbox_age_seconds: float,
    bbox_interpolation: str,
) -> dict[str, Any]:
    detection = row.get("detection") if isinstance(row.get("detection"), Mapping) else row
    attributes = detection.get("attributes") if isinstance(detection, Mapping) and isinstance(detection.get("attributes"), Mapping) else {}
    locate_index = None if locate is None else locate.get("locate_detection_index")
    if locate_index is None and isinstance(detection, Mapping):
        locate_index = detection.get("locate_detection_index", attributes.get("locate_detection_index"))
    association_quality = None if locate is None else locate.get("association_score")
    if association_quality is None and isinstance(detection, Mapping):
        association_quality = detection.get("association_score", attributes.get("association_score"))
    x1, y1, x2, y2 = [float(value) for value in bbox]
    action_code = str(action.get("action_code", "UNKNOWN"))
    action_name_en = _english_action_name(action_code, action.get("action_name_en"))
    payload = {
        "pts_seconds": float(pts),
        "frame_index": int(frame),
        "student_id": student_id,
        "locate_detection_index": locate_index,
        "bbox_px": [x1, y1, x2, y2],
        "bbox_norm": [x1 / max(1.0, width), y1 / max(1.0, height), x2 / max(1.0, width), y2 / max(1.0, height)],
        "bbox_source": bbox_source,
        "bbox_observed": bool(bbox_observed),
        "bbox_reliable": True,
        "bbox_age_seconds": round(max(0.0, float(bbox_age_seconds)), 6),
        "bbox_interpolation": bbox_interpolation,
        "association_quality": association_quality,
        "action_id": action.get("action_id", UNKNOWN_ID),
        "action_code": action_code,
        "action_name_zh": action.get("action_name_zh", "证据不足"),
        "action_name_en": action_name_en,
        "display_label": _english_action_name(action_code, action.get("display_label", action_name_en)),
        "color": action.get("color", GRAY),
        "interval_status": action.get("status", "unobserved"),
        "evidence_event_id": action.get("evidence_event_id"),
    }
    provider, generic = _observation_detector_source(row, locate)
    if generic:
        payload.update({"detector_source": provider or "unknown",
                        "detection_index": _generic_detection_index(row, locate)})
        if provider != "locate_anything":
            payload["locate_detection_index"] = None
    return payload


class _LocateCursor:
    """Read Locate boxes in frame order with only one frame buffered."""

    def __init__(self, path: Path) -> None:
        self._iterator = iter_jsonl(path) if path.is_file() else iter(())
        self._pending: Mapping[str, Any] | None = None
        self._served_frame: int | None = None
        self._served_values: dict[str, Mapping[str, Any]] = {}
        self._advance()

    def _advance(self) -> None:
        try:
            self._pending = next(self._iterator)
        except StopIteration:
            self._pending = None

    def take(self, frame_index: int) -> dict[str, Mapping[str, Any]]:
        if self._served_frame == frame_index:
            return self._served_values
        while self._pending is not None and int(self._pending.get("frame_index", -1)) < frame_index:
            self._advance()
        values: dict[str, Mapping[str, Any]] = {}
        while self._pending is not None and int(self._pending.get("frame_index", -1)) == frame_index:
            student_id = str(self._pending.get("student_id", ""))
            if student_id:
                values[student_id] = self._pending
            self._advance()
        self._served_frame = frame_index
        self._served_values = values
        return values


def iter_overlay_rows(run_dir: Path, info: Mapping[str, Any], intervals: Sequence[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield overlays in global frame order with bounded active state.

    The JSONL observations are consumed one line at a time.  For each video
    frame, all observations at that frame update the active state first, then
    active students are emitted in stable ID order.  This makes sparse tracks
    reliable for at most two seconds while keeping output monotonic for both
    chunking and the renderer.  Memory is proportional to active students,
    not video length.
    """

    observation_path = run_dir / "artifacts/track_observations.jsonl"
    if not observation_path.is_file():
        return
    fps = float(info.get("fps", 1.0) or 1.0)
    width = float(info.get("width", 1.0) or 1.0)
    height = float(info.get("height", 1.0) or 1.0)
    duration = max(0.0, float(info.get("duration_seconds", 0.0) or 0.0))
    frame_count_hint = int(info.get("frame_count", 0) or 0)
    frame_limit = max(frame_count_hint - 1, int(math.ceil(duration * fps)) - 1)
    intervals_by_student = _intervals_by_student(intervals)
    locate_cursor = _LocateCursor(run_dir / "artifacts/locate_student_boxes.jsonl")
    states: dict[str, dict[str, Any]] = {}
    observations = iter(iter_jsonl(observation_path))
    pending: Mapping[str, Any] | None = None

    def advance() -> None:
        nonlocal pending
        try:
            pending = next(observations)
        except StopIteration:
            pending = None

    def point(
        state: Mapping[str, Any],
        *,
        frame: int,
        pts: float,
        bbox: Sequence[float],
        locate: Mapping[str, Any] | None,
        source: str,
        observed: bool,
        age: float,
    ) -> dict[str, Any]:
        student_id = str(state["student_id"])
        values = intervals_by_student.get(student_id, ())
        cursor = int(state.get("interval_cursor", 0))
        action, cursor = _action_at(values, cursor, pts, student_id)
        state["interval_cursor"] = cursor  # type: ignore[index]
        return _overlay_payload(
            row=state["raw"],
            student_id=student_id,
            frame=frame,
            pts=pts,
            bbox=bbox,
            locate=locate,
            width=width,
            height=height,
            action=action,
            bbox_source=source,
            bbox_observed=observed,
            bbox_age_seconds=age,
            bbox_interpolation="none" if observed else "hold",
        )

    advance()
    frame = 0
    while pending is not None or states:
        if pending is None and frame_limit >= 0 and frame > frame_limit:
            break
        frame_rows: dict[str, Mapping[str, Any]] = {}
        # Input artifacts are frame sorted.  Keep only one frame's rows in
        # memory, so a long stream never accumulates observations.
        while pending is not None:
            try:
                pending_frame = int(pending.get("frame_index", -1))
            except (TypeError, ValueError):
                advance()
                continue
            if pending_frame < frame:
                advance()
                continue
            if pending_frame > frame:
                break
            student_id = str(pending.get("student_id") or pending.get("track_id") or "").strip()
            if student_id:
                frame_rows[student_id] = pending
            advance()
        locate_rows = locate_cursor.take(frame) if frame_rows else {}
        for student_id, raw in frame_rows.items():
            bbox = _bbox_from_observation(raw)
            if bbox is None:
                continue
            try:
                pts = float(raw.get("pts_seconds", frame / fps))
            except (TypeError, ValueError):
                pts = frame / fps
            previous = states.get(student_id)
            state = {
                "student_id": student_id,
                "frame": frame,
                "pts": pts,
                "bbox": bbox,
                "raw": raw,
                "interval_cursor": 0 if previous is None else int(previous.get("interval_cursor", 0)),
            }
            states[student_id] = state
        if frame_rows:
            # Real observations win over a hold at this frame, and all real
            # updates happen before output so students are ordered together.
            pass
        current_pts = frame / fps
        for student_id in sorted(states):
            state = states[student_id]
            observed = student_id in frame_rows and _bbox_from_observation(frame_rows[student_id]) is not None
            if observed:
                raw = frame_rows[student_id]
                bbox = state["bbox"]
                locate = locate_rows.get(student_id)
                provider, generic = _observation_detector_source(raw, locate)
                if generic:
                    source = "locate" if provider == "locate_anything" else provider or "unknown"
                else:
                    # Preserve unversioned Locate-only bbox_source bytes;
                    # generic providers above never use this legacy alias.
                    detection = raw.get("detection", {}) if isinstance(raw.get("detection"), Mapping) else {}
                    attributes = detection.get("attributes", {}) if isinstance(detection.get("attributes"), Mapping) else {}
                    legacy_locate = (locate is not None or attributes.get("locate_detection_index") is not None
                                     or raw.get("locate_detection_index") is not None)
                    source = "locate" if legacy_locate else str(raw.get("bbox_source") or "tracked")
                yield point(state, frame=frame, pts=float(state["pts"]), bbox=bbox, locate=locate,
                            source=source, observed=True, age=0.0)
            else:
                age = current_pts - float(state["pts"])
                if age <= MAX_BOX_HOLD_SECONDS + 1e-9:
                    yield point(state, frame=frame, pts=current_pts, bbox=state["bbox"], locate=None, source="held", observed=False, age=age)
        # Once a track has exceeded the reliability horizon, discard its tiny
        # state.  A later real observation recreates it; stale identities do
        # not accumulate across a long lesson.
        for student_id in [
            student_id
            for student_id, state in states.items()
            if current_pts - float(state["pts"]) > MAX_BOX_HOLD_SECONDS + 1e-9
        ]:
            states.pop(student_id, None)
        frame += 1


def build_overlay_rows(run_dir: Path, info: Mapping[str, Any], intervals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility wrapper returning overlays as a list.

    New S7 code can consume :func:`iter_overlay_rows` directly; existing
    callers retain the historical list-returning API.
    """

    return list(iter_overlay_rows(run_dir, info, intervals))


def write_overlay_chunks(run_dir: Path, rows: Iterable[Mapping[str, Any]], *, chunk_seconds: float = OVERLAY_CHUNK_SECONDS) -> list[str]:
    """Write at most one in-memory 60-second overlay chunk at a time."""

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds_must_be_positive")
    directory = run_dir / "frontend" / "overlay_chunks"
    directory.mkdir(parents=True, exist_ok=True)
    uris: list[str] = []
    current_index: int | None = None
    current_rows: list[Mapping[str, Any]] = []

    def flush(index: int, values: list[Mapping[str, Any]]) -> None:
        uri = f"frontend/overlay_chunks/{index:06d}.json"
        atomic_write_json(run_dir / uri, {
            "schema_version": "classroom_hybrid_ai.overlay_chunk.v1",
            "start_pts_seconds": index * chunk_seconds,
            "end_pts_seconds": (index + 1) * chunk_seconds,
            # Overlay chunks are public frontend data.  Keep the source spool
            # intact for audit, but publish ASCII labels so a client can draw
            # the same glyphs as the OpenCV video renderer.
            "rows": [_english_display_row(value) for value in values],
        })
        uris.append(uri)

    for row in rows:
        index = max(0, int(float(row.get("pts_seconds", 0.0)) // chunk_seconds))
        if current_index is None:
            current_index = index
        if index != current_index:
            flush(current_index, current_rows)
            current_rows = []
            current_index = index
        current_rows.append(row)
    if current_index is not None:
        flush(current_index, current_rows)
    if not uris:
        flush(0, [])
    return uris


def _preview_contact_sheet(root: Path, stage_id: str, paths: Sequence[Path], *, label_prefix: str = "") -> str | None:
    """Create one bounded contact sheet from real stage images."""

    valid = [path for path in paths if path.is_file()]
    if not valid:
        return None
    selected = [valid[min(len(valid) - 1, round(index * (len(valid) - 1) / 11))] for index in range(12)]
    tiles: list[Any] = []
    for index, path in enumerate(selected):
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (240, 135), interpolation=cv2.INTER_AREA)
        label = f"{label_prefix}{index + 1:02d} {path.stem}"[:42]
        cv2.putText(image, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    if not tiles:
        return None
    while len(tiles) < 12:
        tiles.append(tiles[-1].copy())
    target = root / "previews" / stage_id / "semantic_contact_12.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.vconcat([cv2.hconcat(tiles[row:row + 4]) for row in range(0, 12, 4)])
    return f"previews/{stage_id}/semantic_contact_12.jpg" if cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 90]) else None


def _render_s2_locate_frames(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Path], int, list[int]]:
    """Render bounded raw detections with their actual provider before S3."""

    usable = [
        row for row in rows
        if row.get("path") or row.get("screenshot_path")
    ]
    if not usable:
        return [], 0, []
    selected = [
        usable[min(len(usable) - 1, round(index * (len(usable) - 1) / 11))]
        for index in range(12)
    ]
    output_dir = root / "previews" / "S2" / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    detection_count = 0
    detection_indices: set[int] = set()
    for ordinal, row in enumerate(selected):
        source = Path(str(row.get("path") or row.get("screenshot_path")))
        image = cv2.imread(str(source))
        if image is None:
            continue
        height, width = image.shape[:2]
        for fallback_index, detection in enumerate(row.get("detections", ()), start=1):
            if not isinstance(detection, Mapping):
                continue
            bbox = detection.get("bbox") or detection.get("bbox_px")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
            except (TypeError, ValueError):
                continue
            x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
            y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
            try:
                locate_index = int(detection.get(
                    "locate_detection_index",
                    detection.get("detection_index", fallback_index),
                ))
            except (TypeError, ValueError):
                locate_index = fallback_index
            detection_indices.add(locate_index)
            detection_count += 1
            provider = str(detection.get("source", row.get("source", "locate_anything")))
            yolo_person = provider == "yolo"
            source_label = "YOLO" if yolo_person else "Locate" if provider == "locate_anything" else provider
            spatial_label = "person (unverified)" if yolo_person else str(detection.get("label", "student"))
            index_label = "D" if yolo_person else "L" if provider == "locate_anything" else "D"
            color = (128, 128, 128) if yolo_person else (0, 165, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1 if yolo_person else 2)
            cv2.putText(
                image,
                f"{source_label} {index_label}{locate_index} {spatial_label}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        frame_index = int(row.get("frame_index", ordinal))
        target = output_dir / f"localization_{frame_index:09d}_{ordinal:02d}.jpg"
        if cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            rendered.append(target)
    return rendered, detection_count, sorted(detection_indices)


def _timeline_preview(root: Path, intervals: Sequence[Mapping[str, Any]], duration: float) -> str | None:
    """Draw S6's merged interval evidence as an auditable, non-model image."""

    if not intervals:
        return None
    students = sorted({str(row.get("student_id", "")) for row in intervals if row.get("student_id")})
    if not students:
        return None
    width, row_height = 1200, 28
    height = max(80, 48 + row_height * min(40, len(students)))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right = 180, width - 20
    cv2.putText(canvas, "S6 merged action intervals (gray=UNKNOWN)", (left, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    scale = (right - left) / max(duration, 1e-6)
    student_index = {student: index for index, student in enumerate(students[:40])}
    for row in intervals:
        student = str(row.get("student_id", ""))
        if student not in student_index:
            continue
        y = 40 + student_index[student] * row_height
        start = max(0.0, float(row.get("start_pts_seconds", 0.0)))
        end = max(start, float(row.get("end_pts_seconds", start)))
        x1, x2 = int(left + start * scale), int(left + end * scale)
        x2 = max(x1 + 1, min(right, x2))
        color = _bgr(str(row.get("color", GRAY)))
        cv2.rectangle(canvas, (x1, y), (x2, y + row_height - 5), color, -1)
    for student, index in student_index.items():
        cv2.putText(canvas, student, (8, 58 + index * row_height), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    target = root / "previews" / "S6" / "timeline_merged.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    return "previews/S6/timeline_merged.jpg" if cv2.imwrite(str(target), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]) else None


def _write_stage_previews(
    root: Path,
    info: Mapping[str, Any],
    *,
    stage_ids: Sequence[str] | None = None,
    preserve_existing_stage_previews: bool = True,
) -> list[str]:
    """Create semantically labelled, auditable previews for selected stages.

    The normal S7 delivery path writes the complete set.  The backend also
    calls this writer immediately before sealing each earlier stage receipt;
    ``stage_ids`` keeps that call scoped to the stage that just completed.
    """

    duration = max(0.0, float(info.get("duration_seconds", 0.0) or 0.0))
    selected_stage_ids = set(stage_ids or ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"))
    writable_stage_ids = {
        stage_id for stage_id in selected_stage_ids
        if not (preserve_existing_stage_previews and (root / "previews" / stage_id / "preview.json").is_file())
    }

    def stage_rows(uri: str, *consumers: str) -> list[dict[str, Any]]:
        return _read_rows(root / uri) if writable_stage_ids.intersection(consumers) else []

    screenshot_dir = root / "artifacts" / "locate_screenshots"
    # A resumed run can retain screenshot directories from earlier invalidated
    # attempts.  The canonical schedule is the stage boundary for the active
    # attempt, so previews must not mix those stale images or inflate counts.
    schedule_rows = stage_rows("artifacts/screenshot_schedule.v1.jsonl", "S1", "S3")
    scheduled_paths = [
        Path(str(row["path"]))
        for row in schedule_rows
        if row.get("path") and Path(str(row["path"])).is_file()
    ]
    screenshot_paths = (
        scheduled_paths
        if scheduled_paths
        else (sorted(screenshot_dir.rglob("*.jpg")) if writable_stage_ids.intersection({"S1", "S3"}) and screenshot_dir.is_dir() else [])
    )
    raw_uri = ("artifacts/localization_raw_results.v1.jsonl"
               if (root / "artifacts/localization_raw_results.v1.jsonl").is_file()
               else "artifacts/locate_raw_results.jsonl")
    locate_raw_rows = stage_rows(raw_uri, "S2")
    # Raw S2 output takes precedence over any stale downstream S3 artifact.
    locate_boxes = stage_rows("artifacts/locate_student_boxes.jsonl", "S2") if not locate_raw_rows else []
    track_rows = stage_rows("artifacts/track_observations.jsonl", "S3")
    associated_uri = ("artifacts/localization_results.v1.jsonl"
                      if (root / "artifacts/localization_results.v1.jsonl").is_file()
                      else "artifacts/locate_results.jsonl")
    associated_rows = stage_rows(associated_uri, "S3")
    crop_queue = stage_rows("artifacts/qwen_crop_queue.jsonl", "S4", "S5")
    selected_candidates = stage_rows("artifacts/candidates.selected.jsonl", "S5")
    evidence_rows = stage_rows("evidence/index.jsonl", "S4", "S5")
    reviews = stage_rows("reviews.classification.jsonl", "S5")
    action_rows = stage_rows("action_observations.jsonl", "S5")
    intervals = stage_rows("action_intervals.jsonl", "S6")
    semantics = {
        "S0": "Input video contact sheet and input audit",
        "S1": "Frame sampling schedule and screenshot manifest",
        "S2": "Actual detector source and frame-local boxes; YOLO people remain unverified",
        "S3": "Confirmed student IDs, detector sources, and gray unverified people",
        "S4": "Student crops, evidence, and Qwen temporal composites",
        "S5": "English action labels, colors, UNKNOWN, and partial timeline observations",
        "S6": "Raw action anchors and non-overlapping merged intervals",
        "S7": "Full video, classroom timeline, report, and frontend index",
    }
    payloads: dict[str, dict[str, Any]] = {}
    for stage_id in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        payloads[stage_id] = {
            "schema_version": "classroom_hybrid_ai.preview.v1",
            "stage_id": stage_id,
            "video": dict(info),
            "semantic": semantics[stage_id],
            "preview_status": "unavailable",
            "reason": "stage_artifacts_not_available",
            "uris": [],
            "counts": {},
        }

    # S0/S1 use only their own input/screenshot artifacts.  A missing image is
    # reported as unavailable rather than silently re-labelled as another stage.
    s0_images = [root / "previews/S0" / f"input_{name}.jpg" for name in ("first", "middle", "last")]
    if any(path.is_file() for path in s0_images):
        payloads["S0"].update({"preview_status": "available", "reason": None, "uris": [path.relative_to(root).as_posix() for path in s0_images if path.is_file()], "counts": {"input_frames": sum(path.is_file() for path in s0_images)}})
    s1_stage_owned = (root / "previews/S1/preview.json").is_file()
    if "S1" in writable_stage_ids and screenshot_paths and (not s1_stage_owned or not preserve_existing_stage_previews):
        contact = _preview_contact_sheet(root, "S1", screenshot_paths, label_prefix="S1 ")
        payloads["S1"].update({"preview_status": "available", "reason": None, "uris": [contact] if contact else [], "counts": {"screenshot_count": len(screenshot_paths)}})

    s2_paths = [Path(str(row.get("annotated_path"))) for row in locate_boxes if row.get("annotated_path")] + [Path(str(row.get("screenshot_path"))) for row in locate_boxes if row.get("screenshot_path")]
    raw_detection_count = 0
    raw_detection_indices: list[int] = []
    if "S2" in writable_stage_ids and not s2_paths and locate_raw_rows:
        s2_paths, raw_detection_count, raw_detection_indices = _render_s2_locate_frames(
            root, locate_raw_rows
        )
    s2_contact = _preview_contact_sheet(root, "S2", s2_paths, label_prefix="detector ") if "S2" in writable_stage_ids else None
    if locate_boxes or locate_raw_rows:
        source_counts: dict[str, int] = {}
        source_frames: dict[str, int] = {}
        locate_indices: set[int] = set()
        for row in locate_raw_rows:
            row_source = str(row.get("source", "locate_anything"))
            source_frames[row_source] = source_frames.get(row_source, 0) + 1
            for index, detection in enumerate(row.get("detections", ()), 1):
                provider = str(detection.get("source", row_source))
                source_counts[provider] = source_counts.get(provider, 0) + 1
                if provider == "locate_anything":
                    locate_indices.add(int(detection.get("locate_detection_index", detection.get("detection_index", index))))
        payloads["S2"].update({"preview_status": "available", "reason": None,
            "uris": [s2_contact] if s2_contact else [], "source_artifact": raw_uri,
            "counts": {"localization_frame_rows": len(locate_raw_rows),
                "locate_frame_rows": source_frames.get("locate_anything", 0),
                "yolo_frame_rows": source_frames.get("yolo", 0),
                "locate_student_box_rows": len(locate_boxes),
                "locate_detection_count": source_counts.get("locate_anything", 0),
                "yolo_detection_count": source_counts.get("yolo", 0),
                "detections_by_source": source_counts,
                "student_ids": sorted({str(row.get("student_id", "")) for row in locate_boxes if row.get("student_id")}),
                "locate_detection_indices": sorted(locate_indices) if locate_raw_rows else sorted({int(row["locate_detection_index"]) for row in locate_boxes if isinstance(row.get("locate_detection_index"), int)})}})

    s3_paths = ([Path(str(row["annotated_path"])) for row in associated_rows if row.get("annotated_path")]
                if associated_rows else sorted((root / "artifacts/locate_annotated").rglob("*.jpg"))
                if "S3" in writable_stage_ids and (root / "artifacts/locate_annotated").is_dir() else screenshot_paths)
    s3_contact = _preview_contact_sheet(root, "S3", s3_paths, label_prefix="ID ") if "S3" in writable_stage_ids else None
    if track_rows or associated_rows:
        unverified = sum(detection.get("student_confirmed") is False
            for row in associated_rows for detection in row.get("detections", ()))
        payloads["S3"].update({"preview_status": "available", "reason": None,
            "uris": [s3_contact] if s3_contact else [], "source_artifact": associated_uri,
            "counts": {"track_observation_rows": len(track_rows), "unverified_person_boxes": unverified,
                "student_ids": sorted({str(row.get("student_id") or row.get("track_id", "")) for row in track_rows
                    if row.get("student_confirmed") is not False and row.get("metadata", {}).get("student_confirmed") is not False})}})

    crop_paths: list[Path] = []
    for row in crop_queue:
        if row.get("composite_path"):
            crop_paths.append(Path(str(row["composite_path"])))
    for row in evidence_rows:
        composite = row.get("qwen_composite")
        if isinstance(composite, Mapping) and composite.get("path"):
            crop_paths.append(Path(str(composite["path"])))
    s4_contact = _preview_contact_sheet(root, "S4", crop_paths, label_prefix="crop ") if "S4" in writable_stage_ids else None
    if crop_queue or evidence_rows:
        payloads["S4"].update({"preview_status": "available", "reason": None, "uris": [s4_contact] if s4_contact else [], "counts": {"qwen_queue_rows": len(crop_queue), "evidence_packets": len(evidence_rows), "student_ids": sorted({str(row.get("student_id", "")) for row in crop_queue if row.get("student_id")})}})

    s5_contact = _preview_contact_sheet(root, "S5", crop_paths, label_prefix="action ") if "S5" in writable_stage_ids else None
    if reviews or action_rows:
        manifest = {}
        manifest_path = root / "run_manifest.v1.json"
        if manifest_path.is_file():
            try:
                loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = loaded_manifest if isinstance(loaded_manifest, Mapping) else {}
            except (OSError, ValueError, TypeError):
                manifest = {}
        profile = manifest.get("profile") if isinstance(manifest.get("profile"), Mapping) else {}
        profile_actions = profile.get("actions") if isinstance(profile.get("actions"), list) else []
        action_lookup: dict[str, dict[str, str]] = {}
        for action in profile_actions:
            if not isinstance(action, Mapping):
                continue
            action_code = str(action.get("code") or action.get("action_id") or "UNKNOWN")
            item = {
                "action_name_zh": str(action.get("name_zh") or action.get("action_name_zh") or action.get("action_id") or "证据不足"),
                "action_name_en": _english_action_name(action_code, action.get("name_en") or action.get("action_name_en")),
                "color": str(action.get("color") or GRAY),
            }
            for key in (action.get("action_id"), action.get("id"), action.get("code")):
                if key:
                    action_lookup[str(key)] = item
        candidate_by_event = {
            str(row.get("event_id") or row.get("candidate_event_id")): row
            for row in selected_candidates
        }
        normalized_rows: list[dict[str, Any]] = []
        for row in action_rows + reviews:
            action_value = str(row.get("action_id") or row.get("primary_action_id") or row.get("behavior") or UNKNOWN_ID)
            action = action_lookup.get(action_value, {"action_name_zh": "证据不足", "action_name_en": "UNKNOWN", "color": GRAY})
            event_id = str(row.get("candidate_event_id") or row.get("event_id") or "")
            candidate = candidate_by_event.get(event_id, {})
            normalized_rows.append({
                **dict(row),
                "student_id": str(row.get("student_id") or row.get("track_id") or candidate.get("track_id") or ""),
                "action_name_zh": action["action_name_zh"],
                "action_name_en": action["action_name_en"],
                "display_label": action["action_name_en"],
                "color": action["color"],
            })
        action_labels = sorted({str(row["action_name_en"]) for row in normalized_rows})
        action_colors = {str(row["action_name_en"]): str(row["color"]) for row in normalized_rows}
        partial_timeline = [
            {
                "student_id": str(row.get("student_id") or row.get("track_id") or ""),
                "pts_seconds": row.get("pts_seconds", row.get("trigger_pts_seconds")),
                "action_name_zh": str(row.get("action_name_zh") or row.get("behavior") or "证据不足"),
                "action_name_en": str(row.get("action_name_en") or row.get("action_code") or "UNKNOWN"),
                "display_label": str(row.get("display_label") or row.get("action_name_en") or row.get("action_code") or "UNKNOWN"),
                "color": str(row.get("color", GRAY)),
                "status": str(row.get("status") or row.get("observability") or "observed"),
            }
            for row in normalized_rows[:200]
        ]
        payloads["S5"].update({"preview_status": "available", "reason": None, "uris": [s5_contact] if s5_contact else [], "counts": {"classification_reviews": len(reviews), "action_observations": len(action_rows), "unknown_observations": sum(str(row.get("action_id") or row.get("primary_action_id")) in {"unknown", "UNKNOWN", "None"} for row in action_rows)}, "action_labels": action_labels, "action_colors": action_colors, "partial_timeline": partial_timeline})

    timeline_path = root / "timeline.v2.json"
    s6_image = _timeline_preview(root, intervals, duration) if "S6" in writable_stage_ids else None
    if "S6" in writable_stage_ids and (timeline_path.is_file() or intervals):
        raw_events = _read_rows(root / "events.jsonl")
        payloads["S6"].update({"preview_status": "available", "reason": None, "uris": [uri for uri in (s6_image, "timeline.v2.json" if timeline_path.is_file() else None) if uri], "counts": {"merged_intervals": len(intervals), "raw_anchor_rows": len(raw_events), "gray_intervals": sum(str(row.get("action_id")) == UNKNOWN_ID for row in intervals)}, "raw_anchor_preview": [{"event_id": row.get("event_id"), "student_id": row.get("track_id") or row.get("student_id"), "start_pts_seconds": row.get("start_pts_seconds", row.get("trigger_pts_seconds")), "end_pts_seconds": row.get("end_pts_seconds")} for row in raw_events[:200]], "merged_interval_preview": [dict(row) for row in intervals[:200]]})

    final_video = root / "videos/annotated_final.mp4"
    report_path = root / "report.html"
    frontend_path = root / "frontend/index.v1.json"
    if final_video.is_file() or timeline_path.is_file() or report_path.is_file():
        payloads["S7"].update({"preview_status": "available", "reason": None, "uris": [uri for uri, path in (("videos/annotated_final.mp4", final_video), ("timeline.v2.json", timeline_path), ("report.html", report_path), ("frontend/index.v1.json", frontend_path)) if path.is_file() or uri == "frontend/index.v1.json"], "counts": {"final_video": final_video.is_file(), "timeline": timeline_path.is_file(), "report": report_path.is_file(), "frontend_index": True}})

    preview_uris: list[str] = []
    for stage_id, payload in payloads.items():
        if stage_id not in selected_stage_ids:
            continue
        directory = root / "previews" / stage_id
        directory.mkdir(parents=True, exist_ok=True)
        existing_preview = directory / "preview.json"
        # Earlier receipts are finalized before S7 runs. Rewriting any
        # stage-owned preview here would invalidate that upstream hash, so S7
        # only indexes existing S0-S6 previews and creates its own S7 entry.
        if preserve_existing_stage_previews and existing_preview.is_file():
            preview_uris.append(f"previews/{stage_id}/preview.json")
            try:
                existing_payload = json.loads(existing_preview.read_text(encoding="utf-8"))
                preview_uris.extend(str(uri) for uri in existing_payload.get("uris", ()) if str(uri).startswith("previews/"))
            except (OSError, ValueError, TypeError):
                pass
            continue
        atomic_write_json(directory / "preview.json", payload)
        preview_uris.append(f"previews/{stage_id}/preview.json")
        preview_uris.extend(str(uri) for uri in payload["uris"] if str(uri).startswith("previews/"))
    return preview_uris


def write_stage_preview(
    run_dir: str | Path,
    stage_id: str,
    *,
    info: Mapping[str, Any],
    preserve_existing_stage_previews: bool = False,
) -> list[str]:
    """Persist the semantic preview owned by one completed stage.

    This narrow wrapper lets the stage orchestrator seal the preview in the
    same receipt transaction without exposing the all-stage implementation.
    """

    if stage_id not in {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}:
        raise ValueError(f"unknown_preview_stage:{stage_id}")
    return _write_stage_previews(
        Path(run_dir).resolve(),
        info,
        stage_ids=(stage_id,),
        preserve_existing_stage_previews=preserve_existing_stage_previews,
    )


def _bgr(hex_color: str) -> tuple[int, int, int]:
    value = str(hex_color).lstrip("#")
    try:
        red, green, blue = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
        return blue, green, red
    except Exception:
        return (184, 184, 184)


def _student_bgr(student_id: str) -> tuple[int, int, int]:
    seed = sum((index + 1) * ord(char) for index, char in enumerate(str(student_id)))
    return (64 + (47 * seed) % 192, 64 + (89 * seed) % 192, 64 + (137 * seed) % 192)


def _compact_student_id(value: Any) -> str:
    """Render a stable, legible ASCII ID without crowding dense classrooms."""

    student_id = str(value or "").strip()
    if student_id.startswith("student-"):
        suffix = student_id.removeprefix("student-")
        if suffix.isdigit():
            return f"S{int(suffix):03d}"
    return student_id[:12] or "S?"


def _display_action_label(row: Mapping[str, Any]) -> str:
    """Return the ASCII action label used by every rendered surface."""

    value = str(
        row.get("display_label")
        or row.get("action_name_en")
        or row.get("action_code")
        or "UNKNOWN"
    ).strip()
    mapped = _english_action_name(row.get("action_code"), value)
    # The video writer is intentionally ASCII-only.  JSON retains canonical
    # ``action_name_zh`` separately for taxonomy compatibility, while every
    # user-facing display label is English and portable across OpenCV builds.
    return mapped.encode("ascii", "ignore").decode("ascii").strip() or "UNKNOWN"


def _draw_overlay(image: Any, row: Mapping[str, Any], *, semantic: str) -> None:
    bbox = row.get("bbox_px")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return
    try:
        age = float(row.get("bbox_age_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        age = MAX_BOX_HOLD_SECONDS + 1.0
    if not bool(row.get("bbox_observed", True)) and age > MAX_BOX_HOLD_SECONDS + 1e-9:
        return
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    height, width = image.shape[:2]
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    status = str(row.get("interval_status", "unobserved"))
    if semantic == "tracking_preview":
        color = _student_bgr(str(row.get("student_id", ""))) if status != "unobserved" else _bgr(GRAY)
        thickness = 2 if status == "confirmed" else 1
        label = _compact_student_id(row.get("student_id"))
        provider = row.get("detector_source")
        if provider and provider != "locate_anything":
            detection_index = row.get("detection_index")
            if detection_index is not None:
                label += f" {'Y' if provider == 'yolo' else 'D'}{detection_index}"
        elif row.get("locate_detection_index") is not None:
            label += f" L{row['locate_detection_index']}"
        elif provider == "locate_anything" and row.get("detection_index") is not None:
            label += f" L{row['detection_index']}"
    else:
        color = _bgr(str(row.get("color", GRAY))) if status != "unobserved" else _bgr(GRAY)
        thickness = 2 if status == "confirmed" else 1
        label = None
        if status != "unobserved":
            label = _display_action_label(row)
        if semantic == "annotated_final":
            compact_id = _compact_student_id(row.get("student_id"))
            label = f"{compact_id} {label}" if label else compact_id
    if status == "propagated" and semantic != "tracking_preview":
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, PROPAGATED_ALPHA, image, 1.0 - PROPAGATED_ALPHA, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (max(0, x1), max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


class _OverlayCursor:
    """Consume a frame-sorted overlay iterator without a by-frame dictionary."""

    def __init__(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self._rows = iter(rows)
        self._pending: Mapping[str, Any] | None = None
        self._advance()

    def _advance(self) -> None:
        try:
            self._pending = next(self._rows)
        except StopIteration:
            self._pending = None

    def take(self, frame_index: int) -> list[Mapping[str, Any]]:
        while self._pending is not None and int(self._pending.get("frame_index", -1)) < frame_index:
            self._advance()
        values: list[Mapping[str, Any]] = []
        while self._pending is not None and int(self._pending.get("frame_index", -1)) == frame_index:
            values.append(self._pending)
            self._advance()
        return values


def _requested_preview_ranges(duration: float) -> list[tuple[str, float, float]]:
    duration = max(0.0, duration)
    if duration <= 0.0:
        return [("first", 0.0, 0.0)]
    raw = [
        ("first", 0.0, min(duration, 10.0)),
        ("middle", max(0.0, duration / 2.0 - 5.0), min(duration, duration / 2.0 + 5.0)),
        ("last", max(0.0, duration - 10.0), duration),
    ]
    return [(label, max(0.0, start), min(duration, max(start, end))) for label, start, end in raw if end > start]


def _merge_preview_ranges(requested: Sequence[tuple[str, float, float]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for label, start, end in sorted(requested, key=lambda item: (item[1], item[2], item[0])):
        if not merged or start > float(merged[-1]["end_pts_seconds"]) + 1e-9:
            merged.append({"start_pts_seconds": float(start), "end_pts_seconds": float(end), "requested_segments": [label]})
        else:
            merged[-1]["end_pts_seconds"] = max(float(merged[-1]["end_pts_seconds"]), float(end))
            merged[-1]["requested_segments"].append(label)
    return merged


def _frame_ranges(clip_ranges: Sequence[Mapping[str, Any]], fps: float, frame_count: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in clip_ranges:
        start = max(0, int(math.floor(float(item["start_pts_seconds"]) * fps + 1e-9)))
        end = min(frame_count, int(math.ceil(float(item["end_pts_seconds"]) * fps - 1e-9)))
        if end > start:
            result.append((start, end))
    return result


def _clip_contains(frame_index: int, ranges: Sequence[tuple[int, int]], cursor: int) -> tuple[bool, int]:
    while cursor < len(ranges) and frame_index >= ranges[cursor][1]:
        cursor += 1
    return bool(cursor < len(ranges) and ranges[cursor][0] <= frame_index < ranges[cursor][1]), cursor


def _ffmpeg_executable(configured: str | Path | None) -> str | None:
    if configured is not None:
        path = Path(configured).expanduser()
        return str(path.resolve()) if path.is_file() else None
    return shutil.which("ffmpeg")


class _NvencWriter:
    """Pipe bounded raw BGR frames to FFmpeg and check the completed stream."""

    def __init__(self, path: Path, fps: float, width: int, height: int, ffmpeg: str) -> None:
        self.path, self.fps = path, fps
        self.width, self.height = width, height
        self.frames = 0
        self.first: np.ndarray | None = None
        self.last: np.ndarray | None = None
        self.validation: dict[str, Any] = {}
        self._stderr = tempfile.TemporaryFile()
        self._closed = False
        command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except BaseException:
            self._stderr.close()
            raise

    def isOpened(self) -> bool:
        return not self._closed and self._process.poll() is None

    def write(self, image: np.ndarray) -> None:
        if self._process.stdin is None or self._closed:
            raise RuntimeError("nvenc_writer_closed")
        started = time.perf_counter()
        try:
            self._process.stdin.write(np.ascontiguousarray(image).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("nvenc_encode_pipe_failed") from exc
        finally:
            record_timing("video.encode_seconds", time.perf_counter() - started)
        if self.first is None:
            self.first = image.copy()
        self.last = image.copy()
        self.frames += 1

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        started = time.perf_counter()
        try:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
            try:
                code = self._process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
                raise RuntimeError("nvenc_finalize_timeout")
            if code != 0:
                self._stderr.seek(0)
                reason = self._stderr.read().decode("utf-8", errors="replace")[-1000:]
                raise RuntimeError(f"nvenc_encode_failed:{code}:{reason}")
            self.validation = self._validate()
        finally:
            self._stderr.close()
            record_timing("video.encode_finalize_seconds", time.perf_counter() - started)

    def _validate(self) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened() or self.frames < 1:
                raise RuntimeError("nvenc_output_unreadable")
            actual_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if actual_count != self.frames or abs(actual_fps - self.fps) > max(1e-3, self.fps * 1e-4):
                raise RuntimeError("nvenc_output_timing_mismatch")
            psnrs: list[float] = []
            for index, reference in ((0, self.first), (self.frames - 1, self.last)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = capture.read()
                if not ok or reference is None or image.shape != reference.shape:
                    raise RuntimeError("nvenc_output_endpoint_unreadable")
                psnr = float(cv2.PSNR(reference, image))
                if psnr < 28.0:
                    raise RuntimeError(f"nvenc_output_quality_below_threshold:{psnr}")
                psnrs.append(psnr)
            return {"status": "passed", "frames": actual_count, "fps": actual_fps, "duration_seconds": actual_count / actual_fps, "endpoint_psnr_db": psnrs, "minimum_endpoint_psnr_db": 28.0}
        finally:
            capture.release()


def _probe_nvenc(ffmpeg: str | None, directory: Path) -> dict[str, Any]:
    if ffmpeg is None:
        return {"status": "unavailable", "reason": "ffmpeg_not_found"}
    try:
        with tempfile.TemporaryDirectory(prefix="nvenc-probe-", dir=directory) as temporary:
            writer = _NvencWriter(Path(temporary) / "probe.mp4", 10.0, 64, 64, ffmpeg)
            try:
                for value in (0, 80, 160):
                    writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
            finally:
                writer.release()
            return {"status": "passed", "ffmpeg": ffmpeg, "validation": writer.validation}
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}:{exc}"[-1000:]}


def _render_videos_once(
    source: Path,
    output_paths: Mapping[str, Path],
    overlays: Iterable[Mapping[str, Any]],
    *,
    info: Mapping[str, Any],
    progress_callback: Callable[[int, int], None] | None = None,
    encoder_backend: str = "opencv",
    ffmpeg_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    if encoder_backend not in {"opencv", "nvenc"}:
        raise ValueError("video_encoder_backend_invalid")
    if encoder_backend == "opencv":
        return _render_videos_once_impl(source, output_paths, overlays, info=info, progress_callback=progress_callback)
    directory = next(iter(output_paths.values())).parent
    directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_executable(ffmpeg_path)
    probe = _probe_nvenc(ffmpeg, directory)
    fallback_reason: str | None = None
    decode_passes = 1
    if probe["status"] != "passed":
        fallback_reason = str(probe.get("reason", "nvenc_probe_failed"))
        result = _render_videos_once_impl(source, output_paths, overlays, info=info, progress_callback=progress_callback)
    else:
        # Spooling rows permits a clean retry after an encoder failure without
        # holding the complete overlay stream or any raw video in memory.
        with tempfile.TemporaryDirectory(prefix="encoder-retry-", dir=directory) as temporary:
            spool = Path(temporary) / "overlays.jsonl"
            atomic_write_jsonl(spool, overlays)
            try:
                result = _render_videos_once_impl(source, output_paths, iter_jsonl(spool), info=info, progress_callback=progress_callback, ffmpeg=ffmpeg)
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                fallback_reason = f"{type(exc).__name__}:{exc}"[-1000:]
                decode_passes = 2
                result = _render_videos_once_impl(source, output_paths, iter_jsonl(spool), info=info, progress_callback=progress_callback)
    for receipt in result.values():
        receipt["encoder"] = {"requested": encoder_backend, "effective": "opencv" if fallback_reason else "nvenc", "probe": probe, "fallback_reason": fallback_reason, "source_decode_passes": decode_passes}
    return result


def _render_videos_once_impl(
    source: Path,
    output_paths: Mapping[str, Path],
    overlays: Iterable[Mapping[str, Any]],
    *,
    info: Mapping[str, Any],
    progress_callback: Callable[[int, int], None] | None = None,
    ffmpeg: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Decode ``source`` once and feed all three semantic writers."""

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"video_open_failed:{source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or info.get("fps", 25.0) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or info.get("width", 0) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or info.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"video_dimensions_invalid:{source}")
    hint_count = int(info.get("frame_count", 0) or 0)
    hint_duration = float(info.get("duration_seconds", 0.0) or 0.0)
    duration = max(hint_duration, hint_count / fps if hint_count > 0 else 0.0)
    requested = _requested_preview_ranges(duration)
    merged = _merge_preview_ranges(requested)
    # The frame count is only a hint until decode completes.  A generous hint
    # still lets the range cursor work for ordinary files; final receipts use
    # actual frame counts.
    range_count = max(hint_count, int(math.ceil(duration * fps)), 1)
    preview_ranges = _frame_ranges(merged, fps, range_count)
    output_paths = {str(key): Path(value) for key, value in output_paths.items()}
    progress_total = max(1, hint_count)
    if progress_callback is not None:
        progress_callback(0, progress_total)
    writers: dict[str, Any] = {}
    generic_tracking_sources: set[str] = set()

    def write_frame(semantic: str, image: np.ndarray) -> None:
        began = time.perf_counter()
        writers[semantic].write(image)
        if ffmpeg is None:
            record_timing("video.encode_seconds", time.perf_counter() - began)
        increment("encoded_frames")

    try:
        for semantic, path in output_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = _NvencWriter(path, fps, width, height, ffmpeg) if ffmpeg else cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            writers[semantic] = writer
            if not writer.isOpened():
                raise RuntimeError(f"video_writer_open_failed:{path}")
        cursor = _OverlayCursor(overlays)
        frame_index = 0
        preview_cursor = 0
        preview_frames = 0
        final_frames = 0
        while True:
            decode_started = time.perf_counter()
            ok, image = capture.read()
            record_timing("video.decode_seconds", time.perf_counter() - decode_started)
            if not ok:
                break
            increment("decoded_frames")
            rows = cursor.take(frame_index)
            generic_tracking_sources.update(str(row["detector_source"]) for row in rows if row.get("detector_source"))
            if "annotated_final" in writers:
                rendered = image.copy()
                for row in rows:
                    _draw_overlay(rendered, row, semantic="annotated_final")
                write_frame("annotated_final", rendered)
                final_frames += 1
            selected, preview_cursor = _clip_contains(frame_index, preview_ranges, preview_cursor)
            if selected:
                if "tracking_preview" in writers:
                    rendered = image.copy()
                    for row in rows:
                        _draw_overlay(rendered, row, semantic="tracking_preview")
                    write_frame("tracking_preview", rendered)
                if "action_preview" in writers:
                    rendered = image.copy()
                    for row in rows:
                        _draw_overlay(rendered, row, semantic="action_preview")
                    write_frame("action_preview", rendered)
                preview_frames += 1
            frame_index += 1
            if progress_callback is not None:
                progress_callback(frame_index, progress_total)
        actual_frames = frame_index
    finally:
        close_error: Exception | None = None
        for writer in writers.values():
            try:
                writer.release()
            except Exception as exc:
                close_error = close_error or exc
        capture.release()
        if close_error is not None:
            raise close_error
    if progress_callback is not None:
        progress_callback(actual_frames, max(1, actual_frames))

    actual_ranges = _frame_ranges(merged, fps, actual_frames)
    actual_clip_receipts = [
        {
            "start_pts_seconds": round(start / fps, 6),
            "end_pts_seconds": round(end / fps, 6),
            "start_frame": start,
            "end_frame_exclusive": end,
            "requested_segments": [label for label, requested_start, requested_end in requested if requested_start < end / fps and requested_end > start / fps],
        }
        for start, end in actual_ranges
    ]
    requested_clip_receipts = [
        {
            "segment": label,
            "start_pts_seconds": float(start),
            "end_pts_seconds": float(end),
        }
        for label, start, end in requested
    ]
    receipts: dict[str, dict[str, Any]] = {}
    semantics = {
        "tracking_preview": {
            "kind": "tracking_preview",
            "description": "Up to 10 seconds from the beginning, middle, and end; shows student IDs, track IDs, and Locate indices only.",
            "labels": ["student_id", "locate_detection_index"],
            "clip_policy": "first_middle_last_10_seconds_deduplicated",
        },
        "action_preview": {
            "kind": "action_preview",
            "description": "Up to 10 seconds from the beginning, middle, and end; shows action colors and English action labels without track IDs.",
            "labels": ["action_name_en", "action_color", "interval_status"],
            "clip_policy": "first_middle_last_10_seconds_deduplicated",
        },
        "annotated_final": {
            "kind": "annotated_final",
            "description": "Full-video action annotation; shows student IDs, action colors, English action labels, and evidence status.",
            "labels": ["student_id", "action_name_en", "action_color", "interval_status"],
            "clip_policy": "full_video",
        },
    }
    if generic_tracking_sources or info.get("localization_route") in {"yolo_only", "yolo_locate_hybrid"}:
        semantics["tracking_preview"].update({
            "description": "Up to 10 seconds from the beginning, middle, and end; shows student IDs and actual detector indices (L=Locate, Y=YOLO, D=other).",
            "labels": ["student_id", "detector_source", "detection_index"],
            "detector_sources": sorted(generic_tracking_sources),
        })
    for semantic, path in output_paths.items():
        frame_total = final_frames if semantic == "annotated_final" else preview_frames
        receipts[semantic] = {
            "path": str(path),
            "uri": f"videos/{path.name}",
            "frames": frame_total,
            "fps": fps,
            "width": width,
            "height": height,
            "clip_ranges": [] if semantic == "annotated_final" else actual_clip_receipts,
            "requested_clip_ranges": [] if semantic == "annotated_final" else requested_clip_receipts,
            "frame_ranges": [] if semantic == "annotated_final" else [{"start_frame": start, "end_frame_exclusive": end} for start, end in actual_ranges],
            "semantic": semantics[semantic],
            "semantic_id": semantic,
            "audio_preserved": False,
            "audio_status": "pending",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "encoding_validation": getattr(writers[semantic], "validation", {"status": "legacy_opencv_backend"}),
        }
    if "annotated_final" in receipts:
        receipts["annotated_final"]["clip_ranges"] = [{"start_pts_seconds": 0.0, "end_pts_seconds": duration, "start_frame": 0, "end_frame_exclusive": actual_frames, "requested_segments": ["full_video"]}]
        receipts["annotated_final"]["requested_clip_ranges"] = receipts["annotated_final"]["clip_ranges"]
        receipts["annotated_final"]["frame_ranges"] = [{"start_frame": 0, "end_frame_exclusive": actual_frames}]
    return receipts


def render_video(source: Path, output: Path, overlays: Mapping[int, Sequence[Mapping[str, Any]]], *, max_frames: int | None = None, alpha_propagated: float = PROPAGATED_ALPHA) -> dict[str, Any]:
    """Render one annotated video for legacy callers.

    ``alpha_propagated`` remains accepted for compatibility; the frozen S7
    contract always uses :data:`PROPAGATED_ALPHA`.
    """

    del alpha_propagated
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"video_open_failed:{source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"video_writer_open_failed:{output}")
    frame = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok or (max_frames is not None and frame >= max_frames):
                break
            for row in overlays.get(frame, ()):
                _draw_overlay(image, row, semantic="annotated_final")
            writer.write(image)
            frame += 1
    finally:
        writer.release(); capture.release()
    return {"path": str(output), "uri": f"videos/{output.name}", "frames": frame, "fps": fps, "sha256": sha256_file(output), "bytes": output.stat().st_size, "clip_ranges": [{"start_pts_seconds": 0.0, "end_pts_seconds": frame / fps, "start_frame": 0, "end_frame_exclusive": frame, "requested_segments": ["full_video"]}], "frame_ranges": [{"start_frame": 0, "end_frame_exclusive": frame}], "semantic": {"kind": "annotated_final", "labels": ["student_id", "action_name_en", "action_color", "interval_status"]}, "semantic_id": "annotated_final"}


def _video_frame_timing(path: Path) -> tuple[int, float]:
    """Return the decoded frame count and fps used to compare muxed outputs."""

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0, 0.0
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


def _mux_audio_pyav(source: Path, rendered: Path, temporary: Path) -> dict[str, Any]:
    """Mux audio without FFmpeg: remux the video packets and encode AAC audio.

    Only called when FFmpeg is absent.  The rendered video stream is copied
    packet-for-packet (no second encode); source audio is decoded, encoded to
    AAC, truncated to the rendered duration, and written next to the video.
    """

    try:
        import av
    except ImportError as exc:
        reason = f"pyav_unavailable:{type(exc).__name__}"
        return {"status": "unavailable", "preserved": False, "muxer": "pyav", "reason": reason, "warning": f"audio_not_preserved:{reason}"}
    temporary.unlink(missing_ok=True)
    source_container = rendered_container = output_container = None
    try:
        source_container = av.open(str(source))
        if not source_container.streams.audio:
            # No track to mux is a fact about the source, not a tool failure.
            return {"status": "unavailable", "preserved": False, "muxer": "pyav", "reason": "source_has_no_audio_stream", "warning": "audio_not_preserved:source_has_no_audio_stream"}
        rendered_frames, rendered_fps = _video_frame_timing(rendered)
        duration_limit = rendered_frames / rendered_fps if rendered_frames > 0 and rendered_fps > 0 else None
        rendered_container = av.open(str(rendered))
        output_container = av.open(str(temporary), mode="w", format="mp4")
        video_in = rendered_container.streams.video[0]
        video_out = output_container.add_stream_from_template(video_in)
        audio_in = source_container.streams.audio[0]
        rate = int(audio_in.rate or 48000)
        layout = "mono" if str(getattr(audio_in.layout, "name", "")) in {"mono"} else "stereo"
        audio_out = output_container.add_stream("aac", rate=rate)
        audio_out.layout = layout
        resampler = av.AudioResampler(format=audio_out.codec_context.format.name, layout=layout, rate=rate)
        for packet in rendered_container.demux(video_in):
            if packet.dts is None:
                continue
            packet.stream = video_out
            output_container.mux(packet)
        # Windows refuses os.replace while the input handles are open.
        rendered_container.close()
        rendered_container = None
        for frame in source_container.decode(audio_in):
            if duration_limit is not None and frame.pts is not None and frame.time_base is not None and float(frame.pts * frame.time_base) >= duration_limit:
                break
            # Rebase the source audio at zero so it starts with the rendered video.
            frame.pts = None
            for converted in resampler.resample(frame):
                _mux_encoded_audio(output_container, audio_out, converted, duration_limit)
        for converted in resampler.resample(None):
            _mux_encoded_audio(output_container, audio_out, converted, duration_limit)
        for packet in audio_out.encode():
            if duration_limit is None or packet.pts is None or float(packet.pts * audio_out.time_base) < duration_limit:
                output_container.mux(packet)
        source_container.close()
        source_container = None
        output_container.close()
        output_container = None
        with av.open(str(temporary)) as probe:
            has_video = bool(probe.streams.video)
            has_audio = bool(probe.streams.audio)
        if not (has_video and has_audio):
            raise OSError("pyav_mux_required_stream_missing")
        muxed_frames, muxed_fps = _video_frame_timing(temporary)
        if muxed_frames != rendered_frames or abs(muxed_fps - rendered_fps) > 1e-3:
            raise OSError("pyav_mux_changed_video_timing")
        os.replace(temporary, rendered)
        return {
            "status": "preserved", "preserved": True, "muxer": "pyav", "reason": "pyav_mux_ok",
            "audio_codec": "aac",
            "validation": {
                "video_stream": "copied_from_rendered_without_reencode",
                "audio_stream": "decoded_from_source_and_encoded_as_aac",
                "required_streams_present": "passed",
                "video_frame_count_and_fps": "passed",
                "audio_truncated_to_video_seconds": duration_limit,
            },
        }
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        reason = f"{type(exc).__name__}:{exc}"
        return {"status": "failed", "preserved": False, "muxer": "pyav", "reason": reason[:500], "warning": f"audio_not_preserved:{reason[:500]}"}
    finally:
        for container in (output_container, rendered_container, source_container):
            if container is not None:
                container.close()


def _mux_encoded_audio(output_container: Any, audio_out: Any, frame: Any, duration_limit: float | None) -> None:
    for packet in audio_out.encode(frame):
        if duration_limit is not None and packet.pts is not None and float(packet.pts * audio_out.time_base) >= duration_limit:
            continue
        output_container.mux(packet)


def _preserve_audio(source: Path, rendered: Path, *, ffmpeg_path: str | Path | None = None, validate_streams: bool = False) -> dict[str, Any]:
    """Best-effort audio mux with an explicit reason for every non-success."""

    ffmpeg = _ffmpeg_executable(ffmpeg_path)
    temporary = rendered.with_name(rendered.stem + ".audio.mp4")
    if ffmpeg is None:
        # PyAV (libav bindings) covers the common environment where the FFmpeg
        # CLI is not on PATH; it always validates streams and frame timing.
        return _mux_audio_pyav(source, rendered, temporary)
    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(rendered), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(temporary)]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, timeout=180)
        if completed.returncode != 0 or not temporary.is_file():
            reason = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1] if completed.stderr else "ffmpeg_mux_failed"
            temporary.unlink(missing_ok=True)
            return {"status": "failed", "preserved": False, "reason": reason[:500], "warning": f"audio_not_preserved:{reason[:500]}"}
        validation: dict[str, Any] = {}
        if validate_streams:
            metadata = []
            for path in (rendered, temporary):
                capture = cv2.VideoCapture(str(path))
                try:
                    if not capture.isOpened():
                        raise OSError("audio_mux_video_unreadable")
                    metadata.append((int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))))
                finally:
                    capture.release()
            if metadata[0][0] != metadata[1][0] or abs(metadata[0][1] - metadata[1][1]) > 1e-3:
                raise OSError("audio_mux_changed_video_timing")
            validation = {"video_frame_count_and_fps": "passed", "audio_stream_mapping": "ffmpeg_required_source_audio_stream", "av_start_time_check": "not_measured_ffprobe_unavailable"}
            sibling = Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
            ffprobe = str(sibling) if sibling.is_file() else shutil.which("ffprobe")
            if ffprobe:
                probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "stream=codec_type,start_time,duration", "-of", "json", str(temporary)], capture_output=True, timeout=30, check=False)
                if probe.returncode != 0:
                    raise OSError("audio_mux_stream_probe_failed")
                streams = json.loads(probe.stdout).get("streams", [])
                video_stream = next((row for row in streams if row.get("codec_type") == "video"), None)
                audio_stream = next((row for row in streams if row.get("codec_type") == "audio"), None)
                if video_stream is None or audio_stream is None:
                    raise OSError("audio_mux_required_stream_missing")
                if video_stream.get("start_time") is not None and audio_stream.get("start_time") is not None:
                    offset = abs(float(video_stream["start_time"]) - float(audio_stream["start_time"]))
                    if offset > 0.1:
                        raise OSError(f"audio_mux_start_offset_exceeded:{offset}")
                    validation.update({"av_start_time_check": "passed", "av_start_offset_seconds": offset})
        os.replace(temporary, rendered)
        return {"status": "preserved", "preserved": True, "reason": "ffmpeg_mux_ok", **({"validation": validation} if validate_streams else {})}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        temporary.unlink(missing_ok=True)
        reason = f"{type(exc).__name__}:{exc}"
        return {"status": "failed", "preserved": False, "reason": reason[:500], "warning": f"audio_not_preserved:{reason[:500]}"}


def _spool_overlay_rows(root: Path, info: Mapping[str, Any], intervals: Sequence[Mapping[str, Any]]) -> tuple[Path, int]:
    """Persist a sorted overlay stream without retaining it in RAM.

    S7 consumes this spool twice: once for 60-second chunk files and once for
    the single source-video decode.  The source is never decoded for the
    spool/chunk pass, and the spool remains auditable as a JSONL artifact.
    """

    path = root / "artifacts" / "overlay_rows.v3.jsonl"
    count = 0
    with JsonlStreamWriter(path) as writer:
        for row in iter_overlay_rows(root, info, intervals):
            writer.write(row)
            count += 1
    return path, count


def _metrics_not_evaluated(intervals: Sequence[Mapping[str, Any]], overlay_row_count: int, video_results: Mapping[str, Mapping[str, Any]], warnings: Sequence[str], *, system_metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Emit auditable run facts without fake evaluation placeholders."""

    unknown_intervals = [row for row in intervals if str(row.get("action_id")) == UNKNOWN_ID or str(row.get("status")) == "unobserved"]
    unknown_seconds = sum(max(0.0, float(row.get("end_pts_seconds", 0.0)) - float(row.get("start_pts_seconds", 0.0))) for row in unknown_intervals)
    total_seconds = sum(max(0.0, float(row.get("end_pts_seconds", 0.0)) - float(row.get("start_pts_seconds", 0.0))) for row in intervals)
    return {
        "schema_version": "classroom_hybrid_ai.metrics.v1",
        "evaluation_status": "not_evaluated",
        "reason": "gold_annotations_not_provided",
        "audit": {
            "interval_count": len(intervals),
            "unknown_interval_count": len(unknown_intervals),
            "unknown_interval_seconds": round(unknown_seconds, 6),
            "covered_interval_seconds": round(max(0.0, total_seconds - unknown_seconds), 6),
            "overlay_row_count": int(overlay_row_count),
            "propagated_alpha": PROPAGATED_ALPHA,
            "maximum_bbox_hold_seconds": MAX_BOX_HOLD_SECONDS,
            "video_decode_passes": max((int(receipt.get("encoder", {}).get("source_decode_passes", 1)) for receipt in video_results.values()), default=1),
            "audio_warning_count": len(warnings),
        },
        "system": {
            **dict(system_metrics or {}),
            "rendered_video_frames": {name: int(receipt.get("frames", 0)) for name, receipt in video_results.items()},
            "rendered_video_count": len(video_results),
        },
    }


def _english_display_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public ASCII JSON form without leaking CJK-only labels."""

    action_code = str(row.get("action_code") or row.get("action_id") or "UNKNOWN")
    label = _english_action_name(
        action_code,
        row.get("display_label") or row.get("action_name_en"),
    )
    result = {
        key: value
        for key, value in row.items()
        if key not in {"action_name_zh", "name_zh"}
    }
    result["action_name_en"] = label
    result["display_label"] = label
    return result


def _write_english_display_json(
    root: Path,
    *,
    timeline: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    profile: ActionProfile,
) -> tuple[str, str, str]:
    """Write the ASCII-only JSON contract consumed by future frontends."""

    frontend = root / "frontend"
    frontend.mkdir(parents=True, exist_ok=True)
    action_uri = "frontend/action_observations_display.v1.jsonl"
    interval_uri = "frontend/action_intervals_display.v1.jsonl"
    timeline_uri = "frontend/timeline_display.v1.json"
    atomic_write_jsonl(frontend / "action_observations_display.v1.jsonl", (
        _english_display_row(row) for row in observations
    ))
    atomic_write_jsonl(frontend / "action_intervals_display.v1.jsonl", (
        _english_display_row(row) for row in intervals
    ))
    actions = [
        {
            "action_id": str(action["action_id"]),
            "action_code": str(action["code"]),
            "action_name_en": _english_action_name(action["code"], action.get("name_en")),
            "display_label": _english_action_name(action["code"], action.get("name_en")),
            "color": str(action["color"]),
        }
        for action in profile.actions
    ]
    raw_video = timeline.get("video", {})
    public_video = {
        key: value
        for key, value in dict(raw_video).items()
        if key != "path"
    } if isinstance(raw_video, Mapping) else {}
    payload = {
        "schema_version": "classroom_hybrid_ai.timeline_display.v1",
        "display_language": "english_ascii",
        # A frontend display artifact must not expose a local source path;
        # this also keeps the portable English JSON contract ASCII-only.
        "video": public_video,
        "student_rows": list(timeline.get("student_rows", ())),
        "start_pts_seconds": timeline.get("start_pts_seconds"),
        "end_pts_seconds": timeline.get("end_pts_seconds"),
        "duration_seconds": timeline.get("duration_seconds"),
        "taxonomy_profile_id": profile.profile_id,
        "taxonomy_contract_sha256": profile.sha256,
        "actions": actions,
        "intervals": [_english_display_row(row) for row in intervals],
        "coverage": dict(timeline.get("coverage", {})),
    }
    atomic_write_json(frontend / "timeline_display.v1.json", payload)
    return action_uri, interval_uri, timeline_uri


_REPORT_STYLE = """
:root { color-scheme: light dark; }
body { margin: 0; padding: 24px 32px 48px; font: 15px/1.55 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; color: #1d2433; background: #f6f7fb; }
header { border-bottom: 3px solid #2f6fed; padding-bottom: 12px; margin-bottom: 20px; }
h1 { margin: 0 0 4px; font-size: 24px; }
h2 { font-size: 17px; margin: 26px 0 8px; padding-left: 8px; border-left: 4px solid #2f6fed; }
.meta { color: #566; font-size: 13px; }
table { width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 1px 2px rgba(20, 30, 60, .08); }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #e6e9f0; vertical-align: top; }
th { background: #eef2fb; font-weight: 600; }
td.key { width: 34%; color: #333c4f; }
td.value { font-variant-numeric: tabular-nums; word-break: break-word; }
.not-recorded { color: #8a8f9c; font-style: italic; }
ul.links { margin: 0; padding-left: 20px; }
ul.links a { color: #1f5fd0; }
footer { margin-top: 30px; color: #7a808d; font-size: 12px; }
"""


def _report_text(value: Any) -> str:
    """Escape any recorded value; missing data is stated, never invented."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return '<span class="not-recorded">未记录</span>'
    return _html_escape(str(value))


def _report_number(value: Any, unit: str = "") -> str:
    try:
        return _html_escape(f"{float(value):.3f}{unit}")
    except (TypeError, ValueError):
        return '<span class="not-recorded">未记录</span>'


def _report_count(value: Any) -> str:
    try:
        return _html_escape(str(int(value)))
    except (TypeError, ValueError):
        return '<span class="not-recorded">未记录</span>'


def _report_lookup(payload: Any, key: str) -> Any:
    """Find the first recorded scalar for ``key`` anywhere in the metrics tree."""

    if isinstance(payload, Mapping):
        for name, value in payload.items():
            if str(name).lower() == key.lower() and value not in (None, ""):
                return value
            found = _report_lookup(value, key)
            if found not in (None, ""):
                return found
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found = _report_lookup(item, key)
            if found not in (None, ""):
                return found
    return None


def _report_row(label: str, value: str) -> str:
    return f'<tr><td class="key">{_html_escape(label)}</td><td class="value">{value}</td></tr>'


def _render_report_html(
    *,
    info: Mapping[str, Any],
    profile: ActionProfile,
    students: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    video_results: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render a self-contained, escaped run report from committed artifacts."""

    audit = metrics.get("audit", {}) if isinstance(metrics.get("audit"), Mapping) else {}
    system = metrics.get("system", {}) if isinstance(metrics.get("system"), Mapping) else {}
    video = dict(info or {})
    annotated = video_results.get("annotated_final", {}) or {}
    source_name = Path(str(video["path"]).strip()).name if str(video.get("path") or "").strip() else None
    duration = video.get("duration_seconds")
    frame_count = video.get("frame_count")
    if frame_count in (None, ""):
        frame_count = annotated.get("frames")
    width, height = video.get("width"), video.get("height")
    resolution = f"{int(width)} × {int(height)}" if isinstance(width, (int, float)) and isinstance(height, (int, float)) else None
    interval_count = audit.get("interval_count", len(intervals))
    unknown_count = audit.get("unknown_interval_count")
    unknown_seconds = audit.get("unknown_interval_seconds")
    covered_seconds = audit.get("covered_interval_seconds")
    if unknown_count in (None, "") or unknown_seconds in (None, "") or covered_seconds in (None, ""):
        unknown_rows = [row for row in intervals if str(row.get("action_id")) == UNKNOWN_ID or str(row.get("status")) == "unobserved"]
        unknown_count = len(unknown_rows)
        unknown_seconds = sum(max(0.0, float(row.get("end_pts_seconds", 0.0)) - float(row.get("start_pts_seconds", 0.0))) for row in unknown_rows)
        covered_seconds = max(0.0, sum(max(0.0, float(row.get("end_pts_seconds", 0.0)) - float(row.get("start_pts_seconds", 0.0))) for row in intervals) - unknown_seconds)
    stage_times = system.get("stage_wall_time_seconds", {})
    stage_times = stage_times if isinstance(stage_times, Mapping) else {}
    stage_rows = "".join(
        _report_row(stage, _report_number(stage_times.get(stage), " s"))
        for stage in (f"S{index}" for index in range(8))
    )
    qwen_calls = system.get("qwen_actual_calls")
    candidate_count = _report_lookup(metrics, "all_candidates")
    selected_count = _report_lookup(metrics, "selected_candidates")
    if selected_count in (None, ""):
        selected_count = _report_lookup(metrics, "fair_selected_candidates")
    vram_rows = "".join((
        _report_row("PyTorch peak reserved (GiB)", _report_number(system.get("peak_reserved_gib"))),
        _report_row("PyTorch peak allocated (GiB)", _report_number(system.get("peak_allocated_gib"))),
        _report_row("Device-wide sampled peak (GiB)", _report_number(system.get("sampled_peak_physical_vram_gib"))),
        _report_row("Recorded generic VRAM peak", _report_number(system.get("peak_vram"))),
    ))
    video_rows = []
    for key in ("tracking_preview", "action_preview", "annotated_final"):
        receipt = video_results.get(key, {}) or {}
        audio = receipt.get("audio", {}) if isinstance(receipt.get("audio"), Mapping) else {}
        uri = receipt.get("uri") or f"videos/{key}.mp4"
        detail = audio.get("reason") or audio.get("warning")
        video_rows.append(
            "<tr>"
            f'<td class="key"><a href="{_html_escape(str(uri))}">{_html_escape(key)}</a></td>'
            f'<td class="value">{_report_count(receipt.get("frames"))} 帧</td>'
            f'<td class="value">{_report_text(receipt.get("audio_status"))}</td>'
            f'<td class="value">{_report_text(detail)}</td>'
            "</tr>"
        )
    evaluation_status = metrics.get("evaluation_status")
    report = "\n".join((
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Classroom Hybrid AI v3 report</title>",
        f"<style>{_REPORT_STYLE}</style></head><body>",
        "<header><h1>Classroom Student Action Report</h1>",
        f'<p class="meta">schema: classroom_hybrid_ai.report.v1 · profile: {_report_text(profile.profile_id)} · '
        f'students: {_report_count(len(students))} · evaluation: {_report_text(evaluation_status)}</p></header>',
        '<section id="video"><h2>视频信息 / Video</h2><table><tr><th>字段</th><th>值</th></tr>',
        _report_row("Source file", _report_text(source_name)),
        _report_row("Duration", _report_number(duration, " s")),
        _report_row("Frames", _report_count(frame_count)),
        _report_row("Resolution", _report_text(resolution)),
        "</table></section>",
        '<section id="coverage"><h2>覆盖率 / Coverage</h2><table><tr><th>字段</th><th>值</th></tr>',
        _report_row("Intervals total", _report_count(interval_count)),
        _report_row("Unknown / gray-zone intervals", _report_count(unknown_count)),
        _report_row("Unknown / gray-zone seconds", _report_number(unknown_seconds, " s")),
        _report_row("Covered seconds", _report_number(covered_seconds, " s")),
        _report_row("Overlay rows", _report_count(audit.get("overlay_row_count"))),
        "</table></section>",
        '<section id="stage-wall-time"><h2>阶段耗时 / Stage wall time</h2><table><tr><th>Stage</th><th>Wall time</th></tr>',
        stage_rows,
        _report_row("Total", _report_number(system.get("total_wall_time_seconds"), " s")),
        "</table></section>",
        '<section id="qwen"><h2>Qwen 调用 / Qwen calls</h2><table><tr><th>字段</th><th>值</th></tr>',
        _report_row("Qwen actual calls", _report_count(qwen_calls)),
        _report_row("Candidates", _report_count(candidate_count)),
        _report_row("Selected candidates", _report_count(selected_count)),
        "</table></section>",
        '<section id="vram"><h2>显存 / VRAM peak</h2><table><tr><th>字段</th><th>值</th></tr>',
        vram_rows,
        "</table></section>",
        '<section id="videos"><h2>视频与音频 / Videos and audio</h2><table>'
        "<tr><th>Video</th><th>Frames</th><th>audio_status</th><th>Audio detail</th></tr>",
        *video_rows,
        "</table></section>",
        '<section id="artifacts"><h2>产物链接 / Artifacts</h2><ul class="links">',
        '<li><a href="metrics.v1.json">metrics.v1.json</a></li>',
        '<li><a href="frontend/timeline_display.v1.json">frontend/timeline_display.v1.json</a></li>',
        '<li><a href="frontend/index.v1.json">frontend/index.v1.json</a></li>',
        '<li><a href="videos/">videos/</a></li>',
        "</ul></section>",
        "<footer>Generated by classroom_hybrid_ai S7. Missing values are marked 未记录; nothing is interpolated.</footer>",
        "</body></html>",
    ))
    return report


def build_v3_outputs(run_dir: str | Path, *, source: str | Path, info: Mapping[str, Any], profile: ActionProfile, config_digest: str, manifest_extra: Mapping[str, Any] | None = None, reuse_stage_artifacts: bool = False, progress_callback: Callable[[int, int], None] | None = None, encoder_backend: str = "opencv", ffmpeg_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir).resolve(); source_path = Path(source).resolve(); duration = float(info.get("duration_seconds", 0.0) or 0.0)
    preview_uris: list[str] = []
    if reuse_stage_artifacts:
        students = _read_rows(root / "students.jsonl")
        observations = _read_rows(root / "action_observations.jsonl")
        intervals = _read_rows(root / "action_intervals.jsonl")
        timeline = json.loads((root / "timeline.v2.json").read_text(encoding="utf-8"))
        taxonomy = json.loads((root / "taxonomy_profile.v1.json").read_text(encoding="utf-8"))
        if timeline.get("taxonomy_contract_sha256") != profile.sha256:
            raise ValueError("timeline_profile_sha256_mismatch")
        if taxonomy.get("contract_sha256") != profile.sha256:
            raise ValueError("taxonomy_profile_sha256_mismatch")
    else:
        students = _student_rows(root, duration)
        atomic_write_jsonl(root / "students.jsonl", students)
        observations = build_action_observations(root, profile)
        atomic_write_jsonl(root / "action_observations.jsonl", observations)
        intervals = normalize_intervals(root, profile, duration, students)
        atomic_write_jsonl(root / "action_intervals.jsonl", intervals)
        atomic_write_json(root / "taxonomy_profile.v1.json", taxonomy_payload(profile))
        timeline = {"schema_version": "classroom_hybrid_ai.timeline.v2", "video": dict(info), "student_rows": [row["student_id"] for row in students], "start_pts_seconds": 0.0, "end_pts_seconds": duration, "duration_seconds": duration, "taxonomy_profile_id": profile.profile_id, "taxonomy_contract_sha256": profile.sha256, "taxonomy_actions": [dict(item) for item in profile.actions], "intervals": intervals, "coverage": {"complete": True, "gray_gaps_included": True}}
        atomic_write_json(root / "timeline.v2.json", timeline)
    # Materialise a sorted overlay spool atomically, not a Python list.  Both
    # chunking and rendering consume the durable stream incrementally.
    overlay_spool, overlay_count = _spool_overlay_rows(root, info, intervals)
    chunks = write_overlay_chunks(root, iter_jsonl(overlay_spool))
    action_display_uri, interval_display_uri, timeline_display_uri = _write_english_display_json(
        root,
        timeline=timeline,
        observations=observations,
        intervals=intervals,
        profile=profile,
    )
    videos = root / "videos"; videos.mkdir(exist_ok=True)
    render_info = dict(info)
    routing_path = root / "artifacts/localization_routing.v1.json"
    if routing_path.is_file():
        routing = json.loads(routing_path.read_text(encoding="utf-8"))
        if isinstance(routing, Mapping) and routing.get("route") in {"yolo_only", "yolo_locate_hybrid"}:
            render_info["localization_route"] = routing["route"]
    video_results = _render_videos_once(
        source_path,
        {
            "tracking_preview": videos / "tracking_preview.mp4",
            "action_preview": videos / "action_preview.mp4",
            "annotated_final": videos / "annotated_final.mp4",
        },
        iter_jsonl(overlay_spool),
        info=render_info,
        progress_callback=progress_callback,
        encoder_backend=encoder_backend,
        ffmpeg_path=ffmpeg_path,
    )
    warnings: list[str] = []
    for key, receipt in video_results.items():
        if key == "annotated_final":
            audio_options: dict[str, Any] = {"ffmpeg_path": ffmpeg_path} if ffmpeg_path is not None else {}
            if encoder_backend == "nvenc":
                audio_options["validate_streams"] = True
            audio = _preserve_audio(source_path, Path(receipt["path"]), **audio_options)
        else:
            # The preview consists of disjoint/overlapping temporal clips.
            # Muxing the complete source track would shift audio relative to
            # the concatenated visual clips, so leave it out explicitly until
            # an atrim+concat audio pipeline is available.  This is a declared
            # contract of the preview artifacts, not a delivery failure: the
            # final annotated video carries the preserved audio track.
            audio = {
                "status": "unavailable",
                "preserved": False,
                "by_design": True,
                "reason": "preview_noncontiguous_clips_audio_not_muxed",
            }
        # Keep the historical boolean while exposing a structured state and
        # an explicit warning/reason for unavailable or failed muxing.
        receipt["audio"] = audio
        receipt["audio_preserved"] = bool(audio.get("preserved", False))
        receipt["audio_status"] = str(audio.get("status", "unknown"))
        if audio.get("warning"):
            warnings.append(f"{key}:{audio['warning']}")
        receipt["sha256"] = sha256_file(Path(receipt["path"]))
        receipt["bytes"] = Path(receipt["path"]).stat().st_size
    from .paper_evaluation import collect_system_metrics

    # S7 is still producing artifacts. Capture the committed upstream stages
    # now; its own final receipt and full-run timing remain explicitly missing.
    # A read-only evaluation after completion can report the final S7 timing.
    metrics = _metrics_not_evaluated(
        intervals, overlay_count, video_results, warnings,
        system_metrics=collect_system_metrics(root, video_info=info, stage_ids=tuple(f"S{index}" for index in range(7))),
    )
    atomic_write_json(root / "metrics.v1.json", metrics)
    report = _render_report_html(
        info=info,
        profile=profile,
        students=students,
        intervals=intervals,
        metrics=metrics,
        video_results=video_results,
    )
    (root / "report.html").write_text(report, encoding="utf-8")
    # Render S0-S7 semantic payloads after S7 files exist, so the S7 preview
    # can reference the final video, timeline, and report.
    preview_uris = _write_stage_previews(root, info)
    artifact_rows: list[dict[str, Any]] = []
    internal_diagnostics: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "artifact_index.v1.json"} or ".tmp" in path.name:
            continue
        relative_uri = path.relative_to(root).as_posix()
        if relative_uri == "frontend/index.v1.json":
            continue
        row = {"uri": relative_uri, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        if path.relative_to(root).parts and path.relative_to(root).parts[0] == "diagnostics":
            internal_diagnostics.append(row)
        else:
            artifact_rows.append(row)
    frontend = {"schema_version": "classroom_hybrid_ai.frontend_index.v1", "display_language": "english_ascii", "run_uri": "run_state.v1.json", "stages_uri": "stage_events.v1.jsonl", "students_uri": "students.jsonl", "actions_uri": action_display_uri, "intervals_uri": interval_display_uri, "timeline_uri": timeline_display_uri, "raw_actions_uri": "action_observations.jsonl", "raw_intervals_uri": "action_intervals.jsonl", "raw_timeline_uri": "timeline.v2.json", "metrics_uri": "metrics.v1.json", "artifact_index_uri": "frontend/artifact_index.v1.json", "video_uris": {key.removesuffix(".mp4"): f"videos/{key}" for key in ("tracking_preview.mp4", "action_preview.mp4", "annotated_final.mp4")}, "video_metadata": {name.removesuffix(".mp4"): {"uri": receipt["uri"], "semantic": receipt["semantic"], "semantic_id": receipt["semantic_id"], "clip_ranges": receipt["clip_ranges"], "frame_ranges": receipt["frame_ranges"], "frames": receipt["frames"], "audio_status": receipt["audio_status"]} for name, receipt in ((f"{key}.mp4", value) for key, value in video_results.items())}, "stage_preview_semantics": {"S0": "input contact sheet", "S1": "sampling schedule", "S2": "detections with actual detector source", "S3": "confirmed student IDs and unverified people", "S4": "student crops and Qwen composites", "S5": "English action labels and partial timeline", "S6": "raw anchors and merged intervals", "S7": "video, timeline, report, and frontend index"}, "preview_uris": preview_uris, "overlay_spool_uri": "artifacts/overlay_rows.v3.jsonl", "evidence_uri": "evidence/", "overlay_chunks": chunks, "students": students, "actions": [{"action_id": str(action["action_id"]), "action_code": str(action["code"]), "action_name_en": _english_action_name(action["code"], action.get("name_en")), "display_label": _english_action_name(action["code"], action.get("name_en")), "color": str(action["color"])} for action in profile.actions], "taxonomy_profile_id": profile.profile_id, "taxonomy_contract_sha256": profile.sha256, "endpoints": {"run": "/api/v1/run", "stages": "/api/v1/stages", "students": "/api/v1/students", "detections": "/api/v1/detections?start=&end=", "actions": "/api/v1/actions?student_id=&start=&end=", "timeline": "/api/v1/timeline", "metrics": "/api/v1/metrics", "artifacts": "/api/v1/artifacts/<path>", "events": "/api/v1/events"}, "contract_sha256": profile.sha256}
    atomic_write_json(root / "frontend" / "index.v1.json", frontend)
    frontend_index_path = root / "frontend" / "index.v1.json"
    artifact_rows.append({"uri": "frontend/index.v1.json", "sha256": sha256_file(frontend_index_path), "bytes": frontend_index_path.stat().st_size})
    artifact_rows.sort(key=lambda row: str(row["uri"]))
    artifact_index_path = root / "frontend" / "artifact_index.v1.json"
    atomic_write_json(artifact_index_path, {"schema_version": "classroom_hybrid_ai.artifact_index.v1", "artifacts": artifact_rows})
    manifest_artifacts = [*artifact_rows, {"uri": "frontend/artifact_index.v1.json", "sha256": sha256_file(artifact_index_path), "bytes": artifact_index_path.stat().st_size}]
    manifest = {"schema_version": "classroom_hybrid_ai.manifest.v3", "video": {**dict(info), "path": str(source_path), "size_bytes": source_path.stat().st_size, "sha256": sha256_file(source_path)}, "profile": profile.to_dict(), "profile_sha256": profile.sha256, "config_sha256": config_digest, "source_sha256": sha256_file(source_path), "runtime": {}, "warnings": warnings, "videos": video_results, "artifacts": manifest_artifacts, "internal_diagnostics": internal_diagnostics, "output_contract": {"timeline": "timeline.v2.json", "frontend": "frontend/index.v1.json", "final_video": "videos/annotated_final.mp4"}, **dict(manifest_extra or {})}
    atomic_write_json(root / "manifest.json", manifest)
    return {"students": len(students), "intervals": len(intervals), "overlays": overlay_count, "overlay_chunks": len(chunks), "videos": video_results, "unknown_intervals": sum(1 for row in intervals if row.get("action_id") == UNKNOWN_ID), "warnings": warnings, "manifest": "manifest.json"}


__all__ = ["build_v3_outputs", "build_overlay_rows", "iter_overlay_rows", "normalize_intervals", "render_video", "taxonomy_payload", "write_overlay_chunks", "write_stage_preview", "PROPAGATED_ALPHA", "MAX_BOX_HOLD_SECONDS"]
