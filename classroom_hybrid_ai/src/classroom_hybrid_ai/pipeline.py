"""Bounded-memory, sequential-GPU classroom video pipeline."""

from __future__ import annotations

import json
import math
import time
from collections import Counter, deque
from itertools import groupby
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib import request as urllib_request
from urllib import error as urllib_error

import cv2

from .adapters import (
    LocateAnythingAdapter,
    LocateAnythingHealthError,
    Qwen35Adapter,
    YoloAnomalyAdapter,
    YoloTrackerAdapter,
    qwen35_9b_8gb_device_map,
)
from .qwen_worker import QwenProcessClient
from .config import ProjectConfig
from .contracts import CandidateEvent, Detection, ReviewDecision, TrackObservation
from .evidence import (
    build_event_evidence,
    build_temporal_composite,
    extract_frames_at_times,
    filter_observations_for_candidates,
)
from .fusion import StrictFusionPolicy
from .hashing import canonical_sha256, project_config_digest, sha256_file
from .pose_rules import keypoint_displacement_score, pose_proxy_scores
from .preflight import run_preflight
from .runtime import GpuLease, release_model
from .storage import RunStore, iter_jsonl, atomic_write_json, append_jsonl
from .performance import measured, record_model_performance
from .open_vocab import (
    ActionDescriptor,
    ActionObservation,
    ActionRegistrySnapshot,
    TimelineDocument,
    build_timeline,
    normalize_phrase,
    session_vocabulary,
    session_vocabulary_digest,
)
from .registry import append_candidates, descriptors_from_discovery, load_latest
from .semantic import LocalBgeRetriever, SemanticCandidate, canonicalize_discoveries
from .tracking import fair_time_stratified_selection, stitch_tracklets, summarize_tracklet
from .taxonomy import FIXED_EIGHT_VISUAL_RULES, PROMPT_PRIORITY_CODES, fixed_registry, load_fixed_taxonomy
from .locate_tracking import (
    associate_locate_rows,
    candidates_from_locate_observations,
    materialize_locate_screenshots,
    render_locate_student_annotations,
    student_id_sort_key,
)
from .locate_service import LocateServiceError, LocateServiceSession
from .profiles import (
    ActionProfile,
    ActionProfileSchemaError,
    CORE8_PROFILE_ID,
    load_profile,
)
from .triggers import CooldownGate, EventDeduplicator, KOfMThresholdTrigger, PeriodicSampler, TriggerEngine
from .video import SceneChangeDetector, estimate_workload, iter_video, probe_video


_FROZEN_BENCHMARK_CODES = frozenset({"dx", "dk", "tt", "zt", "js", "zl", "xt", "jz"})


class PipelineError(RuntimeError):
    """Raised when an invariant required for auditable inference is violated."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    status: str
    run_dir: str
    video: dict[str, Any]
    counts: dict[str, int]
    stages: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["warnings"] = list(self.warnings)
        return value


@dataclass(slots=True)
class _RoundRobinTracks:
    cursor: int = 0

    def select(self, rows: Sequence[TrackObservation], limit: int) -> list[TrackObservation]:
        if limit <= 0 or not rows:
            return []
        ordered = sorted(rows, key=lambda row: row.track_id)
        take = min(limit, len(ordered))
        start = self.cursor % len(ordered)
        selected = [ordered[(start + offset) % len(ordered)] for offset in range(take)]
        self.cursor = (start + take) % len(ordered)
        return selected


def _event_id(observation: TrackObservation, event_type: str) -> str:
    return canonical_sha256({
        "stream_id": observation.stream_id,
        "track_id": observation.track_id,
        "event_type": event_type,
        "frame_index": observation.frame_index,
        "pts_seconds": round(observation.pts_seconds, 6),
    })[:32]


def _periodic_candidate(
    observation: TrackObservation,
    event_type: str,
    *,
    priority: float = 0.0,
) -> CandidateEvent:
    metadata = {"trigger": event_type, "small_model_assertion": False}
    if priority > 0:
        metadata["priority"] = float(priority)
    return CandidateEvent(
        event_id=_event_id(observation, event_type),
        track_id=observation.track_id,
        event_type=event_type,
        trigger_frame_index=observation.frame_index,
        trigger_pts_seconds=observation.pts_seconds,
        stream_id=observation.stream_id,
        score=None,
        evidence_frame_indices=(observation.frame_index,),
        source="periodic_audit",
        metadata=metadata,
    )


def _fair_candidates(candidates: Sequence[CandidateEvent], *, limit: int, bucket_seconds: float) -> list[CandidateEvent]:
    selected, _audit = fair_candidate_selection(
        candidates,
        limit=limit,
        bucket_seconds=bucket_seconds,
    )
    return selected


def fair_candidate_selection(
    candidates: Sequence[CandidateEvent],
    *,
    limit: int,
    bucket_seconds: float,
) -> tuple[list[CandidateEvent], list[dict[str, Any]]]:
    """Select a review queue with explicit, auditable coverage reasons.

    The queue reserves signal rows (anomaly/action-change/new-ID/edge-seat),
    then interleaves temporal buckets from both ends of the video and rotates
    tracks inside each bucket. This makes the budget independent of input
    ordering and gives every student a chance before a track receives a second
    request. The returned audit rows explain every selected and unselected
    candidate; callers persist them beside the queue.
    """

    if limit <= 0 or not candidates:
        return [], [{
            **candidate.to_dict(),
            "selection_status": "unselected",
            "selection_reason": "id_uncertain_no_qwen_request" if bool(candidate.metadata.get("id_uncertain")) else "qwen_budget_zero",
            "canonical_reason": "id_uncertain" if bool(candidate.metadata.get("id_uncertain")) else "budget_exhausted",
        } for candidate in candidates]
    if float(bucket_seconds) <= 0:
        raise ValueError("bucket_seconds must be positive")
    unique: dict[str, CandidateEvent] = {candidate.event_id: candidate for candidate in candidates}
    all_ordered = sorted(unique.values(), key=lambda item: (item.trigger_pts_seconds, student_id_sort_key(item.track_id), item.event_id))
    ordered = [
        candidate for candidate in all_ordered
        if not bool(candidate.metadata.get("id_uncertain"))
    ]
    max_pts = max((float(item.trigger_pts_seconds) for item in ordered), default=0.0)
    bucket_rows: dict[int, dict[str, list[CandidateEvent]]] = {}
    for candidate in ordered:
        bucket = int(float(candidate.trigger_pts_seconds) // float(bucket_seconds))
        bucket_rows.setdefault(bucket, {}).setdefault(candidate.track_id, []).append(candidate)
    for tracks in bucket_rows.values():
        for rows in tracks.values():
            rows.sort(key=lambda item: (
                -float(item.metadata.get("priority", 0.0) or 0.0) if isinstance(item.metadata, Mapping) else 0.0,
                item.trigger_pts_seconds,
                item.event_id,
            ))

    def signals(candidate: CandidateEvent) -> set[str]:
        metadata = candidate.metadata if isinstance(candidate.metadata, Mapping) else {}
        nested = metadata.get("selection_features") if isinstance(metadata.get("selection_features"), Mapping) else {}
        values: set[str] = set()
        for name, aliases in {
            "anomaly": ("anomaly", "custom_anomaly", "anomaly_candidate"),
            "action_change": ("action_change", "motion_change", "change"),
            "new_id": ("new_id", "new_student_id", "new_track"),
            "edge": ("edge", "edge_seat", "boundary"),
        }.items():
            if any(bool(metadata.get(alias)) or bool(nested.get(alias)) for alias in aliases):
                values.add(name)
        source = f"{candidate.source} {candidate.event_type}".lower()
        if "anomaly" in source:
            values.add("anomaly")
        if "change" in source or "motion" in source:
            values.add("action_change")
        if "new_track" in source:
            values.add("new_id")
        return values

    # Choose a stable temporal order that alternates early/late buckets. This
    # protects the latter half even when the budget is much smaller than the
    # number of source candidates.
    bucket_ids = sorted(bucket_rows)
    coverage_order: list[int] = []
    left, right = 0, len(bucket_ids) - 1
    while left <= right:
        coverage_order.append(bucket_ids[left])
        left += 1
        if left <= right:
            coverage_order.append(bucket_ids[right])
            right -= 1
    selected: list[CandidateEvent] = []
    selected_ids: set[str] = set()
    reasons: dict[str, list[str]] = {candidate.event_id: [] for candidate in ordered}

    def add(candidate: CandidateEvent, reason: str) -> bool:
        if len(selected) >= int(limit) or candidate.event_id in selected_ids:
            return False
        selected.append(candidate)
        selected_ids.add(candidate.event_id)
        reasons[candidate.event_id].append(reason)
        return True

    # Signal coverage gets the first claim on the budget. The round-robin
    # candidate order avoids a source-order or score-only first-N bias.
    all_track_ids = sorted({candidate.track_id for candidate in ordered}, key=student_id_sort_key)
    for signal_name in ("anomaly", "action_change", "new_id", "edge"):
        signal_rows = [candidate for candidate in ordered if signal_name in signals(candidate)]
        signal_rows.sort(key=lambda item: (
            0 if float(item.trigger_pts_seconds) >= max_pts * 0.5 else 1,
            -float(item.metadata.get("priority", 0.0) or 0.0) if isinstance(item.metadata, Mapping) else 0.0,
            item.trigger_pts_seconds,
            item.event_id,
        ))
        for candidate in signal_rows:
            # Do not let several signals from one student consume the slots
            # needed to give unseen students their first fair opportunity.
            if candidate.track_id in {item.track_id for item in selected} and len({item.track_id for item in selected}) < min(len(all_track_ids), int(limit)):
                continue
            if add(candidate, f"selected_{signal_name}_coverage"):
                break

    # First give each student at most one request. Choose a row across its
    # complete time range, not the first row encountered in source order.
    for track_id in all_track_ids:
        track_rows = [candidate for candidate in ordered if candidate.track_id == track_id and candidate.event_id not in selected_ids]
        track_rows.sort(key=lambda item: (
            0 if float(item.trigger_pts_seconds) >= max_pts * 0.5 else 1,
            -len(signals(item)),
            item.trigger_pts_seconds,
            item.event_id,
        ))
        if track_rows:
            add(track_rows[0], "selected_student_fairness_quota")

    # If a signal was attached to a student already used by another signal,
    # revisit it after every student has one slot. This preserves explicit
    # anomaly/action/edge/new-ID coverage whenever the budget has room.
    for signal_name in ("anomaly", "action_change", "new_id", "edge"):
        if len(selected) >= int(limit) or any(signal_name in signals(item) for item in selected):
            continue
        signal_rows = [
            candidate for candidate in ordered
            if candidate.event_id not in selected_ids and signal_name in signals(candidate)
        ]
        if not signal_rows:
            continue
        selected_counts = Counter(item.track_id for item in selected)
        signal_rows.sort(key=lambda item: (
            selected_counts.get(item.track_id, 0),
            0 if float(item.trigger_pts_seconds) >= max_pts * 0.5 else 1,
            float(item.trigger_pts_seconds),
            item.event_id,
        ))
        add(signal_rows[0], f"selected_{signal_name}_coverage")

    # Fill remaining capacity by choosing among the least-served students.
    # The temporal rank still alternates early/late buckets, so this step
    # cannot silently turn into a source-order first-N loop.
    bucket_rank = {bucket: rank for rank, bucket in enumerate(coverage_order)}
    while len(selected) < int(limit):
        remaining = [candidate for candidate in ordered if candidate.event_id not in selected_ids]
        if not remaining:
            break
        selected_counts = Counter(item.track_id for item in selected)
        # A track can exhaust its source rows before the global budget is
        # filled.  Excluding exhausted tracks is essential: otherwise their
        # lower selected count defines ``minimum_count`` while no remaining
        # candidate can satisfy it, leaving ``least_served`` empty.
        remaining_track_ids = {candidate.track_id for candidate in remaining}
        minimum_count = min(
            (selected_counts.get(track_id, 0) for track_id in remaining_track_ids),
            default=0,
        )
        least_served = [
            candidate for candidate in remaining
            if selected_counts.get(candidate.track_id, 0) == minimum_count
        ]
        latter_half_missing = not any(float(item.trigger_pts_seconds) >= max_pts * 0.5 for item in selected)
        least_served.sort(key=lambda item: (
            0 if latter_half_missing and float(item.trigger_pts_seconds) >= max_pts * 0.5 else 1,
            bucket_rank.get(int(float(item.trigger_pts_seconds) // float(bucket_seconds)), len(bucket_rank)),
            -len(signals(item)),
            float(item.trigger_pts_seconds),
            student_id_sort_key(item.track_id),
            item.event_id,
        ))
        candidate = least_served[0]
        if float(candidate.trigger_pts_seconds) >= max_pts * 0.5:
            reason = "selected_latter_half_coverage"
        else:
            reason = "selected_temporal_fairness_round"
        add(candidate, reason)

    # Defensive fill for sparse buckets: choose high-priority rows from all
    # remaining candidates, still preferring the second half and signals.
    if len(selected) < int(limit):
        remaining = [candidate for candidate in ordered if candidate.event_id not in selected_ids]
        remaining.sort(key=lambda item: (
            -len(signals(item)),
            0 if float(item.trigger_pts_seconds) >= max_pts * 0.5 else 1,
            -float(item.metadata.get("priority", 0.0) or 0.0) if isinstance(item.metadata, Mapping) else 0.0,
            item.trigger_pts_seconds,
            item.event_id,
        ))
        for candidate in remaining:
            if not add(candidate, "selected_budget_fill"):
                break

    audit: list[dict[str, Any]] = []
    selected_rank = {candidate.event_id: rank for rank, candidate in enumerate(selected, 1)}
    for candidate in all_ordered:
        if candidate.event_id in selected_ids:
            signal_values = signals(candidate)
            canonical_reason = "scheduled"
            audit.append({
                **candidate.to_dict(),
                "selection_status": "selected",
                "selection_rank": selected_rank[candidate.event_id],
                "selection_reason": ";".join(reasons[candidate.event_id]) or "selected_fair_budget",
                "selection_signals": sorted(signal_values),
                "canonical_reason": canonical_reason,
            })
            continue
        metadata = candidate.metadata if isinstance(candidate.metadata, Mapping) else {}
        if bool(metadata.get("id_uncertain")):
            reason = "id_uncertain_no_qwen_request"
            canonical_reason = "id_uncertain"
        else:
            if len(selected) >= int(limit):
                reason = "qwen_budget_exhausted_after_fair_coverage"
            elif candidate.track_id in {item.track_id for item in selected}:
                reason = "student_quota_after_fair_coverage"
            elif float(candidate.trigger_pts_seconds) < max_pts * 0.5:
                reason = "early_half_deprioritized_after_latter_half_coverage"
            else:
                reason = "temporal_bucket_already_covered"
            canonical_reason = "budget_exhausted"
        audit.append({
            **candidate.to_dict(),
            "selection_status": "unselected",
            "selection_reason": reason,
            "selection_signals": sorted(signals(candidate)),
            "canonical_reason": canonical_reason,
            "selection_metadata": {"priority": metadata.get("priority", 0.0)},
        })
    return selected, audit


def _candidate_priority(candidate: CandidateEvent) -> tuple[float, float, int, str]:
    metadata_priority = 0.0
    try:
        metadata_priority = float(candidate.metadata.get("priority", 0.0) or 0.0)
    except (TypeError, ValueError):
        metadata_priority = 0.0
    source = f"{candidate.source} {candidate.event_type}".lower()
    anomaly_priority = 1.0 if "anomaly" in source else 0.0
    return (
        max(metadata_priority, anomaly_priority),
        float(candidate.score or 0.0),
        -int(candidate.trigger_frame_index),
        candidate.event_id,
    )


def _merge_candidate_events(
    candidates: Sequence[CandidateEvent],
    *,
    window_seconds: float = 1.0,
) -> tuple[list[CandidateEvent], int]:
    """Merge nearby same-track triggers while retaining an audit trail."""

    if window_seconds < 0:
        raise ValueError("candidate_merge_window_must_be_non_negative")
    grouped: dict[tuple[str, str], list[CandidateEvent]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.stream_id, candidate.track_id), []).append(candidate)
    merged: list[CandidateEvent] = []
    merged_groups = 0
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item.trigger_pts_seconds, item.event_id))
        cluster: list[CandidateEvent] = []
        cluster_end = -float("inf")

        def flush() -> None:
            nonlocal merged_groups
            if not cluster:
                return
            if len(cluster) == 1:
                merged.append(cluster[0])
                return
            merged_groups += 1
            winner = max(cluster, key=_candidate_priority)
            evidence = tuple(sorted({
                frame
                for item in cluster
                for frame in item.evidence_frame_indices
            }))
            starts = [item.start_pts_seconds for item in cluster if item.start_pts_seconds is not None]
            ends = [item.end_pts_seconds for item in cluster if item.end_pts_seconds is not None]
            metadata = dict(winner.metadata)
            metadata.update({
                "merged_trigger_count": len(cluster),
                "merged_event_ids": [item.event_id for item in sorted(cluster, key=lambda item: item.event_id)],
                "merged_event_types": sorted({item.event_type for item in cluster}),
                "merge_window_seconds": float(window_seconds),
            })
            merged.append(CandidateEvent(
                event_id=winner.event_id,
                track_id=winner.track_id,
                event_type=winner.event_type,
                trigger_frame_index=winner.trigger_frame_index,
                trigger_pts_seconds=winner.trigger_pts_seconds,
                stream_id=winner.stream_id,
                start_pts_seconds=min(starts) if starts else winner.start_pts_seconds,
                end_pts_seconds=max(ends) if ends else winner.end_pts_seconds,
                score=max((item.score or 0.0) for item in cluster) if any(item.score is not None for item in cluster) else None,
                evidence_frame_indices=evidence,
                source=winner.source,
                metadata=metadata,
            ))

        for candidate in ordered:
            candidate_start = candidate.start_pts_seconds or candidate.trigger_pts_seconds
            if cluster and candidate_start - cluster_end > window_seconds:
                flush()
                cluster.clear()
            cluster.append(candidate)
            candidate_end = candidate.end_pts_seconds or candidate.trigger_pts_seconds
            cluster_end = max(cluster_end, candidate_end)
        flush()
    return sorted(merged, key=lambda item: (item.trigger_pts_seconds, item.event_id)), merged_groups


def _merge_and_select_candidates(
    candidates: Sequence[CandidateEvent],
    *,
    merge_window_seconds: float,
    limit: int,
    bucket_seconds: float,
) -> tuple[list[CandidateEvent], list[CandidateEvent], int]:
    """Merge nearby triggers before applying the fair review queue.

    Keeping this ordering in one helper prevents a later cap/queue operation
    from accidentally reintroducing the raw first-500 candidate bias.
    """

    merged, merged_groups = _merge_candidate_events(
        candidates,
        window_seconds=merge_window_seconds,
    )
    selected = _fair_candidates(
        merged,
        limit=limit,
        bucket_seconds=bucket_seconds,
    )
    return merged, selected, merged_groups


def _qwen_budget(duration_seconds: float, config: ProjectConfig) -> tuple[int, int, int]:
    """Return total, discovery and classification limits within an all-call cap.

    Two calls are held back for order-inverted text synonym adjudication.  They
    may go unused when BGE finds no close candidate, but may never push a long
    lesson above its configured Qwen request maximum.
    """

    minutes = max(0.0, duration_seconds) / 60.0
    total = min(
        int(config.get("open_vocabulary.total_qwen_requests_maximum", 240)),
        max(
            int(config.get("open_vocabulary.total_qwen_requests_minimum", 48)),
            int(math.ceil(float(config.get("open_vocabulary.total_qwen_requests_per_minute", 2.0)) * minutes)),
        ),
    )
    if str(config.get("qwen.action_mode", "open_vocabulary")) == "fixed_eight":
        return total, 0, total
    semantic_reserve = min(2, total)
    discovery = max(12, int(math.ceil(total * 0.20)))
    discovery = min(discovery, total - semantic_reserve)
    return total, discovery, total - discovery - semantic_reserve


class _QwenCallBudget:
    """Hard per-run call ledger shared by review and text adjudication passes."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or int(limit) < 0:
            raise ValueError("qwen_call_budget_must_be_non_negative")
        self.limit = int(limit)
        self.used = 0
        self.semantic_calls = 0

    def consume(self, *, semantic: bool = False) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        if semantic:
            self.semantic_calls += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


class _QwenBudgetExhausted(RuntimeError):
    """Internal signal used to serialize a budget-exhausted gray result."""


def _load_yolo_resume_checkpoint(
    path: Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load a completed tracking checkpoint only when its contract is intact."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "classroom_hybrid_ai.yolo_checkpoint.v1":
            return None
        if expected is not None:
            for key, expected_value in expected.items():
                if value.get(key) != expected_value:
                    return None
        candidates = [CandidateEvent.from_dict(row) for row in value["candidates"]]
        scenes = value["scenes"]
        stages = value["stages"]
        if not isinstance(scenes, list) or not isinstance(stages, Mapping):
            return None
        if any(not isinstance(row, Mapping) or not Path(str(row.get("path", ""))).is_file() for row in scenes):
            return None
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return {"candidates": candidates, "scenes": scenes, "stages": dict(stages)}


def _student_catalog(
    observation_path: Path,
    *,
    max_stitch_gap_seconds: float = 2.0,
    max_normalized_center_distance: float = 0.6,
    max_appearance_distance: float = 0.35,
    minimum_match_margin: float = 0.15,
) -> tuple[dict[str, str], list[dict[str, Any]], float]:
    """Build bounded per-track rows for the timeline backend.

    Per-track centre samples are capped because a class can run for hours; the
    row ordering remains stable enough for seat-based frontend placement.
    """

    if max_stitch_gap_seconds < 0 or max_normalized_center_distance < 0 or max_appearance_distance < 0:
        raise ValueError("tracklet_stitch_thresholds_must_be_non_negative")
    if minimum_match_margin < 0:
        raise ValueError("minimum_match_margin_must_be_non_negative")
    track_rows: dict[str, deque[TrackObservation]] = {}
    counts: dict[str, int] = {}
    maximum_pts = 0.0
    for row in iter_jsonl(observation_path):
        try:
            observation = TrackObservation.from_dict(row)
        except (TypeError, ValueError):
            continue
        track_rows.setdefault(observation.track_id, deque(maxlen=2048)).append(observation)
        counts[observation.track_id] = counts.get(observation.track_id, 0) + 1
        maximum_pts = max(maximum_pts, observation.pts_seconds)
    summaries = [summarize_tracklet(tuple(rows)) for rows in track_rows.values() if rows]
    locate_ids_are_persistent = bool(summaries) and all(
        item.track_id.startswith("student-")
        and item.track_id.removeprefix("student-").isdigit()
        for item in summaries
    )
    if locate_ids_are_persistent:
        # Locate-first S3 has already performed the cross-screenshot identity
        # association. Restitching these IDs with the legacy YOLO epoch parser
        # corrupts ``student-001`` into a duplicated camera/segment prefix and
        # breaks Qwen-to-timeline traceability. Preserve the authoritative IDs.
        mapping = {item.track_id: item.track_id for item in summaries}
        rows = [{
            "student_id": item.track_id,
            "source_track_ids": [item.track_id],
            "first_pts_seconds": round(item.first_pts_seconds, 6),
            "last_pts_seconds": round(item.last_pts_seconds, 6),
            "seat_center_xy": [round(item.center_x, 3), round(item.center_y, 3)],
            "observation_count": counts[item.track_id],
            "identity_scope": "locate_assigned_same_camera_video_run",
            "stitch_policy": {
                "applied": False,
                "provider": "LocateStudentIdAssigner",
                "reason": "persistent_locate_student_id_already_assigned",
            },
        } for item in summaries]
        rows.sort(key=lambda row: (
            row["seat_center_xy"][1],
            row["seat_center_xy"][0],
            student_id_sort_key(str(row["student_id"])),
        ))
        return mapping, rows, maximum_pts
    by_epoch: dict[str, list[Any]] = {}
    for summary in summaries:
        epoch = summary.track_id.split(":", 1)[0]
        by_epoch.setdefault(epoch, []).append(summary)
    mapping: dict[str, str] = {}
    clusters: dict[str, list[Any]] = {}
    for epoch, items in by_epoch.items():
        stitched = stitch_tracklets(
            items,
            max_gap_seconds=max_stitch_gap_seconds,
            max_normalized_center_distance=max_normalized_center_distance,
            max_appearance_distance=max_appearance_distance,
            minimum_match_margin=minimum_match_margin,
        )
        segment = epoch.replace("s", "segment-")
        for raw_track_id, local_student in stitched.items():
            student_id = f"camera-default-{segment}-{local_student}"
            mapping[raw_track_id] = student_id
            clusters.setdefault(student_id, []).append(next(item for item in items if item.track_id == raw_track_id))
    rows: list[dict[str, Any]] = []
    for student_id, items in clusters.items():
        xs = sorted(item.center_x for item in items)
        ys = sorted(item.center_y for item in items)
        rows.append({
            "student_id": student_id,
            "source_track_ids": sorted(item.track_id for item in items),
            "first_pts_seconds": round(min(item.first_pts_seconds for item in items), 6),
            "last_pts_seconds": round(max(item.last_pts_seconds for item in items), 6),
            "seat_center_xy": [round(xs[len(xs) // 2], 3), round(ys[len(ys) // 2], 3)],
            "observation_count": sum(counts[item.track_id] for item in items),
            "identity_scope": "same_camera_continuous_segment",
            "stitch_policy": {
                "gap_seconds_max": max_stitch_gap_seconds,
                "normalized_center_distance_max": max_normalized_center_distance,
                "appearance_distance_max": max_appearance_distance,
                "best_margin_min": minimum_match_margin,
            },
        })
    rows.sort(key=lambda row: (row["seat_center_xy"][1], row["seat_center_xy"][0], row["student_id"]))
    return mapping, rows, maximum_pts


def _hsv_histogram(frame_bgr: Any, bbox: Sequence[float]) -> list[float] | None:
    """Small normalized HSV signature used only to reconnect short tracklets."""

    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    total = float(histogram.sum())
    if total <= 0:
        return None
    return [round(float(value / total), 6) for value in histogram]


def _prepare_session_vocabulary(
    config: ProjectConfig,
    store: RunStore,
    *,
    discovery_reviews: Sequence[ReviewDecision],
    candidates: Sequence[CandidateEvent],
    action_profile: ActionProfile | None = None,
    qwen_adapter: Any | None = None,
    qwen_call_budget: _QwenCallBudget | None = None,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    if action_profile is not None:
        # Bind the exact profile selected at run start.  The legacy fixed
        # taxonomy loader is intentionally bypassed here so an extension11
        # experiment cannot inherit only the baseline eight actions.
        descriptors = tuple(
            ActionDescriptor(
                action_id=str(action["action_id"]),
                name_zh=str(action["name_zh"]),
                definition=str(action["definition"]),
                aliases=(str(action["code"]),),
                attributes={"profile_id": action_profile.profile_id, "profile_action_code": str(action["code"])},
                color=str(action["color"]),
                status="shared_candidate",
            )
            for action in action_profile.actions
        )
        registry = ActionRegistrySnapshot.create(descriptors)
        return registry, registry.actions, {
            "status": "skipped",
            "reason": "action_profile_taxonomy",
            "profile_id": action_profile.profile_id,
            "profile_sha256": action_profile.sha256,
            "phrases": 0,
            "merged": 0,
            "active_registry_actions": len(registry.actions),
            "provisional_actions": 0,
            "session_actions": len(registry.actions),
        }
    if str(config.get("qwen.action_mode", "open_vocabulary")) == "fixed_eight":
        actions = load_fixed_taxonomy(config.behavior_taxonomy_path)
        registry = fixed_registry(actions)
        vocabulary = registry.actions
        store.write_jsonl("artifacts/open_vocabulary_discoveries.jsonl", [])
        return registry, vocabulary, {
            "status": "skipped",
            "reason": "fixed_eight_taxonomy",
            "phrases": 0,
            "merged": 0,
            "active_registry_actions": len(vocabulary),
            "provisional_actions": 0,
            "session_actions": len(vocabulary),
        }
    track_to_student, _, _ = _student_catalog(
        store.run_dir / "artifacts/track_observations.jsonl",
        max_stitch_gap_seconds=float(config.get("tracking.max_stitch_gap_seconds", 2.0)),
        max_normalized_center_distance=float(config.get("tracking.max_normalized_center_distance", 0.6)),
        max_appearance_distance=float(config.get("tracking.max_appearance_distance", 0.35)),
        minimum_match_margin=float(config.get("tracking.minimum_match_margin", 0.15)),
    )
    by_event = {candidate.event_id: candidate for candidate in candidates}
    discoveries: list[dict[str, Any]] = []
    for review in discovery_reviews:
        candidate = by_event.get(review.candidate_event_id)
        normalized_behavior = normalize_phrase(review.behavior or "")
        if (
            candidate is None
            or review.decision != "confirmed"
            or not normalized_behavior
            or normalized_behavior.upper() == "UNKNOWN"
            or normalized_behavior.lower() in _FROZEN_BENCHMARK_CODES
        ):
            continue
        details = review.metadata.get("model_details", {})
        if not isinstance(details, Mapping):
            details = {}
        discoveries.append({
            "candidate_event_id": candidate.event_id,
            "raw_phrase": normalized_behavior,
            "student_id": track_to_student.get(candidate.track_id, candidate.track_id),
            "pts_seconds": candidate.trigger_pts_seconds,
            "evidence_quality": 0.8 if len(review.evidence_frame_indices) >= 2 else 0.5,
            "attributes": {
                key: details.get(key)
                for key in ("posture", "movement_state", "head_direction", "hand_action", "visible_objects", "interaction_target")
                if details.get(key) is not None
            },
        })
    registry = load_latest(config.action_registry_root)
    embedding_path = config.model_path("open_vocabulary.embedding_model_dir")
    retriever = LocalBgeRetriever(embedding_path)
    try:
        discoveries, semantic_stage = canonicalize_discoveries(
            discoveries,
            registry_actions=registry.actions,
            retriever=retriever,
            threshold=float(config.get("open_vocabulary.semantic_retrieval_threshold", 0.72)),
            top_k=int(config.get("open_vocabulary.semantic_retrieval_top_k", 5)),
            batch_adjudicator=(
                _semantic_batch_adjudicator(qwen_adapter, qwen_call_budget)
                if qwen_adapter is not None else None
            ),
        )
    finally:
        retriever.close()
    store.write_jsonl("artifacts/open_vocabulary_discoveries.jsonl", discoveries)
    discovered_shared, provisional = descriptors_from_discovery(
        discoveries,
        minimum_windows=int(config.get("open_vocabulary.shared_promotion_min_windows")),
        same_student_gap_seconds=float(config.get("open_vocabulary.shared_promotion_same_student_gap_seconds")),
        minimum_quality=float(config.get("open_vocabulary.shared_promotion_min_evidence_quality")),
        initial_colour_index=len(registry.actions),
    )
    registry = append_candidates(config.action_registry_root, registry, discovered_shared)
    active_registry_ids = {
        str(row["canonical_action_id"])
        for row in discoveries if row.get("canonical_action_id")
    } | {item.action_id for item in discovered_shared}
    active_shared = [item for item in registry.actions if item.action_id in active_registry_ids]
    vocabulary = session_vocabulary(
        active_shared,
        provisional,
        maximum_actions=int(config.get("open_vocabulary.maximum_session_actions")),
    )
    semantic_stage.update({
        "active_registry_actions": len(active_shared),
        "provisional_actions": len(provisional),
        "session_actions": len(vocabulary),
    })
    return registry, vocabulary, semantic_stage


def _write_open_vocab_outputs(
    config: ProjectConfig,
    store: RunStore,
    *,
    discovery_reviews: Sequence[ReviewDecision],
    classification_reviews: Sequence[ReviewDecision],
    candidates: Sequence[CandidateEvent],
    registry: Any,
    vocabulary: Sequence[Any],
    duration_seconds: float,
    video: Mapping[str, Any] | None = None,
    write_session_vocabulary: bool = True,
) -> dict[str, Any]:
    track_to_student, students, observed_duration = _student_catalog(
        store.run_dir / "artifacts/track_observations.jsonl",
        max_stitch_gap_seconds=float(config.get("tracking.max_stitch_gap_seconds", 2.0)),
        max_normalized_center_distance=float(config.get("tracking.max_normalized_center_distance", 0.6)),
        max_appearance_distance=float(config.get("tracking.max_appearance_distance", 0.35)),
        minimum_match_margin=float(config.get("tracking.minimum_match_margin", 0.15)),
    )
    by_event = {candidate.event_id: candidate for candidate in candidates}
    discovery_by_event = {review.candidate_event_id: review for review in discovery_reviews}
    discovery_rows_by_event: dict[str, Mapping[str, Any]] = {}
    discovery_artifact = store.run_dir / "artifacts/open_vocabulary_discoveries.jsonl"
    if discovery_artifact.is_file():
        for row in iter_jsonl(discovery_artifact):
            event_id = str(row.get("candidate_event_id", "")).strip()
            if event_id:
                discovery_rows_by_event[event_id] = row
    action_observations: list[ActionObservation] = []
    allowed_ids = {item.action_id for item in vocabulary}
    for review in classification_reviews:
        candidate = by_event.get(review.candidate_event_id)
        if candidate is None:
            continue
        resolved_action_id = review.behavior if review.decision == "confirmed" and review.behavior in allowed_ids else None
        discovery = discovery_by_event.get(candidate.event_id)
        discovery_row = discovery_rows_by_event.get(candidate.event_id, {})
        details = discovery.metadata.get("model_details", {}) if discovery is not None else {}
        classification_details = review.metadata.get("model_details", {})
        observability = (
            classification_details.get("observability")
            if isinstance(classification_details, Mapping)
            else None
        )
        if observability not in {"clear", "partial", "none"}:
            observability = "none"
        unobserved_reason: str | None = None
        # A single crop is too weak to paint a behavioral interval.  Keep the
        # absence of sufficient visual evidence explicit as gray/unobserved.
        if resolved_action_id is None or len(review.evidence_frame_indices) < 2:
            observability = "none"
            if review.reviewer == "pipeline_budget" or review.rationale == "qwen_call_budget_exhausted":
                unobserved_reason = "unreviewed"
            elif "conflict" in review.rationale.lower():
                unobserved_reason = "conflict"
            elif "out of memory" in review.rationale.lower() or "cuda oom" in review.rationale.lower():
                unobserved_reason = "model_oom"
            elif review.decision == "rejected":
                unobserved_reason = "model_rejected"
            elif (
                review.behavior == "UNKNOWN"
                or "frozen_vocabulary" in review.rationale
                or (
                    discovery_row.get("canonical_action_id") is not None
                    and str(discovery_row.get("canonical_action_id")) not in allowed_ids
                )
                or (
                    discovery_row.get("canonical_action_id") is None
                    and normalize_phrase(str(discovery_row.get("canonical_phrase") or discovery_row.get("raw_phrase") or ""))
                    and normalize_phrase(str(discovery_row.get("canonical_phrase") or discovery_row.get("raw_phrase") or ""))
                    not in {normalize_phrase(item.name_zh) for item in vocabulary}
                )
            ):
                unobserved_reason = "out_of_vocab"
            elif not review.schema_valid:
                unobserved_reason = "model_schema_error"
            elif len(review.evidence_frame_indices) < 2:
                unobserved_reason = "insufficient_evidence"
            else:
                unobserved_reason = "unreviewed"
        if observability == "none":
            resolved_action_id = None
            if unobserved_reason is None:
                unobserved_reason = "insufficient_evidence"
        secondary = details.get("secondary_behaviors", ()) if isinstance(details, Mapping) else ()
        if not isinstance(secondary, (list, tuple)) or any(not isinstance(item, str) for item in secondary):
            secondary = ()
        action_observations.append(ActionObservation(
            student_id=track_to_student.get(candidate.track_id, candidate.track_id),
            pts_seconds=candidate.trigger_pts_seconds,
            raw_phrase=normalize_phrase(discovery.behavior or "") or None if discovery is not None else None,
            primary_action_id=resolved_action_id,
            secondary_phrases=tuple(secondary),
            observability=observability,
            evidence_frame_indices=review.evidence_frame_indices,
            evidence_quality=0.8 if len(review.evidence_frame_indices) >= 2 else 0.5,
            source="qwen_classification",
            metadata={
                "candidate_event_id": candidate.event_id,
                "decision": review.decision,
                "visible_facts": details if isinstance(details, Mapping) else {},
            },
            unobserved_reason=unobserved_reason,
        ))
    # The timeline contract covers the complete source video.  A missing tail
    # after the last visible observation is a gray/unobserved interval, not a
    # reason to shorten the document.  ``observed_duration`` remains useful in
    # the audit payload below, especially for bounded development runs.
    timeline_duration = duration_seconds
    intervals = build_timeline(
        action_observations, vocabulary,
        student_ids=[row["student_id"] for row in students],
        duration_seconds=timeline_duration,
        max_hold_seconds=float(config.get("open_vocabulary.max_hold_seconds")),
        max_gap_seconds=float(config.get("open_vocabulary.max_unobserved_bridge_seconds")),
    )
    store.write_jsonl("students.jsonl", students)
    store.write_jsonl("action_observations.jsonl", action_observations)
    store.write_jsonl("action_intervals.jsonl", intervals)
    vocabulary_payload = {
        "schema_version": "classroom_hybrid_ai.session_vocabulary.v1",
        "registry_sha256": registry.sha256,
        "actions": [item.to_dict() for item in vocabulary],
    }
    vocabulary_payload["sha256"] = session_vocabulary_digest(
        vocabulary,
        registry_sha256=registry.sha256,
    )
    if write_session_vocabulary:
        store.write_json("session_vocabulary.v1.json", vocabulary_payload)
    timeline_document = TimelineDocument(
        video=dict(video or {}),
        student_rows=tuple(row["student_id"] for row in students),
        session_vocabulary_sha256=vocabulary_payload["sha256"],
        registry_sha256=registry.sha256,
        start_pts_seconds=0.0,
        end_pts_seconds=timeline_duration,
        intervals=tuple(intervals),
    )
    store.write_json("timeline.v1.json", timeline_document.to_dict())
    return {
        "students": len(students),
        "vocabulary_actions": len(vocabulary),
        "intervals": len(intervals),
        "registry_sha256": registry.sha256,
        "observed_duration_seconds": observed_duration,
        "timeline_duration_seconds": timeline_duration,
        "timeline_schema_version": timeline_document.schema_version,
    }


def _pose_xyc(detection: Mapping[str, Any]) -> list[list[float]] | None:
    attributes = detection.get("attributes")
    pose = attributes.get("pose") if isinstance(attributes, Mapping) else None
    if not isinstance(pose, Mapping):
        return None
    points = pose.get("keypoints")
    if not isinstance(points, list):
        return None
    confidences = pose.get("scores")
    output: list[list[float]] = []
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        confidence = 1.0
        if isinstance(confidences, list) and index < len(confidences):
            confidence = float(confidences[index])
        output.append([float(point[0]), float(point[1]), confidence])
    return output or None


def _proxy_scores(
    detection: Mapping[str, Any],
    config: ProjectConfig,
    *,
    keypoint_motion: float = 0.0,
    interaction_change: float = 0.0,
) -> dict[str, float]:
    confidence = float(detection.get("score") or 0.0)
    scores = pose_proxy_scores(_pose_xyc(detection), detection["bbox"], confidence)
    rules = config.get("yolo.proxy_candidate_rules", {})
    if isinstance(rules, Mapping) and rules.get("enabled") is False:
        return {}
    enabled = {
        "head_down": bool(rules.get("head_down", False)),
        "stand": bool(rules.get("standing", False)),
        "raise_hand": bool(rules.get("raised_hand", False)),
        "low_detection_confidence": "low_detection_confidence" in rules,
        "keypoint_motion": bool(rules.get("keypoint_motion", True)),
        "interaction_change": bool(rules.get("interaction_change", True)),
    }
    # A k-of-m window must receive an explicit zero whenever a visible track
    # does not exhibit a configured proxy.  Omitting that update lets sparse
    # positives accumulate across arbitrarily long periods.
    values = {key: float(scores.get(key, 0.0)) for key, active in enabled.items() if active}
    if enabled["keypoint_motion"]:
        values["keypoint_motion"] = max(0.0, min(1.0, float(keypoint_motion)))
    if enabled["interaction_change"]:
        values["interaction_change"] = max(0.0, min(1.0, float(interaction_change)))
    return values


def _tracker_observation(
    raw: Mapping[str, Any],
    *,
    epoch: int,
    tracklet_id: str | None = None,
    frame_index: int,
    pts_seconds: float,
    scene_score: float,
) -> TrackObservation | None:
    raw_track_id = raw.get("track_id")
    if raw_track_id is None:
        return None
    detection = Detection(
        label=str(raw.get("label") or "person"),
        bbox=tuple(float(value) for value in raw["bbox"]),
        score=None if raw.get("score") is None else float(raw["score"]),
        source=str(raw.get("source") or "yolo26_pose"),
        class_id=raw.get("class_id"),
        attributes=dict(raw.get("attributes") or {}),
    )
    return TrackObservation(
        track_id=f"s{epoch}:{tracklet_id or raw_track_id}",
        frame_index=frame_index,
        pts_seconds=pts_seconds,
        detection=detection,
        anomaly_scores={},
        metadata={
            "raw_track_id": str(raw_track_id),
            "tracklet_id": str(tracklet_id or raw_track_id),
            "scene_epoch": epoch,
            "scene_score": scene_score,
        },
    )


def _make_trigger_engine(config: ProjectConfig) -> TriggerEngine:
    k, m = config.get("triggering.k_of_m")
    return TriggerEngine(
        KOfMThresholdTrigger(
            k=k,
            m=m,
            high_threshold=float(config.get("triggering.high_threshold")),
            low_threshold=float(config.get("triggering.low_threshold")),
            require_full_window=bool(config.get("triggering.require_full_window", True)),
        ),
        cooldown=CooldownGate(float(config.get("triggering.cooldown_seconds"))),
        deduplicator=EventDeduplicator(float(config.get("triggering.merge_window_seconds"))),
        event_id_factory=_event_id,
    )


def _save_scene_frame(frame: Any, directory: Path, frame_index: int, pts: float, quality: int) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"scene_{frame_index:09d}.jpg"
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"image_write_failed:{path}")
    return {
        "path": path.as_posix(),
        "frame_index": frame_index,
        "pts_seconds": round(pts, 6),
        "sha256": sha256_file(path),
    }


def _track_pass(
    config: ProjectConfig,
    video_path: Path,
    store: RunStore,
    *,
    max_frames: int | None,
    tracker_factory: Callable[..., Any] = YoloTrackerAdapter,
) -> tuple[list[CandidateEvent], list[dict[str, Any]], dict[str, Any]]:
    tracker = tracker_factory(
        config.model_path("yolo.tracker_model"),
        config_dir=config.resolve(config.get("runtime.yolo_config_dir")),
        tracker=config.get("yolo.tracker"),
        device=config.get("runtime.device"),
        default_kwargs={
            "conf": float(config.get("yolo.confidence")),
            "imgsz": int(config.get("yolo.image_size")),
        },
    )
    trigger_engine = _make_trigger_engine(config)
    scene_detector = SceneChangeDetector(float(config.get("video.scene_change_threshold")))
    mode = str(config.get("triggering.mode"))
    if mode == "fixed_stride":
        sampler = PeriodicSampler(frame_stride=int(config.get("triggering.fixed_frame_stride")))
        audit_limit = int(config.get("triggering.max_tracks_per_fixed_stride"))
        audit_type = "fixed_stride_review"
    elif mode == "hybrid":
        sampler = PeriodicSampler(
            frame_stride=int(config.get("triggering.fixed_frame_stride")),
            pts_interval_seconds=float(config.get("triggering.periodic_audit_seconds")),
        )
        audit_limit = int(config.get("triggering.max_tracks_per_periodic_audit"))
        audit_type = "periodic_review"
    else:
        raise PipelineError(f"unsupported_triggering_mode:{mode}")
    selector = _RoundRobinTracks()
    candidates: list[CandidateEvent] = []
    scene_frames: list[dict[str, Any]] = []
    max_scenes = int(config.get("locate_anything.max_scene_images", 32))
    epoch = 0
    last_seen_pts: dict[str, float] = {}
    first_seen_pts: dict[str, float] = {}
    stable_track_emitted: set[str] = set()
    stable_track_seconds = float(config.get("triggering.new_track_stable_seconds", 1.0))
    reset_gap_seconds = float(config.get("triggering.reset_gap_seconds", 0.5))
    forced_audit_seconds = float(config.get("triggering.forced_audit_seconds", 120.0))
    if forced_audit_seconds <= 0:
        raise PipelineError("triggering.forced_audit_seconds_must_be_positive")
    max_stitch_gap_seconds = float(config.get("tracking.max_stitch_gap_seconds", 2.0))
    raw_tracklet_serial: dict[tuple[int, str], int] = {}
    raw_tracklet_last_seen: dict[tuple[int, str], float] = {}
    raw_tracklet_ids: dict[tuple[int, str], str] = {}
    pose_history: dict[str, deque[tuple[float, list[list[float]], tuple[float, float, float, float]]]] = {}
    previous_interactions: dict[str, dict[str, float]] = {}
    last_forced_audit_pts: dict[str, float] = {}
    scene_boundary_candidates = 0
    reconnect_candidates = 0
    forced_audit_candidates = 0
    frames = observations = unconfirmed = 0
    last_frame_index = -1
    writer = store.stream_jsonl("artifacts/track_observations.jsonl")
    teardown: dict[str, Any] = {}
    try:
        with writer:
            for decoded in iter_video(
                video_path, stride=int(config.get("video.detector_frame_stride", 1))
            ):
                if max_frames is not None and frames >= max_frames:
                    break
                last_frame_index = decoded.frame_index
                changed, scene_score = scene_detector.update(decoded.image_bgr)
                scene_boundary = bool(changed and decoded.frame_index > 0)
                if changed and decoded.frame_index > 0:
                    epoch += 1
                    trigger_engine.reset()
                    last_seen_pts.clear()
                    first_seen_pts.clear()
                    stable_track_emitted.clear()
                    pose_history.clear()
                    previous_interactions.clear()
                    last_forced_audit_pts.clear()
                if changed and len(scene_frames) < max_scenes:
                    scene_frames.append(_save_scene_frame(
                        decoded.image_bgr,
                        store.artifacts_dir / "locate_inputs",
                        decoded.frame_index,
                        decoded.pts_seconds,
                        int(config.get("evidence.jpeg_quality", 90)),
                    ))
                result = tracker.track(
                    decoded.image_bgr,
                    persist=bool(config.get("yolo.persist_tracks", True)),
                )
                current: list[TrackObservation] = []
                reconnected_track_ids: set[str] = set()
                pending: list[tuple[TrackObservation, dict[str, float], float | None, float, float, bool]] = []
                for raw in result.get("detections", []):
                    raw_track_id = raw.get("track_id")
                    if raw_track_id is None:
                        unconfirmed += 1
                        continue
                    raw_key = (epoch, str(raw_track_id))
                    previous_raw_seen = raw_tracklet_last_seen.get(raw_key)
                    tracklet_restarted = bool(
                        previous_raw_seen is not None
                        and decoded.pts_seconds - previous_raw_seen > max_stitch_gap_seconds
                    )
                    if (
                        previous_raw_seen is None
                        or decoded.pts_seconds - previous_raw_seen > max_stitch_gap_seconds
                    ):
                        serial = raw_tracklet_serial.get(raw_key, -1) + 1
                        raw_tracklet_serial[raw_key] = serial
                        raw_tracklet_ids[raw_key] = f"{raw_track_id}~{serial}"
                    raw_tracklet_last_seen[raw_key] = decoded.pts_seconds
                    observation = _tracker_observation(
                        raw,
                        epoch=epoch,
                        tracklet_id=raw_tracklet_ids[raw_key],
                        frame_index=decoded.frame_index,
                        pts_seconds=decoded.pts_seconds,
                        scene_score=scene_score,
                    )
                    if observation is None:
                        unconfirmed += 1
                        continue
                    if tracklet_restarted:
                        reconnected_track_ids.add(observation.track_id)
                    previous_seen = last_seen_pts.get(observation.track_id)
                    if previous_seen is not None and decoded.pts_seconds - previous_seen > reset_gap_seconds:
                        trigger_engine.reset_track(observation.stream_id, observation.track_id)
                        first_seen_pts[observation.track_id] = decoded.pts_seconds
                        stable_track_emitted.discard(observation.track_id)
                        pose_history.pop(observation.track_id, None)
                        previous_interactions.pop(observation.track_id, None)
                        last_forced_audit_pts.pop(observation.track_id, None)
                        reconnected_track_ids.add(observation.track_id)
                    last_seen_pts[observation.track_id] = decoded.pts_seconds
                    first_seen = first_seen_pts.setdefault(observation.track_id, decoded.pts_seconds)
                    keypoints = _pose_xyc(raw)
                    motion_score = 0.0
                    history = pose_history.get(observation.track_id)
                    if keypoints and history:
                        target_pts = decoded.pts_seconds - float(config.get("yolo.keypoint_motion_window_seconds", 0.5))
                        eligible = [item for item in history if item[0] <= target_pts + 0.05]
                        if eligible:
                            previous_pts, previous_keypoints, previous_bbox = min(
                                eligible, key=lambda item: abs(item[0] - target_pts)
                            )
                            if decoded.pts_seconds - previous_pts >= float(config.get("yolo.keypoint_motion_min_gap_seconds", 0.25)):
                                motion_score = keypoint_displacement_score(
                                    previous_keypoints,
                                    previous_bbox,
                                    keypoints,
                                    observation.detection.bbox,
                                    threshold=float(config.get("yolo.keypoint_motion_threshold", 0.15)),
                                )
                    scores = _proxy_scores(raw, config, keypoint_motion=motion_score)
                    appearance = _hsv_histogram(decoded.image_bgr, observation.detection.bbox)
                    observation = TrackObservation(
                        track_id=observation.track_id,
                        frame_index=observation.frame_index,
                        pts_seconds=observation.pts_seconds,
                        detection=observation.detection,
                        stream_id=observation.stream_id,
                        anomaly_scores=scores,
                        metadata={
                            **observation.metadata,
                            **({"hsv_histogram": appearance} if appearance is not None else {}),
                        },
                    )
                    if keypoints:
                        pose_history.setdefault(observation.track_id, deque(maxlen=64)).append(
                            (decoded.pts_seconds, keypoints, observation.detection.bbox)
                        )
                    pending.append((
                        observation,
                        scores,
                        appearance,
                        first_seen,
                        motion_score,
                        observation.track_id in reconnected_track_ids,
                    ))
                    current.append(observation)

                # Interaction relations require the complete set of visible
                # students in this frame.  Compute the change before writing
                # observations so the audit row contains every routing score.
                finalized: list[TrackObservation] = []
                for observation, scores, appearance, first_seen, motion_score, reconnected in pending:
                    signature = _interaction_signature(observation, current)
                    interaction_score = _interaction_change_score(
                        previous_interactions.get(observation.track_id), signature
                    )
                    scores = dict(scores)
                    if bool(config.get("yolo.proxy_candidate_rules.interaction_change", True)):
                        scores["interaction_change"] = interaction_score
                    final_observation = TrackObservation(
                        track_id=observation.track_id,
                        frame_index=observation.frame_index,
                        pts_seconds=observation.pts_seconds,
                        detection=observation.detection,
                        stream_id=observation.stream_id,
                        anomaly_scores=scores,
                        metadata={
                            **observation.metadata,
                            "keypoint_motion_score": motion_score,
                            "interaction_change_score": interaction_score,
                            "interaction_relations": signature,
                        },
                    )
                    writer.write(final_observation)
                    observations += 1
                    finalized.append(final_observation)
                    previous_interactions[final_observation.track_id] = signature
                    if scene_boundary:
                        candidates.append(_periodic_candidate(
                            final_observation,
                            "scene_boundary_review",
                            priority=50.0,
                        ))
                        scene_boundary_candidates += 1
                    if reconnected:
                        candidates.append(_periodic_candidate(
                            final_observation,
                            "track_reconnect_review",
                            priority=40.0,
                        ))
                        reconnect_candidates += 1
                    if (
                        final_observation.track_id not in stable_track_emitted
                        and final_observation.pts_seconds - first_seen >= stable_track_seconds
                    ):
                        candidates.append(_periodic_candidate(final_observation, "stable_new_track_review"))
                        stable_track_emitted.add(final_observation.track_id)
                    last_forced = last_forced_audit_pts.setdefault(
                        final_observation.track_id,
                        first_seen,
                    )
                    if final_observation.pts_seconds - last_forced >= forced_audit_seconds:
                        candidates.append(_periodic_candidate(
                            final_observation,
                            "forced_120s_audit",
                            priority=30.0,
                        ))
                        last_forced_audit_pts[final_observation.track_id] = final_observation.pts_seconds
                        forced_audit_candidates += 1
                    for event_type, score in scores.items():
                        candidate = trigger_engine.update(final_observation, event_type, score)
                        if candidate is not None:
                            candidates.append(candidate)
                current = finalized
                if sampler.should_sample(
                    decoded.frame_index, decoded.pts_seconds
                ):
                    for observation in selector.select(current, audit_limit):
                        candidates.append(_periodic_candidate(observation, audit_type))
                frames += 1
    finally:
        teardown = release_model(tracker)
    return candidates, scene_frames, {
        "status": "ok",
        "frames_processed": frames,
        "last_frame_index": last_frame_index,
        "observations": observations,
        "unconfirmed_track_detections": unconfirmed,
        "candidates": len(candidates),
        "scene_boundary_candidates": scene_boundary_candidates,
        "track_reconnect_candidates": reconnect_candidates,
        "forced_audit_candidates": forced_audit_candidates,
        "candidate_cap_reached": False,
        "tracklet_policy": {
            "max_stitch_gap_seconds": max_stitch_gap_seconds,
            "max_normalized_center_distance": float(config.get("tracking.max_normalized_center_distance", 0.6)),
            "max_appearance_distance": float(config.get("tracking.max_appearance_distance", 0.35)),
            "minimum_match_margin": float(config.get("tracking.minimum_match_margin", 0.15)),
        },
        "observation_artifact": writer.receipt,
        "teardown": teardown,
    }


def _box_overlap(subject: Sequence[float], event: Sequence[float]) -> float:
    """Association score that works for both person-sized and tiny object boxes."""

    sx1, sy1, sx2, sy2 = (float(value) for value in subject)
    ex1, ey1, ex2, ey2 = (float(value) for value in event)
    ix1, iy1, ix2, iy2 = max(sx1, ex1), max(sy1, ey1), min(sx2, ex2), min(sy2, ey2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    subject_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
    event_area = max(1.0, (ex2 - ex1) * (ey2 - ey1))
    union = subject_area + event_area - intersection
    return max(intersection / union, intersection / event_area)


def _interaction_signature(
    observation: TrackObservation,
    current: Sequence[TrackObservation],
) -> dict[str, float]:
    """Summarize visible person-person, hand-to-person and hand-object relations.

    This is deliberately a change signal, not an interaction classifier.  The
    signature is kept per track and fed to the same k-of-m trigger as pose
    movement; Qwen remains responsible for deciding what the relation means.
    An object detector is optional: when a tracker supplies object rows under
    ``detection.attributes.objects`` they are included without turning the
    stock pose checkpoint into an object/anomaly classifier.
    """

    x1, y1, x2, y2 = observation.detection.bbox
    subject_width = max(1.0, x2 - x1)
    subject_height = max(1.0, y2 - y1)
    subject_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    relations: dict[str, float] = {}
    pose = observation.detection.attributes.get("pose", {})
    keypoints = pose.get("keypoints", []) if isinstance(pose, Mapping) else []
    wrists = []
    if isinstance(keypoints, Sequence):
        for index in (9, 10):
            if index < len(keypoints) and isinstance(keypoints[index], (list, tuple)) and len(keypoints[index]) >= 2:
                wrists.append((float(keypoints[index][0]), float(keypoints[index][1])))
    for other in current:
        if other.track_id == observation.track_id:
            continue
        ox1, oy1, ox2, oy2 = other.detection.bbox
        other_center = ((ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0)
        scale = max(subject_width, subject_height, ox2 - ox1, oy2 - oy1, 1.0)
        distance = ((subject_center[0] - other_center[0]) ** 2 + (subject_center[1] - other_center[1]) ** 2) ** 0.5 / scale
        proximity = max(_box_overlap(observation.detection.bbox, other.detection.bbox), max(0.0, 1.0 - distance / 2.0))
        if proximity >= 0.15:
            relations[f"peer:{other.track_id}"] = max(0.0, min(1.0, proximity))
        for wrist in wrists:
            margin = max(4.0, scale * 0.10)
            if ox1 - margin <= wrist[0] <= ox2 + margin and oy1 - margin <= wrist[1] <= oy2 + margin:
                relations[f"hand:{other.track_id}"] = 1.0
                break

    # Optional object evidence can be attached by a separate local detector or
    # by a custom tracker adapter.  Keep the contract deliberately permissive
    # (``bbox``/``bbox_xyxy`` and ``label``) so the default pose-only path stays
    # dependency-free and emits no fabricated object detections.
    object_rows = observation.detection.attributes.get("objects", ())
    if isinstance(object_rows, Sequence) and not isinstance(object_rows, (str, bytes)):
        for index, item in enumerate(object_rows):
            if not isinstance(item, Mapping):
                continue
            object_box = item.get("bbox", item.get("bbox_xyxy"))
            if not isinstance(object_box, (list, tuple)) or len(object_box) != 4:
                continue
            try:
                object_box = tuple(float(value) for value in object_box)
                overlap = _box_overlap(observation.detection.bbox, object_box)
            except (TypeError, ValueError):
                continue
            label = str(item.get("label") or "object").strip() or "object"
            hand_score = 0.0
            for wrist in wrists:
                margin = max(3.0, subject_width * 0.06)
                if (
                    object_box[0] - margin <= wrist[0] <= object_box[2] + margin
                    and object_box[1] - margin <= wrist[1] <= object_box[3] + margin
                ):
                    hand_score = 1.0
                    break
            if overlap >= 0.02 or hand_score:
                relation_key = f"object:{label}:{index}"
                relations[relation_key] = max(0.0, min(1.0, max(overlap, hand_score)))
                if hand_score:
                    relations[f"hand_object:{label}:{index}"] = hand_score
    return relations


def _interaction_change_score(
    previous: Mapping[str, float] | None,
    current: Mapping[str, float],
) -> float:
    if previous is None:
        return 0.0
    keys = set(previous) | set(current)
    if not keys:
        return 0.0
    deltas = [abs(float(current.get(key, 0.0)) - float(previous.get(key, 0.0))) for key in keys]
    changed = sum(delta >= 0.15 for delta in deltas) / len(deltas)
    magnitude = sum(deltas) / len(deltas)
    return max(0.0, min(1.0, max(changed, magnitude)))


class _FrameObservationCursor:
    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self._groups = iter(
            (frame, list(group))
            for frame, group in groupby(rows, key=lambda row: int(row["frame_index"]))
        )
        self._current: tuple[int, list[dict[str, Any]]] | None = next(self._groups, None)

    def at(self, frame_index: int) -> list[dict[str, Any]]:
        while self._current is not None and self._current[0] < frame_index:
            self._current = next(self._groups, None)
        if self._current is not None and self._current[0] == frame_index:
            return self._current[1]
        return []


def _custom_anomaly_pass(
    config: ProjectConfig,
    video_path: Path,
    observation_path: Path,
    store: RunStore,
    *,
    last_frame_index: int,
    candidate_limit: int,
    adapter_factory: Callable[..., Any] = YoloAnomalyAdapter,
) -> tuple[list[CandidateEvent], dict[str, Any]]:
    model_path = config.model_path("yolo.anomaly_model")
    if model_path is None:
        store.write_jsonl("artifacts/custom_anomaly_detections.jsonl", [])
        return [], {"status": "skipped", "reason": "no_trained_local_anomaly_checkpoint"}
    class_map_raw = config.get("yolo.anomaly_class_map", {})
    if not isinstance(class_map_raw, Mapping) or not class_map_raw:
        raise PipelineError("yolo.anomaly_class_map_required_for_custom_checkpoint")
    class_map = {int(key): str(value) for key, value in class_map_raw.items()}
    adapter = adapter_factory(
        model_path,
        class_map=class_map,
        config_dir=config.resolve(config.get("runtime.yolo_config_dir")),
        device=config.get("runtime.device"),
        default_kwargs={
            "conf": float(config.get("yolo.confidence")),
            "imgsz": int(config.get("yolo.image_size")),
        },
    )
    cursor = _FrameObservationCursor(iter_jsonl(observation_path))
    engine = _make_trigger_engine(config)
    threshold = float(config.get("yolo.anomaly_track_iou_threshold", 0.2))
    stride = int(config.get("yolo.anomaly_frame_stride", 5))
    reset_gap_seconds = float(config.get("triggering.reset_gap_seconds", 0.5))
    anomaly_labels = tuple(dict.fromkeys(class_map.values()))
    candidates: list[CandidateEvent] = []
    last_seen_by_track: dict[str, float] = {}
    detections = matched = frames = 0
    writer = store.stream_jsonl("artifacts/custom_anomaly_detections.jsonl")
    try:
        with writer:
            for decoded in iter_video(video_path, stride=stride):
                if last_frame_index >= 0 and decoded.frame_index > last_frame_index:
                    break
                frames += 1
                track_rows = cursor.at(decoded.frame_index)
                base_observations: dict[str, TrackObservation] = {}
                scores_by_track: dict[str, dict[str, float]] = {}
                for track_row in track_rows:
                    try:
                        base = TrackObservation.from_dict(track_row)
                    except (TypeError, ValueError):
                        continue
                    previous_seen = last_seen_by_track.get(base.track_id)
                    if previous_seen is not None and decoded.pts_seconds - previous_seen > reset_gap_seconds:
                        engine.reset_track(base.stream_id, base.track_id)
                    last_seen_by_track[base.track_id] = decoded.pts_seconds
                    base_observations[base.track_id] = TrackObservation(
                        track_id=base.track_id,
                        frame_index=decoded.frame_index,
                        pts_seconds=decoded.pts_seconds,
                        detection=Detection(
                            label=base.detection.label,
                            bbox=base.detection.bbox,
                            score=base.detection.score,
                            source="yolo26_custom_anomaly",
                            class_id=base.detection.class_id,
                            attributes=dict(base.detection.attributes),
                        ),
                        stream_id=base.stream_id,
                        metadata={"custom_checkpoint_verified": True},
                    )
                    scores_by_track[base.track_id] = {label: 0.0 for label in anomaly_labels}
                result = adapter.predict(decoded.image_bgr)
                for anomaly in result.get("detections", []):
                    detections += 1
                    event_box = anomaly.get("bbox")
                    if not isinstance(event_box, (list, tuple)) or len(event_box) != 4:
                        continue
                    ranked = sorted(
                        (
                            (_box_overlap(row["detection"]["bbox"], event_box), row)
                            for row in track_rows
                            if isinstance(row.get("detection"), Mapping)
                        ),
                        key=lambda item: item[0],
                        reverse=True,
                    )
                    if not ranked or ranked[0][0] < threshold:
                        writer.write({
                            "frame_index": decoded.frame_index,
                            "pts_seconds": decoded.pts_seconds,
                            "status": "unassigned",
                            "anomaly": anomaly,
                        })
                        continue
                    association, track_row = ranked[0]
                    matched += 1
                    label = str(anomaly.get("label"))
                    score = float(anomaly.get("score") or 0.0)
                    track_id = str(track_row["track_id"])
                    if track_id in scores_by_track and label in scores_by_track[track_id]:
                        scores_by_track[track_id][label] = max(scores_by_track[track_id][label], score)
                    writer.write({
                        "frame_index": decoded.frame_index,
                        "pts_seconds": decoded.pts_seconds,
                        "status": "matched",
                        "track_id": track_id,
                        "association_score": association,
                        "anomaly": anomaly,
                    })
                # Every visible track receives an explicit zero for every
                # configured anomaly class.  Sparse positive detections cannot
                # accumulate across a long unobserved gap.
                for track_id, score_map in scores_by_track.items():
                    observation = base_observations[track_id]
                    for label, score in score_map.items():
                        candidate = engine.update(observation, label, score)
                        if candidate is not None:
                            candidates.append(CandidateEvent(
                                event_id=candidate.event_id,
                                track_id=candidate.track_id,
                                event_type=candidate.event_type,
                                trigger_frame_index=candidate.trigger_frame_index,
                                trigger_pts_seconds=candidate.trigger_pts_seconds,
                                stream_id=candidate.stream_id,
                                start_pts_seconds=candidate.start_pts_seconds,
                                end_pts_seconds=candidate.end_pts_seconds,
                                score=candidate.score,
                                evidence_frame_indices=candidate.evidence_frame_indices,
                                source=candidate.source,
                                metadata={
                                    **candidate.metadata,
                                    "priority": 100,
                                    "candidate_class": "custom_anomaly",
                                },
                            ))
    finally:
        teardown = release_model(adapter)
    return candidates, {
        "status": "ok",
        "frames_processed": frames,
        "detections": detections,
        "matched_to_tracks": matched,
        "candidates": len(candidates),
        "candidate_cap_reached": len(candidates) > candidate_limit,
        "artifact": writer.receipt,
        "teardown": teardown,
    }


def _post_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib_request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout) as opened:
        value = json.loads(opened.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise PipelineError("remote_unload_response_must_be_object")
    return value


def _is_connection_refused(error: BaseException) -> bool:
    """Recognize a dead optional HTTP worker across urllib/Windows/Python clients."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ConnectionRefusedError):
            return True
        errno_value = getattr(current, "errno", None)
        # POSIX ECONNREFUSED and Windows WSAECONNREFUSED.
        if errno_value in {111, 10061}:
            return True
        message = str(current).lower()
        if "connection refused" in message or "actively refused" in message:
            return True
        if isinstance(current, urllib_error.URLError):
            reason = getattr(current, "reason", None)
            if isinstance(reason, BaseException):
                current = reason
                continue
        current = current.__cause__ or current.__context__
    return False


def _is_service_unavailable(error: BaseException) -> bool:
    """Classify an unreachable optional worker without masking request state."""

    if _is_connection_refused(error):
        return True
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        message = str(current).lower()
        if "timed out" in message or "timeout" in message:
            return True
        if isinstance(current, urllib_error.URLError):
            reason = getattr(current, "reason", None)
            if isinstance(reason, BaseException):
                current = reason
                continue
        current = current.__cause__ or current.__context__
    return False


def _is_oom_error(error: BaseException) -> bool:
    """Detect CUDA/allocator OOMs without importing torch in the orchestrator."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        message = str(current).lower()
        if "outofmemory" in name or "out of memory" in message or "cuda oom" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _locate_anchor_frames(
    config: ProjectConfig,
    video_path: Path,
    candidates: Sequence[CandidateEvent],
    store: RunStore,
    *,
    fallback_scenes: Sequence[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize the exact candidate frames that will precede Qwen review.

    One Locate request localizes every student in an anchor frame, so multiple
    student candidates at the same frame share the same spatial result.
    """

    limit = int(config.get("locate_anything.max_scene_images", 32))
    if limit < 1:
        raise PipelineError("locate_max_scene_images_must_be_positive")
    ordered = _fair_candidates(
        candidates,
        limit=len(candidates),
        bucket_seconds=float(config.get("open_vocabulary.candidate_bucket_seconds", 300.0)),
    )
    selected: list[CandidateEvent] = []
    seen_frames: set[int] = set()
    for candidate in ordered:
        frame_index = int(candidate.trigger_frame_index)
        if frame_index in seen_frames:
            continue
        seen_frames.add(frame_index)
        selected.append(candidate)
        if len(selected) >= limit:
            break

    if not selected:
        rows = [dict(scene) for scene in fallback_scenes[:limit]]
        return rows, {
            "requested_candidates": len(candidates),
            "unique_candidate_frames": 0,
            "materialized_anchor_frames": 0,
            "fallback_scene_frames": len(rows),
            "truncated": False,
        }

    packets = extract_frames_at_times(
        video_path,
        [candidate.trigger_pts_seconds for candidate in selected],
    )
    output_dir = store.run_dir / "artifacts" / "locate_anchors"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen_actual_frames: set[int] = set()
    for candidate in selected:
        packet = packets.get(max(0.0, float(candidate.trigger_pts_seconds)))
        if packet is None:
            continue
        frame_index, pts_seconds, image = packet
        if frame_index in seen_actual_frames:
            continue
        seen_actual_frames.add(frame_index)
        path = output_dir / f"frame_{frame_index:09d}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"locate_anchor_write_failed:{path}")
        rows.append({
            "path": str(path),
            "frame_index": int(frame_index),
            "pts_seconds": float(pts_seconds),
            "anchor_event_id": candidate.event_id,
            "anchor_track_id": candidate.track_id,
            "anchor_source": "qwen_candidate",
        })
    return rows, {
        "requested_candidates": len(candidates),
        "unique_candidate_frames": len({int(item.trigger_frame_index) for item in candidates}),
        "selected_anchor_frames": len(selected),
        "materialized_anchor_frames": len(rows),
        "fallback_scene_frames": 0,
        "truncated": len({int(item.trigger_frame_index) for item in candidates}) > limit,
    }


def _locate_pass(
    config: ProjectConfig,
    scene_frames: Sequence[dict[str, Any]],
    store: RunStore,
    *,
    skip: bool,
    adapter_factory: Callable[..., Any] = LocateAnythingAdapter,
    progress_callback: Callable[[int, int], None] | None = None,
    request_fingerprint: str | None = None,
    reuse_responses: bool = False,
) -> dict[str, Any]:
    total_scenes = len(scene_frames)

    def report_progress(processed: int) -> None:
        if progress_callback is not None:
            progress_callback(min(max(0, int(processed)), total_scenes), total_scenes)

    if skip or not bool(config.get("locate_anything.enabled", False)):
        store.write_jsonl("artifacts/locate_results.jsonl", [])
        report_progress(0)
        return {"status": "skipped", "reason": "disabled_or_cli_skip", "requests": 0}
    backend = str(config.get("locate_anything.backend", "http"))
    adapter = adapter_factory(
        backend=backend,
        model_path=config.model_path("locate_anything.model_dir") if backend == "local" else None,
        base_url=str(config.get("locate_anything.service_url")),
        health_url=str(config.get("locate_anything.health_url", "")) or None,
        timeout=float(config.get("locate_anything.request_timeout_seconds")),
        max_new_tokens=int(config.get("locate_anything.max_new_tokens", 256)),
        maximum_image_side=config.get("locate_anything.maximum_image_side"),
        generation_mode=str(config.get("locate_anything.generation_mode", "hybrid")),
        release_cuda_cache=bool(config.get("locate_anything.release_cuda_cache", False)),
        profile_timings=bool(config.get("performance.profile_model_timings", False)),
    )
    rows: list[dict[str, Any]] = []
    request_count = 0
    reused_count = 0
    effective_parameters = {
        "max_new_tokens": int(config.get("locate_anything.max_new_tokens", 256)),
        "maximum_image_side": config.get("locate_anything.maximum_image_side"),
        "generation_mode": str(config.get("locate_anything.generation_mode", "hybrid")),
        "random_seed": int(config.get("research.random_seed", 0)),
    }
    error: str | None = None
    reason: str | None = None
    health: dict[str, Any] | None = None
    # A service can load its model before a timeout or malformed response.  Once
    # a request is attempted it is treated as GPU-resident until /unload ACKs.
    used_remote = False
    try:
        health_method = getattr(adapter, "health", None)
        if backend == "http" and callable(health_method):
            try:
                health_value = health_method()
                if not isinstance(health_value, Mapping):
                    raise LocateAnythingHealthError("health_response_must_be_object")
                health = dict(health_value)
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
                reason = "service_unavailable" if _is_service_unavailable(exc) else "health_check_failed"

        if error is None:
            labels = list(config.get("locate_anything.prompts", []))
            student_labels = [
                label for label in labels if str(label).strip().lower() in {"student", "person", "pupil", "learner"}
            ]
            context_labels = [label for label in labels if label not in student_labels]
            # Student localization is an explicit first request. Combining it
            # with books/desks/teachers in one open-vocabulary sentence made
            # the number and meaning of returned student boxes unpredictable.
            prompt_groups = [[student_labels[0]]] if student_labels else []
            if context_labels:
                prompt_groups.append(context_labels)
            if not prompt_groups and labels:
                prompt_groups = [labels]
            for scene_index, scene in enumerate(scene_frames):
                image = cv2.imread(scene["path"])
                if image is None:
                    rows.append({**scene, "status": "failed", "error": "scene_image_decode_failed"})
                    report_progress(scene_index + 1)
                    continue
                height, width = image.shape[:2]
                screenshot_sha256 = sha256_file(Path(scene["path"]))
                used_remote = used_remote or backend == "http"
                merged_detections: list[dict[str, Any]] = []
                raw_answers: list[str] = []
                for group_index, prompt_group in enumerate(prompt_groups):
                    request_key = canonical_sha256({
                        "stage_fingerprint": request_fingerprint,
                        "image_sha256": screenshot_sha256,
                        "image_size": [width, height],
                        "frame_index": scene.get("frame_index"),
                        "pts_seconds": scene.get("pts_seconds"),
                        "labels": prompt_group,
                        "parameters": effective_parameters,
                    })
                    cache_path = store.run_dir / "responses" / "locate" / f"{request_key}.json"
                    response = None
                    if request_fingerprint and reuse_responses and cache_path.is_file():
                        try:
                            cached = json.loads(cache_path.read_text(encoding="utf-8"))
                            candidate_response = cached["response"]
                            if (cached.get("request_id") == request_key
                                and cached.get("response_sha256") == canonical_sha256(candidate_response)
                                and isinstance(candidate_response, dict)
                                and isinstance(candidate_response.get("detections"), list)):
                                response = candidate_response
                                reused_count += 1
                        except (OSError, ValueError, KeyError, TypeError):
                            pass
                    if response is None:
                        request_count += 1
                        request_started = time.perf_counter()
                        try:
                            response = adapter.detect(
                                scene["path"], prompt_group, image_size=(width, height)
                            )
                        finally:
                            elapsed = time.perf_counter() - request_started
                            telemetry = getattr(adapter, "last_performance", {})
                            record_model_performance("locate", telemetry, wall_seconds=elapsed)
                            append_jsonl(store.run_dir / "diagnostics" / "locate_requests.performance.jsonl", {
                                "request_id": request_key, "wall_seconds": elapsed,
                                "performance": telemetry,
                            })
                        if request_fingerprint:
                            atomic_write_json(cache_path, {
                                "schema_version": "classroom_hybrid_ai.locate_response.v1",
                                "request_id": request_key,
                                "response_sha256": canonical_sha256(response),
                                "response": response,
                            })
                    raw_answers.append(str(response.get("raw_answer", "")))
                    for detection in response.get("detections", ()):
                        if not isinstance(detection, Mapping):
                            continue
                        normalized = dict(detection)
                        normalized["detection_index"] = len(merged_detections) + 1
                        attributes = dict(normalized.get("attributes") or {})
                        attributes["locate_query_group"] = group_index
                        attributes["locate_query_labels"] = list(prompt_group)
                        normalized["attributes"] = attributes
                        merged_detections.append(normalized)
                rows.append({
                    **scene,
                    "status": "ok",
                    "detections": merged_detections,
                    "raw_answer": "\n".join(raw_answers),
                    "metadata": {
                        "mode": "student_first_spatial_localization",
                        "request_count": len(prompt_groups),
                        "prompt_groups": prompt_groups,
                        "provides_track_ids": False,
                    },
                })
                report_progress(scene_index + 1)
    except Exception as exc:
        if error is None:
            error = f"{type(exc).__name__}:{exc}"
        reason = reason or ("service_unavailable" if _is_service_unavailable(exc) else "request_failed")
        # Refusal means there was no listening service to have loaded GPU state.
        # A timeout or malformed response remains unsafe and must acknowledge
        # unload before Qwen can start.
        if _is_connection_refused(exc):
            used_remote = False
    finally:
        unload: dict[str, Any] | None = None
        # Keep the unload request in a nested finally.  A failure while closing
        # the client or writing the Locate audit artifact must not skip the
        # remote GPU teardown handshake.
        try:
            teardown = release_model(adapter)
        finally:
            try:
                store.write_jsonl("artifacts/locate_results.jsonl", rows)
            finally:
                if used_remote:
                    try:
                        unload = _post_json(
                            str(config.get("locate_anything.unload_url")),
                            min(60.0, float(config.get("locate_anything.request_timeout_seconds"))),
                        )
                        acknowledged = unload.get("status") in {"ok", "unloaded"} and not unload.get("loaded", False)
                        if bool(config.get("locate_anything.require_unload_ack_before_qwen", True)) and not acknowledged:
                            raise PipelineError("locate_remote_did_not_acknowledge_gpu_unload")
                    except Exception as exc:
                        raise PipelineError(f"locate_remote_unload_failed:{type(exc).__name__}:{exc}") from exc
    if error and bool(config.get("locate_anything.required", False)):
        raise PipelineError(f"locate_stage_failed:{error}")
    return {
        "status": "warning" if error else "ok",
        "error": error,
        "reason": reason,
        "health": health,
        "requests": request_count,
        "reused_requests": reused_count,
        "effective_parameters": effective_parameters,
        "scene_images": len(rows),
        "detections": sum(len(row.get("detections", [])) for row in rows),
        "remote_unload": unload,
        "teardown": teardown,
    }


def _legacy_discovery_prompt() -> str:
    """No legacy action labels are included: discovery starts from an empty vocabulary."""
    return (
        "当前证据是一张 480x416 时序拼图：上排从左到右是 t-0.75 秒、t、t+0.75 秒的学生局部；下排是带学生框和轨迹标签的全局上下文。只能依据这张图，不得把拼图外的信息当作证据。\n"
        "Include observability with exactly one of: clear, partial, none. clear means the action is visibly supported across the supplied frames; partial means only incomplete evidence; none means not observable.\n"
        "只分析当前请求中同一学生的短时序图片。不要使用任何预置课堂动作类别；"
        "仅描述画面中可见的主动作，用具体中文短语（不得使用缩写、数据集代码或空泛单字），"
        "尽量包含可见的动作和对象，但不能推断情绪、意图、注意力或纪律。"
        "遮挡、模糊、证据冲突或无法确认持续动作时输出 uncertain。不要由轨迹ID推断身份，也不能把低头直接称作睡觉。\n"
        "只输出一个严格 JSON 对象，不得输出 Markdown。字段必须为：candidate_event_id、decision、"
        "reviewed_event_type、behavior、secondary_behaviors、posture、movement_state、head_direction、hand_action、"
        "visible_objects、interaction_target、rationale、evidence_frame_indices、self_reported_confidence。"
        "secondary_behaviors 与 visible_objects 必须是字符串数组；其余事实字段必须是字符串或 null。"
        "decision 只能是 confirmed、rejected、uncertain。"
    )


def _legacy_semantic_batch_adjudicator(adapter: Any) -> Callable[[Sequence[tuple[str, Sequence[SemanticCandidate]]]], Mapping[str, str | None]]:
    """Return a bounded, stateless two-pass text synonym adjudicator.

    ``canonicalize_discoveries`` invokes the returned function exactly twice:
    once in normal candidate order and once with each candidate list reversed.
    The response is intentionally a tiny strict JSON document and contains no
    video evidence or legacy eight-class labels.
    """

    def decide(proposals: Sequence[tuple[str, Sequence[SemanticCandidate]]]) -> Mapping[str, str | None]:
        if not proposals:
            return {}
        request_rows = [
            {
                "phrase": phrase,
                "candidates": [item.to_prompt_dict() for item in choices],
            }
            for phrase, choices in proposals
        ]
        prompt = (
            "只做中文动作短语同义归并，不分析图像，不推断意图、情绪、注意力或纪律。"
            "对每个 phrase，仅当它与一个候选表示相同的可见动作，且姿态、手部、物体和交互对象不冲突时，"
            "返回该候选 action_id；否则 action_id 为 null。"
            "只输出一个严格 JSON 对象，不得输出 Markdown。对象只能有 mappings 字段；"
            "mappings 是数组，每项只能有 phrase 和 action_id 字段，必须覆盖每个输入 phrase 一次。\n"
            + json.dumps({"proposals": request_rows}, ensure_ascii=False, separators=(",", ":"))
        )
        raw = adapter.complete(prompt, images=(), max_new_tokens=128)
        if set(raw) != {"mappings"} or not isinstance(raw["mappings"], list):
            raise PipelineError("qwen_semantic_response_schema_invalid")
        allowed = {phrase: {item.action_id for item in choices} for phrase, choices in proposals}
        output: dict[str, str | None] = {}
        for row in raw["mappings"]:
            if not isinstance(row, Mapping) or set(row) != {"phrase", "action_id"}:
                raise PipelineError("qwen_semantic_mapping_schema_invalid")
            phrase, selected = row["phrase"], row["action_id"]
            if not isinstance(phrase, str) or phrase not in allowed or phrase in output:
                raise PipelineError("qwen_semantic_mapping_phrase_invalid")
            if selected is not None and (not isinstance(selected, str) or selected not in allowed[phrase]):
                raise PipelineError("qwen_semantic_mapping_action_invalid")
            output[phrase] = selected
        if set(output) != set(allowed):
            raise PipelineError("qwen_semantic_mapping_incomplete")
        return output

    return decide


def _legacy_classification_prompt(vocabulary: Sequence[Any], vocabulary_sha256: str) -> str:
    choices = [
        {"action_id": item.action_id, "name_zh": item.name_zh, "definition": item.definition}
        for item in vocabulary
    ]
    return (
        "当前证据是一张 480x416 时序拼图：上排从左到右是 t-0.75 秒、t、t+0.75 秒的学生局部；下排是带学生框和轨迹标签的全局上下文。只能依据这张图，不得把拼图外的信息当作证据。\n"
        "Include observability with exactly one of: clear, partial, none. clear means the selected action is visibly supported across the supplied frames; partial means incomplete evidence; none means not observable.\n"
        "只分析当前请求中同一学生的短时序图片。请从下面冻结的课程动作词表选择一个主动作；"
        "不清楚、遮挡、词表没有合适动作时输出 uncertain 且 behavior 为 null。"
        "不能推断情绪、意图、注意力或纪律。behavior 若 confirmed 必须恰好是 action_id，不得输出中文名称。\n"
        "词表 SHA256=" + vocabulary_sha256 + "\n词表="
        + json.dumps(choices, ensure_ascii=False, separators=(",", ":"))
        + "\n只输出一个严格 JSON 对象，不得输出 Markdown。字段必须为：candidate_event_id、decision、"
        "reviewed_event_type、behavior、rationale、evidence_frame_indices、self_reported_confidence、vocabulary_sha256。"
        "decision 只能是 confirmed、rejected、uncertain。"
    )


# The source tree is used on Windows and on WSL. Keep the effective semantic
# prompt in ASCII source as well, so a host code-page conversion cannot turn
# the two-pass synonym adjudication request into mojibake.
def _semantic_batch_adjudicator(
    adapter: Any,
    call_budget: _QwenCallBudget | None = None,
) -> Callable[[Sequence[tuple[str, Sequence[SemanticCandidate]]]], Mapping[str, str | None]]:
    """Build a conservative, stateless two-pass synonym adjudicator.

    Semantic merging is an optional acceleration for the shared registry.  A
    malformed or truncated text response must leave phrases provisional and
    continue the course timeline; it must never abort the whole video after
    the visual discovery pass has already produced usable evidence.
    """

    def decide(proposals: Sequence[tuple[str, Sequence[SemanticCandidate]]]) -> Mapping[str, str | None]:
        if not proposals:
            return {}
        request_rows = [
            {
                "phrase": phrase,
                "candidates": [item.to_prompt_dict() for item in choices],
            }
            for phrase, choices in proposals
        ]
        prompt = (
            "Map each discovered Chinese action phrase to one candidate only when the visible action is the same. "
            "Reject the mapping when posture, hand action, visible object, or interaction target conflicts, even "
            "if the words look similar. "
            "Do not analyze images and do not infer intent, emotion, attention, or discipline. "
            "If no candidate is a safe synonym, return null. Return exactly one JSON object with a mappings array; "
            "each item must contain only phrase and action_id, and every input phrase must appear exactly once.\n"
            + json.dumps({"proposals": request_rows}, ensure_ascii=False, separators=(",", ":"))
        )
        if call_budget is not None and not call_budget.consume(semantic=True):
            return {phrase: None for phrase, _ in proposals}
        try:
            raw = adapter.complete(prompt, images=(), max_new_tokens=128)
            if not isinstance(raw, Mapping) or set(raw) != {"mappings"} or not isinstance(raw["mappings"], list):
                raise PipelineError("qwen_semantic_response_schema_invalid")
            allowed = {phrase: {item.action_id for item in choices} for phrase, choices in proposals}
            output: dict[str, str | None] = {}
            for row in raw["mappings"]:
                if not isinstance(row, Mapping) or set(row) != {"phrase", "action_id"}:
                    raise PipelineError("qwen_semantic_mapping_schema_invalid")
                phrase, selected = row["phrase"], row["action_id"]
                if not isinstance(phrase, str) or phrase not in allowed or phrase in output:
                    raise PipelineError("qwen_semantic_mapping_phrase_invalid")
                if selected is not None and (not isinstance(selected, str) or selected not in allowed[phrase]):
                    raise PipelineError("qwen_semantic_mapping_action_invalid")
                output[phrase] = selected
            if set(output) != set(allowed):
                raise PipelineError("qwen_semantic_mapping_incomplete")
            return output
        except Exception:
            # A failed text adjudication is conservative: keep the discovered
            # phrase provisional instead of aborting the video timeline.  This
            # also covers strict-schema failures, not only transport errors.
            return {phrase: None for phrase, _ in proposals}

    return decide


# The source tree is used on Windows and on WSL. Keep the effective prompts in
# ASCII source so they cannot be damaged by a host code-page conversion. Qwen
# still receives explicit Chinese-language instructions for the discovered
# behavior phrases. The legacy prompt helpers above remain named for old
# imports, while the pipeline uses the strict helpers below.
def _discovery_prompt(action_profile: ActionProfile | None = None) -> str:
    profile_context = (
        f"Run action profile is {action_profile.profile_id} with SHA256={action_profile.sha256}. "
        "This discovery pass records a phrase only; it must not add or remove profile actions.\n"
        if action_profile is not None else ""
    )
    return (
        profile_context
        + "Evidence is one 480x416 temporal composite. The top row contains the same student's crop at "
        "t-0.75s, t, and t+0.75s from left to right. The bottom row is full-class context with "
        "student boxes and track labels. Use only pixels in this composite; do not invent context.\n"
        "Set observability to exactly one of: clear, partial, none. clear means the action is visibly "
        "supported across the supplied frames; partial means incomplete evidence; none means not observable.\n"
        "Analyze only the requested student's short temporal sequence. Do not use any preset classroom "
        "action taxonomy. Describe the main visible action with a concrete Chinese verb-object phrase; "
        "never use an abbreviation, dataset code, or vague one-character label. Include a visible object "
        "when useful, but do not infer emotion, intent, attention, or discipline. If occluded, blurry, "
        "contradictory, or not confirmable, use decision=uncertain and behavior=null. Do not infer identity "
        "from a track ID, and do not call a lowered head sleeping without visible evidence.\n"
        "Return exactly one strict JSON object and no Markdown. Required fields: candidate_event_id, decision, "
        "reviewed_event_type, behavior, secondary_behaviors, posture, movement_state, head_direction, hand_action, "
        "visible_objects, interaction_target, rationale, evidence_frame_indices, self_reported_confidence, "
        "observability. secondary_behaviors and visible_objects must be string arrays; other factual fields "
        "must be strings or null. decision must be confirmed, rejected, or uncertain. "
        "Output one compact JSON line with no spaces or newlines. Keep rationale under 8 Chinese characters; "
        "keep every factual string under 8 characters, use at most 2 short array items, and set "
        "self_reported_confidence to null. Never add optional fields.\n"
    )


def _classification_prompt(
    vocabulary: Sequence[Any],
    vocabulary_sha256: str,
    *,
    action_profile: ActionProfile | None = None,
    prompt_version: str = "legacy_v1",
) -> str:
    if prompt_version not in {"legacy_v1", "compact_v1"}:
        raise ValueError(f"unsupported_prompt_version:{prompt_version}")
    if action_profile is not None:
        # A selected profile is immutable for the complete run.  Include all
        # fields (code, detailed name, definition, colour and digest) so the
        # prompt cannot silently drift from timeline/manifest output.
        choices = [
            {
                "action_id": str(item["action_id"]),
                "code": str(item["code"]),
                "name_zh": str(item["name_zh"]),
                "definition": str(item["definition"]),
                "color": str(item["color"]),
            }
            for item in action_profile.actions
        ]
        fixed_rules = FIXED_EIGHT_VISUAL_RULES if action_profile.is_core8 else ""
        profile_line = (
            f"Action profile={action_profile.profile_id}; profile SHA256={action_profile.sha256}. "
            "Use only this immutable profile.\n"
            f"Profile contract={json.dumps(action_profile.prompt_payload(), ensure_ascii=False, separators=(',', ':'))}\n"
        )
    else:
        priority = {code: index for index, code in enumerate(PROMPT_PRIORITY_CODES)}
        ordered_vocabulary = sorted(
            vocabulary,
            key=lambda item: priority.get(
                str(getattr(item, "attributes", {}).get("fixed_eight_code", "")),
                len(priority),
            ),
        )
        choices = [
            {"action_id": item.action_id, "name_zh": item.name_zh, "definition": item.definition}
            for item in ordered_vocabulary
        ]
        fixed_rules = (
            FIXED_EIGHT_VISUAL_RULES
            if len(vocabulary) == 8
            and all(getattr(item, "attributes", {}).get("fixed_eight_code") for item in vocabulary)
            else ""
        )
        profile_line = ""
    if prompt_version == "compact_v1":
        choices = [{key: item[key] for key in ("action_id", "name_zh", "definition")} for item in choices]
        profile_line = ""
    return (
        "Evidence is one 480x416 temporal composite. The top row contains the same student's crop at "
        "t-0.75s, t, and t+0.75s. The bottom row is full-class context with student boxes and track labels. "
        "Use only visible evidence in this composite.\n"
        "Set observability to exactly one of: clear, partial, none. clear means the selected action is visibly "
        "supported across the supplied frames; partial means incomplete evidence; none means it is not observable.\n"
        "Analyze only the requested student. Choose exactly one primary action from the frozen course vocabulary "
        "below. If no vocabulary action fits or evidence is insufficient, use decision=uncertain and the exact "
        "JSON string behavior=\"UNKNOWN\". Never emit JSON null, the string \"null\", or any label outside the "
        "frozen vocabulary. "
        "Do not infer identity from a track ID. "
        + fixed_rules
        + "\n"
        + profile_line
        + f"Vocabulary SHA256={vocabulary_sha256}\n"
        + f"Frozen vocabulary={json.dumps(choices, ensure_ascii=False, separators=(',', ':'))}\n"
        "Return exactly one compact JSON object and no Markdown with only these model fields: decision, behavior, "
        "rationale, evidence_frame_indices, observability. The pipeline attaches the exact candidate_event_id, "
        "reviewed_event_type, self_reported_confidence=null, and vocabulary_sha256 after parsing; do not output "
        "those request-bound fields. decision must be confirmed, rejected, or uncertain. Output one line with no "
        "spaces or newlines; keep rationale under 8 Chinese characters and never add optional fields.\n"
    )


def _make_qwen_adapter(
    config: ProjectConfig,
    *,
    adapter_factory: Callable[..., Any] = Qwen35Adapter,
) -> Any:
    backend = str(config.get("qwen.backend", "local"))
    model = (
        str(config.model_path("qwen.model_dir"))
        if backend == "local"
        else str(config.get("qwen.model_family", "qwen3.5-9b"))
    )
    offload = config.resolve(config.get("qwen.disk_offload_dir"))
    device_map_policy = str(config.get("qwen.device_map_policy", "auto"))
    device_map: Any = (
        qwen35_9b_8gb_device_map(int(config.get("runtime.gpu_index", 0)))
        if device_map_policy == "explicit_qwen35_9b_8gb"
        else device_map_policy
    )
    device_strategy = "legacy_offload"
    if device_map_policy == "cpu_embedding_gpu_head":
        from .adapters.qwen import qwen35_9b_cpu_embedding_gpu_head_device_map
        device_map = qwen35_9b_cpu_embedding_gpu_head_device_map(int(config.get("runtime.gpu_index", 0)))
        device_strategy = "cpu_embedding_gpu_head"
    adapter_kwargs = {
        "model": model,
        "backend": backend,
        "base_url": str(config.get("qwen.service_url")),
        "api_key": str(config.get("qwen.api_key", "EMPTY")),
        "timeout": float(config.get("qwen.request_timeout_seconds")),
        "max_new_tokens": int(config.get("qwen.max_new_tokens")),
        "max_oom_retries": max(0, len(config.get("qwen.oom_image_fallback", [])) - 1),
        "device_map": device_map,
        "max_memory": {
            int(config.get("runtime.gpu_index", 0)): f"{float(config.get('qwen.gpu_memory_limit_gib'))}GiB",
            "cpu": f"{float(config.get('qwen.cpu_offload_gib'))}GiB",
        } if backend == "local" else None,
        "offload_folder": offload if backend == "local" else None,
        "offload_state_dict": True,
        "enable_thinking": bool(config.get("qwen.enable_thinking", False)),
        "release_cuda_cache": bool(config.get("qwen.release_cuda_cache", False)),
        "device_strategy": device_strategy,
        "profile_timings": bool(config.get("performance.profile_model_timings", False)),
    }
    # Production local Qwen is isolated for the whole video.  Test doubles and
    # explicitly supplied adapter factories stay in-process so unit tests can
    # observe their call counters and failure circuits directly.
    if (
        backend == "local"
        and adapter_factory is Qwen35Adapter
        and bool(config.get("qwen.process_isolation", True))
    ):
        return QwenProcessClient(
            adapter_kwargs,
            timeout=float(config.get("qwen.request_timeout_seconds", 300.0)),
        )
    adapter_kwargs.pop("model")
    return adapter_factory(model, **adapter_kwargs)


def _classification_conflict_groups(
    candidates: Sequence[CandidateEvent],
    reviews: Sequence[ReviewDecision],
) -> list[list[CandidateEvent]]:
    """Find same-track anchors whose confirmed actions disagree."""

    by_event = {candidate.event_id: candidate for candidate in candidates}
    review_by_event = {review.candidate_event_id: review for review in reviews}
    eligible = [
        candidate for candidate in candidates
        if (
            (review := review_by_event.get(candidate.event_id)) is not None
            and review.schema_valid
            and review.decision == "confirmed"
            and isinstance(review.behavior, str)
            and review.behavior not in {"", "UNKNOWN"}
        )
    ]
    graph: dict[str, set[str]] = {candidate.event_id: set() for candidate in eligible}
    for index, left in enumerate(eligible):
        left_review = review_by_event[left.event_id]
        left_frames = set(left_review.evidence_frame_indices)
        for right in eligible[index + 1:]:
            if left.track_id != right.track_id:
                continue
            right_review = review_by_event[right.event_id]
            if left_review.behavior == right_review.behavior:
                continue
            right_frames = set(right_review.evidence_frame_indices)
            if (
                abs(left.trigger_pts_seconds - right.trigger_pts_seconds) <= 1.0
                or bool(left_frames & right_frames)
            ):
                graph[left.event_id].add(right.event_id)
                graph[right.event_id].add(left.event_id)
    groups: list[list[CandidateEvent]] = []
    visited: set[str] = set()
    for candidate in eligible:
        if candidate.event_id in visited or not graph[candidate.event_id]:
            continue
        stack = [candidate.event_id]
        component: list[CandidateEvent] = []
        while stack:
            event_id = stack.pop()
            if event_id in visited:
                continue
            visited.add(event_id)
            component.append(by_event[event_id])
            stack.extend(graph[event_id] - visited)
        groups.append(sorted(component, key=lambda item: (item.trigger_pts_seconds, item.event_id)))
    return groups


def _mark_conflict_review(
    review: ReviewDecision,
    *,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> ReviewDecision:
    return ReviewDecision(
        candidate_event_id=review.candidate_event_id,
        decision="uncertain",
        reviewed_event_type=review.reviewed_event_type,
        behavior=None,
        rationale=reason,
        evidence_frame_indices=(),
        reviewer="pipeline_conflict",
        schema_valid=False,
        metadata={
            **dict(review.metadata),
            "conflict": True,
            **dict(metadata or {}),
        },
    )


def _target_locate_evidence(
    candidate: CandidateEvent,
    locate_rows: Sequence[Mapping[str, Any]],
    *,
    maximum_delta_seconds: float = 2.0,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return bounded Locate metadata for only the current student."""

    if hasattr(locate_rows, "query"):
        return locate_rows.query(candidate, maximum_delta_seconds=maximum_delta_seconds, limit=limit)

    matches: list[tuple[float, int, int, dict[str, Any]]] = []
    for row in locate_rows:
        try:
            pts_seconds = float(row.get("pts_seconds", -1.0))
            frame_index = int(row.get("frame_index", -1))
        except (TypeError, ValueError):
            continue
        delta = abs(pts_seconds - candidate.trigger_pts_seconds)
        if delta > maximum_delta_seconds:
            continue
        for fallback_index, detection in enumerate(row.get("detections", ()), start=1):
            if not isinstance(detection, Mapping):
                continue
            if str(detection.get("student_id", "")) != str(candidate.track_id):
                continue
            bbox = detection.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            locate_index = int(detection.get(
                "locate_detection_index",
                detection.get("detection_index", fallback_index),
            ))
            compact = {
                "frame_index": frame_index,
                "pts_seconds": pts_seconds,
                "student_id": str(candidate.track_id),
                "locate_detection_index": locate_index,
                "bbox": [float(value) for value in bbox],
            }
            matches.append((delta, frame_index, locate_index, compact))
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in matches[: max(0, int(limit))]]


def _resolve_classification_conflicts(
    config: ProjectConfig,
    candidates: Sequence[CandidateEvent],
    evidence: Sequence[dict[str, Any]],
    reviews: Sequence[ReviewDecision],
    store: RunStore,
    *,
    adapter: Any,
    prompt: str,
    expected_vocabulary_sha256: str,
    allowed_action_ids: set[str],
    action_profile: ActionProfile | None = None,
    locate_rows: Sequence[Mapping[str, Any]],
    call_budget: _QwenCallBudget | None,
) -> tuple[list[ReviewDecision], dict[str, int]]:
    """Use at most one fresh Qwen vote per conflicting anchor group."""

    groups = _classification_conflict_groups(candidates, reviews)
    if not groups:
        return list(reviews), {"conflict_groups": 0, "conflict_rechecks": 0, "conflict_unresolved": 0}
    by_event = {candidate.event_id: candidate for candidate in candidates}
    evidence_by_event = {str(row.get("event_id")): row for row in evidence}
    review_by_event = {review.candidate_event_id: review for review in reviews}
    rechecks = unresolved = 0
    for group in groups:
        group_ids = [candidate.event_id for candidate in group]
        if call_budget is not None and not call_budget.consume():
            unresolved += 1
            for event_id in group_ids:
                review_by_event[event_id] = _mark_conflict_review(
                    review_by_event[event_id],
                    reason="classification_conflict_budget_exhausted",
                )
            continue
        anchor = group[-1]
        packet = evidence_by_event.get(anchor.event_id, {})
        composite = build_temporal_composite(
            packet,
            output_path=store.evidence_dir / anchor.event_id / "temporal_composite_conflict_recheck.jpg",
            jpeg_quality=int(config.get("evidence.jpeg_quality", 90)),
        )
        if composite is None:
            unresolved += 1
            for event_id in group_ids:
                review_by_event[event_id] = _mark_conflict_review(
                    review_by_event[event_id],
                    reason="classification_conflict_no_evidence",
                )
            continue
        frames = list(composite.get("frame_indices", ()))
        request_evidence = {
            "images": [composite["path"]],
            "candidate_source": anchor.source,
            "target_student_id": anchor.track_id,
            "target_locate_evidence": _target_locate_evidence(anchor, locate_rows),
            "target_highlighted": bool(packet.get("target_highlighted", False)),
            "metadata_scope": "current_student_only",
        }
        if action_profile is not None:
            request_evidence["action_profile"] = action_profile.prompt_payload()
        recheck_prompt = (
            prompt
            + "\nIndependent conflict recheck: the same student's nearby reviews disagree. "
              "Return one fresh compact decision from the frozen vocabulary; do not use any earlier answer."
        )
        rechecks += 1
        try:
            raw = adapter.review(
                anchor.event_id,
                request_evidence,
                event_type=anchor.event_type,
                evidence_frame_indices=frames,
                prompt=recheck_prompt,
                compact_response=True,
            )
            if "vocabulary_sha256" not in raw:
                raw = {
                    **raw,
                    "candidate_event_id": anchor.event_id,
                    "reviewed_event_type": anchor.event_type,
                    "self_reported_confidence": None,
                    "vocabulary_sha256": expected_vocabulary_sha256,
                }
            if action_profile is not None:
                raw = {**raw, "profile_id": action_profile.profile_id, "profile_sha256": action_profile.sha256}
            if raw.get("candidate_event_id") != anchor.event_id:
                raise PipelineError("qwen_conflict_recheck_candidate_event_id_mismatch")
            if raw.get("reviewed_event_type") != anchor.event_type:
                raise PipelineError("qwen_conflict_recheck_event_type_mismatch")
            raw = _validate_classification_response(
                raw,
                frames,
                allowed_action_ids=allowed_action_ids,
                action_profile=action_profile,
                target_highlighted=_target_highlighted(request_evidence),
            )
            if raw.get("vocabulary_sha256") != expected_vocabulary_sha256:
                raise PipelineError("qwen_conflict_recheck_vocabulary_sha256_mismatch")
            independent = ReviewDecision.from_dict(raw)
            store.write_json(
                f"responses/classification/{anchor.event_id}.conflict_recheck.json",
                raw,
            )
        except Exception as exc:
            unresolved += 1
            for event_id in group_ids:
                review_by_event[event_id] = _mark_conflict_review(
                    review_by_event[event_id],
                    reason="classification_conflict_recheck_failed",
                    metadata={"error": f"{type(exc).__name__}:{exc}"[:1000]},
                )
            continue

        group_behaviors = {
            review_by_event[event_id].behavior
            for event_id in group_ids
            if review_by_event[event_id].decision == "confirmed"
        }
        winning_behavior = (
            independent.behavior
            if independent.decision == "confirmed" and independent.behavior in group_behaviors
            else None
        )
        if winning_behavior is None:
            unresolved += 1
            for event_id in group_ids:
                review_by_event[event_id] = _mark_conflict_review(
                    review_by_event[event_id],
                    reason="classification_conflict",
                    metadata={"recheck_behavior": independent.behavior, "recheck_decision": independent.decision},
                )
        else:
            for event_id in group_ids:
                current = review_by_event[event_id]
                if current.behavior != winning_behavior:
                    review_by_event[event_id] = _mark_conflict_review(
                        current,
                        reason="classification_conflict_lost_recheck",
                        metadata={"winning_behavior": winning_behavior},
                    )
                else:
                    review_by_event[event_id] = ReviewDecision(
                        candidate_event_id=current.candidate_event_id,
                        decision=current.decision,
                        reviewed_event_type=current.reviewed_event_type,
                        behavior=current.behavior,
                        rationale=current.rationale,
                        evidence_frame_indices=current.evidence_frame_indices,
                        reviewer=current.reviewer,
                        schema_valid=current.schema_valid,
                        self_reported_confidence=current.self_reported_confidence,
                        metadata={
                            **dict(current.metadata),
                            "conflict": True,
                            "independent_recheck_behavior": winning_behavior,
                        },
                    )
        for event_id in group_ids:
            store.write_json(
                f"responses/classification/{event_id}.json",
                review_by_event[event_id].to_dict(),
            )
    return [review_by_event[review.candidate_event_id] for review in reviews], {
        "conflict_groups": len(groups),
        "conflict_rechecks": rechecks,
        "conflict_unresolved": unresolved,
    }


def _qwen_pass(
    config: ProjectConfig,
    candidates: Sequence[CandidateEvent],
    evidence: Sequence[dict[str, Any]],
    store: RunStore,
    *,
    skip: bool,
    prompt: str | None = None,
    response_prefix: str = "responses",
    expected_vocabulary_sha256: str | None = None,
    allowed_action_ids: set[str] | None = None,
    action_profile: ActionProfile | None = None,
    locate_rows: Sequence[Mapping[str, Any]] = (),
    response_contract: str = "review",
    adapter: Any | None = None,
    call_budget: _QwenCallBudget | None = None,
    release_adapter: bool = True,
    resolve_conflicts: bool = True,
    review_output: str = "reviews.jsonl",
    adapter_factory: Callable[..., Any] = Qwen35Adapter,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[ReviewDecision], dict[str, Any]]:
    total_candidates = len(candidates)

    def report_progress(processed: int) -> None:
        if progress_callback is not None:
            progress_callback(min(max(0, int(processed)), total_candidates), total_candidates)

    by_event = {row["event_id"]: row for row in evidence}
    if not candidates:
        store.write_jsonl(review_output, [])
        report_progress(0)
        return [], {
            "status": "skipped", "reason": "no_candidates", "reviews": 0,
            "stateless_requests": True, "history_messages_per_request": 0,
            "cross_request_kv_cache_reuse": False,
        }
    if skip or not bool(config.get("qwen.enabled", True)):
        reviews = [
            ReviewDecision(
                candidate_event_id=candidate.event_id,
                decision="uncertain",
                reviewed_event_type=candidate.event_type,
                rationale="qwen_review_skipped",
                evidence_frame_indices=candidate.evidence_frame_indices,
                reviewer="pipeline",
                schema_valid=False,
            )
            for candidate in candidates
        ]
        store.write_jsonl(review_output, reviews)
        report_progress(total_candidates)
        return reviews, {
            "status": "skipped", "reviews": len(reviews),
            "stateless_requests": True, "history_messages_per_request": 0,
            "cross_request_kv_cache_reuse": False,
        }
    adapter = adapter or _make_qwen_adapter(config, adapter_factory=adapter_factory)
    active_prompt = prompt or _discovery_prompt()
    from .localization_evidence import TargetLocalizationIndex
    from .research_policy import localization_route
    localization_index = TargetLocalizationIndex(locate_rows)
    generic_localization = localization_route(config) != "locate_only"
    profile_template = action_profile.prompt_payload() if (
        action_profile is not None and str(config.get("qwen.prompt_version", "legacy_v1")) == "legacy_v1") else None
    reviews: list[ReviewDecision] = []
    failures = 0
    budget_exhausted = 0
    schema_retries = 0
    compact_transport_responses = 0
    oom_circuit_open = False
    conflict_stats = {"conflict_groups": 0, "conflict_rechecks": 0, "conflict_unresolved": 0}
    started = time.perf_counter()
    try:
        warmup = getattr(adapter, "warmup", None)
        if callable(warmup):
            try:
                warmup()
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
                failures = len(candidates)
                reviews = [ReviewDecision(
                    candidate_event_id=candidate.event_id,
                    decision="uncertain",
                    reviewed_event_type=candidate.event_type,
                    rationale="qwen_load_failed",
                    evidence_frame_indices=(),
                    reviewer="pipeline_load_error",
                    schema_valid=False,
                    metadata={"error": error[:2000]},
                ) for candidate in candidates]
                store.write_jsonl(review_output, reviews)
                report_progress(total_candidates)
                return reviews, {
                    "status": "warning", "reviews": len(reviews), "failures": failures,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "stateless_requests": True, "load_circuit_open": True,
                    "history_messages_per_request": 0,
                    "cross_request_kv_cache_reuse": False,
                    "oom_circuit_open": _is_oom_error(exc),
                }
        for candidate_index, candidate in enumerate(candidates):
            packet = by_event.get(candidate.event_id, {})
            existing_composite = packet.get("qwen_composite") if isinstance(packet, Mapping) else None
            composite = (
                dict(existing_composite)
                if isinstance(existing_composite, Mapping)
                and Path(str(existing_composite.get("path", ""))).is_file()
                else build_temporal_composite(
                    packet,
                    output_path=store.evidence_dir / candidate.event_id / "temporal_composite.jpg",
                    jpeg_quality=int(config.get("evidence.jpeg_quality", 90)),
                )
            )
            images = [composite["path"]] if composite is not None else []
            frames = list(composite["frame_indices"]) if composite is not None else []
            if not images:
                reviews.append(ReviewDecision(
                    candidate_event_id=candidate.event_id,
                    decision="uncertain",
                    reviewed_event_type=candidate.event_type,
                    rationale="insufficient_visual_evidence",
                    evidence_frame_indices=(),
                    reviewer="pipeline",
                    schema_valid=False,
                ))
                failures += 1
                report_progress(candidate_index + 1)
                continue
            try:
                context_highlighted = bool(
                    packet.get("target_highlighted")
                    if isinstance(packet, Mapping) and "target_highlighted" in packet
                    else (composite.get("context_target_highlighted", False) if isinstance(composite, Mapping) else False)
                )
                request_evidence = {
                    "images": images,
                    "candidate_source": candidate.source,
                    "target_student_id": candidate.track_id,
                    "target_locate_evidence": _target_locate_evidence(candidate, localization_index),
                    "target_highlighted": context_highlighted,
                    "metadata_scope": "current_student_only",
                }
                if generic_localization:
                    request_evidence.pop("target_locate_evidence")
                    request_evidence["target_localization_evidence"] = localization_index.query(candidate, generic=True)
                    request_evidence["localization_contract_version"] = "classroom_hybrid_ai.localization_evidence.v2"
                if profile_template is not None:
                    request_evidence["action_profile"] = profile_template
                retry_attempts = 0
                while True:
                    try:
                        request_prompt = active_prompt
                        if retry_attempts:
                            request_prompt += (
                                "\nFresh retry: do not rely on any earlier response; return only the required "
                                "strict JSON object with no additional properties. For abstention use the exact "
                                "JSON string behavior=\"UNKNOWN\"; never emit JSON null or the string \"null\"."
                            )
                        if call_budget is not None and not call_budget.consume():
                            raise _QwenBudgetExhausted("qwen_call_budget_exhausted")
                        request_started = time.perf_counter()
                        try:
                            raw = adapter.review(
                                candidate.event_id,
                                request_evidence,
                                event_type=candidate.event_type,
                                evidence_frame_indices=frames,
                                prompt=request_prompt,
                                compact_response=response_contract == "classification",
                            )
                        finally:
                            elapsed = time.perf_counter() - request_started
                            telemetry = getattr(adapter, "last_performance", {})
                            record_model_performance("qwen", telemetry, wall_seconds=elapsed)
                            append_jsonl(store.run_dir / "diagnostics" / "qwen_requests.performance.jsonl", {
                                "event_id": candidate.event_id, "wall_seconds": elapsed,
                                "performance": telemetry,
                                "memory_snapshot": getattr(adapter, "last_metrics", {}),
                            })
                        if response_contract == "classification" and "vocabulary_sha256" not in raw:
                            if expected_vocabulary_sha256 is None:
                                raise PipelineError("qwen_compact_classification_hash_missing_from_context")
                            compact_transport_responses += 1
                            raw = {
                                **raw,
                                "candidate_event_id": candidate.event_id,
                                "reviewed_event_type": candidate.event_type,
                                "self_reported_confidence": None,
                                "vocabulary_sha256": expected_vocabulary_sha256,
                            }
                        if response_contract == "classification" and action_profile is not None:
                            raw = {
                                **raw,
                                "profile_id": action_profile.profile_id,
                                "profile_sha256": action_profile.sha256,
                            }
                        # The model must not be allowed to bind a valid JSON
                        # response to a different event or trigger type.  The
                        # compact classification transport already injects
                        # these fields; discovery and test doubles receive the
                        # same strict request-bound check here.
                        if raw.get("candidate_event_id") != candidate.event_id:
                            raise PipelineError("qwen_candidate_event_id_mismatch")
                        if raw.get("reviewed_event_type") != candidate.event_type:
                            raise PipelineError("qwen_reviewed_event_type_mismatch")
                        if response_contract == "discovery":
                            _validate_discovery_response(raw, frames)
                        else:
                            raw = _validate_classification_response(
                                raw,
                                frames,
                                allowed_action_ids=allowed_action_ids,
                                action_profile=action_profile,
                                target_highlighted=_target_highlighted(request_evidence),
                            )
                        if expected_vocabulary_sha256 is not None:
                            if raw.get("vocabulary_sha256") != expected_vocabulary_sha256:
                                raise PipelineError("qwen_vocabulary_sha256_mismatch")
                        review = ReviewDecision.from_dict(raw)
                        if retry_attempts:
                            review = ReviewDecision(
                                candidate_event_id=review.candidate_event_id,
                                decision=review.decision,
                                reviewed_event_type=review.reviewed_event_type,
                                behavior=review.behavior,
                                rationale=review.rationale,
                                evidence_frame_indices=review.evidence_frame_indices,
                                reviewer=review.reviewer,
                                schema_valid=review.schema_valid,
                                self_reported_confidence=review.self_reported_confidence,
                                metadata={**review.metadata, "fresh_retry_attempts": retry_attempts},
                            )
                        store.write_json(f"{response_prefix}/{candidate.event_id}.json", raw)
                        break
                    except _QwenBudgetExhausted:
                        raise
                    except Exception as exc:
                        if retry_attempts == 0 and not _is_oom_error(exc):
                            retry_attempts = 1
                            schema_retries += 1
                            continue
                        raise
            except _QwenBudgetExhausted as exc:
                budget_exhausted += 1
                review = ReviewDecision(
                    candidate_event_id=candidate.event_id,
                    decision="uncertain",
                    reviewed_event_type=candidate.event_type,
                    rationale="qwen_call_budget_exhausted",
                    evidence_frame_indices=tuple(frames),
                    reviewer="pipeline_budget",
                    schema_valid=False,
                    metadata={"error": str(exc), "fresh_retry_attempts": retry_attempts},
                )
                store.write_json(
                    f"{response_prefix}/{candidate.event_id}.json",
                    {
                        "candidate_event_id": candidate.event_id,
                        "status": "gray",
                        "reason": "qwen_call_budget_exhausted",
                        "stateless_request": False,
                        "fresh_retry_attempts": retry_attempts,
                    },
                )
            except Exception as exc:
                failures += 1
                error = f"{type(exc).__name__}:{exc}"
                is_model_schema_error = "model_schema_error:" in str(exc)
                review = ReviewDecision(
                    candidate_event_id=candidate.event_id,
                    decision="uncertain",
                    reviewed_event_type=candidate.event_type,
                    # Out-of-profile labels and missing extension context are
                    # explicit UNKNOWN/model_schema_error observations.  They
                    # must never become an unlabelled generic transport error.
                    behavior="UNKNOWN" if is_model_schema_error else None,
                    rationale=("model_schema_error:" + str(exc).split("model_schema_error:", 1)[-1])[:1000]
                    if is_model_schema_error else error[:1000],
                    evidence_frame_indices=tuple(frames),
                    reviewer="pipeline_schema_error" if is_model_schema_error else "pipeline_error",
                    schema_valid=False,
                    metadata={
                        "error": error[:2000],
                        "profile_id": action_profile.profile_id if action_profile is not None else None,
                        "profile_sha256": action_profile.sha256 if action_profile is not None else None,
                        "unobserved_reason": "model_schema_error" if is_model_schema_error else None,
                    },
                )
                store.write_json(
                    f"{response_prefix}/{candidate.event_id}.json",
                    {
                        "candidate_event_id": candidate.event_id,
                        "status": "error",
                        "error": error[:2000],
                        "stateless_request": True,
                        "fresh_retry_attempts": retry_attempts,
                    },
                )
                if _is_oom_error(exc):
                    oom_circuit_open = True
                    # Do not send another request after an OOM.  Remaining
                    # candidates are explicit gray outputs, not silent drops.
                    reviews.append(review)
                    for remaining in candidates[candidate_index + 1:]:
                        failures += 1
                        remaining_review = ReviewDecision(
                            candidate_event_id=remaining.event_id,
                            decision="uncertain",
                            reviewed_event_type=remaining.event_type,
                            rationale="qwen_oom_circuit_open",
                            evidence_frame_indices=(),
                            reviewer="pipeline_oom_circuit",
                            schema_valid=False,
                            metadata={"circuit_open": True},
                        )
                        reviews.append(remaining_review)
                        store.write_json(
                            f"{response_prefix}/{remaining.event_id}.json",
                            {
                                "candidate_event_id": remaining.event_id,
                                "status": "gray",
                                "reason": "qwen_oom_circuit_open",
                                "stateless_request": False,
                            },
                        )
                    report_progress(total_candidates)
                    break
            reviews.append(review)
            report_progress(candidate_index + 1)
        if (
            resolve_conflicts
            and
            response_contract == "classification"
            and not oom_circuit_open
            and reviews
            and expected_vocabulary_sha256 is not None
            and allowed_action_ids is not None
        ):
            reviews, conflict_stats = _resolve_classification_conflicts(
                config,
                candidates,
                evidence,
                reviews,
                store,
                adapter=adapter,
                prompt=active_prompt,
                expected_vocabulary_sha256=expected_vocabulary_sha256,
                allowed_action_ids=allowed_action_ids,
                action_profile=action_profile,
                locate_rows=locate_rows,
                call_budget=call_budget,
            )
    finally:
        teardown = release_model(adapter) if release_adapter else {"released": False, "shared_adapter": True}
    store.write_jsonl(review_output, reviews)
    return reviews, {
        "status": "warning" if failures else "ok",
        "reviews": len(reviews),
        "failures": failures,
        "budget_exhausted": budget_exhausted,
        "fresh_schema_retries": schema_retries,
        "compact_transport_responses": compact_transport_responses,
        **conflict_stats,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stateless_requests": True,
        "history_messages_per_request": 0,
        "cross_request_kv_cache_reuse": False,
        "model_weights_reused_within_video": True,
        "worker_process_exits_after_video": True,
        "request_metadata_scope": "current_student_only",
        "maximum_target_locate_records_per_request": 3,
        "oom_circuit_open": oom_circuit_open,
        "teardown": teardown,
    }


def _validate_discovery_response(
    value: Mapping[str, Any], evidence_frames: Sequence[int]
) -> None:
    required = {
        "secondary_behaviors", "posture", "movement_state", "head_direction", "hand_action", "observability",
        "visible_objects", "interaction_target",
    }
    missing = required.difference(value)
    if missing:
        raise PipelineError(f"qwen_discovery_missing_fields:{sorted(missing)}")
    for name in ("secondary_behaviors", "visible_objects"):
        item = value[name]
        if not isinstance(item, list) or any(not isinstance(part, str) for part in item):
            raise PipelineError(f"qwen_discovery_{name}_invalid")
    for name in ("posture", "movement_state", "head_direction", "hand_action", "interaction_target"):
        item = value[name]
        if item is not None and not isinstance(item, str):
            raise PipelineError(f"qwen_discovery_{name}_invalid")
    if value["observability"] not in {"clear", "partial", "none"}:
        raise PipelineError("qwen_discovery_observability_invalid")
    behavior = value.get("behavior")
    if value.get("decision") == "confirmed":
        if not isinstance(behavior, str) or len(normalize_phrase(behavior)) < 2:
            raise PipelineError("qwen_discovery_behavior_phrase_invalid")
        if normalize_phrase(behavior).lower() in _FROZEN_BENCHMARK_CODES:
            raise PipelineError("qwen_discovery_legacy_label_forbidden")
    elif behavior is not None and not isinstance(behavior, str):
        raise PipelineError("qwen_discovery_behavior_invalid")
    _validate_response_shape(
        value, allowed_extra=required, evidence_frames=evidence_frames
    )


def _target_highlighted(evidence: Mapping[str, Any] | None) -> bool:
    """Read the explicit target-highlight audit bit from a Qwen packet."""

    if not isinstance(evidence, Mapping):
        return False
    for key in ("target_highlighted", "context_target_highlighted"):
        if isinstance(evidence.get(key), bool):
            return bool(evidence[key])
    nested = evidence.get("global_context")
    if isinstance(nested, Mapping) and isinstance(nested.get("target_highlighted"), bool):
        return bool(nested["target_highlighted"])
    return False


def _validate_classification_response(
    value: Mapping[str, Any],
    evidence_frames: Sequence[int],
    *,
    allowed_action_ids: set[str] | None = None,
    action_profile: ActionProfile | None = None,
    target_highlighted: bool | None = None,
) -> dict[str, Any]:
    normalized_value: dict[str, Any] = dict(value)
    if "vocabulary_sha256" not in value or not isinstance(value["vocabulary_sha256"], str):
        raise PipelineError("qwen_classification_vocabulary_sha256_missing")
    if value.get("observability") not in {"clear", "partial", "none"}:
        raise PipelineError("qwen_classification_observability_invalid")
    behavior = value.get("behavior")
    allowed_extra = {"vocabulary_sha256", "observability"}
    if action_profile is not None:
        if "profile_id" in value and value.get("profile_id") != action_profile.profile_id:
            raise PipelineError("model_schema_error:profile_id_mismatch")
        if "profile_sha256" in value and value.get("profile_sha256") != action_profile.sha256:
            raise PipelineError("model_schema_error:profile_sha256_mismatch")
        try:
            canonical_behavior, action = action_profile.normalize_model_behavior(behavior)
        except ActionProfileSchemaError as exc:
            raise PipelineError(str(exc)) from exc
        normalized_value["behavior"] = canonical_behavior
        if action_profile.requires_target_highlight(behavior):
            highlighted = bool(target_highlighted)
            if not highlighted:
                raise PipelineError(
                    f"model_schema_error:target_highlight_required:{action_profile.profile_id}:{action['code']}"
                )
        allowed_extra.update({"profile_id", "profile_sha256"})
        expected_ids = set(action_profile.action_ids)
        if allowed_action_ids is None:
            allowed_action_ids = expected_ids
    elif behavior not in {None, "UNKNOWN"} and (
        not isinstance(behavior, str)
        or allowed_action_ids is None
        or behavior not in allowed_action_ids
    ):
        raise PipelineError("qwen_classification_behavior_not_in_frozen_vocabulary")
    _validate_response_shape(normalized_value, allowed_extra=allowed_extra, evidence_frames=evidence_frames)
    return normalized_value


def _validate_response_shape(
    value: Mapping[str, Any],
    *,
    allowed_extra: set[str],
    evidence_frames: Sequence[int] = (),
) -> None:
    base = {
        "candidate_event_id", "decision", "reviewed_event_type", "behavior", "rationale",
        "evidence_frame_indices", "self_reported_confidence", "reviewer", "schema_valid",
    }
    missing = base.difference(value)
    if missing:
        raise PipelineError(f"qwen_response_required_fields_missing:{sorted(missing)}")
    unknown = set(value).difference(base | allowed_extra)
    if unknown:
        raise PipelineError(f"qwen_response_additional_properties:{sorted(unknown)}")
    if value.get("decision") not in {"confirmed", "rejected", "uncertain"}:
        raise PipelineError("qwen_response_decision_invalid")
    if not isinstance(value.get("candidate_event_id"), str) or not value["candidate_event_id"].strip():
        raise PipelineError("qwen_response_candidate_event_id_invalid")
    if not isinstance(value.get("rationale"), str):
        raise PipelineError("qwen_response_rationale_invalid")
    if not isinstance(value.get("schema_valid"), bool):
        raise PipelineError("qwen_response_schema_valid_invalid")
    indices = value.get("evidence_frame_indices")
    if (
        not isinstance(indices, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in indices)
        or not set(indices).issubset(set(evidence_frames))
    ):
        raise PipelineError("qwen_response_evidence_indices_not_in_request")


class HybridPipeline:
    """Orchestrate each GPU-heavy stage with explicit teardown in between."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        tracker_factory: Callable[..., Any] = YoloTrackerAdapter,
        anomaly_factory: Callable[..., Any] = YoloAnomalyAdapter,
        locate_factory: Callable[..., Any] = LocateAnythingAdapter,
        qwen_factory: Callable[..., Any] = Qwen35Adapter,
        action_profile: ActionProfile | None = None,
    ) -> None:
        self.config = config
        self.tracker_factory = tracker_factory
        self.anomaly_factory = anomaly_factory
        self.locate_factory = locate_factory
        self.qwen_factory = qwen_factory
        # ResearchBackend supplies one frozen profile at run construction.  A
        # direct legacy HybridPipeline caller may omit it and retain the old
        # fixed/open-vocabulary compatibility behaviour.
        self.action_profile = action_profile

    def plan(self, video_path: str | Path, *, average_tracks: int = 30) -> dict[str, Any]:
        info = probe_video(video_path)
        if str(self.config.get("pipeline.primary_detector", "yolo")) == "locate_anything":
            stride = int(self.config.get("locate_first.screenshot_frame_stride", 10))
            screenshots = int(math.ceil(info.frame_count / stride))
            maximum = self.config.get("locate_first.maximum_screenshots")
            if maximum not in {None, 0, ""}:
                screenshots = min(screenshots, int(maximum))
            return {
                "video": info.to_dict(),
                "selected_policy": "locate_first_every_screenshot_student_box",
                "primary_detector": "locate_anything",
                "screenshot_frame_stride": stride,
                "estimated_locate_screenshots": screenshots,
                "estimated_qwen_requests": screenshots * max(0, int(average_tracks)),
                "qwen_request_order": "frame_then_student_id",
                "qwen_request_context": "one_fresh_user_message_no_history",
                "gpu_order": ["locateanything", "unload_locate", "qwen3.5_9b_child_process"],
                "simultaneous_gpu_models": 1,
            }
        return {
            "video": info.to_dict(),
            "workload": estimate_workload(
                info,
                average_tracks=average_tracks,
                fixed_stride=int(self.config.get("triggering.fixed_frame_stride")),
                periodic_seconds=float(self.config.get("triggering.periodic_audit_seconds")),
                tracks_per_periodic_audit=int(
                    self.config.get("triggering.max_tracks_per_periodic_audit")
                ),
            ),
            "selected_policy": self.config.get("triggering.mode"),
            "gpu_order": ["yolo26_pose", "optional_yolo26_custom_anomaly", "locate_anything", "qwen3.5_9b"],
            "simultaneous_gpu_models": 1,
        }

    def run(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None = None,
        skip_locate: bool = False,
        skip_qwen: bool = False,
        resume: bool = False,
        require_preflight: bool = True,
    ) -> PipelineResult:
        source = Path(video_path).resolve()
        info = probe_video(source)
        preflight = run_preflight(self.config, verify_hashes=False)
        if require_preflight and preflight["status"] != "ok":
            raise PipelineError("preflight_failed:" + ",".join(preflight["failed_checks"]))
        source_stat = source.stat()
        run_digest = canonical_sha256({
            "config": self.config.payload,
            "source": str(source),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "max_frames": max_frames,
            "skip_locate": skip_locate,
            "skip_qwen": skip_qwen,
        })
        store = RunStore(self.config.output_root, source.stem, run_digest)
        store.write_json("preflight.json", preflight)
        stages: dict[str, Any] = {}
        warnings: list[str] = []
        primary_detector = str(self.config.get("pipeline.primary_detector", "yolo"))
        locate_first_mode = primary_detector == "locate_anything"
        if locate_first_mode and skip_locate:
            raise PipelineError("cannot_skip_locate_when_it_is_the_primary_detector")
        checkpoint_path = store.run_dir / "artifacts/yolo_checkpoint.v1.json"
        checkpoint = _load_yolo_resume_checkpoint(
            checkpoint_path,
            expected={
                "run_digest": run_digest,
                "source_path": str(source),
                "config_digest": project_config_digest(self.config),
                "max_frames": max_frames,
            },
        ) if resume and not locate_first_mode else None
        if resume and locate_first_mode:
            warnings.append("locate_first_resume_not_implemented_full_run_used")
        preloaded_locate_rows: list[dict[str, Any]] | None = None
        preloaded_locate_stage: dict[str, Any] | None = None
        lease_path = self.config.resolve(".runtime/classroom_hybrid_ai/gpu.lock")
        assert lease_path is not None
        with GpuLease(lease_path):
            if locate_first_mode:
                maximum_screenshots_value = self.config.get("locate_first.maximum_screenshots")
                maximum_screenshots = (
                    None
                    if maximum_screenshots_value in {None, 0, ""}
                    else int(maximum_screenshots_value)
                )
                scenes, stages["locate_screenshot_sampling"] = materialize_locate_screenshots(
                    source,
                    store.run_dir / "artifacts" / "locate_screenshots",
                    frame_stride=int(self.config.get("locate_first.screenshot_frame_stride", 10)),
                    max_frames=max_frames,
                    maximum_screenshots=maximum_screenshots,
                    jpeg_quality=int(self.config.get("locate_first.jpeg_quality", 95)),
                )
                locate_service = LocateServiceSession(self.config, store.run_dir)
                try:
                    with locate_service:
                        preloaded_locate_stage = _locate_pass(
                            self.config,
                            scenes,
                            store,
                            skip=False,
                            adapter_factory=self.locate_factory,
                        )
                except LocateServiceError as exc:
                    raise PipelineError(
                        f"locate_service_prepare_failed:{type(exc).__name__}:{exc}"
                    ) from exc
                preloaded_locate_stage["service_process"] = dict(locate_service.receipt)
                preloaded_locate_stage["execution_order"] = "before_id_assignment_and_qwen"
                if preloaded_locate_stage["status"] == "warning":
                    locate_reason = str(preloaded_locate_stage.get("reason") or "warning")
                    warnings.append(f"locate_anything_{locate_reason}")
                raw_locate_rows = list(iter_jsonl(store.run_dir / "artifacts/locate_results.jsonl"))
                observations, preloaded_locate_rows, locate_boxes, locate_tracking_stage = associate_locate_rows(
                    raw_locate_rows,
                    max_gap_seconds=float(self.config.get("locate_first.max_gap_seconds", 2.0)),
                    minimum_iou=float(self.config.get("locate_first.minimum_iou", 0.10)),
                    maximum_center_distance=float(self.config.get("locate_first.maximum_center_distance", 0.75)),
                    minimum_match_score=float(self.config.get("locate_first.minimum_match_score", 0.30)),
                    minimum_match_margin=float(self.config.get("locate_first.minimum_match_margin", 0.10)),
                )
                preloaded_locate_rows, locate_visual_stage = render_locate_student_annotations(
                    preloaded_locate_rows,
                    store.run_dir / "artifacts" / "locate_annotated",
                    jpeg_quality=int(self.config.get("locate_first.jpeg_quality", 95)),
                )
                annotated_by_frame = {
                    int(row["frame_index"]): str(row["annotated_path"])
                    for row in preloaded_locate_rows
                    if row.get("annotated_path")
                }
                locate_boxes = [
                    {
                        **row,
                        "annotated_path": annotated_by_frame.get(int(row["frame_index"])),
                    }
                    for row in locate_boxes
                ]
                # The canonical Locate artifact now contains both the model's
                # frame-local L-number and the persistent student_id assigned
                # by the dedicated cross-frame association component.
                store.write_jsonl("artifacts/locate_results.jsonl", preloaded_locate_rows)
                store.write_jsonl("artifacts/locate_student_boxes.jsonl", locate_boxes)
                store.write_jsonl("artifacts/track_observations.jsonl", observations)
                locate_tracking_stage.update({
                    "last_frame_index": max((row.frame_index for row in observations), default=-1),
                    "model": "LocateAnything-3B",
                    "yolo_model_loaded": False,
                    "student_box_artifact": "artifacts/locate_student_boxes.jsonl",
                })
                stages["student_id_assignment"] = dict(locate_tracking_stage)
                stages["locate_visual_audit"] = locate_visual_stage
                # Keep the legacy key for downstream counts/output contracts;
                # its source field makes clear that YOLO was not run.
                stages["yolo_tracking"] = dict(locate_tracking_stage)
                stages["yolo_custom_anomaly"] = {
                    "status": "skipped",
                    "reason": "locate_anything_is_primary_detector",
                    "candidates": 0,
                }
                candidates = candidates_from_locate_observations(observations)
                if bool(self.config.get("locate_anything.required", False)) and not observations:
                    # In strict Locate-first research mode an empty detection
                    # artifact is a hard boundary: Qwen must never be started
                    # to fill a missing detector result with guesses.
                    raise PipelineError("locate_stage_output_empty_before_qwen")
                stages["candidate_selection"] = {
                    "raw_candidates": len(candidates),
                    "merged_candidates": len(candidates),
                    "merged_trigger_groups": 0,
                    "selected_candidates": len(candidates),
                    "raw_priority_candidates": 0,
                    "selected_priority_candidates": 0,
                    "priority_selection_policy": "frame_then_student_id",
                    "policy": "every_locate_screenshot_student_box",
                }
            elif checkpoint is not None:
                candidates = checkpoint["candidates"]
                scenes = checkpoint["scenes"]
                stages.update(checkpoint["stages"])
                stages["resume"] = {"status": "resumed", "checkpoint": checkpoint_path.as_posix()}
            else:
                candidates, scenes, stages["yolo_tracking"] = _track_pass(
                    self.config,
                    source,
                    store,
                    max_frames=max_frames,
                    tracker_factory=self.tracker_factory,
                )
                anomaly_candidates, stages["yolo_custom_anomaly"] = _custom_anomaly_pass(
                    self.config,
                    source,
                    store.run_dir / "artifacts/track_observations.jsonl",
                    store,
                    last_frame_index=int(stages["yolo_tracking"]["last_frame_index"]),
                    candidate_limit=int(self.config.get("triggering.max_candidates_per_video", 500)),
                    adapter_factory=self.anomaly_factory,
                )
                candidates.extend(anomaly_candidates)
                raw_candidates = list(candidates)
                raw_candidate_count = len(raw_candidates)
                merged_candidates, candidates, merged_trigger_groups = _merge_and_select_candidates(
                    raw_candidates,
                    merge_window_seconds=float(self.config.get("triggering.merge_window_seconds", 1.0)),
                    limit=int(self.config.get("triggering.max_candidates_per_video", 500)),
                    bucket_seconds=float(self.config.get("open_vocabulary.candidate_bucket_seconds", 300.0)),
                )
                is_priority = lambda candidate: (
                    "anomaly" in str(candidate.source).lower()
                    or "anomaly" in str(candidate.event_type).lower()
                    or float(candidate.metadata.get("priority", 0.0) or 0.0) > 0
                )
                candidates.sort(key=lambda candidate: (candidate.trigger_pts_seconds, candidate.event_id))
                stages["candidate_selection"] = {
                    "raw_candidates": raw_candidate_count,
                    "merged_candidates": len(merged_candidates),
                    "merged_trigger_groups": merged_trigger_groups,
                    "selected_candidates": len(candidates),
                    "raw_priority_candidates": sum(is_priority(candidate) for candidate in raw_candidates),
                    "selected_priority_candidates": sum(is_priority(candidate) for candidate in candidates),
                    "priority_selection_policy": "custom_anomaly_before_time_and_student_stratification",
                    "policy": "time_and_student_stratified",
                }
                store.write_json("artifacts/yolo_checkpoint.v1.json", {
                    "schema_version": "classroom_hybrid_ai.yolo_checkpoint.v1",
                    "run_digest": run_digest,
                    "source_path": str(source),
                    "config_digest": project_config_digest(self.config),
                    "max_frames": max_frames,
                    "candidates": [item.to_dict() for item in candidates],
                    "scenes": scenes,
                    "stages": {
                        "yolo_tracking": stages["yolo_tracking"],
                        "yolo_custom_anomaly": stages["yolo_custom_anomaly"],
                        "candidate_selection": stages["candidate_selection"],
                    },
                })
            if locate_first_mode:
                # v3 cap: one fresh request per selected student packet, with
                # a bounded long-course queue.  The exhaustive validation mode
                # still samples every ten frames, but it observes the same
                # hardware-safe upper bound.
                effective_students = max(1, len({candidate.track_id for candidate in candidates}))
                minutes = max(0.0, float(info.duration_seconds) / 60.0)
                configured_cap = min(240, max(48, 2 * effective_students, int(math.ceil(2.0 * minutes))))
                total_budget = min(len(candidates), configured_cap)
                discovery_budget, classification_budget = 0, total_budget
            else:
                total_budget, discovery_budget, classification_budget = _qwen_budget(info.duration_seconds, self.config)
            qwen_call_budget = _QwenCallBudget(total_budget)
            discovery_candidates = _fair_candidates(
                candidates,
                limit=min(discovery_budget, len(candidates)),
                bucket_seconds=float(self.config.get("open_vocabulary.candidate_bucket_seconds", 300.0)),
            )
            classification_candidates = _fair_candidates(
                candidates,
                limit=min(classification_budget, len(candidates)),
                bucket_seconds=float(self.config.get("open_vocabulary.candidate_bucket_seconds", 300.0)),
            )
            if locate_first_mode:
                classification_candidates.sort(
                    key=lambda candidate: (
                        candidate.trigger_frame_index,
                        student_id_sort_key(candidate.track_id),
                        candidate.event_id,
                    )
                )
            qwen_candidate_ids = {candidate.event_id for candidate in [*discovery_candidates, *classification_candidates]}
            evidence_candidates = [candidate for candidate in candidates if candidate.event_id in qwen_candidate_ids]
            if locate_first_mode:
                evidence_candidates.sort(
                    key=lambda candidate: (
                        candidate.trigger_frame_index,
                        student_id_sort_key(candidate.track_id),
                        candidate.event_id,
                    )
                )
            stages["qwen_budget"] = {
                "total_requests": total_budget,
                "discovery_request_limit": discovery_budget,
                "classification_request_limit": classification_budget,
                "semantic_adjudication_reserved_calls": total_budget - discovery_budget - classification_budget,
                "discovery_selected": len(discovery_candidates),
                "classification_selected": len(classification_candidates),
                "unreviewed_candidates": len(candidates) - len({candidate.event_id for candidate in classification_candidates}),
                "request_order": "frame_then_student_id" if locate_first_mode else "time_student_fair",
                "stateless_request_per_student_box": locate_first_mode,
            }
            if locate_first_mode:
                assert preloaded_locate_stage is not None and preloaded_locate_rows is not None
                stages["locate_anything"] = preloaded_locate_stage
                stages["locate_anything"]["anchor_selection"] = {
                    "requested_candidates": len(evidence_candidates),
                    "materialized_anchor_frames": len(scenes),
                    "reason": "fixed_stride_screenshots_before_tracking",
                }
                locate_rows = preloaded_locate_rows
            else:
                if skip_locate or not bool(self.config.get("locate_anything.enabled", False)):
                    locate_frames = list(scenes)
                    locate_anchor_stage = {
                        "requested_candidates": len(evidence_candidates),
                        "materialized_anchor_frames": 0,
                        "reason": "disabled_or_cli_skip",
                    }
                else:
                    locate_frames, locate_anchor_stage = _locate_anchor_frames(
                        self.config,
                        source,
                        evidence_candidates,
                        store,
                        fallback_scenes=scenes,
                    )
                stages["locate_anything"] = _locate_pass(
                    self.config,
                    locate_frames,
                    store,
                    skip=skip_locate,
                    adapter_factory=self.locate_factory,
                )
                stages["locate_anything"]["anchor_selection"] = locate_anchor_stage
                if stages["locate_anything"]["status"] == "warning":
                    locate_reason = str(stages["locate_anything"].get("reason") or "warning")
                    warnings.append(f"locate_anything_{locate_reason}")
                locate_rows = list(iter_jsonl(store.run_dir / "artifacts/locate_results.jsonl"))
            observation_path = store.run_dir / "artifacts/track_observations.jsonl"
            evidence_cfg = self.config.section("evidence")
            filtered = filter_observations_for_candidates(
                iter_jsonl(observation_path),
                evidence_candidates,
                before_seconds=float(evidence_cfg["before_seconds"]),
                after_seconds=float(evidence_cfg["after_seconds"]),
                include_all_tracks_in_windows=True,
            )
            track_to_student, _, _ = _student_catalog(
                observation_path,
                max_stitch_gap_seconds=float(self.config.get("tracking.max_stitch_gap_seconds", 2.0)),
                max_normalized_center_distance=float(self.config.get("tracking.max_normalized_center_distance", 0.6)),
                max_appearance_distance=float(self.config.get("tracking.max_appearance_distance", 0.35)),
                minimum_match_margin=float(self.config.get("tracking.minimum_match_margin", 0.15)),
            )
            filtered = [
                {
                    **row,
                    "student_id": track_to_student.get(
                        str(row.get("track_id", row.get("local_track_id", ""))),
                        str(row.get("track_id", row.get("local_track_id", ""))),
                    ),
                }
                for row in filtered
            ]
            evidence = build_event_evidence(
                video_path=source,
                candidates=evidence_candidates,
                observations=filtered,
                locate_rows=locate_rows,
                output_dir=store.evidence_dir,
                before_seconds=float(evidence_cfg["before_seconds"]),
                after_seconds=float(evidence_cfg["after_seconds"]),
                frames_per_event=int(evidence_cfg["frames_per_event"]),
                context_ratio=float(evidence_cfg["crop_context_ratio"]),
                maximum_crop_side=int(evidence_cfg["maximum_crop_side"]),
                maximum_context_side=int(evidence_cfg.get("maximum_context_side", 768)),
                jpeg_quality=int(evidence_cfg["jpeg_quality"]),
            )
            if locate_first_mode:
                evidence_by_event = {row["event_id"]: row for row in evidence}
                crop_queue: list[dict[str, Any]] = []
                for request_ordinal, candidate in enumerate(classification_candidates, start=1):
                    packet = evidence_by_event.get(candidate.event_id, {})
                    composite = build_temporal_composite(
                        packet,
                        output_path=(
                            store.evidence_dir
                            / candidate.event_id
                            / "temporal_composite.jpg"
                        ),
                        jpeg_quality=int(evidence_cfg["jpeg_quality"]),
                    )
                    if composite is not None and isinstance(packet, dict):
                        packet["qwen_composite"] = composite
                    subject_rows = [
                        item
                        for item in packet.get("evidence", ())
                        if isinstance(item, Mapping) and item.get("kind") == "subject_crop"
                    ]
                    crop_queue.append({
                        "request_ordinal": request_ordinal,
                        "frame_index": candidate.trigger_frame_index,
                        "pts_seconds": candidate.trigger_pts_seconds,
                        "student_id": candidate.track_id,
                        "candidate_event_id": candidate.event_id,
                        "status": "ready" if composite is not None else "insufficient_evidence",
                        "composite_path": None if composite is None else composite["path"],
                        "subject_crop_paths": [
                            str(item["path"])
                            for item in subject_rows
                            if item.get("path")
                        ],
                        "evidence_frame_indices": [] if composite is None else composite["frame_indices"],
                        "localization_sources": sorted({
                            str(item.get("localization_source")) for item in subject_rows
                        }),
                        "locate_detection_indices": sorted({
                            int(item["locate_detection_index"])
                            for item in subject_rows
                            if item.get("locate_detection_index") is not None
                        }),
                        "request_is_stateless": True,
                        "stateless_context_id": f"stateless:{candidate.event_id}",
                        "messages_in_request": 1,
                        "history_messages": 0,
                        "previous_student_context_included": False,
                    })
                store.write_jsonl("artifacts/qwen_crop_queue.jsonl", crop_queue)
                stages["student_crop_router"] = {
                    "status": "ok",
                    "provider": "locate_box_temporal_crop_router",
                    "queue_order": "frame_then_student_id",
                    "jobs": len(crop_queue),
                    "ready": sum(row["status"] == "ready" for row in crop_queue),
                    "artifact": "artifacts/qwen_crop_queue.jsonl",
                    "one_student_per_qwen_request": True,
                }
            store.write_jsonl("evidence/index.jsonl", evidence)
            locate_crop_count = sum(
                1
                for receipt in evidence
                for item in receipt.get("evidence", ())
                if isinstance(item, Mapping)
                and item.get("kind") == "subject_crop"
                and item.get("localization_source") == "locate_anything"
            )
            locate_event_count = sum(
                any(
                    isinstance(item, Mapping)
                    and item.get("kind") == "subject_crop"
                    and item.get("localization_source") == "locate_anything"
                    for item in receipt.get("evidence", ())
                )
                for receipt in evidence
            )
            stages["evidence"] = {
                "status": "ok",
                "events": len(evidence),
                "insufficient": sum(row["status"] != "ok" for row in evidence),
                "filtered_observations_in_memory": len(filtered),
                "locate_localized_events": locate_event_count,
                "locate_localized_subject_crops": locate_crop_count,
                "locate_event_coverage": (
                    locate_event_count / len(evidence) if evidence else 0.0
                ),
            }
            qwen_adapter = None if (
                skip_qwen
                or not bool(self.config.get("qwen.enabled", True))
                or not candidates
            ) else _make_qwen_adapter(
                self.config, adapter_factory=self.qwen_factory,
            )
            try:
                discovery_reviews, stages["qwen_discovery"] = _qwen_pass(
                    self.config,
                    discovery_candidates,
                    evidence,
                    store,
                    skip=skip_qwen,
                    prompt=_discovery_prompt(self.action_profile),
                    action_profile=self.action_profile,
                    response_prefix="responses/discovery",
                    locate_rows=locate_rows,
                    response_contract="discovery",
                    adapter=qwen_adapter,
                    call_budget=qwen_call_budget,
                    release_adapter=False,
                    review_output="reviews.discovery.jsonl",
                    adapter_factory=self.qwen_factory,
                )
                discovery_circuit_open = bool(
                    stages["qwen_discovery"].get("load_circuit_open")
                    or stages["qwen_discovery"].get("oom_circuit_open")
                )
                if discovery_circuit_open:
                    # A failed model load or an OOM is a phase-level circuit:
                    # do not retry warmup, semantic adjudication or
                    # classification in the same video.  Preserve the current
                    # registry and emit explicit gray records instead.
                    registry = load_latest(self.config.action_registry_root)
                    vocabulary = ()
                    store.write_jsonl("artifacts/open_vocabulary_discoveries.jsonl", [])
                    circuit_reason = (
                        "qwen_oom_circuit_open"
                        if stages["qwen_discovery"].get("oom_circuit_open")
                        else "qwen_load_circuit_open"
                    )
                    stages["semantic_canonicalization"] = {
                        "status": "skipped",
                        "reason": circuit_reason,
                        "phrases": 0,
                        "merged": 0,
                    }
                    reviews = [ReviewDecision(
                        candidate_event_id=candidate.event_id,
                        decision="uncertain",
                        reviewed_event_type=candidate.event_type,
                        rationale=circuit_reason,
                        evidence_frame_indices=(),
                        reviewer="pipeline_circuit",
                        schema_valid=False,
                        metadata={"circuit_open": True},
                    ) for candidate in classification_candidates]
                    store.write_jsonl("reviews.classification.jsonl", reviews)
                    stages["qwen_classification"] = {
                        "status": "skipped",
                        "reason": circuit_reason,
                        "reviews": len(reviews),
                    }
                else:
                    registry, vocabulary, stages["semantic_canonicalization"] = _prepare_session_vocabulary(
                        self.config,
                        store,
                        discovery_reviews=discovery_reviews,
                        candidates=discovery_candidates,
                        action_profile=self.action_profile,
                        qwen_adapter=qwen_adapter,
                        qwen_call_budget=qwen_call_budget,
                    )
                    # Use the exact digest that will be written to
                    # session_vocabulary.v1.json (without the self-referential
                    # ``sha256`` field).  Classification responses can therefore
                    # be joined to the immutable vocabulary artifact directly.
                    vocabulary_sha256 = session_vocabulary_digest(
                        vocabulary,
                        registry_sha256=registry.sha256,
                    )
                    if vocabulary:
                        reviews, stages["qwen_classification"] = _qwen_pass(
                            self.config,
                            classification_candidates,
                            evidence,
                            store,
                            skip=skip_qwen,
                            prompt=_classification_prompt(
                                vocabulary,
                                vocabulary_sha256,
                                action_profile=self.action_profile,
                                prompt_version=str(self.config.get("qwen.prompt_version", "legacy_v1")),
                            ),
                            response_prefix="responses/classification",
                            expected_vocabulary_sha256=vocabulary_sha256,
                            allowed_action_ids={item.action_id for item in vocabulary},
                            action_profile=self.action_profile,
                            locate_rows=locate_rows,
                            adapter=qwen_adapter,
                            call_budget=qwen_call_budget,
                            release_adapter=False,
                            review_output="reviews.classification.jsonl",
                            adapter_factory=self.qwen_factory,
                        )
                    else:
                        reviews = [ReviewDecision(
                            candidate_event_id=candidate.event_id,
                            decision="uncertain",
                            reviewed_event_type=candidate.event_type,
                            rationale="session_vocabulary_empty",
                            evidence_frame_indices=(),
                            reviewer="pipeline",
                            schema_valid=False,
                        ) for candidate in classification_candidates]
                        store.write_jsonl("reviews.classification.jsonl", reviews)
                        stages["qwen_classification"] = {"status": "skipped", "reason": "session_vocabulary_empty", "reviews": len(reviews)}

                # Every merged candidate must have an auditable classification
                # record, even when the fair queue or the request budget did
                # not select it.  Leaving these IDs out of the JSONL would
                # make a gray timeline gap look like a missing pipeline row.
                reviewed_ids = {review.candidate_event_id for review in reviews}
                unreviewed_reviews: list[ReviewDecision] = []
                for candidate in candidates:
                    if candidate.event_id in reviewed_ids:
                        continue
                    unreviewed = ReviewDecision(
                        candidate_event_id=candidate.event_id,
                        decision="uncertain",
                        reviewed_event_type=candidate.event_type,
                        rationale="qwen_candidate_unreviewed",
                        evidence_frame_indices=(),
                        reviewer="pipeline_budget",
                        schema_valid=False,
                        metadata={"unreviewed": True, "reason": "qwen_candidate_unreviewed"},
                    )
                    unreviewed_reviews.append(unreviewed)
                    store.write_json(
                        f"responses/classification/{candidate.event_id}.json",
                        {
                            "candidate_event_id": candidate.event_id,
                            "status": "gray",
                            "reason": "qwen_candidate_unreviewed",
                            "stateless_request": False,
                        },
                    )
                if unreviewed_reviews:
                    reviews.extend(unreviewed_reviews)
                    store.write_jsonl("reviews.classification.jsonl", reviews)
                stages["qwen_classification"].update({
                    "reviewed_candidates": len(reviewed_ids),
                    "unreviewed_candidates": len(unreviewed_reviews),
                    "total_review_records": len(reviews),
                })
            finally:
                if qwen_adapter is None:
                    stages["qwen_release"] = {"released": False, "reason": "skipped"}
                else:
                    worker_pid = getattr(qwen_adapter, "pid", None)
                    worker_isolated = isinstance(qwen_adapter, QwenProcessClient)
                    release_receipt = release_model(qwen_adapter)
                    release_receipt.update({
                        "worker_isolated": worker_isolated,
                        "worker_pid": worker_pid,
                        "worker_exitcode": getattr(qwen_adapter, "exitcode", None),
                    })
                    stages["qwen_release"] = release_receipt
            stages["qwen_budget"].update({
                "actual_calls": qwen_call_budget.used,
                "semantic_adjudication_calls": qwen_call_budget.semantic_calls,
                "remaining_calls": qwen_call_budget.remaining,
                "budget_exhausted_reviews": int(stages["qwen_discovery"].get("budget_exhausted", 0))
                + int(stages["qwen_classification"].get("budget_exhausted", 0)),
            })
        if locate_first_mode:
            action_lookup = {item.action_id: item for item in vocabulary}
            review_lookup = {item.candidate_event_id: item for item in reviews}
            student_actions: list[dict[str, Any]] = []
            for request_ordinal, candidate in enumerate(classification_candidates, start=1):
                review = review_lookup.get(candidate.event_id)
                descriptor = (
                    action_lookup.get(str(review.behavior))
                    if review is not None and review.decision == "confirmed"
                    else None
                )
                student_actions.append({
                    "request_ordinal": request_ordinal,
                    "frame_index": candidate.trigger_frame_index,
                    "pts_seconds": candidate.trigger_pts_seconds,
                    "student_id": candidate.track_id,
                    "candidate_event_id": candidate.event_id,
                    "decision": None if review is None else review.decision,
                    "action_id": None if descriptor is None else descriptor.action_id,
                    "action_code": (
                        None
                        if descriptor is None
                        else descriptor.attributes.get("fixed_eight_code")
                    ),
                    "action_name_zh": None if descriptor is None else descriptor.name_zh,
                    "color": "#B8B8B8" if descriptor is None else descriptor.color,
                    "schema_valid": False if review is None else review.schema_valid,
                    "request_is_stateless": True,
                    "history_messages": 0,
                    "cross_request_kv_cache_reused": False,
                })
            store.write_jsonl("artifacts/student_actions.jsonl", student_actions)
            stages["student_action_output"] = {
                "status": "ok",
                "rows": len(student_actions),
                "artifact": "artifacts/student_actions.jsonl",
                "order": "frame_then_student_id",
                "fixed_action_count": len(vocabulary),
            }
        policy = StrictFusionPolicy(min_evidence_frames=1)
        reviews_by_id = {review.candidate_event_id: review for review in reviews}
        final_events = [policy.fuse(candidate, reviews_by_id.get(candidate.event_id)) for candidate in candidates]
        store.write_jsonl("candidates.jsonl", candidates)
        store.write_jsonl("events.jsonl", final_events)
        stages["open_vocabulary"] = _write_open_vocab_outputs(
            self.config,
            store,
            discovery_reviews=discovery_reviews,
            classification_reviews=reviews,
            candidates=candidates,
            registry=registry,
            vocabulary=vocabulary,
            duration_seconds=info.duration_seconds,
            video=info.to_dict(),
        )
        counts = {
            "observations": int(stages["yolo_tracking"]["observations"]),
            "candidates": len(candidates),
            "confirmed": sum(event.status == "confirmed" for event in final_events),
            "rejected": sum(event.status == "rejected" for event in final_events),
            "uncertain": sum(event.status == "uncertain" for event in final_events),
            "students": int(stages["open_vocabulary"]["students"]),
            "timeline_intervals": int(stages["open_vocabulary"]["intervals"]),
        }
        result = PipelineResult(
            status="warning" if warnings or stages["qwen_discovery"].get("failures") or stages["qwen_classification"].get("failures") else "ok",
            run_dir=str(store.run_dir),
            video=info.to_dict(),
            counts=counts,
            stages=stages,
            warnings=tuple(warnings),
        )
        store.write_json("manifest.json", {
            "schema_version": "classroom_hybrid_ai.run.v1",
            **result.to_dict(),
            "config_source": str(self.config.source_path),
            "config_digest": project_config_digest(self.config),
            "source_sha256": None,
            "source_hash_note": "omitted_by_default_to_avoid_an_extra_full_read_of_long_video",
        })
        return result


__all__ = ["HybridPipeline", "PipelineError", "PipelineResult"]
