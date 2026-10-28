"""Conservative same-camera tracklet stitching and fair review scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .contracts import TrackObservation


@dataclass(frozen=True)
class TrackletSummary:
    track_id: str
    first_pts_seconds: float
    last_pts_seconds: float
    center_x: float
    center_y: float
    scale: float
    appearance: tuple[float, ...] = ()


def summarize_tracklet(rows: Sequence[TrackObservation]) -> TrackletSummary:
    if not rows:
        raise ValueError("tracklet_empty")
    ordered = sorted(rows, key=lambda item: item.pts_seconds)
    centers = []
    scales = []
    for item in ordered:
        x1, y1, x2, y2 = item.detection.bbox
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        scales.append(max(1.0, ((x2 - x1) * (y2 - y1)) ** 0.5))
    midpoint = len(centers) // 2
    return TrackletSummary(
        track_id=ordered[0].track_id,
        first_pts_seconds=ordered[0].pts_seconds,
        last_pts_seconds=ordered[-1].pts_seconds,
        center_x=sorted(point[0] for point in centers)[midpoint],
        center_y=sorted(point[1] for point in centers)[midpoint],
        scale=sorted(scales)[len(scales) // 2],
        appearance=_median_appearance(ordered),
    )


def stitch_score(
    left: TrackletSummary,
    right: TrackletSummary,
    *,
    max_gap_seconds: float = 2.0,
    max_normalized_center_distance: float = 0.6,
    max_size_delta: float = 0.5,
    max_appearance_distance: float = 0.35,
) -> float | None:
    """Return a comparable score or ``None`` when joining is forbidden.

    The defaults mirror the conservative policy in the project specification.
    They are keyword-only so callers can expose the policy in a run manifest
    without changing the compact positional API used by older callers.
    """

    for value, name in (
        (max_gap_seconds, "max_gap_seconds"),
        (max_normalized_center_distance, "max_normalized_center_distance"),
        (max_size_delta, "max_size_delta"),
        (max_appearance_distance, "max_appearance_distance"),
    ):
        if float(value) < 0:
            raise ValueError(f"{name} must be non-negative")

    gap = right.first_pts_seconds - left.last_pts_seconds
    if gap < 0.0 or gap > float(max_gap_seconds):
        return None
    scale = max(1.0, (left.scale + right.scale) / 2.0)
    distance = (((left.center_x - right.center_x) ** 2 + (left.center_y - right.center_y) ** 2) ** 0.5) / scale
    if distance > float(max_normalized_center_distance):
        return None
    size_delta = abs(left.scale - right.scale) / scale
    if size_delta > float(max_size_delta):
        return None
    appearance_distance = _appearance_distance(left.appearance, right.appearance)
    if appearance_distance is not None and appearance_distance > float(max_appearance_distance):
        return None
    return 1.0 - min(1.0, 0.55 * distance + 0.20 * size_delta + 0.25 * (appearance_distance or 0.0))


def stitch_tracklets(
    tracklets: Sequence[TrackletSummary],
    *,
    max_gap_seconds: float = 2.0,
    max_normalized_center_distance: float = 0.6,
    max_size_delta: float = 0.5,
    max_appearance_distance: float = 0.35,
    minimum_match_margin: float = 0.15,
) -> Mapping[str, str]:
    """Map raw tracklets to conservative stable IDs.

    A link needs a best-versus-runner-up margin of 0.15, so dense overlapping
    students are left separate rather than silently joined to the wrong person.
    """

    ordered = sorted(tracklets, key=lambda item: (item.first_pts_seconds, item.track_id))
    assigned: dict[str, str] = {}
    counter = 0
    for current in ordered:
        candidates = []
        for previous in ordered:
            if previous.track_id == current.track_id or previous.track_id not in assigned:
                continue
            score = stitch_score(
                previous,
                current,
                max_gap_seconds=max_gap_seconds,
                max_normalized_center_distance=max_normalized_center_distance,
                max_size_delta=max_size_delta,
                max_appearance_distance=max_appearance_distance,
            )
            if score is not None:
                candidates.append((score, previous))
        candidates.sort(key=lambda item: (-item[0], item[1].track_id))
        if candidates and (
            len(candidates) == 1
            or candidates[0][0] - candidates[1][0] >= float(minimum_match_margin)
        ):
            assigned[current.track_id] = assigned[candidates[0][1].track_id]
        else:
            counter += 1
            assigned[current.track_id] = f"student-{counter:03d}"
    return assigned


def fair_time_stratified_selection(
    candidates: Sequence[Mapping[str, object]],
    *,
    limit: int,
    bucket_seconds: float = 300.0,
) -> list[Mapping[str, object]]:
    """Select candidates across time and people without hiding anomalies.

    ``source``/``event_type`` values containing ``custom_anomaly`` or
    ``anomaly`` are treated as high-priority review signals.  They are selected
    before ordinary periodic/audit rows, so a long video cannot consume the
    entire cap with early periodic candidates.  Remaining capacity is filled
    with one row per time bucket and track before score ordering is used.
    """

    if limit <= 0:
        return []
    if float(bucket_seconds) <= 0:
        raise ValueError("bucket_seconds must be positive")
    chosen: list[Mapping[str, object]] = []
    seen: set[tuple[int, str]] = set()
    selected_ids: set[str] = set()
    ordered = sorted(
        candidates,
        key=lambda row: (float(row.get("pts_seconds", 0.0)), str(row.get("track_id", "")), str(row.get("event_id", ""))),
    )
    def priority(row: Mapping[str, object]) -> tuple[int, float, float, str, str]:
        source = str(row.get("source", "")).lower()
        event_type = str(row.get("event_type", "")).lower()
        metadata = row.get("metadata")
        metadata_priority = 0.0
        if isinstance(metadata, Mapping):
            try:
                metadata_priority = float(metadata.get("priority", 0.0) or 0.0)
            except (TypeError, ValueError):
                metadata_priority = 0.0
        high_priority = int(
            metadata_priority > 0
            or "custom_anomaly" in source
            or "custom_anomaly" in event_type
            or "anomaly" in source
            or "anomaly" in event_type
        )
        return (
            high_priority,
            metadata_priority,
            float(row.get("score") or 0.0),
            -float(row.get("pts_seconds", 0.0)),
            str(row.get("event_id", "")),
        )

    def bucket_key(row: Mapping[str, object]) -> tuple[int, str]:
        return (
            int(float(row.get("pts_seconds", 0.0)) // bucket_seconds),
            str(row.get("track_id", "")),
        )

    def round_robin(rows: Sequence[Mapping[str, object]], *, high_priority: bool) -> list[Mapping[str, object]]:
        """Take one row per student/time bucket per round, including late video buckets."""

        grouped: dict[int, dict[str, list[Mapping[str, object]]]] = {}
        for row in rows:
            bucket, track = bucket_key(row)
            grouped.setdefault(bucket, {}).setdefault(track, []).append(row)
        for tracks in grouped.values():
            for track_rows in tracks.values():
                track_rows.sort(
                    key=(
                        (lambda item: priority(item)) if high_priority
                        else (lambda item: (float(item.get("pts_seconds", 0.0)), str(item.get("event_id", ""))))
                    ),
                    reverse=high_priority,
                )
        output: list[Mapping[str, object]] = []
        for round_index in range(max((len(rows) for tracks in grouped.values() for rows in tracks.values()), default=0)):
            for bucket in sorted(grouped):
                for track in sorted(grouped[bucket]):
                    track_rows = grouped[bucket][track]
                    if round_index < len(track_rows):
                        output.append(track_rows[round_index])
        return output

    # Reserve the cap for verified/custom anomaly signals first.  If there are
    # more anomaly rows than the cap, preserve late buckets and students before
    # falling back to priority/score ordering.
    priority_rows = sorted(
        [row for row in ordered if priority(row)[0]],
        key=priority,
        reverse=True,
    )
    for row in round_robin(priority_rows, high_priority=True):
        identifier = str(row.get("event_id", id(row)))
        if identifier in selected_ids:
            continue
        chosen.append(row)
        selected_ids.add(identifier)
        seen.add((int(float(row.get("pts_seconds", 0.0)) // bucket_seconds), str(row.get("track_id", ""))))
        if len(chosen) == limit:
            return chosen

    for row in priority_rows:
        identifier = str(row.get("event_id", id(row)))
        if identifier in selected_ids:
            continue
        chosen.append(row)
        selected_ids.add(identifier)
        if len(chosen) == limit:
            return chosen

    ordinary_rows = [row for row in ordered if not priority(row)[0]]
    for row in round_robin(ordinary_rows, high_priority=False):
        key = bucket_key(row)
        if key not in seen:
            identifier = str(row.get("event_id", id(row)))
            if identifier in selected_ids:
                continue
            chosen.append(row)
            selected_ids.add(identifier)
            seen.add(key)
            if len(chosen) == limit:
                return chosen
    for row in sorted(ordered, key=lambda item: (-float(item.get("score") or 0.0), float(item.get("pts_seconds", 0.0)))):
        identifier = str(row.get("event_id", id(row)))
        if identifier in selected_ids:
            continue
        chosen.append(row)
        selected_ids.add(identifier)
        if len(chosen) == limit:
            break
    return chosen


def _median_appearance(rows: Sequence[TrackObservation]) -> tuple[float, ...]:
    values = []
    for item in rows:
        appearance = item.metadata.get("hsv_histogram") if isinstance(item.metadata, Mapping) else None
        if isinstance(appearance, (tuple, list)) and appearance:
            values.append(tuple(float(value) for value in appearance))
    if not values:
        return ()
    width = min(len(value) for value in values)
    return tuple(sorted(value[index] for value in values)[len(values) // 2] for index in range(width))


def _appearance_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if not left or not right:
        return None
    width = min(len(left), len(right))
    if width == 0:
        return None
    return sum(abs(left[index] - right[index]) for index in range(width)) / width


__all__ = ["TrackletSummary", "fair_time_stratified_selection", "stitch_score", "stitch_tracklets", "summarize_tracklet"]
