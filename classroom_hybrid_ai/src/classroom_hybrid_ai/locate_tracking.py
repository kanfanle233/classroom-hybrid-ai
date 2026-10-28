"""LocateAnything-first screenshot sampling and persistent student ID association."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2

from .contracts import CandidateEvent, Detection, TrackObservation
from .hashing import canonical_sha256
from .video import iter_video


STUDENT_LABELS = frozenset({"student", "person", "pupil", "learner"})


def student_id_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    suffix = text.rsplit("-", 1)[-1]
    return (int(suffix), text) if suffix.isdigit() else (10**9, text)


def materialize_locate_screenshots(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    frame_stride: int,
    max_frames: int | None = None,
    maximum_screenshots: int | None = None,
    jpeg_quality: int = 95,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Save every configured screenshot before any detector or Qwen call."""

    if frame_stride < 1:
        raise ValueError("locate_screenshot_frame_stride_must_be_positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames_must_be_positive")
    if maximum_screenshots is not None and maximum_screenshots < 1:
        raise ValueError("maximum_screenshots_must_be_positive_or_null")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    decoded = 0
    truncated = False
    for frame in iter_video(video_path):
        if max_frames is not None and frame.frame_index >= max_frames:
            break
        decoded += 1
        if frame.frame_index % frame_stride != 0:
            continue
        if maximum_screenshots is not None and len(rows) >= maximum_screenshots:
            truncated = True
            break
        path = destination / f"frame_{frame.frame_index:09d}.jpg"
        if not cv2.imwrite(str(path), frame.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            raise OSError(f"locate_screenshot_write_failed:{path}")
        rows.append({
            "path": str(path),
            "frame_index": frame.frame_index,
            "pts_seconds": frame.pts_seconds,
            "anchor_source": "fixed_frame_stride",
        })
    return rows, {
        "status": "warning" if truncated else "ok",
        "frame_stride": frame_stride,
        "decoded_frames": decoded,
        "screenshots": len(rows),
        "maximum_screenshots": maximum_screenshots,
        "truncated": truncated,
    }


def materialize_scheduled_screenshots(
    video_path: str | Path,
    output_dir: str | Path,
    schedule: Iterable[Mapping[str, Any]],
    *,
    max_frames: int | None = None,
    jpeg_quality: int = 95,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode once and save the exact frame indices chosen by the v3 scheduler."""

    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames_must_be_positive")
    requested: dict[int, dict[str, Any]] = {}
    for raw in schedule:
        frame_index = int(raw.get("frame_index", -1))
        if frame_index < 0 or (max_frames is not None and frame_index >= max_frames):
            continue
        requested.setdefault(frame_index, dict(raw))
    if not requested:
        raise ValueError("scheduled_screenshots_empty")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    remaining = set(requested)
    rows: list[dict[str, Any]] = []
    decoded = 0
    if progress_callback is not None:
        progress_callback(0, len(requested))
    for frame in iter_video(video_path):
        if max_frames is not None and frame.frame_index >= max_frames:
            break
        decoded += 1
        if frame.frame_index not in remaining:
            continue
        requested_row = requested[frame.frame_index]
        path = destination / f"frame_{frame.frame_index:09d}.jpg"
        if not cv2.imwrite(str(path), frame.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
            raise OSError(f"locate_screenshot_write_failed:{path}")
        rows.append({
            **requested_row,
            "path": str(path),
            "frame_index": int(frame.frame_index),
            "pts_seconds": float(frame.pts_seconds),
            "scheduled_pts_seconds": float(requested_row.get("pts_seconds", frame.pts_seconds)),
        })
        remaining.remove(frame.frame_index)
        if progress_callback is not None:
            progress_callback(len(rows), len(requested))
        if not remaining:
            break
    rows.sort(key=lambda item: int(item["frame_index"]))
    if not rows:
        raise RuntimeError("scheduled_screenshot_decode_produced_no_frames")
    return rows, {
        "status": "warning" if remaining else "ok",
        "requested": len(requested),
        "screenshots": len(rows),
        "decoded_frames": decoded,
        "missing_frame_indices": sorted(remaining),
        "exact_schedule_applied": True,
    }


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_center_distance(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    distance = math.hypot((ax1 + ax2 - bx1 - bx2) / 2.0, (ay1 + ay2 - by1 - by2) / 2.0)
    scale = max(
        1.0,
        0.5 * (math.hypot(ax2 - ax1, ay2 - ay1) + math.hypot(bx2 - bx1, by2 - by1)),
    )
    return distance / scale


def _association_score(first: Sequence[float], second: Sequence[float]) -> tuple[float, float, float]:
    """Return a bounded spatial score plus its auditable components."""

    overlap = _iou(first, second)
    center_distance = _normalized_center_distance(first, second)
    # A reciprocal distance term remains informative after a long occlusion,
    # unlike ``max(0, 1-distance)`` which collapses every distant seat to the
    # same zero score.  IoU still dominates normal consecutive-frame matches.
    score = 0.65 * overlap + 0.35 / (1.0 + center_distance)
    return score, overlap, center_distance


def _maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> list[int]:
    """Solve a rectangular maximum-weight assignment without SciPy.

    The Hungarian implementation expects at least as many columns as rows.
    Callers add one private dummy column per detection so every row can be
    assigned deterministically while real tracks remain one-to-one.
    """

    if not weights:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if column_count < row_count or any(len(row) != column_count for row in weights):
        raise ValueError("assignment_matrix_must_be_rectangular_with_columns_ge_rows")
    # Hungarian algorithm for minimum cost; negating weights maximizes them.
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        column0 = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column0] = True
            current_row = matched_row[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = -float(weights[current_row - 1][column - 1]) - u[current_row] - v[column]
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("assignment_solver_left_unmatched_row")
    return assignment


def _blend_bbox(
    first: Sequence[float], second: Sequence[float], *, alpha: float
) -> tuple[float, float, float, float]:
    return tuple(
        (1.0 - alpha) * float(old) + alpha * float(new)
        for old, new in zip(first, second)
    )  # type: ignore[return-value]


@dataclass(slots=True)
class _TrackState:
    student_id: str
    bbox: tuple[float, float, float, float]
    anchor_bbox: tuple[float, float, float, float]
    frame_index: int
    pts_seconds: float
    observations: int = 1


class LocateStudentIdAssigner:
    """Associate Locate boxes across screenshots without inventing model IDs.

    IDs are scoped to one fixed-camera video run.  Ambiguous observations keep
    the globally optimal seat ID and carry an explicit uncertainty flag so
    Qwen can skip them; ambiguity must not recursively manufacture new IDs.
    """

    def __init__(
        self,
        *,
        max_gap_seconds: float = 2.0,
        minimum_iou: float = 0.10,
        maximum_center_distance: float = 0.75,
        minimum_match_score: float = 0.30,
        minimum_match_margin: float = 0.10,
    ) -> None:
        self.max_gap_seconds = float(max_gap_seconds)
        self.minimum_iou = float(minimum_iou)
        self.maximum_center_distance = float(maximum_center_distance)
        self.minimum_match_score = float(minimum_match_score)
        self.minimum_match_margin = float(minimum_match_margin)
        self._states: dict[str, _TrackState] = {}
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"student-{self._counter:03d}"

    def update(
        self,
        *,
        frame_index: int,
        pts_seconds: float,
        detections: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        students: list[dict[str, Any]] = []
        for fallback_index, raw in enumerate(detections, start=1):
            if str(raw.get("label", "")).strip().lower() not in STUDENT_LABELS:
                continue
            bbox = raw.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            values = tuple(float(value) for value in bbox)
            if values[2] <= values[0] or values[3] <= values[1]:
                continue
            students.append({
                **dict(raw),
                "bbox": list(values),
                "locate_detection_index": int(raw.get("detection_index", fallback_index)),
            })
        students.sort(key=lambda row: (
            (row["bbox"][1] + row["bbox"][3]) / 2.0,
            (row["bbox"][0] + row["bbox"][2]) / 2.0,
            row["locate_detection_index"],
        ))

        had_tracks = bool(self._states)
        track_ids = sorted(self._states, key=student_id_sort_key)
        pair_details: list[list[dict[str, Any]]] = []
        real_weights: list[list[float]] = []
        for detection in students:
            detail_row: list[dict[str, Any]] = []
            weight_row: list[float] = []
            for track_id in track_ids:
                state = self._states[track_id]
                gap = max(0.0, float(pts_seconds) - state.pts_seconds)
                recent = _association_score(detection["bbox"], state.bbox)
                anchor = _association_score(detection["bbox"], state.anchor_bbox)
                # Identity in the frozen-camera classroom is anchored to the
                # student's seat.  Using the latest box as the primary key lets
                # a walking teacher or one bad detection push an entire row of
                # IDs sideways.  Keep the recent score only as a diagnostic;
                # the slowly updated seat anchor controls the assignment.
                score, overlap, center_distance = anchor
                source = "seat_anchor"
                detail_row.append({
                    "score": score,
                    "overlap": overlap,
                    "center_distance": center_distance,
                    "recent_score": recent[0],
                    "gap_seconds": gap,
                    "source": source,
                })
                weight_row.append(score)
            pair_details.append(detail_row)
            real_weights.append(weight_row)

        # A very low dummy score permits a genuinely new spatial region while
        # preferring auditable uncertain reassociation over ID fragmentation.
        dummy_score = min(0.02, max(0.0, self.minimum_match_score * 0.05))
        weights = [
            [*row, *([dummy_score] * len(students))]
            for row in real_weights
        ]
        assigned_columns = _maximum_weight_assignment(weights) if students else []

        output: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(students):
            assigned_column = assigned_columns[detection_index]
            if assigned_column >= len(track_ids):
                student_id, score, method = self._new_id(), None, "new_track"
                margin = None
                track_margin = None
                ambiguous = had_tracks
                ambiguity_reason = "unmatched_new_track" if had_tracks else None
                gap_seconds = None
            else:
                student_id = track_ids[assigned_column]
                detail = pair_details[detection_index][assigned_column]
                score = float(detail["score"])
                method = (
                    "seat_anchor_match"
                    if float(detail["gap_seconds"]) <= self.max_gap_seconds
                    else "seat_anchor_reacquisition"
                )
                alternatives = sorted(real_weights[detection_index], reverse=True)
                best_other = next(
                    (
                        real_weights[detection_index][column]
                        for column in sorted(
                            range(len(track_ids)),
                            key=lambda column: real_weights[detection_index][column],
                            reverse=True,
                        )
                        if column != assigned_column
                    ),
                    0.0,
                )
                margin = score - float(best_other)
                competing_detections = [
                    real_weights[row][assigned_column]
                    for row in range(len(students))
                    if row != detection_index
                ]
                track_margin = score - max(competing_detections, default=0.0)
                spatial_gate_failed = (
                    float(detail["overlap"]) < self.minimum_iou
                    and float(detail["center_distance"]) > self.maximum_center_distance
                )
                ambiguous_reasons: list[str] = []
                if score < self.minimum_match_score:
                    ambiguous_reasons.append("low_match_score")
                if margin < self.minimum_match_margin:
                    ambiguous_reasons.append("detection_margin")
                if track_margin < self.minimum_match_margin:
                    ambiguous_reasons.append("track_margin")
                if spatial_gate_failed:
                    ambiguous_reasons.append("outside_recent_spatial_gate")
                ambiguous = bool(ambiguous_reasons)
                ambiguity_reason = ",".join(ambiguous_reasons) or None
                gap_seconds = float(detail["gap_seconds"])
            bbox_tuple = tuple(float(value) for value in detection["bbox"])
            prior = self._states.get(student_id)
            if prior is None:
                anchor_bbox = bbox_tuple
                observations = 1
            else:
                observations = prior.observations + 1
                alpha = 1.0 / float(min(observations, 20))
                anchor_bbox = (
                    prior.anchor_bbox
                    if ambiguous
                    else _blend_bbox(prior.anchor_bbox, bbox_tuple, alpha=alpha)
                )
            self._states[student_id] = _TrackState(
                student_id, bbox_tuple, anchor_bbox, int(frame_index), float(pts_seconds), observations
            )
            output.append({
                **detection,
                "student_id": student_id,
                "association_score": score,
                "association_method": method,
                "association_margin": margin,
                "association_track_margin": track_margin,
                "association_ambiguous": ambiguous,
                "association_ambiguity_reason": ambiguity_reason,
                "association_gap_seconds": gap_seconds,
                "id_scope": "single_camera_video_run",
            })
        output.sort(key=lambda row: student_id_sort_key(str(row["student_id"])))
        return output


def associate_locate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_gap_seconds: float,
    minimum_iou: float,
    maximum_center_distance: float,
    minimum_match_score: float,
    minimum_match_margin: float,
) -> tuple[list[TrackObservation], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assigner = LocateStudentIdAssigner(
        max_gap_seconds=max_gap_seconds,
        minimum_iou=minimum_iou,
        maximum_center_distance=maximum_center_distance,
        minimum_match_score=minimum_match_score,
        minimum_match_margin=minimum_match_margin,
    )
    observations: list[TrackObservation] = []
    enriched_rows: list[dict[str, Any]] = []
    flat_boxes: list[dict[str, Any]] = []
    method_counts: dict[str, int] = {}
    ambiguous_associations = 0
    maximum_frame_students = 0
    for row in sorted(rows, key=lambda item: (int(item.get("frame_index", 0)), float(item.get("pts_seconds", 0.0)))):
        enriched = dict(row)
        if row.get("status") != "ok":
            enriched_rows.append(enriched)
            continue
        assigned = assigner.update(
            frame_index=int(row["frame_index"]),
            pts_seconds=float(row["pts_seconds"]),
            detections=[item for item in row.get("detections", ()) if isinstance(item, Mapping)],
        )
        maximum_frame_students = max(maximum_frame_students, len(assigned))
        non_students = [
            dict(item)
            for item in row.get("detections", ())
            if isinstance(item, Mapping)
            and str(item.get("label", "")).strip().lower() not in STUDENT_LABELS
        ]
        enriched["detections"] = [*assigned, *non_students]
        enriched["student_ids_assigned"] = True
        enriched_rows.append(enriched)
        for detection in assigned:
            attributes = dict(detection.get("attributes") or {})
            attributes.update({
                "locate_detection_index": detection["locate_detection_index"],
                "association_score": detection["association_score"],
                "association_method": detection["association_method"],
                "association_margin": detection["association_margin"],
                "association_track_margin": detection["association_track_margin"],
                "association_ambiguous": detection["association_ambiguous"],
                "association_ambiguity_reason": detection["association_ambiguity_reason"],
                "association_gap_seconds": detection["association_gap_seconds"],
                "id_scope": detection["id_scope"],
            })
            method = str(detection["association_method"])
            method_counts[method] = method_counts.get(method, 0) + 1
            ambiguous_associations += int(bool(detection["association_ambiguous"]))
            observation = TrackObservation(
                track_id=str(detection["student_id"]),
                frame_index=int(row["frame_index"]),
                pts_seconds=float(row["pts_seconds"]),
                detection=Detection(
                    label="student",
                    bbox=tuple(float(value) for value in detection["bbox"]),
                    score=(
                        float(detection["score"])
                        if isinstance(detection.get("score"), (int, float))
                        else None
                    ),
                    source="locate_anything",
                    attributes=attributes,
                ),
                metadata={
                    "identity_provider": "locate_box_association",
                    "locate_frame_detection_id": f"L{detection['locate_detection_index']}",
                    "association_ambiguous": bool(detection["association_ambiguous"]),
                    "id_uncertain": bool(detection["association_ambiguous"]),
                },
            )
            observations.append(observation)
            flat_boxes.append({
                "frame_index": observation.frame_index,
                "pts_seconds": observation.pts_seconds,
                "student_id": observation.track_id,
                "bbox": list(observation.detection.bbox),
                "locate_detection_index": detection["locate_detection_index"],
                "association_score": detection["association_score"],
                "association_method": detection["association_method"],
                "association_margin": detection["association_margin"],
                "association_track_margin": detection["association_track_margin"],
                "association_ambiguous": detection["association_ambiguous"],
                "association_ambiguity_reason": detection["association_ambiguity_reason"],
                "association_gap_seconds": detection["association_gap_seconds"],
                "id_scope": detection["id_scope"],
                "screenshot_path": str(row.get("path", "")),
            })

    # A track that is geometrically ambiguous for almost the entire lesson is
    # not a student identity merely because its first-frame seed was unique.
    # Promote per-observation uncertainty to the whole track before candidate
    # selection so such seeds consume neither fairness quota nor Qwen calls.
    valid_frame_count = sum(row.get("status") == "ok" for row in enriched_rows)
    minimum_reliable_observations = min(
        max(1, valid_frame_count),
        max(3, int(math.ceil(valid_frame_count * 0.05))),
    )
    track_quality: dict[str, dict[str, Any]] = {}
    for observation in observations:
        quality = track_quality.setdefault(observation.track_id, {
            "observations": 0,
            "reliable_observations": 0,
        })
        quality["observations"] = int(quality["observations"]) + 1
        if not bool(observation.detection.attributes.get("association_ambiguous")):
            quality["reliable_observations"] = int(quality["reliable_observations"]) + 1
    uncertain_tracks: set[str] = set()
    for track_id, quality in track_quality.items():
        total = int(quality["observations"])
        reliable = int(quality["reliable_observations"])
        ratio = reliable / total if total else 0.0
        quality.update({
            "reliable_ratio": ratio,
            "minimum_reliable_observations": minimum_reliable_observations,
            "minimum_reliable_ratio": 0.10,
            "status": (
                "reliable"
                if reliable >= minimum_reliable_observations and ratio >= 0.10
                else "uncertain"
            ),
        })
        if quality["status"] == "uncertain":
            uncertain_tracks.add(track_id)

    if uncertain_tracks:
        promoted: list[TrackObservation] = []
        for observation in observations:
            if observation.track_id not in uncertain_tracks:
                promoted.append(observation)
                continue
            attributes = dict(observation.detection.attributes)
            existing_reason = str(attributes.get("association_ambiguity_reason") or "").strip(",")
            reasons = [reason for reason in existing_reason.split(",") if reason]
            if "track_quality_insufficient" not in reasons:
                reasons.append("track_quality_insufficient")
            attributes.update({
                "association_ambiguous": True,
                "association_ambiguity_reason": ",".join(reasons),
                "track_identity_reliable": False,
                "id_uncertain": True,
            })
            promoted.append(TrackObservation(
                track_id=observation.track_id,
                frame_index=observation.frame_index,
                pts_seconds=observation.pts_seconds,
                detection=Detection(
                    label=observation.detection.label,
                    bbox=observation.detection.bbox,
                    score=observation.detection.score,
                    source=observation.detection.source,
                    class_id=observation.detection.class_id,
                    attributes=attributes,
                ),
                stream_id=observation.stream_id,
                anomaly_scores=observation.anomaly_scores,
                metadata={
                    **dict(observation.metadata),
                    "association_ambiguous": True,
                    "id_uncertain": True,
                    "track_identity_reliable": False,
                },
            ))
        observations = promoted
        for row in flat_boxes:
            if str(row.get("student_id")) in uncertain_tracks:
                existing_reason = str(row.get("association_ambiguity_reason") or "").strip(",")
                reasons = [reason for reason in existing_reason.split(",") if reason]
                if "track_quality_insufficient" not in reasons:
                    reasons.append("track_quality_insufficient")
                row.update({
                    "association_ambiguous": True,
                    "association_ambiguity_reason": ",".join(reasons),
                    "track_identity_reliable": False,
                    "id_uncertain": True,
                })
        for row in enriched_rows:
            detections = []
            for raw in row.get("detections", ()):
                detection = dict(raw) if isinstance(raw, Mapping) else raw
                if isinstance(detection, dict) and str(detection.get("student_id")) in uncertain_tracks:
                    existing_reason = str(detection.get("association_ambiguity_reason") or "").strip(",")
                    reasons = [reason for reason in existing_reason.split(",") if reason]
                    if "track_quality_insufficient" not in reasons:
                        reasons.append("track_quality_insufficient")
                    detection.update({
                        "association_ambiguous": True,
                        "association_ambiguity_reason": ",".join(reasons),
                        "track_identity_reliable": False,
                        "id_uncertain": True,
                    })
                detections.append(detection)
            if "detections" in row:
                row["detections"] = detections

    observations.sort(key=lambda item: (item.frame_index, student_id_sort_key(item.track_id)))
    flat_boxes.sort(key=lambda item: (item["frame_index"], student_id_sort_key(item["student_id"])))
    unique_students = len({item.track_id for item in observations})
    ambiguous_associations = sum(
        bool(item.detection.attributes.get("association_ambiguous")) for item in observations
    )
    return observations, enriched_rows, flat_boxes, {
        "status": "ok",
        "source": "locate_anything",
        "identity_provider": "locate_box_association",
        "association_algorithm": "global one-to-one persistent-seat-anchor association",
        "tracker": "LocateStudentIdAssigner",
        "input_boxes_source": "LocateAnything",
        "yolo_model_loaded": False,
        "observations": len(observations),
        "students": unique_students,
        "reliable_students": unique_students - len(uncertain_tracks),
        "uncertain_students": len(uncertain_tracks),
        "uncertain_student_ids": sorted(uncertain_tracks, key=student_id_sort_key),
        "track_quality": {
            track_id: track_quality[track_id]
            for track_id in sorted(track_quality, key=student_id_sort_key)
        },
        "frames": len(enriched_rows),
        "maximum_frame_students": maximum_frame_students,
        "association_method_counts": dict(sorted(method_counts.items())),
        "ambiguous_associations": ambiguous_associations,
        "ambiguous_association_rate": (
            ambiguous_associations / len(observations) if observations else 0.0
        ),
        "provides_cross_frame_ids": True,
        "locate_native_ids_are_frame_local": True,
    }


def render_locate_student_annotations(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    jpeg_quality: int = 95,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render the saved Locate boxes and persistent IDs for visual audit."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rendered_rows: list[dict[str, Any]] = []
    rendered = 0
    failed = 0
    boxes = 0
    for raw_row in rows:
        row = dict(raw_row)
        if row.get("status") != "ok":
            rendered_rows.append(row)
            continue
        source = Path(str(row.get("path", "")))
        image = cv2.imread(str(source))
        if image is None:
            row["annotation_status"] = "failed"
            row["annotation_error"] = "screenshot_decode_failed"
            rendered_rows.append(row)
            failed += 1
            continue
        height, width = image.shape[:2]
        for detection in row.get("detections", ()):
            if not isinstance(detection, Mapping) or not detection.get("student_id"):
                continue
            bbox = detection.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
            y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            student_id = str(detection["student_id"])
            locate_index = int(detection.get("locate_detection_index", 0))
            ambiguous = bool(
                detection.get("association_ambiguous")
                or detection.get("id_uncertain")
            )
            color_seed = max(1, student_id_sort_key(student_id)[0])
            color = (128, 128, 128) if ambiguous else (
                64 + (47 * color_seed) % 192,
                64 + (89 * color_seed) % 192,
                64 + (137 * color_seed) % 192,
            )
            label = f"{student_id}{'?' if ambiguous else ''} | Locate L{locate_index}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1 if ambiguous else 2)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_top = max(0, y1 - text_height - baseline - 5)
            cv2.rectangle(
                image,
                (x1, label_top),
                (min(width - 1, x1 + text_width + 6), y1),
                color,
                thickness=-1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 3, max(text_height + 1, y1 - baseline - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            boxes += 1
        output = destination / f"frame_{int(row.get('frame_index', 0)):09d}.jpg"
        if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
            row["annotation_status"] = "failed"
            row["annotation_error"] = "annotation_write_failed"
            failed += 1
        else:
            row["annotated_path"] = str(output)
            row["annotation_status"] = "ok"
            rendered += 1
        rendered_rows.append(row)
    return rendered_rows, {
        "status": "warning" if failed else "ok",
        "rendered_screenshots": rendered,
        "failed_screenshots": failed,
        "rendered_student_boxes": boxes,
        "artifact_dir": str(destination),
        "label_format": "student_id[? if uncertain] | Locate L<frame-local-index>",
    }


def candidates_from_locate_observations(
    observations: Sequence[TrackObservation],
) -> list[CandidateEvent]:
    ordered = sorted(observations, key=lambda item: (item.frame_index, student_id_sort_key(item.track_id)))
    first_frame_by_track: dict[str, int] = {}
    previous_by_track: dict[str, TrackObservation] = {}
    candidates: list[CandidateEvent] = []
    all_boxes = [tuple(float(value) for value in item.detection.bbox) for item in ordered]
    max_x = max((box[2] for box in all_boxes), default=1.0)
    max_y = max((box[3] for box in all_boxes), default=1.0)
    normalized_coordinates = max(max_x, max_y) <= 1.5
    for observation in ordered:
        previous = previous_by_track.get(observation.track_id)
        bbox = tuple(float(value) for value in observation.detection.bbox)
        if normalized_coordinates:
            x1, y1, x2, y2 = bbox
            edge_seat = x1 <= 0.08 or y1 <= 0.08 or x2 >= 0.92 or y2 >= 0.92
        else:
            edge_margin_x = max(5.0, max_x * 0.08)
            edge_margin_y = max(5.0, max_y * 0.08)
            x1, y1, x2, y2 = bbox
            edge_seat = x1 <= edge_margin_x or y1 <= edge_margin_y or x2 >= max_x - edge_margin_x or y2 >= max_y - edge_margin_y
        action_change = False
        if previous is not None:
            previous_bbox = previous.detection.bbox
            iou = _iou(bbox, previous_bbox)
            previous_center = ((previous_bbox[0] + previous_bbox[2]) / 2.0, (previous_bbox[1] + previous_bbox[3]) / 2.0)
            current_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            scale = max(1.0, math.hypot(max_x, max_y))
            center_delta = math.hypot(current_center[0] - previous_center[0], current_center[1] - previous_center[1]) / scale
            action_change = iou < 0.45 or center_delta >= 0.08
        new_id = observation.track_id not in first_frame_by_track
        first_frame_by_track.setdefault(observation.track_id, observation.frame_index)
        attributes = observation.detection.attributes if isinstance(observation.detection.attributes, Mapping) else {}
        action_change = action_change or any(
            bool(attributes.get(key))
            for key in ("action_change", "motion_change", "change")
        )
        edge_seat = edge_seat or any(
            bool(attributes.get(key))
            for key in ("edge", "edge_seat", "boundary")
        )
        anomaly = any(
            bool(attributes.get(key))
            for key in ("anomaly", "custom_anomaly", "anomaly_candidate")
        ) or any(
            float(score) >= 0.5
            for score in observation.anomaly_scores.values()
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        )
        id_uncertain = bool(
            observation.metadata.get("id_uncertain")
            or observation.metadata.get("association_ambiguous")
            or attributes.get("id_uncertain")
            or attributes.get("association_ambiguous")
        )
        metadata = {
            "trigger": "every_locate_screenshot",
            "small_model_assertion": False,
            "qwen_request_order": "frame_then_student_id",
            "selection_features": {
                "edge_seat": bool(edge_seat),
                "new_student_id": bool(new_id),
                "action_change": bool(action_change),
                "anomaly": bool(anomaly),
                "id_uncertain": bool(id_uncertain),
            },
            "edge_seat": bool(edge_seat),
            "new_student_id": bool(new_id),
            "action_change": bool(action_change),
            "anomaly": bool(anomaly),
            "id_uncertain": bool(id_uncertain),
        }
        generic_source = bool(observation.metadata.get("localization_contract")) or observation.detection.source != "locate_anything"
        if generic_source:
            metadata.update({
                "trigger": "every_localization_screenshot",
                "detector_source": observation.detection.source,
                "localization_contract": observation.metadata.get("localization_contract", "classroom_hybrid_ai.localization.v1"),
                # IDs and reviewed_event_type retain their compatibility
                # namespace so source reporting cannot change fair ordering.
                "event_namespace": "locate_snapshot_review_compatibility",
            })
        if anomaly:
            metadata.update({"priority": 100.0, "candidate_class": "custom_anomaly"})
        candidates.append(CandidateEvent(
            event_id=canonical_sha256({
                "source": "locate_first",
                "track_id": observation.track_id,
                "frame_index": observation.frame_index,
                "pts_seconds": round(observation.pts_seconds, 6),
            })[:32],
            track_id=observation.track_id,
            event_type="locate_snapshot_review",
            trigger_frame_index=observation.frame_index,
            trigger_pts_seconds=observation.pts_seconds,
            source=observation.detection.source if generic_source else "locate_anything",
            metadata=metadata,
        ))
        previous_by_track[observation.track_id] = observation
    return sorted(
        candidates,
        key=lambda item: (item.trigger_frame_index, student_id_sort_key(item.track_id), item.event_id),
    )


__all__ = [
    "LocateStudentIdAssigner",
    "associate_locate_rows",
    "candidates_from_locate_observations",
    "materialize_locate_screenshots",
    "materialize_scheduled_screenshots",
    "render_locate_student_annotations",
    "student_id_sort_key",
]
