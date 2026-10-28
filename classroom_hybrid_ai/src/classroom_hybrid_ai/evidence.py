"""Build bounded, per-event visual evidence from a second video pass."""

from __future__ import annotations

import bisect
import math
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .hashing import sha256_file
from .video import iter_video, iter_video_target_windows
from .performance import increment, record_timing


def expand_box(
    box_xyxy: Sequence[float],
    width: int,
    height: int,
    context_ratio: float,
) -> tuple[int, int, int, int]:
    if len(box_xyxy) != 4:
        raise ValueError("bbox_must_have_four_values")
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_must_have_positive_area")
    margin_x = (x2 - x1) * context_ratio
    margin_y = (y2 - y1) * context_ratio
    return (
        max(0, int(math.floor(x1 - margin_x))),
        max(0, int(math.floor(y1 - margin_y))),
        min(width, int(math.ceil(x2 + margin_x))),
        min(height, int(math.ceil(y2 + margin_y))),
    )


def resize_max_side(image: np.ndarray, maximum_side: int) -> np.ndarray:
    if maximum_side <= 0:
        raise ValueError("maximum_side_must_be_positive")
    height, width = image.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def temporal_sample_times(
    trigger_seconds: float,
    before_seconds: float,
    after_seconds: float,
    count: int,
) -> list[float]:
    if count < 1 or before_seconds < 0 or after_seconds < 0:
        raise ValueError("invalid_temporal_sampling_arguments")
    start = max(0.0, trigger_seconds - before_seconds)
    end = max(start, trigger_seconds + after_seconds)
    if count == 1:
        return [trigger_seconds]
    return [start + index * (end - start) / (count - 1) for index in range(count)]


def _observation_pts(row: dict[str, Any]) -> float:
    for key in ("pts_seconds", "timestamp_seconds", "local_start_seconds"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(row.get("local_start_us"), int):
        return row["local_start_us"] / 1_000_000.0
    raise ValueError("observation_pts_missing")


def _observation_box(row: dict[str, Any]) -> Sequence[float] | None:
    """Accept both flattened legacy rows and the canonical observation contract."""

    for key in ("bbox_xyxy", "box_xyxy", "bbox"):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return value
    detection = row.get("detection")
    if isinstance(detection, dict):
        for key in ("bbox", "bbox_xyxy", "box_xyxy"):
            value = detection.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 4:
                return value
    return None


class ObservationIndex:
    def __init__(self, observations: Iterable[dict[str, Any]]) -> None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in observations:
            track_id = row.get("track_id", row.get("local_track_id"))
            if track_id is None:
                continue
            normalized = dict(row)
            grouped[str(track_id)].append(normalized)
            frame_index = normalized.get("frame_index")
            if isinstance(frame_index, int) and not isinstance(frame_index, bool) and frame_index >= 0:
                by_frame[frame_index].append(normalized)
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._times: dict[str, list[float]] = {}
        for track_id, rows in grouped.items():
            rows.sort(key=_observation_pts)
            self._rows[track_id] = rows
            self._times[track_id] = [_observation_pts(row) for row in rows]
        self._by_frame = dict(by_frame)

    def nearest(self, track_id: str, pts_seconds: float) -> dict[str, Any] | None:
        times = self._times.get(str(track_id))
        if not times:
            return None
        position = bisect.bisect_left(times, pts_seconds)
        choices = [idx for idx in (position - 1, position) if 0 <= idx < len(times)]
        selected = min(choices, key=lambda idx: abs(times[idx] - pts_seconds))
        return self._rows[str(track_id)][selected]

    def at_frame(self, frame_index: int) -> list[dict[str, Any]]:
        """Return all visible track rows for one decoded context frame."""

        return list(self._by_frame.get(int(frame_index), ()))


class LocateIndex:
    """Nearest-scene access for LocateAnything boxes.

    Locate indices are scoped to one model response.  Persistent student IDs
    still come from the video tracker and are associated below by box overlap.
    """

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        timed: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            if row.get("status") != "ok" or not isinstance(row.get("detections"), list):
                continue
            try:
                timed.append((_observation_pts(row), dict(row)))
            except ValueError:
                continue
        timed.sort(key=lambda item: item[0])
        self._times = [item[0] for item in timed]
        self._rows = [item[1] for item in timed]

    def nearest(self, pts_seconds: float, *, maximum_delta_seconds: float) -> dict[str, Any] | None:
        if not self._times:
            return None
        position = bisect.bisect_left(self._times, float(pts_seconds))
        choices = [idx for idx in (position - 1, position) if 0 <= idx < len(self._times)]
        selected = min(choices, key=lambda idx: abs(self._times[idx] - pts_seconds))
        if abs(self._times[selected] - pts_seconds) > maximum_delta_seconds:
            return None
        return self._rows[selected]


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    if intersection <= 0:
        return 0.0
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(
        0.0, bx2 - bx1
    ) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _matching_locate_student(
    target_box: Sequence[float],
    row: Mapping[str, Any] | None,
    *,
    minimum_iou: float,
) -> tuple[int, Mapping[str, Any]] | None:
    if row is None:
        return None
    matches: list[tuple[float, int, Mapping[str, Any]]] = []
    for fallback_index, detection in enumerate(row.get("detections", ()), start=1):
        if not isinstance(detection, Mapping):
            continue
        label = str(detection.get("label", "")).strip().lower()
        if label not in {"student", "person", "pupil", "learner"}:
            continue
        box = detection.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        overlap = _box_iou(target_box, box)
        if overlap >= minimum_iou:
            index = detection.get("detection_index", fallback_index)
            matches.append((overlap, int(index), detection))
    if not matches:
        return None
    _, index, detection = max(matches, key=lambda item: item[0])
    return index, detection


def extract_frames_at_times(
    video_path: str | Path,
    target_seconds: Sequence[float],
    *,
    decode_strategy: str = "sequential",
) -> dict[float, tuple[int, float, np.ndarray]]:
    """Retain the nearest PTS frame; auto can seek verified CFR inputs."""

    targets = sorted(set(max(0.0, float(value)) for value in target_seconds))
    if not targets:
        return {}
    output: dict[float, tuple[int, float, np.ndarray]] = {}
    target_index = 0
    previous: tuple[int, float, np.ndarray] | None = None
    for decoded in (iter_video(video_path) if decode_strategy == "sequential" else iter_video_target_windows(video_path, targets, strategy=decode_strategy)):
        current = (decoded.frame_index, decoded.pts_seconds, decoded.image_bgr)
        while target_index < len(targets) and decoded.pts_seconds >= targets[target_index]:
            target = targets[target_index]
            candidates = [current]
            if previous is not None:
                candidates.append(previous)
            nearest = min(candidates, key=lambda item: abs(item[1] - target))
            output[target] = (nearest[0], nearest[1], nearest[2].copy())
            target_index += 1
        previous = current
        if target_index >= len(targets):
            break
    if previous is not None:
        while target_index < len(targets):
            target = targets[target_index]
            output[target] = (previous[0], previous[1], previous[2].copy())
            target_index += 1
    return output


def _candidate_value(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _candidate_trigger_seconds(candidate: Any) -> float:
    for name in ("trigger_pts_seconds", "pts_seconds", "trigger_seconds"):
        value = _candidate_value(candidate, name)
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError("candidate_trigger_time_missing")


def merge_candidate_windows(
    candidates: Sequence[Any],
    *,
    before_seconds: float,
    after_seconds: float,
) -> dict[str, list[tuple[float, float]]]:
    """Merge event windows per track so a long JSONL file can be filtered once."""

    if before_seconds < 0 or after_seconds < 0:
        raise ValueError("evidence_window_must_be_non_negative")
    raw: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for candidate in candidates:
        track_id = str(_candidate_value(candidate, "track_id"))
        trigger = _candidate_trigger_seconds(candidate)
        raw[track_id].append((max(0.0, trigger - before_seconds), trigger + after_seconds))
    merged: dict[str, list[tuple[float, float]]] = {}
    for track_id, windows in raw.items():
        output: list[tuple[float, float]] = []
        for start, end in sorted(windows):
            if output and start <= output[-1][1]:
                output[-1] = (output[-1][0], max(output[-1][1], end))
            else:
                output.append((start, end))
        merged[track_id] = output
    return merged


def filter_observations_for_candidates(
    observations: Iterable[dict[str, Any]],
    candidates: Sequence[Any],
    *,
    before_seconds: float,
    after_seconds: float,
    include_all_tracks_in_windows: bool = False,
) -> list[dict[str, Any]]:
    """Retain rows for candidate crops and, optionally, full-frame context.

    ``include_all_tracks_in_windows`` is used by the temporal-composite path:
    it keeps every visible student in the small candidate time windows so the
    annotated context tile can show peer relationships without loading the
    complete long-video observation log into memory.
    """

    windows = merge_candidate_windows(
        candidates,
        before_seconds=before_seconds,
        after_seconds=after_seconds,
    )
    all_windows: list[tuple[float, float]] = []
    for start, end in sorted(window for values in windows.values() for window in values):
        if all_windows and start <= all_windows[-1][1]:
            all_windows[-1] = (all_windows[-1][0], max(end, all_windows[-1][1]))
        else:
            all_windows.append((start, end))
    starts_by_track = {track: [start for start, _ in values] for track, values in windows.items()}
    all_starts = [start for start, _ in all_windows]
    selected: list[dict[str, Any]] = []
    for row in observations:
        track_id = row.get("track_id", row.get("local_track_id"))
        if track_id is None:
            continue
        pts = _observation_pts(row)
        if include_all_tracks_in_windows:
            selected_windows = all_windows
            starts = all_starts
        else:
            selected_windows = windows.get(str(track_id), ())
            starts = starts_by_track.get(str(track_id), ())
        position = bisect.bisect_right(starts, pts) - 1
        if position >= 0 and pts <= selected_windows[position][1]:
            selected.append(row)
    return selected


def build_event_evidence(
    *,
    video_path: str | Path,
    candidates: Sequence[Any],
    observations: Sequence[dict[str, Any]],
    locate_rows: Sequence[dict[str, Any]] = (),
    output_dir: str | Path,
    before_seconds: float,
    after_seconds: float,
    frames_per_event: int,
    context_ratio: float,
    maximum_crop_side: int,
    maximum_context_side: int = 768,
    locate_maximum_delta_seconds: float = 2.0,
    locate_minimum_iou: float = 0.10,
    jpeg_quality: int = 90,
    progress_callback: Callable[[int, int], None] | None = None,
    decode_strategy: str = "sequential",
    preprocess_workers: int = 1,
    preprocess_queue_size: int = 2,
) -> list[dict[str, Any]]:
    """Build ordered evidence with a configurable, bounded preprocessing queue."""

    if preprocess_workers not in {1, 2, 4}:
        raise ValueError("evidence_preprocess_workers_must_be_1_2_or_4")
    if preprocess_queue_size < 1:
        raise ValueError("evidence_preprocess_queue_size_must_be_positive")
    if decode_strategy not in {"sequential", "auto"}:
        raise ValueError("video_decode_strategy_invalid")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    index = ObservationIndex(observations)
    locate_index = LocateIndex(locate_rows)
    receipts_by_event: dict[str, dict[str, Any]] = {}
    ordered_event_ids: list[str] = []
    samples: list[dict[str, Any]] = []
    for candidate in candidates:
        event_id = str(_candidate_value(candidate, "event_id"))
        if event_id in receipts_by_event:
            raise ValueError(f"duplicate_candidate_event_id:{event_id}")
        track_id = str(_candidate_value(candidate, "track_id"))
        trigger = _candidate_trigger_seconds(candidate)
        times = temporal_sample_times(trigger, before_seconds, after_seconds, frames_per_event)
        event_dir = destination / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        receipts_by_event[event_id] = {
            "event_id": event_id,
            "track_id": track_id,
            "status": "insufficient_evidence",
            "evidence": [],
            "target_highlighted": False,
            "_context_written": False,
        }
        ordered_event_ids.append(event_id)
        for ordinal, target in enumerate(times):
            observation = index.nearest(track_id, target)
            if observation is None:
                continue
            box = _observation_box(observation)
            if box is None:
                continue
            samples.append({
                "target": target,
                "event_id": event_id,
                "track_id": track_id,
                "event_dir": event_dir,
                "ordinal": ordinal,
                "write_context": ordinal >= len(times) // 2,
                "box": box,
            })
    samples.sort(key=lambda item: (item["target"], item["event_id"], item["ordinal"]))
    if progress_callback is not None:
        progress_callback(0, len(samples))

    def materialize(sample: dict[str, Any], packet: tuple[int, float, np.ndarray]) -> None:
        frame_index, actual_pts, frame = packet
        height, width = frame.shape[:2]
        locate_row = locate_index.nearest(
            actual_pts, maximum_delta_seconds=locate_maximum_delta_seconds
        )
        locate_match = _matching_locate_student(
            sample["box"], locate_row, minimum_iou=locate_minimum_iou
        )
        source_box = sample["box"]
        if locate_match is not None:
            source_box = locate_match[1]["bbox"]
        x1, y1, x2, y2 = expand_box(source_box, width, height, context_ratio)
        if x2 <= x1 or y2 <= y1:
            return
        crop = resize_max_side(frame[y1:y2, x1:x2], maximum_crop_side)
        if locate_match is not None:
            locate_number, locate_detection = locate_match
            cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (255, 180, 0), 2)
            cv2.putText(
                crop,
                f"Locate L{locate_number}:student id={sample['track_id']}",
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )
        crop_path = sample["event_dir"] / f"subject_{sample['ordinal']:02d}.jpg"
        if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            raise OSError(f"image_write_failed:{crop_path}")
        receipt = receipts_by_event[sample["event_id"]]
        receipt["evidence"].append({
            "kind": "subject_crop",
            "path": crop_path.as_posix(),
            "frame_index": frame_index,
            "requested_pts_seconds": round(sample["target"], 6),
            "actual_pts_seconds": round(actual_pts, 6),
            "bbox_xyxy": [x1, y1, x2, y2],
            "localization_source": "locate_anything" if locate_match is not None else "yolo_track",
            "locate_detection_index": locate_match[0] if locate_match is not None else None,
            "locate_label": str(locate_match[1].get("label")) if locate_match is not None else None,
            "sha256": sha256_file(crop_path),
        })
        if sample["write_context"] and not receipt["_context_written"]:
            annotated = resize_max_side(frame, maximum_context_side)
            scale_x = annotated.shape[1] / width
            scale_y = annotated.shape[0] / height
            # Show the spatial model's actual per-image boxes in the image
            # that Qwen receives.  L1/L2/... are detection numbers for this
            # Locate response; the tracker ID is drawn separately below.
            if locate_row is not None:
                for fallback_index, detection in enumerate(locate_row.get("detections", ()), start=1):
                    if not isinstance(detection, Mapping):
                        continue
                    locate_box = detection.get("bbox")
                    if not isinstance(locate_box, (list, tuple)) or len(locate_box) != 4:
                        continue
                    try:
                        lx1, ly1, lx2, ly2 = expand_box(locate_box, width, height, 0.0)
                    except ValueError:
                        continue
                    detection_index = int(detection.get("detection_index", fallback_index))
                    label = str(detection.get("label") or "object")
                    colour = (255, 180, 0)
                    alx1, aly1 = round(lx1 * scale_x), round(ly1 * scale_y)
                    alx2, aly2 = round(lx2 * scale_x), round(ly2 * scale_y)
                    cv2.rectangle(annotated, (alx1, aly1), (alx2, aly2), colour, 1)
                    cv2.putText(
                        annotated,
                        f"L{detection_index}:{label}",
                        (alx1, max(12, aly1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.30,
                        colour,
                        1,
                        cv2.LINE_AA,
                    )
            # The lower row must preserve classroom context.  Draw every
            # student visible at the selected frame, with the requested
            # student highlighted in yellow.  The input is already bounded
            # to candidate time windows, so this does not retain a full-video
            # observation index.
            context_rows = index.at_frame(frame_index)
            target_drawn = False
            for context_row in context_rows:
                context_box = _observation_box(context_row)
                context_track = context_row.get("track_id", context_row.get("local_track_id"))
                if context_box is None or context_track is None:
                    continue
                try:
                    cx1, cy1, cx2, cy2 = expand_box(context_box, width, height, 0.0)
                except ValueError:
                    continue
                is_target = str(context_track) == str(sample["track_id"])
                target_drawn = target_drawn or is_target
                colour = (0, 255, 255) if is_target else (80, 220, 80)
                thickness = 2 if is_target else 1
                ax1, ay1 = round(cx1 * scale_x), round(cy1 * scale_y)
                ax2, ay2 = round(cx2 * scale_x), round(cy2 * scale_y)
                cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), colour, thickness)
                display_id = context_row.get("student_id", context_track)
                label = f"id={display_id}"
                cv2.putText(
                    annotated,
                    label,
                    (ax1, max(14, ay1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38 if not is_target else 0.5,
                    colour,
                    1 if not is_target else 2,
                    cv2.LINE_AA,
                )
            if not target_drawn:
                # Keep the target annotation even when a caller supplies only
                # target-track observations (legacy/test evidence packets).
                cv2.rectangle(
                    annotated,
                    (round(x1 * scale_x), round(y1 * scale_y)),
                    (round(x2 * scale_x), round(y2 * scale_y)),
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    f"id={sample['track_id']}",
                    (round(x1 * scale_x), max(20, round(y1 * scale_y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                # The fallback box is still a genuine target highlight in
                # the global context tile, even for legacy packets that only
                # provide the target track.
                target_drawn = True
            context_path = sample["event_dir"] / "context.jpg"
            if not cv2.imwrite(
                str(context_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            ):
                raise OSError(f"image_write_failed:{context_path}")
            receipt["evidence"].append({
                "kind": "annotated_context",
                "path": context_path.as_posix(),
                "frame_index": frame_index,
                "actual_pts_seconds": round(actual_pts, 6),
                "sha256": sha256_file(context_path),
            })
            receipt["target_highlighted"] = bool(target_drawn)
            receipt["_context_written"] = True

    sample_index = 0
    completed = 0
    pending: deque[tuple[str, Future[None]]] = deque()
    pool = ThreadPoolExecutor(max_workers=preprocess_workers, thread_name_prefix="evidence") if preprocess_workers > 1 else None

    def work(sample: dict[str, Any], packet: tuple[int, float, np.ndarray]) -> None:
        started = time.perf_counter()
        try:
            materialize(sample, packet)
        finally:
            record_timing("evidence.preprocess_seconds", time.perf_counter() - started)

    def finish_one() -> None:
        nonlocal completed
        _, future = pending.popleft()
        future.result()
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, len(samples))

    def submit(sample: dict[str, Any], packet: tuple[int, float, np.ndarray]) -> None:
        nonlocal completed
        if pool is None:
            work(sample, packet)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(samples))
            return
        # Each event's receipt/context is modified in original sample order.
        # Different events can crop/encode concurrently with a strict bound.
        while pending and (len(pending) >= preprocess_queue_size or any(event == sample["event_id"] for event, _ in pending)):
            finish_one()
        pending.append((sample["event_id"], pool.submit(copy_context().run, work, sample, packet)))

    previous: tuple[int, float, np.ndarray] | None = None
    try:
        stream = iter_video(video_path) if decode_strategy == "sequential" else iter_video_target_windows(video_path, [sample["target"] for sample in samples], strategy=decode_strategy)
        for decoded in stream if samples else ():
            current = (decoded.frame_index, decoded.pts_seconds, decoded.image_bgr)
            while sample_index < len(samples) and decoded.pts_seconds >= samples[sample_index]["target"]:
                target = samples[sample_index]["target"]
                choices = [current] if previous is None else [previous, current]
                packet = min(choices, key=lambda item: abs(item[1] - target))
                submit(samples[sample_index], packet)
                sample_index += 1
            previous = current
            if sample_index >= len(samples):
                break
        if previous is not None:
            while sample_index < len(samples):
                submit(samples[sample_index], previous)
                sample_index += 1
        while pending:
            finish_one()
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    receipts: list[dict[str, Any]] = []
    for event_id in ordered_event_ids:
        receipt = receipts_by_event[event_id]
        receipt.pop("_context_written", None)
        receipt["status"] = "ok" if receipt["evidence"] else "insufficient_evidence"
        receipts.append(receipt)
    return receipts


def build_temporal_composite(
    receipt: Mapping[str, Any],
    *,
    output_path: str | Path,
    tile_width: int = 160,
    tile_height: int = 208,
    jpeg_quality: int = 90,
) -> dict[str, Any] | None:
    """Build the one-image Qwen evidence packet for the 8 GB profile.

    Top row contains three temporal subject crops; bottom row repeats the
    annotated context so relations to a teacher/peer remain visible.  All
    source images are local evidence artifacts, never additional video decodes.
    """

    items = [item for item in receipt.get("evidence", ()) if isinstance(item, Mapping)]
    subjects = [item for item in items if item.get("kind") == "subject_crop" and item.get("path")]
    contexts = [item for item in items if item.get("kind") == "annotated_context" and item.get("path")]
    if not subjects:
        return None
    if len(subjects) >= 3:
        indices = [0, len(subjects) // 2, len(subjects) - 1]
        subjects = [subjects[index] for index in indices]
    else:
        subjects = subjects + [subjects[-1]] * (3 - len(subjects))
    canvas = np.full((tile_height * 2, tile_width * 3, 3), 24, dtype=np.uint8)
    selected = subjects + [contexts[0] if contexts else subjects[1]] * 3
    tiles: dict[str, np.ndarray] = {}
    for index, item in enumerate(selected):
        source_path = str(item["path"])
        image = tiles.get(source_path)
        if image is None:
            started = time.perf_counter()
            image = cv2.imread(source_path)
            increment("composite_image_reads")
            if image is None:
                return None
            increment("composite_image_read_bytes", Path(source_path).stat().st_size)
            image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            record_timing("evidence.composite_read_resize_seconds", time.perf_counter() - started)
            tiles[source_path] = image
        row, column = divmod(index, 3)
        y1, y2 = row * tile_height, (row + 1) * tile_height
        x1, x2 = column * tile_width, (column + 1) * tile_width
        canvas[y1:y2, x1:x2] = image
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise OSError(f"composite_write_failed:{destination}")
    frames = [int(item["frame_index"]) for item in subjects if item.get("frame_index") is not None]
    return {
        "path": destination.as_posix(),
        "frame_indices": frames,
        "shape": [int(canvas.shape[1]), int(canvas.shape[0])],
        "kind": "temporal_composite",
        "context_target_highlighted": bool(receipt.get("target_highlighted", False)),
        "sha256": sha256_file(destination),
    }
