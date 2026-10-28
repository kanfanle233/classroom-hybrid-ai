"""Video timing helpers. Time decisions use PTS seconds, not only frame numbers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import cv2
import numpy as np

from .performance import increment, record_timing


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DecodedFrame:
    frame_index: int
    pts_seconds: float
    image_bgr: np.ndarray


def probe_video(path: str | Path) -> VideoInfo:
    source = Path(path).resolve()
    if not source.is_file():
        raise VideoError(f"video_not_found:{source}")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise VideoError(f"video_open_failed:{source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
            raise VideoError(
                f"invalid_video_metadata:fps={fps}:frames={count}:size={width}x{height}"
            )
        return VideoInfo(
            path=str(source),
            fps=fps,
            frame_count=count,
            duration_seconds=count / fps,
            width=width,
            height=height,
        )
    finally:
        capture.release()


def iter_video(path: str | Path, *, stride: int = 1) -> Iterator[DecodedFrame]:
    if stride < 1:
        raise ValueError("stride_must_be_positive")
    source = Path(path).resolve()
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise VideoError(f"video_open_failed:{source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        index = 0
        last_pts = -1.0
        while True:
            started = time.perf_counter()
            ok, image = capture.read()
            record_timing("video.decode_seconds", time.perf_counter() - started)
            if not ok:
                break
            increment("decoded_frames")
            if index % stride:
                index += 1
                continue
            msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            pts = msec / 1000.0 if msec >= 0 else index / fps
            # Some OpenCV backends report zero for every frame. Fall back to
            # the nominal FPS while retaining monotonic time.
            if pts <= last_pts and index > 0:
                pts = index / fps
            last_pts = pts
            yield DecodedFrame(index, pts, image)
            index += 1
    finally:
        capture.release()


@lru_cache(maxsize=16)
def _constant_frame_timing(path: str, size: int, modified_ns: int) -> tuple[float, int] | None:
    """Validate the entire compressed PTS stream before using frame-number seeks.

    Demuxing does not decode image pixels. VFR, duplicate/missing/nonzero-start
    timestamps and unavailable PyAV all select the original sequential reader.
    The stat-keyed cache is process-local; it is never a recovery integrity check.
    """
    started = time.perf_counter()
    packet_bytes = 0
    try:
        import av

        with av.open(path) as container:
            stream = container.streams.video[0]
            if stream.average_rate is None:
                return None
            fps = float(stream.average_rate)
            if not math.isfinite(fps) or fps <= 0:
                return None
            seen: set[int] = set()
            for packet in container.demux(stream):
                if not packet.size:
                    continue
                packet_bytes += int(packet.size)
                if packet.pts is None or packet.time_base is None:
                    return None
                position = float(packet.pts * packet.time_base) * fps
                index = round(position)
                if index < 0 or abs(position - index) > 1e-4 or index in seen:
                    return None
                seen.add(index)
            if not seen or min(seen) != 0 or max(seen) != len(seen) - 1:
                return None
            return fps, len(seen)
    except (ImportError, OSError, ValueError, IndexError):
        return None
    finally:
        record_timing("video.timing_probe_seconds", time.perf_counter() - started)
        increment("timing_probe_packet_bytes", packet_bytes)


def iter_video_target_windows(
    path: str | Path,
    targets: Sequence[float],
    *,
    strategy: str = "sequential",
) -> Iterator[DecodedFrame]:
    """Yield the bracketing frames needed for sorted nearest-PTS requests.

    ``auto`` seeks only after verifying a CFR packet timeline. Nearby windows
    merge into one decode; dense requests and VFR retain sequential decoding.
    Every seek checks the backend's actual frame position and PTS. Unsupported
    or inaccurate seeks restart sequentially, never relabel a sought frame.
    """
    if strategy not in {"sequential", "auto"}:
        raise ValueError("video_decode_strategy_invalid")
    values = sorted(set(max(0.0, float(value)) for value in targets))
    if not values:
        return
    if not all(math.isfinite(value) for value in values):
        raise ValueError("video_target_time_must_be_finite")
    source = Path(path).resolve()
    timing = None
    if strategy == "auto":
        stat = source.stat()
        timing = _constant_frame_timing(str(source), stat.st_size, stat.st_mtime_ns)
    if timing is None:
        increment("sequential_decode_passes")
        yield from iter_video(source)
        return
    fps, count = timing
    windows: list[tuple[int, int]] = []
    for target in values:
        center = min(count - 1, max(0, int(math.floor(target * fps))))
        start, end = max(0, center - 1), min(count - 1, center + 2)
        if windows and start - windows[-1][1] <= max(16, round(2 * fps)):
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))
    if sum(end - start + 1 for start, end in windows) >= (windows[-1][1] + 1) * 0.5:
        increment("sequential_decode_passes")
        yield from iter_video(source)
        return
    capture = cv2.VideoCapture(str(source))
    last_emitted = -1
    fallback = False
    try:
        if not capture.isOpened():
            raise VideoError(f"video_open_failed:{source}")
        for start, end in windows:
            increment("video_seeks")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                fallback = True
                break
            for expected_index in range(start, end + 1):
                began = time.perf_counter()
                ok, pixels = capture.read()
                record_timing("video.decode_seconds", time.perf_counter() - began)
                if not ok:
                    fallback = True
                    break
                increment("decoded_frames")
                position = float(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if abs(position - expected_index) > 0.01 or not math.isfinite(pts) or abs(pts * fps - expected_index) > 1e-3:
                    fallback = True
                    break
                last_emitted = expected_index
                yield DecodedFrame(expected_index, pts, pixels)
            if fallback:
                break
    finally:
        capture.release()
    if fallback:
        increment("seek_fallbacks")
        for frame in iter_video(source):
            if frame.frame_index > last_emitted:
                yield frame


class SceneChangeDetector:
    """Cheap scene-cut candidate detector used to reset track epochs."""

    def __init__(self, threshold: float = 0.32) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("scene_change_threshold_out_of_range")
        self.threshold = threshold
        self._previous: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._previous is None:
            self._previous = small
            return True, 1.0
        score = float(np.mean(np.abs(small - self._previous)) / 255.0)
        self._previous = small
        return score >= self.threshold, score


def detect_cpu_change_anchors(
    path: str | Path,
    *,
    fps: float | None = None,
    frame_count: int | None = None,
    duration_seconds: float | None = None,
    threshold: float = 0.08,
    sample_stride: int = 10,
    minimum_gap_seconds: float = 0.5,
    maximum_anchors: int = 300,
    progress_callback: Callable[[int, int | None], None] | None = None,
    frame_callback: Callable[[DecodedFrame, bool], None] | None = None,
) -> list[dict[str, Any]]:
    """Find scene/global-change screenshot anchors with a CPU-only pass.

    The pass compares 64x36 grayscale thumbnails and never imports or calls a
    detector. Candidate peaks are kept in temporal order with non-maximum
    suppression, so S1 can reserve a bounded number of true changes while its
    regular PTS anchors continue to cover the full video.
    """

    if float(threshold) < 0.0 or float(threshold) > 1.0:
        raise ValueError("change_anchor_threshold_out_of_range")
    if int(sample_stride) < 1:
        raise ValueError("change_anchor_sample_stride_must_be_positive")
    if float(minimum_gap_seconds) < 0.0:
        raise ValueError("change_anchor_minimum_gap_must_be_non_negative")
    if int(maximum_anchors) < 0:
        raise ValueError("change_anchor_maximum_must_be_non_negative")
    source = Path(path).resolve()
    if not source.is_file():
        raise VideoError(f"video_not_found:{source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoError(f"video_open_failed:{source}")
    effective_fps = float(fps or capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if effective_fps <= 0:
        capture.release()
        raise VideoError("change_anchor_fps_invalid")
    previous: np.ndarray | None = None
    last_pts = -1.0
    callback_last_pts = -1.0
    peaks: list[dict[str, Any]] = []
    index = 0
    progress_total = int(frame_count) if frame_count is not None and int(frame_count) > 0 else None
    if progress_callback is not None:
        progress_callback(0, progress_total)
    try:
        while True:
            decode_started = time.perf_counter()
            ok, image = capture.read()
            record_timing("video.decode_seconds", time.perf_counter() - decode_started)
            if not ok:
                break
            increment("decoded_frames")
            callback_pts = 0.0
            if frame_callback is not None:
                callback_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if callback_pts < 0 or (callback_pts <= callback_last_pts and index > 0):
                    callback_pts = index / effective_fps
                callback_last_pts = callback_pts
            change_candidate = False
            if index % int(sample_stride):
                if frame_callback is not None:
                    frame_callback(DecodedFrame(index, callback_pts, image), False)
                index += 1
                if progress_callback is not None:
                    progress_callback(index, progress_total)
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
            if previous is not None:
                score = float(np.mean(np.abs(small - previous)) / 255.0)
                if score >= float(threshold):
                    change_candidate = True
                    msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    pts = msec / 1000.0 if msec > 0 else index / effective_fps
                    if pts <= last_pts and index > 0:
                        # Several OpenCV backends report zero (or a repeated
                        # timestamp) for every decoded frame. Keep anchors
                        # tied to the sampled frame index in that case.
                        pts = index / effective_fps
                    peaks.append({
                        "frame_index": int(index),
                        "pts_seconds": float(pts if pts >= 0 else index / effective_fps),
                        "anchor_source": "cpu_scene_change",
                        "anchor_reason": "global_thumbnail_difference",
                        "change_score": score,
                        "detector": "cpu_global_frame_difference",
                    })
                    last_pts = pts
            previous = small
            if frame_callback is not None:
                frame_callback(DecodedFrame(index, callback_pts, image), change_candidate)
            index += 1
            if progress_callback is not None:
                progress_callback(index, progress_total)
    finally:
        capture.release()
    if progress_callback is not None:
        progress_callback(index, index if progress_total is None else progress_total)
    if not peaks or maximum_anchors == 0:
        return []
    # Pick strongest local changes first, then restore temporal order. The
    # scheduler performs a second cap/coverage pass; this local limit keeps a
    # pathological cut-heavy video bounded before it reaches the scheduler.
    ranked = sorted(peaks, key=lambda row: (-float(row["change_score"]), int(row["frame_index"])))
    selected: list[dict[str, Any]] = []
    min_gap_frames = max(1, int(round(float(minimum_gap_seconds) * effective_fps)))
    for row in ranked:
        if len(selected) >= int(maximum_anchors):
            break
        if any(abs(int(row["frame_index"]) - int(item["frame_index"])) < min_gap_frames for item in selected):
            continue
        selected.append(row)
    selected.sort(key=lambda row: (int(row["frame_index"]), float(row["pts_seconds"])))
    return selected


def scan_and_materialize_screenshots(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    scheduler: Any,
    cache_dir: str | Path | None = None,
    maximum_cache_bytes: int = 2 * 1024 ** 3,
    max_frames: int | None = None,
    jpeg_quality: int = 95,
    change_options: Mapping[str, Any] | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Share S1's change-analysis decode with screenshot materialization.

    Cache only baseline frames and above-threshold changes, encoded once with
    the production JPEG settings. The scheduler remains the sole authority.
    Overflow, disk errors or PTS-to-frame remapping cache misses use its exact
    original materializer after the scan. Temporary files always stay bounded
    and are removed when the helper exits.
    """
    from .locate_tracking import materialize_scheduled_screenshots

    if maximum_cache_bytes < 0:
        raise ValueError("screenshot_cache_limit_must_be_non_negative")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames_must_be_positive")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache_parent = Path(cache_dir).resolve() if cache_dir is not None else destination.parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    baseline_indices = {int(row["frame_index"]) for row in scheduler.schedule()}
    cache: dict[int, tuple[Path, float]] = {}
    cache_bytes = 0
    decoded_frames = 0
    fallback_reason: str | None = None
    with tempfile.TemporaryDirectory(prefix="s1-cache-", dir=cache_parent) as directory:
        temporary = Path(directory)

        def save_frame(frame: DecodedFrame, changed: bool) -> None:
            nonlocal cache_bytes, decoded_frames, fallback_reason
            decoded_frames += 1
            if fallback_reason or (max_frames is not None and frame.frame_index >= max_frames):
                return
            if not changed and frame.frame_index not in baseline_indices:
                return
            began = time.perf_counter()
            ok, encoded = cv2.imencode(".jpg", frame.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            record_timing("image.encode_seconds", time.perf_counter() - began)
            if not ok:
                fallback_reason = "cache_encode_failed"
                return
            if cache_bytes + encoded.nbytes > maximum_cache_bytes:
                fallback_reason = "cache_limit_exceeded"
                return
            cached_path = temporary / f"frame_{frame.frame_index:09d}.jpg"
            try:
                cached_path.write_bytes(encoded.tobytes())
            except OSError:
                fallback_reason = "cache_write_failed"
                return
            cache_bytes += encoded.nbytes
            increment("screenshot_cache_written_bytes", encoded.nbytes)
            cache[frame.frame_index] = (cached_path, frame.pts_seconds)

        anchors = detect_cpu_change_anchors(
            video_path,
            fps=scheduler.fps,
            frame_count=scheduler.frame_count,
            duration_seconds=scheduler.duration_seconds,
            frame_callback=save_frame,
            progress_callback=progress_callback,
            **dict(change_options or {}),
        )
        requested = [row for row in scheduler.schedule(change_anchors=anchors) if max_frames is None or int(row["frame_index"]) < max_frames]
        if any(int(row["frame_index"]) not in cache for row in requested):
            fallback_reason = fallback_reason or "scheduled_frame_not_cached"
        if fallback_reason:
            rows, sampling = materialize_scheduled_screenshots(video_path, destination, requested, max_frames=max_frames, jpeg_quality=jpeg_quality)
            sampling["decoded_frames"] += decoded_frames
            sampling.update({"decode_passes": 2, "cache_fallback_reason": fallback_reason, "cache_peak_bytes": cache_bytes, "cache_limit_bytes": maximum_cache_bytes})
            return rows, sampling, anchors
        if not requested:
            raise ValueError("scheduled_screenshots_empty")
        rows: list[dict[str, Any]] = []
        for row in requested:
            frame_index = int(row["frame_index"])
            cached_path, pts = cache[frame_index]
            target = destination / cached_path.name
            shutil.copyfile(cached_path, target)
            increment("screenshot_written_bytes", target.stat().st_size)
            rows.append({**row, "path": str(target), "frame_index": frame_index, "pts_seconds": pts, "scheduled_pts_seconds": float(row.get("pts_seconds", pts))})
        rows.sort(key=lambda row: int(row["frame_index"]))
        return rows, {"status": "ok", "requested": len(requested), "screenshots": len(rows), "decoded_frames": decoded_frames, "missing_frame_indices": [], "exact_schedule_applied": True, "decode_passes": 1, "cache_fallback_reason": None, "cache_peak_bytes": cache_bytes, "cache_limit_bytes": maximum_cache_bytes}, anchors


def estimate_workload(
    info: VideoInfo,
    *,
    average_tracks: int,
    fixed_stride: int,
    periodic_seconds: float,
    tracks_per_periodic_audit: int,
    estimated_anomalies_per_minute: float = 0.0,
) -> dict[str, float | int | str]:
    if average_tracks < 0 or fixed_stride < 1 or periodic_seconds <= 0:
        raise ValueError("invalid_workload_arguments")
    sampled_frames = (info.frame_count + fixed_stride - 1) // fixed_stride
    all_track_requests = sampled_frames * average_tracks
    audit_ticks = int(info.duration_seconds // periodic_seconds) + 1
    periodic_requests = audit_ticks * min(average_tracks, tracks_per_periodic_audit)
    anomaly_requests = round(info.duration_seconds / 60.0 * estimated_anomalies_per_minute)
    hybrid_requests = periodic_requests + anomaly_requests
    return {
        "video_seconds": round(info.duration_seconds, 3),
        "fixed_stride": fixed_stride,
        "sampled_frame_count": sampled_frames,
        "fixed_stride_all_tracks_qwen_requests": all_track_requests,
        "periodic_audit_qwen_requests": periodic_requests,
        "estimated_anomaly_qwen_requests": anomaly_requests,
        "estimated_hybrid_qwen_requests": hybrid_requests,
        "recommendation": "hybrid" if hybrid_requests < all_track_requests else "measure_both",
    }
