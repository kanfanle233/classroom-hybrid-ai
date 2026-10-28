from __future__ import annotations

from classroom_hybrid_ai.video import VideoInfo, estimate_workload

import cv2
import numpy as np
import pytest

from classroom_hybrid_ai.evidence import extract_frames_at_times
from classroom_hybrid_ai.performance import PerformanceRecorder, recording
from classroom_hybrid_ai.video import detect_cpu_change_anchors, scan_and_materialize_screenshots
from classroom_hybrid_ai.stage_runtime import AdaptiveScreenshotScheduler
from classroom_hybrid_ai.locate_tracking import materialize_scheduled_screenshots


def test_stride_ten_means_every_tenth_frame() -> None:
    info = VideoInfo("demo.mp4", 30.0, 108_000, 3600.0, 1920, 1080)
    result = estimate_workload(
        info,
        average_tracks=30,
        fixed_stride=10,
        periodic_seconds=3.0,
        tracks_per_periodic_audit=4,
        estimated_anomalies_per_minute=2,
    )
    assert result["sampled_frame_count"] == 10_800
    assert result["fixed_stride_all_tracks_qwen_requests"] == 324_000
    assert result["estimated_hybrid_qwen_requests"] < 5_000


def _video(path, frames=300, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((48, 64, 3), (index // 25 % 3) * 90, dtype=np.uint8)
        cv2.rectangle(image, (index % 40, 10), (index % 40 + 8, 20), (10, 220, 30), -1)
        writer.write(image)
    writer.release()


def test_sparse_targets_match_pixels_pts_indices_and_decode_fewer_frames(tmp_path):
    pytest.importorskip("av")
    source = tmp_path / "cfr.avi"
    _video(source)
    targets = [0.0, 15.05, 29.9, 33.0]
    reference = extract_frames_at_times(source, targets)
    recorder = PerformanceRecorder("sparse-test")
    with recording(recorder):
        sparse = extract_frames_at_times(source, targets, decode_strategy="auto")
    assert set(reference) == set(sparse)
    for target in reference:
        assert reference[target][:2] == sparse[target][:2]
        np.testing.assert_array_equal(reference[target][2], sparse[target][2])
    assert recorder.counts["decoded_frames"] < 30
    assert recorder.counts["video_seeks"] >= 2


def test_unverified_timestamps_fall_back_to_sequential(tmp_path, monkeypatch):
    import classroom_hybrid_ai.video as video_module
    source = tmp_path / "fallback.avi"
    _video(source, frames=40)
    monkeypatch.setattr(video_module, "_constant_frame_timing", lambda *args: None)
    recorder = PerformanceRecorder("fallback-test")
    with recording(recorder):
        sparse = extract_frames_at_times(source, [0.0, 2.2, 10.0], decode_strategy="auto")
    reference = extract_frames_at_times(source, [0.0, 2.2, 10.0])
    assert recorder.counts["sequential_decode_passes"] == 1
    for target in reference:
        assert reference[target][:2] == sparse[target][:2]
        np.testing.assert_array_equal(reference[target][2], sparse[target][2])


def test_failed_opencv_seek_restarts_sequentially_without_relabeling(tmp_path, monkeypatch):
    import classroom_hybrid_ai.video as video_module
    source = tmp_path / "seek-failure.avi"
    _video(source)
    targets = [0.0, 15.05, 29.9]
    expected = extract_frames_at_times(source, targets)
    original_capture = cv2.VideoCapture

    class UnsupportedSeek:
        def __init__(self, path):
            self.capture = original_capture(path)

        def __getattr__(self, name):
            return getattr(self.capture, name)

        def set(self, key, value):
            if key == cv2.CAP_PROP_POS_FRAMES and value > 0:
                return False
            return self.capture.set(key, value)

    monkeypatch.setattr(video_module, "_constant_frame_timing", lambda *args: (10.0, 300))
    monkeypatch.setattr(video_module.cv2, "VideoCapture", UnsupportedSeek)
    recorder = PerformanceRecorder("seek-failure")
    with recording(recorder):
        actual = extract_frames_at_times(source, targets, decode_strategy="auto")
    assert recorder.counts["seek_fallbacks"] == 1
    for target in expected:
        assert actual[target][:2] == expected[target][:2]
        np.testing.assert_array_equal(actual[target][2], expected[target][2])


def test_vfr_packet_timeline_uses_sequential_reader(tmp_path):
    av = pytest.importorskip("av")
    from fractions import Fraction
    from classroom_hybrid_ai.video import _constant_frame_timing
    source = tmp_path / "vfr.mp4"
    with av.open(str(source), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for index, pts in enumerate((0, 1, 3, 4, 8)):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 40, np.uint8), format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    stat = source.stat()
    assert _constant_frame_timing(str(source), stat.st_size, stat.st_mtime_ns) is None
    reference = extract_frames_at_times(source, [0.0, 0.25, 0.6, 4.0])
    candidate = extract_frames_at_times(source, [0.0, 0.25, 0.6, 4.0], decode_strategy="auto")
    for target in reference:
        assert reference[target][:2] == candidate[target][:2]
        np.testing.assert_array_equal(reference[target][2], candidate[target][2])


@pytest.mark.parametrize("cache_bytes,passes", [(2 * 1024 ** 3, 1), (0, 2)])
def test_shared_s1_scan_is_exact_and_cache_overflow_falls_back(tmp_path, cache_bytes, passes):
    source = tmp_path / "s1.avi"
    _video(source, frames=100)
    scheduler = AdaptiveScreenshotScheduler(fps=10, frame_count=100, duration_seconds=10, locate_cap=20, reserved_change_events=5)
    options = {"threshold": 0.08, "sample_stride": 5, "minimum_gap_seconds": 0.5, "maximum_anchors": 5}
    anchors = detect_cpu_change_anchors(source, fps=10, frame_count=100, **options)
    expected, _ = materialize_scheduled_screenshots(source, tmp_path / "old", scheduler.schedule(change_anchors=anchors))
    rows, sampling, found = scan_and_materialize_screenshots(source, tmp_path / "new", scheduler=scheduler, maximum_cache_bytes=cache_bytes, change_options=options)
    assert found == anchors
    assert sampling["decode_passes"] == passes
    assert sampling["cache_peak_bytes"] <= cache_bytes
    assert [{key: value for key, value in row.items() if key != "path"} for row in rows] == [{key: value for key, value in row.items() if key != "path"} for row in expected]
    from pathlib import Path
    for actual, old in zip(rows, expected):
        assert Path(actual["path"]).read_bytes() == Path(old["path"]).read_bytes()
    assert not list(tmp_path.glob("s1-cache-*"))


def test_empty_evidence_targets_never_open_video(tmp_path):
    assert extract_frames_at_times(tmp_path / "missing.mp4", [], decode_strategy="auto") == {}
