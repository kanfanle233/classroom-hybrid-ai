"""Optional, bounded performance observations independent of action labels.

Counters describe measured work, not estimates. Timings may be nested and must
not be summed to manufacture an end-to-end wall time.
"""
from __future__ import annotations

import contextvars
import math
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

_ACTIVE: contextvars.ContextVar[PerformanceRecorder | None] = contextvars.ContextVar("performance", default=None)


@dataclass
class PerformanceRecorder:
    scope: str
    timings_seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)
    model_requests: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def timing(self, name: str, seconds: float) -> None:
        if math.isfinite(seconds) and seconds >= 0:
            with self._lock:
                self.timings_seconds[name] = self.timings_seconds.get(name, 0.0) + seconds

    def count(self, name: str, value: float = 1) -> None:
        if math.isfinite(value):
            with self._lock:
                self.counts[name] = self.counts.get(name, 0) + value

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": "classroom_hybrid_ai.performance_stage.v1",
            "scope": self.scope,
            "timings_seconds": dict(sorted(self.timings_seconds.items())),
            "counts": dict(sorted(self.counts.items())),
            "model_requests": list(self.model_requests),
            "timing_semantics": "nested_observed_durations_do_not_sum",
        }


@contextmanager
def recording(recorder: PerformanceRecorder) -> Iterator[PerformanceRecorder]:
    token = _ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.reset(token)


def record_timing(name: str, seconds: float) -> None:
    recorder = _ACTIVE.get()
    if recorder is not None:
        recorder.timing(name, float(seconds))


def increment(name: str, value: float = 1) -> None:
    recorder = _ACTIVE.get()
    if recorder is not None:
        recorder.count(name, value)


@contextmanager
def measured(name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        record_timing(name, time.perf_counter() - started)


def record_model_performance(model: str, value: Mapping[str, Any] | None, *, wall_seconds: float) -> None:
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    row = {**dict(value or {}), "model": model, "wall_seconds": wall_seconds}
    with recorder._lock:
        recorder.model_requests.append(row)
    recorder.count(f"{model}.request_attempts")
    counts = (value or {}).get("counts", {})
    calls = counts.get("inference_calls") if isinstance(counts, Mapping) else None
    if isinstance(calls, (int, float)) and not isinstance(calls, bool) and math.isfinite(calls) and calls >= 0:
        recorder.count(f"{model}.actual_calls", calls)
    else:
        recorder.count(f"{model}.requests_without_inference_count")
    recorder.timing(f"{model}.request_seconds", wall_seconds)


class ResourceSampler:
    """Sample process-tree RSS and optional device-wide GPU usage off-thread.

    RSS is the sum of reported working sets and can include shared pages. GPU
    samples include desktop applications under WDDM; allocator peaks come from
    the model worker separately. Neither sampling series is an instantaneous
    hardware peak measurement.
    """
    def __init__(self, *, gpu: bool = False, interval: float = 0.2) -> None:
        self.gpu = gpu
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self.missing_reasons: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peaks: dict[str, float] = {}
        self._first: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def _observe(self, row: dict[str, float]) -> None:
        # Lifetime extrema survive downsampling of the bounded raw series.
        for key, value in row.items():
            if key == "monotonic_seconds":
                continue
            self._peaks[key] = max(self._peaks.get(key, value), value)
            self._first.setdefault(key, value)
            self._last[key] = value
        self.samples.append(row)
        if len(self.samples) > 18000:
            self.samples[1:9000] = self.samples[1:9000:2]

    def __enter__(self) -> ResourceSampler:
        self._thread = threading.Thread(target=self._run, name="performance-sampler", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            import psutil
            parent = psutil.Process()
        except ImportError:
            self.missing_reasons["process_tree_rss_bytes"] = "psutil_unavailable"
            return
        executable = shutil.which("nvidia-smi") if self.gpu else None
        if self.gpu and executable is None:
            self.missing_reasons["device_total_gpu_mib"] = "nvidia_smi_unavailable"
        last_gpu = -math.inf
        while not self._stop.is_set():
            row: dict[str, float] = {"monotonic_seconds": time.monotonic()}
            try:
                processes = [parent, *parent.children(recursive=True)]
                memory = 0
                for process in processes:
                    try:
                        memory += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                row["process_tree_rss_bytes"] = memory
                row["system_available_ram_bytes"] = psutil.virtual_memory().available
                io = parent.io_counters()
                row["parent_read_bytes"] = io.read_bytes
                row["parent_write_bytes"] = io.write_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError) as exc:
                self.missing_reasons["process_metrics"] = type(exc).__name__
            if executable and row["monotonic_seconds"] - last_gpu >= 1:
                last_gpu = row["monotonic_seconds"]
                try:
                    result = subprocess.run(
                        [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
                        capture_output=True, text=True, timeout=3,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if result.returncode == 0:
                        row["device_total_gpu_mib"] = float(result.stdout.strip().splitlines()[0])
                except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                    self.missing_reasons["device_total_gpu_mib"] = type(exc).__name__
            self._observe(row)
            self._stop.wait(self.interval)

    def __exit__(self, *args: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4)

    def payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_count": len(self.samples), "interval_seconds": self.interval,
            "scope": "process_tree_working_set_sum_and_device_total_WDDM",
            "missing_reasons": dict(self.missing_reasons),
        }
        for key in ("process_tree_rss_bytes", "device_total_gpu_mib"):
            if key in self._peaks:
                result[f"sampled_peak_{key}"] = self._peaks[key]
                result[f"first_{key}"] = self._first[key]
                result[f"last_{key}"] = self._last[key]
            else:
                result["missing_reasons"].setdefault(key, "no_observation")
        for key in ("parent_read_bytes", "parent_write_bytes"):
            if key in self._first:
                result[f"observed_delta_{key}"] = max(0, self._last[key] - self._first[key])
            else:
                result["missing_reasons"].setdefault(key, "no_observation")
        result["io_scope"] = "parent_process_counter_delta_between_first_and_last_sample; excludes worker IO"
        return result
