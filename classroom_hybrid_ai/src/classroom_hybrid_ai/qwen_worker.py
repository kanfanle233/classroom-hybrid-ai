"""One-video Qwen worker process.

The local 9B model is intentionally owned by a short-lived child process.  The
parent sends one review request at a time over a pipe; every request contains a
single current user message because :class:`Qwen35Adapter` does not retain
conversation state.  Keeping the model in the child makes the end of a video a
hard CUDA-context boundary instead of relying on allocator cache cleanup in a
long-running orchestration process.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any, Mapping


class QwenWorkerError(RuntimeError):
    """A request failed inside the isolated Qwen process."""


def _apply_cuda_alloc_conf(alloc_conf: str) -> str | None:
    """Apply the allocator tuning before the worker touches CUDA.

    ``PYTORCH_CUDA_ALLOC_CONF`` is only read when the allocator is created, so
    this has to run inside the child process ahead of any torch import.  An
    operator-supplied value always wins; the configured default is a fallback
    for the isolated worker only and never leaks into the parent process.
    """

    value = str(alloc_conf or "").strip()
    if not value:
        return None
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if existing:
        return existing
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value
    return value


def _cuda_memory_metrics(*, probe: bool = True) -> dict[str, Any]:
    """Return allocator metrics from the process that owns the model."""

    metrics: dict[str, Any] = {
        "allocated_bytes": None,
        "reserved_bytes": None,
        "inactive_split_bytes": None,
        "num_alloc_retries": None,
        "num_ooms": None,
        "max_allocated_bytes": None,
        "max_reserved_bytes": None,
        "request_peak_allocated_bytes": None,
        "request_peak_reserved_bytes": None,
        "scope": "PyTorch allocator in owning worker; excludes other processes and WDDM driver allocations",
    }
    if not probe:
        return metrics
    try:
        import torch

        if not torch.cuda.is_available():
            return metrics
        metrics["allocated_bytes"] = int(torch.cuda.memory_allocated())
        metrics["reserved_bytes"] = int(torch.cuda.memory_reserved())
        metrics["max_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        metrics["max_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        metrics["request_peak_allocated_bytes"] = metrics["max_allocated_bytes"]
        metrics["request_peak_reserved_bytes"] = metrics["max_reserved_bytes"]
        stats = torch.cuda.memory_stats()
        for name in ("inactive_split_bytes.all.current", "inactive_split_bytes"):  # version variants
            if name in stats:
                metrics["inactive_split_bytes"] = int(stats[name])
                break
        for name in ("num_alloc_retries", "num_alloc_retries.all"):  # version variants
            if name in stats:
                metrics["num_alloc_retries"] = int(stats[name])
                break
        for name in ("num_ooms", "num_ooms.all"):  # version variants
            if name in stats:
                metrics["num_ooms"] = int(stats[name])
                break
    except Exception as exc:  # metrics are audit data, never a request failure
        metrics["metrics_error"] = f"{type(exc).__name__}:{exc}"[:500]
    return metrics


def _reset_cuda_peaks() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:4000],
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:],
    }


def _worker_main(connection: Connection, adapter_kwargs: Mapping[str, Any]) -> None:
    """Child entrypoint; it must remain top-level for Windows spawn."""

    adapter = None
    stage_peaks: dict[str, int] = {}
    lifecycle: dict[str, Any] = {
        "schema_version": "classroom_hybrid_ai.qwen_worker_lifecycle.v1",
        "worker_pid": os.getpid(),
        "timings_seconds": {"model_load": 0.0, "warmup": 0.0},
        "counts": {"model_load_attempts": 0, "model_load_successes": 0, "warmup_requests": 0},
        "scope": "cumulative_per_worker; deduplicate_by_worker_pid; timings_may_overlap",
    }

    def capture_lifecycle(operation: str, loaded_before: bool, started: float) -> None:
        performance = adapter.last_performance
        load_seconds = performance.get("timings_seconds", {}).get("model_load")
        if not loaded_before and isinstance(load_seconds, (int, float)) and load_seconds > 0:
            lifecycle["timings_seconds"]["model_load"] += load_seconds
            lifecycle["counts"]["model_load_attempts"] += 1
            lifecycle["counts"]["model_load_successes"] += int(bool(adapter.loaded))
        if operation == "warmup":
            lifecycle["timings_seconds"]["warmup"] += time.perf_counter() - started
            lifecycle["counts"]["warmup_requests"] += 1
        lifecycle["model_loaded"] = bool(adapter.loaded)
        performance["worker_lifecycle"] = lifecycle

    def collect_metrics() -> dict[str, Any]:
        metrics = _cuda_memory_metrics()
        for key in ("max_allocated_bytes", "max_reserved_bytes"):
            value = metrics.get(key)
            if isinstance(value, int):
                stage_peaks[key] = max(stage_peaks.get(key, 0), value)
                metrics[key] = stage_peaks[key]
        return metrics
    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Must precede the adapter import: the allocator reads its config the
        # first time CUDA initializes inside this process.
        worker_kwargs = dict(adapter_kwargs)
        _apply_cuda_alloc_conf(str(worker_kwargs.pop("cuda_alloc_conf", "") or ""))
        started = time.perf_counter()
        from .adapters.qwen import Qwen35Adapter
        lifecycle["timings_seconds"]["adapter_import"] = time.perf_counter() - started
        started = time.perf_counter()
        adapter = Qwen35Adapter(**worker_kwargs)
        lifecycle["timings_seconds"]["adapter_initialization"] = time.perf_counter() - started
        while True:
            try:
                request = connection.recv()
            except (EOFError, OSError):
                break
            if not isinstance(request, dict):
                response = {"ok": False, "error": {"type": "TypeError", "message": "worker_request_must_be_mapping"}}
                try:
                    connection.send(response)
                except (EOFError, OSError):
                    break
                continue
            operation = request.get("op")
            if operation == "close":
                failed_load = bool(
                    adapter is not None
                    and getattr(adapter, "_load_failure", None) is not None
                    and not bool(getattr(adapter, "loaded", False))
                )
                if failed_load:
                    # ``Qwen35Adapter._load_local`` records the exception and
                    # may have imported CUDA/Transformers without producing a
                    # model.  Calling adapter.close() here can block in
                    # interpreter teardown on Windows.  There are no live
                    # model weights to release; acknowledge the close and let
                    # the failed-load worker exit at the process boundary.
                    adapter = None
                    try:
                        # The failed warmup response already carried allocator
                        # diagnostics.  Probing CUDA again during close can
                        # block behind another test/process on Windows and
                        # exhaust the parent's bounded shutdown grace period,
                        # converting this clean path into exit code -15.
                        connection.send({"ok": True, "result": None, "metrics": None})
                    except (EOFError, OSError):
                        pass
                    try:
                        connection.close()
                    finally:
                        os._exit(0)
                try:
                    adapter.close()
                    adapter.last_performance["worker_lifecycle"] = lifecycle
                    performance = dict(adapter.last_performance)
                    adapter = None
                    connection.send({"ok": True, "result": None, "metrics": collect_metrics(), "performance": performance})
                except BaseException as exc:
                    try:
                        connection.send({"ok": False, "error": _error_payload(exc), "metrics": _cuda_memory_metrics()})
                    except (EOFError, OSError):
                        pass
                break
            operation_started = time.perf_counter()
            loaded_before = bool(adapter.loaded)
            try:
                metrics_started = time.perf_counter()
                _reset_cuda_peaks()
                lifecycle["timings_seconds"].setdefault("first_cuda_metrics_initialization", time.perf_counter() - metrics_started)
                if operation == "warmup":
                    adapter.warmup()
                    result: Any = None
                elif operation == "complete":
                    # Text-only semantic adjudication uses the same isolated
                    # model process and the same stateless JSON contract as
                    # visual review.  Keeping this operation in the worker is
                    # essential: the production pipeline passes a
                    # QwenProcessClient into the two-pass synonym adjudicator.
                    result = adapter.complete(
                        str(request["prompt"]),
                        images=tuple(request.get("images", ())),
                        max_new_tokens=request.get("max_new_tokens"),
                    )
                elif operation == "review":
                    result = adapter.review(
                        str(request["candidate_event_id"]),
                        request.get("evidence", {}),
                        event_type=request.get("event_type"),
                        evidence_frame_indices=tuple(request.get("evidence_frame_indices", ())),
                        prompt=request.get("prompt"),
                        compact_response=bool(request.get("compact_response", False)),
                    )
                else:
                    raise ValueError(f"unknown_qwen_worker_operation:{operation}")
                metrics = collect_metrics()
                adapter.last_performance["operation"] = operation
                adapter.last_performance["memory"] = metrics
                response = {"ok": True, "result": result, "metrics": metrics, "performance": adapter.last_performance}
            except BaseException as exc:
                metrics = collect_metrics()
                adapter.last_performance["operation"] = operation
                adapter.last_performance["memory"] = metrics
                adapter.last_performance["error"] = f"{type(exc).__name__}:{exc}"[:500]
                response = {"ok": False, "error": _error_payload(exc), "metrics": metrics, "performance": adapter.last_performance}
            capture_lifecycle(str(operation), loaded_before, operation_started)
            try:
                connection.send(response)
            except (EOFError, OSError):
                break
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass
        try:
            connection.close()
        except Exception:
            pass


class QwenProcessClient:
    """Parent-side adapter facade for one short-lived Qwen worker.

    The facade deliberately implements only the methods consumed by the
    pipeline.  ``last_metrics`` is populated by the child after every request,
    which lets the smoke gate measure the allocator that actually owns Qwen.
    """

    def __init__(
        self,
        adapter_kwargs: Mapping[str, Any],
        *,
        timeout: float = 300.0,
        start_method: str = "spawn",
    ) -> None:
        self.adapter_kwargs = dict(adapter_kwargs)
        self.timeout = max(1.0, float(timeout))
        self.start_method = start_method
        self._connection: Connection | None = None
        self._process: mp.Process | None = None
        self._closed = False
        self._exitcode: int | None = None
        self.last_metrics: dict[str, Any] = _cuda_memory_metrics(probe=False)
        self.last_performance: dict[str, Any] = {}
        self.worker_lifecycle: dict[str, Any] = {}
        self._parent_lifecycle_timings: dict[str, float] = {}
        self._started_at: float | None = None

    @property
    def loaded(self) -> bool:
        return self._process is not None and self._process.is_alive() and not self._closed

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def exitcode(self) -> int | None:
        if self._process is not None:
            return self._process.exitcode
        return self._exitcode

    def start(self) -> None:
        if self._closed:
            raise QwenWorkerError("qwen_worker_already_closed")
        if self._process is not None:
            return
        self._started_at = time.perf_counter()
        try:
            context = mp.get_context(self.start_method)
            parent, child = context.Pipe(duplex=True)
            process = context.Process(target=_worker_main, args=(child, self.adapter_kwargs), name="classroom-qwen-worker")
            process.daemon = True
            process.start()
            child.close()
        except Exception as exc:
            self._connection = None
            self._process = None
            raise QwenWorkerError(f"qwen_worker_start_failed:{type(exc).__name__}:{exc}") from exc
        self._connection = parent
        self._process = process
        self._parent_lifecycle_timings["process_spawn"] = time.perf_counter() - self._started_at

    def _accept_performance(self, performance: Any) -> None:
        if not isinstance(performance, dict):
            return
        self.last_performance = dict(performance)
        lifecycle = performance.get("worker_lifecycle")
        if isinstance(lifecycle, dict):
            self.worker_lifecycle = {
                **lifecycle,
                "parent_timings_seconds": dict(self._parent_lifecycle_timings),
            }
            self.last_performance["worker_lifecycle"] = self.worker_lifecycle

    def _request(self, payload: dict[str, Any]) -> Any:
        self.start()
        assert self._connection is not None
        assert self._process is not None
        try:
            self._connection.send(payload)
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("qwen_worker_request_timeout")
                if self._connection.poll(min(remaining, 0.5)):
                    response = self._connection.recv()
                    break
                if not self._process.is_alive():
                    raise QwenWorkerError(f"qwen_worker_exited:{self._process.exitcode}")
        except TimeoutError:
            # TimeoutError is an OSError subclass. Handle it first so a late
            # reply cannot be mistaken for the following close ACK.
            self._terminate()
            raise
        except (EOFError, OSError) as exc:
            raise QwenWorkerError(f"qwen_worker_pipe_failed:{type(exc).__name__}:{exc}") from exc
        if not isinstance(response, dict):
            raise QwenWorkerError("qwen_worker_response_must_be_mapping")
        metrics = response.get("metrics")
        if isinstance(metrics, dict):
            self.last_metrics = dict(metrics)
        performance = response.get("performance")
        if payload.get("op") == "warmup" and self._started_at is not None:
            self._parent_lifecycle_timings.setdefault("startup_to_warmup_ack", time.perf_counter() - self._started_at)
        self._accept_performance(performance)
        if not response.get("ok"):
            error = response.get("error", {})
            if isinstance(error, Mapping):
                message = f"{error.get('type', 'QwenWorkerError')}:{error.get('message', '')}"
            else:
                message = str(error)
            raise QwenWorkerError(message[:5000])
        return response.get("result")

    def warmup(self) -> None:
        self._request({"op": "warmup"})

    def review(
        self,
        candidate_event_id: str,
        evidence: Any,
        *,
        event_type: str | None = None,
        evidence_frame_indices: Any = (),
        prompt: str | None = None,
        compact_response: bool = False,
    ) -> dict[str, Any]:
        result = self._request({
            "op": "review",
            "candidate_event_id": candidate_event_id,
            "evidence": evidence,
            "event_type": event_type,
            "evidence_frame_indices": list(evidence_frame_indices),
            "prompt": prompt,
            "compact_response": compact_response,
        })
        if not isinstance(result, dict):
            raise QwenWorkerError("qwen_worker_review_result_must_be_mapping")
        return result

    def complete(
        self,
        prompt: str,
        *,
        images: Any = (),
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run one stateless JSON completion in the isolated worker.

        The pipeline uses this for text-only candidate synonym adjudication;
        it deliberately shares the same child process as visual reviews so a
        video never creates a second Qwen model instance.
        """

        result = self._request({
            "op": "complete",
            "prompt": prompt,
            "images": list(images),
            "max_new_tokens": max_new_tokens,
        })
        if not isinstance(result, dict):
            raise QwenWorkerError("qwen_worker_complete_result_must_be_mapping")
        return result

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
        process.join(timeout=10.0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        connection = self._connection
        process = self._process
        close_failure: Exception | None = None
        try:
            if connection is not None and process is not None and process.is_alive():
                try:
                    self._request_close(connection, process)
                except Exception as exc:
                    # A child can report a teardown failure and still exit
                    # normally.  Preserve the protocol error independently
                    # of its exit code, after releasing all owned handles.
                    close_failure = exc
                    self._terminate()
            elif process is not None:
                process.join(timeout=1.0)
        finally:
            if process is not None and process.is_alive():
                self._terminate()
            if process is not None:
                self._exitcode = process.exitcode
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            self._connection = None
            self._process = None
        if close_failure is not None:
            raise close_failure

    def _request_close(self, connection: Connection, process: mp.Process) -> None:
        connection.send({"op": "close"})
        if not connection.poll(10.0):
            raise QwenWorkerError("qwen_worker_close_ack_timeout")
        response = connection.recv()
        if not isinstance(response, dict):
            raise QwenWorkerError("qwen_worker_close_response_must_be_mapping")
        if isinstance(response.get("metrics"), dict):
            self.last_metrics = dict(response["metrics"])
        if isinstance(response.get("performance"), dict):
            self._accept_performance(response["performance"])
        if response.get("ok") is not True:
            error = response.get("error", "missing_success_ack")
            raise QwenWorkerError(f"qwen_worker_close_failed:{error}")
        # The child sends its ACK before the Windows Python/PyTorch runtime
        # finishes interpreter and CUDA-context teardown.  Ten seconds was
        # occasionally too short and converted a clean shutdown into SIGTERM
        # (-15).  Keep a bounded grace period so the delivery gate can require
        # an actual zero exit code.
        process.join(timeout=30.0)

    def __enter__(self) -> "QwenProcessClient":
        self.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = ["QwenProcessClient", "QwenWorkerError", "_apply_cuda_alloc_conf", "_cuda_memory_metrics"]
