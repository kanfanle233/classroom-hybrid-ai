"""Screenshot YOLO process with an acknowledged, verified CUDA boundary."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any, Mapping, Sequence


class YoloWorkerError(RuntimeError):
    pass


def _worker_main(connection: Any, adapter_kwargs: Mapping[str, Any]) -> None:
    adapter = None
    try:
        from .adapters.yolo import YoloTrackerAdapter
        adapter = YoloTrackerAdapter(**dict(adapter_kwargs))
        while True:
            request = connection.recv()
            if request.get("op") == "close":
                try:
                    adapter.close()
                    adapter = None
                    connection.send({"ok": True, "result": None, "pid": os.getpid()})
                except BaseException as exc:
                    connection.send({"ok": False, "error": f"{type(exc).__name__}:{exc}"})
                break
            try:
                if request.get("op") != "predict_batch":
                    raise ValueError("unknown_yolo_operation")
                started = time.perf_counter()
                loaded_before = adapter.loaded
                if not loaded_before:
                    adapter._ensure_model()
                model_load = time.perf_counter() - started if not loaded_before else 0.0
                result = adapter.predict_batch(request["images"], **request["kwargs"])
                memory: dict[str, Any] = {}
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory = {
                            "allocated_bytes": int(torch.cuda.memory_allocated()),
                            "reserved_bytes": int(torch.cuda.memory_reserved()),
                            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                            "num_ooms": int(torch.cuda.memory_stats().get("num_ooms", 0)),
                            "scope": "worker PyTorch allocator; excludes WDDM and other processes",
                        }
                except Exception as exc:
                    memory["metrics_error"] = f"{type(exc).__name__}:{exc}"[:500]
                connection.send({
                    "ok": True, "result": result, "pid": os.getpid(),
                    "performance": {"model_load_seconds": model_load,
                                    "request_wall_seconds": time.perf_counter() - started,
                                    "memory": memory},
                })
            except BaseException as exc:
                connection.send({"ok": False, "error": f"{type(exc).__name__}:{exc}"[:4000]})
    except (EOFError, OSError):
        pass
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass
        connection.close()


class YoloProcessClient:
    """Own one worker for all YOLO screenshots, then require normal exit."""

    def __init__(self, adapter_kwargs: Mapping[str, Any], *, timeout: float = 300.0) -> None:
        self.adapter_kwargs = dict(adapter_kwargs)
        self.timeout = float(timeout)
        self._connection: Any = None
        self._process: Any = None
        self._closed = False
        self.last_performance: dict[str, Any] = {}
        self.receipt: dict[str, Any] = {
            "schema_version": "classroom_hybrid_ai.yolo_worker_lifecycle.v1",
            "owned_process": True, "model_started": False, "pid": None,
            "close_acknowledged": False, "exitcode": None,
        }

    def _start(self) -> None:
        if self._closed:
            raise YoloWorkerError("yolo_worker_already_closed")
        if self._process is not None:
            return
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_worker_main, args=(child, self.adapter_kwargs), name="classroom-yolo-worker")
        process.daemon = True
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self._connection, self._process = parent, process
        self.receipt.update({"model_started": True, "pid": process.pid})

    def predict_batch(self, images: Sequence[str], **kwargs: Any) -> list[dict[str, Any]]:
        self._start()
        self._connection.send({"op": "predict_batch", "images": list(images), "kwargs": kwargs})
        deadline = time.monotonic() + self.timeout
        while not self._connection.poll(0.25):
            if time.monotonic() >= deadline:
                raise TimeoutError("yolo_worker_request_timeout")
            if not self._process.is_alive():
                raise YoloWorkerError(f"yolo_worker_exited:{self._process.exitcode}")
        response = self._connection.recv()
        if not isinstance(response, Mapping) or response.get("ok") is not True:
            raise YoloWorkerError(f"yolo_worker_request_failed:{response}")
        self.last_performance = dict(response.get("performance", {}))
        result = response.get("result")
        if not isinstance(result, list) or len(result) != len(images):
            raise YoloWorkerError("yolo_worker_result_count_mismatch")
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process, connection = self._process, self._connection
        if process is None:
            self.receipt.update({"close_acknowledged": True, "exitcode": 0, "status": "not_started"})
            return
        failure: BaseException | None = None
        try:
            if not process.is_alive():
                raise YoloWorkerError(f"yolo_worker_exited_without_close_ack:{process.exitcode}")
            connection.send({"op": "close"})
            if not connection.poll(30.0):
                raise YoloWorkerError("yolo_worker_close_ack_timeout")
            response = connection.recv()
            if not isinstance(response, Mapping) or response.get("ok") is not True:
                raise YoloWorkerError(f"yolo_worker_close_failed:{response}")
            self.receipt["close_acknowledged"] = True
            process.join(timeout=30.0)
            if process.is_alive() or type(process.exitcode) is not int or process.exitcode != 0:
                raise YoloWorkerError(f"yolo_worker_normal_exit_missing:{process.exitcode}")
        except BaseException as exc:
            failure = exc
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
            self.receipt.update({"exitcode": process.exitcode,
                                 "status": "failed" if failure else "ok",
                                 "close_error": str(failure) if failure else None})
            connection.close()
            self._connection, self._process = None, None
        if failure is not None:
            raise failure
