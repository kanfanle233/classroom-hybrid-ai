"""Single-GPU process lease and explicit model teardown."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from types import TracebackType
from typing import Any


class GpuLease:
    """Cross-process exclusive lease using atomic file creation.

    The lease stores a PID for diagnosis. Stale files are removed only when the
    recorded process is verifiably absent on the same host.
    """

    def __init__(self, path: Path, *, timeout_seconds: float = 1800.0, poll_seconds: float = 0.25) -> None:
        self.path = path.resolve()
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._owned = False

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            import psutil

            return psutil.pid_exists(pid)
        except Exception:
            return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        payload = json.dumps({"pid": os.getpid(), "created_unix": time.time()})
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(descriptor, payload.encode("utf-8"))
                finally:
                    os.close(descriptor)
                self._owned = True
                return
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                    pid = int(existing.get("pid", -1))
                except Exception:
                    pid = -1
                if pid > 0 and not self._pid_exists(pid):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"gpu_lease_timeout:{self.path}:owner_pid={pid}")
                time.sleep(self.poll_seconds)

    def release(self) -> None:
        if not self._owned:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) == os.getpid():
                self.path.unlink(missing_ok=True)
        finally:
            self._owned = False

    def __enter__(self) -> "GpuLease":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def release_model(model_or_adapter: Any) -> dict[str, Any]:
    """Close an adapter, drop references and clear reusable CUDA allocations."""

    close_error: str | None = None
    if model_or_adapter is not None and hasattr(model_or_adapter, "close"):
        try:
            model_or_adapter.close()
        except Exception as exc:  # teardown should remain auditable
            close_error = f"{type(exc).__name__}:{exc}"
    gc.collect()
    cuda = {"available": False, "allocated_gib": 0.0, "reserved_gib": 0.0}
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            cuda["allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 4)
            cuda["reserved_gib"] = round(torch.cuda.memory_reserved() / 2**30, 4)
    except Exception as exc:
        cuda["teardown_error"] = f"{type(exc).__name__}:{exc}"
    return {"close_error": close_error, "cuda_after_release": cuda}

