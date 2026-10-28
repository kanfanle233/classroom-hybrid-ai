"""Lifecycle control for the isolated LocateAnything HTTP worker.

The production pipeline owns a worker only when it had to start one.  An
already-running compatible worker is reused, never terminated by this class.
"""

from __future__ import annotations

import json
import importlib.resources as importlib_resources
import os
import subprocess
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .config import ProjectConfig


class LocateServiceError(RuntimeError):
    """The configured isolated LocateAnything worker could not be prepared."""


def _health_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise LocateServiceError("locate_health_response_must_be_object")
    result = dict(payload)
    if result.get("status") != "ok":
        raise LocateServiceError(f"locate_health_not_ready:{result.get('status')!r}")
    return result


def _shutdown_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=b"{}",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise LocateServiceError("locate_shutdown_response_must_be_object")
    return dict(payload)


class LocateServiceSession:
    """Start a local worker when absent and stop only the process we own."""

    def __init__(
        self,
        config: ProjectConfig,
        run_dir: str | Path,
        *,
        health_probe: Callable[[str, float], dict[str, Any]] = _health_json,
        shutdown_request: Callable[[str, float], dict[str, Any]] = _shutdown_json,
        popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.health_probe = health_probe
        self.shutdown_request = shutdown_request
        self.popen_factory = popen_factory
        self.sleep = sleep
        self.process: subprocess.Popen[Any] | None = None
        self._log_handle: BinaryIO | None = None
        self.receipt: dict[str, Any] = {
            "status": "not_started",
            "backend": str(config.get("locate_anything.backend", "http")),
            "owned_process": False,
            "pid": None,
        }

    def __enter__(self) -> "LocateServiceSession":
        backend = str(self.config.get("locate_anything.backend", "http"))
        if backend != "http":
            self.receipt.update({"status": "not_applicable", "reason": "non_http_backend"})
            return self

        health_url = str(self.config.get("locate_anything.health_url", "")).strip()
        if not health_url:
            raise LocateServiceError("locate_health_url_missing")
        expected_model = self.config.model_path("locate_anything.model_dir")
        if expected_model is None or not expected_model.is_dir():
            raise LocateServiceError(f"locate_model_dir_missing:{expected_model}")

        try:
            try:
                health = self.health_probe(health_url, 1.0)
            except Exception as first_error:
                if not bool(self.config.get("locate_anything.auto_start_service", True)):
                    self.receipt.update({
                        "status": "unavailable",
                        "reason": f"{type(first_error).__name__}:{first_error}",
                        "auto_start_enabled": False,
                    })
                    return self
                self._start_worker(expected_model, health_url)
                health = self._wait_until_ready(health_url)
            self._validate_worker_model(health, expected_model)
        except Exception:
            # Python does not call __exit__ when __enter__ fails.  Tear down a
            # worker started by this attempt here so a failed health/model
            # check cannot leave Locate occupying VRAM before Qwen starts.
            if self.process is not None:
                self._stop_owned_process(status="start_failed_cleaned_up")
            self._close_log()
            raise
        self.receipt.update({
            "status": "ready",
            "health": health,
            "model_dir": str(expected_model.resolve()),
        })
        return self

    def _start_worker(self, model_dir: Path, health_url: str) -> None:
        parsed = urlparse(health_url)
        host = parsed.hostname or "127.0.0.1"
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise LocateServiceError(f"locate_auto_start_requires_loopback_host:{host}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        script = self.config.project_root / "scripts" / "locate_anything_service.py"
        if not script.is_file():
            packaged = importlib_resources.files("classroom_hybrid_ai.resources").joinpath("locate_anything_service.py")
            if packaged.is_file():
                script = Path(str(packaged))
            else:
                raise LocateServiceError(f"locate_service_script_missing:{script}")
        python = self.config.python_executable
        if not python.is_file():
            raise LocateServiceError(f"locate_python_missing:{python}")

        artifact_dir = self.run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_path = artifact_dir / "locate_service.log"
        self._log_handle = log_path.open("ab")
        command = [
            str(python),
            str(script),
            "--model-dir",
            str(model_dir),
            "--host",
            host,
            "--port",
            str(port),
        ]
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        try:
            self.process = self.popen_factory(
                command,
                cwd=str(self.config.project_root),
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except Exception:
            self._close_log()
            raise
        self.receipt.update({
            "status": "starting",
            "owned_process": True,
            "pid": getattr(self.process, "pid", None),
            "command": command,
            "log_path": str(log_path),
            "python_executable": str(python),
        })

    def _wait_until_ready(self, health_url: str) -> dict[str, Any]:
        timeout = max(
            1.0,
            float(self.config.get("locate_anything.service_start_timeout_seconds", 30.0)),
        )
        deadline = time.monotonic() + timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise LocateServiceError(
                    f"locate_service_exited_during_start:{self.process.returncode}"
                )
            try:
                return self.health_probe(health_url, min(1.0, timeout))
            except Exception as exc:
                last_error = exc
                self.sleep(0.25)
        raise LocateServiceError(
            f"locate_service_start_timeout:{type(last_error).__name__}:{last_error}"
        )

    @staticmethod
    def _validate_worker_model(health: Mapping[str, Any], expected_model: Path) -> None:
        actual_value = health.get("model_dir")
        if not isinstance(actual_value, str) or not actual_value.strip():
            raise LocateServiceError("locate_health_model_dir_missing")
        actual = Path(actual_value).resolve()
        if actual != expected_model.resolve():
            raise LocateServiceError(
                f"locate_service_model_mismatch:expected={expected_model.resolve()}:actual={actual}"
            )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        process = self.process
        if process is not None:
            self._stop_owned_process(status="stopped")
        elif self.receipt.get("status") == "ready":
            self.receipt.update({
                "status": "reused_existing_service",
                "stopped_after_locate_stage": False,
            })
        self._close_log()
        # Preserve the original pipeline failure. A teardown failure remains
        # visible in the receipt and cannot make us hide a Locate/Qwen error.
        return False

    def _stop_owned_process(self, *, status: str) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    shutdown = self.shutdown_request(
                        str(self.config.get("locate_anything.shutdown_url")), 5.0
                    )
                    acknowledged = (
                        shutdown.get("status") == "stopping"
                        and shutdown.get("loaded") is False
                    )
                    self.receipt.update({
                        "graceful_shutdown": shutdown,
                        "graceful_shutdown_acknowledged": acknowledged,
                    })
                    if not acknowledged:
                        raise LocateServiceError("locate_shutdown_not_acknowledged")
                    process.wait(timeout=10.0)
                except Exception as shutdown_exc:
                    self.receipt["graceful_shutdown_error"] = (
                        f"{type(shutdown_exc).__name__}:{shutdown_exc}"
                    )
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5.0)
            self.receipt.update({
                "status": status,
                "exitcode": process.returncode,
                "stopped_after_locate_stage": status == "stopped",
            })
        except Exception as stop_exc:
            self.receipt.update({
                "status": "stop_failed",
                "stop_error": f"{type(stop_exc).__name__}:{stop_exc}",
            })

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


__all__ = ["LocateServiceError", "LocateServiceSession"]
