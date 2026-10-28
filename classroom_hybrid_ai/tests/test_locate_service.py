from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError

import pytest

from classroom_hybrid_ai.config import ProjectConfig
from classroom_hybrid_ai.locate_service import LocateServiceError, LocateServiceSession


def _config(tmp_path: Path, *, backend: str = "http") -> ProjectConfig:
    model = tmp_path / "model"
    model.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "locate_anything_service.py").write_text("# test worker\n", encoding="utf-8")
    payload = {
        "runtime": {"python_executable": sys.executable},
        "locate_anything": {
            "backend": backend,
            "model_dir": "model",
            "health_url": "http://127.0.0.1:8011/health",
            "auto_start_service": True,
            "service_start_timeout_seconds": 2,
        },
    }
    return ProjectConfig(tmp_path / "default.yaml", tmp_path, tmp_path, payload)


def test_non_http_locate_backend_does_not_start_process(tmp_path: Path) -> None:
    session = LocateServiceSession(
        _config(tmp_path, backend="local"),
        tmp_path / "run",
        popen_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    with session:
        pass
    assert session.receipt["status"] == "not_applicable"
    assert session.receipt["owned_process"] is False


def test_missing_http_worker_is_started_with_configured_python_then_stopped(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls = 0

    def health(_url: str, _timeout: float):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise URLError(ConnectionRefusedError(10061, "refused"))
        return {"status": "ok", "model_dir": str(tmp_path / "model"), "loaded": False}

    class Process:
        pid = 321
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Process()
    launched: list[tuple[list[str], dict]] = []

    def popen(command, **kwargs):
        launched.append((list(command), dict(kwargs)))
        return process

    def shutdown(_url: str, _timeout: float):
        process.returncode = 0
        return {"status": "stopping", "loaded": False}

    session = LocateServiceSession(
        config,
        tmp_path / "run",
        health_probe=health,
        shutdown_request=shutdown,
        popen_factory=popen,
        sleep=lambda _seconds: None,
    )
    with session:
        assert session.receipt["status"] == "ready"
        assert session.receipt["owned_process"] is True
    assert session.receipt["status"] == "stopped"
    assert session.receipt["exitcode"] == 0
    assert launched[0][0][0] == str(config.python_executable)
    assert launched[0][0][2:4] == ["--model-dir", str(tmp_path / "model")]
    assert Path(session.receipt["log_path"]).is_file()


def test_failed_start_cleans_up_owned_process(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def health(_url: str, _timeout: float):
        raise URLError(ConnectionRefusedError(10061, "refused"))

    class Process:
        pid = 654
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Process()
    session = LocateServiceSession(
        config,
        tmp_path / "run",
        health_probe=health,
        shutdown_request=lambda *_args: (_ for _ in ()).throw(URLError("not ready")),
        popen_factory=lambda *_args, **_kwargs: process,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(LocateServiceError, match="start_timeout"):
        with session:
            pass
    assert process.returncode == 0
    assert session.receipt["status"] == "start_failed_cleaned_up"
