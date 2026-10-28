from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from classroom_hybrid_ai.qwen_worker import QwenProcessClient, QwenWorkerError


class _CloseProcess:
    """The worker exits normally as its close response becomes available."""

    pid = 4321

    def __init__(self):
        self.alive = True
        self.exitcode = None
        self.join_timeouts = []
        self.terminated = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.exitcode = -15


class _ClosePipe:
    def __init__(self, process, response, *, acknowledge=True):
        self.process = process
        self.response = response
        self.acknowledge = acknowledge
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def poll(self, timeout):
        self.process.alive = False
        self.process.exitcode = 0
        return self.acknowledge

    def recv(self):
        return self.response

    def close(self):
        self.closed = True


def _client_at_close(response, *, acknowledge=True):
    client = QwenProcessClient({})
    process = _CloseProcess()
    pipe = _ClosePipe(process, response, acknowledge=acknowledge)
    client._process = process
    client._connection = pipe
    return client, process, pipe


def test_close_error_survives_zero_exit_and_releases_handles():
    client, process, pipe = _client_at_close({"ok": False, "error": {"message": "CUDA teardown failed"}})
    with pytest.raises(QwenWorkerError, match="CUDA teardown failed"):
        client.close()
    assert client.exitcode == 0
    assert client._process is None and client._connection is None
    assert pipe.closed
    assert not process.terminated
    # A repeated cleanup must neither resend a close nor mask the first error
    # with a secondary closed-pipe failure.
    client.close()
    assert pipe.sent == [{"op": "close"}]


@pytest.mark.parametrize("response", [None, [], {}, {"ok": 1}])
def test_invalid_close_ack_cannot_succeed_with_zero_exit(response):
    client, _, pipe = _client_at_close(response)
    with pytest.raises(QwenWorkerError, match="qwen_worker_close"):
        client.close()
    assert client.exitcode == 0
    assert pipe.closed
    assert client._process is None and client._connection is None


def test_missing_close_ack_cannot_succeed_with_zero_exit():
    client, _, pipe = _client_at_close(None, acknowledge=False)
    with pytest.raises(QwenWorkerError, match="qwen_worker_close_ack_timeout"):
        client.close()
    assert client.exitcode == 0
    assert pipe.closed


def test_successful_close_preserves_ack_metrics_and_exit_grace():
    client, process, pipe = _client_at_close({"ok": True, "metrics": {"allocated_bytes": 0}})
    client.close()
    assert client.exitcode == 0
    assert client.last_metrics == {"allocated_bytes": 0}
    assert process.join_timeouts == [30.0]
    assert pipe.closed


def test_request_timeout_terminates_before_late_response_can_be_close_ack(monkeypatch):
    import classroom_hybrid_ai.qwen_worker as worker
    client, process, pipe = _client_at_close({'ok': False, 'error':'late warmup reply'})
    clock = iter((0., 31.))
    monkeypatch.setattr(worker.time, 'monotonic', lambda: next(clock))
    client.timeout = 30.
    with pytest.raises(TimeoutError, match='qwen_worker_request_timeout'):
        client.warmup()
    assert process.terminated
    client.close()
    assert pipe.sent == [{'op':'warmup'}]
    assert pipe.closed and client.exitcode == -15


def test_release_model_records_child_close_error_even_with_zero_exit(monkeypatch):
    from classroom_hybrid_ai.runtime import release_model

    client, _, pipe = _client_at_close({"ok": False, "error": "child cleanup failed"})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    result = release_model(client)
    assert "child cleanup failed" in result["close_error"]
    assert client.exitcode == 0
    assert pipe.closed


def _fake_http_client(_payload: dict) -> dict:
    """Pickleable OpenAI-compatible stub used inside a spawned worker."""

    return {"choices": [{"message": {"content": '{"mappings":[]}'}}]}


class _TinyProcessor:
    def apply_chat_template(self, *_args, **_kwargs):
        return {"input_ids": [[1, 2]]}

    def decode(self, *_args, **_kwargs):
        return '"mappings":[]}'


class _TinyModel:
    device = None

    def generate(self, **_kwargs):
        return [[1, 2, 3]]


def _tiny_local_loader(_path):
    return _TinyModel(), _TinyProcessor()


def test_model_loading_lifecycle_survives_resident_requests(tmp_path):
    client = QwenProcessClient({"model": str(tmp_path), "backend": "local", "local_loader": _tiny_local_loader}, timeout=30)
    try:
        client.warmup()
        first = client.worker_lifecycle
        assert first["counts"]["model_load_attempts"] == 1
        assert first["counts"]["model_load_successes"] == 1
        assert first["timings_seconds"]["model_load"] > 0
        assert first["parent_timings_seconds"]["startup_to_warmup_ack"] > 0
        for _ in range(2):
            assert client.complete("return mappings") == {"mappings": []}
            assert client.last_performance["timings_seconds"]["model_load"] == 0
            lifecycle = client.last_performance["worker_lifecycle"]
            assert lifecycle["timings_seconds"]["model_load"] == first["timings_seconds"]["model_load"]
            assert lifecycle["counts"]["model_load_attempts"] == 1
            assert lifecycle["counts"]["warmup_requests"] == 1
    finally:
        client.close()
    assert client.exitcode == 0


def test_qwen_worker_is_a_short_lived_process_and_surfaces_load_errors(tmp_path: Path) -> None:
    model_dir = tmp_path / "not-a-model"
    model_dir.mkdir()
    client = QwenProcessClient(
        {
            "model": str(model_dir),
            "backend": "local",
            "max_new_tokens": 1,
            "max_oom_retries": 0,
            "device_map": "cpu",
            "release_cuda_cache": False,
        },
        timeout=30,
    )
    try:
        with pytest.raises(QwenWorkerError, match="qwen_worker|OSError|config"):
            client.warmup()
        assert client.pid is not None
        assert client.loaded is True
    finally:
        client.close()
    assert client.loaded is False
    assert client.exitcode == 0


def test_qwen_worker_forwards_text_completion_for_semantic_adjudication() -> None:
    client = QwenProcessClient(
        {
            "model": "local-http-stub",
            "backend": "http",
            "http_client": _fake_http_client,
            "max_new_tokens": 8,
            "max_oom_retries": 0,
            "release_cuda_cache": False,
        },
        timeout=30,
    )
    try:
        assert client.complete("return a mappings array") == {"mappings": []}
        assert client.pid is not None
        assert client.last_performance["schema_version"] == "model_performance.v1"
        assert client.last_performance["counts"]["inference_calls"] == 1
        assert client.last_performance["timings_seconds"]["total"] >= 0
        assert "max_allocated_bytes" in client.last_metrics
    finally:
        client.close()
    assert client.exitcode == 0
    assert client.last_performance["timings_seconds"]["teardown"] >= 0
