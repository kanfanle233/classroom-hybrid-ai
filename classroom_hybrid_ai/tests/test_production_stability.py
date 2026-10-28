"""No-GPU acceptance and negative-path tests for production stability reports."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from classroom_hybrid_ai.contracts import CandidateEvent
from classroom_hybrid_ai.hashing import sha256_file
import classroom_hybrid_ai.performance_stability as stability


@pytest.fixture
def harness(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic fixture; no decoding or model execution in these tests")
    fixture = tmp_path / "fixture"
    artifacts = fixture / "artifacts"
    artifacts.mkdir(parents=True)
    (fixture / "evidence").mkdir()
    manifest = {"source": {"path": str(source), "sha256": sha256_file(source)}, "profile_sha256": "profile", "mode": "production_adaptive"}
    (fixture / "run_manifest.v1.json").write_text(json.dumps(manifest), encoding="utf-8")
    candidate = CandidateEvent(event_id="event-1", track_id="student-1", event_type="review", trigger_frame_index=0, trigger_pts_seconds=0.0)
    (artifacts / "candidates.selected.jsonl").write_text(json.dumps(candidate.to_dict()) + "\n", encoding="utf-8")
    (artifacts / "qwen_crop_queue.jsonl").write_text(json.dumps({"candidate_event_id": candidate.event_id, "status": "ready"}) + "\n", encoding="utf-8")
    (artifacts / "locate_results.jsonl").write_text("", encoding="utf-8")
    (fixture / "evidence/index.jsonl").write_text(json.dumps({"event_id": candidate.event_id}) + "\n", encoding="utf-8")
    config = SimpleNamespace(payload={}, get=lambda key, default=None: default, resolve=lambda value: tmp_path / value)
    profile = SimpleNamespace(sha256="profile", action_ids={"UNKNOWN"})
    spec = SimpleNamespace(input_sha256=manifest["source"]["sha256"], config_sha256="config", code_sha256="code")
    backend = SimpleNamespace(profile=lambda: profile, _build_spec=lambda **kwargs: spec)
    monkeypatch.setattr(stability, "ResearchBackend", lambda _: backend)
    monkeypatch.setattr(stability, "_profile_vocabulary", lambda _: (profile, (), "vocab"))
    monkeypatch.setattr(stability, "_classification_prompt", lambda *args, **kwargs: "fixed production prompt substitute")
    monkeypatch.setattr(stability, "GpuLease", lambda _: nullcontext())
    sampler = SimpleNamespace(payload=lambda: {})
    monkeypatch.setattr(stability, "ResourceSampler", lambda **kwargs: nullcontext(sampler))
    monkeypatch.setattr(psutil, "Process", lambda pid=None: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100 * 2**20)))
    adapter = SimpleNamespace(pid=os.getpid(), exitcode=None, last_metrics={}, last_performance={})
    created = []
    calls = []
    behavior = {"inference_calls": 1, "allocated_bytes": 100 * 2**20, "schema_valid": True, "crash_at": None, "missing_metric": None, "close_error": None, "exitcode": 0, "growth_per_call": 0, "growth_start": 0, "num_ooms": 0}

    def make_adapter(_):
        created.append(adapter)
        return adapter

    def run_pass(config, candidates, evidence, store, **kwargs):
        calls.append(kwargs)
        if behavior["crash_at"] == len(calls):
            raise RuntimeError("simulated worker crash")
        metrics = {"allocated_bytes": behavior["allocated_bytes"] + behavior["growth_per_call"] * max(0, len(calls) - behavior["growth_start"]), "reserved_bytes": 200 * 2**20, "max_allocated_bytes": 300 * 2**20, "max_reserved_bytes": 400 * 2**20, "request_peak_allocated_bytes": 300 * 2**20, "request_peak_reserved_bytes": 400 * 2**20, "num_ooms": behavior["num_ooms"]}
        if isinstance(behavior["allocated_bytes"], bool):
            metrics["allocated_bytes"] = behavior["allocated_bytes"]
        if behavior["missing_metric"]:
            metrics.pop(behavior["missing_metric"], None)
        adapter.last_metrics = metrics
        adapter.last_performance = {"counts": {"inference_calls": behavior["inference_calls"], "oom_retries": 0}}
        return [SimpleNamespace(schema_valid=behavior["schema_valid"], decision="uncertain", behavior="UNKNOWN")], {"reviews": 1, "status": "ok"}

    def release(value):
        value.exitcode = behavior["exitcode"]
        return {"close_error": behavior["close_error"]}

    monkeypatch.setattr(stability, "_make_qwen_adapter", make_adapter)
    monkeypatch.setattr(stability, "_qwen_pass", run_pass)
    monkeypatch.setattr(stability, "release_model", release)
    return SimpleNamespace(config=config, fixture=fixture, output=tmp_path / "stability.json", behavior=behavior, created=created, calls=calls)


def run(harness, requests=100):
    return stability.run_production_stability(harness.config, harness.fixture, output=harness.output, requests=requests)


def test_100_actual_calls_reuse_one_adapter_and_preserve_fixture(harness):
    before = {path.relative_to(harness.fixture).as_posix(): path.read_bytes() for path in harness.fixture.rglob("*") if path.is_file()}
    result = run(harness)
    assert result["status"] == "passed"
    assert result["completed_calls"] == result["completed_reviews"] == 100
    assert len(harness.created) == 1
    assert len(harness.calls) == 100
    assert all(call["adapter"] is harness.created[0] and call["release_adapter"] is False for call in harness.calls)
    assert {path.relative_to(harness.fixture).as_posix(): path.read_bytes() for path in harness.fixture.rglob("*") if path.is_file()} == before


def test_reviews_without_actual_calls_are_incomplete(harness):
    harness.behavior["inference_calls"] = 0
    result = run(harness)
    assert result["completed_reviews"] == 100
    assert result["completed_calls"] == 0
    assert result["status"] == "incomplete"


def test_missing_current_allocation_is_incomplete(harness):
    harness.behavior["missing_metric"] = "allocated_bytes"
    assert run(harness)["status"] == "incomplete"


def test_worker_crash_is_failed_even_with_clean_teardown(harness):
    harness.behavior["crash_at"] = 4
    result = run(harness)
    assert result["status"] == "failed"
    assert result["completed_reviews"] == 3
    assert "simulated worker crash" in result["failure"]


def test_sustained_memory_growth_fails(harness):
    harness.behavior["growth_per_call"] = 3 * 2**20
    assert run(harness)["status"] == "failed"


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -1, True])
def test_invalid_memory_measurement_cannot_pass(harness, metric):
    harness.behavior["allocated_bytes"] = metric
    assert run(harness)["status"] != "passed"


@pytest.mark.parametrize("metric", ["request_peak_allocated_bytes", "request_peak_reserved_bytes"])
def test_missing_request_peak_cannot_pass(harness, metric):
    harness.behavior["missing_metric"] = metric
    assert run(harness)["status"] != "passed"


def test_boolean_inference_counter_is_not_one_actual_call(harness):
    harness.behavior["inference_calls"] = True
    assert run(harness)["status"] != "passed"


def test_teardown_error_cannot_pass(harness):
    harness.behavior["close_error"] = "simulated CUDA teardown error"
    assert run(harness)["status"] != "passed"


def test_observed_oom_counter_cannot_pass(harness):
    harness.behavior["num_ooms"] = 1
    assert run(harness)["status"] != "passed"


def test_growth_after_first_hundred_requests_cannot_pass(harness):
    harness.behavior["growth_start"] = 100
    harness.behavior["growth_per_call"] = 8 * 2**20
    assert run(harness, requests=120)["status"] != "passed"


@pytest.mark.parametrize("requests", [True, 0, -1, 100.0])
def test_invalid_request_count_rejected_before_any_worker(harness, requests):
    with pytest.raises((ValueError, TypeError)):
        run(harness, requests=requests)
    assert not harness.created
    assert not harness.output.exists()


def test_nested_output_rejected_before_mutating_fixture(harness):
    before = {path.relative_to(harness.fixture).as_posix() for path in harness.fixture.rglob("*")}
    harness.output = harness.fixture / "stability.json"
    with pytest.raises(ValueError):
        run(harness)
    assert {path.relative_to(harness.fixture).as_posix() for path in harness.fixture.rglob("*")} == before


def test_cooperative_stop_closes_worker_and_preserves_finished_requests(harness, monkeypatch):
    original = stability._qwen_pass
    def stop_after_three(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(harness.calls) == 3:
            harness.output.with_name(harness.output.name + ".stop").touch()
        return result
    monkeypatch.setattr(stability, "_qwen_pass", stop_after_three)
    result = run(harness)
    assert result["status"] == "paused_by_user"
    assert result["interruption_reason"] == "stop_file"
    assert result["completed_calls"] == 3
    assert len(harness.calls) == 3
    assert result["worker_exitcode"] == 0
    assert json.loads(harness.output.read_text(encoding="utf-8"))["status"] == "paused_by_user"


def test_stop_before_first_request_does_not_start_worker(harness):
    harness.output.with_name(harness.output.name + ".stop").touch()
    result = run(harness)
    assert result["status"] == "paused_by_user"
    assert result["completed_calls"] == 0
    assert not harness.created


def test_keyboard_interrupt_saves_terminal_report_and_releases_worker(harness, monkeypatch):
    original = stability._qwen_pass
    def interrupt_after_two(*args, **kwargs):
        if len(harness.calls) == 2:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)
    monkeypatch.setattr(stability, "_qwen_pass", interrupt_after_two)
    result = run(harness)
    assert result["status"] == "paused_by_user"
    assert result["interruption_reason"] == "keyboard_interrupt"
    assert result["completed_calls"] == 2
    assert result["worker_exitcode"] == 0
    assert json.loads(harness.output.with_name(harness.output.name + ".progress.json").read_text(encoding="utf-8"))["status"] == "paused_by_user"


@pytest.mark.parametrize("status,exitcode", [("passed", 0), ("failed", 1), ("incomplete", 2), ("paused_by_user", 130)])
def test_stability_cli_only_passed_returns_success(harness, monkeypatch, status, exitcode):
    from classroom_hybrid_ai import cli
    monkeypatch.setattr(cli, "load_config", lambda *_args, **_kwargs: harness.config)
    monkeypatch.setattr(stability, "run_production_stability", lambda *_args, **_kwargs: {"status": status})
    assert cli.main(["benchmark-qwen-stability", str(harness.fixture), "--output", str(harness.output)]) == exitcode
