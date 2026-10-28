"""A single isolated Qwen worker replaying actual S4 production evidence."""
from __future__ import annotations

from pathlib import Path
import json
import math
import statistics
import time
from typing import Any

from .config import ProjectConfig
from .contracts import CandidateEvent
from .hashing import canonical_sha256, sha256_file
from .performance import PerformanceRecorder, ResourceSampler, recording
from .performance_benchmark import _copy_fixture
from .pipeline import _classification_prompt, _make_qwen_adapter, _QwenCallBudget, _qwen_pass
from .research_backend import ResearchBackend
from .research_stages import _profile_vocabulary
from .runtime import GpuLease, release_model
from .smoke import _growth_metrics
from .storage import RunStore, atomic_write_json, iter_jsonl


_MEMORY_MEASUREMENTS = (
    "allocated_bytes", "reserved_bytes", "process_rss_bytes",
    "request_peak_allocated_bytes", "request_peak_reserved_bytes",
)


def _valid_memory_measurement(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def _valid_counter(value: Any) -> bool:
    return type(value) is int and value >= 0


def _memory_growth(values: list[float]) -> dict[str, Any]:
    """Retain the original 100-request audit and inspect the actual run tail."""
    result = _growth_metrics(values[:100])
    baseline = statistics.median(values[10:30]) if len(values) >= 30 else None
    recent_windows = [statistics.median(values[start:start + 10]) for start in range(len(values) - 30, len(values), 10)] if len(values) >= 30 else []
    recent_rising = len(recent_windows) == 3 and recent_windows[0] < recent_windows[1] < recent_windows[2]
    tail = values[-10:]
    tail_steps = [right - left for left, right in zip(tail, tail[1:])]
    result.update({
        "observed_requests": len(values),
        "recent_ten_request_window_medians_mib": recent_windows,
        "recent_three_windows_rising": recent_rising,
        "tail_median_growth_from_postwarmup_mib": None if baseline is None else statistics.median(tail) - baseline,
        "tail_median_step_mib_per_request": statistics.median(tail_steps) if tail_steps else None,
    })
    return result


def run_production_stability(config: ProjectConfig, run_dir: Path, *, output: Path, requests: int = 100) -> dict[str, Any]:
    if type(requests) is not int or requests < 1:
        raise ValueError("stability_requests_must_be_positive_integer")
    run_dir = run_dir.resolve()
    output = output.resolve()
    assets_root = output.with_name(output.stem + ".assets").resolve()
    if any(path == run_dir or run_dir in path.parents for path in (output, assets_root)):
        raise ValueError("stability_output_must_be_outside_fixture")
    if output.exists():
        raise FileExistsError(f"stability_output_exists:{output}")
    manifest = json.loads((run_dir / "run_manifest.v1.json").read_text(encoding="utf-8"))
    source = Path(manifest["source"]["path"])
    if sha256_file(source) != manifest["source"]["sha256"]:
        raise ValueError("stability_fixture_source_changed")
    backend = ResearchBackend(config)
    profile = backend.profile()
    if manifest["profile_sha256"] != profile.sha256:
        raise ValueError("stability_fixture_profile_mismatch")
    store = RunStore(assets_root, source.stem, canonical_sha256(config.payload), unique_run=True)
    _copy_fixture(run_dir.resolve(), store.run_dir)
    spec = backend._build_spec(
        store=store, source=source, profile=profile, mode=str(manifest["mode"]),
        max_frames=manifest.get("max_frames"), skip_qwen=False, allow_incomplete_preflight=False,
        effective_config=config, experiment_digest=canonical_sha256(config.payload),
    )
    ready = {str(row["candidate_event_id"]) for row in iter_jsonl(store.run_dir / "artifacts/qwen_crop_queue.jsonl") if row.get("status") == "ready"}
    candidates = [CandidateEvent.from_dict(row) for row in iter_jsonl(store.run_dir / "artifacts/candidates.selected.jsonl") if str(row.get("event_id")) in ready]
    if not candidates:
        raise ValueError("stability_fixture_has_no_ready_candidates")
    evidence = list(iter_jsonl(store.run_dir / "evidence/index.jsonl"))
    locate_rows = list(iter_jsonl(store.run_dir / "artifacts/locate_results.jsonl"))
    _, vocabulary, vocabulary_sha = _profile_vocabulary(spec)
    prompt = _classification_prompt(vocabulary, vocabulary_sha, action_profile=profile, prompt_version=str(config.get("qwen.prompt_version", "legacy_v1")))
    result: dict[str, Any] = {
        "schema_version": "classroom_hybrid_ai.production_stability.v1", "status": "running",
        "requested_calls": requests, "fixture_candidates": len(candidates),
        "source_sha256": spec.input_sha256, "profile_sha256": profile.sha256,
        "config_sha256": spec.config_sha256, "code_sha256": spec.code_sha256,
        "device_strategy": config.get("qwen.device_map_policy"), "prompt_version": config.get("qwen.prompt_version", "legacy_v1"),
        "max_new_tokens": config.get("qwen.max_new_tokens"), "evaluation_status": "not_evaluated",
        "one_worker": False, "input_rotation": "production_candidates_round_robin", "samples": [],
    }
    progress = output.with_name(output.name + ".progress.json")
    stop_file = output.with_name(output.name + ".stop")
    result["stop_file"] = str(stop_file)
    result["stop_semantics"] = "finish_current_request_then_close_worker; restart_100_in_new_report"
    atomic_write_json(progress, result)
    recorder = PerformanceRecorder("production_stability")
    adapter = None
    started = time.perf_counter()
    try:
        with ResourceSampler(gpu=True) as sampler, GpuLease(config.resolve(".runtime/classroom_hybrid_ai/gpu.lock")), recording(recorder):
            try:
                for index in range(requests):
                    if stop_file.exists():
                        result["interruption_reason"] = "stop_file"
                        break
                    if adapter is None:
                        adapter = _make_qwen_adapter(config)
                    candidate = candidates[index % len(candidates)]
                    call_started = time.perf_counter()
                    reviews, stats = _qwen_pass(
                        config, [candidate], evidence, store, skip=False, prompt=prompt,
                        response_prefix=f"responses/stability/{index:04d}",
                        expected_vocabulary_sha256=vocabulary_sha, allowed_action_ids=set(profile.action_ids),
                        action_profile=profile, locate_rows=locate_rows, response_contract="classification",
                        adapter=adapter, call_budget=_QwenCallBudget(1), release_adapter=False,
                        resolve_conflicts=False, review_output=f"diagnostics/reviews.{index:04d}.jsonl",
                    )
                    review = reviews[0]
                    sample = {
                        "iteration": index + 1, "event_id": candidate.event_id,
                        "wall_seconds": time.perf_counter() - call_started,
                        "schema_valid": review.schema_valid, "decision": review.decision,
                        "behavior": review.behavior, "stats": stats,
                        "worker_pid": getattr(adapter, "pid", None),
                        "memory": dict(getattr(adapter, "last_metrics", {})),
                        "performance": dict(getattr(adapter, "last_performance", {})),
                    }
                    try:
                        import psutil
                        sample["memory"]["process_rss_bytes"] = psutil.Process(adapter.pid).memory_info().rss
                    except Exception:
                        sample["memory"]["process_rss_unavailable"] = True
                    invalid_memory = [key for key in _MEMORY_MEASUREMENTS if not _valid_memory_measurement(sample["memory"].get(key))]
                    # Invalid readings are explicitly missing in the report;
                    # JSON NaN/Infinity must never masquerade as measured bytes.
                    for key, value in list(sample["memory"].items()):
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                            sample["memory"][key] = None
                    sample["invalid_memory_measurements"] = invalid_memory
                    result["samples"].append(sample)
                    atomic_write_json(progress, result)
                    if getattr(adapter, "exitcode", None) is not None or not review.schema_valid:
                        result["failure"] = "worker_exited_or_production_schema_failed"
                        break
                if stop_file.exists():
                    result["interruption_reason"] = "stop_file"
            finally:
                result["teardown"] = release_model(adapter) if adapter is not None else {"reason": "not_started"}
                result["worker_exitcode"] = getattr(adapter, "exitcode", None)
        result["resources"] = sampler.payload()
    except KeyboardInterrupt:
        result["interruption_reason"] = "keyboard_interrupt"
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}:{exc}"
    result["wall_seconds"] = time.perf_counter() - started
    result["performance"] = recorder.payload()
    growth = {}
    for metric in ("allocated_bytes", "reserved_bytes", "process_rss_bytes"):
        values = [float(row["memory"][metric]) / 2**20 for row in result["samples"] if _valid_memory_measurement(row["memory"].get(metric))]
        growth[metric] = _memory_growth(values) if values and len(values) == len(result["samples"]) else {"missing_reason": "not_all_requests_observed_with_valid_finite_nonnegative_values"}
    result["memory_growth"] = growth
    growing = any(
        (value.get("three_consecutive_windows_rising", False)
         and float(value.get("growth_11_30_to_81_100_mib") or 0) > limit)
        or ((value.get("recent_three_windows_rising", False)
             or float(value.get("tail_median_step_mib_per_request") or 0) > 0)
            and float(value.get("tail_median_growth_from_postwarmup_mib") or 0) > limit)
        for key, limit in (("allocated_bytes", 64), ("reserved_bytes", 128), ("process_rss_bytes", 512))
        for value in (growth[key],)
    )
    attempts = [row["performance"].get("counts", {}).get("inference_calls") for row in result["samples"]]
    completed = sum(value for value in attempts if _valid_counter(value))
    pids = [row.get("worker_pid") for row in result["samples"]]
    result["one_worker"] = bool(pids) and all(type(pid) is int and pid > 0 for pid in pids) and len(set(pids)) == 1
    result["worker_pid"] = pids[0] if result["one_worker"] else None
    oom_counters = [(row["memory"].get("num_ooms"), row["performance"].get("counts", {}).get("oom_retries")) for row in result["samples"]]
    oom_measured = bool(oom_counters) and all(_valid_counter(count) for pair in oom_counters for count in pair)
    oom_observed = any(_valid_counter(count) and count > 0 for pair in oom_counters for count in pair) or any(row.get("stats", {}).get("oom_circuit_open") for row in result["samples"])
    measured_all = (
        len(attempts) == requests
        and all(_valid_counter(value) and value == 1 for value in attempts)
        and all(not row["invalid_memory_measurements"] for row in result["samples"])
        and all("missing_reason" not in value for value in growth.values())
        and oom_measured and result["one_worker"]
    )
    teardown = result.get("teardown", {})
    teardown_error = teardown.get("close_error") or teardown.get("cuda_after_release", {}).get("teardown_error")
    exitcode = result.get("worker_exitcode")
    abnormal_exit = type(exitcode) is int and exitcode != 0
    failed = result.get("failure") or growing or oom_observed or teardown_error or abnormal_exit
    result["gate_checks"] = {"all_requests_and_peaks_measured": measured_all, "oom_counters_measured": oom_measured, "oom_observed": bool(oom_observed), "sustained_memory_growth": growing, "teardown_error": teardown_error, "one_worker_pid": result["one_worker"]}
    result["status"] = "failed" if failed else "passed" if completed >= 100 and measured_all and type(exitcode) is int and exitcode == 0 else "incomplete"
    if result.get("interruption_reason"):
        result["status_before_interruption_classification"] = result["status"]
        result["status"] = "paused_by_user"
        result["stability_acceptance"] = "not_completed"
    result["completed_calls"] = completed
    result["completed_reviews"] = len(result["samples"])
    result["gate"] = "100_production_requests_no_oom_no_schema_failure_no_sustained_growth"
    atomic_write_json(output, result)
    atomic_write_json(progress, result)
    return result
