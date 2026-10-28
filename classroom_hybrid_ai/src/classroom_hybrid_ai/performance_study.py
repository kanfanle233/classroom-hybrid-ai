"""Locked, resumable, whole-video studies using the production benchmark child.

Study setup and re-verification are outside trial timing. Every measured trial
starts a new interpreter and performs S0-S7, including all deliverable videos.
Algorithm comparisons never use equality of opaque student IDs as quality.
"""
from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
from typing import Any, Mapping

from .performance_benchmark import _canonical, _code_state, _launch, _read, _sha, _target_project, _write, percentile
from .taxonomy import CORE8_CODES


SCHEMA = "classroom_hybrid_ai.performance_study.v1"
_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_METRICS = {"duration_weighted_macro_f1": .01, "student_recall": .01,
            "idf1": .01, "hota": .01, "segmental_f1": .01,
            "unknown_time_fraction": .01}


def asset_sha256(path: Path) -> str:
    """Full content hash, including tokenizer/config/custom code in model dirs."""
    path = path.resolve()
    if path.is_file():
        return _sha(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = {item.relative_to(path).as_posix(): _sha(item) for item in sorted(path.rglob("*"))
             if item.is_file() and "__pycache__" not in item.parts and not item.name.endswith(".lock")}
    if not files:
        raise ValueError(f"empty_locked_directory:{path}")
    return _canonical(files)


def _path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("study_path_required")
    item = Path(value)
    return (item if item.is_absolute() else base / item).resolve()


def _locked_asset(row: Any, base: Path) -> dict[str, str]:
    if not isinstance(row, dict) or not _SHA.fullmatch(str(row.get("sha256", ""))):
        raise ValueError("study_asset_requires_sha256")
    path = _path(row.get("path"), base)
    actual = asset_sha256(path)
    if row["sha256"] != actual:
        raise ValueError(f"study_asset_sha_mismatch:{path}")
    return {**row, "path": str(path), "sha256": actual}


def _valid_number(value: Any, *, minimum: float = 0, maximum: float = 1) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and minimum <= value <= maximum


def load_study(manifest: Path) -> dict[str, Any]:
    """Validate every declared byte before allowing any GPU work or resume."""
    import yaml
    manifest = manifest.resolve()
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported_study_schema")
    kind = payload.get("comparison_type")
    if kind not in {"engineering_equivalence", "algorithm_quality_cost"}:
        raise ValueError("study_comparison_type_required")
    if type(payload.get("repeats", 3)) is not int or payload.get("repeats", 3) != 3:
        raise ValueError("study_requires_three_repeats")
    if type(payload.get("seed", 0)) is not int or not 0 <= payload.get("seed", 0) < 2**32:
        raise ValueError("study_seed_must_be_uint32")
    if payload.get("delivery") != "whole_pipeline_s0_s7_all_videos":
        raise ValueError("study_full_delivery_required")
    for key in ("max_frames", "max_requests", "run_dir", "skip_qwen", "skip_stages"):
        if key in payload:
            raise ValueError(f"study_bounded_or_skipped_workload_forbidden:{key}")
    base = manifest.parent
    root = _path(payload.get("repository_root"), base)
    result = {**payload, "manifest_path": str(manifest), "manifest_sha256": _sha(manifest),
              "repository_root": str(root), "repeats": 3, "seed": payload.get("seed", 0)}
    for field in ("videos", "variants"):
        values = payload.get(field)
        if not isinstance(values, list) or not values or any(not isinstance(item, dict) for item in values):
            raise ValueError(f"study_{field}_required")
        ids = [item.get("id") for item in values]
        if any(not isinstance(item, str) or not _ID.fullmatch(item) for item in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"study_{field}_ids_invalid_or_duplicate")
    result["videos"] = [_locked_asset(row, base) for row in payload["videos"]]
    result["variants"] = []
    model_cache: dict[tuple[str, str], dict[str, str]] = {}
    for row in payload["variants"]:
        for key in ("max_frames", "max_requests", "run_dir", "skip_qwen", "skip_stages", "overrides"):
            if key in row:
                raise ValueError(f"study_variant_override_forbidden:{key}")
        source = _path(row.get("source_path"), base)
        project = _target_project(source)
        state = _code_state(source, project)
        if row.get("source_sha256") != state:
            raise ValueError(f"study_source_sha_mismatch:{row['id']}")
        config = _path(row.get("config"), base)
        if row.get("config_sha256") != _sha(config):
            raise ValueError(f"study_config_sha_mismatch:{row['id']}")
        policy = yaml.safe_load(config.read_text(encoding="utf-8"))
        if not isinstance(policy, dict) or policy.get("qwen", {}).get("enabled") is not True:
            raise ValueError(f"study_qwen_must_be_enabled:{row['id']}")
        if policy.get("research", {}).get("mode", "production_adaptive") != "production_adaptive":
            raise ValueError("study_requires_production_adaptive_mode")
        route = policy.get("pipeline", {}).get("localization_route", "locate_only")
        if route not in {"locate_only", "yolo_only", "yolo_locate_hybrid"}:
            raise ValueError("study_localization_route_invalid")
        models = row.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError("study_model_locks_required")
        locked = []
        for model in models:
            location = str(_path(model.get("path"), base))
            key = (location, str(model.get("sha256")))
            if key not in model_cache:
                model_cache[key] = _locked_asset(model, base)
            locked.append(model_cache[key])
        required = [policy["qwen"].get("model_dir")]
        if route != "yolo_only":
            required.append(policy.get("locate_anything", {}).get("model_dir"))
        if route != "locate_only":
            required.append(policy.get("yolo", {}).get("tracker_model"))
        locked_paths = {item["path"] for item in locked}
        if any(str(_path(item, root)) not in locked_paths for item in required):
            raise ValueError(f"study_effective_model_not_locked:{row['id']}")
        result["variants"].append({**row, "source_path": str(source), "config": str(config),
                                   "models": locked, "effective_localization_route": route,
                                   "declared_config": policy})
    if len(result["variants"]) < 2 or payload.get("baseline_variant") not in {r["id"] for r in result["variants"]}:
        raise ValueError("study_baseline_and_candidate_required")
    if payload.get("quality_report") is not None:
        result["quality_report"] = _locked_asset(payload["quality_report"], base)
    return result


_SEED_STARTUP = '''"""Trial-local RNG initialization; inherited by owned workers."""
import json, os, time
from pathlib import Path
if os.environ.get("CLASSROOM_STUDY_SEED") is not None:
    started = time.perf_counter()
    root = Path(os.environ["CLASSROOM_STUDY_SEED_AUDIT"])
    root.mkdir(parents=True, exist_ok=True)
    row = {"pid": os.getpid(), "parent_pid": os.getppid(), "seed": int(os.environ["CLASSROOM_STUDY_SEED"])}
    try:
        import random, numpy, torch
        random.seed(row["seed"])
        numpy.random.seed(row["seed"])
        torch.manual_seed(row["seed"])
        row.update(status="ok", targets=["python", "numpy", "torch_cpu", "torch_cuda_lazy"])
    except Exception as exc:
        row.update(status="failed", error=type(exc).__name__ + ":" + str(exc))
    row["initialization_seconds"] = time.perf_counter() - started
    (root / (str(os.getpid()) + ".json")).write_text(json.dumps(row), encoding="utf-8")
    if row["status"] != "ok":
        os._exit(78)
'''


def prepare_seed_control(directory: Path, seed: int) -> dict[str, str]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("study_seed_must_be_uint32")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "sitecustomize.py").write_text(_SEED_STARTUP, encoding="utf-8")
    return {"PYTHONPATH": os.pathsep.join(filter(None, (str(directory.resolve()), os.environ.get("PYTHONPATH")))),
            "CLASSROOM_STUDY_SEED": str(seed), "PYTHONHASHSEED": str(seed),
            "CLASSROOM_STUDY_SEED_AUDIT": str((directory / "audit").resolve()),
            "PYTHONIOENCODING": "utf-8"}


def verify_seed_control(directory: Path, seed: int, trial: dict[str, Any]) -> dict[str, Any]:
    audits = [_read(path) for path in sorted((directory / "audit").glob("*.json"))]
    ok = {row.get("pid") for row in audits if row.get("status") == "ok" and row.get("seed") == seed}
    required = {trial.get("child_pid")}
    errors = []
    requests = trial.get("request_observations", [])
    for row in requests:
        if row.get("model") in {"qwen", "yolo"}:
            required.add(row.get("worker_pid") or row.get("caller_pid"))
    if any(row.get("model") == "locate" for row in requests):
        receipt = Path(trial.get("run_dir", "")) / "diagnostics/locate_service_receipt.v1.json"
        service = _read(receipt) if receipt.is_file() else {}
        if service.get("owned_process") is True:
            required.add(service.get("pid"))
        else:
            errors.append("locate_worker_not_owned_or_pid_unavailable")
    if any(type(pid) is not int or pid not in ok for pid in required):
        errors.append("not_all_execution_worker_pids_have_successful_seed_audit")
    if any(row.get("status") != "ok" for row in audits):
        errors.append("seed_startup_failed")
    return {"status": "failed" if errors else "passed", "seed": seed,
            "required_pids": sorted(pid for pid in required if type(pid) is int), "audits": audits,
            "startup_sha256": _sha(directory / "sitecustomize.py"), "errors": errors,
            "deterministic_cuda_algorithms_forced": False}


def estimate_study(study: dict[str, Any]) -> dict[str, Any]:
    """Metadata-only estimates; no screenshots, models, or invented sub-timings."""
    from .video import probe_video
    rows = []
    for video in study["videos"]:
        try:
            info = probe_video(video["path"])
            duration = info.duration_seconds
            frames = info.frame_count
        except (OSError, RuntimeError, ValueError) as exc:
            duration, frames = None, None
            metadata_error = f"{type(exc).__name__}:{exc}"
        else:
            metadata_error = None
        for variant in study["variants"]:
            policy = variant["declared_config"]
            cap = int(policy.get("locate_first", {}).get("locate_cap", 1500))
            screenshots_upper = min(cap, frames) if frames is not None else cap
            budget = policy.get("research_budget", {})
            mode = budget.get("selection_mode", "fair")
            labels = list(policy.get("locate_anything", {}).get("prompts", []))
            students = [label for label in labels if str(label).strip().lower() in {"student", "person", "pupil", "learner"}]
            context = [label for label in labels if label not in students]
            groups = int(bool(students)) + int(bool(context))
            rows.append({"video_id": video["id"], "variant": variant["id"], "duration_seconds": duration,
                         "source_frames": frames, "screenshots": {"lower": 0, "upper": screenshots_upper,
                         "reason": "actual_adaptive_schedule_requires_s1_change_scan"},
                         "locate_calls": {"lower": 0, "upper": screenshots_upper * groups if variant["effective_localization_route"] != "yolo_only" else 0,
                         "prompt_groups": groups, "scope": "all_configured_prompt_groups; difficult_queue_requires_yolo_and_association"},
                         "qwen_calls": {"lower": 0, "upper": 0 if variant["effective_localization_route"] == "yolo_only" else None if mode == "all_candidates" else int(budget.get("maximum", 240)),
                         "reason": "yolo_only_has_no_student_confirmation" if variant["effective_localization_route"] == "yolo_only" else "valid_student_count_and_ready_evidence_not_known_before_s2_s4"},
                         "estimated_wall_seconds_range": None,
                         "timing_missing_reason": "no_comparable_whole_video_calibration_for_this_locked_variant",
                         "metadata_error": metadata_error})
    return {"status": "estimated", "trials": len(rows) * 3, "per_trial": rows,
            "actual_measurements": None, "model_loads_per_trial": "cold; number_depends_on_route_and_valid_candidates"}


def _artifact_inventory(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError("study_trial_run_directory_missing")
    return {path.relative_to(root).as_posix(): _sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and not path.name.endswith((".tmp", ".lock"))}


def _quality_gate(study: dict[str, Any], trials: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    missing = {"status": "not_evaluated", "reason": "independent_video_gold_and_bound_quality_report_required"}
    locked = study.get("quality_report")
    if not locked:
        return missing
    if _sha(Path(locked["path"])) != locked["sha256"]:
        return {"status": "failed", "errors": ["quality_report_content_changed"]}
    report = _read(Path(locked["path"]))
    if not isinstance(report, dict) or report.get("schema_version") != "performance_study_quality.v1":
        return {"status": "failed", "errors": ["quality_report_schema_invalid"]}
    if report.get("status") != "evaluated":
        return missing
    errors = []
    if report.get("independent_gold") is not True or report.get("provenance_review_status") != "passed":
        errors.append("independent_gold_provenance_not_verified")
    if report.get("source_video_sha256") != {row["id"]: row["sha256"] for row in study["videos"]}:
        errors.append("quality_video_binding_mismatch")
    binding = {row["trial_id"]: row.get("semantic_sha256") for row in trials}
    if report.get("trial_semantic_sha256") != binding:
        errors.append("quality_trial_binding_mismatch")
    assets = {row["id"]: {"source": row["source_sha256"], "config": row["config_sha256"],
                          "models": {model["id"]: model["sha256"] for model in row["models"]}}
              for row in study["variants"]}
    if report.get("variant_asset_sha256") != assets:
        errors.append("quality_variant_source_config_model_binding_mismatch")
    variants = report.get("variants", {})
    if not isinstance(variants, dict):
        return {"status": "failed", "errors": errors + ["quality_variants_invalid"]}
    baseline = variants.get(study["baseline_variant"], {})
    proposed = variants.get(candidate, {})
    if not isinstance(baseline, dict) or not isinstance(proposed, dict):
        return {"status": "failed", "errors": errors + ["quality_variant_metrics_invalid"]}
    for name, packet in (("baseline", baseline), ("candidate", proposed)):
        if not isinstance(packet, dict):
            errors.append(f"quality_metrics_invalid:{name}")
            continue
        lessons = packet.get("per_lesson_metrics")
        if not isinstance(lessons, list) or len(lessons) < 2 or any(not isinstance(row, dict) or not row.get("lesson_id") or any(not _valid_number(row.get(metric)) for metric in _METRICS) for row in lessons):
            errors.append(f"quality_per_lesson_metrics_missing:{name}")
        elif len({row["lesson_id"] for row in lessons}) != len(lessons):
            errors.append(f"quality_lesson_ids_duplicate:{name}")
        uncertainty = packet.get("uncertainty", {})
        if not isinstance(uncertainty, dict) or uncertainty.get("method") != "lesson_bootstrap" or uncertainty.get("confidence_level") != .95:
            errors.append(f"quality_lesson_uncertainty_missing:{name}")
        else:
            intervals = uncertainty.get("intervals", {})
            for metric in _METRICS:
                interval = intervals.get(metric) if isinstance(intervals, dict) else None
                if not isinstance(interval, list) or len(interval) != 2 or not all(_valid_number(value) for value in interval) or interval[0] > interval[1]:
                    errors.append(f"quality_uncertainty_interval_invalid:{name}:{metric}")
        if not isinstance(packet.get("per_class"), dict) or any(not isinstance(row, dict) for row in packet["per_class"].values()):
            return {"status": "failed", "errors": errors + [f"quality_per_class_invalid:{name}"]}
    for metric, limit in _METRICS.items():
        left, right = baseline.get(metric), proposed.get(metric)
        if not _valid_number(left) or not _valid_number(right):
            errors.append(f"quality_metric_missing_or_invalid:{metric}")
        else:
            regression = right - left if metric == "unknown_time_fraction" else left - right
            if regression > limit + 1e-12:
                errors.append(f"quality_regression:{metric}")
    for code in CORE8_CODES:
        left = baseline.get("per_class", {}).get(code, {})
        right = proposed.get("per_class", {}).get(code, {})
        if any(not _valid_number(row.get("support_seconds"), minimum=1e-12, maximum=1e12)
               or not _valid_number(row.get("f1")) for row in (left, right)):
            errors.append(f"quality_class_support_or_f1_invalid:{code}")
        elif abs(left["support_seconds"] - right["support_seconds"]) > 1e-9:
            errors.append(f"quality_class_support_disagrees:{code}")
        elif left["f1"] - right["f1"] > .02 + 1e-12:
            errors.append(f"quality_class_regression:{code}")
    return {"status": "failed" if errors else "passed", "errors": errors, "quality_report_sha256": locked["sha256"]}


def summarize_study(study: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    variants = {}
    baseline_id = study["baseline_variant"]
    for variant in study["variants"]:
        name = variant["id"]
        grouped = {}
        for video in study["videos"]:
            rows = [row for row in trials if row["variant"] == name and row["video_id"] == video["id"]]
            passed = [row for row in rows if row.get("status") == "ok" and row.get("semantic_validation", {}).get("status") == "passed"
                      and row.get("seed_control", {}).get("status") == "passed" and _valid_number(row.get("wall_seconds"), minimum=1e-12, maximum=1e12)]
            values = [row["wall_seconds"] for row in passed]
            requests = [request["wall_seconds"] for row in passed for request in row.get("request_observations", [])
                        if _valid_number(request.get("wall_seconds"), maximum=1e12)]
            grouped[video["id"]] = {"completed_trials": len(passed), "expected_trials": 3,
                "median_wall_seconds": statistics.median(values) if len(values) == 3 else None,
                "request_p50_seconds": statistics.median(requests) if requests else None,
                "request_p95_seconds": percentile(requests, .95) if requests else None,
                "semantic_sha256": sorted({row["semantic_sha256"] for row in passed}),
                "request_counts": {model: sum(request.get("model") == model for row in passed for request in row.get("request_observations", [])) for model in ("locate", "qwen", "yolo")}}
        variants[name] = grouped
    comparisons = {}
    for variant in study["variants"]:
        name = variant["id"]
        if name == baseline_id:
            continue
        baseline, candidate = variants[baseline_id], variants[name]
        complete = all(packet["median_wall_seconds"] is not None for group in (baseline, candidate) for packet in group.values())
        equivalent = complete and all(len(baseline[key]["semantic_sha256"]) == 1 and baseline[key]["semantic_sha256"] == candidate[key]["semantic_sha256"] for key in baseline)
        reduction = (1 - sum(row["median_wall_seconds"] for row in candidate.values()) / sum(row["median_wall_seconds"] for row in baseline.values())) if complete else None
        quality = _quality_gate(study, trials, name)
        equality_required = study["comparison_type"] == "engineering_equivalence"
        eligible = complete and reduction >= .20 and quality["status"] == "passed" and (not equality_required or equivalent)
        comparisons[name] = {"performance_status": "incomplete" if not complete else "target_met" if reduction >= .20 else "target_not_met",
            "median_reduction_fraction": reduction, "outputs_equivalent": equivalent if equality_required else None,
            "quality": quality, "default_promotion_eligible": eligible,
            "promotion_basis": "strict_engineering_output_equivalence" if equality_required else "independent_quality_gate",
            "aggregate_timing": "sum_of_per_video_three_trial_medians", "target_reduction_fraction": .20}
        comparisons[name]["route_interpretation"] = (
            "person_localization_diagnostic; no_Locate_confirmation_means_zero_confirmed_students_and_Qwen_calls; quality_and_coverage_gates_still_required"
            if variant.get("effective_localization_route") == "yolo_only" else "student_action_pipeline")
    return {"variants": variants, "comparisons": comparisons,
            "evaluation_status": "evaluated" if comparisons and all(row["quality"]["status"] == "passed" for row in comparisons.values()) else "not_evaluated"}


def run_study(manifest: Path, *, output: Path, dry_run: bool = False, python: Path | None = None) -> dict[str, Any]:
    study = load_study(manifest)
    output = output.resolve()
    for variant in study["variants"]:
        for protected in (Path(variant["source_path"]), Path(variant["config"]).parent, *(Path(row["path"]) for row in variant["models"])):
            if output == protected or protected in output.parents:
                raise ValueError("study_output_overlaps_locked_inputs")
    estimates = estimate_study(study)
    if dry_run:
        return {"schema_version": SCHEMA, "status": "planned", "manifest_sha256": study["manifest_sha256"],
                "estimates": estimates, "locks_verified": True, "models_loaded": False,
                "evaluation_status": "not_evaluated"}
    output.mkdir(parents=True, exist_ok=True)
    final_path, progress_path = output / "study.json", output / "progress.json"
    stop = output / "STOP"
    if progress_path.exists():
        report = _read(progress_path)
        if report.get("manifest_sha256") != study["manifest_sha256"]:
            raise ValueError("study_resume_manifest_changed")
    else:
        report = {"schema_version": SCHEMA, "status": "running", "manifest_sha256": study["manifest_sha256"],
                  "manifest": study, "estimates": estimates, "trials": [], "stop_file": str(stop),
                  "stop_semantics": "finish_active_whole_video_trial_then_pause",
                  "order": "round_rotation_of_variants_then_each_video", "model_history_reused": False,
                  "timing_scope": "cold_child_process_lifecycle_including_setup_and_media_audit; lock_checks_excluded"}
    verified = []
    for row in report["trials"]:
        receipt_path = output / "receipts" / f"{row['trial_id']}.json"
        receipt = _read(receipt_path)
        if receipt.get("trial") != row or receipt.get("manifest_sha256") != study["manifest_sha256"]:
            raise ValueError("study_resume_receipt_mismatch")
        if row.get("run_dir") and receipt["artifacts"] != _artifact_inventory(Path(row["run_dir"])):
            raise ValueError("study_resume_artifact_changed")
        verified.append(row["trial_id"])
        if row.get("status") != "ok":
            report.update(status="needs_attention", reason="failed_trial_preserved; use_a_new_manifest_output_for_an_explicit_retry")
            _write(final_path, report)
            return report
    report["status"] = "running"
    report["resumed_verified_trials"] = verified
    assets = output / "trials"
    assets.mkdir(exist_ok=True)
    _write(progress_path, report)
    variants = study["variants"]
    for iteration in range(3):
        rotated = variants[iteration % len(variants):] + variants[:iteration % len(variants)]
        for variant in rotated:
            for video in study["videos"]:
                name = f"{iteration + 1:03d}-{variant['id']}-{video['id']}"
                if name in verified:
                    continue
                if stop.exists():
                    report.update(status="paused_by_user", active_trial=None)
                    _write(final_path, report)
                    _write(progress_path, report)
                    return report
                # Re-read source/config/video/model bytes at each independent
                # boundary; the source child separately checks before/after.
                fresh = load_study(manifest)
                if fresh["manifest_sha256"] != study["manifest_sha256"]:
                    raise ValueError("study_manifest_changed_during_execution")
                recovered_receipt = output / "receipts" / f"{name}.json"
                if recovered_receipt.is_file():
                    receipt = _read(recovered_receipt)
                    trial = receipt.get("trial", {})
                    if receipt.get("manifest_sha256") != study["manifest_sha256"] or trial.get("trial_id") != name or not trial.get("run_dir") or receipt.get("artifacts") != _artifact_inventory(Path(trial["run_dir"])):
                        raise ValueError("study_resume_uncommitted_receipt_invalid")
                    report["trials"].append(trial)
                    report.update(summarize_study(study, report["trials"]))
                    verified.append(name)
                    _write(progress_path, report)
                    if trial.get("status") != "ok":
                        report.update(status="needs_attention", reason="recovered_failed_trial_receipt")
                        _write(final_path, report)
                        _write(progress_path, report)
                        return report
                    continue
                if (assets / f"{name}.job.json").exists():
                    report.update(status="needs_attention", active_trial=name,
                                  reason="interrupted_trial_has_no_verified_receipt; no_automatic_duplicate_execution")
                    _write(final_path, report)
                    _write(progress_path, report)
                    return report
                job = {"name": name, "source_path": variant["source_path"], "config": variant["config"],
                       "repository_root": study["repository_root"], "video": video["path"],
                       "component": "pipeline", "max_frames": None, "max_requests": None, "run_dir": None,
                       "trial_dir": str(assets / name), "seed": study["seed"], "profile_model_timings": False}
                report["active_trial"] = name
                _write(progress_path, report)
                trial = _launch(job, assets, python or Path(sys.executable))
                trial.update(trial_id=name, variant=variant["id"], video_id=video["id"], iteration=iteration + 1)
                # Model content changes in flight cannot acquire a valid receipt.
                load_study(manifest)
                inventory = _artifact_inventory(Path(trial["run_dir"])) if trial.get("run_dir") else {}
                _write(output / "receipts" / f"{name}.json", {"schema_version": "performance_study_trial.v1",
                       "manifest_sha256": study["manifest_sha256"], "trial": trial, "artifacts": inventory})
                report["trials"].append(trial)
                report.update(summarize_study(study, report["trials"]))
                report["active_trial"] = None
                if trial.get("status") != "ok":
                    report["status"] = "failed"
                    _write(final_path, report)
                    _write(progress_path, report)
                    return report
                _write(progress_path, report)
    report["status"] = "ok"
    report.update(summarize_study(study, report["trials"]))
    _write(final_path, report)
    _write(progress_path, report)
    return report


def write_study_manifest(path: Path, *, repository_root: Path, videos: list[dict[str, Any]],
                         variants: list[dict[str, Any]], comparison_type: str,
                         baseline_variant: str) -> dict[str, Any]:
    """Freeze explicit local assets; hashing full model contents is intentional.

    Sources/configs must already be final. This helper never edits a source
    variant and never assigns a guessed digest or a placeholder model hash.
    """
    import yaml
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    root = repository_root.resolve()
    locked_variants = []
    models: dict[str, dict[str, str]] = {}
    for row in variants:
        source, config = Path(row["source_path"]).resolve(), Path(row["config"]).resolve()
        policy = yaml.safe_load(config.read_text(encoding="utf-8"))
        route = policy.get("pipeline", {}).get("localization_route", "locate_only")
        required = [("qwen", policy["qwen"]["model_dir"])]
        if route != "yolo_only":
            required.append(("locate", policy["locate_anything"]["model_dir"]))
        if route != "locate_only":
            required.append(("yolo", policy["yolo"]["tracker_model"]))
        references = []
        for model_id, location in required:
            resolved = str(_path(location, root))
            if resolved not in models:
                models[resolved] = {"id": model_id, "path": resolved, "sha256": asset_sha256(Path(resolved))}
            references.append(models[resolved])
        locked_variants.append({**row, "source_path": str(source), "source_sha256": _code_state(source, _target_project(source)),
                                "config": str(config), "config_sha256": _sha(config), "models": references})
    result = {"schema_version": SCHEMA, "comparison_type": comparison_type,
              "repository_root": str(root), "baseline_variant": baseline_variant,
              "repeats": 3, "seed": 0, "delivery": "whole_pipeline_s0_s7_all_videos",
              "videos": [{**row, "path": str(Path(row["path"]).resolve()), "sha256": _sha(Path(row["path"]))} for row in videos],
              "variants": locked_variants}
    _write(path, result)
    return result


def assess_study_quality(output: Path, quality_report: Path) -> dict[str, Any]:
    """Attach later independent gold to completed immutable trials, without GPU.

    The original manifest and performance report are retained unchanged. This
    allows gold annotation to finish after the performance run, without making
    a manifest change invalidate otherwise valid expensive trial receipts.
    """
    output = output.resolve()
    report = _read(output / "study.json")
    if report.get("status") != "ok":
        raise ValueError("study_quality_requires_completed_performance_report")
    original = report["manifest"]
    manifest = Path(original["manifest_path"])
    locked = load_study(manifest)
    if locked["manifest_sha256"] != report["manifest_sha256"]:
        raise ValueError("study_quality_original_manifest_changed")
    for trial in report["trials"]:
        receipt = _read(output / "receipts" / f"{trial['trial_id']}.json")
        if receipt.get("trial") != trial or receipt.get("manifest_sha256") != report["manifest_sha256"] or receipt.get("artifacts") != _artifact_inventory(Path(trial["run_dir"])):
            raise ValueError("study_quality_original_trial_artifacts_changed")
    quality = quality_report.resolve()
    locked["quality_report"] = {"path": str(quality), "sha256": _sha(quality)}
    result = {"schema_version": "performance_study_quality_assessment.v1", "original_study_sha256": _sha(output / "study.json"),
              "manifest_sha256": locked["manifest_sha256"], "quality_report": locked["quality_report"],
              **summarize_study(locked, report["trials"])}
    result["status"] = "passed" if result["comparisons"] and all(row["quality"]["status"] == "passed" for row in result["comparisons"].values()) else "not_passed"
    _write(output / "study.quality.json", result)
    return result


def _method_span(source: str, name: str) -> ast.FunctionDef:
    nodes = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"engineering_variant_requires_one_method:{name}")
    return nodes[0]


def _replace_method(source: str, name: str, replacement: str) -> str:
    node = _method_span(source, name)
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [textwrap.indent(textwrap.dedent(replacement).rstrip(), " " * node.col_offset) + "\n"]
    return "".join(lines)


def build_engineering_variants(baseline_project: Path, candidate_project: Path, *, output: Path,
                               python: Path | None = None) -> dict[str, Any]:
    """Build a real two-factor ablation from immutable historical source.

    The request factor is precisely routine Locate CUDA/GC cleanup removal.
    The video factor is S1 shared scan + S4 image reuse/bounded two-thread work.
    This intentionally does not label unrelated round-two changes as either
    factor. Both factors retain baseline prompts and generation parameters.
    """
    import yaml
    baseline_project, candidate_project, output = baseline_project.resolve(), candidate_project.resolve(), output.resolve()
    for project in (baseline_project, candidate_project):
        _target_project(project / "src")
        if output == project or project in output.parents:
            raise ValueError("engineering_variants_output_must_be_outside_inputs")
    if output.exists():
        raise FileExistsError(output)
    original_sha = _code_state(baseline_project / "src", baseline_project)
    donor_sha = _code_state(candidate_project / "src", candidate_project)
    output.mkdir(parents=True)
    variants = []
    for name, request_factor, video_factor in (("original", False, False), ("request_overhead", True, False),
                                              ("video_cache", False, True), ("combined", True, True)):
        project = output / name
        for directory in ("src", "scripts", "configs"):
            shutil.copytree(baseline_project / directory, project / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        modifications = []
        if request_factor:
            target = project / "src/classroom_hybrid_ai/adapters/locate_anything.py"
            contents = target.read_text(encoding="utf-8")
            node = _method_span(contents, "_local_answer")
            removed = 0
            for block in ast.walk(node):
                if isinstance(block, ast.Try):
                    kept = []
                    for statement in block.finalbody:
                        call = statement.value if isinstance(statement, ast.Expr) else None
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_empty_cuda_cache":
                            removed += 1
                        else:
                            kept.append(statement)
                    if len(kept) != len(block.finalbody):
                        block.finalbody = kept
                        block.handlers.append(ast.ExceptHandler(type=ast.Name(id="BaseException", ctx=ast.Load()),
                            body=[ast.Expr(value=ast.Call(func=ast.Name(id="_empty_cuda_cache", ctx=ast.Load()), args=[], keywords=[])), ast.Raise()]))
            if removed != 1:
                raise ValueError("engineering_request_cleanup_pattern_changed")
            ast.fix_missing_locations(node)
            target.write_text(_replace_method(contents, "_local_answer", ast.unparse(node)), encoding="utf-8")
            modifications.append("adapters/locate_anything.py:_local_answer; routine_cleanup_removed_exception_cleanup_retained")
        if video_factor:
            for filename in ("video.py", "evidence.py", "performance.py"):
                shutil.copy2(candidate_project / "src/classroom_hybrid_ai" / filename, project / "src/classroom_hybrid_ai" / filename)
                modifications.append(filename)
            stage_path = project / "src/classroom_hybrid_ai/research_stages.py"
            contents = stage_path.read_text(encoding="utf-8")
            donor = (candidate_project / "src/classroom_hybrid_ai/research_stages.py").read_text(encoding="utf-8")
            node = _method_span(donor, "_run_s1")
            replacement = "\n".join(donor.splitlines()[node.lineno - 1:node.end_lineno])
            contents = _replace_method(contents, "_run_s1", replacement)
            needle = "from .video import detect_cpu_change_anchors"
            if contents.count(needle) != 1:
                raise ValueError("engineering_s1_video_import_pattern_changed")
            contents = contents.replace(needle, needle + ", scan_and_materialize_screenshots", 1)
            tree = ast.parse(contents)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "build_event_evidence"]
            if len(calls) != 1 or any(key.arg in {"preprocess_workers", "preprocess_queue_size"} for key in calls[0].keywords):
                raise ValueError("engineering_s4_evidence_call_pattern_changed")
            call = calls[0]
            # Insert actual consumed kwargs in the copied S4 source.
            lines = contents.splitlines(keepends=True)
            lines.insert(call.end_lineno - 1, "            preprocess_workers=2, preprocess_queue_size=2,\n")
            stage_path.write_text("".join(lines), encoding="utf-8")
            config_path = project / "configs/default.yaml"
            policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            policy.setdefault("video", {})["shared_screenshot_scan"] = True
            policy["video"]["screenshot_cache_bytes"] = 2 * 1024**3
            config_path.write_text(yaml.safe_dump(policy, allow_unicode=True, sort_keys=False), encoding="utf-8")
            modifications.extend(["research_stages.py:_run_s1", "research_stages.py:_run_s4:two_preprocess_workers_queue_two", "configs/default.yaml:consumed_video_scan_flags"])
        checked = 0
        for file in (project / "src").rglob("*.py"):
            compile(file.read_text(encoding="utf-8"), str(file), "exec")
            checked += 1
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        check = subprocess.run([str(python or sys.executable), "-B", "-c",
            "import classroom_hybrid_ai.research_stages, classroom_hybrid_ai.adapters.locate_anything"],
            env=env, cwd=project, capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if check.returncode:
            _write(output / "build_failure.json", {"variant": name, "stderr": check.stderr, "status": "failed"})
            raise ValueError(f"engineering_variant_import_failed:{name}:{check.stderr[-500:]}")
        variants.append({"id": name, "source_path": str(project / "src"), "config": str(project / "configs/default.yaml"),
                         "source_sha256": _code_state(project / "src", project), "config_sha256": _sha(project / "configs/default.yaml"),
                         "applied_changes": modifications, "python_files_compiled": checked, "import_validation": "passed"})
    if _code_state(baseline_project / "src", baseline_project) != original_sha or _code_state(candidate_project / "src", candidate_project) != donor_sha:
        raise ValueError("engineering_input_changed_during_build")
    result = {"schema_version": "performance_engineering_variants.v1", "status": "built_not_performance_validated",
              "baseline_source_sha256": original_sha, "donor_source_sha256": donor_sha,
              "factor_scope": {"request_overhead": "Locate routine GC/CUDA cleanup", "video_cache": "S1 shared scan and S4 bounded evidence image processing"},
              "unrepresented_optimizations": ["S0 preview seek", "S7 lazy preview", "stage dependency fingerprints", "HOTA", "round2 algorithm routes"],
              "variants": variants, "output_equivalence": "requires_measured_whole_video_trials"}
    _write(output / "build_receipt.json", result)
    return result
