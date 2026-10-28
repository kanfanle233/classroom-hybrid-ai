"""Isolated production benchmarks, including an immutable source baseline.

The launcher deliberately imports project code only inside the child. This
lets the same measurement harness run an older source snapshot without copying
new instrumentation into that snapshot. Missing old telemetry stays missing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile_requires_observations")
    at = (len(ordered) - 1) * fraction
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def summarize_trials(trials: list[dict[str, Any]], *, target_reduction: float = 0.20) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in dict.fromkeys(row["variant"] for row in trials):
        rows = [row for row in trials if row["variant"] == variant]
        successful = [row for row in rows if row.get("status") == "ok" and (
            row.get("component") != "pipeline" or (
                isinstance(row.get("semantic_validation"), Mapping)
                and row["semantic_validation"].get("schema_version") == "pipeline_semantic_validation.v1"
                and row["semantic_validation"].get("status") == "passed"
            )
        )]
        item: dict[str, Any] = {"trials": len(rows), "successful_trials": len(successful)}
        if successful:
            durations = [float(row["wall_seconds"]) for row in successful]
            item.update(median_wall_seconds=statistics.median(durations), p95_wall_seconds=percentile(durations, .95))
            requests = [float(request["wall_seconds"]) for row in successful for request in row.get("request_observations", row.get("requests", [])) if "wall_seconds" in request]
            if requests:
                item.update(request_p50_seconds=statistics.median(requests), request_p95_seconds=percentile(requests, .95))
            item["semantic_digests"] = sorted({row["semantic_sha256"] for row in successful})
        summary[variant] = item
    comparison: dict[str, Any] = {"status": "baseline_not_provided", "target_reduction_fraction": target_reduction}
    if "baseline" in summary and "candidate" in summary:
        baseline, candidate = summary["baseline"], summary["candidate"]
        if baseline["trials"] != candidate["trials"] or any(item["successful_trials"] != item["trials"] for item in (baseline, candidate)):
            comparison["status"] = "incomplete_or_failed"
        else:
            denominator = baseline["median_wall_seconds"]
            reduction = 1 - candidate["median_wall_seconds"] / denominator if denominator else 0.0
            equivalent = baseline["semantic_digests"] == candidate["semantic_digests"] and len(baseline["semantic_digests"]) == 1
            comparison.update(
                status="target_met" if equivalent and reduction >= target_reduction else "target_not_met",
                median_reduction_fraction=reduction, outputs_equivalent_on_benchmark=equivalent,
            )
    comparison["timing_scope"] = "cold_child_process_lifecycle_including_setup_and_audit; stage_seconds_reported_separately"
    return {"variants": summary, "comparison": comparison}


def _relocate_paths(value: Any, source: Path, destination: Path, *, protected: set[str] | None = None) -> Any:
    """Replace paths inside a copied fixture, preserving the original video."""
    if isinstance(value, dict):
        return {key: _relocate_paths(item, source, destination, protected=protected) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate_paths(item, source, destination, protected=protected) for item in value]
    if isinstance(value, str) and value not in (protected or set()):
        try:
            candidate = Path(value)
            if candidate.is_absolute():
                relative = candidate.resolve().relative_to(source.resolve())
                return str(destination / relative)
        except (ValueError, OSError):
            pass
    return value


def _input_assets(root: Path) -> dict[str, Any]:
    """Candidate order, evidence pixels and timing are part of fair workload."""
    candidates = _rows(root / "artifacts/candidates.selected.jsonl")
    evidence = _rows(root / "evidence/index.jsonl")

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items() if key not in {"created_at", "performance"}}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            try:
                path = Path(value)
                if path.is_absolute():
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        return {"image_sha256": _sha(path)}
                    return "<run>/" + path.resolve().relative_to(root.resolve()).as_posix()
            except (ValueError, OSError):
                pass
        return value
    return {"candidates": normalize(candidates), "evidence": normalize(evidence)}


_PIPELINE_JSONL = (
    "artifacts/screenshot_schedule.v1.jsonl", "artifacts/locate_raw_results.jsonl",
    "artifacts/locate_results.jsonl", "artifacts/locate_student_boxes.jsonl",
    "artifacts/track_observations.jsonl", "artifacts/student_catalog.s3.jsonl",
    "artifacts/candidates.all.jsonl", "artifacts/candidates.selected.jsonl",
    "artifacts/qwen_crop_queue.jsonl", "evidence/index.jsonl",
    "reviews.classification.jsonl", "events.jsonl", "students.jsonl",
    "action_observations.jsonl", "action_intervals.jsonl",
    "artifacts/overlay_rows.v3.jsonl", "frontend/action_observations_display.v1.jsonl",
    "frontend/action_intervals_display.v1.jsonl",
)
_PIPELINE_JSON = (
    "timeline.v2.json", "frontend/timeline_display.v1.json",
    "artifacts/association_audit.v1.json",
)
_PIPELINE_VIDEOS = ("tracking_preview", "action_preview", "annotated_final")
_SEMANTIC_RUNTIME_FIELDS = frozenset({
    "created_at", "updated_at", "generated_at", "run_id", "run_dir",
    "config_sha256", "code_sha256", "experiment_digest", "performance",
    "timings_seconds", "wall_seconds", "elapsed_seconds", "worker_pid",
})


def _media_signature(path: Path) -> dict[str, Any]:
    """Independently decode every video/audio frame; never trust file metadata alone.

    PyAV supplies actual PTS rather than OpenCV's nominal-FPS fallback. RGB and
    decoded native audio samples are hashed with their timing and shape. This is
    strict equality, not a perceptual threshold that could hide changed labels.
    The audit runs after production stage timing has stopped.
    """
    try:
        import av

        with av.open(str(path)) as container:
            video_streams = list(container.streams.video)
            audio_count = len(container.streams.audio)
            if len(video_streams) != 1:
                raise ValueError("expected_one_video_stream")
            stream = video_streams[0]
            fps = float(stream.average_rate or 0)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("video_rate_unverifiable")
            pixels, timing = hashlib.sha256(), hashlib.sha256()
            count, first, last, shape = 0, None, None, None
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                    raise ValueError("video_pts_missing_or_frame_corrupt")
                pts = float(frame.pts * frame.time_base)
                if not math.isfinite(pts) or (last is not None and pts <= last):
                    raise ValueError("video_pts_not_strictly_increasing")
                image = frame.to_ndarray(format="rgb24")
                if shape is not None and shape != list(image.shape):
                    raise ValueError("video_dimensions_changed")
                shape = list(image.shape)
                pixels.update(image.tobytes(order="C"))
                timing.update((_canonical([count, round(pts, 12), shape]) + "\n").encode())
                first = pts if first is None else first
                last, count = pts, count + 1
            if not count:
                raise ValueError("video_has_no_decodable_frames")
            if stream.frames and int(stream.frames) != count:
                raise ValueError("video_stream_frame_count_mismatch")
            video = {
                "frames": count, "fps": fps, "shape": shape,
                "first_pts_seconds": first, "last_pts_seconds": last,
                "duration_seconds": last - first + 1.0 / fps,
                "pts_sha256": timing.hexdigest(), "decoded_rgb_sha256": pixels.hexdigest(),
                "codec": stream.codec_context.name,
            }
        audio = []
        for index in range(audio_count):
            digest = hashlib.sha256()
            frames, samples, first, last, end = 0, 0, None, None, None
            with av.open(str(path)) as container:
                for frame in container.decode(audio=index):
                    if frame.pts is None or frame.time_base is None or not frame.sample_rate or frame.is_corrupt:
                        raise ValueError("audio_pts_or_samples_unverifiable")
                    pts = float(frame.pts * frame.time_base)
                    if not math.isfinite(pts) or (last is not None and pts <= last):
                        raise ValueError("audio_pts_not_strictly_increasing")
                    array = frame.to_ndarray()
                    digest.update((_canonical([round(pts, 12), frame.sample_rate, frame.samples,
                                               frame.layout.name, str(array.dtype), list(array.shape)]) + "\n").encode())
                    digest.update(array.tobytes(order="C"))
                    first = pts if first is None else first
                    last, end = pts, pts + frame.samples / frame.sample_rate
                    frames, samples = frames + 1, samples + frame.samples
            if not frames or not samples:
                raise ValueError("audio_stream_has_no_decodable_samples")
            audio.append({"stream_ordinal": index, "decoded_frames": frames, "samples": samples,
                          "first_pts_seconds": first, "end_pts_seconds": end,
                          "decoded_samples_and_pts_sha256": digest.hexdigest()})
        return {"status": "passed", "video": video, "audio": {"stream_count": audio_count, "streams": audio}}
    except Exception as exc:
        # Decoder-specific failures also mean "not verified". Never turn an
        # unrecognised media exception into successful semantic equivalence.
        return {"status": "failed", "reason": f"media_unverifiable:{type(exc).__name__}:{exc}"}


def _pipeline_semantics(root: Path) -> dict[str, Any]:
    """Validate the durable S2-S7 contract, including all rendered media.

    Empty JSONL files are legitimate for videos without students/candidates;
    an absent file is not equivalent to an empty result. Both old and current
    v3.1 outputs expose these paths; no new production telemetry is required.
    """
    errors: list[str] = []
    images: dict[Path, str] = {}

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items() if key not in _SEMANTIC_RUNTIME_FIELDS}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            try:
                path = Path(value)
                if path.is_absolute():
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        if path not in images:
                            images[path] = _sha(path)
                        return {"image_sha256": images[path]}
                    return "<run>/" + path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                pass  # Absolute source/model paths outside the run stay explicit.
        return value

    def read(relative: str, *, rows: bool = False) -> Any:
        path = root / relative
        try:
            if not path.is_file():
                raise ValueError("missing")
            value = _rows(path) if rows else _read(path)
            if (rows and (not isinstance(value, list) or any(not isinstance(row, dict) for row in value))) or (not rows and not isinstance(value, dict)):
                raise ValueError("invalid_structure")
            normalized = normalize(value)
            # Reject NaN/Infinity, which are not valid portable JSON semantics.
            json.dumps(normalized, allow_nan=False)
            return normalized
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"artifact_unverifiable:{relative}:{type(exc).__name__}")
            return None

    artifacts = {name: read(name, rows=True) for name in _PIPELINE_JSONL}
    for name in ("artifacts/localization_raw_results.v1.jsonl", "artifacts/localization_results.v1.jsonl", "artifacts/localization_boxes.v1.jsonl"):
        if (root / name).is_file():
            artifacts[name] = read(name, rows=True)
    artifacts.update({name: read(name) for name in _PIPELINE_JSON})
    frontend = read("frontend/index.v1.json") or {}
    manifest = read("manifest.json") or {}
    # Only production display fields are compared; resource/manifest receipts
    # contain intentionally different timing, source hashes and code versions.
    frontend_fields = ("display_language", "students_uri", "actions_uri", "intervals_uri", "timeline_uri",
                       "raw_actions_uri", "raw_intervals_uri", "raw_timeline_uri", "video_uris",
                       "video_metadata", "overlay_spool_uri", "students", "actions", "taxonomy_profile_id", "taxonomy_contract_sha256")
    artifacts["frontend/index.v1.json"] = {key: frontend.get(key) for key in frontend_fields}
    for key in ("video_uris", "video_metadata"):
        if not isinstance(frontend.get(key), dict) or any(name not in frontend[key] for name in _PIPELINE_VIDEOS):
            errors.append(f"frontend_{key}_incomplete")
    chunks = frontend.get("overlay_chunks")
    if not isinstance(chunks, list) or not chunks or any(not isinstance(name, str) for name in chunks):
        errors.append("overlay_chunks_missing_or_invalid")
        chunks = []
    overlay_chunks = []
    for name in chunks:
        path = (root / name).resolve()
        if root.resolve() not in path.parents or path.suffix != ".json":
            errors.append("overlay_chunk_path_invalid")
            continue
        value = read(name)
        if value is not None and (not isinstance(value.get("rows"), list) or any(not isinstance(row, dict) for row in value["rows"])):
            errors.append(f"overlay_chunk_rows_invalid:{name}")
        overlay_chunks.append({"uri": name, "content": value})
    artifacts["frontend/overlay_chunks"] = overlay_chunks
    timeline = artifacts.get("timeline.v2.json")
    if timeline is not None:
        if not isinstance(timeline.get("intervals"), list) or not isinstance(timeline.get("student_rows"), list):
            errors.append("timeline_structure_invalid")
        elif timeline["intervals"] != artifacts["action_intervals.jsonl"]:
            errors.append("timeline_intervals_disagree_with_action_intervals")
        students = artifacts.get("students.jsonl")
        if isinstance(students, list) and timeline.get("student_rows") != [row.get("student_id") for row in students]:
            errors.append("timeline_student_rows_disagree_with_students")
    media = {}
    for name in _PIPELINE_VIDEOS:
        relative = f"videos/{name}.mp4"
        receipt = manifest.get("videos", {}).get(name) if isinstance(manifest.get("videos"), dict) else None
        if not isinstance(receipt, dict):
            errors.append(f"video_receipt_missing:{name}")
            receipt = {}
        if receipt.get("uri") != relative or receipt.get("semantic_id") != name:
            errors.append(f"video_receipt_semantics_invalid:{name}")
        if not isinstance(receipt.get("clip_ranges"), list) or not isinstance(receipt.get("frame_ranges"), list):
            errors.append(f"video_ranges_unverifiable:{name}")
        if type(receipt.get("audio_preserved")) is not bool:
            errors.append(f"video_audio_preservation_state_unverifiable:{name}")
        try:
            if not isinstance(receipt.get("sha256"), str) or _sha(root / relative) != receipt["sha256"]:
                errors.append(f"video_receipt_sha_mismatch:{name}")
        except OSError:
            errors.append(f"video_file_missing_or_unreadable:{name}")
        signature = _media_signature(root / relative)
        if signature["status"] != "passed":
            errors.append(f"video_unverifiable:{name}")
        else:
            observed = signature["video"]
            expected_frames = receipt.get("frames")
            if type(expected_frames) is not int or expected_frames != observed["frames"]:
                errors.append(f"video_frame_count_mismatch:{name}")
            expected_fps = receipt.get("fps")
            if isinstance(expected_fps, bool) or not isinstance(expected_fps, (int, float)) or not math.isfinite(expected_fps) or abs(expected_fps - observed["fps"]) > 1e-3:
                errors.append(f"video_fps_mismatch:{name}")
            if receipt.get("audio_status") not in {"preserved", "unavailable", "failed", "absent"}:
                errors.append(f"video_audio_status_unverifiable:{name}")
            if (receipt.get("audio_status") == "preserved") != (receipt.get("audio_preserved") is True):
                errors.append(f"video_audio_receipt_inconsistent:{name}")
            if receipt.get("audio_preserved") is True and signature["audio"]["stream_count"] == 0:
                errors.append(f"video_claimed_audio_missing:{name}")
        encoder = receipt.get("encoder") or {}
        signature["encoding_strategy"] = {key: encoder.get(key, "opencv") for key in ("requested", "effective")}
        signature["output_contract"] = {key: receipt.get(key) for key in (
            "semantic_id", "semantic", "clip_ranges", "frame_ranges", "audio_status", "audio_preserved")}
        media[name] = signature
    return {
        "artifacts": artifacts, "videos": media,
        "validation": {"schema_version": "pipeline_semantic_validation.v1", "status": "failed" if errors else "passed",
                       "errors": errors, "video_equality": "all_decoded_rgb_pixels_and_pts_same_encoding_strategy",
                       "audio_equality": "verified_stream_presence_and_decoded_samples_with_pts; shared_baseline_audio_omissions_remain_explicit"},
    }


def _semantic_outputs(root: Path, component: str) -> dict[str, Any]:
    """Compare model/temporal decisions and input pixels, excluding diagnostics."""
    result: dict[str, Any] = {}
    if component in {"io", "locate", "pipeline"}:
        screenshots = []
        for row in _rows(root / "artifacts/screenshot_schedule.v1.jsonl"):
            screenshots.append({
                "frame_index": row.get("frame_index"), "pts_seconds": row.get("pts_seconds"),
                "anchor_source": row.get("anchor_source"),
                "image_sha256": _sha(Path(row["path"])),
            })
        result["screenshots"] = screenshots
    if component in {"locate", "pipeline"}:
        localization_path = root / "artifacts/localization_raw_results.v1.jsonl"
        if not localization_path.is_file():
            localization_path = root / "artifacts/locate_raw_results.jsonl"
        result["locate"] = [
            {"frame_index": row.get("frame_index"), "pts_seconds": row.get("pts_seconds"),
             "detections": row.get("detections", []), "raw_answer": row.get("raw_answer")}
            for row in _rows(localization_path)
        ]
    if component in {"qwen", "pipeline"}:
        result["inputs"] = _input_assets(root)
        result["reviews"] = [
            {key: row.get(key) for key in ("candidate_event_id", "decision", "behavior", "schema_valid", "evidence_frame_indices")}
            for row in _rows(root / "reviews.classification.jsonl")
        ]
    if component == "pipeline":
        result["complete_pipeline"] = _pipeline_semantics(root)
    return result


def _copy_fixture(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("benchmark_fixture_destination_must_be_separate")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("artifacts", "evidence"):
        if (source / name).is_dir():
            shutil.copytree(source / name, destination / name, dirs_exist_ok=True)
    for name in ("run_manifest.v1.json", "session_vocabulary.v1.json", "taxonomy_profile.v1.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    manifest = _read(source / "run_manifest.v1.json") if (source / "run_manifest.v1.json").is_file() else {}
    protected = {str(manifest.get("source", {}).get("path", ""))}
    for path in destination.rglob("*"):
        if path.suffix == ".json" and path.is_file():
            _write(path, _relocate_paths(_read(path), source, destination, protected=protected))
        elif path.suffix == ".jsonl" and path.is_file():
            rows = _relocate_paths(_rows(path), source, destination, protected=protected)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _target_project(source: Path) -> Path:
    source = source.resolve()
    project = source.parent
    if (project / "configs").is_dir() and (project / "scripts").is_dir():
        return project
    # Installed package includes its own config and packaged service resource.
    packaged = source / "classroom_hybrid_ai"
    if (packaged / "configs").is_dir() and (packaged / "resources/locate_anything_service.py").is_file():
        return packaged
    raise ValueError(f"benchmark_source_requires_matching_configs_and_service:{source}")


def _bound_qwen_fixture(root: Path, limit: int) -> None:
    selected = _rows(root / "artifacts/candidates.selected.jsonl")
    queue = _rows(root / "artifacts/qwen_crop_queue.jsonl")
    ready = {str(row.get("candidate_event_id")) for row in queue if row.get("status") == "ready"}
    ids = [str(row.get("event_id")) for row in selected if str(row.get("event_id")) in ready][:limit]
    wanted = set(ids)
    if not wanted:
        raise ValueError("benchmark_qwen_fixture_has_no_ready_candidates")
    for relative, rows in (
        ("artifacts/candidates.selected.jsonl", [row for row in selected if str(row.get("event_id")) in wanted]),
        ("artifacts/qwen_crop_queue.jsonl", [row for row in queue if str(row.get("candidate_event_id")) in wanted]),
    ):
        (root / relative).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _code_state(source: Path, project: Path) -> str:
    paths = {}
    for label, directory in (("package", source / "classroom_hybrid_ai"), ("scripts", project / "scripts"), ("configs", project / "configs")):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".json"}:
                paths[f"{label}/{path.relative_to(directory).as_posix()}"] = _sha(path)
    return _canonical(paths)


def _collect_model_lifecycles(*observations: Any) -> list[dict[str, Any]]:
    """Keep one cumulative loading record per worker, never one per review."""
    workers: dict[int, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            lifecycle = value.get("worker_lifecycle")
            if isinstance(lifecycle, Mapping):
                pid = lifecycle.get("worker_pid")
                if isinstance(pid, int) and not isinstance(pid, bool):
                    previous = workers.setdefault(pid, {"worker_pid": pid, "model": "qwen"})
                    for key, item in lifecycle.items():
                        if key in {"timings_seconds", "parent_timings_seconds", "counts"} and isinstance(item, Mapping):
                            target = previous.setdefault(key, {})
                            for metric, number in item.items():
                                if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number):
                                    target[metric] = max(target.get(metric, number), number)
                        else:
                            previous[key] = item
            for key, item in value.items():
                if key != "worker_lifecycle":
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    for observation in observations:
        visit(observation)
    return [workers[pid] for pid in sorted(workers)]


@contextmanager
def _observe_requests(observations: list[dict[str, Any]]):
    """Identical outer host timing for old and new production adapters."""
    import importlib
    originals = []
    specifications = (
        ("classroom_hybrid_ai.adapters.locate_anything", "LocateAnythingAdapter", "locate", "locate"),
        ("classroom_hybrid_ai.adapters.qwen", "Qwen35Adapter", "review", "qwen"),
        ("classroom_hybrid_ai.qwen_worker", "QwenProcessClient", "review", "qwen"),
        ("classroom_hybrid_ai.yolo_worker", "YoloProcessClient", "predict_batch", "yolo"),
    )

    def instrument(original, model):
        @wraps(original)
        def invoke(instance, *args, **kwargs):
            started = time.perf_counter()
            outcome = "ok"
            error = None
            try:
                return original(instance, *args, **kwargs)
            except BaseException as exc:
                outcome = "failed"
                error = type(exc).__name__
                raise
            finally:
                observations.append({"model": model, "wall_seconds": time.perf_counter() - started,
                                     "status": outcome, "error_type": error,
                                     "caller_pid": os.getpid(),
                                     "worker_pid": getattr(instance, "pid", None) or (getattr(instance, "receipt", {}) or {}).get("pid"),
                                     "timing_scope": "outer_production_adapter_host_wall; same_harness_for_both_variants"})
        return invoke
    try:
        for module_name, class_name, method_name, model in specifications:
            try:
                cls = getattr(importlib.import_module(module_name), class_name)
                original = getattr(cls, method_name)
            except (ImportError, AttributeError):
                continue
            originals.append((cls, method_name, original))
            setattr(cls, method_name, instrument(original, model))
        yield
    finally:
        for cls, method_name, original in reversed(originals):
            setattr(cls, method_name, original)


def _child(job: dict[str, Any]) -> dict[str, Any]:
    child_started = time.perf_counter()
    project_root = _target_project(Path(job["source_path"]))
    original_code_sha = _code_state(Path(job["source_path"]), project_root)
    original_config_sha = _sha(Path(job["config"]))
    sys.path.insert(0, job["source_path"])
    from classroom_hybrid_ai.config import load_config, ProjectConfig
    from classroom_hybrid_ai.research_backend import ResearchBackend
    from classroom_hybrid_ai.research_stages import ResearchStageExecutor
    from classroom_hybrid_ai.storage import RunStore
    import classroom_hybrid_ai.research_backend as backend_module

    config = load_config(job["config"])
    payload = json.loads(json.dumps(config.payload))
    payload.setdefault("paths", {})["output_root"] = str(Path(job["trial_dir"]) / "runs")
    # The fixture is copied for each measured trial: no inherited response cache.
    payload.setdefault("performance", {})["profile_model_timings"] = bool(job.get("profile_model_timings", False))
    config = ProjectConfig(config.source_path, project_root, Path(job["repository_root"]), payload)
    backend = ResearchBackend(config)
    video = Path(job["video"])
    component = job["component"]
    profile = backend.profile()
    mode = str(config.get("research.mode", "production_adaptive"))
    config = backend_module._effective_config(config, profile=profile, mode=mode)
    payload = config.payload
    backend = ResearchBackend(config)
    setup_start = time.perf_counter()
    store = None
    if component != "pipeline":
        store = RunStore(config.output_root, video.stem, _canonical(payload), unique_run=True)
        if job.get("run_dir"):
            _copy_fixture(Path(job["run_dir"]), store.run_dir)
            if component == "qwen" and job.get("max_requests") is not None:
                _bound_qwen_fixture(store.run_dir, job["max_requests"])
        spec = backend._build_spec(
            store=store, source=video, profile=profile, mode=mode,
            max_frames=job.get("max_frames"), skip_qwen=False,
            allow_incomplete_preflight=component == "io", effective_config=config,
            experiment_digest=_canonical({"benchmark": job, "profile": profile.sha256}),
        )
        executor = ResearchStageExecutor(spec)
        # IO never loads a model or claims to have passed production preflight.
        if component == "io":
            executor.preflight_runner = lambda *a, **k: {
                "status": "ok", "warnings": [], "scope": "io_benchmark_only",
                "model_preflight": "not_performed_io_component",
            }
    setup_seconds = time.perf_counter() - setup_start
    stage_results = {}
    request_observations: list[dict[str, Any]] = []
    start = time.perf_counter()
    with _observe_requests(request_observations):
        if component == "pipeline":
            pipeline_result = backend.run(video, max_frames=job.get("max_frames"))
            root = Path(pipeline_result.run_dir)
            stage_results = pipeline_result.stages
            status = "ok" if pipeline_result.status == "ok" or (
                pipeline_result.status == "warning" and all("audio_not_preserved:" in warning for warning in pipeline_result.warnings)
            ) else "failed"
        else:
            root = store.run_dir
            stages = {"io": ("S0", "S1"), "locate": ("S2",), "qwen": ("S5",),
                      "prepare_locate": ("S0", "S1"), "prepare_qwen": ("S0", "S1", "S2", "S3", "S4")}[component]
            for stage in stages:
                stage_start = time.perf_counter()
                outcome = executor.execute(stage, input_fingerprint_sha256=_canonical({"config": payload, "stage": stage}), partial_outputs_reusable=False)
                stage_results[stage] = {"wall_seconds": time.perf_counter() - stage_start, "stats": dict(outcome.stats), "status": outcome.status,
                                        "warnings": list(outcome.warnings), "errors": list(outcome.errors)}
                if outcome.status not in {"completed", "ok"} or outcome.errors:
                    break
            status = "ok" if all(row["status"] in {"completed", "ok"} and not row["errors"] for row in stage_results.values()) else "failed"
            if component == "qwen" and not stage_results.get("S5", {}).get("stats", {}).get("model_started"):
                status = "failed"
                stage_results.setdefault("S5", {}).setdefault("errors", []).append("benchmark_qwen_model_not_started")
    wall = time.perf_counter() - start
    execution_wall = time.perf_counter() - child_started
    audit_started = time.perf_counter()
    requests = []
    for name in ("locate_requests.performance.jsonl", "qwen_requests.performance.jsonl"):
        requests.extend(_rows(root / "diagnostics" / name))
    diagnostics = {}
    for file in sorted((root / "diagnostics/performance").glob("*.json")):
        diagnostics[file.stem] = _read(file)
    lifecycles = _collect_model_lifecycles(requests, diagnostics)
    code_sha = _code_state(Path(job["source_path"]), project_root)
    source_changed = code_sha != original_code_sha or _sha(Path(job["config"])) != original_config_sha
    if source_changed:
        status = "failed"
    semantic = _semantic_outputs(root, component)
    workload_errors = []
    semantic_validation = semantic.get("complete_pipeline", {}).get("validation")
    if component == "pipeline" and (not semantic_validation or semantic_validation.get("status") != "passed"):
        workload_errors.extend((semantic_validation or {}).get("errors") or ["pipeline_semantic_validation_missing"])
    if component in {"io", "locate", "pipeline"} and not semantic.get("screenshots"):
        workload_errors.append("benchmark_has_no_production_screenshots")
    if component == "qwen" and not semantic.get("inputs", {}).get("candidates"):
        workload_errors.append("benchmark_has_no_selected_candidates")
    if workload_errors:
        status = "failed"
    manifest = _read(root / "run_manifest.v1.json") if (root / "run_manifest.v1.json").is_file() else {}
    model_fingerprints = manifest.get("fingerprints", {})
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "safetensors",
                 "numpy", "opencv-python", "av", "Pillow", "PyYAML", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "status": status, "run_dir": str(root), "component": component, "child_pid": os.getpid(),
        "errors": workload_errors,
        "wall_seconds": wall, "stage_execution_seconds": wall,
        "child_execution_seconds": execution_wall, "setup_seconds": setup_seconds,
        "postrun_audit_seconds": time.perf_counter() - audit_started,
        "cold_model_load_included": component not in {"io", "prepare_locate"}, "stage_results": stage_results,
        "stage_performance": diagnostics, "requests": requests,
        "request_observations": request_observations,
        "model_lifecycles": lifecycles,
        "model_lifecycle_semantics": "one_cumulative_record_per_worker; model_load_is_separate_from_resident_request_model_load_zero",
        "semantic_sha256": _canonical(semantic), "semantic_outputs": semantic,
        "semantic_validation": semantic_validation,
        "source_sha256": _sha(video), "code_sha256": code_sha,
        "code_sha256_before_execution": original_code_sha, "source_changed_during_trial": source_changed,
        "config_sha256": _canonical(payload), "effective_config": payload,
        "profile_sha256": profile.sha256,
        "model_fingerprints": {"locate": model_fingerprints.get("locate_model_sha256"), "qwen": model_fingerprints.get("qwen_model_sha256")}
        if component == "pipeline" else {"locate": spec.locate_model_sha256, "qwen": spec.qwen_model_sha256},
        "target_project_root": str(project_root),
        "missing_reasons": {"request_substage_metrics": "baseline_has_no_instrumentation"} if not requests and component not in {"io", "prepare_locate"} else {},
        "evaluation_status": "not_evaluated", "reason": "performance_benchmark_without_gold",
        "versions": versions,
    }


def _launch(job: dict[str, Any], assets: Path, python: Path) -> dict[str, Any]:
    name = job["name"]
    job_file, result_file = assets / f"{name}.job.json", assets / f"{name}.result.json"
    if result_file.exists() or job_file.exists():
        return {"status": "failed", "error": "benchmark_trial_artifacts_already_exist", "wall_seconds": 0.0}
    job["result_file"] = str(result_file)
    _write(job_file, job)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    seed_directory = None
    if job.get("seed") is not None:
        # The startup hook is isolated to this trial, including its spawned
        # Qwen and HTTP Locate workers. Never modify the installed environment.
        from classroom_hybrid_ai.performance_study import prepare_seed_control
        seed_directory = assets / f"{name}.seed"
        environment.update(prepare_seed_control(seed_directory, job["seed"]))
    started = time.perf_counter()
    try:
        with (assets / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, (assets / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                [str(python), "-B", str(Path(__file__).resolve()), "--job", str(job_file)],
                env=environment, stdout=stdout, stderr=stderr, cwd=job["repository_root"],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        lifecycle_seconds = time.perf_counter() - started
        value = _read(result_file) if result_file.is_file() else {"status": "failed", "error": "worker_result_missing"}
        if not isinstance(value, dict):
            raise ValueError("worker_result_must_be_object")
    except (OSError, ValueError) as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}:{exc}", "wall_seconds": time.perf_counter() - started}
    value.update(process_exitcode=result.returncode, process_lifecycle_seconds=lifecycle_seconds,
                 wall_seconds=lifecycle_seconds,
                 wall_scope="cold_child_process_lifecycle_including_fixture_setup_postrun_audit_and_interpreter_teardown")
    if result.returncode:
        value["status"] = "failed"
    if seed_directory is not None:
        from classroom_hybrid_ai.performance_study import verify_seed_control
        value["seed_control"] = verify_seed_control(seed_directory, job["seed"], value)
        if value["seed_control"]["status"] != "passed":
            value["status"] = "failed"
    return value


def run_benchmark(config: Any, video: Path, *, component: str = "pipeline", repeats: int = 3,
                  output: Path, baseline_source: Path | None = None, baseline_config: Path | None = None,
                  run_dir: Path | None = None, max_frames: int | None = None,
                  max_requests: int | None = None,
                  profile_model_timings: bool = False) -> dict[str, Any]:
    if component not in {"pipeline", "locate", "qwen", "io"} or isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("invalid_benchmark_component_or_repeats")
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError("max_frames_must_be_positive")
    if max_requests is not None and (component != "qwen" or isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1):
        raise ValueError("max_requests_requires_qwen_and_positive_integer")
    source_sha = _sha(video.resolve())
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"benchmark_output_exists:{output}")
    if run_dir is not None and component not in {"locate", "qwen"}:
        raise ValueError("run_dir_fixture_supported_only_for_model_components")
    assets = output.with_name(output.stem + ".assets")
    if run_dir is not None:
        fixture_root = run_dir.resolve()
        if any(path == fixture_root or fixture_root in path.parents for path in (output, assets)):
            raise ValueError("benchmark_output_must_be_outside_fixture")
    current_source = Path(__file__).resolve().parents[1]
    base_job = {
        "source_path": str(current_source), "config": str(config.source_path),
        "repository_root": str(config.repository_root), "video": str(video.resolve()),
        "component": component, "max_frames": max_frames,
        "max_requests": max_requests,
        "profile_model_timings": profile_model_timings,
    }
    if run_dir is not None:
        manifest = _read(run_dir / "run_manifest.v1.json")
        if manifest["source"]["sha256"] != source_sha:
            raise ValueError("benchmark_fixture_source_mismatch")
    # Reject incomplete snapshots before allocating output artifacts or GPU work.
    _target_project(current_source)
    if baseline_source is not None:
        _target_project(baseline_source)
    assets.mkdir(parents=True, exist_ok=False)
    preparation = None
    if component in {"locate", "qwen"} and run_dir is None:
        prep_job = {**base_job, "name": "prepare", "component": f"prepare_{component}", "trial_dir": str(assets / "prepare")}
        preparation = _launch(prep_job, assets, config.python_executable)
        if preparation.get("status") != "ok":
            report = {"schema_version": "classroom_hybrid_ai.performance_benchmark.v1", "status": "failed", "preparation": preparation}
            _write(output, report)
            return report
        run_dir = Path(preparation["run_dir"])
    base_job["run_dir"] = str(run_dir.resolve()) if run_dir is not None else None
    variants = [("candidate", current_source, config.source_path)]
    if baseline_source is not None or baseline_config is not None:
        source = (baseline_source or current_source).resolve()
        if not (source / "classroom_hybrid_ai/research_backend.py").is_file():
            raise ValueError(f"baseline_source_must_contain_package:{source}")
        variants.insert(0, ("baseline", source, (baseline_config or config.source_path).resolve()))
    report: dict[str, Any] = {
        "schema_version": "classroom_hybrid_ai.performance_benchmark.v1", "status": "running",
        "component": component, "source_sha256": source_sha, "video": str(video.resolve()),
        "max_frames": max_frames, "max_requests": max_requests,
        "bounded_workload": max_frames is not None or max_requests is not None,
        "evaluation_status": "not_evaluated", "quality_gate": "independent_gold_required_for_changed_predictions",
        "preparation": preparation, "trials": [], "repeats_per_variant": repeats,
        "execution_order": "baseline_then_candidate_each_round", "model_history_reused": False,
        "sampling_seed_controlled": False,
        "sampling_note": "production Locate hybrid sampling is retained; exact-output equivalence may fail across independent samples",
        "timing_scope": "cold_child_process_lifecycle; fixture_setup_and_postrun_audit_reported_separately",
    }
    progress_path = output.with_name(output.name + ".progress.json")
    _write(progress_path, report)
    for iteration in range(repeats):
        for variant, source, config_path in variants:
            name = f"{iteration + 1:03d}-{variant}"
            job = {**base_job, "name": name, "source_path": str(source), "config": str(config_path), "trial_dir": str(assets / name)}
            _write(progress_path, {**report, "active_trial": name})
            trial = _launch(job, assets, config.python_executable)
            trial.update(variant=variant, iteration=iteration + 1)
            report["trials"].append(trial)
            report.update(summarize_trials(report["trials"]))
            _write(progress_path, report)
            if trial.get("status") != "ok":
                report["status"] = "failed"
                _write(output, report)
                _write(progress_path, report)
                return report
    report["status"] = "ok"
    report.update(summarize_trials(report["trials"]))
    _write(output, report)
    _write(progress_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    job = _read(parser.parse_args().job)
    # Load the harness sampler from this source, while keeping target package
    # imports free to resolve to the immutable baseline.
    import importlib.util
    module_spec = importlib.util.spec_from_file_location("benchmark_resource_sampler", Path(__file__).with_name("performance.py"))
    sampler_module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = sampler_module
    module_spec.loader.exec_module(sampler_module)
    sampler = sampler_module.ResourceSampler(gpu=job["component"] not in {"io", "prepare_locate"})
    try:
        with sampler:
            result = _child(job)
        result["resources"] = sampler.payload()
        _write(Path(job["result_file"]), result)
        raise SystemExit(0 if result["status"] == "ok" else 1)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _write(Path(job["result_file"]), {
            "status": "failed", "error": f"{type(exc).__name__}:{exc}",
            "component": job["component"], "trial_dir": job["trial_dir"],
            "source_path": job["source_path"], "resources": sampler.payload(),
        })
        raise SystemExit(1)
