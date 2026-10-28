"""Repeatable v3 hardware gate for the selected Qwen3.5-9B 4-bit snapshot."""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from .adapters import Qwen35Adapter, qwen35_9b_8gb_device_map
from .config import ProjectConfig
from .hashing import canonical_sha256, project_config_digest, python_source_digest, sha256_file
from .qwen_worker import QwenProcessClient
from .runtime import GpuLease, release_model
from .storage import atomic_write_json


def _physical_gpu_used_mib(gpu_index: int) -> float | None:
    """Read whole-device use.

    On Windows WDDM this includes desktop/driver/other-process allocations and
    is therefore retained as a warning metric rather than a leak gate.
    """

    try:
        completed = subprocess.run(
            [
                "nvidia-smi", f"--id={gpu_index}",
                "--query-gpu=memory.used", "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return float(completed.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _physical_gpu_measurement_scope() -> str:
    """Describe whether whole-device NVML readings are delivery-gate safe.

    Windows WDDM owns and pages device memory across desktop, driver and all
    applications.  Its whole-device value is useful diagnostics, but it is not
    attributable to the isolated Qwen worker.  On non-Windows hosts the same
    NVML device reading is treated as a reliable headroom measurement unless a
    future backend records a more specific scope.
    """

    return (
        "device_total_wddm_warning_only"
        if platform.system().lower() == "windows"
        else "device_total_nvml_reliable"
    )


def _physical_gpu_total_mib(gpu_index: int, torch: Any) -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", f"--id={gpu_index}",
                "--query-gpu=memory.total", "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return float(completed.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        try:
            return float(torch.cuda.get_device_properties(gpu_index).total_memory / 2**20)
        except Exception:
            return None


def _physical_gpu_process_used_mib(gpu_index: int, pid: int | None) -> float | None:
    """Read this worker's physical allocation where NVML exposes it.

    Linux compute mode normally returns rows.  Windows WDDM commonly returns
    no usable per-process rows, so ``None`` is an expected result there.
    """

    if pid is None:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", f"--id={gpu_index}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True, timeout=10,
        )
        total = 0.0
        found = False
        for line in completed.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 2:
                continue
            try:
                if int(fields[0]) == int(pid):
                    total += float(fields[1])
                    found = True
            except ValueError:
                continue
        return total if found else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _smoke_image(config: ProjectConfig, media: str | Path) -> tuple[Path, str]:
    source = Path(media).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"smoke_media_not_found:{source}")
    import cv2

    is_image = source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if is_image:
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"smoke_image_decode_failed:{source}")
        source_kind = "image"
    else:
        capture = cv2.VideoCapture(str(source))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"smoke_video_first_frame_decode_failed:{source}")
        source_kind = "video_first_frame"

    # The production Qwen path always receives one 480x416 temporal
    # composite.  Normalize smoke inputs to the same visual-token profile;
    # otherwise a 720p source frame makes the hardware gate much slower and
    # measures a workload the pipeline never sends.
    target_width, target_height = 480, 416
    if int(frame.shape[1]) == target_width and int(frame.shape[0]) == target_height and is_image:
        return source, "image_480x416"
    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    directory = config.resolve(".runtime/classroom_hybrid_ai/smoke_inputs")
    assert directory is not None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        canonical_sha256({
            "path": str(source),
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "shape": [target_width, target_height],
        })[:20]
        + ".jpg"
    )
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"smoke_frame_write_failed:{path}")
    return path, f"{source_kind}_resized_480x416"


def _smoke_response_contract(response: Any) -> tuple[bool, str | None]:
    """Validate the candidate binding used by the stability workload."""

    if not isinstance(response, dict):
        return False, "response_not_mapping"
    decision = response.get("decision")
    behavior = response.get("behavior")
    if decision == "confirmed" and behavior != "action_raise_hand":
        return False, "confirmed_behavior_must_match_candidate"
    if decision in {"rejected", "uncertain"} and behavior is not None:
        return False, "non_confirmed_behavior_must_be_null"
    if response.get("observability") not in {"clear", "partial", "none"}:
        return False, "observability_invalid"
    return True, None


def _allocator_snapshot(adapter: Any, torch: Any) -> dict[str, Any]:
    """Read allocator counters from the process that owns Qwen when possible."""

    metrics = getattr(adapter, "last_metrics", None)
    if isinstance(metrics, dict) and metrics.get("allocated_bytes") is not None:
        return dict(metrics)
    result: dict[str, Any] = {
        "allocated_bytes": None,
        "reserved_bytes": None,
        "inactive_split_bytes": None,
        "num_alloc_retries": None,
        "num_ooms": None,
    }
    try:
        if not torch.cuda.is_available():
            return result
        result["allocated_bytes"] = int(torch.cuda.memory_allocated())
        result["reserved_bytes"] = int(torch.cuda.memory_reserved())
        stats = torch.cuda.memory_stats()
        for key in ("inactive_split_bytes.all.current", "inactive_split_bytes"):
            if key in stats:
                result["inactive_split_bytes"] = int(stats[key])
                break
        for key in ("num_alloc_retries", "num_alloc_retries.all"):
            if key in stats:
                result["num_alloc_retries"] = int(stats[key])
                break
        for key in ("num_ooms", "num_ooms.all"):
            if key in stats:
                result["num_ooms"] = int(stats[key])
                break
    except Exception as exc:
        result["metrics_error"] = f"{type(exc).__name__}:{exc}"[:500]
    return result


def _median_window(values: Sequence[float], start: int, end: int) -> float | None:
    window = [float(value) for value in values[start:end]]
    return None if not window else float(statistics.median(window))


def _robust_slope(values: Sequence[float], *, start: int = 10) -> float | None:
    """Theil-Sen-like median slope, expressed in MiB per request."""

    points = [(index, float(value)) for index, value in enumerate(values[start:], start=start)]
    slopes: list[float] = []
    for left_index, (left_x, left_y) in enumerate(points):
        for right_x, right_y in points[left_index + 1:]:
            if right_x != left_x:
                slopes.append((right_y - left_y) / (right_x - left_x))
    return None if not slopes else float(statistics.median(slopes))


def _growth_metrics(values: Sequence[float]) -> dict[str, Any]:
    first = _median_window(values, 10, 30)
    last = _median_window(values, 80, 100)
    windows = [
        _median_window(values, 10, 30),
        _median_window(values, 40, 60),
        _median_window(values, 70, 90),
    ]
    rising = (
        len(windows) == 3
        and all(value is not None for value in windows)
        and float(windows[1]) > float(windows[0])
        and float(windows[2]) > float(windows[1])
    )
    return {
        "warmup_excluded_first_window_mib": first,
        "steady_last_window_mib": last,
        "growth_11_30_to_81_100_mib": None if first is None or last is None else last - first,
        "robust_slope_mib_per_request": _robust_slope(values),
        "window_medians_mib": windows,
        "three_consecutive_windows_rising": rising,
    }


def _process_rss_mib(process: Any, pid: int | None) -> float | None:
    if process is None:
        return None
    try:
        target = process if pid is None else process(pid)
        return float(target.memory_info().rss / 2**20)
    except Exception:
        return None


def _qwen_adapter_for_smoke(config: ProjectConfig, model_path: Path) -> Any:
    gpu_index = int(config.get("runtime.gpu_index", 0))
    device_map_policy = str(config.get("qwen.device_map_policy", "auto"))
    device_map: Any = (
        qwen35_9b_8gb_device_map(gpu_index)
        if device_map_policy == "explicit_qwen35_9b_8gb" else device_map_policy
    )
    kwargs: dict[str, Any] = {
        "model": str(model_path),
        "backend": "local",
        "max_new_tokens": min(64, int(config.get("qwen.max_new_tokens", 128))),
        "max_oom_retries": 0,
        "device_map": device_map,
        "max_memory": {
            gpu_index: f"{float(config.get('qwen.gpu_memory_limit_gib'))}GiB",
            "cpu": f"{float(config.get('qwen.cpu_offload_gib'))}GiB",
        },
        "offload_folder": config.resolve(config.get("qwen.disk_offload_dir")),
        "offload_state_dict": True,
        "enable_thinking": False,
        "release_cuda_cache": bool(config.get("qwen.release_cuda_cache", False)),
    }
    if bool(config.get("qwen.process_isolation", True)):
        return QwenProcessClient(
            kwargs,
            timeout=float(config.get("qwen.request_timeout_seconds", 300.0)),
        )
    kwargs.pop("model")
    return Qwen35Adapter(str(model_path), **kwargs)


def _smoke_media_list(media: str | Path | Sequence[str | Path]) -> tuple[str | Path, ...]:
    if isinstance(media, (str, Path)):
        return (media,)
    values = tuple(media)
    if not values:
        raise ValueError("smoke_media_list_must_not_be_empty")
    return values


def run_qwen_smoke(
    config: ProjectConfig,
    image: str | Path | Sequence[str | Path],
    *,
    repeats: int | None = None,
    gate: str = "none",
    command: Sequence[str] | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Issue stateless requests and evaluate allocator stability.

    ``development`` requires 20 requests and ``delivery`` requires 100.  The
    delivery contract uses allocator metrics from the isolated worker, excludes
    requests 1--10 from growth estimates, and records WDDM whole-device usage
    only as a warning.
    """

    required_repeats = {"none": 1, "development": 20, "delivery": 100}.get(gate)
    if required_repeats is None:
        raise ValueError("smoke_gate_must_be_none_development_or_delivery")
    if repeats is None:
        repeats = required_repeats
    if repeats < 1:
        raise ValueError("smoke_repeats_must_be_positive")
    smoke_inputs = [
        _smoke_image(config, media)
        for media in _smoke_media_list(image)
    ]
    image_path, image_source = smoke_inputs[0]
    model_path = config.model_path("qwen.model_dir")
    if model_path is None or not model_path.is_dir():
        raise FileNotFoundError(f"qwen_model_not_found:{model_path}")
    try:
        import psutil

        psutil_process = psutil.Process()
        before_available = psutil.virtual_memory().available / 2**30
        before_rss = psutil_process.memory_info().rss / 2**20
    except Exception:
        psutil = None
        psutil_process = None
        before_available = before_rss = None
    import torch

    package_versions: dict[str, str | None] = {}
    for distribution in ("torch", "transformers", "bitsandbytes", "Pillow", "opencv-python"):
        try:
            package_versions[distribution] = version(distribution)
        except PackageNotFoundError:
            package_versions[distribution] = None
    driver_version: str | None = None
    try:
        driver_probe = subprocess.run(
            ["nvidia-smi", f"--id={int(config.get('runtime.gpu_index', 0))}", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        driver_version = driver_probe.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass

    code_digest = python_source_digest()
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_not_available")
    gpu_index = int(config.get("runtime.gpu_index", 0))
    total_mib = _physical_gpu_total_mib(gpu_index, torch)
    physical_measurement_scope = _physical_gpu_measurement_scope()
    physical_headroom_warning_only = physical_measurement_scope == "device_total_wddm_warning_only"
    adapter: Any = None
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    progress_path = None
    if checkpoint_path is not None:
        destination = Path(checkpoint_path).resolve()
        progress_path = destination.with_name(destination.stem + ".progress.json")

    def write_progress(*, status: str, worker_pid_value: int | None = None) -> None:
        if progress_path is None:
            return
        atomic_write_json(progress_path, {
            "schema_version": "classroom_hybrid_ai.qwen_smoke.progress.v1",
            "status": status,
            "gate": gate,
            "required_repeats": required_repeats,
            "completed_requests": len(samples),
            "valid_responses": sum(row.get("status") == "valid" for row in samples),
            "oom_count": sum(bool(row.get("oom")) for row in samples),
            "worker_pid": worker_pid_value,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "inputs": [str(path) for path, _source in smoke_inputs],
            "samples": samples,
        })

    write_progress(status="starting")
    lease_path = config.resolve(".runtime/classroom_hybrid_ai/gpu.lock")
    assert lease_path is not None
    teardown: dict[str, Any] = {}
    worker_pid: int | None = None
    worker_exitcode: int | None = None
    warmup_error: str | None = None
    try:
        with GpuLease(lease_path):
            adapter = _qwen_adapter_for_smoke(config, model_path)
            try:
                warmup = getattr(adapter, "warmup", None)
                if callable(warmup):
                    warmup()
            except Exception as exc:
                warmup_error = f"{type(exc).__name__}:{exc}"[:2000]
            worker_pid = getattr(adapter, "pid", None)
            smoke_process = psutil.Process if psutil is not None else None
            prompt = (
                "Inspect only the supplied 480x416 classroom evidence image. The candidate action is "
                "action_raise_hand: a student visibly raises one or both hands away from the desk to at least "
                "shoulder height. Confirm only this candidate; reject it when a different action is visible; use "
                "uncertain when the student or hands cannot be judged. Return exactly one compact JSON object and no Markdown "
                "with only these fields: decision, behavior, rationale, evidence_frame_indices, observability. "
                "decision must be confirmed, rejected, or uncertain; observability must be clear, partial, or none. "
                "Use behavior=action_raise_hand only when confirmed; for rejected or uncertain use behavior=null. "
                "Use evidence_frame_indices=[0] only when the image visibly supports the decision, otherwise []. "
                "Keep rationale under 8 Chinese characters."
            )
            for iteration in range(repeats):
                iteration_started = time.perf_counter()
                request_image_path, request_image_source = smoke_inputs[iteration % len(smoke_inputs)]
                try:
                    if warmup_error is not None:
                        raise RuntimeError(f"qwen_warmup_failed:{warmup_error}")
                    response = adapter.review(
                        f"qwen35-9b-smoke-{iteration:03d}",
                        {"images": [str(request_image_path)], "purpose": "hardware_smoke_only"},
                        event_type="observable_action",
                        evidence_frame_indices=(0,),
                        prompt=prompt,
                        compact_response=True,
                    )
                    semantic_ok, semantic_error = _smoke_response_contract(response)
                    if not semantic_ok:
                        raise ValueError(f"qwen_smoke_semantic_contract_failed:{semantic_error}")
                    result: dict[str, Any] = {
                        "status": "valid",
                        "response": response,
                        "semantic_contract_ok": True,
                        "oom": False,
                    }
                except Exception as exc:
                    error = f"{type(exc).__name__}:{exc}"
                    result = {
                        "status": "error",
                        "error": error[:1000],
                        "oom": "out of memory" in error.lower() or "cuda oom" in error.lower(),
                    }
                metrics = _allocator_snapshot(adapter, torch)
                child_rss = _process_rss_mib(smoke_process, worker_pid)
                samples.append({
                    "iteration": iteration + 1,
                    "input_index": iteration % len(smoke_inputs),
                    "input_path": str(request_image_path),
                    "input_source": request_image_source,
                    "input_sha256": sha256_file(request_image_path),
                    "elapsed_seconds": round(time.perf_counter() - iteration_started, 3),
                    "allocated_bytes": metrics.get("allocated_bytes"),
                    "reserved_bytes": metrics.get("reserved_bytes"),
                    "inactive_split_bytes": metrics.get("inactive_split_bytes"),
                    "allocated_mib": None if metrics.get("allocated_bytes") is None else round(float(metrics["allocated_bytes"]) / 2**20, 3),
                    "reserved_mib": None if metrics.get("reserved_bytes") is None else round(float(metrics["reserved_bytes"]) / 2**20, 3),
                    "inactive_split_mib": None if metrics.get("inactive_split_bytes") is None else round(float(metrics["inactive_split_bytes"]) / 2**20, 3),
                    "num_alloc_retries": metrics.get("num_alloc_retries"),
                    "num_ooms": metrics.get("num_ooms"),
                    "allocator_measurement_scope": "qwen_worker_process" if isinstance(adapter, QwenProcessClient) else "current_process",
                    "physical_gpu_used_mib": _physical_gpu_used_mib(gpu_index),
                    "physical_gpu_measurement_scope": physical_measurement_scope,
                    "physical_gpu_process_used_mib": _physical_gpu_process_used_mib(
                        gpu_index, worker_pid if isinstance(adapter, QwenProcessClient) else None
                    ),
                    "physical_gpu_process_measurement_scope": "nvidia_smi_compute_app_pid_or_unavailable",
                    "process_rss_mib": child_rss if isinstance(adapter, QwenProcessClient) else _process_rss_mib(psutil_process, None),
                    **result,
                })
                write_progress(status="running", worker_pid_value=worker_pid)
    finally:
        teardown = release_model(adapter)
        worker_exitcode = getattr(adapter, "exitcode", None) if adapter is not None else None
    try:
        after_available = psutil.virtual_memory().available / 2**30 if psutil is not None else None
        after_rss = psutil_process.memory_info().rss / 2**20 if psutil_process is not None else None
    except Exception:
        after_available = after_rss = None

    allocated = [float(row["allocated_mib"]) for row in samples if row.get("allocated_mib") is not None]
    reserved = [float(row["reserved_mib"]) for row in samples if row.get("reserved_mib") is not None]
    physical = [row.get("physical_gpu_used_mib") for row in samples]
    physical_values = [float(value) for value in physical if value is not None]
    physical_process = [row.get("physical_gpu_process_used_mib") for row in samples]
    physical_process_values = [float(value) for value in physical_process if value is not None]
    rss_values = [float(row["process_rss_mib"]) for row in samples if row.get("process_rss_mib") is not None]
    allocated_growth = _growth_metrics(allocated)
    reserved_growth = _growth_metrics(reserved)
    physical_growth = (
        max(float(value) for value in physical[-10:] if value is not None)
        - min(float(value) for value in physical[:10] if value is not None)
        if len(samples) >= 20 and all(value is not None for value in physical[:10] + physical[-10:])
        else None
    )
    physical_process_growth = (
        max(float(value) for value in physical_process[-10:] if value is not None)
        - min(float(value) for value in physical_process[:10] if value is not None)
        if len(samples) >= 20 and all(value is not None for value in physical_process[:10] + physical_process[-10:])
        else None
    )
    rss_growth = (
        _median_window(rss_values, 80, 100) - _median_window(rss_values, 10, 30)
        if _median_window(rss_values, 10, 30) is not None and _median_window(rss_values, 80, 100) is not None
        else None
    )
    physical_limit_mib = None if total_mib is None else total_mib - 512.0
    blockers: list[str] = []
    warnings: list[str] = []
    if repeats < required_repeats:
        blockers.append(f"requires_{required_repeats}_repeats")
    if warmup_error is not None:
        blockers.append("qwen_warmup_failed")
    if any(row.get("oom") for row in samples):
        blockers.append("oom_observed")
    if any(row.get("status") != "valid" for row in samples):
        blockers.append("strict_json_or_process_contract_failed")
    if not allocated or not reserved:
        blockers.append("allocator_memory_measurement_unavailable")
    if gate == "delivery" and (len(allocated) != repeats or len(reserved) != repeats):
        blockers.append("allocator_memory_samples_incomplete")
    if gate == "delivery":
        if repeats != required_repeats:
            blockers.append("delivery_requires_exactly_100_requests")
        if len(samples) < 100 or sum(row.get("status") == "valid" for row in samples) != repeats:
            blockers.append("delivery_requires_100_strict_json_responses")
        for name, metrics, threshold in (
            ("allocated", allocated_growth, 64.0),
            ("reserved", reserved_growth, 128.0),
        ):
            growth = metrics["growth_11_30_to_81_100_mib"]
            if growth is None:
                blockers.append(f"{name}_steady_window_unavailable")
            elif float(growth) > threshold:
                blockers.append(f"{name}_growth_exceeds_{int(threshold)}_mib")
            slope = metrics["robust_slope_mib_per_request"]
            if slope is None:
                blockers.append(f"{name}_slope_unavailable")
            elif float(slope) > 1.0:
                blockers.append(f"{name}_robust_slope_exceeds_1_mib_per_request")
            if metrics["three_consecutive_windows_rising"]:
                blockers.append(f"{name}_three_windows_rising")
    if isinstance(adapter, QwenProcessClient) and gate == "delivery" and worker_exitcode != 0:
        blockers.append("qwen_worker_did_not_exit_cleanly")
    if gate == "delivery" and physical_process_growth is not None and physical_process_growth > 256.0:
        blockers.append("qwen_worker_physical_growth_exceeds_256_mib")
    if not physical_values:
        warnings.append("physical_gpu_memory_measurement_unavailable_wddm")
    else:
        if physical_limit_mib is not None and max(physical_values) > physical_limit_mib:
            if physical_headroom_warning_only:
                warnings.append("physical_gpu_peak_headroom_below_512_mib_wddm")
            else:
                blockers.append("physical_gpu_peak_headroom_below_512_mib")
        if physical_growth is not None and physical_growth > 256.0:
            warnings.append("physical_gpu_total_growth_over_256_mib_wddm")
    if isinstance(adapter, QwenProcessClient) and not physical_process_values:
        warnings.append("physical_gpu_process_memory_unavailable_wddm")
    status = "ok" if not blockers else (
        "incomplete" if blockers == [f"requires_{required_repeats}_repeats"] else "failed"
    )
    peak_allocated = max(allocated, default=0.0) / 1024.0
    peak_reserved = max(reserved, default=0.0) / 1024.0
    receipt = {
        "schema_version": "classroom_hybrid_ai.qwen_smoke.v3",
        "status": status,
        "model_path": str(model_path),
        "quantization": "bitsandbytes_nf4",
        "batch_size": 1,
        "stateless_messages": repeats,
        "worker_isolated": isinstance(adapter, QwenProcessClient),
        "worker_pid": worker_pid,
        "worker_exitcode": worker_exitcode,
        "image": str(image_path),
        "image_source": image_source,
        "input_sha256": sha256_file(image_path),
        "input_bytes": image_path.stat().st_size,
        "inputs": [
            {
                "path": str(path),
                "source": source,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "shape": [480, 416],
            }
            for path, source in smoke_inputs
        ],
        "input_rotation": "round_robin",
        "response_contract": "compact_classification_transport",
        "command": list(command) if command is not None else None,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions,
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": driver_version,
        "config_digest": project_config_digest(config),
        "code_digest": code_digest,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_torch_allocated_gib": round(peak_allocated, 3),
        "peak_torch_reserved_gib": round(peak_reserved, 3),
        "physical_gpu_peak_mib": None if not physical_values else round(max(physical_values), 3),
        "physical_gpu_measurement_scope": physical_measurement_scope,
        "physical_gpu_headroom_gate": "warning" if physical_headroom_warning_only else "hard",
        "physical_gpu_limit_mib": physical_limit_mib,
        "physical_gpu_peak_headroom_mib": None if total_mib is None or not physical_values else round(total_mib - max(physical_values), 3),
        "physical_gpu_process_peak_mib": None if not physical_process_values else round(max(physical_process_values), 3),
        "physical_gpu_process_growth_first10_to_last10_mib": None if physical_process_growth is None else round(physical_process_growth, 3),
        "reserved_growth_first10_to_last10_mib": None,
        "physical_gpu_growth_first10_to_last10_mib": None if physical_growth is None else round(physical_growth, 3),
        "process_rss_growth_first10_to_last10_mib": None if rss_growth is None else round(rss_growth, 3),
        "process_rss_growth_limit_mib": 512.0,
        "process_rss_total_lifecycle_growth_mib": None if before_rss is None or after_rss is None else round(after_rss - before_rss, 3),
        "allocator_metrics": {
            "allocated": allocated_growth,
            "reserved": reserved_growth,
            "thresholds": {
                "allocated_growth_mib": 64.0,
                "reserved_growth_mib": 128.0,
                "robust_slope_mib_per_request": 1.0,
                "warmup_requests_excluded": 10,
            },
        },
        "system_available_gib_before": None if before_available is None else round(before_available, 3),
        "system_available_gib_after": None if after_available is None else round(after_available, 3),
        "gate": gate,
        "required_repeats": required_repeats,
        "valid_responses": sum(row.get("status") == "valid" for row in samples),
        "strict_json_success_rate": round(
            sum(row.get("status") == "valid" for row in samples) / repeats, 6
        ) if repeats else 0.0,
        "gray_responses": sum(row.get("status") != "valid" for row in samples),
        "oom_count": sum(bool(row.get("oom")) for row in samples),
        "warnings": warnings,
        "blockers": blockers,
        "samples": samples,
        "teardown": teardown,
    }
    write_progress(status="complete", worker_pid_value=worker_pid)
    return receipt


__all__ = [
    "run_qwen_smoke",
    "_growth_metrics",
    "_physical_gpu_measurement_scope",
    "_physical_gpu_used_mib",
    "_smoke_media_list",
    "_smoke_response_contract",
]
