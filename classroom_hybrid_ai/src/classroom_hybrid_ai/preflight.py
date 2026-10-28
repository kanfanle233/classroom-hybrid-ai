"""Fail-fast checks for the local-only pipeline."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

from .config import ProjectConfig
from .hashing import sha256_file
from .registry import audit_registry
from .research_policy import localization_route


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: str
    detail: str


def _check_path(name: str, path: Path | None, *, required: bool = True) -> Check:
    if path is None:
        return Check(name, "failed" if required else "skipped", "not_configured")
    exists = path.is_file() or path.is_dir()
    return Check(name, "ok" if exists else ("failed" if required else "warning"), str(path))


def _load_model_manifest(config: ProjectConfig) -> dict[str, Any]:
    with config.model_manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError("model_manifest_invalid")
    return payload


def _check_transformers_snapshot(
    name: str, path: Path | None, *, minimum_weight_bytes: int = 100_000_000
) -> list[Check]:
    """Reject half-downloaded directories that happen to exist."""

    if path is None or not path.is_dir():
        return []
    required = ["config.json", "tokenizer_config.json"]
    checks = [
        Check(
            f"{name}:{filename}",
            "ok" if (path / filename).is_file() else "failed",
            str(path / filename),
        )
        for filename in required
    ]
    weights = list(path.glob("*.safetensors"))
    weight_bytes = sum(item.stat().st_size for item in weights)
    incomplete = list(path.rglob("*.incomplete"))
    checks.append(Check(
        f"{name}:weights",
        "ok" if weights and weight_bytes >= minimum_weight_bytes else "failed",
        f"files={len(weights)} bytes={weight_bytes}",
    ))
    checks.append(Check(
        f"{name}:no_incomplete_downloads",
        "ok" if not incomplete else "failed",
        "none" if not incomplete else ",".join(str(item) for item in incomplete[:5]),
    ))
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index.get("weight_map", {}).values())
            missing = sorted(shard for shard in shards if not (path / shard).is_file())
            checks.append(Check(
                f"{name}:weight_index",
                "ok" if shards and not missing else "failed",
                f"shards={len(shards)} missing={missing}",
            ))
        except Exception as exc:
            checks.append(Check(f"{name}:weight_index", "failed", f"{type(exc).__name__}:{exc}"))
    return checks


def run_preflight(config: ProjectConfig, *, verify_hashes: bool = False) -> dict[str, Any]:
    checks: list[Check] = []
    checks.append(_check_path("python_executable", config.python_executable))
    interpreter_matches = Path(sys.executable).resolve() == config.python_executable
    checks.append(Check(
        "active_python_matches_config",
        "ok" if interpreter_matches else "failed",
        f"active={Path(sys.executable).resolve()} expected={config.python_executable}",
    ))
    checks.append(_check_path("dataset_root", config.dataset_root))
    checks.append(_check_path("behavior_taxonomy", config.behavior_taxonomy_path))
    checks.append(_check_path("model_manifest", config.model_manifest_path))
    try:
        registry = audit_registry(config.action_registry_root)
        registry_status = str(registry.get("status"))
        registry_ok = registry_status in {"ok", "empty"} and not registry.get("errors")
        checks.append(Check(
            "action_registry",
            "ok" if registry_ok else "failed",
            json.dumps(registry, ensure_ascii=False, sort_keys=True),
        ))
    except Exception as exc:
        checks.append(Check("action_registry", "failed", f"{type(exc).__name__}:{exc}"))
    route = localization_route(config)
    yolo_required = route != "locate_only"
    checks.append(_check_path(
        "yolo_tracker_model",
        config.model_path("yolo.tracker_model"),
        required=yolo_required,
    ))
    locate_path = config.model_path("locate_anything.model_dir")
    qwen_path = config.model_path("qwen.model_dir")
    embedding_path = config.model_path("open_vocabulary.embedding_model_dir")
    checks.append(_check_path("locate_anything_model", locate_path, required=route != "yolo_only"))
    checks.append(_check_path("qwen35_9b_4bit_model", qwen_path))
    checks.append(_check_path(
        "bge_small_zh_v1_5_cpu_model",
        embedding_path,
        required=str(config.get("qwen.action_mode", "fixed_eight")) == "open_vocabulary",
    ))
    if route != "yolo_only":
        checks.extend(_check_transformers_snapshot("locate_anything_model", locate_path))
    checks.extend(_check_transformers_snapshot("qwen35_9b_4bit_model", qwen_path))
    checks.extend(_check_transformers_snapshot(
        "bge_small_zh_v1_5_cpu_model", embedding_path, minimum_weight_bytes=50_000_000,
    ))
    if qwen_path is not None and (qwen_path / "config.json").is_file():
        try:
            qwen_config = json.loads((qwen_path / "config.json").read_text(encoding="utf-8"))
            quantization = qwen_config.get("quantization_config", {})
            valid = (
                qwen_config.get("model_type") == "qwen3_5"
                and quantization.get("load_in_4bit") is True
                and quantization.get("quant_method") == "bitsandbytes"
            )
            checks.append(Check(
                "qwen35_9b_architecture_and_quantization",
                "ok" if valid else "failed",
                f"model_type={qwen_config.get('model_type')} quantization={quantization.get('quant_method')} load_in_4bit={quantization.get('load_in_4bit')}",
            ))
        except Exception as exc:
            checks.append(Check(
                "qwen35_9b_architecture_and_quantization",
                "failed",
                f"{type(exc).__name__}:{exc}",
            ))
    anomaly_required = yolo_required and bool(config.get("yolo.anomaly_model"))
    checks.append(_check_path(
        "yolo_classroom_anomaly_model",
        config.model_path("yolo.anomaly_model"),
        required=anomaly_required,
    ))

    package_versions: dict[str, str | None] = {}
    for distribution in (
        "torch",
        "torchvision",
        "transformers",
        "ultralytics",
        "bitsandbytes",
        "opencv-python",
        "Pillow",
        "PyYAML",
    ):
        try:
            from importlib.metadata import version

            package_versions[distribution] = version(distribution)
        except Exception:
            package_versions[distribution] = None
            checks.append(Check(f"package:{distribution}", "failed", "not_installed"))

    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            index = int(config.get("runtime.gpu_index", 0))
            properties = torch.cuda.get_device_properties(index)
            cuda.update({
                "device_name": properties.name,
                "total_memory_gib": round(properties.total_memory / 2**30, 3),
                "torch_cuda": torch.version.cuda,
            })
        checks.append(Check("cuda", "ok" if cuda["available"] else "failed", json.dumps(cuda)))
    except Exception as exc:
        checks.append(Check("cuda", "failed", f"{type(exc).__name__}:{exc}"))

    manifest: dict[str, Any] | None = None
    try:
        manifest = _load_model_manifest(config)
        checks.append(Check("model_manifest_parse", "ok", config.model_manifest_path.as_posix()))
    except Exception as exc:
        checks.append(Check("model_manifest_parse", "failed", f"{type(exc).__name__}:{exc}"))

    hash_receipts: list[dict[str, Any]] = []
    if verify_hashes and manifest:
        for model_id, record in manifest["models"].items():
            if not isinstance(record, dict) or not record.get("sha256"):
                continue
            path = config.resolve(record.get("weight_file", record.get("path")))
            if path is None or not path.is_file():
                continue
            actual = sha256_file(path)
            expected = str(record["sha256"]).lower()
            status = "ok" if actual == expected else "failed"
            checks.append(Check(f"sha256:{model_id}", status, f"expected={expected} actual={actual}"))
            hash_receipts.append({"model_id": model_id, "path": str(path), "sha256": actual})

    failed = [check.name for check in checks if check.status == "failed"]
    warnings = [check.name for check in checks if check.status == "warning"]
    return {
        "schema_version": "classroom_hybrid_ai.preflight.v1",
        "status": "failed" if failed else "ok",
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "packages": package_versions,
        "cuda": cuda,
        "checks": [asdict(check) for check in checks],
        "failed_checks": failed,
        "warnings": warnings,
        "hash_receipts": hash_receipts,
        "runtime_network_allowed": False,
        "gpu_execution": "sequential_only",
    }
