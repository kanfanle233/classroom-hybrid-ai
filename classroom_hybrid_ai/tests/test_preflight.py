from __future__ import annotations

import inspect
import sys
from pathlib import Path

from classroom_hybrid_ai.config import ProjectConfig
from classroom_hybrid_ai.preflight import run_preflight


def _minimal_config(tmp_path: Path) -> ProjectConfig:
    project_root = tmp_path / "project"
    repository_root = tmp_path / "repository"
    configs = project_root / "configs"
    dataset = repository_root / "data"
    registry = repository_root / ".runtime" / "action_registry"
    configs.mkdir(parents=True)
    dataset.mkdir(parents=True)
    registry.mkdir(parents=True)
    (configs / "models.local.yaml").write_text("models: {}\n", encoding="utf-8")
    (configs / "behaviors.yaml").write_text("actions: []\n", encoding="utf-8")
    source = configs / "default.yaml"
    source.write_text("schema_version: classroom_hybrid_ai.config.v1\n", encoding="utf-8")
    payload = {
        "runtime": {"python_executable": sys.executable, "gpu_index": 0},
        "paths": {
            "dataset_root": str(dataset),
            "model_manifest": "configs/models.local.yaml",
            "behavior_taxonomy": "configs/behaviors.yaml",
            "action_registry_root": str(registry),
        },
        "pipeline": {"primary_detector": "locate_anything"},
        "locate_anything": {"model_dir": None},
        "qwen": {
            "enabled": True,
            "model_dir": None,
            "action_mode": "fixed_eight",
        },
        "open_vocabulary": {"embedding_model_dir": None},
        "yolo": {"tracker_model": None, "anomaly_model": None},
    }
    return ProjectConfig(source, project_root, repository_root, payload)


def test_run_preflight_signature_has_no_receipt_gate_switch() -> None:
    signature = inspect.signature(run_preflight)

    assert tuple(signature.parameters) == ("config", "verify_hashes")
    assert signature.parameters["verify_hashes"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["verify_hashes"].default is False


def test_preflight_does_not_depend_on_qwen_receipt(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)

    result = run_preflight(config)

    check_names = {check["name"] for check in result["checks"]}
    assert "qwen_selected_stability_gate" not in check_names
    assert all("delivery_receipt" not in check["detail"] for check in result["checks"])
