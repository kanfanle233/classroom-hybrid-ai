from __future__ import annotations

import copy
from pathlib import Path
from importlib import resources

import pytest
import yaml

from classroom_hybrid_ai.config import ConfigError, load_config
from classroom_hybrid_ai.hashing import project_config_digest, project_config_payload


PROJECT = Path(__file__).resolve().parents[1]


def test_default_config_is_stateless_single_gpu() -> None:
    config = load_config(PROJECT / "configs" / "default.yaml")
    assert config.python_executable == Path("E:/miniconda/envs/py310/python.exe")
    assert config.get("qwen.model_size") == "9b"
    assert config.get("qwen.quantization") == "bnb4"
    assert config.get("qwen.batch_size") == 1
    assert config.get("qwen.stateless_requests") is True
    assert config.get("qwen.process_isolation") is True
    # Idle cached blocks are returned after every request, and large blocks are
    # never split, so the measured 7.4 GiB device budget holds without relying
    # on Windows-only paging.
    assert config.get("qwen.release_cuda_cache") is True
    assert config.get("qwen.cuda_alloc_conf") == "max_split_size_mb:128"
    # The enforced 8 GiB bound: a hard allocator fraction, not an advisory
    # byte cap that an explicit device_map cannot honour.
    assert config.get("qwen.gpu_memory_fraction") == 0.80
    assert config.get("runtime.execution_layout") == "sequential_offline_passes"
    assert config.section("qwen_candidates") == {
        "minimum_event_macro_f1_gain_pp": 3.0,
        "maximum_any_class_f1_drop_pp": 2.0,
    }
    assert config.section("research") == {
        "profile": "core8_v1",
        "mode": "production_adaptive",
        "strict": True,
        "random_seed": 0,
    }


def test_invalid_history_policy_fails(tmp_path: Path) -> None:
    payload = yaml.safe_load((PROJECT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    payload["qwen"]["stateless_requests"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="stateless_requests"):
        load_config(path)


def test_packaged_style_config_can_resolve_external_workspace_from_cwd(tmp_path: Path, monkeypatch) -> None:
    payload = yaml.safe_load((PROJECT / "src" / "classroom_hybrid_ai" / "configs" / "default.yaml").read_text(encoding="utf-8"))
    config_path = tmp_path / "configs" / "default.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    monkeypatch.chdir(PROJECT.parent)
    config = load_config(config_path)
    assert config.repository_root == PROJECT.parent.resolve()


def test_source_and_packaged_default_yaml_differ_only_in_repository_root() -> None:
    source_payload = yaml.safe_load(
        (PROJECT / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    packaged_payload = yaml.safe_load(
        (PROJECT / "src" / "classroom_hybrid_ai" / "configs" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    source = copy.deepcopy(source_payload)
    packaged = copy.deepcopy(packaged_payload)
    assert source["paths"]["repository_root"] != packaged["paths"]["repository_root"]
    source["paths"]["repository_root"] = "<normalized_repository_root>"
    packaged["paths"]["repository_root"] = "<normalized_repository_root>"
    assert packaged == source


def test_packaged_config_resources_include_preflight_manifests() -> None:
    package_configs = resources.files("classroom_hybrid_ai").joinpath("configs")
    for name in ("default.yaml", "models.local.yaml", "behaviors.yaml"):
        assert package_configs.joinpath(name).is_file()


def test_source_and_packaged_configs_have_identical_effective_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_config(PROJECT / "configs" / "default.yaml")
    monkeypatch.chdir(source.repository_root)
    packaged = load_config(
        PROJECT / "src" / "classroom_hybrid_ai" / "configs" / "default.yaml"
    )
    assert project_config_payload(source) == project_config_payload(packaged)
    assert project_config_digest(source) == project_config_digest(packaged)

    for name in (
        "action_profiles.yaml",
        "behaviors.yaml",
        "models.local.yaml",
    ):
        source_payload = yaml.safe_load((PROJECT / "configs" / name).read_text(encoding="utf-8"))
        packaged_payload = yaml.safe_load(
            (PROJECT / "src" / "classroom_hybrid_ai" / "configs" / name).read_text(
                encoding="utf-8"
            )
        )
        assert source_payload == packaged_payload, name
