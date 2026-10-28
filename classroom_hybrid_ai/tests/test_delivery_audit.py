from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_delivery_receipt.py"
SPEC = importlib.util.spec_from_file_location("delivery_audit_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
delivery_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery_audit)


def _valid_receipt() -> dict[str, object]:
    config = delivery_audit.load_config(ROOT / "configs" / "default.yaml")
    samples = [
        {
            "request_index": index,
            "status": "valid",
            "semantic_contract_ok": True,
            "oom": False,
        }
        for index in range(1, 101)
    ]
    stable_allocator = {
        "growth_11_30_to_81_100_mib": 0.0,
        "robust_slope_mib_per_request": 0.0,
        "three_consecutive_windows_rising": False,
    }
    return {
        "schema_version": "classroom_hybrid_ai.qwen_smoke.v3",
        "status": "ok",
        "gate": "delivery",
        "stateless_messages": 100,
        "required_repeats": 100,
        "samples": samples,
        "strict_json_success_rate": 1.0,
        "worker_isolated": True,
        "worker_exitcode": 0,
        "oom_count": 0,
        "physical_gpu_peak_headroom_mib": 512.0,
        "physical_gpu_measurement_scope": "device_total_nvml_reliable",
        "platform": "Linux-test",
        "allocator_metrics": {
            "allocated": dict(stable_allocator),
            "reserved": dict(stable_allocator),
        },
        "teardown": {"close_error": None},
        "blockers": [],
        "code_digest": delivery_audit.current_code_digest(),
        "config_digest": delivery_audit.project_config_digest(config),
        "python_executable": str(config.python_executable),
        "model_path": str(config.resolve(config.get("qwen.model_dir"))),
    }


def test_delivery_audit_separates_metrics_from_current_source_provenance(tmp_path: Path) -> None:
    receipt = _valid_receipt()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    report = delivery_audit.audit_receipt(
        ROOT / "configs" / "default.yaml",
        receipt_path,
    )
    assert report["delivery_metrics"]["passed"] is True
    assert report["provenance"]["passed"] is True
    assert report["status"] == "stability_passed_action_quality_pending"


def test_delivery_audit_treats_only_wddm_device_headroom_as_warning() -> None:
    receipt = _valid_receipt()
    receipt["physical_gpu_peak_headroom_mib"] = 100.0
    receipt["physical_gpu_measurement_scope"] = "device_total_wddm_warning_only"
    receipt["platform"] = "Windows-11-10.0.26200-SP0"
    assert delivery_audit._metric_contract(receipt)["checks"]["physical_headroom"] is True

    receipt["physical_gpu_measurement_scope"] = "device_total_nvml_reliable"
    receipt["platform"] = "Linux-6.8"
    assert delivery_audit._metric_contract(receipt)["checks"]["physical_headroom"] is False


def test_delivery_audit_cli_requires_explicit_receipt() -> None:
    with pytest.raises(SystemExit) as exc_info:
        delivery_audit._parser().parse_args([])
    assert exc_info.value.code == 2
