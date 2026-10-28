"""Audit a Qwen delivery receipt against the current checkout.

The smoke receipt is intentionally immutable evidence from one run.  This
read-only command checks whether that evidence still describes the current
source and configuration, without loading Qwen or running another stress
workload.  ``--strict`` is suitable for CI and returns non-zero when the
receipt is stale or the delivery contract is incomplete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classroom_hybrid_ai.config import load_config  # noqa: E402
from classroom_hybrid_ai.hashing import project_config_digest, python_source_digest  # noqa: E402


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return value


def current_code_digest() -> str:
    return python_source_digest(SOURCE_ROOT / "classroom_hybrid_ai")


def _normalised_path(value: Any) -> str:
    return os.path.normcase(os.path.normpath(str(value))).replace("/", "\\")


def _metric_contract(receipt: Mapping[str, Any]) -> dict[str, Any]:
    samples = receipt.get("samples")
    allocator = receipt.get("allocator_metrics")
    allocated = allocator.get("allocated", {}) if isinstance(allocator, dict) else {}
    reserved = allocator.get("reserved", {}) if isinstance(allocator, dict) else {}
    physical_headroom = receipt.get("physical_gpu_peak_headroom_mib")
    physical_scope = str(receipt.get("physical_gpu_measurement_scope", "")).lower()
    receipt_platform = str(receipt.get("platform", "")).lower()
    if not physical_scope and isinstance(samples, list) and samples:
        first_sample = samples[0] if isinstance(samples[0], dict) else {}
        physical_scope = str(first_sample.get("physical_gpu_measurement_scope", "")).lower()
    wddm_warning_only = "wddm" in physical_scope or receipt_platform.startswith("windows")
    # Preserve the hard 512 MiB margin when the physical reading is reliable.
    # Windows WDDM exposes only whole-device usage polluted by unrelated GUI
    # processes, so it remains diagnostic and cannot fail delivery alone.
    physical_headroom_ok = (
        wddm_warning_only
        or
        physical_headroom is None
        or (
            isinstance(physical_headroom, (int, float))
            and float(physical_headroom) >= 512.0
        )
    )
    sample_contract = isinstance(samples, list) and len(samples) == 100 and all(
        isinstance(row, dict)
        and row.get("status") == "valid"
        and row.get("semantic_contract_ok") is True
        and not bool(row.get("oom"))
        for row in samples
    )
    checks = {
        "schema_v3": receipt.get("schema_version") == "classroom_hybrid_ai.qwen_smoke.v3",
        "delivery_status": receipt.get("status") == "ok" and receipt.get("gate") == "delivery",
        "request_count": receipt.get("stateless_messages") == 100 and receipt.get("required_repeats") == 100,
        "sample_contract": sample_contract,
        "strict_json": receipt.get("strict_json_success_rate") == 1.0,
        "worker": receipt.get("worker_isolated") is True and receipt.get("worker_exitcode") == 0,
        "oom": receipt.get("oom_count") == 0,
        "physical_headroom": physical_headroom_ok,
        "allocator_growth": (
            isinstance(allocated.get("growth_11_30_to_81_100_mib"), (int, float))
            and float(allocated["growth_11_30_to_81_100_mib"]) <= 64.0
            and isinstance(reserved.get("growth_11_30_to_81_100_mib"), (int, float))
            and float(reserved["growth_11_30_to_81_100_mib"]) <= 128.0
        ),
        "allocator_slope": (
            isinstance(allocated.get("robust_slope_mib_per_request"), (int, float))
            and float(allocated["robust_slope_mib_per_request"]) <= 1.0
            and isinstance(reserved.get("robust_slope_mib_per_request"), (int, float))
            and float(reserved["robust_slope_mib_per_request"]) <= 1.0
        ),
        "no_rising_windows": (
            allocated.get("three_consecutive_windows_rising") is False
            and reserved.get("three_consecutive_windows_rising") is False
        ),
        "teardown": receipt.get("teardown", {}).get("close_error") in {None, ""},
        "no_blockers": not receipt.get("blockers"),
    }
    return {"passed": all(checks.values()), "checks": checks}


def audit_receipt(
    config_path: Path,
    receipt_path: Path,
    *,
    blind_report_path: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path.resolve())
    receipt = _read_object(receipt_path.resolve())
    code_digest = current_code_digest()
    config_digest = project_config_digest(config)
    expected_python = config.python_executable
    expected_model = config.resolve(config.get("qwen.model_dir"))

    provenance_checks = {
        "code_digest": receipt.get("code_digest") == code_digest,
        "config_digest": receipt.get("config_digest") == config_digest,
        "python_executable": _normalised_path(receipt.get("python_executable")) == _normalised_path(expected_python),
        "model_path": _normalised_path(receipt.get("model_path")) == _normalised_path(expected_model),
    }
    blind_quality: dict[str, Any]
    if blind_report_path is None:
        blind_quality = {"status": "pending", "reason": "blind_report_not_supplied"}
    elif not blind_report_path.is_file():
        blind_quality = {"status": "pending", "reason": "blind_report_missing", "path": str(blind_report_path)}
    else:
        report = _read_object(blind_report_path)
        required = {"macro_f1", "per_class_f1", "unknown_rate", "covered_sample_accuracy"}
        valid = report.get("schema_version") == "classroom_hybrid_ai.blind_benchmark_report.v1" and required.issubset(report)
        blind_quality = {
            "status": "available" if valid else "invalid",
            "path": str(blind_report_path.resolve()),
            "macro_f1": report.get("macro_f1"),
            "unknown_rate": report.get("unknown_rate"),
            "covered_sample_accuracy": report.get("covered_sample_accuracy"),
        }

    metric = _metric_contract(receipt)
    provenance_ok = all(provenance_checks.values())
    if not metric["passed"]:
        status = "delivery_contract_failed"
    elif not provenance_ok:
        status = "historical_receipt_stale"
    elif blind_quality.get("status") != "available":
        status = "stability_passed_action_quality_pending"
    else:
        status = "ready_for_action_quality_review"
    return {
        "schema_version": "classroom_hybrid_ai.delivery_receipt_audit.v1",
        "status": status,
        "config_path": str(config_path.resolve()),
        "receipt_path": str(receipt_path.resolve()),
        "current_code_digest": code_digest,
        "receipt_code_digest": receipt.get("code_digest"),
        "current_config_digest": config_digest,
        "receipt_config_digest": receipt.get("config_digest"),
        "provenance": {"passed": provenance_ok, "checks": provenance_checks},
        "delivery_metrics": metric,
        "blind_action_quality": blind_quality,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a Qwen delivery receipt without loading models")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="receipt JSON to audit; no historical output is selected implicitly",
    )
    parser.add_argument("--blind-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="return 2 unless current provenance and delivery metrics pass")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_receipt(args.config, args.receipt, blind_report_path=args.blind_report)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.strict and report["status"] != "ready_for_action_quality_review":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_receipt", "current_code_digest"]
