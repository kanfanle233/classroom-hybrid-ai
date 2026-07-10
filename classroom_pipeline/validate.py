from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts.schemas import (
    validate_event_query_record,
    validate_json_file,
    validate_jsonl_file,
    validate_pipeline_manifest,
    validate_verified_event_record,
    validate_verified_event_v3_record,
)


REQUIRED_CASE_FILES = (
    "pipeline_manifest.json",
    "actions.fusion_v2.jsonl",
    "event_queries.fusion_v2.jsonl",
    "verified_events.jsonl",
)


def _validate_object_jsonl(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path.name}:{line_number}: row must be a JSON object")
    return errors


def validate_case(case_dir: Path) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    if not case_dir.exists():
        errors.append(f"case directory not found: {case_dir}")
    else:
        for name in REQUIRED_CASE_FILES:
            path = case_dir / name
            if not path.exists():
                errors.append(f"missing required artifact: {name}")
            elif path.is_file() and path.stat().st_size == 0:
                errors.append(f"empty artifact: {name}")

        manifest_path = case_dir / "pipeline_manifest.json"
        if manifest_path.exists() and manifest_path.stat().st_size > 0:
            ok, contract_errors = validate_json_file(manifest_path, validate_pipeline_manifest)
            if not ok:
                errors.extend(contract_errors)

        actions_path = case_dir / "actions.fusion_v2.jsonl"
        if actions_path.exists() and actions_path.stat().st_size > 0:
            errors.extend(_validate_object_jsonl(actions_path))

        query_path = case_dir / "event_queries.fusion_v2.jsonl"
        if query_path.exists() and query_path.stat().st_size > 0:
            ok, _, contract_errors = validate_jsonl_file(query_path, validate_event_query_record)
            if not ok:
                errors.extend(contract_errors)

        verified_path = case_dir / "verified_events.jsonl"
        if verified_path.exists() and verified_path.stat().st_size > 0:
            ok, _, contract_errors = validate_jsonl_file(verified_path, validate_verified_event_record)
            if not ok:
                errors.extend(contract_errors)

        coherence_path = case_dir / "verified_events.coherence_v3.jsonl"
        if coherence_path.exists():
            if coherence_path.stat().st_size == 0:
                errors.append(f"empty artifact: {coherence_path.name}")
            else:
                ok, _, contract_errors = validate_jsonl_file(coherence_path, validate_verified_event_v3_record)
                if not ok:
                    errors.extend(contract_errors)
        else:
            warnings.append("missing optional artifact: verified_events.coherence_v3.jsonl")

    payload = {
        "status": "ok" if not errors else "failed",
        "case_dir": str(case_dir),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a completed pipeline case directory.")
    parser.add_argument("--case-dir", required=True)
    args = parser.parse_args(argv)
    return validate_case(Path(args.case_dir).resolve())
