from pathlib import Path
from typing import List, Tuple

from classroom_pipeline.contracts.schemas import (
    validate_align_file,
    validate_event_query_record,
    validate_json_file,
    validate_jsonl_file,
    validate_pipeline_manifest,
    validate_pose_uq_record,
    validate_verifier_calibration_report,
    validate_verifier_eval_report,
    validate_verifier_sample_record,
    validate_verified_event_record,
)


def verify_event_queries(path: Path) -> Tuple[bool, int, List[str]]:
    return validate_jsonl_file(path, validate_event_query_record)


def verify_pose_uq(path: Path) -> Tuple[bool, int, List[str]]:
    return validate_jsonl_file(path, validate_pose_uq_record)


def verify_aligned(path: Path) -> Tuple[bool, List[str]]:
    return validate_json_file(path, validate_align_file)


def verify_verifier_samples(path: Path) -> Tuple[bool, int, List[str]]:
    return validate_jsonl_file(path, validate_verifier_sample_record)


def verify_verified_events(path: Path) -> Tuple[bool, int, List[str]]:
    return validate_jsonl_file(path, validate_verified_event_record)


def verify_eval_report(path: Path) -> Tuple[bool, List[str]]:
    return validate_json_file(path, validate_verifier_eval_report)


def verify_calibration_report(path: Path) -> Tuple[bool, List[str]]:
    return validate_json_file(path, validate_verifier_calibration_report)


def verify_pipeline_manifest(path: Path) -> Tuple[bool, List[str]]:
    return validate_json_file(path, validate_pipeline_manifest)
