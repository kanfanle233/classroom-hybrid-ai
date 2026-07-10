import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def _report_jsonl(title: str, ok: bool, count: int, errors: List[str]) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {title}: {count} rows")
    if errors:
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... {len(errors) - 5} more")
    return ok


def _report_json(title: str, ok: bool, errors: List[str]) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {title}")
    if errors:
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... {len(errors) - 5} more")
    return ok


def _resolve(base: Path, value: str, default_rel: str = "") -> Path:
    if value:
        path = Path(value)
    else:
        path = base / default_rel if default_rel else Path("")
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _validate_examples(base_dir: Path, examples_dir: Path) -> Tuple[bool, List[str]]:
    checks: List[Tuple[str, Path, str]] = [
        ("event_queries(sample)", examples_dir / "event_queries.sample.jsonl", "jsonl_event"),
        ("pose_tracks_smooth_uq(sample)", examples_dir / "pose_tracks_smooth_uq.sample.jsonl", "jsonl_pose"),
        ("align_multimodal(sample)", examples_dir / "align_multimodal.sample.json", "json_align"),
        ("verifier_samples_train(sample)", examples_dir / "verifier_samples_train.sample.jsonl", "jsonl_samples"),
        ("verified_events(sample)", examples_dir / "verified_events.sample.jsonl", "jsonl_verified"),
        ("verifier_eval_report(sample)", examples_dir / "verifier_eval_report.sample.json", "json_eval"),
        ("verifier_calibration_report(sample)", examples_dir / "verifier_calibration_report.sample.json", "json_cal"),
        ("pipeline_manifest(sample)", examples_dir / "pipeline_manifest.sample.json", "json_manifest"),
    ]
    all_ok = True
    errors_flat: List[str] = []
    for title, path, kind in checks:
        if kind == "jsonl_event":
            ok, n, errors = validate_jsonl_file(path, validate_event_query_record)
            all_ok &= _report_jsonl(title, ok, n, errors)
        elif kind == "jsonl_pose":
            ok, n, errors = validate_jsonl_file(path, validate_pose_uq_record)
            all_ok &= _report_jsonl(title, ok, n, errors)
        elif kind == "json_align":
            ok, errors = validate_json_file(path, validate_align_file)
            all_ok &= _report_json(title, ok, errors)
        elif kind == "jsonl_samples":
            ok, n, errors = validate_jsonl_file(path, validate_verifier_sample_record)
            all_ok &= _report_jsonl(title, ok, n, errors)
        elif kind == "jsonl_verified":
            ok, n, errors = validate_jsonl_file(path, validate_verified_event_record)
            all_ok &= _report_jsonl(title, ok, n, errors)
        elif kind == "json_eval":
            ok, errors = validate_json_file(path, validate_verifier_eval_report)
            all_ok &= _report_json(title, ok, errors)
        elif kind == "json_cal":
            ok, errors = validate_json_file(path, validate_verifier_calibration_report)
            all_ok &= _report_json(title, ok, errors)
        else:
            ok, errors = validate_json_file(path, validate_pipeline_manifest)
            all_ok &= _report_json(title, ok, errors)
        if not ok:
            errors_flat.extend(errors)
    return all_ok, errors_flat


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Validate formal verifier contract artifacts.")
    parser.add_argument(
        "--case-dir",
        type=str,
        default="",
        help="Validate the standard artifact set in one pipeline case directory.",
    )
    parser.add_argument("--event_queries", type=str, default="")
    parser.add_argument("--pose_uq", type=str, default="")
    parser.add_argument("--align", type=str, default="")
    parser.add_argument("--verifier_samples", type=str, default="")
    parser.add_argument("--verified_events", type=str, default="")
    parser.add_argument("--eval_report", type=str, default="")
    parser.add_argument("--calibration_report", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--examples_dir", type=str, default="")
    args = parser.parse_args()

    explicit = {
        "event_queries": args.event_queries,
        "pose_uq": args.pose_uq,
        "align": args.align,
        "verifier_samples": args.verifier_samples,
        "verified_events": args.verified_events,
        "eval_report": args.eval_report,
        "calibration_report": args.calibration_report,
        "manifest": args.manifest,
    }
    case_dir = _resolve(base_dir, args.case_dir) if args.case_dir else None
    validate_all = case_dir is not None or not any(explicit.values())
    default_root = case_dir if case_dir is not None else base_dir / "output"
    default_names = {
        "event_queries": "event_queries.fusion_v2.jsonl" if case_dir else "event_queries.jsonl",
        "pose_uq": "pose_tracks_smooth_uq.jsonl",
        "align": "align_multimodal.json",
        "verifier_samples": "verifier_samples_train.jsonl",
        "verified_events": "verified_events.jsonl",
        "eval_report": "verifier_eval_report.json",
        "calibration_report": "verifier_calibration_report.json",
        "manifest": "pipeline_manifest.json",
    }

    def selected_path(name: str) -> Path | None:
        value = explicit[name]
        if value:
            return _resolve(base_dir, value)
        if validate_all:
            return (default_root / default_names[name]).resolve()
        return None

    all_ok = True
    jsonl_checks = [
        ("event_queries", "event_queries", validate_event_query_record),
        ("pose_tracks_smooth_uq", "pose_uq", validate_pose_uq_record),
        ("verifier_samples_train", "verifier_samples", validate_verifier_sample_record),
        ("verified_events", "verified_events", validate_verified_event_record),
    ]
    for title, name, validator in jsonl_checks:
        path = selected_path(name)
        if path is None:
            continue
        ok, n, errors = validate_jsonl_file(path, validator)
        all_ok &= _report_jsonl(title, ok, n, errors)

    json_checks = [
        ("align_multimodal", "align", validate_align_file),
        ("verifier_eval_report", "eval_report", validate_verifier_eval_report),
        ("verifier_calibration_report", "calibration_report", validate_verifier_calibration_report),
        ("pipeline_manifest", "manifest", validate_pipeline_manifest),
    ]
    for title, name, validator in json_checks:
        path = selected_path(name)
        if path is None:
            continue
        ok, errors = validate_json_file(path, validator)
        all_ok &= _report_json(title, ok, errors)

    if args.examples_dir:
        examples_dir = _resolve(base_dir, args.examples_dir)
        print(f"[INFO] validating examples in {examples_dir}")
        examples_ok, _ = _validate_examples(base_dir, examples_dir)
        all_ok &= examples_ok

    if not all_ok:
        raise SystemExit(1)
    print("[DONE] all contract checks passed.")


if __name__ == "__main__":
    main()
