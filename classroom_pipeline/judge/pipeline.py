#!/usr/bin/env python3
"""LLM judge pipeline orchestrator.

Orchestrates the full pipeline:
  1. build_llm_evidence  (structured multimodal evidence → JSONL)
  2. run_llm_teacher     (LLM teacher → silver labels)
  3. train_student       (student judge training → .pt checkpoint)
  4. run_all_checks      (anti-drift smoke tests → report)

Usage:
  python -m judge.pipeline --event_queries ... --aligned ... --out_dir ...
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path when this file is executed directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, validate_jsonl_file, validate_llm_silver_label_record
from classroom_pipeline.judge.build_evidence import build_llm_evidence, smoke_check_evidence
from classroom_pipeline.judge.llm_teacher import LLMTeacherConfig, run_llm_teacher_batch, smoke_check_silver_labels
from classroom_pipeline.judge.student_train import train_student, smoke_check_student
from classroom_pipeline.judge.smoketest import run_all_checks


def _resolve(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _write_marker(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or f"DONE {datetime.now().isoformat()}", encoding="utf-8")


def _find_next_run(out_dir_root: Path) -> Path:
    """Find next available run_NNN directory."""
    out_dir_root.mkdir(parents=True, exist_ok=True)
    max_run = 0
    for child in out_dir_root.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                n = int(child.name.split("_")[-1])
                max_run = max(max_run, n)
            except ValueError:
                continue
    return out_dir_root / f"run_{max_run + 1:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Judge Pipeline: evidence → teacher → student → smoke test"
    )
    parser.add_argument("--event_queries", required=True, type=str,
                        help="event_queries.fusion_v2.jsonl")
    parser.add_argument("--aligned", required=True, type=str,
                        help="align_multimodal.json")
    parser.add_argument("--actions", default="", type=str,
                        help="actions.jsonl (optional)")
    parser.add_argument("--pose_uq", default="", type=str,
                        help="pose_uq.jsonl (optional)")
    parser.add_argument("--out_dir", default="output/llm_judge_pipeline", type=str,
                        help="output root directory")
    parser.add_argument("--teacher_model", default="simulate", type=str,
                        help="LLM model: simulate, openai:gpt-4o, claude:sonnet-4-20250506")
    parser.add_argument("--api_key", default="", type=str,
                        help="LLM API key (or set LLM_API_KEY env var)")
    parser.add_argument("--teacher_temperature", type=float, default=0.1)
    parser.add_argument("--student_epochs", type=int, default=120)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--student_hidden_dim", type=int, default=16)
    parser.add_argument("--target_threshold", type=float, default=0.60,
                        help="silver_p_match >= this → positive target")
    parser.add_argument("--max_teacher_records", type=int, default=0,
                        help="limit teacher records for testing (0 = all)")
    parser.add_argument("--skip_teacher", action="store_true",
                        help="skip LLM teacher step (use existing silver labels)")
    parser.add_argument("--skip_student", action="store_true",
                        help="skip student training step")
    parser.add_argument("--existing_silver_labels", default="", type=str,
                        help="use existing silver labels instead of running teacher")
    parser.add_argument("--previous_smoketest", default="", type=str,
                        help="previous smoketest_report.json for drift comparison")
    args = parser.parse_args()

    t_start = time.time()

    # ── Setup ──────────────────────────────────────────────────────────
    event_queries_path = _resolve(args.event_queries, _PROJECT_ROOT)
    aligned_path = _resolve(args.aligned, _PROJECT_ROOT)
    actions_path = _resolve(args.actions, _PROJECT_ROOT) if args.actions else None
    pose_uq_path = _resolve(args.pose_uq, _PROJECT_ROOT) if args.pose_uq else None

    out_dir = _resolve(args.out_dir, _PROJECT_ROOT)
    run_dir = _find_next_run(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PIPELINE] Output directory: {run_dir}")

    # Paths for intermediate and final artifacts
    evidence_path = run_dir / "llm_evidence.jsonl"
    silver_labels_path = run_dir / "llm_silver_labels.jsonl"
    teacher_batch_report_path = run_dir / "llm_teacher_batch_report.json"
    student_samples_path = run_dir / "student_samples.jsonl"
    student_model_path = run_dir / "student_judge.pt"
    student_report_path = run_dir / "student_train_report.json"
    smoketest_path = run_dir / "smoketest_report.json"
    done_path = run_dir / "DONE"

    # ── Step 1: Build LLM evidence ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("[STEP 1/3] Building LLM evidence...")
    print("=" * 60)

    if evidence_path.exists():
        print(f"  Evidence exists, reusing: {evidence_path}")
        evidence_list = []
        with evidence_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        evidence_list.append(json.loads(line))
                    except Exception:
                        continue
    else:
        evidence_list = build_llm_evidence(
            event_queries_path=event_queries_path,
            aligned_path=aligned_path,
            actions_path=actions_path,
            pose_uq_path=pose_uq_path,
        )
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        from classroom_pipeline.contracts.schemas import write_jsonl
        write_jsonl(evidence_path, evidence_list)
        print(f"  Generated {len(evidence_list)} evidence records")

    # Smoke check Step 1
    ev_check = smoke_check_evidence(evidence_list, aligned_path)
    if not ev_check["passed"]:
        print(f"[FAIL] Evidence smoke check: {ev_check['detail']}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] Evidence smoke check: {ev_check['detail']}")

    # ── Step 2: LLM Teacher ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[STEP 2/3] Running LLM teacher...")
    print("=" * 60)

    if args.skip_teacher:
        if args.existing_silver_labels:
            silver_labels_path = _resolve(args.existing_silver_labels, _PROJECT_ROOT)
            print(f"  Skipping teacher, using existing: {silver_labels_path}")
        else:
            print("  Skipping teacher step (--skip_teacher)")
    else:
        teacher_config = LLMTeacherConfig(
            model=args.teacher_model,
            api_key=args.api_key,
            temperature=args.teacher_temperature,
        )
        silver_labels = run_llm_teacher_batch(
            evidence_path=evidence_path,
            output_path=silver_labels_path,
            config=teacher_config,
            max_records=args.max_teacher_records,
        )

        # Write batch report
        sim_count = sum(1 for s in silver_labels if s.get("is_simulated", True))
        real_count = len(silver_labels) - sim_count
        batch_report = {
            "model": args.teacher_model,
            "temperature": args.teacher_temperature,
            "total_records": len(silver_labels),
            "simulated": sim_count,
            "real_api_calls": real_count,
            "is_simulated": args.teacher_model == "simulate",
        }
        with teacher_batch_report_path.open("w", encoding="utf-8") as f:
            json.dump(batch_report, f, ensure_ascii=False, indent=2)

    # Smoke check Step 2
    silver_labels_list: List[Dict[str, Any]] = []
    with silver_labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    silver_labels_list.append(json.loads(line))
                except Exception:
                    continue

    silver_check = smoke_check_silver_labels(silver_labels_list)
    if not silver_check["passed"]:
        print(f"[FAIL] Silver label smoke check: {silver_check['detail']}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] Silver label smoke check: {silver_check['detail']}")

    # Schema validation
    ok, count, errors = validate_jsonl_file(silver_labels_path, validate_llm_silver_label_record)
    if not ok:
        print(f"[FAIL] Silver label schema validation: {errors[0]}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] Silver label schema: {count} records valid")

    # ── Step 3: Train Student Judge ────────────────────────────────────
    print("\n" + "=" * 60)
    print("[STEP 3/3] Training student judge...")
    print("=" * 60)

    if args.skip_student:
        print("  Skipping student training (--skip_student)")
    else:
        report = train_student(
            silver_labels_path=silver_labels_path,
            event_queries_path=event_queries_path,
            aligned_path=aligned_path,
            out_model=student_model_path,
            out_report=student_report_path,
            out_samples=student_samples_path,
            epochs=args.student_epochs,
            lr=args.student_lr,
            hidden_dim=args.student_hidden_dim,
            target_threshold=args.target_threshold,
        )

        # Smoke check Step 3
        student_check = smoke_check_student(report)
        if student_check.get("warnings"):
            for w in student_check["warnings"]:
                print(f"  [WARN] {w}")
        if not student_check["passed"]:
            print(f"[FAIL] Student training: {student_check['detail']}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"[PASS] Student training: {student_check['detail']}")

    # ── Step 4: Anti-drift smoke tests ─────────────────────────────────
    print("\n" + "=" * 60)
    print("[STEP 4/3] Running anti-drift smoke tests...")
    print("=" * 60)

    smoketest_report = run_all_checks(
        silver_labels_path=silver_labels_path,
        student_samples_path=student_samples_path if student_samples_path.exists() else None,
        student_checkpoint_path=student_model_path if student_model_path.exists() else None,
        previous_smoketest_path=_resolve(args.previous_smoketest, _PROJECT_ROOT)
        if args.previous_smoketest else None,
        kl_threshold=0.20,
        consistency_threshold=0.15,
        cross_case_std_threshold=0.10,
        loocv_epochs=60,
    )

    with smoketest_path.open("w", encoding="utf-8") as f:
        json.dump(smoketest_report, f, ensure_ascii=False, indent=2)

    for name, check in smoketest_report.get("checks", {}).items():
        icon = "PASS" if check.get("passed", False) else "FAIL"
        print(f"  [{icon}] {name}")

    if not smoketest_report.get("all_checks_passed", False):
        print(f"[FAIL] Some anti-drift checks failed — see {smoketest_path}", file=sys.stderr)
        sys.exit(1)

    # ── Done ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    _write_marker(done_path)

    print("\n" + "=" * 60)
    print(f"[DONE] Pipeline complete in {elapsed:.1f}s")
    print(f"[DONE] All artifacts in: {run_dir}")
    if (student_model_path.exists()):
        print(f"[DONE] Student checkpoint: {student_model_path}")
        print(f"[HINT] Use with: --verifier_model {student_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
