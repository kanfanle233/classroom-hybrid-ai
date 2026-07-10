#!/usr/bin/env python3
"""Batch runner for LLM teacher pipeline across all available cases.

Produces:
  - teacher_labels/*.jsonl          per-case silver labels
  - silver_label_summary.json       aggregate over all cases
  - agreement_with_rule_verifier.json  cross-tabulation
  - skipped_events_report.json      any events that were skipped
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from classroom_pipeline.contracts.llm_teacher_schema import (
    SCHEMA_VERSION,
    PROMPT_VERSION,
    validate_llm_teacher_jsonl,
    write_teacher_jsonl,
)
from classroom_pipeline.contracts.schemas import write_json

# Import 06h_run_llm_teacher via importlib (filename starts with digit)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "llm_teacher_step",
    _PROJECT_ROOT / "scripts" / "pipeline" / "06h_run_llm_teacher.py",
)
_llm_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_llm_mod)
build_evidence_pairs = _llm_mod.build_evidence_pairs
run_llm_teacher = _llm_mod.run_llm_teacher


# ── Config ───────────────────────────────────────────────────────────────

OUTPUT_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "teacher_labels"

CASES = [
    {
        "name": "front_002_full_pose020_hybrid",
        "event_queries": "output/codex_reports/front_002_full_pose020_hybrid/event_queries.fusion_v2.jsonl",
        "aligned": "output/codex_reports/front_002_full_pose020_hybrid/align_multimodal.json",
        "actions": "output/codex_reports/front_002_full_pose020_hybrid/actions.fusion_v2.jsonl",
        "pose_uq": "output/codex_reports/front_002_full_pose020_hybrid/pose_tracks_smooth_uq.jsonl",
        "verified": "output/codex_reports/front_002_full_pose020_hybrid/verified_events.jsonl",
    },
    {
        "name": "front_002_rear_row_sliced_pose020_hybrid",
        "event_queries": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/event_queries.fusion_v2.jsonl",
        "aligned": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/align_multimodal.json",
        "actions": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/actions.fusion_v2.jsonl",
        "pose_uq": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/pose_tracks_smooth_uq.jsonl",
        "verified": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/verified_events.jsonl",
    },
    {
        "name": "front_1885_full",
        "event_queries": "output/codex_reports/front_1885_full/event_queries.fusion_v2.jsonl",
        "aligned": "output/codex_reports/front_1885_full/align_multimodal.json",
        "actions": "output/codex_reports/front_1885_full/actions.fusion_v2.jsonl",
        "pose_uq": "output/codex_reports/front_1885_full/pose_tracks_smooth_uq.jsonl",
        "verified": "output/codex_reports/front_1885_full/verified_events.jsonl",
    },
    {
        "name": "front_22259_full",
        "event_queries": "output/codex_reports/front_22259_full/event_queries.fusion_v2.jsonl",
        "aligned": "output/codex_reports/front_22259_full/align_multimodal.json",
        "actions": "output/codex_reports/front_22259_full/actions.fusion_v2.jsonl",
        "pose_uq": "output/codex_reports/front_22259_full/pose_tracks_smooth_uq.jsonl",
        "verified": "output/codex_reports/front_22259_full/verified_events.jsonl",
    },
    {
        "name": "front_26729_full",
        "event_queries": "output/codex_reports/front_26729_full/event_queries.fusion_v2.jsonl",
        "aligned": "output/codex_reports/front_26729_full/align_multimodal.json",
        "actions": "output/codex_reports/front_26729_full/actions.fusion_v2.jsonl",
        "pose_uq": "output/codex_reports/front_26729_full/pose_tracks_smooth_uq.jsonl",
        "verified": "output/codex_reports/front_26729_full/verified_events.jsonl",
    },
    {
        "name": "front_45618_full",
        "event_queries": "output/codex_reports/front_45618_full/event_queries.fusion_v2.jsonl",
        "aligned": "output/codex_reports/front_45618_full/align_multimodal.json",
        "actions": "output/codex_reports/front_45618_full/actions.fusion_v2.jsonl",
        "pose_uq": "output/codex_reports/front_45618_full/pose_tracks_smooth_uq.jsonl",
        "verified": "output/codex_reports/front_45618_full/verified_events.jsonl",
    },
]


def _resolve(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_verifier_labels(verified_path: Path) -> Dict[str, str]:
    """Load verifier labels keyed by event_id+track_id."""
    labels = {}
    for row in _load_jsonl(verified_path):
        eid = str(row.get("event_id", row.get("query_id", "")))
        tid = int(row.get("track_id", -1))
        key = f"{eid}_{tid}"
        label = str(row.get("label", row.get("match_label", "")))
        labels[key] = label
    return labels


def compute_agreement(
    all_records: List[Dict[str, Any]],
    verifier_labels: Dict[str, str],
) -> Dict[str, Any]:
    """Cross-tabulate LLM labels vs verifier labels."""
    cross = defaultdict(lambda: defaultdict(int))
    llm_only = 0
    verifier_only = 0
    matched = 0
    total = 0

    for rec in all_records:
        eid = rec["event_id"]
        tid = rec["track_id"]
        key = f"{eid}_{tid}"
        llm_label = rec["llm_label"]
        ver_label = verifier_labels.get(key)

        if ver_label is None:
            llm_only += 1
            continue

        total += 1
        cross[llm_label][ver_label] += 1
        if llm_label == ver_label:
            matched += 1

    # Build clean cross-tab matrix
    labels_order = ["match", "uncertain", "mismatch"]
    matrix = {}
    for ll in labels_order:
        row = {}
        for vl in labels_order:
            row[vl] = cross[ll].get(vl, 0)
        row["total"] = sum(cross[ll].values())
        matrix[ll] = row

    agreement_rate = matched / max(1, total)

    return {
        "total_comparable": total,
        "matched": matched,
        "agreement_rate": round(agreement_rate, 4),
        "llm_only_records": llm_only,
        "cross_tabulation": matrix,
        "note": "LLM silver labels vs rule-based verifier labels. "
                "Agreement does not imply correctness — verifier uses heuristic rules, "
                "LLM uses semantic understanding.",
    }


def main() -> None:
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    per_case_stats: Dict[str, Dict[str, Any]] = {}
    global_label_dist: Counter = Counter()
    global_provider_dist: Counter = Counter()
    global_confidences: List[float] = []
    verifier_labels_all: Dict[str, str] = {}
    skipped_events: List[Dict[str, Any]] = []
    total_expected = 0
    total_actual = 0

    for case in CASES:
        name = case["name"]
        print(f"\n{'='*60}")
        print(f"[BATCH] Processing: {name}")
        print(f"{'='*60}")

        eq_path = _resolve(case["event_queries"])
        al_path = _resolve(case["aligned"])
        ac_path = _resolve(case["actions"])
        uq_path = _resolve(case["pose_uq"])
        vf_path = _resolve(case["verified"])

        # Build evidence
        pairs = build_evidence_pairs(
            event_queries_path=eq_path,
            aligned_path=al_path,
            actions_path=ac_path,
            pose_uq_path=uq_path,
            case_id=name,
        )
        total_expected += len(pairs)
        print(f"  Evidence pairs: {len(pairs)}")

        if not pairs:
            skipped_events.append({
                "case_id": name,
                "reason": "no evidence pairs generated",
                "n_pairs": 0,
            })
            continue

        # Run teacher (simulate, all records)
        records = run_llm_teacher(
            pairs,
            model_name="simulate",
            max_records=0,
            verbose=True,
        )
        total_actual += len(records)

        # Track skipped
        n_skipped = len(pairs) - len(records)
        if n_skipped > 0:
            skipped_events.append({
                "case_id": name,
                "reason": f"{n_skipped} records failed schema validation",
                "n_pairs": len(pairs),
                "n_records": len(records),
                "n_skipped": n_skipped,
            })

        # Write per-case output
        case_output = OUTPUT_DIR / f"llm_teacher_output.{name}.jsonl"
        try:
            write_teacher_jsonl(case_output, records)
            print(f"  Output: {case_output}")
        except ValueError as e:
            print(f"  [ERROR] Schema validation failed for {name}: {e}", file=sys.stderr)
            skipped_events.append({
                "case_id": name,
                "reason": f"schema validation error: {e}",
                "n_pairs": len(pairs),
                "n_records": len(records),
            })
            continue

        # Validate output file
        ok, count, errors = validate_llm_teacher_jsonl(case_output)
        print(f"  Schema validation: {'PASS' if ok else 'FAIL'} ({count} valid, {len(errors)} errors)")
        if not ok:
            skipped_events.append({
                "case_id": name,
                "reason": f"file-level schema validation failed: {errors[0] if errors else 'unknown'}",
                "n_records": count,
            })
            continue

        # Per-case stats
        labels = Counter(r["llm_label"] for r in records)
        providers = Counter(r["provider_mode"] for r in records)
        confs = [r["llm_confidence"] for r in records]

        per_case_stats[name] = {
            "case_id": name,
            "n_pairs": len(pairs),
            "n_records": len(records),
            "label_distribution": dict(labels),
            "provider_distribution": dict(providers),
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
        }

        print(f"  Labels: {dict(labels)}")
        print(f"  Providers: {dict(providers)}")

        # Accumulate
        all_records.extend(records)
        global_label_dist.update(labels)
        global_provider_dist.update(providers)
        global_confidences.extend(confs)

        # Load verifier labels for agreement
        verifier_labels_all.update(load_verifier_labels(vf_path))

    # ── Aggregate Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[BATCH] Generating aggregate reports")
    print(f"{'='*60}")

    # Sanity check
    if not all_records:
        print("[ERROR] No records produced across any case", file=sys.stderr)
        sys.exit(1)

    # Summary
    summary = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "disclaimer": (
            "These are SILVER LABELS produced by an LLM teacher in late-fusion mode. "
            "They are NOT gold labels. They represent the LLM's judgment based on "
            "structured multimodal evidence (text/JSON only, no video). "
            "Use for training student judge models, not for final evaluation."
        ),
        "total_cases": len(CASES),
        "cases_with_output": sum(1 for c in per_case_stats.values()),
        "total_expected_pairs": total_expected,
        "total_actual_records": total_actual,
        "simulate_only": global_provider_dist.get("simulate", 0) == total_actual,
        "provider_summary": {
            "simulate": global_provider_dist.get("simulate", 0),
            "real": global_provider_dist.get("real", 0),
        },
        "global_label_distribution": dict(global_label_dist),
        "global_mean_confidence": round(sum(global_confidences) / len(global_confidences), 4) if global_confidences else 0.0,
        "per_case": per_case_stats,
        "output_dir": str(OUTPUT_DIR),
    }

    summary_path = OUTPUT_DIR / "silver_label_summary.json"
    write_json(summary_path, summary)
    print(f"[DONE] Summary: {summary_path}")

    # ── Agreement with rule verifier ──
    agreement = compute_agreement(all_records, verifier_labels_all)
    agreement["note"] = (
        "Silver labels (LLM teacher) vs rule-based verifier labels cross-tabulation. "
        "These are NOT gold labels. Disagreement is expected — the LLM uses semantic "
        "understanding while the verifier uses heuristic rules."
    )

    agreement_path = OUTPUT_DIR / "agreement_with_rule_verifier.json"
    write_json(agreement_path, agreement)
    print(f"[DONE] Agreement: {agreement_path}")
    print(f"  Comparable: {agreement['total_comparable']}, "
          f"Agreement: {agreement['agreement_rate']:.2%}")
    if agreement["cross_tabulation"]:
        print(f"  Cross-tabulation:")
        for ll, row in agreement["cross_tabulation"].items():
            print(f"    LLM={ll:10s}: {dict(row)}")

    # ── Skipped events report ──
    skipped_path = OUTPUT_DIR / "skipped_events_report.json"
    skipped_report = {
        "schema_version": SCHEMA_VERSION,
        "total_expected_pairs": total_expected,
        "total_actual_records": total_actual,
        "total_skipped": total_expected - total_actual,
        "skipped_events": skipped_events if skipped_events else [],
        "note": "Empty list means all events were processed without error.",
    }
    write_json(skipped_path, skipped_report)
    print(f"[DONE] Skipped events: {skipped_path}")

    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[BATCH] Complete in {elapsed:.1f}s")
    print(f"  Cases: {len(per_case_stats)}/{len(CASES)}")
    print(f"  Records: {total_actual}/{total_expected} ({total_expected - total_actual} skipped)")
    print(f"  Provider: simulate={global_provider_dist.get('simulate', 0)}, "
          f"real={global_provider_dist.get('real', 0)}")
    print(f"  Label distribution: {dict(global_label_dist)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
