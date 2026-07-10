#!/usr/bin/env python3
"""Step 2: Balanced Sampling attempt. STOP if minimums not met."""
import json, sys, hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except: continue
    return out


def main() -> None:
    td = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "teacher_labels"
    REPORT_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all teacher labels
    all_records = []
    for f in sorted(td.glob("llm_teacher_output.front_*.jsonl")):
        all_records.extend(load_jsonl(f))

    if not all_records:
        print("[ERROR] No records", file=sys.stderr)
        sys.exit(1)

    # ── Dedup by input_signature ────────────────────────────────
    deduped: Dict[str, Dict[str, Any]] = {}
    for r in all_records:
        sig = r.get("input_signature", "")
        if sig not in deduped:
            deduped[sig] = r
        else:
            # Keep the one with lower mismatch_score (less certain = better as negative)
            existing_mis = deduped[sig].get("llm_mismatch_score", 1.0)
            new_mis = r.get("llm_mismatch_score", 1.0)
            if r["llm_label"] == "match" or (new_mis < existing_mis):
                deduped[sig] = r

    deduped_list = list(deduped.values())

    # ── Hard negative threshold ─────────────────────────────────
    # Hard negative = mismatch score < 0.99 (less certain)
    # Hard negative = mismatch score < 0.99 (less certain)
    HARD_NEG_THRESH = 0.99
    matches = [r for r in deduped_list if r["llm_label"] == "match"]
    uncertains = [r for r in deduped_list if r["llm_label"] == "uncertain"]
    hard_negatives = [r for r in deduped_list
                      if r["llm_label"] == "mismatch" and r["llm_mismatch_score"] < HARD_NEG_THRESH]
    soft_negatives = [r for r in deduped_list
                      if r["llm_label"] == "mismatch" and r["llm_mismatch_score"] >= HARD_NEG_THRESH]

    # ── Minimum requirements ────────────────────────────────────
    MIN_MATCH = 30
    MIN_UNCERTAIN = 10
    MIN_MISMATCH = 30

    n_match = len(matches)
    n_uncertain = len(uncertains)
    n_hard_neg = len(hard_negatives)
    n_soft_neg = len(soft_negatives)

    # ── Multi-case coverage check ───────────────────────────────
    case_match = Counter(r.get("case_id", "?") for r in matches)
    case_uncertain = Counter(r.get("case_id", "?") for r in uncertains)
    case_hard = Counter(r.get("case_id", "?") for r in hard_negatives)

    # ── Determine if we can proceed ─────────────────────────────
    blockers = []
    if n_match < MIN_MATCH:
        blockers.append(f"match={n_match} < MIN_MATCH={MIN_MATCH}")
    if n_uncertain < MIN_UNCERTAIN:
        blockers.append(f"uncertain={n_uncertain} < MIN_UNCERTAIN={MIN_UNCERTAIN}")
    if n_hard_neg < MIN_MISMATCH:
        blockers.append(f"hard_neg={n_hard_neg} < MIN_MISMATCH={MIN_MISMATCH}")

    can_proceed = len(blockers) == 0

    # ── Build balanced sampling plan ────────────────────────────
    if can_proceed:
        # Balance: keep all match + uncertain + hard negatives
        # Downsample hard_neg if needed to match ratio
        target_mismatch = min(n_hard_neg, max(n_match, n_uncertain) * 3)
        selected = matches + uncertains + hard_negatives[:target_mismatch]
    else:
        # Show what would be selected if minimums were met
        selected = matches + uncertains + hard_negatives[:min(n_hard_neg, 30)]

    # ── Build output ────────────────────────────────────────────
    result: Dict[str, Any] = {
        "audit_version": "2.0",
        "step": "2_balanced_sampling",
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "source": {
            "raw_total": len(all_records),
            "deduped_total": len(deduped_list),
            "deduped_from": len(set(r["input_signature"] for r in all_records)),
        },

        "class_counts_after_dedup": {
            "match": n_match,
            "uncertain": n_uncertain,
            "hard_negatives": n_hard_neg,
            "soft_negatives": n_soft_neg,
            "total": n_match + n_uncertain + n_hard_neg + n_soft_neg,
        },

        "hard_negative_threshold": HARD_NEG_THRESH,

        "minimum_requirements": {
            "min_match": MIN_MATCH,
            "min_uncertain": MIN_UNCERTAIN,
            "min_mismatch": MIN_MISMATCH,
        },

        "multi_case_coverage": {
            "match_cases": dict(case_match),
            "uncertain_cases": dict(case_uncertain),
            "hard_negative_cases": dict(case_hard),
        },

        "can_proceed_to_training": can_proceed,
        "blocking_reasons": blockers,

        "status": "PASS" if can_proceed else "BLOCKED — cannot proceed to training",
    }

    # ── What WOULD be selected ──────────────────────────────────
    selected_sigs = [r.get("input_signature", "") for r in selected]
    result["balanced_set_would_be"] = {
        "n_selected": len(selected),
        "n_match": sum(1 for r in selected if r["llm_label"] == "match"),
        "n_uncertain": sum(1 for r in selected if r["llm_label"] == "uncertain"),
        "n_mismatch": sum(1 for r in selected if r["llm_label"] == "mismatch"),
        "cases_covered": sorted(set(r.get("case_id", "?") for r in selected)),
    }

    # ── Write output ────────────────────────────────────────────
    out_path = REPORT_DIR / "step2_balanced_sampling.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── Write what samples would be used (for traceability) ─────
    samples_path = REPORT_DIR / "step2_proposed_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for r in selected:
            sub = {
                "input_signature": r.get("input_signature", ""),
                "case_id": r.get("case_id", ""),
                "event_id": r.get("event_id", ""),
                "track_id": r.get("track_id", -1),
                "llm_label": r.get("llm_label", ""),
                "llm_match_score": r.get("llm_match_score", 0),
                "llm_mismatch_score": r.get("llm_mismatch_score", 0),
                "llm_confidence": r.get("llm_confidence", 0),
                "provider_mode": r.get("provider_mode", "?"),
            }
            f.write(json.dumps(sub, ensure_ascii=False) + "\n")

    # ── Print ───────────────────────────────────────────────────
    print(f"=== Step 2: Balanced Sampling ===")
    print(f"Match: {n_match} (need >= {MIN_MATCH})")
    print(f"Uncertain: {n_uncertain} (need >= {MIN_UNCERTAIN})")
    print(f"Hard negatives: {n_hard_neg} (need >= {MIN_MISMATCH})")
    print(f"Soft negatives: {n_soft_neg}")
    print(f"Can proceed: {can_proceed}")
    if blockers:
        for b in blockers:
            print(f"  BLOCKED: {b}")
    print(f"\n[DONE] Report: {out_path} ({out_path.stat().st_size} bytes)")
    print(f"[DONE] Proposed samples: {samples_path} ({samples_path.stat().st_size} bytes)")
    print(f"[STATUS] {result['status']}")

    if not can_proceed:
        sys.exit(2)  # exit code 2 = blocked, not error


if __name__ == "__main__":
    main()
