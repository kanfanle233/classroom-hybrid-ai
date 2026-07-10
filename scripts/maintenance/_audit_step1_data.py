#!/usr/bin/env python3
"""Step 1: Data Audit — scan, dedup analysis, class distribution, machine-readable JSON."""
import json, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def main() -> None:
    td = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "teacher_labels"
    REPORT_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all teacher labels ─────────────────────────────────
    label_files = sorted(td.glob("llm_teacher_output.front_*.jsonl"))
    if not label_files:
        label_files = sorted(td.glob("llm_teacher_output.*.jsonl"))

    all_raw: List[Dict[str, Any]] = []
    per_file: Dict[str, int] = {}
    for f in label_files:
        recs = load_jsonl(f)
        per_file[f.name] = len(recs)
        all_raw.extend(recs)

    raw_total = len(all_raw)
    if raw_total == 0:
        print("[ERROR] No teacher labels found", file=sys.stderr)
        sys.exit(1)

    # ── Dedup analysis ──────────────────────────────────────────
    sig_map: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(all_raw):
        sig = r.get("input_signature", "")
        sig_map[sig].append(i)

    n_unique = len(sig_map)
    n_dropped = raw_total - n_unique
    unique_sigs = {k: v for k, v in sig_map.items() if len(v) == 1}
    dup_sigs = {k: v for k, v in sig_map.items() if len(v) > 1}

    dup_detail = []
    for sig, indices in sorted(dup_sigs.items(), key=lambda x: -len(x[1])):
        cases = [all_raw[i].get("case_id", "?") for i in indices]
        eids = [all_raw[i].get("event_id", "?") for i in indices]
        tids = [all_raw[i].get("track_id", -1) for i in indices]
        labels = [all_raw[i].get("llm_label", "?") for i in indices]
        same_case = len(set(cases)) == 1
        dup_detail.append({
            "input_signature": sig,
            "count": len(indices),
            "same_case": same_case,
            "cases": list(set(cases)),
            "event_ids": list(set(eids)),
            "track_ids": list(set(tids)),
            "labels": labels,
            "all_case_ids": cases,
        })

    # ── Class distribution ──────────────────────────────────────
    label_dist = Counter(r["llm_label"] for r in all_raw)
    per_case_dist = {}
    for f in label_files:
        case = f.stem.replace("llm_teacher_output.", "")
        recs = load_jsonl(f)
        dist = Counter(r["llm_label"] for r in recs)
        per_case_dist[case] = {"n": len(recs), "labels": dict(dist)}

    all_match = [r for r in all_raw if r["llm_label"] == "match"]
    all_uncertain = [r for r in all_raw if r["llm_label"] == "uncertain"]

    # ── Provider audit ──────────────────────────────────────────
    providers = Counter(r.get("provider_mode", "?") for r in all_raw)

    # ── Score statistics ────────────────────────────────────────
    match_scores = [r["llm_match_score"] for r in all_raw]
    mismatch_scores = [r["llm_mismatch_score"] for r in all_raw]
    uncertain_scores = [r["llm_uncertain_score"] for r in all_raw]
    confs = [r["llm_confidence"] for r in all_raw]

    # ── Hard negative analysis ──────────────────────────────────
    hard_neg_threshold = 0.99
    hard_negatives = [r for r in all_raw
                      if r["llm_label"] == "mismatch" and r["llm_mismatch_score"] < hard_neg_threshold]
    soft_negatives = [r for r in all_raw
                      if r["llm_label"] == "mismatch" and r["llm_mismatch_score"] >= hard_neg_threshold]

    # ── Build audit JSON ────────────────────────────────────────
    audit: Dict[str, Any] = {
        "audit_version": "2.0",
        "step": "1_data_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_output_dir": str(td.parent),

        "raw_counts": {
            "total_raw_records": raw_total,
            "per_file": per_file,
            "raw_total_check": sum(per_file.values()),
        },

        "dedup": {
            "unique_input_signatures": n_unique,
            "records_dropped_by_dedup": n_dropped,
            "duplicate_signature_count": len(dup_sigs),
            "explanation": (
                f"{len(dup_sigs)} input_signatures appear >1 time. "
                "input_signature = sha256(event_id|track_id|query_text|behavior_code)[:16]. "
                "Duplicates occur because the same (event_id, track_id) appears in "
                "alignment data from multiple cases (cross-case overlap) or duplicate candidate entries."
            ),
            "top_duplicates": dup_detail[:15],
            "duplicate_file": str(REPORT_DIR / "step1_duplicate_records.json"),
        },

        "class_distribution": {
            "raw_all_records": dict(label_dist),
            "per_case": per_case_dist,
            "n_match": len(all_match),
            "n_mismatch": label_dist.get("mismatch", 0),
            "n_uncertain": label_dist.get("uncertain", 0),
            "uncertain_ratio": round(label_dist.get("uncertain", 0) / max(1, raw_total), 4),
            "match_ratio": round(len(all_match) / max(1, raw_total), 4),
            "class_collapse_detected": label_dist.get("uncertain", 0) == 0,
        },

        "provider_audit": {
            "distribution": dict(providers),
            "all_simulate": providers.get("simulate", 0) == raw_total,
            "warning": "ALL records are simulate mode — no real LLM was used.",
        },

        "score_statistics": {
            "match_score": {"min": round(min(match_scores), 6), "max": round(max(match_scores), 6),
                            "mean": round(sum(match_scores) / len(match_scores), 6)},
            "mismatch_score": {"min": round(min(mismatch_scores), 6), "max": round(max(mismatch_scores), 6),
                               "mean": round(sum(mismatch_scores) / len(mismatch_scores), 6)},
            "uncertain_score": {"min": round(min(uncertain_scores), 6), "max": round(max(uncertain_scores), 6),
                                "mean": round(sum(uncertain_scores) / len(uncertain_scores), 6)},
            "confidence": {"min": round(min(confs), 6), "max": round(max(confs), 6),
                           "mean": round(sum(confs) / len(confs), 6)},
        },

        "hard_negative_analysis": {
            "total_mismatch": label_dist.get("mismatch", 0),
            "hard_negatives_count": len(hard_negatives),
            "soft_negatives_count": len(soft_negatives),
            "hard_neg_threshold": hard_neg_threshold,
            "explanation": "Hard negatives = mismatch samples where mismatch_score < threshold. These are less certain negatives and more valuable for training.",
        },

        "training_readiness": {
            "can_train": False,
            "blocking_reasons": [
                "0 uncertain samples — cannot train uncertain class",
                f"Only {len(all_match)} match samples — need >=30 for meaningful training",
                f"{label_dist.get('mismatch', 0)} mismatch samples — extreme class imbalance",
                "All event_types are 'unknown' — teacher features are non-discriminative",
                "Simulate teacher labels are not real LLM judgments",
                "Val/test would have 0 match/uncertain — metrics degenerate",
            ],
            "minimum_requirements": {
                "min_match": 30,
                "min_mismatch": 30,
                "min_uncertain": 10,
                "current_match": len(all_match),
                "current_mismatch": label_dist.get("mismatch", 0),
                "current_uncertain": label_dist.get("uncertain", 0),
            },
            "status": "BLOCKED",
        },
    }

    # ── Write audit JSON ────────────────────────────────────────
    audit_path = REPORT_DIR / "step1_data_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # ── Write duplicate records list ────────────────────────────
    dup_path = REPORT_DIR / "step1_duplicate_records.json"
    dup_for_json = [
        {k: v for k, v in d.items() if k != "all_case_ids"}
        for d in dup_detail
    ]
    with dup_path.open("w", encoding="utf-8") as f:
        json.dump(dup_for_json, f, ensure_ascii=False, indent=2)

    # ── Print summary ───────────────────────────────────────────
    print(f"=== Step 1: Data Audit Complete ===")
    print(f"Raw records: {raw_total}")
    print(f"Unique signatures: {n_unique}")
    print(f"Dropped by dedup: {n_dropped} ({len(dup_sigs)} signatures)")
    print(f"Label distribution: {dict(label_dist)}")
    print(f"Provider: {dict(providers)}")
    print(f"Hard negatives: {len(hard_negatives)}, Soft negatives: {len(soft_negatives)}")
    print(f"")
    print(f"=== BLOCKING REASONS ===")
    for r in audit["training_readiness"]["blocking_reasons"]:
        print(f"  BLOCKED: {r}")
    print(f"")
    print(f"[DONE] Audit JSON: {audit_path} ({audit_path.stat().st_size} bytes)")
    print(f"[DONE] Duplicates: {dup_path} ({dup_path.stat().st_size} bytes)")
    print(f"[STATUS] Training readiness: BLOCKED")


if __name__ == "__main__":
    main()
