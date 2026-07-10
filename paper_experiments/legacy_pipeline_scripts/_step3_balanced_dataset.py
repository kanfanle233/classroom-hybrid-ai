#!/usr/bin/env python3
"""Step 3: Build balanced hard-case dataset from teacher labels + verifier evidence."""
import json, csv, sys, hashlib
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


def load_json(path: Path) -> Any:
    if not path.exists(): return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x, d=0.0):
    try: return float(x)
    except: return d


def main() -> None:
    REPORT_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "reports"
    OUT_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all teacher labels ──────────────────────────────
    td = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "teacher_labels"
    all_teacher = []
    for f in sorted(td.glob("llm_teacher_output.front_*.jsonl")):
        all_teacher.extend(load_jsonl(f))

    # Dedup by input_signature — keep the one with lowest mismatch_score
    deduped: Dict[str, Dict[str, Any]] = {}
    for r in all_teacher:
        sig = r.get("input_signature", "")
        if sig not in deduped:
            deduped[sig] = r
        else:
            existing_mis = deduped[sig].get("llm_mismatch_score", 1.0)
            new_mis = r.get("llm_mismatch_score", 1.0)
            if r["llm_label"] == "match" or new_mis < existing_mis:
                deduped[sig] = r

    deduped_list = list(deduped.values())
    print(f"Raw teacher: {len(all_teacher)}, deduped: {len(deduped_list)}, dropped: {len(all_teacher)-len(deduped_list)}")

    # ── 2. Categorize ───────────────────────────────────────────
    matches = [r for r in deduped_list if r["llm_label"] == "match"]
    uncertains = [r for r in deduped_list if r["llm_label"] == "uncertain"]

    # ── 3. Build verifier label lookup ───────────────────────────
    CASE_CONFIGS = {
        "front_002_full_pose020_hybrid": "output/codex_reports/front_002_full_pose020_hybrid/verified_events.jsonl",
        "front_002_rear_row_sliced_pose020_hybrid": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/verified_events.jsonl",
        "front_1885_full": "output/codex_reports/front_1885_full/verified_events.jsonl",
        "front_22259_full": "output/codex_reports/front_22259_full/verified_events.jsonl",
        "front_26729_full": "output/codex_reports/front_26729_full/verified_events.jsonl",
        "front_45618_full": "output/codex_reports/front_45618_full/verified_events.jsonl",
    }

    verifier_labels: Dict[str, Dict[str, Any]] = {}
    for case_id, rel_path in CASE_CONFIGS.items():
        vp = _PROJECT_ROOT / rel_path
        if not vp.exists():
            continue
        for row in load_jsonl(vp):
            eid = str(row.get("event_id", row.get("query_id", "")))
            tid = row.get("track_id", -1)
            key = f"{case_id}|{eid}|{tid}"
            label = str(row.get("label", row.get("match_label", "")))
            p_match = _safe_float(row.get("p_match", 0.0))
            evidence = row.get("evidence", {}) if isinstance(row.get("evidence"), dict) else {}
            verifier_labels[key] = {
                "verifier_label": label,
                "verifier_p_match": p_match,
                "verifier_fusion": evidence.get("fusion_mode", ""),
            }

    # ── 4. Score every mismatch candidate for hard-negative potential ──
    scored_negatives = []
    for r in deduped_list:
        if r["llm_label"] != "mismatch":
            continue
        case = r.get("case_id", "")
        eid = r.get("event_id", "")
        tid = r.get("track_id", -1)
        key = f"{case}|{eid}|{tid}"
        vinfo = verifier_labels.get(key, {})

        # Score = sum of hard-negative criteria (higher = harder)
        score = 0.0

        # Criterion 1: verifier says match but teacher says mismatch (disagreement)
        if vinfo.get("verifier_label") == "match":
            score += 2.0
        elif vinfo.get("verifier_label") == "uncertain":
            score += 1.0

        # Criterion 2: high verifier p_match but teacher says mismatch
        vpm = vinfo.get("verifier_p_match", 0.0)
        if vpm > 0.6 and r.get("llm_match_score", 0) < 0.4:
            score += 1.0

        # Criterion 3: teacher mismatch_score is NOT at ceiling (less certain)
        mismatch_score = r.get("llm_mismatch_score", 1.0)
        if mismatch_score < 0.99:
            score += 1.5
        if mismatch_score < 0.95:
            score += 1.0

        # Criterion 4: teacher confidence is lower (more informative)
        confidence = r.get("llm_confidence", 1.0)
        if confidence < 0.95:
            score += 0.5
        if confidence < 0.90:
            score += 0.5

        scored_negatives.append({
            "record": r,
            "hardness_score": round(score, 2),
            "verifier_label": vinfo.get("verifier_label", "unknown"),
            "verifier_p_match": vpm,
            "teacher_mismatch_score": mismatch_score,
            "teacher_confidence": confidence,
        })

    # Sort by hardness (highest first)
    scored_negatives.sort(key=lambda x: x["hardness_score"], reverse=True)

    # ── 5. Select balanced set ──────────────────────────────────
    TARGET_MISMATCH = 30  # balanced target

    selected_match_records = matches  # all 8
    selected_uncertain_records = uncertains  # all 0
    selected_mismatch_records = [sn["record"] for sn in scored_negatives[:TARGET_MISMATCH]]

    # Build final balanced dataset
    all_selected = []
    all_selected.extend(selected_match_records)
    all_selected.extend(selected_uncertain_records)
    all_selected.extend(selected_mismatch_records)

    # Add provenance
    balanced_records = []
    for r in all_selected:
        balanced_records.append({
            "case_id": r.get("case_id", ""),
            "event_id": r.get("event_id", ""),
            "track_id": r.get("track_id", -1),
            "input_signature": r.get("input_signature", ""),
            "llm_label": r.get("llm_label", ""),
            "llm_match_score": r.get("llm_match_score", 0.0),
            "llm_mismatch_score": r.get("llm_mismatch_score", 0.0),
            "llm_uncertain_score": r.get("llm_uncertain_score", 0.0),
            "llm_confidence": r.get("llm_confidence", 0.0),
            "provider_mode": r.get("provider_mode", "simulate"),
            "model_name": r.get("model_name", "simulate"),
        })

    # ── 6. Output ───────────────────────────────────────────────
    # JSONL
    jsonl_path = OUT_DIR / "balanced_teacher_dataset_v2.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in balanced_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # CSV
    csv_path = OUT_DIR / "balanced_teacher_dataset_v2.csv"
    if balanced_records:
        fieldnames = list(balanced_records[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(balanced_records)

    # ── 7. Report ───────────────────────────────────────────────
    label_dist = Counter(r["llm_label"] for r in balanced_records)
    case_dist = Counter(r["case_id"] for r in balanced_records)
    hard_neg_info = []
    for sn in scored_negatives[:TARGET_MISMATCH]:
        hard_neg_info.append({
            "case_id": sn["record"].get("case_id", ""),
            "hardness_score": sn["hardness_score"],
            "verifier_label": sn["verifier_label"],
            "verifier_p_match": sn["verifier_p_match"],
            "teacher_mismatch_score": sn["teacher_mismatch_score"],
        })

    # Check if still at risk
    n_match = label_dist.get("match", 0)
    n_uncertain = label_dist.get("uncertain", 0)
    n_mismatch = label_dist.get("mismatch", 0)

    single_class_risk = (
        n_uncertain == 0 or n_match < 5
    )

    report = {
        "step": "3_balanced_dataset",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "raw_teacher_labels": len(all_teacher),
            "deduped": len(deduped_list),
            "dropped_by_dedup": len(all_teacher) - len(deduped_list),
            "dedup_explanation": "Duplicated input_signature = same (event_id, track_id, query_text, behavior_code). 126 of 864 records were duplicates. 95 signatures had >1 occurrence. Kept the record with lowest mismatch_score (or kept match label)."
        },
        "selection": {
            "match_all": n_match,
            "uncertain_all": n_uncertain,
            "mismatch_hard_negatives": n_mismatch,
            "total": n_match + n_uncertain + n_mismatch,
            "hard_neg_available": len(scored_negatives),
            "hard_neg_selected": n_mismatch,
        },
        "label_distribution": dict(label_dist),
        "case_coverage": dict(case_dist),
        "hard_negative_details": hard_neg_info,
        "single_class_risk": single_class_risk,
        "risk_note": "0 uncertain samples – this is a simulate teacher limitation. Not fixable without real LLM or cases with known event_types." if n_uncertain == 0 else "",
        "output_files": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
        },
    }

    report_path = REPORT_DIR / "step3_balanced_sampling_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Print ────────────────────────────────────────────────────
    print(f"=== Step 3: Balanced Hard-Case Dataset ===")
    print(f"Match (all): {n_match}")
    print(f"Uncertain (all): {n_uncertain}")
    print(f"Mismatch (hard negatives): {n_mismatch} / {len(scored_negatives)} available")
    print(f"Total: {len(balanced_records)}")
    print(f"Label distribution: {dict(label_dist)}")
    print(f"Case coverage: {dict(case_dist)}")
    print(f"Single-class risk: {single_class_risk}")
    print(f"[DONE] {jsonl_path} ({jsonl_path.stat().st_size} bytes)")
    print(f"[DONE] {csv_path} ({csv_path.stat().st_size} bytes)")
    print(f"[DONE] {report_path} ({report_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
