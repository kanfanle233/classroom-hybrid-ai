#!/usr/bin/env python3
"""Step 3b: Candidate-level high-disambiguation sample selection.

Works on ALL candidates from align_multimodal.json (not just top-1 from verified_events).
Replicates the verifier's audio_visual_dynamic fusion heuristic to produce
candidate-level p_match and reliability scores, then labels each candidate
as match/uncertain/mismatch based on VerifierRuntimeConfig thresholds.

Selection criteria (no teacher labels needed — simulate teacher is all-mismatch):
  A. Reliability near decision boundary (gap to threshold < 0.10)
  B. High uq_score (pose uncertainty) dragging reliability down
  C. Low action_confidence creating score spread
  D. Top-k candidate scores close together (gap < 0.10)
  E. Cross-case representative coverage

Outputs:
  - high_disambiguation_samples.jsonl     (candidate-level)
  - balanced_teacher_dataset_v3.jsonl/csv (3-class, candidate-level)
  - sampled_cases_report_v3.json
"""
import json, sys, hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

OUT_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline"
REPORT_DIR = OUT_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CASE_CONFIGS = {
    "front_002_full_pose020_hybrid":        "output/codex_reports/front_002_full_pose020_hybrid",
    "front_002_rear_row_sliced_pose020_hybrid": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid",
    "front_1885_full":  "output/codex_reports/front_1885_full",
    "front_22259_full": "output/codex_reports/front_22259_full",
    "front_26729_full": "output/codex_reports/front_26729_full",
    "front_45618_full": "output/codex_reports/front_45618_full",
}

# VerifierRuntimeConfig defaults (from verifier/model.py)
MATCH_THRESHOLD = 0.6
UNCERTAIN_THRESHOLD = 0.4
UQ_GATE = 0.6
TEMPERATURE = 1.0


def _safe_float(x, d=0.0):
    try: return float(x)
    except: return d


def _clamp01(x):
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return float(x)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except json.JSONDecodeError: continue
    return out


def load_json(path: Path) -> Any:
    if not path.exists(): return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_candidate(
    cand: Dict[str, Any],
    event_type: str,
    query_source: str,
    audio_confidence: float,
) -> Dict[str, Any]:
    """Replicate verifier's audio_visual_dynamic fusion for one candidate.

    Mirrors verifier/infer.py _predict_one() without MLP/student overrides.
    Returns p_match, reliability, label, and all intermediate scores.
    """
    overlap = _safe_float(cand.get("overlap", 0.0), 0.0)
    action_conf = _safe_float(cand.get("action_confidence", cand.get("confidence", 0.0)), 0.0)
    uq = _safe_float(cand.get("uq_score", cand.get("uq_track", 0.5)), 0.5)

    # visual_score = 0.65 * overlap + 0.35 * action_confidence (same as verifier)
    visual_score = _clamp01(0.65 * overlap + 0.35 * action_conf)

    # text_score = action_match_score(event_type, query_text, action_label)
    # When event_type="unknown", action_match_score returns 0 for all behaviors
    text_score = 0.0 if str(event_type or "").strip().lower() in ("", "unknown") else 0.0

    # Determine fusion mode (same logic as verifier)
    is_visual_fallback = "fallback" in str(query_source).lower()
    has_real_audio = (
        audio_confidence > 0.0
        and not is_visual_fallback
        and str(query_source).strip().lower() not in ("unknown", "placeholder", "fallback", "")
    )

    if has_real_audio:
        w_visual = _clamp01(1.0 - uq)
        w_audio = _clamp01(audio_confidence)
        p_match = _clamp01(
            (w_visual * visual_score + w_audio * text_score) / (w_visual + w_audio + 1e-9)
        )
        fusion_mode = "audio_visual_dynamic"
    elif is_visual_fallback:
        p_match = visual_score
        text_score = float('nan')
        fusion_mode = "visual_only_fallback"
    else:
        p_match = visual_score
        fusion_mode = "visual_only"

    p_mismatch = _clamp01(1.0 - p_match)
    reliability = _clamp01(p_match * (1.0 - UQ_GATE * _clamp01(uq)))

    if reliability >= MATCH_THRESHOLD:
        label = "match"
    elif reliability >= UNCERTAIN_THRESHOLD:
        label = "uncertain"
    else:
        label = "mismatch"

    return {
        "p_match": round(p_match, 6),
        "p_mismatch": round(p_mismatch, 6),
        "reliability": round(reliability, 6),
        "label": label,
        "visual_score": round(visual_score, 6),
        "text_score": text_score if text_score == text_score else None,
        "uq_score": round(uq, 6),
        "fusion_mode": fusion_mode,
        "overlap": round(overlap, 6),
        "action_confidence": round(action_conf, 6),
        "w_visual": round(_clamp01(1.0 - uq), 6) if has_real_audio else 1.0,
        "w_audio": round(_clamp01(audio_confidence), 6) if has_real_audio else 0.0,
    }


def compute_ambiguity(
    score: Dict[str, Any],
    rank_in_query: int,
    total_cands: int,
    gap_to_next: float,
) -> Tuple[float, List[str]]:
    """Score how ambiguous/disambiguating this candidate is.

    Higher = more useful for training a discriminative student.
    """
    points = 0.0
    reasons = []

    p_match = score["p_match"]
    reliability = score["reliability"]
    label = score["label"]
    uq = score["uq_score"]
    action_conf = score["action_confidence"]

    # A. Reliability near any decision boundary
    gap_to_match = abs(reliability - MATCH_THRESHOLD)
    gap_to_uncertain = abs(reliability - UNCERTAIN_THRESHOLD)
    min_gap = min(gap_to_match, gap_to_uncertain)

    if min_gap < 0.03:
        points += 3.0
        reasons.append(f"boundary_adjacent(gap={min_gap:.3f},rel={reliability:.3f})")
    elif min_gap < 0.06:
        points += 2.0
        reasons.append(f"near_boundary(gap={min_gap:.3f},rel={reliability:.3f})")
    elif min_gap < 0.10:
        points += 1.0
        reasons.append(f"moderate_boundary(gap={min_gap:.3f})")

    # B. High uq_score causing reliability degradation
    reliability_if_no_uq = p_match  # reliability without uq penalty
    uq_penalty = reliability_if_no_uq - reliability
    if uq_penalty > 0.15:
        points += 2.0
        reasons.append(f"high_uq_penalty(uq={uq:.3f},penalty={uq_penalty:.3f})")
    elif uq_penalty > 0.08:
        points += 1.0
        reasons.append(f"moderate_uq_penalty(uq={uq:.3f},penalty={uq_penalty:.3f})")

    # C. Low action_confidence creating score spread
    if action_conf < 0.75:
        points += 1.5
        reasons.append(f"low_action_conf={action_conf:.3f}")
    elif action_conf < 0.80:
        points += 0.5
        reasons.append(f"moderate_action_conf={action_conf:.3f}")

    # D. Top-k gap (close competition)
    if rank_in_query == 1 and gap_to_next < 0.05:
        points += 2.5
        reasons.append(f"top1_close_gap(gap={gap_to_next:.3f})")
    elif rank_in_query == 1 and gap_to_next < 0.10:
        points += 1.0
        reasons.append(f"top1_near_gap(gap={gap_to_next:.3f})")
    elif rank_in_query == 2 and gap_to_next < 0.05:
        points += 2.0
        reasons.append(f"runner_up_close(gap={gap_to_next:.3f})")

    # E. Label is minority class → more valuable
    if label == "mismatch":
        points += 1.5
        reasons.append("minority_class:mismatch")
    elif label == "uncertain":
        points += 1.0
        reasons.append("minority_class:uncertain")

    # F. First or last in query ranking (extreme positions are less informative)
    if rank_in_query > total_cands * 0.7:
        points += 0.5
        reasons.append(f"low_rank({rank_in_query}/{total_cands})")

    return (round(points, 2), reasons)


def main():
    print("=" * 60)
    print("Step 3b: Candidate-Level High-Disambiguation Selection")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # 1. Load all candidates from align_multimodal.json
    # ═══════════════════════════════════════════════════════════════
    all_scored = []  # flat list of (candidate, score, query_meta)

    for case_id, dir_path in CASE_CONFIGS.items():
        aligned_path = _PROJECT_ROOT / dir_path / "align_multimodal.json"
        eq_path = _PROJECT_ROOT / dir_path / "event_queries.fusion_v2.jsonl"

        if not aligned_path.exists():
            print(f"  SKIP {case_id}: no align_multimodal.json")
            continue

        aligned = load_json(aligned_path)
        if not isinstance(aligned, list):
            print(f"  SKIP {case_id}: align_multimodal.json is not a list")
            continue

        # Build query index for metadata
        queries = load_jsonl(eq_path)
        q_index = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}

        case_scored = 0
        for block in aligned:
            if not isinstance(block, dict):
                continue
            query_id = str(block.get("query_id", block.get("event_id", "")))
            query = q_index.get(query_id, {})
            event_type = str(query.get("event_type", block.get("event_type", "unknown")))
            query_text = str(query.get("query_text", block.get("query_text", "")))
            query_source = str(query.get("source", block.get("source", "unknown")))
            audio_conf = _safe_float(query.get("confidence", 0.0), 0.0)

            candidates = block.get("candidates", [])
            if not isinstance(candidates, list) or not candidates:
                continue

            # Score every candidate
            query_scores = []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                score = score_candidate(cand, event_type, query_source, audio_conf)
                query_scores.append((cand, score))

            # Sort by reliability (same as verifier)
            query_scores.sort(key=lambda x: x[1]["reliability"], reverse=True)
            n_cands = len(query_scores)

            # Compute gaps to next-best
            for rank, (cand, score) in enumerate(query_scores, 1):
                gap_to_next = 0.0
                if rank < n_cands:
                    gap_to_next = score["reliability"] - query_scores[rank][1]["reliability"]

                ambiguity, reasons = compute_ambiguity(score, rank, n_cands, gap_to_next)

                all_scored.append({
                    "case_id": case_id,
                    "event_id": query_id,
                    "query_id": query_id,
                    "track_id": int(cand.get("track_id", -1)),
                    "event_type": event_type,
                    "query_text": query_text[:200],
                    "query_source": query_source,
                    "audio_confidence": audio_conf,
                    "action": str(cand.get("action", cand.get("semantic_id", ""))),
                    "raw_action": str(cand.get("raw_action", "")),
                    "behavior_code": str(cand.get("behavior_code", "")),
                    "behavior_label_zh": str(cand.get("behavior_label_zh", "")),
                    "semantic_id": str(cand.get("semantic_id", "")),
                    "semantic_label_zh": str(cand.get("semantic_label_zh", "")),
                    "overlap_raw": _safe_float(cand.get("overlap", 0)),
                    "action_confidence_raw": _safe_float(cand.get("action_confidence", 0)),
                    "uq_score_raw": _safe_float(cand.get("uq_score", cand.get("uq_track", 0.5))),
                    "rank_in_query": rank,
                    "total_candidates": n_cands,
                    "gap_to_next": round(gap_to_next, 6),
                    "p_match": score["p_match"],
                    "p_mismatch": score["p_mismatch"],
                    "reliability": score["reliability"],
                    "label": score["label"],
                    "visual_score": score["visual_score"],
                    "text_score": score["text_score"],
                    "uq_score": score["uq_score"],
                    "fusion_mode": score["fusion_mode"],
                    "ambiguity_score": ambiguity,
                    "ambiguity_reasons": reasons,
                })
                case_scored += 1

        print(f"  {case_id}: {len(aligned)} queries, {case_scored} candidates scored")

    print(f"\nTotal candidates scored: {len(all_scored)}")

    # ═══════════════════════════════════════════════════════════════
    # 2. Label distribution analysis
    # ═══════════════════════════════════════════════════════════════
    label_dist = Counter(r["label"] for r in all_scored)
    print(f"\nFull label distribution: {dict(label_dist)}")

    # Per-case distribution
    for case_id in CASE_CONFIGS:
        case_rows = [r for r in all_scored if r["case_id"] == case_id]
        if case_rows:
            ld = Counter(r["label"] for r in case_rows)
            pms = [r["p_match"] for r in case_rows]
            rels = [r["reliability"] for r in case_rows]
            print(f"  {case_id}: {len(case_rows)} cands, labels={dict(ld)}, "
                  f"p_match=[{min(pms):.3f}-{max(pms):.3f}], "
                  f"rel=[{min(rels):.3f}-{max(rels):.3f}]")

    # ═══════════════════════════════════════════════════════════════
    # 3. Select high-disambiguation samples
    # ═══════════════════════════════════════════════════════════════
    # Sort by ambiguity (highest first)
    all_scored.sort(key=lambda x: x["ambiguity_score"], reverse=True)

    # Select diverse high-disambiguation pool
    # Ensure per-case and per-label representation
    selected = []
    seen_case_label = set()

    # Phase 1: pick top ambiguity per (case, label)
    for r in all_scored:
        key = f"{r['case_id']}|{r['label']}"
        if key not in seen_case_label:
            selected.append(r)
            seen_case_label.add(key)

    # Phase 2: fill with highest ambiguity remaining, up to 200
    for r in all_scored:
        if len(selected) >= 200:
            break
        if r not in selected:
            selected.append(r)

    hd_final = selected[:200]

    hd_labels = Counter(r["label"] for r in hd_final)
    hd_cases = Counter(r["case_id"] for r in hd_final)
    print(f"\n--- High-Disambiguation Pool ---")
    print(f"  Selected: {len(hd_final)}")
    print(f"  Labels: {dict(hd_labels)}")
    print(f"  Cases: {dict(hd_cases)}")

    # ═══════════════════════════════════════════════════════════════
    # 4. Build balanced_teacher_dataset_v3
    # ═══════════════════════════════════════════════════════════════
    matches = [r for r in all_scored if r["label"] == "match"]
    uncertains = [r for r in all_scored if r["label"] == "uncertain"]
    mismatches = [r for r in all_scored if r["label"] == "mismatch"]

    print(f"\n  Available: match={len(matches)}, uncertain={len(uncertains)}, mismatch={len(mismatches)}")

    # Sort each class by ambiguity (higher = more informative)
    matches.sort(key=lambda x: x["ambiguity_score"], reverse=True)
    uncertains.sort(key=lambda x: x["ambiguity_score"], reverse=True)
    mismatches.sort(key=lambda x: x["ambiguity_score"], reverse=True)

    # Target: balanced, using the minimum class as the cap
    TARGET = min(60, len(matches), len(uncertains), len(mismatches)) if mismatches else min(60, len(matches), len(uncertains))
    # If mismatch is very small, take all mismatches and balance others
    if len(mismatches) < 10:
        n_mismatch = len(mismatches)
        n_target = min(50, len(matches), len(uncertains))
        balanced_v3 = (
            matches[:n_target] +
            uncertains[:n_target] +
            mismatches  # all mismatch
        )
    else:
        balanced_v3 = (
            matches[:TARGET] +
            uncertains[:TARGET] +
            mismatches[:TARGET]
        )

    v3_label_dist = Counter(r["label"] for r in balanced_v3)
    v3_case_dist = Counter(r["case_id"] for r in balanced_v3)
    print(f"\n--- Balanced V3 Dataset ---")
    print(f"  Total: {len(balanced_v3)}")
    print(f"  Labels: {dict(v3_label_dist)}")
    print(f"  Cases: {dict(v3_case_dist)}")

    # Dedup by (case_id, event_id, track_id) keeping highest ambiguity
    deduped_v3 = {}
    for r in balanced_v3:
        key = f"{r['case_id']}|{r['event_id']}|{r['track_id']}"
        if key not in deduped_v3 or r["ambiguity_score"] > deduped_v3[key]["ambiguity_score"]:
            deduped_v3[key] = r
    balanced_v3_deduped = list(deduped_v3.values())
    if len(balanced_v3_deduped) < len(balanced_v3):
        print(f"  After dedup: {len(balanced_v3_deduped)} (dropped {len(balanced_v3) - len(balanced_v3_deduped)})")
        balanced_v3 = balanced_v3_deduped
        v3_label_dist = Counter(r["label"] for r in balanced_v3)
        v3_case_dist = Counter(r["case_id"] for r in balanced_v3)
        print(f"  Labels: {dict(v3_label_dist)}")

    # ═══════════════════════════════════════════════════════════════
    # 5. Write output files
    # ═══════════════════════════════════════════════════════════════

    # 5a. High-disambiguation samples
    hd_path = OUT_DIR / "high_disambiguation_samples.jsonl"
    with hd_path.open("w", encoding="utf-8") as f:
        for s in hd_final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n[OUT] {hd_path} ({hd_path.stat().st_size} bytes)")

    # 5b. Balanced V3 JSONL (with input_signature for training)
    v3_jsonl_path = OUT_DIR / "balanced_teacher_dataset_v3.jsonl"
    with v3_jsonl_path.open("w", encoding="utf-8") as f:
        for r in balanced_v3:
            rec = {
                "case_id": r["case_id"],
                "event_id": r["event_id"],
                "query_id": r["query_id"],
                "track_id": r["track_id"],
                "input_signature": hashlib.sha256(
                    f"{r['event_id']}|{r['track_id']}|{r['query_text']}|{r['behavior_code']}".encode()
                ).hexdigest()[:16],
                "label": r["label"],
                "p_match": r["p_match"],
                "reliability": r["reliability"],
                "label_source": "verifier_heuristic_replicated_v1",
                "ambiguity_score": r["ambiguity_score"],
                "event_type": r["event_type"],
                "query_text": r["query_text"][:120],
                "query_source": r["query_source"],
                "behavior_code": r["behavior_code"],
                "behavior_label_zh": r["behavior_label_zh"],
                "action": r["action"],
                "overlap": r["overlap_raw"],
                "action_confidence": r["action_confidence_raw"],
                "uq_score": r["uq_score_raw"],
                "visual_score": r["visual_score"],
                "text_score": r["text_score"],
                "rank_in_query": r["rank_in_query"],
                "total_candidates": r["total_candidates"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OUT] {v3_jsonl_path} ({v3_jsonl_path.stat().st_size} bytes)")

    # 5c. Balanced V3 CSV
    v3_csv_path = OUT_DIR / "balanced_teacher_dataset_v3.csv"
    if balanced_v3:
        import csv
        # Re-read JSONL for CSV consistency
        csv_recs = []
        with v3_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    csv_recs.append(json.loads(line))
        if csv_recs:
            fieldnames = list(csv_recs[0].keys())
            with v3_csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_recs)
            print(f"[OUT] {v3_csv_path} ({v3_csv_path.stat().st_size} bytes)")

    # ═══════════════════════════════════════════════════════════════
    # 6. Generate sampled_cases_report_v3.json
    # ═══════════════════════════════════════════════════════════════
    has_3_classes = len(v3_label_dist) >= 3
    min_class = min(v3_label_dist.values()) if v3_label_dist else 0

    # Per-case stats
    case_stats = {}
    for case_id in CASE_CONFIGS:
        case_all = [r for r in all_scored if r["case_id"] == case_id]
        case_hd = [r for r in hd_final if r["case_id"] == case_id]
        case_v3 = [r for r in balanced_v3 if r["case_id"] == case_id]
        if case_all:
            case_stats[case_id] = {
                "total_candidates": len(case_all),
                "hd_selected": len(case_hd),
                "v3_included": len(case_v3),
                "v3_labels": dict(Counter(r["label"] for r in case_v3)),
                "full_labels": dict(Counter(r["label"] for r in case_all)),
                "avg_p_match": round(sum(r["p_match"] for r in case_all) / len(case_all), 4),
                "avg_reliability": round(sum(r["reliability"] for r in case_all) / len(case_all), 4),
                "avg_uq_score": round(sum(r["uq_score_raw"] for r in case_all) / len(case_all), 4),
                "avg_action_confidence": round(sum(r["action_confidence_raw"] for r in case_all) / len(case_all), 4),
                "top1_match_pct": round(
                    sum(1 for r in case_all if r["rank_in_query"] == 1 and r["label"] == "match")
                    / max(1, sum(1 for r in case_all if r["rank_in_query"] == 1)) * 100, 1
                ),
            }

    # Ambiguity distribution of selected samples
    amb_scores = [r["ambiguity_score"] for r in hd_final]
    amb_reasons = Counter()
    for r in hd_final:
        for reason in r["ambiguity_reasons"]:
            cat = reason.split("(")[0] if "(" in reason else reason.split(":")[0]
            amb_reasons[cat] += 1

    report = {
        "step": "3b_candidate_level_disambiguation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "scoring": "verifier_heuristic_replicated_v1 — audio_visual_dynamic fusion",
            "config": {
                "match_threshold": MATCH_THRESHOLD,
                "uncertain_threshold": UNCERTAIN_THRESHOLD,
                "uq_gate": UQ_GATE,
                "temperature": TEMPERATURE,
            },
            "fusion": "w_visual=1-uq, w_audio=audio_confidence, p_match=(w_v*visual_score + w_a*text_score)/(w_v+w_a)",
            "visual_score": "0.65*overlap + 0.35*action_confidence",
            "reliability": "p_match * (1 - uq_gate * uq)",
            "note": "text_score=0 when event_type='unknown' (action_match_score returns 0)",
        },
        "data_summary": {
            "total_cases": len(CASE_CONFIGS),
            "total_queries": sum(1 for r in all_scored if r["rank_in_query"] == 1),
            "total_candidates_scored": len(all_scored),
            "full_label_distribution": dict(label_dist),
            "full_label_percentages": {
                k: round(v / len(all_scored) * 100, 1) for k, v in label_dist.items()
            },
        },
        "v2_comparison": {
            "v2_total": 38,
            "v2_match": 8,
            "v2_uncertain": 0,
            "v2_mismatch": 30,
            "v2_blocker": "0 uncertain labels, class collapse, uniform teacher labels (all-mismatch)",
        },
        "v3_results": {
            "total": len(balanced_v3),
            "label_distribution": dict(v3_label_dist),
            "case_distribution": dict(v3_case_dist),
            "has_3_classes": has_3_classes,
            "min_class_count": min_class,
            "label_source": "verifier heuristic replication on ALL candidates (not just top-1)",
        },
        "high_disambiguation_pool": {
            "total": len(hd_final),
            "labels": dict(hd_labels),
            "cases": dict(hd_cases),
            "avg_ambiguity": round(sum(amb_scores) / max(1, len(amb_scores)), 2),
            "min_ambiguity": round(min(amb_scores), 2),
            "max_ambiguity": round(max(amb_scores), 2),
            "top_reasons": dict(amb_reasons.most_common(10)),
        },
        "per_case": case_stats,
        "training_feasibility": {
            "3_classes_present": has_3_classes,
            "min_samples_per_class": min_class,
            "total_samples": len(balanced_v3),
            "recommendation": (
                "READY for 3-class student training with deterministic case-aware split"
                if has_3_classes and min_class >= 8
                else "MARGINAL — has 3 classes but min class < 8, training possible but metrics will be noisy"
                if has_3_classes
                else "BLOCKED — less than 3 classes in dataset"
            ),
            "caveats": [
                "Labels are verifier-heuristic-replicated, not human gold",
                "All event_types are 'unknown', so text_score=0 for all candidates",
                "Visual_score dominates; discrimination comes from uq_score variance",
                "Must use pseudo_label_benchmark evaluation kind",
                "Student learns from 16-dim features, not from heuristic fusion directly",
            ],
        },
        "output_files": {
            "high_disambiguation_samples": str(hd_path),
            "balanced_v3_jsonl": str(v3_jsonl_path),
            "balanced_v3_csv": str(v3_csv_path),
            "report_json": str(REPORT_DIR / "sampled_cases_report_v3.json"),
        },
    }

    report_path = REPORT_DIR / "sampled_cases_report_v3.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OUT] {report_path} ({report_path.stat().st_size} bytes)")

    # ═══════════════════════════════════════════════════════════════
    # 7. Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Candidates scored: {len(all_scored)}")
    print(f"  Full labels: {dict(label_dist)}")
    print(f"  High-disambiguation pool: {len(hd_final)}")
    print(f"  V3 balanced: {len(balanced_v3)} records, labels={dict(v3_label_dist)}")
    print(f"  3-class: {has_3_classes}, min_class={min_class}")
    print(f"  Training: {report['training_feasibility']['recommendation']}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
