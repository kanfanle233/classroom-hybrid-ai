"""Build structured multimodal evidence for LLM teacher consumption.

Reads pipeline intermediate artifacts and produces one evidence record per
(query, candidate) pair — text/JSON only, no video frames.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, write_jsonl
from classroom_pipeline.verifier.model import build_feature_vector


# ── helpers ──────────────────────────────────────────────────────────────

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _load_uq_index(path: Optional[Path]) -> Dict[int, float]:
    if path is None or not path.exists():
        return {}
    rows = _load_jsonl(path)
    by_track: Dict[int, List[float]] = {}
    for row in rows:
        persons = row.get("persons")
        if isinstance(persons, list):
            for person in persons:
                if not isinstance(person, dict):
                    continue
                tid = person.get("track_id")
                uq = person.get("uq_track", person.get("uq_score"))
                if isinstance(tid, int) and isinstance(uq, (int, float)):
                    by_track.setdefault(tid, []).append(float(uq))
            continue
        tid = row.get("track_id")
        uq = row.get("uq_score", row.get("uq_track"))
        if isinstance(tid, int) and isinstance(uq, (int, float)):
            by_track.setdefault(tid, []).append(float(uq))
    return {tid: (sum(vals) / len(vals) if vals else 0.5) for tid, vals in by_track.items()}


def _compute_visual_score(overlap: float, action_confidence: float) -> float:
    """Replicate the visual_score computation from verifier/infer.py _predict_one."""
    return _clamp01(0.65 * overlap + 0.35 * action_confidence)


# ── main builder ─────────────────────────────────────────────────────────

def build_llm_evidence(
    *,
    event_queries_path: Path,
    aligned_path: Path,
    actions_path: Optional[Path] = None,
    pose_uq_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build one evidence record per (query, candidate) pair.

    Returns a list of dicts, each a self-contained evidence record for the LLM teacher.
    """
    queries = _load_jsonl(event_queries_path)
    q_index = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}

    aligned_obj = _load_json(aligned_path)
    aligned = aligned_obj if isinstance(aligned_obj, list) else []

    uq_index: Dict[int, float] = {}
    if pose_uq_path is not None:
        uq_index = _load_uq_index(pose_uq_path)

    # Optional: load actions.jsonl for fallback/extra context
    actions_map: Dict[str, Dict[str, Any]] = {}
    if actions_path is not None and actions_path.exists():
        for a in _load_jsonl(actions_path):
            tid = a.get("track_id")
            if isinstance(tid, int):
                key = f"track_{tid}"
                if key not in actions_map:
                    actions_map[key] = a

    evidence_list: List[Dict[str, Any]] = []

    for block in aligned:
        if not isinstance(block, dict):
            continue
        query_id = str(block.get("query_id", block.get("event_id", "")))
        query = q_index.get(query_id, {})
        event_type = str(query.get("event_type", block.get("event_type", "unknown")))
        query_text = str(query.get("query_text", block.get("query_text", "")))
        query_source = str(query.get("source", block.get("source", "unknown")))
        audio_confidence = _safe_float(query.get("confidence", 0.0), 0.0)

        window = block.get("window", {})
        if not isinstance(window, dict):
            window = {
                "start": _safe_float(block.get("window_start", 0.0), 0.0),
                "end": _safe_float(block.get("window_end", 0.0), 0.0),
            }
        w_start = _safe_float(window.get("start", 0.0), 0.0)
        w_end = _safe_float(window.get("end", max(w_start, 0.0)), max(w_start, 0.0))
        if w_end < w_start:
            w_start, w_end = w_end, w_start

        candidates = block.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        align_candidates_count = len(candidates)

        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            track_id = int(cand.get("track_id", -1))
            overlap = _safe_float(cand.get("overlap", 0.0), 0.0)
            action_conf = _safe_float(
                cand.get("action_confidence", cand.get("confidence", cand.get("conf", 0.0))), 0.0
            )
            uq = _safe_float(
                cand.get("uq_score", cand.get("uq_track", uq_index.get(track_id, 0.5))), 0.5
            )
            action_label = str(cand.get("semantic_id", cand.get("action", ""))).strip().lower()
            raw_action = str(cand.get("action", "")).strip().lower()
            behavior_code = str(cand.get("behavior_code", "")).strip().lower()
            behavior_label_zh = str(cand.get("behavior_label_zh", "")).strip()
            behavior_label_en = str(cand.get("behavior_label_en", "")).strip()
            semantic_label_zh = str(cand.get("semantic_label_zh", "")).strip()
            semantic_label_en = str(cand.get("semantic_label_en", "")).strip()

            # Build the 4-dim feature vector using the canonical builder
            feat = build_feature_vector(
                event_type=event_type,
                query_text=query_text,
                action_label=action_label,
                overlap=overlap,
                action_confidence=action_conf,
                uq_score=uq,
            )

            visual_score = _compute_visual_score(overlap, action_conf)
            text_score = feat[2]  # from action_match_score

            evidence_id = f"ev_{query_id}_track{track_id}"

            record: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "evidence_id": evidence_id,
                "query_id": query_id,
                "track_id": track_id,
                "query": {
                    "event_id": query_id,
                    "query_text": query_text,
                    "event_type": event_type,
                    "query_source": query_source,
                    "audio_confidence": audio_confidence,
                    "window_start": round(w_start, 4),
                    "window_end": round(w_end, 4),
                },
                "candidate": {
                    "track_id": track_id,
                    "action": action_label,
                    "raw_action": raw_action,
                    "behavior_code": behavior_code,
                    "behavior_label_zh": behavior_label_zh,
                    "behavior_label_en": behavior_label_en,
                    "semantic_label_zh": semantic_label_zh,
                    "semantic_label_en": semantic_label_en,
                    "start_time": _safe_float(cand.get("start_time", 0.0), 0.0),
                    "end_time": _safe_float(cand.get("end_time", 0.0), 0.0),
                    "overlap": overlap,
                    "action_confidence": action_conf,
                    "uq_score": uq,
                },
                "feature_vector": [round(v, 6) for v in feat],
                "derived_scores": {
                    "visual_score": round(visual_score, 6),
                    "text_score": round(text_score, 6),
                },
                "align_candidates_count": align_candidates_count,
            }
            evidence_list.append(record)

    return evidence_list


# ── smoke check ──────────────────────────────────────────────────────────

def smoke_check_evidence(
    evidence_list: List[Dict[str, Any]],
    aligned_path: Path,
) -> Dict[str, Any]:
    """Verify evidence completeness and feature correctness."""
    aligned_obj = _load_json(aligned_path)
    aligned = aligned_obj if isinstance(aligned_obj, list) else []

    expected_count = sum(
        len(block.get("candidates", []))
        for block in aligned
        if isinstance(block, dict)
    )

    actual_count = len(evidence_list)

    # Verify feature vectors match independent recomputation
    feat_errors = 0
    for ev in evidence_list:
        q = ev["query"]
        c = ev["candidate"]
        expected_feat = build_feature_vector(
            event_type=q["event_type"],
            query_text=q["query_text"],
            action_label=c["action"],
            overlap=c["overlap"],
            action_confidence=c["action_confidence"],
            uq_score=c["uq_score"],
        )
        actual = ev["feature_vector"]
        for i in range(4):
            if abs(actual[i] - expected_feat[i]) > 1e-5:
                feat_errors += 1
                break

    passed = (actual_count == expected_count) and (feat_errors == 0)

    return {
        "passed": passed,
        "expected_candidates": expected_count,
        "actual_evidence_records": actual_count,
        "feature_vector_mismatches": feat_errors,
        "detail": (
            f"OK: {actual_count}/{expected_count} records"
            if passed
            else f"MISMATCH: {actual_count} records vs {expected_count} expected, "
            f"{feat_errors} feature vector errors"
        ),
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Build LLM evidence from pipeline artifacts"
    )
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--aligned", required=True, type=str)
    parser.add_argument("--actions", default="", type=str)
    parser.add_argument("--pose_uq", default="", type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--smoke_report", default="", type=str,
                        help="output path for smoke check JSON")
    args = parser.parse_args()

    event_queries_path = Path(args.event_queries)
    aligned_path = Path(args.aligned)
    actions_path = Path(args.actions) if args.actions else None
    pose_uq_path = Path(args.pose_uq) if args.pose_uq else None
    out_path = Path(args.out)
    smoke_path = Path(args.smoke_report) if args.smoke_report else None

    ev = build_llm_evidence(
        event_queries_path=event_queries_path,
        aligned_path=aligned_path,
        actions_path=actions_path,
        pose_uq_path=pose_uq_path,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, ev)
    print(f"[DONE] LLM evidence: {out_path} ({len(ev)} records)")

    # Smoke check
    check = smoke_check_evidence(ev, aligned_path)
    if smoke_path:
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        with smoke_path.open("w", encoding="utf-8") as f:
            json.dump(check, f, ensure_ascii=False, indent=2)

    if not check["passed"]:
        print(f"[SMOKE FAIL] {check['detail']}")
        sys.exit(1)
    else:
        print(f"[SMOKE PASS] {check['detail']}")


if __name__ == "__main__":
    main()
