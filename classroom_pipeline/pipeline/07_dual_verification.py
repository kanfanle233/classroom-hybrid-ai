import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, validate_jsonl_file, validate_verified_event_record
from classroom_pipeline.verifier.infer import infer_verified_rows, resolve_llm_student_model_path


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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
                rows.append(obj)
    return rows


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    if inter <= 0:
        return 0.0
    denom = max(1e-6, min(a1 - a0, b1 - b0))
    return inter / denom


def _normalize_actions(rows: List[Dict[str, Any]], *, require_semantic: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    missing_semantic = 0
    for row in rows:
        tid = row.get("track_id")
        if not isinstance(tid, int):
            continue
        st = _safe_float(row.get("start_time", row.get("start", row.get("t", 0.0))), 0.0)
        ed = _safe_float(row.get("end_time", row.get("end", st + 0.2)), st + 0.2)
        if ed < st:
            st, ed = ed, st
        if ed <= st:
            ed = st + 0.2
        has_semantic = bool(str(row.get("semantic_id", "")).strip()) and bool(str(row.get("behavior_code", "")).strip())
        if require_semantic and not has_semantic:
            missing_semantic += 1
            continue
        action_name = str(row.get("semantic_id", row.get("action", row.get("label", "")))).strip().lower()
        out.append(
            {
                "track_id": tid,
                "action": action_name,
                "raw_action": str(row.get("action", row.get("label", ""))).strip().lower(),
                "behavior_code": str(row.get("behavior_code", "")).strip().lower(),
                "behavior_label_zh": str(row.get("behavior_label_zh", "")).strip(),
                "behavior_label_en": str(row.get("behavior_label_en", "")).strip(),
                "semantic_id": str(row.get("semantic_id", action_name)).strip().lower(),
                "semantic_label_zh": str(row.get("semantic_label_zh", "")).strip(),
                "semantic_label_en": str(row.get("semantic_label_en", "")).strip(),
                "taxonomy_version": str(row.get("taxonomy_version", "")).strip(),
                "start_time": st,
                "end_time": ed,
                "action_confidence": _safe_float(row.get("confidence", row.get("conf", 0.5)), 0.5),
            }
        )
    out.sort(key=lambda x: (x["start_time"], x["track_id"]))
    if require_semantic and missing_semantic > 0:
        raise ValueError(f"semantic fields missing on {missing_semantic} action rows")
    return out


def _load_uq_by_track(path: Path) -> Dict[int, float]:
    out: Dict[int, List[float]] = {}
    for row in _load_jsonl(path):
        if isinstance(row.get("persons"), list):
            for person in row["persons"]:
                if not isinstance(person, dict):
                    continue
                tid = person.get("track_id")
                uq = person.get("uq_track", person.get("uq_score"))
                if isinstance(tid, int) and isinstance(uq, (int, float)):
                    out.setdefault(tid, []).append(float(uq))
            continue
        tid = row.get("track_id")
        uq = row.get("uq_score", row.get("uq_track"))
        if isinstance(tid, int) and isinstance(uq, (int, float)):
            out.setdefault(tid, []).append(float(uq))
    return {tid: (sum(vals) / len(vals) if vals else 0.5) for tid, vals in out.items()}


def _build_aligned_fallback(
    *,
    queries: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    uq_by_track: Dict[int, float],
    default_window: float = 1.5,
) -> List[Dict[str, Any]]:
    aligned = []
    for q in queries:
        event_id = str(q.get("event_id", q.get("query_id", "")))
        event_type = str(q.get("event_type", "unknown"))
        query_text = str(q.get("query_text", ""))
        t_center = _safe_float(q.get("timestamp", q.get("t_center", q.get("start", 0.0))), 0.0)
        w_start = max(0.0, t_center - default_window)
        w_end = t_center + default_window
        candidates = []
        for a in actions:
            ov = _interval_overlap(w_start, w_end, a["start_time"], a["end_time"])
            if ov <= 0:
                continue
            tid = int(a["track_id"])
            candidates.append(
                {
                    "track_id": tid,
                    "action": a["action"],
                    "raw_action": a["raw_action"],
                    "behavior_code": a["behavior_code"],
                    "behavior_label_zh": a["behavior_label_zh"],
                    "behavior_label_en": a["behavior_label_en"],
                    "semantic_id": a["semantic_id"],
                    "semantic_label_zh": a["semantic_label_zh"],
                    "semantic_label_en": a["semantic_label_en"],
                    "taxonomy_version": a["taxonomy_version"],
                    "start_time": a["start_time"],
                    "end_time": a["end_time"],
                    "overlap": ov,
                    "action_confidence": a["action_confidence"],
                    "uq_track": float(uq_by_track.get(tid, 0.5)),
                    "uq_score": float(uq_by_track.get(tid, 0.5)),
                }
            )
        candidates.sort(key=lambda x: (x["overlap"], x["action_confidence"]), reverse=True)
        aligned.append(
            {
                "event_id": event_id,
                "query_id": event_id,
                "event_type": event_type,
                "query_text": query_text,
                "window_start": round(w_start, 3),
                "window_end": round(w_end, 3),
                "window_center": round(t_center, 3),
                "window_size": round(default_window, 3),
                "basis_motion": 0.0,
                "basis_uq": 0.0,
                "window": {
                    "start": round(w_start, 3),
                    "end": round(w_end, 3),
                    "center": round(t_center, 3),
                    "size": round(default_window, 3),
                },
                "motion_basis": 0.0,
                "uq_basis": 0.0,
                "candidates": candidates[:8],
            }
        )
    return aligned


def _write_per_person_compat(
    *,
    per_person_out: Path,
    actions: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    verified_rows: List[Dict[str, Any]],
) -> None:
    people: Dict[int, Dict[str, Any]] = {}
    for a in actions:
        tid = int(a["track_id"])
        people.setdefault(
            tid,
            {
                "track_id": tid,
                "person_id": tid,
                "visual_sequence": [],
                "speech_sequence": [],
                "verified_sequence": [],
            },
        )
        people[tid]["visual_sequence"].append(
            {
                "track_id": tid,
                "action": a["action"],
                "raw_action": a.get("raw_action", ""),
                "behavior_code": a.get("behavior_code", ""),
                "behavior_label_zh": a.get("behavior_label_zh", ""),
                "behavior_label_en": a.get("behavior_label_en", ""),
                "semantic_id": a.get("semantic_id", ""),
                "semantic_label_zh": a.get("semantic_label_zh", ""),
                "semantic_label_en": a.get("semantic_label_en", ""),
                "taxonomy_version": a.get("taxonomy_version", ""),
                "confidence": float(a["action_confidence"]),
                "start_time": float(a["start_time"]),
                "end_time": float(a["end_time"]),
            }
        )

    speech = []
    for q in queries:
        st = _safe_float(q.get("start", q.get("t_center", 0.0)), 0.0)
        ed = _safe_float(q.get("end", q.get("t_center", st + 0.2)), st + 0.2)
        speech.append(
            {
                "query_id": str(q.get("query_id", q.get("event_id", ""))),
                "event_type": str(q.get("event_type", "unknown")),
                "start": st,
                "end": ed,
                "text": str(q.get("trigger_text", q.get("query_text", ""))),
            }
        )
    for p in people.values():
        p["speech_sequence"] = speech

    for row in verified_rows:
        tid = int(row.get("track_id", -1))
        if tid >= 0 and tid in people:
            people[tid]["verified_sequence"].append(row)

    payload = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "total_people": len(people),
            "total_queries": len(queries),
            "total_verified": len(verified_rows),
            "note": "compat export generated from verified_events",
        },
        "speech_sequence": speech,
        "people": [people[k] for k in sorted(people.keys())],
    }
    per_person_out.parent.mkdir(parents=True, exist_ok=True)
    with per_person_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Step07 orchestration: query + align + verifier -> verified_events.jsonl")
    parser.add_argument("--actions", required=True, type=str, help="actions.jsonl")
    parser.add_argument("--event_queries", required=True, type=str, help="event_queries.jsonl")
    parser.add_argument("--pose_uq", required=True, type=str, help="pose_tracks_smooth_uq.jsonl")
    parser.add_argument("--aligned", default="", type=str, help="align_multimodal.json")
    parser.add_argument("--out", required=True, type=str, help="verified_events.jsonl")
    parser.add_argument("--verifier_model", default="", type=str, help="trained verifier checkpoint (.pt)")
    parser.add_argument("--verifier_config", default="", type=str, help="compat alias for --verifier_model")
    parser.add_argument("--llm_student_model", default="auto", type=str,
                        help="student judge .joblib model path; auto uses V4 default, off disables")
    parser.add_argument("--keep_all_candidates", type=int, default=0)
    parser.add_argument("--per_person_out", default="", type=str, help="optional compatibility export")
    parser.add_argument("--validate", type=int, default=1)
    parser.add_argument("--require_semantic", type=int, default=0)
    args = parser.parse_args()

    action_path = Path(args.actions)
    query_path = Path(args.event_queries)
    uq_path = Path(args.pose_uq)
    aligned_path = Path(args.aligned) if args.aligned else None
    out_path = Path(args.out)
    model_path = Path(args.verifier_model or args.verifier_config) if (args.verifier_model or args.verifier_config) else None
    llm_student_model_path = resolve_llm_student_model_path(args.llm_student_model)
    per_person_out = Path(args.per_person_out) if args.per_person_out else None

    if not action_path.is_absolute():
        action_path = (base_dir / action_path).resolve()
    if not query_path.is_absolute():
        query_path = (base_dir / query_path).resolve()
    if not uq_path.is_absolute():
        uq_path = (base_dir / uq_path).resolve()
    if aligned_path and (not aligned_path.is_absolute()):
        aligned_path = (base_dir / aligned_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    if model_path and (not model_path.is_absolute()):
        model_path = (base_dir / model_path).resolve()
    if per_person_out and (not per_person_out.is_absolute()):
        per_person_out = (base_dir / per_person_out).resolve()

    queries = _load_jsonl(query_path)
    q_index = {str(q.get("event_id", q.get("query_id", ""))): q for q in queries}
    actions = _normalize_actions(_load_jsonl(action_path), require_semantic=bool(int(args.require_semantic)))
    uq_by_track = _load_uq_by_track(uq_path)

    # Ensure aligned input exists. If not provided, generate fallback alignment.
    if aligned_path is None or (not aligned_path.exists()):
        fallback_aligned = _build_aligned_fallback(
            queries=queries,
            actions=actions,
            uq_by_track=uq_by_track,
        )
        aligned_path = out_path.with_name("align_multimodal.fallback.json")
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        with aligned_path.open("w", encoding="utf-8") as f:
            json.dump(fallback_aligned, f, ensure_ascii=False, indent=2)
        print(f"[INFO] aligned file missing, fallback generated: {aligned_path}")

    raw_rows = infer_verified_rows(
        event_queries_path=query_path,
        aligned_path=aligned_path,
        pose_uq_path=uq_path,
        model_path=model_path,
        keep_all_candidates=bool(int(args.keep_all_candidates)),
        llm_student_model_path=llm_student_model_path,
    )

    model_version = f"verifier:{model_path.stem}" if (model_path and model_path.exists()) else "heuristic_v1"
    verified_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        event_id = str(row.get("event_id", row.get("query_id", "")))
        query_row = q_index.get(event_id, {})
        query_time = _safe_float(
            query_row.get("timestamp", query_row.get("t_center", query_row.get("start", 0.0))),
            _safe_float(row.get("window_start", row.get("window", {}).get("center", 0.0)), 0.0),
        )
        window_obj = row.get("window", {})
        window_start = _safe_float(row.get("window_start", window_obj.get("start", 0.0)), 0.0)
        window_end = _safe_float(row.get("window_end", window_obj.get("end", window_start)), window_start)
        reliability = _safe_float(row.get("reliability_score", 0.0), 0.0)
        uncertainty = _safe_float(row.get("uncertainty", 1.0 - reliability), 1.0 - reliability)
        threshold_source = str(
            row.get(
                "threshold_source",
                "model_runtime_config" if (model_path and model_path.exists()) else "heuristic_default",
            )
        )
        runtime_cfg = row.get("runtime_config", {})
        if not isinstance(runtime_cfg, dict):
            runtime_cfg = {}
        label = str(row.get("label", row.get("match_label", "mismatch")))
        evidence = row.get("evidence", {})
        out_row = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "query_id": event_id,
            "track_id": int(row.get("track_id", -1)),
            "event_type": str(row.get("event_type", query_row.get("event_type", "unknown"))),
            "query_text": str(row.get("query_text", query_row.get("query_text", ""))),
            "query_time": query_time,
            "window_start": window_start,
            "window_end": window_end,
            "window": {"start": window_start, "end": window_end},
            "p_match": _safe_float(row.get("p_match", 0.0), 0.0),
            "p_mismatch": _safe_float(row.get("p_mismatch", 1.0), 1.0),
            "reliability_score": reliability,
            "uncertainty": max(0.0, min(1.0, uncertainty)),
            "label": label,
            "match_label": label,
            "threshold_source": threshold_source,
            "model_version": model_version,
            "thresholds": runtime_cfg,
            "action": str(row.get("action", "")),
            "raw_action": str(row.get("raw_action", "")),
            "behavior_code": str(row.get("behavior_code", "")),
            "behavior_label_zh": str(row.get("behavior_label_zh", "")),
            "behavior_label_en": str(row.get("behavior_label_en", "")),
            "semantic_id": str(row.get("semantic_id", "")),
            "semantic_label_zh": str(row.get("semantic_label_zh", "")),
            "semantic_label_en": str(row.get("semantic_label_en", "")),
            "taxonomy_version": str(row.get("taxonomy_version", "")),
            "evidence": {k: v for k, v in evidence.items() if v is not None},
        }
        verified_rows.append(out_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in verified_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if int(args.validate) == 1:
        ok, _, errors = validate_jsonl_file(out_path, validate_verified_event_record)
        if not ok:
            first_error = errors[0] if errors else "unknown schema error"
            raise ValueError(f"invalid verified event schema: {first_error}")

    if per_person_out is not None:
        _write_per_person_compat(
            per_person_out=per_person_out,
            actions=actions,
            queries=queries,
            verified_rows=verified_rows,
        )
        print(f"[DONE] compat per_person: {per_person_out}")

    print(f"[DONE] verified events: {out_path}")
    print(f"[INFO] rows: {len(verified_rows)}")


if __name__ == "__main__":
    main()
