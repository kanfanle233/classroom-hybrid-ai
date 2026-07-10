import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import (
    SCHEMA_VERSION,
    validate_jsonl_file,
    validate_verified_event_v3_record,
    write_jsonl,
)
from classroom_pipeline.verifier.model import action_match_score


COHERENCE_SCHEMA_VERSION = f"{SCHEMA_VERSION}+coherence_v3"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        x = float(value)
        if not math.isfinite(x):
            return default
        return x
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _load_json(path: Optional[Path]) -> Any:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path is None or not path.exists():
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


def _event_id(row: Dict[str, Any]) -> str:
    return str(row.get("event_id") or row.get("query_id") or "")


def _index_by_event(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        eid = _event_id(row)
        if eid and eid not in out:
            out[eid] = row
    return out


def _query_source(query: Dict[str, Any], verified: Dict[str, Any]) -> str:
    return str(query.get("source") or verified.get("query_source") or verified.get("source") or "unknown").strip().lower()


def _asr_ok(asr_report: Any) -> bool:
    if not isinstance(asr_report, dict):
        return False
    return str(asr_report.get("status", "")).strip().lower() == "ok" and int(asr_report.get("segments_good", 0) or 0) > 0


def _selected_candidate(align: Dict[str, Any], track_id: int) -> Tuple[Optional[Dict[str, Any]], int, int]:
    candidates = align.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return None, 0, 0
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        overlap = _safe_float(cand.get("overlap"), 0.0)
        action_conf = _safe_float(cand.get("action_confidence"), 0.0)
        scored.append((0.65 * overlap + 0.35 * action_conf, cand))
    scored.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, cand) in enumerate(scored, start=1):
        if int(_safe_float(cand.get("track_id"), -1)) == track_id:
            return cand, rank, len(scored)
    return scored[0][1], 1, len(scored)


def _object_support(
    objects_by_frame: Dict[int, List[Dict[str, Any]]],
    *,
    action: str,
    start: float,
    end: float,
    fps: float,
) -> Optional[float]:
    if not objects_by_frame:
        return None
    action = str(action or "").strip().lower()
    if not action:
        return None
    start_frame = max(0, int(math.floor(start * fps)))
    end_frame = max(start_frame, int(math.ceil(end * fps)))
    supports: List[float] = []
    for frame in range(start_frame, end_frame + 1):
        for obj in objects_by_frame.get(frame, []):
            support = str(obj.get("semantic_support") or obj.get("support") or "").strip().lower()
            conf = _safe_float(obj.get("conf", obj.get("confidence", 0.0)), 0.0)
            if support == action:
                supports.append(_clamp01(conf))
    if not supports:
        return None
    return max(supports)


def _load_objects_by_frame(path: Optional[Path]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in _load_jsonl(path):
        frame = row.get("frame")
        if not isinstance(frame, int):
            frame = int(_safe_float(row.get("frame_idx"), -1))
        if frame < 0:
            continue
        objects = row.get("objects")
        if isinstance(objects, list):
            out.setdefault(frame, []).extend([obj for obj in objects if isinstance(obj, dict)])
    return out


def _nullable_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return _clamp01(x)


def _coherence_score(scores: List[Tuple[float, float]], align_score: float) -> Tuple[bool, Optional[float]]:
    usable = [(s, w) for s, w in scores if w > 0.0]
    if len(usable) < 2:
        return False, None
    numerator = 0.0
    denominator = 0.0
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            si, wi = usable[i]
            sj, wj = usable[j]
            pair_w = wi * wj
            numerator += pair_w * abs(si - sj)
            denominator += pair_w
    agreement = 1.0 - numerator / (denominator + 1e-9)
    return True, round(_clamp01(agreement * align_score), 6)


def _conflict_type(
    *,
    channel_valid: Dict[str, bool],
    scores: Dict[str, Optional[float]],
    align_score: float,
    coherence_valid: bool,
    coherence_score: Optional[float],
    reliability: float,
    quality_flags: List[str],
) -> str:
    severe_flags = {"high_uq", "no_candidates", "low_asr_quality"}
    valid_semantic = [name for name in ("visual", "text", "object") if channel_valid.get(name)]
    if reliability < 0.4 or any(flag in severe_flags for flag in quality_flags) or not valid_semantic:
        return "low_evidence"
    if align_score < 0.4 and any((scores.get(name) or 0.0) >= 0.6 for name in valid_semantic):
        return "temporal_mismatch"
    vals = [scores[name] for name in valid_semantic if scores.get(name) is not None]
    if vals and max(vals) >= 0.6 and min(vals) < 0.4:
        return "semantic_mismatch"
    if valid_semantic == ["visual"] and (scores.get("visual") or 0.0) >= 0.6:
        return "visual_only"
    if valid_semantic == ["text"] and (scores.get("text") or 0.0) >= 0.6:
        return "text_only"
    if len(valid_semantic) < 2:
        return "low_evidence"
    if coherence_valid and coherence_score is not None and coherence_score >= 0.6:
        return "none"
    return "low_evidence"


def diagnose_rows(
    *,
    verified_rows: List[Dict[str, Any]],
    align_rows: List[Dict[str, Any]],
    query_rows: List[Dict[str, Any]],
    asr_report: Any,
    objects_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    case_id: str = "",
    fps: float = 25.0,
) -> List[Dict[str, Any]]:
    align_index = _index_by_event(align_rows)
    query_index = _index_by_event(query_rows)
    objects_by_frame = objects_by_frame or {}
    asr_available = _asr_ok(asr_report)

    out: List[Dict[str, Any]] = []
    for row in verified_rows:
        new_row = dict(row)
        new_row["schema_version"] = COHERENCE_SCHEMA_VERSION
        eid = _event_id(row)
        track_id = int(_safe_float(row.get("track_id"), -1))
        query = query_index.get(eid, {})
        align = align_index.get(eid, {})
        selected, selected_rank, candidate_count = _selected_candidate(align, track_id) if align else (None, 0, 0)
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}

        action = str(row.get("semantic_id") or row.get("action") or (selected or {}).get("semantic_id") or (selected or {}).get("action") or "")
        event_type = str(row.get("event_type") or query.get("event_type") or "")
        query_text = str(row.get("query_text") or query.get("query_text") or "")
        source = _query_source(query, row)

        action_conf = _safe_float((selected or {}).get("action_confidence", row.get("action_confidence")), 0.0)
        visual_from_evidence = _nullable_score(evidence.get("visual_score", evidence.get("c_visual")))
        if visual_from_evidence is None:
            semantic = action_match_score(event_type=event_type, query_text=query_text, action_label=action)
            c_visual = _clamp01(0.65 * action_conf + 0.35 * semantic)
        else:
            c_visual = visual_from_evidence

        text_raw = _nullable_score(evidence.get("text_score", evidence.get("c_text")))
        has_real_text = asr_available and source not in {"visual_fallback", "fallback", "placeholder", "unknown", ""}
        c_text = text_raw if has_real_text else None

        win = row.get("window") if isinstance(row.get("window"), dict) else {}
        start = _safe_float(row.get("window_start", win.get("start")), 0.0)
        end = _safe_float(row.get("window_end", win.get("end")), start)
        align_score = _clamp01(_safe_float((selected or {}).get("overlap"), 0.0))
        c_object = _object_support(objects_by_frame, action=action, start=start, end=end, fps=fps)

        uq_score = _clamp01(_safe_float(evidence.get("uq_score", row.get("uncertainty")), 0.0))
        reliability = _clamp01(_safe_float(row.get("reliability_score"), 0.0))
        asr_conf = _clamp01(_safe_float(query.get("confidence"), 0.0)) if has_real_text else 0.0

        flags: List[str] = []
        if candidate_count == 0:
            flags.append("no_candidates")
        if uq_score >= 0.35:
            flags.append("high_uq")
        if action_conf and action_conf < 0.4:
            flags.append("low_visual_confidence")
        if align_score < 0.4:
            flags.append("low_alignment")
        if not has_real_text:
            flags.append("missing_text")
        elif asr_conf < 0.4:
            flags.append("low_asr_quality")
        if c_object is None:
            flags.append("missing_object")

        weights = {
            "visual": round(_clamp01((1.0 - uq_score) * max(action_conf, c_visual)), 6),
            "text": round(asr_conf, 6),
            "object": round(c_object if c_object is not None else 0.0, 6),
            "temporal": round(align_score, 6),
        }
        channel_valid = {
            "visual": c_visual is not None and weights["visual"] > 0.0,
            "text": c_text is not None and weights["text"] > 0.0,
            "object": c_object is not None and weights["object"] > 0.0,
            "temporal": align_score > 0.0,
        }
        score_map = {
            "visual": c_visual,
            "text": c_text,
            "object": c_object,
            "temporal": align_score,
        }
        coherence_valid, coherence_value = _coherence_score(
            [
                (c_visual if c_visual is not None else 0.0, weights["visual"] if channel_valid["visual"] else 0.0),
                (c_text if c_text is not None else 0.0, weights["text"] if channel_valid["text"] else 0.0),
                (c_object if c_object is not None else 0.0, weights["object"] if channel_valid["object"] else 0.0),
            ],
            align_score,
        )
        conflict = _conflict_type(
            channel_valid=channel_valid,
            scores=score_map,
            align_score=align_score,
            coherence_valid=coherence_valid,
            coherence_score=coherence_value,
            reliability=reliability,
            quality_flags=flags,
        )

        new_row["coherence"] = {
            "c_visual": round(c_visual, 6) if c_visual is not None else None,
            "c_text": round(c_text, 6) if c_text is not None else None,
            "c_object": round(c_object, 6) if c_object is not None else None,
            "align_score": round(align_score, 6),
            "coherence_score": coherence_value,
            "coherence_valid": coherence_valid,
            "conflict_type": conflict,
            "quality_flags": sorted(set(flags)),
            "channel_weights": weights,
            "channel_valid": channel_valid,
        }
        new_row["evidence_refs"] = {
            "video": {"case_id": case_id or "unknown", "start_sec": round(start, 3), "end_sec": round(max(start, end), 3)},
            "query": {"file": "event_queries.fusion_v2.jsonl", "event_id": eid},
            "alignment": {
                "file": "align_multimodal.json",
                "selected_rank": selected_rank,
                "candidate_count": candidate_count,
            },
            "verified_event": {"file": "verified_events.jsonl"},
            "quality": {"uq_file": "pose_tracks_smooth_uq.jsonl", "asr_file": "asr_quality_report.json"},
        }
        out.append(new_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Append v3 multimodal coherence diagnosis to verified events.")
    parser.add_argument("--verified", required=True, type=str)
    parser.add_argument("--aligned", required=True, type=str)
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--asr_quality", default="", type=str)
    parser.add_argument("--objects", default="", type=str)
    parser.add_argument("--case_id", default="", type=str)
    parser.add_argument("--fps", default=25.0, type=float)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--validate", default=1, type=int)
    args = parser.parse_args()

    verified_path = Path(args.verified).resolve()
    aligned_path = Path(args.aligned).resolve()
    query_path = Path(args.event_queries).resolve()
    asr_path = Path(args.asr_quality).resolve() if args.asr_quality else None
    objects_path = Path(args.objects).resolve() if args.objects else None
    out_path = Path(args.out).resolve()

    aligned_obj = _load_json(aligned_path)
    align_rows = aligned_obj if isinstance(aligned_obj, list) else []
    rows = diagnose_rows(
        verified_rows=_load_jsonl(verified_path),
        align_rows=align_rows,
        query_rows=_load_jsonl(query_path),
        asr_report=_load_json(asr_path),
        objects_by_frame=_load_objects_by_frame(objects_path),
        case_id=args.case_id or verified_path.parent.name,
        fps=float(args.fps),
    )
    write_jsonl(out_path, rows)

    if int(args.validate) == 1:
        ok, count, errors = validate_jsonl_file(out_path, validate_verified_event_v3_record)
        if not ok:
            first_error = errors[0] if errors else "unknown schema error"
            raise ValueError(f"invalid coherence v3 output after {count} rows: {first_error}")

    print(f"[DONE] coherence v3 events: {out_path} ({len(rows)})")


if __name__ == "__main__":
    main()
