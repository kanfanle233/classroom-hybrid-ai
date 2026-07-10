import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import validate_align_record


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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
                "action_confidence": _safe_float(row.get("conf", row.get("confidence", 0.5)), 0.5),
            }
        )
    out.sort(key=lambda x: (x["start_time"], x["track_id"]))
    if require_semantic and missing_semantic > 0:
        raise ValueError(f"semantic fields missing on {missing_semantic} action rows")
    return out


def _build_uq_index(rows: List[Dict[str, Any]]) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Build per-frame per-track map:
      frame_idx -> track_id -> {"uq_score":..., "motion_stability":...}
    """
    out: Dict[int, Dict[int, Dict[str, float]]] = {}
    for row in rows:
        # Contract v1: frame-level row with persons[].
        if isinstance(row.get("frame"), int) and isinstance(row.get("persons"), list):
            frame_idx = int(row["frame"])
            for person in row["persons"]:
                if not isinstance(person, dict):
                    continue
                track_id = person.get("track_id")
                if not isinstance(track_id, int):
                    continue
                uq_track = _safe_float(person.get("uq_track", person.get("uq_score", 0.5)), 0.5)
                uq_motion = _safe_float(person.get("uq_motion", 0.5), 0.5)
                motion_stability = _safe_float(person.get("motion_stability", 1.0 - uq_motion), 1.0 - uq_motion)
                out.setdefault(frame_idx, {})[track_id] = {
                    "uq_score": uq_track,
                    "motion_stability": motion_stability,
                }
            continue

        # Legacy compatibility: one row per (frame_idx, track_id).
        frame_idx = row.get("frame_idx")
        track_id = row.get("track_id")
        if not isinstance(frame_idx, int) or not isinstance(track_id, int):
            continue
        out.setdefault(frame_idx, {})[track_id] = {
            "uq_score": _safe_float(row.get("uq_score", row.get("uq_track", 0.5)), 0.5),
            "motion_stability": _safe_float(row.get("motion_stability", 0.5), 0.5),
        }
    return out


def _mean_uq_near(uq_index: Dict[int, Dict[int, Dict[str, float]]], t: float, fps: float, radius_sec: float = 1.0) -> Tuple[float, float, Dict[int, float]]:
    center_frame = int(round(t * fps))
    radius = max(1, int(round(radius_sec * fps)))
    frames = range(max(0, center_frame - radius), center_frame + radius + 1)
    uq_vals: List[float] = []
    motion_vals: List[float] = []
    by_track: Dict[int, List[float]] = {}

    for fr in frames:
        row = uq_index.get(fr, {})
        for tid, values in row.items():
            uq = _safe_float(values.get("uq_score", 0.5), 0.5)
            motion_stability = _safe_float(values.get("motion_stability", 0.5), 0.5)
            uq_vals.append(uq)
            motion_vals.append(1.0 - motion_stability)
            by_track.setdefault(tid, []).append(uq)

    uq_mean = sum(uq_vals) / len(uq_vals) if uq_vals else 0.0
    motion_mean = sum(motion_vals) / len(motion_vals) if motion_vals else 0.0
    track_uq = {tid: (sum(vals) / len(vals) if vals else uq_mean) for tid, vals in by_track.items()}
    return uq_mean, motion_mean, track_uq


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    if inter <= 0:
        return 0.0
    denom = max(1e-6, min(a1 - a0, b1 - b0))
    return inter / denom


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Adaptive multimodal aligner (query + visual candidates + UQ basis).")
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--actions", required=True, type=str)
    parser.add_argument("--pose_uq", required=True, type=str)
    parser.add_argument("--out", required=True, type=str, help="align_multimodal.json")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--base_window", type=float, default=1.0)
    parser.add_argument("--alpha_motion", type=float, default=1.2)
    parser.add_argument("--beta_uq", type=float, default=0.8)
    parser.add_argument("--min_window", type=float, default=0.6)
    parser.add_argument("--max_window", type=float, default=4.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--require_semantic", type=int, default=0)
    args = parser.parse_args()

    event_path = Path(args.event_queries)
    action_path = Path(args.actions)
    uq_path = Path(args.pose_uq)
    out_path = Path(args.out)
    if not event_path.is_absolute():
        event_path = (base_dir / event_path).resolve()
    if not action_path.is_absolute():
        action_path = (base_dir / action_path).resolve()
    if not uq_path.is_absolute():
        uq_path = (base_dir / uq_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    queries = _load_jsonl(event_path)
    actions = _normalize_actions(_load_jsonl(action_path), require_semantic=bool(int(args.require_semantic)))
    uq_index = _build_uq_index(_load_jsonl(uq_path))

    aligned: List[Dict[str, Any]] = []
    for q in queries:
        event_id = str(q.get("event_id", q.get("query_id", "")))
        event_type = str(q.get("event_type", "unknown"))
        query_text = str(q.get("query_text", ""))
        t_center = _safe_float(q.get("timestamp", q.get("t_center", q.get("start", 0.0))), 0.0)

        uq_basis, motion_basis, track_uq = _mean_uq_near(uq_index, t=t_center, fps=float(args.fps), radius_sec=1.0)
        window_size = _clamp(
            float(args.base_window) + float(args.alpha_motion) * motion_basis + float(args.beta_uq) * uq_basis,
            float(args.min_window),
            float(args.max_window),
        )
        w_start = max(0.0, t_center - window_size)
        w_end = t_center + window_size

        candidates: List[Dict[str, Any]] = []
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
                    "start_time": round(a["start_time"], 3),
                    "end_time": round(a["end_time"], 3),
                    "overlap": round(float(ov), 6),
                    "action_confidence": round(float(a["action_confidence"]), 6),
                    "uq_track": round(float(track_uq.get(tid, uq_basis)), 6),
                    # Compatibility alias.
                    "uq_score": round(float(track_uq.get(tid, uq_basis)), 6),
                }
            )
        candidates.sort(key=lambda c: (c["overlap"], c["action_confidence"]), reverse=True)
        candidates = candidates[: max(1, int(args.topk))]

        row = {
            "event_id": event_id,
            "query_id": event_id,
            "event_type": event_type,
            "query_text": query_text,
            "window_start": round(w_start, 3),
            "window_end": round(w_end, 3),
            "window_center": round(t_center, 3),
            "window_size": round(window_size, 3),
            "basis_motion": round(motion_basis, 6),
            "basis_uq": round(uq_basis, 6),
            # Compatibility aliases for existing readers.
            "window": {
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "center": round(t_center, 3),
                "size": round(window_size, 3),
            },
            "motion_basis": round(motion_basis, 6),
            "uq_basis": round(uq_basis, 6),
            "candidates": candidates,
        }
        ok, msg = validate_align_record(row)
        if not ok:
            raise ValueError(f"align schema invalid for event_id={event_id}: {msg}")
        aligned.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    print(f"[DONE] aligned multimodal events: {out_path}")
    print(f"[INFO] events: {len(aligned)}")


if __name__ == "__main__":
    main()
