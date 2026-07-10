import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, validate_jsonl_file, validate_pose_uq_record


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return float(x)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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
                yield obj


def _visible_ratio(kpts: List[Dict[str, Any]], c_th: float = 0.25) -> float:
    if not kpts:
        return 0.0
    vis = 0
    for kp in kpts:
        c = _safe_float(kp.get("c", 0.0), 0.0)
        if c >= c_th:
            vis += 1
    return vis / max(1, len(kpts))


def _kpt_motion(prev_kpts: List[Dict[str, Any]], cur_kpts: List[Dict[str, Any]], bbox: List[Any]) -> float:
    if not prev_kpts or not cur_kpts or len(prev_kpts) != len(cur_kpts):
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    diag = max(1.0, math.hypot(x2 - x1, y2 - y1))
    deltas = []
    for p, c in zip(prev_kpts, cur_kpts):
        dx = _safe_float(c.get("x", 0.0)) - _safe_float(p.get("x", 0.0))
        dy = _safe_float(c.get("y", 0.0)) - _safe_float(p.get("y", 0.0))
        deltas.append(math.hypot(dx, dy))
    mean_delta = sum(deltas) / max(1, len(deltas))
    # Normalize by person scale. 0 => stable, 1 => highly unstable.
    return _clamp01(mean_delta / (0.20 * diag))


def _bbox_stability(prev_bbox: List[Any], cur_bbox: List[Any]) -> float:
    if not prev_bbox or not cur_bbox:
        return 0.5
    px1, py1, px2, py2 = [float(v) for v in prev_bbox]
    cx1, cy1, cx2, cy2 = [float(v) for v in cur_bbox]
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    p_diag = max(1.0, math.hypot(pw, ph))
    p_cx, p_cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    c_cx, c_cy = (cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5
    shift = math.hypot(c_cx - p_cx, c_cy - p_cy)
    instability = _clamp01(shift / (0.20 * p_diag))
    return _clamp01(1.0 - instability)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Estimate track uncertainty and export fixed UQ schema.")
    parser.add_argument(
        "--in",
        "--tracks",
        dest="in_path",
        required=True,
        type=str,
        help="pose_tracks_smooth.jsonl",
    )
    parser.add_argument("--out", dest="out_path", required=True, type=str, help="pose_tracks_smooth_uq.jsonl")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--validate", type=int, default=1, help="1=validate output schema")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.is_absolute():
        in_path = (base_dir / in_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"pose tracks not found: {in_path}")

    prev_kpts: Dict[int, List[Dict[str, Any]]] = {}
    prev_bbox: Dict[int, List[Any]] = {}
    rows_written = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for frame_row in _iter_jsonl(in_path):
            frame_idx = int(frame_row.get("frame", 0))
            t_sec = _safe_float(frame_row.get("t", frame_idx / max(1e-6, float(args.fps))), frame_idx / max(1e-6, float(args.fps)))
            persons = frame_row.get("persons", [])
            if not isinstance(persons, list):
                continue

            person_out: List[Dict[str, Any]] = []
            for person in persons:
                if not isinstance(person, dict):
                    continue
                track_id = person.get("track_id")
                if not isinstance(track_id, int):
                    continue

                bbox = person.get("bbox", [0, 0, 1, 1])
                if not isinstance(bbox, list) or len(bbox) < 4:
                    bbox = [0, 0, 1, 1]
                kpts = person.get("keypoints", [])
                if not isinstance(kpts, list):
                    kpts = []

                visible_ratio = _visible_ratio(kpts)
                motion_instability = _kpt_motion(prev_kpts.get(track_id, []), kpts, bbox)
                motion_stability = _clamp01(1.0 - motion_instability)
                bbox_stability = _bbox_stability(prev_bbox.get(track_id, []), bbox)

                # UQ score: larger when visibility is low or motion/bbox are unstable.
                uq_score = _clamp01(
                    0.45 * (1.0 - visible_ratio)
                    + 0.35 * (1.0 - motion_stability)
                    + 0.20 * (1.0 - bbox_stability)
                )

                uq_source: List[str] = []
                if visible_ratio < 0.65:
                    uq_source.append("low_conf")
                if motion_stability < 0.55:
                    uq_source.append("motion_jump")
                if bbox_stability < 0.60:
                    uq_source.append("bbox_shift")
                if not uq_source:
                    uq_source.append("stable")

                person_out.append(
                    {
                        "track_id": int(track_id),
                        "uq_track": round(uq_score, 6),
                        "uq_conf": round(1.0 - visible_ratio, 6),
                        "uq_motion": round(1.0 - motion_stability, 6),
                        "uq_kpt": round(1.0 - bbox_stability, 6),
                        # Reserved output for later learned UQ heads.
                        "log_sigma2": round(math.log(max(1e-6, uq_score + 1e-4)), 6),
                        # Compatibility fields (legacy reader support).
                        "uq_score": round(uq_score, 6),
                        "uq_source": uq_source,
                        "visible_kpt_ratio": round(visible_ratio, 6),
                        "motion_stability": round(motion_stability, 6),
                        "bbox_stability": round(bbox_stability, 6),
                    }
                )

                prev_kpts[track_id] = kpts
                prev_bbox[track_id] = bbox

            if not person_out:
                continue
            uq_frame = sum(float(p["uq_track"]) for p in person_out) / len(person_out)
            out_row = {
                "schema_version": SCHEMA_VERSION,
                "uq_type": "heuristic_statistical",
                "uq_scope": "track_sequence",
                "presence_aware": False,
                "variance_head": False,
                "frame": frame_idx,
                "t": round(t_sec, 6),
                "uq_frame": round(uq_frame, 6),
                "persons": person_out,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            rows_written += 1

    if int(args.validate) == 1:
        ok, _, errors = validate_jsonl_file(out_path, validate_pose_uq_record)
        if not ok:
            first_error = errors[0] if errors else "unknown schema error"
            raise ValueError(f"invalid pose UQ schema: {first_error}")

    print(f"[DONE] pose UQ exported: {out_path}")
    print(f"[INFO] rows written: {rows_written}")


if __name__ == "__main__":
    main()
