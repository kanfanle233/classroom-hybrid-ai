import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COCO_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
]

BEHAVIOR_COLORS = {
    "tt": (255, 80, 40),
    "dx": (255, 220, 0),
    "dk": (0, 220, 80),
    "zt": (0, 160, 255),
    "xt": (220, 80, 255),
    "js": (40, 40, 255),
    "zl": (0, 255, 255),
    "jz": (180, 80, 255),
}

TRACK_COLORS = [
    (64, 180, 255),
    (80, 220, 80),
    (255, 160, 80),
    (220, 120, 255),
    (255, 220, 80),
    (80, 220, 255),
    (180, 255, 120),
    (255, 120, 120),
    (120, 160, 255),
    (180, 180, 80),
]


def _resolve(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _valid_bbox(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except Exception:
        return None
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _track_color(track_id: int) -> Tuple[int, int, int]:
    return TRACK_COLORS[abs(int(track_id)) % len(TRACK_COLORS)]


def _behavior_color(code: str) -> Tuple[int, int, int]:
    return BEHAVIOR_COLORS.get(str(code or "").lower(), (235, 235, 235))


def _index_person_rows(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    by_frame: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        frame = _safe_int(row.get("frame"), -1)
        if frame < 0:
            continue
        persons = row.get("persons", row.get("people", []))
        if isinstance(persons, dict):
            persons = list(persons.values())
        if not isinstance(persons, list):
            continue
        for person in persons:
            if not isinstance(person, dict):
                continue
            track_id = _safe_int(person.get("track_id"), -1)
            if track_id < 0:
                continue
            by_frame[frame][track_id] = person
    return by_frame


def _index_behavior_rows(rows: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        frame = _safe_int(row.get("frame"), -1)
        if frame < 0:
            continue
        behaviors = row.get("behaviors", row.get("detections", []))
        if isinstance(behaviors, dict):
            behaviors = list(behaviors.values())
        if not isinstance(behaviors, list):
            continue
        for behavior in behaviors:
            if isinstance(behavior, dict):
                by_frame[frame].append(behavior)
    return by_frame


def _index_actions(
    rows: Iterable[Dict[str, Any]],
    *,
    fps: float,
    max_frame: int,
    min_conf: float,
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    by_frame: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not isinstance(row, dict):
            continue
        track_id = _safe_int(row.get("track_id"), -1)
        if track_id < 0:
            continue
        conf = _safe_float(row.get("conf", row.get("confidence", 0.0)), 0.0)
        if conf < min_conf:
            continue
        if "start_frame" in row and "end_frame" in row:
            start = _safe_int(row.get("start_frame"), -1)
            end = _safe_int(row.get("end_frame"), start)
        else:
            start_t = _safe_float(row.get("start_time", row.get("t", 0.0)), 0.0)
            end_t = _safe_float(row.get("end_time", start_t), start_t)
            start = int(round(start_t * fps))
            end = int(round(end_t * fps))
        if start < 0:
            continue
        if end < start:
            start, end = end, start
        end = min(end, max_frame - 1 if max_frame > 0 else end)
        for frame in range(start, end + 1):
            prev = by_frame[frame].get(track_id)
            if prev is None or conf >= _safe_float(prev.get("conf", 0.0), 0.0):
                by_frame[frame][track_id] = row
    return by_frame


def _kp_xyc(raw: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(raw, dict):
        x = raw.get("x")
        y = raw.get("y")
        c = raw.get("c", raw.get("conf", 1.0))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        x = raw[0]
        y = raw[1]
        c = raw[2] if len(raw) >= 3 else 1.0
    else:
        return None
    try:
        return float(x), float(y), float(c)
    except Exception:
        return None


def _draw_skeleton(frame: Any, keypoints: Any, color: Tuple[int, int, int], kp_conf: float) -> int:
    if not isinstance(keypoints, list):
        return 0
    pts: List[Optional[Tuple[int, int, float]]] = []
    for kp in keypoints:
        parsed = _kp_xyc(kp)
        if parsed is None:
            pts.append(None)
            continue
        x, y, c = parsed
        if c < kp_conf:
            pts.append(None)
        else:
            pts.append((int(round(x)), int(round(y)), c))

    drawn = 0
    for a, b in COCO_EDGES:
        if a >= len(pts) or b >= len(pts):
            continue
        pa = pts[a]
        pb = pts[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame, (pa[0], pa[1]), (pb[0], pb[1]), color, 2, cv2.LINE_AA)
        drawn += 1
    for pt in pts:
        if pt is not None:
            cv2.circle(frame, (pt[0], pt[1]), 3, color, -1, cv2.LINE_AA)
    return drawn


def _draw_box(frame: Any, bbox: List[float], color: Tuple[int, int, int], thickness: int) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _draw_dashed_box(
    frame: Any,
    bbox: List[float],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash: int = 7,
    gap: int = 5,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    def draw_dashed_line(p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
        x_start, y_start = p1
        x_end, y_end = p2
        dx = x_end - x_start
        dy = y_end - y_start
        length = max(1, int((dx * dx + dy * dy) ** 0.5))
        step = dash + gap
        for offset in range(0, length, step):
            end = min(offset + dash, length)
            sx = int(round(x_start + dx * (offset / length)))
            sy = int(round(y_start + dy * (offset / length)))
            ex = int(round(x_start + dx * (end / length)))
            ey = int(round(y_start + dy * (end / length)))
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

    draw_dashed_line((x1, y1), (x2, y1))
    draw_dashed_line((x2, y1), (x2, y2))
    draw_dashed_line((x2, y2), (x1, y2))
    draw_dashed_line((x1, y2), (x1, y1))


def _draw_text_box(
    frame: Any,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    height, width = frame.shape[:2]
    x0 = max(0, min(int(x), max(0, int(width) - tw - 8)))
    y0 = max(0, min(int(y0), max(0, int(height) - th - baseline - 8)))
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 6, y0 + th + baseline + 6), (20, 20, 20), -1)
    cv2.putText(frame, text, (x0 + 3, y0 + th + 2), font, scale, color, thickness, cv2.LINE_AA)


def _draw_plain_text(
    frame: Any,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.35,
    thickness: int = 1,
) -> None:
    x, y = origin
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x0 = max(0, min(int(x), max(0, int(width) - tw - 2)))
    y0 = max(th + 1, min(int(y), max(th + 1, int(height) - 2)))
    cv2.putText(frame, text, (x0 + 1, y0 + 1), font, scale, (25, 25, 25), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x0, y0), font, scale, color, thickness, cv2.LINE_AA)


def _label_from_frame_person(person: Optional[Dict[str, Any]], min_conf: float) -> Optional[Dict[str, Any]]:
    if not person:
        return None
    status = str(person.get("behavior_match_status", "")).strip().lower()
    if status and status != "matched":
        return None
    code = str(person.get("behavior_code", "")).strip().lower()
    semantic = str(person.get("semantic_id", person.get("action", ""))).strip().lower()
    conf = _safe_float(person.get("det_conf", person.get("conf", 0.0)), 0.0)
    if not code or conf < min_conf:
        return None
    return {
        "behavior_code": code,
        "semantic_id": semantic,
        "conf": conf,
        "source": "frame",
    }


def _label_from_action(action: Optional[Dict[str, Any]], min_conf: float) -> Optional[Dict[str, Any]]:
    if not action:
        return None
    code = str(action.get("behavior_code", "")).strip().lower()
    semantic = str(action.get("semantic_id", action.get("action", ""))).strip().lower()
    conf = _safe_float(action.get("conf", action.get("confidence", 0.0)), 0.0)
    if not code or conf < min_conf:
        return None
    return {
        "behavior_code": code,
        "semantic_id": semantic,
        "conf": conf,
        "source": "segment",
    }


def _make_label(
    *,
    track_id: int,
    frame_person: Optional[Dict[str, Any]],
    action: Optional[Dict[str, Any]],
    label_source: str,
    min_conf: float,
    compact_label: bool,
    show_unmatched_label: bool,
) -> Tuple[str, str, float, str]:
    label: Optional[Dict[str, Any]] = None
    if label_source == "segment":
        label = _label_from_action(action, min_conf)
        if label is None:
            label = _label_from_frame_person(frame_person, min_conf)
    else:
        label = _label_from_frame_person(frame_person, min_conf)
    if label is None:
        if show_unmatched_label:
            return f"S{track_id:02d} no_behavior_match", "", 0.0, ""
        return "", "", 0.0, ""
    code = str(label["behavior_code"]).lower()
    semantic = str(label.get("semantic_id", "")).lower()
    conf = float(label["conf"])
    source = str(label.get("source", "frame"))
    if compact_label:
        return f"S{track_id:02d} {code} {conf:.2f}", code, conf, source
    semantic_part = f" {semantic}" if semantic and semantic != code else ""
    return f"S{track_id:02d} {code}{semantic_part} {conf:.2f} {source}", code, conf, source


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay pose skeletons and 8-class behavior labels into one demo video.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--pose_tracks", required=True, type=str)
    parser.add_argument("--student_tracks", required=True, type=str)
    parser.add_argument("--actions", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--preview_out", default="", type=str)
    parser.add_argument("--report", default="", type=str)
    parser.add_argument("--label_source", choices=["frame", "segment"], default="segment")
    parser.add_argument("--show_behavior_bbox", type=int, default=1)
    parser.add_argument("--min_conf", type=float, default=0.25)
    parser.add_argument("--keypoint_conf", type=float, default=0.35)
    parser.add_argument("--show_unmatched_label", type=int, default=0)
    parser.add_argument("--compact_label", type=int, default=1)
    parser.add_argument("--unmatched_behaviors", default="", type=str)
    parser.add_argument("--show_unlinked_behavior_bbox", type=int, default=1)
    parser.add_argument("--show_unlinked_behavior_label", type=int, default=0)
    parser.add_argument("--show_unlinked_behavior_legend", type=int, default=1)
    parser.add_argument("--unlinked_behavior_min_conf", type=float, default=0.25)
    args = parser.parse_args()

    video_path = _resolve(args.video)
    pose_path = _resolve(args.pose_tracks)
    student_path = _resolve(args.student_tracks)
    actions_path = _resolve(args.actions)
    unmatched_path = _resolve(args.unmatched_behaviors) if args.unmatched_behaviors else None
    out_path = _resolve(args.out)
    preview_path = _resolve(args.preview_out) if args.preview_out else out_path.with_name(out_path.stem + "_preview.jpg")
    report_path = _resolve(args.report) if args.report else out_path.with_name("pose_behavior_video.report.json")

    for path, label in [
        (video_path, "video"),
        (pose_path, "pose_tracks"),
        (student_path, "student_tracks"),
        (actions_path, "actions"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pose_by_frame = _index_person_rows(_read_jsonl(pose_path))
    student_by_frame = _index_person_rows(_read_jsonl(student_path))
    action_by_frame = _index_actions(_read_jsonl(actions_path), fps=fps, max_frame=frame_count, min_conf=float(args.min_conf))
    unlinked_by_frame = _index_behavior_rows(_read_jsonl(unmatched_path)) if unmatched_path else {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {out_path}")

    stats = {
        "stage": "overlay_pose_behavior_video",
        "video": str(video_path),
        "pose_tracks": str(pose_path),
        "student_tracks": str(student_path),
        "actions": str(actions_path),
        "unmatched_behaviors": str(unmatched_path) if unmatched_path else "",
        "output": str(out_path),
        "preview_output": str(preview_path),
        "label_source": str(args.label_source),
        "show_behavior_bbox": bool(int(args.show_behavior_bbox)),
        "show_unmatched_label": bool(int(args.show_unmatched_label)),
        "compact_label": bool(int(args.compact_label)),
        "show_unlinked_behavior_bbox": bool(int(args.show_unlinked_behavior_bbox)),
        "show_unlinked_behavior_label": bool(int(args.show_unlinked_behavior_label)),
        "show_unlinked_behavior_legend": bool(int(args.show_unlinked_behavior_legend)),
        "min_conf": float(args.min_conf),
        "unlinked_behavior_min_conf": float(args.unlinked_behavior_min_conf),
        "keypoint_conf": float(args.keypoint_conf),
        "fps": float(fps),
        "frame_size": [int(width), int(height)],
        "frames_read": 0,
        "frames_written": 0,
        "pose_persons_drawn": 0,
        "pose_skeleton_edges_drawn": 0,
        "behavior_boxes_drawn": 0,
        "behavior_labels_drawn": 0,
        "unmatched_persons_drawn": 0,
        "unlinked_behavior_boxes_drawn": 0,
        "unlinked_behavior_labels_drawn": 0,
        "unlinked_behavior_legend_drawn": 0,
        "segment_labels_drawn": 0,
        "frame_labels_drawn": 0,
    }

    preview_written = False
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stats["frames_read"] += 1
        pose_people = pose_by_frame.get(frame_idx, {})
        student_people = student_by_frame.get(frame_idx, {})
        frame_actions = action_by_frame.get(frame_idx, {})

        if int(args.show_unlinked_behavior_bbox) == 1:
            frame_unlinked_drawn = 0
            for behavior in unlinked_by_frame.get(frame_idx, []):
                conf = _safe_float(behavior.get("conf", behavior.get("confidence", 0.0)), 0.0)
                if conf < float(args.unlinked_behavior_min_conf):
                    continue
                behavior_bbox = _valid_bbox(behavior.get("bbox"))
                if behavior_bbox is None:
                    continue
                _draw_dashed_box(frame, behavior_bbox, (150, 150, 150), thickness=1)
                stats["unlinked_behavior_boxes_drawn"] += 1
                frame_unlinked_drawn += 1
                if int(args.show_unlinked_behavior_label) == 1:
                    code = str(behavior.get("behavior_code", behavior.get("label", ""))).strip().lower()
                    label = f"unlinked {code}" if code else "unlinked"
                    ux1, uy1, _, _ = [int(round(v)) for v in behavior_bbox]
                    _draw_plain_text(frame, label, (ux1, max(14, uy1 - 4)), (190, 190, 190), scale=0.32)
                    stats["unlinked_behavior_labels_drawn"] += 1
            if frame_unlinked_drawn > 0 and int(args.show_unlinked_behavior_legend) == 1:
                _draw_plain_text(
                    frame,
                    f"gray dashed = unlinked behavior det ({frame_unlinked_drawn})",
                    (12, height - 10),
                    (190, 190, 190),
                    scale=0.38,
                )
                stats["unlinked_behavior_legend_drawn"] += 1

        for track_id in sorted(pose_people):
            pose_person = pose_people[track_id]
            student_person = student_people.get(track_id)
            action = frame_actions.get(track_id)
            pose_bbox = _valid_bbox(pose_person.get("bbox"))
            if pose_bbox is None:
                continue
            track_color = _track_color(track_id)
            label_text, code, _, label_origin = _make_label(
                track_id=track_id,
                frame_person=student_person,
                action=action,
                label_source=str(args.label_source),
                min_conf=float(args.min_conf),
                compact_label=bool(int(args.compact_label)),
                show_unmatched_label=bool(int(args.show_unmatched_label)),
            )
            label_color = _behavior_color(code) if code else (170, 170, 170)

            _draw_box(frame, pose_bbox, track_color, 2)
            stats["pose_skeleton_edges_drawn"] += _draw_skeleton(
                frame,
                pose_person.get("keypoints", []),
                track_color,
                kp_conf=float(args.keypoint_conf),
            )
            stats["pose_persons_drawn"] += 1

            if int(args.show_behavior_bbox) == 1 and student_person:
                behavior_bbox = _valid_bbox(student_person.get("behavior_bbox"))
                if behavior_bbox is not None:
                    _draw_box(frame, behavior_bbox, label_color, 2)
                    stats["behavior_boxes_drawn"] += 1

            x1, y1, _, _ = [int(round(v)) for v in pose_bbox]
            if label_text:
                _draw_text_box(frame, label_text, (x1, max(18, y1 - 6)), label_color)
            if code:
                stats["behavior_labels_drawn"] += 1
                stats[f"{label_origin}_labels_drawn"] += 1
            else:
                stats["unmatched_persons_drawn"] += 1

        cv2.putText(
            frame,
            f"frame={frame_idx} t={frame_idx / fps:.2f}s",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        stats["frames_written"] += 1
        if not preview_written:
            cv2.imwrite(str(preview_path), frame)
            preview_written = True
        frame_idx += 1

    cap.release()
    writer.release()
    stats["status"] = "ok" if stats["frames_written"] > 0 else "failed"
    _write_json(report_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
