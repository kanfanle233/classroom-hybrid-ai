import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


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


def _resolve(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _read_jsonl(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return by_frame
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            frame = int(row.get("frame", -1))
            if frame < 0:
                continue
            persons = row.get("persons", [])
            if isinstance(persons, list):
                by_frame[frame] = [p for p in persons if isinstance(p, dict)]
    return by_frame


def _grab_frame(video: Path, frame_idx: int) -> Any:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx} from {video}")
    return frame


def _draw_text(img: Any, text: str, org: Tuple[int, int], color=(255, 255, 255), scale=0.55) -> None:
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _kp_xyc(raw: Any) -> Tuple[float, float, float]:
    if isinstance(raw, dict):
        return float(raw.get("x", 0.0)), float(raw.get("y", 0.0)), float(raw.get("c", raw.get("conf", 1.0)) or 0.0)
    return float(raw[0]), float(raw[1]), float(raw[2]) if len(raw) > 2 and raw[2] is not None else 1.0


def _draw_pose(img: Any, persons: Sequence[Dict[str, Any]], *, kp_conf: float = 0.35) -> Any:
    out = img.copy()
    for person in persons:
        bbox = person.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            color = (0, 220, 255) if str(person.get("source", "")) == "tile" else (255, 160, 60)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{person.get('source', 'full')} {float(person.get('conf', 0.0)):.2f}"
            _draw_text(out, label, (x1, max(16, y1 - 4)), color=color, scale=0.4)
        keypoints = person.get("keypoints", [])
        pts = []
        for kp in keypoints:
            try:
                x, y, c = _kp_xyc(kp)
            except Exception:
                pts.append(None)
                continue
            pts.append((int(round(x)), int(round(y)), float(c)) if c >= kp_conf else None)
        for a, b in COCO_EDGES:
            if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                cv2.line(out, (pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]), (80, 255, 80), 2, cv2.LINE_AA)
        for pt in pts:
            if pt:
                cv2.circle(out, (pt[0], pt[1]), 3, (80, 255, 80), -1, cv2.LINE_AA)
    _draw_text(out, f"sliced pose persons={len(persons)}", (12, 28), color=(80, 255, 80), scale=0.55)
    return out


def _resize_h(img: Any, height: int) -> Any:
    scale = height / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rear-row enhancement comparison contact sheet.")
    parser.add_argument("--pose_demo_video", required=True)
    parser.add_argument("--fusion_video", required=True)
    parser.add_argument("--pose_jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--frames", default="0,25,50,100")
    args = parser.parse_args()

    pose_demo_video = _resolve(args.pose_demo_video)
    fusion_video = _resolve(args.fusion_video)
    pose_jsonl = _resolve(args.pose_jsonl)
    out_path = _resolve(args.out)
    report_path = _resolve(args.report) if args.report else out_path.with_suffix(".report.json")
    frame_ids = [int(x.strip()) for x in str(args.frames).split(",") if x.strip()]

    pose_by_frame = _read_jsonl(pose_jsonl)
    rows = []
    report_frames = []
    for frame_id in frame_ids:
        pose_demo = _grab_frame(pose_demo_video, frame_id)
        fusion = _grab_frame(fusion_video, frame_id)
        sliced = _draw_pose(pose_demo, pose_by_frame.get(frame_id, []))
        h = min(pose_demo.shape[0], sliced.shape[0], fusion.shape[0])
        cells = [_resize_h(pose_demo, h), _resize_h(sliced, h), _resize_h(fusion, h)]
        bar = np.full((h, 8, 3), 255, dtype=np.uint8)
        row = cv2.hconcat([cells[0], bar, cells[1], bar, cells[2]])
        row = cv2.copyMakeBorder(row, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        _draw_text(row, f"frame {frame_id}: step01 full", (10, 22), color=(40, 40, 40), scale=0.52)
        _draw_text(row, "sliced pose jsonl", (cells[0].shape[1] + 20, 22), color=(40, 40, 40), scale=0.52)
        _draw_text(row, "fusion overlay", (cells[0].shape[1] + cells[1].shape[1] + 36, 22), color=(40, 40, 40), scale=0.52)
        rows.append(row)
        report_frames.append({"frame": frame_id, "sliced_pose_persons": len(pose_by_frame.get(frame_id, []))})

    width = max(r.shape[1] for r in rows)
    padded = []
    for row in rows:
        if row.shape[1] < width:
            row = cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded.append(row)
    sheet = cv2.vconcat(padded)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    report_path.write_text(
        json.dumps(
            {
                "stage": "rear_row_contact_sheet",
                "output": str(out_path),
                "frames": report_frames,
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out_path), "frames": len(frame_ids), "status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
