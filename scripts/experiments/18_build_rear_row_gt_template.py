import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classroom_pipeline.utils.sliced_inference_utils import resolve_roi


def _resolve(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_frames(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x.strip()) for x in part.split(":") if x.strip()]
            if len(bits) == 2:
                start, end = bits
                step = 1
            elif len(bits) == 3:
                start, end, step = bits
            else:
                raise ValueError(f"bad frame range: {part}")
            out.extend(range(start, end + 1, max(1, step)))
        else:
            out.append(int(part))
    return sorted(set(x for x in out if x >= 0))


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create a rear-row GT annotation template JSONL and frame images."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--roi", default="auto_rear")
    parser.add_argument("--frames", default="", help="Comma list or start:end:step, e.g. 0,25,50 or 0:1500:25")
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--duration_sec", type=float, default=60.0)
    parser.add_argument("--every_sec", type=float, default=1.0)
    args = parser.parse_args()

    video_path = _resolve(base_dir, args.video)
    out_jsonl = _resolve(base_dir, args.out_jsonl)
    image_dir = _resolve(base_dir, args.image_dir) if args.image_dir else out_jsonl.with_suffix("") / "gt_frames"
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        fps = 25.0

    if str(args.frames).strip():
        frames = _parse_frames(args.frames)
    else:
        start = max(0, int(round(float(args.start_sec) * fps)))
        end = min(max(0, total_frames - 1), int(round((float(args.start_sec) + float(args.duration_sec)) * fps)))
        step = max(1, int(round(float(args.every_sec) * fps)))
        frames = list(range(start, end + 1, step))

    image_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for frame_idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        roi = resolve_roi(frame.shape, args.roi)
        x1, y1, x2, y2 = [int(round(v)) for v in roi]
        preview = frame.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            preview,
            f"frame={frame_idx} t={frame_idx / fps:.2f}s ROI={args.roi}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"frame={frame_idx} t={frame_idx / fps:.2f}s ROI={args.roi}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        image_path = image_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(image_path), preview)
        rows.append(
            {
                "frame": int(frame_idx),
                "t": round(float(frame_idx) / fps, 3),
                "roi": [round(float(v), 3) for v in roi],
                "image": str(image_path),
                "persons": [],
                "actions": [],
                "annotation_schema": {
                    "persons[]": {
                        "bbox": "required [x1,y1,x2,y2] in original video pixels",
                        "track_id": "required stable video-local ID for IDF1/IDSW",
                        "keypoints": "optional 17 COCO keypoints as [[x,y,v], ...] for PCK/OKS",
                        "behavior_code": "optional 8-class code if visible for behavior F1/frame-mAP",
                        "occlusion": "optional none|partial|heavy",
                    },
                    "actions[]": {
                        "track_id": "optional stable ID",
                        "behavior_code": "optional 8-class code",
                        "start_frame/end_frame": "optional temporal segment boundaries",
                        "start_time/end_time": "optional temporal segment boundaries in seconds",
                    },
                },
            }
        )
    cap.release()

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "status": "ok",
        "video": str(video_path),
        "out_jsonl": str(out_jsonl),
        "image_dir": str(image_dir),
        "frames": len(rows),
        "fps": fps,
        "video_size": [width, height],
        "roi": args.roi,
        "note": "Fill persons[] bbox+track_id manually. Add keypoints/behavior/actions when evaluating PCK/OKS, behavior F1, and temporal mAP.",
    }
    _write_json(out_jsonl.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
