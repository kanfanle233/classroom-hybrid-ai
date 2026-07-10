import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

from ultralytics import YOLO
from classroom_pipeline.model_resolver import resolve_model_reference

from classroom_pipeline.utils.sliced_inference_utils import (
    apply_sr,
    build_tiles,
    count_in_roi,
    crop_tile_from_sr_roi,
    dedupe_detections,
    load_sr_cache_metadata,
    read_sr_cache_frame,
    resolve_roi,
    summarize_diagnostics,
    write_diagnostics,
)


DEFAULT_LABEL_TO_ACTION = {
    "dx": "write",       # low-head writing
    "dk": "read",        # low-head reading
    "tt": "listen",      # head-up listening
    "zt": "turn_head",   # turning around
    "js": "raise_hand",  # hand raising
    "zl": "stand",       # standing
    "xt": "group_discussion",
    "jz": "teacher_interaction",
}


def _resolve(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _safe_label(name: Any) -> str:
    return str(name or "").strip().lower()


def _load_names(model: YOLO) -> Dict[int, str]:
    names = getattr(model.model, "names", None) if hasattr(model, "model") else getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _predict_behavior(
    model: YOLO,
    image: Any,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    names: Dict[int, str],
    offset=(0.0, 0.0),
    scale: float = 1.0,
    source: str = "full",
    tile_id: str = "",
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    results = model.predict(
        image,
        verbose=False,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device if str(device).strip() else None,
    )
    row0 = results[0]
    boxes = getattr(row0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections
    ox, oy = float(offset[0]), float(offset[1])
    scale = max(1e-9, float(scale))
    for box in boxes:
        cls_id = int(box.cls[0])
        det_conf = float(box.conf[0])
        raw_xyxy = [float(v) for v in box.xyxy[0].tolist()]
        xyxy = [
            raw_xyxy[0] / scale + ox,
            raw_xyxy[1] / scale + oy,
            raw_xyxy[2] / scale + ox,
            raw_xyxy[3] / scale + oy,
        ]
        label = _safe_label(names.get(cls_id, "unknown"))
        item = {
            "cls_id": cls_id,
            "label": label,
            "action": DEFAULT_LABEL_TO_ACTION.get(label, label),
            "conf": det_conf,
            "bbox": xyxy,
            "source": source,
        }
        if tile_id:
            item["tile_id"] = tile_id
        detections.append(item)
    return detections


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Export 8-class behavior detections to jsonl.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out", required=True, type=str, help="behavior_det.jsonl")
    parser.add_argument("--model", default="runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt", type=str)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.50, type=float)
    parser.add_argument("--imgsz", default=832, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--device", default="", type=str, help="0/cpu/empty(auto)")
    parser.add_argument("--infer_mode", choices=["full", "sliced", "full_sliced"], default="full")
    parser.add_argument("--slice_grid", default="2x2", type=str, help="NxM, auto/adaptive, rear_adaptive, or rear_dense")
    parser.add_argument("--slice_overlap", default=0.25, type=float)
    parser.add_argument("--slice_roi", default="auto_rear", type=str)
    parser.add_argument(
        "--sr_backend",
        choices=["off", "opencv", "realesrgan", "basicvsrpp", "realbasicvsr", "nvidia_vsr", "maxine_vfx"],
        default="off",
    )
    parser.add_argument("--sr_scale", default=2.0, type=float)
    parser.add_argument("--sr_cache_dir", default="", type=str)
    parser.add_argument("--diagnostics_out", default="", type=str)
    args = parser.parse_args()

    video_path = _resolve(base_dir, args.video)
    out_path = _resolve(base_dir, args.out)
    model_ref = resolve_model_reference(args.model, project_root=PROJECT_ROOT, required=True)
    diag_path = _resolve(base_dir, args.diagnostics_out) if args.diagnostics_out else None

    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_ref))
    names = _load_names(model)
    sr_cache_dir = _resolve(base_dir, args.sr_cache_dir) if args.sr_cache_dir else None
    sr_cache_meta = load_sr_cache_metadata(sr_cache_dir)
    if str(args.sr_backend).strip().lower() not in {"off", "none", "opencv"} and sr_cache_meta is None:
        raise RuntimeError(
            f"sr_backend={args.sr_backend} requires a valid ROI SR cache. "
            "Run 02c_build_rear_roi_sr_cache.py or provide --sr_cache_dir."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0

    frame_idx = 0
    wrote = 0
    diag_rows: List[Dict[str, Any]] = []
    roi_bbox = None
    with out_path.open("w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % max(1, int(args.stride)) != 0:
                frame_idx += 1
                continue

            infer_mode = str(args.infer_mode).strip().lower()
            roi_bbox = resolve_roi(frame.shape, args.slice_roi)
            full_detections: List[Dict[str, Any]] = []
            tile_detections: List[Dict[str, Any]] = []
            if infer_mode in {"full", "full_sliced"}:
                full_detections = _predict_behavior(
                    model,
                    frame,
                    conf=float(args.conf),
                    iou=float(args.iou),
                    imgsz=int(args.imgsz),
                    device=args.device,
                    names=names,
                    source="full",
                )
            if infer_mode in {"sliced", "full_sliced"}:
                sr_roi = read_sr_cache_frame(sr_cache_meta, frame_idx)
                sr_roi_bbox = sr_cache_meta.get("roi", roi_bbox) if sr_cache_meta else roi_bbox
                sr_scale = float(sr_cache_meta.get("scale", args.sr_scale)) if sr_cache_meta else float(args.sr_scale)
                for tile in build_tiles(frame.shape, grid=args.slice_grid, overlap=args.slice_overlap, roi=args.slice_roi):
                    x1, y1, x2, y2 = [int(v) for v in tile["bbox"]]
                    if sr_roi is not None:
                        crop, scale = crop_tile_from_sr_roi(sr_roi, sr_roi_bbox, [x1, y1, x2, y2], sr_scale)
                    else:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        crop, scale = apply_sr(crop, backend=args.sr_backend, scale=float(args.sr_scale))
                    if crop.size == 0:
                        continue
                    tile_detections.extend(
                        _predict_behavior(
                            model,
                            crop,
                            conf=float(args.conf),
                            iou=float(args.iou),
                            imgsz=int(args.imgsz),
                            device=args.device,
                            names=names,
                            offset=(x1, y1),
                            scale=scale,
                            source="tile",
                            tile_id=str(tile.get("tile_id", "")),
                        )
                    )
            detections = dedupe_detections(
                full_detections + tile_detections,
                frame_shape=frame.shape,
                class_sensitive=True,
                iou_thres=0.45,
                overlap_min_thres=0.55,
                center_ratio_thres=0.025,
            )
            diag_rows.append(
                {
                    "frame": int(frame_idx),
                    "full_raw": len(full_detections),
                    "tile_raw": len(tile_detections),
                    "merged": len(detections),
                    "rear_merged": count_in_roi(detections, roi_bbox, float(args.conf)),
                    "avg_conf": round(sum(float(d.get("conf", 0.0)) for d in detections) / max(1, len(detections)), 4),
                }
            )

            if detections:
                row = {
                    "frame": int(frame_idx),
                    "t": round(float(frame_idx) / float(fps), 3),
                    "behaviors": detections,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                wrote += 1

            frame_idx += 1

    cap.release()
    if diag_path:
        write_diagnostics(
            diag_path,
            {
                "stage": "behavior_sliced_inference",
                "input": str(video_path),
                "output": str(out_path),
                "model": str(model_path),
                "conf": float(args.conf),
                "iou": float(args.iou),
                "imgsz": int(args.imgsz),
                "infer_mode": str(args.infer_mode),
                "slice_grid": str(args.slice_grid),
                "slice_overlap": float(args.slice_overlap),
                "slice_roi": str(args.slice_roi),
                "sr_backend": str(args.sr_backend),
                "sr_scale": float(args.sr_scale),
                "sr_cache_dir": str(sr_cache_dir) if sr_cache_dir else "",
                "sr_cache_status": str(sr_cache_meta.get("status", "")) if sr_cache_meta else "",
                "summary": summarize_diagnostics(diag_rows, mode=str(args.infer_mode), roi=roi_bbox or [0, 0, 0, 0]),
                "frames": diag_rows,
                "status": "ok",
            },
        )
    print(f"[DONE] behavior detections: {out_path} (rows={wrote})")


if __name__ == "__main__":
    main()
