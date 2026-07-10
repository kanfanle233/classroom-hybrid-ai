import argparse
import os
import json
import sys
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

from ultralytics import YOLO

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


def interpolate_occluded_keypoints(kpts, conf_thres=0.2):
    visible = [p for p in kpts if p["c"] is not None and p["c"] >= conf_thres]
    if not visible:
        return kpts
    mx = sum(p["x"] for p in visible) / len(visible)
    my = sum(p["y"] for p in visible) / len(visible)
    out = []
    for p in kpts:
        if p["c"] is None or p["c"] < conf_thres:
            out.append({"x": float(mx), "y": float(my), "c": p["c"]})
        else:
            out.append(p)
    return out


def _predict_pose(
    model,
    image,
    *,
    conf_thres,
    imgsz,
    device,
    use_half,
    offset=(0.0, 0.0),
    scale=1.0,
    source="full",
    tile_id="",
):
    r = model.predict(
        image,
        verbose=False,
        conf=conf_thres,
        imgsz=imgsz,
        device=device,
        half=use_half,
    )[0]
    persons = []
    if r.boxes is None or r.keypoints is None:
        return persons
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    xy = r.keypoints.xy.cpu().numpy()
    kc = r.keypoints.conf.cpu().numpy()
    ox, oy = float(offset[0]), float(offset[1])
    scale = max(1e-9, float(scale))
    n = min(len(boxes), len(xy))
    for pid in range(n):
        bbox = [
            float(boxes[pid][0]) / scale + ox,
            float(boxes[pid][1]) / scale + oy,
            float(boxes[pid][2]) / scale + ox,
            float(boxes[pid][3]) / scale + oy,
        ]
        kpts = []
        for k in range(xy.shape[1]):
            kpts.append(
                {
                    "x": float(xy[pid, k, 0]) / scale + ox,
                    "y": float(xy[pid, k, 1]) / scale + oy,
                    "c": float(kc[pid, k]) if kc is not None else None,
                }
            )
        persons.append(
            {
                "person_idx": int(pid),
                "conf": float(scores[pid]),
                "bbox": bbox,
                "keypoints": kpts,
                "_source": source,
                "_tile_id": tile_id,
            }
        )
    return persons


def main():
    base_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/videos/demo1.mp4", help="input video path (relative to project root)")
    parser.add_argument("--out", type=str, default="output/pose_keypoints_v2.jsonl", help="output jsonl path (relative to project root)")
    parser.add_argument("--model", type=str, default="yolo11x-pose.pt", help="model path (relative to project root)")
    parser.add_argument("--conf", type=float, default=0.20, help="person conf threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="inference image size")
    parser.add_argument("--device", type=str, default="", help="device: 0/cuda/cpu; empty=auto")
    parser.add_argument("--half", type=int, default=0, help="1=fp16 (cuda only), 0=fp32")
    parser.add_argument("--interpolate_occluded", type=int, default=0, help="1=interpolate low-conf keypoints")
    parser.add_argument("--occlusion_conf_thres", type=float, default=0.2, help="low-conf threshold for occlusion")
    parser.add_argument("--infer_mode", choices=["full", "sliced", "full_sliced", "roi_sr_sliced"], default="full")
    parser.add_argument("--slice_grid", type=str, default="2x2", help="NxM, auto/adaptive, rear_adaptive, or rear_dense")
    parser.add_argument("--slice_overlap", type=float, default=0.25)
    parser.add_argument("--slice_roi", type=str, default="auto_rear")
    parser.add_argument(
        "--sr_backend",
        choices=["off", "opencv", "realesrgan", "basicvsrpp", "realbasicvsr", "nvidia_vsr", "maxine_vfx"],
        default="off",
    )
    parser.add_argument("--sr_scale", type=float, default=2.0)
    parser.add_argument("--sr_cache_dir", type=str, default="")
    parser.add_argument("--diagnostics_out", type=str, default="")
    args = parser.parse_args()

    video_path = (base_dir / args.video).resolve()
    jsonl_path = (base_dir / args.out).resolve()
    model_arg = args.model

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print("[CHECK] video exists?", video_path.exists(), video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    model_path = Path(model_arg)
    if model_path.is_absolute():
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model_ref = str(model_path)
    else:
        local_model = (base_dir / model_arg).resolve()
        model_ref = str(local_model) if local_model.exists() else model_arg
    print("[CHECK] model ref:", model_ref)

    model = YOLO(model_ref)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0

    conf_thres = float(args.conf)
    device = args.device if args.device else None
    use_half = bool(int(args.half))
    infer_mode = str(args.infer_mode).strip().lower()
    diag_path = (base_dir / args.diagnostics_out).resolve() if args.diagnostics_out else None
    diag_rows = []
    roi_bbox = None
    sr_cache_dir = (base_dir / args.sr_cache_dir).resolve() if args.sr_cache_dir else None
    sr_cache_meta = load_sr_cache_metadata(sr_cache_dir)
    if str(args.sr_backend).strip().lower() not in {"off", "none", "opencv"} and sr_cache_meta is None:
        raise RuntimeError(
            f"sr_backend={args.sr_backend} requires a valid ROI SR cache. "
            "Run 02c_build_rear_roi_sr_cache.py or provide --sr_cache_dir."
        )

    with open(str(jsonl_path), "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = frame_idx / fps
            roi_bbox = resolve_roi(frame.shape, args.slice_roi)
            full_persons = []
            tile_persons = []
            if infer_mode in {"full", "full_sliced"}:
                full_persons = _predict_pose(
                    model,
                    frame,
                    conf_thres=conf_thres,
                    imgsz=args.imgsz,
                    device=device,
                    use_half=use_half,
                    source="full",
                )
            if infer_mode in {"sliced", "full_sliced", "roi_sr_sliced"}:
                sr_backend = "opencv" if infer_mode == "roi_sr_sliced" and args.sr_backend == "off" else args.sr_backend
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
                        crop, scale = apply_sr(crop, backend=sr_backend, scale=float(args.sr_scale))
                    if crop.size == 0:
                        continue
                    tile_persons.extend(
                        _predict_pose(
                            model,
                            crop,
                            conf_thres=conf_thres,
                            imgsz=args.imgsz,
                            device=device,
                            use_half=use_half,
                            offset=(x1, y1),
                            scale=scale,
                            source="tile",
                            tile_id=str(tile.get("tile_id", "")),
                        )
                    )

            persons = dedupe_detections(
                full_persons + tile_persons,
                frame_shape=frame.shape,
                class_sensitive=False,
                iou_thres=0.42,
                overlap_min_thres=0.50,
                center_ratio_thres=0.030,
            )

            for pid, person in enumerate(persons):
                kpts = person.get("keypoints", [])
                if int(args.interpolate_occluded) == 1:
                    kpts = interpolate_occluded_keypoints(kpts, conf_thres=float(args.occlusion_conf_thres))
                vis = [1 for p in kpts if p["c"] is not None and p["c"] >= float(args.occlusion_conf_thres)]
                occlusion_score = 1.0 - (len(vis) / max(1, len(kpts)))
                person["person_idx"] = int(pid)
                person["keypoints"] = kpts
                person["occlusion_score"] = float(round(occlusion_score, 4))
                person["source"] = person.pop("_source", person.get("source", "full"))
                tile_id = person.pop("_tile_id", "")
                if tile_id:
                    person["tile_id"] = tile_id

            diag_rows.append(
                {
                    "frame": int(frame_idx),
                    "full_raw": len(full_persons),
                    "tile_raw": len(tile_persons),
                    "merged": len(persons),
                    "rear_merged": count_in_roi(persons, roi_bbox, conf_thres),
                    "avg_conf": round(sum(float(p.get("conf", 0.0)) for p in persons) / max(1, len(persons)), 4),
                }
            )

            f.write(json.dumps({"frame": frame_idx, "t": float(t), "persons": persons}, ensure_ascii=False) + "\n")

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"[INFO] exported {frame_idx} frames")

    cap.release()
    if diag_path:
        write_diagnostics(
            diag_path,
            {
                "stage": "pose_sliced_inference",
                "input": str(video_path),
                "output": str(jsonl_path),
                "model": str(model_ref),
                "conf": conf_thres,
                "imgsz": int(args.imgsz),
                "infer_mode": infer_mode,
                "slice_grid": str(args.slice_grid),
                "slice_overlap": float(args.slice_overlap),
                "slice_roi": str(args.slice_roi),
                "sr_backend": str(args.sr_backend),
                "sr_scale": float(args.sr_scale),
                "sr_cache_dir": str(sr_cache_dir) if sr_cache_dir else "",
                "sr_cache_status": str(sr_cache_meta.get("status", "")) if sr_cache_meta else "",
                "summary": summarize_diagnostics(diag_rows, mode=infer_mode, roi=roi_bbox or [0, 0, 0, 0]),
                "frames": diag_rows,
                "status": "ok",
            },
        )
    print("[DONE] JSONL:", jsonl_path)


if __name__ == "__main__":
    main()
