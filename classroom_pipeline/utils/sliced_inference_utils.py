import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


BBox = List[float]
SR_BACKENDS = {
    "off",
    "opencv",
    "realesrgan",
    "basicvsrpp",
    "realbasicvsr",
    "nvidia_vsr",
    "maxine_vfx",
}
PREPROCESS_MODES = {"off", "denoise", "deblock", "deblur", "artifact_deblur", "clahe"}


def parse_grid(raw: str) -> Tuple[int, int]:
    text = str(raw or "2x2").strip().lower().replace("*", "x")
    if "x" not in text:
        n = max(1, int(text))
        return n, n
    left, right = text.split("x", 1)
    return max(1, int(left)), max(1, int(right))


def resolve_roi(frame_shape: Sequence[int], raw: str) -> BBox:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    text = str(raw or "auto_rear").strip().lower()
    if text in {"full", "all"}:
        return [0.0, 0.0, float(w), float(h)]
    if text in {"auto_rear", "rear", "mid_rear"}:
        # Classroom front-view videos compress rear rows into the upper half.
        # Include part of the middle row because the difficult small targets are not only at the top edge.
        return [0.0, 0.0, float(w), float(round(h * 0.62))]
    if text in {"auto_upper", "upper"}:
        return [0.0, 0.0, float(w), float(round(h * 0.50))]

    parts = [float(p.strip()) for p in text.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"Invalid ROI: {raw!r}. Use full, auto_rear, or x1,y1,x2,y2.")
    x1, y1, x2, y2 = parts
    if max(parts) <= 1.0:
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    return [
        max(0.0, min(float(w), x1)),
        max(0.0, min(float(h), y1)),
        max(0.0, min(float(w), x2)),
        max(0.0, min(float(h), y2)),
    ]


def build_tiles(frame_shape: Sequence[int], *, grid: str, overlap: float, roi: str) -> List[Dict[str, Any]]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = resolve_roi(frame_shape, roi)
    grid_text = str(grid or "2x2").strip().lower()
    if grid_text in {"rear_dense", "dense"}:
        roi_w = max(1.0, x2 - x1)
        roi_h = max(1.0, y2 - y1)
        rows, cols = (3, 4) if roi_w / max(1.0, roi_h) >= 2.0 else (4, 4)
    elif grid_text in {"rear_adaptive"}:
        roi_w = max(1.0, x2 - x1)
        roi_h = max(1.0, y2 - y1)
        rows, cols = (3, 4) if roi_w / max(1.0, roi_h) >= 2.0 else (3, 3)
    elif grid_text in {"auto", "adaptive"}:
        roi_w = max(1.0, x2 - x1)
        roi_h = max(1.0, y2 - y1)
        # Front classroom videos are wide; extra columns reduce rear-row tiny-person shrinkage.
        rows, cols = (2, 3) if roi_w / max(1.0, roi_h) >= 2.0 else (3, 3)
    else:
        rows, cols = parse_grid(grid)
    roi_w = max(1.0, x2 - x1)
    roi_h = max(1.0, y2 - y1)
    ov = max(0.0, min(0.75, float(overlap)))
    tile_w = roi_w / max(1.0, cols - (cols - 1) * ov)
    tile_h = roi_h / max(1.0, rows - (rows - 1) * ov)
    step_x = tile_w * (1.0 - ov)
    step_y = tile_h * (1.0 - ov)

    tiles: List[Dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            tx1 = x1 + c * step_x
            ty1 = y1 + r * step_y
            tx2 = tx1 + tile_w
            ty2 = ty1 + tile_h
            if c == cols - 1:
                tx2 = x2
                tx1 = max(x1, tx2 - tile_w)
            if r == rows - 1:
                ty2 = y2
                ty1 = max(y1, ty2 - tile_h)
            ix1 = int(max(0, min(w - 1, round(tx1))))
            iy1 = int(max(0, min(h - 1, round(ty1))))
            ix2 = int(max(ix1 + 1, min(w, round(tx2))))
            iy2 = int(max(iy1 + 1, min(h, round(ty2))))
            tiles.append({"tile_id": f"r{r}c{c}", "bbox": [ix1, iy1, ix2, iy2], "row": r, "col": c})
    return tiles


def apply_sr(crop: Any, *, backend: str, scale: float) -> Tuple[Any, float]:
    backend = str(backend or "off").strip().lower()
    scale = max(1.0, float(scale))
    if backend in {"off", "none"} or scale <= 1.0:
        return crop, 1.0
    if backend == "opencv":
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return enlarged, scale
    if backend in {"realesrgan", "basicvsrpp", "realbasicvsr", "nvidia_vsr", "maxine_vfx"}:
        raise RuntimeError(
            f"sr_backend={backend} requires a prebuilt ROI SR cache or an external SR command. "
            "Run 02c_build_rear_roi_sr_cache.py first, or use sr_backend=opencv/off."
        )
    raise ValueError(f"Unknown sr_backend: {backend}")


def apply_roi_preprocess(image: Any, *, mode: str) -> Any:
    mode = str(mode or "off").strip().lower()
    if mode in {"off", "none"}:
        return image
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unknown sr_preprocess: {mode}")

    out = image
    if mode in {"denoise", "artifact_deblur"}:
        # Conservative chroma denoise: enough to reduce compression speckle without erasing faces.
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    if mode in {"deblock", "artifact_deblur"}:
        out = cv2.bilateralFilter(out, d=5, sigmaColor=24, sigmaSpace=7)
    if mode in {"clahe", "artifact_deblur"}:
        ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycc)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        y = clahe.apply(y)
        out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    if mode in {"deblur", "artifact_deblur"}:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
        out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)
    return out


def sr_external_command_env(backend: str) -> str:
    backend = str(backend or "").strip().lower()
    return {
        "realesrgan": "REALESRGAN_SR_COMMAND",
        "basicvsrpp": "BASICVSRPP_SR_COMMAND",
        "realbasicvsr": "REALBASICVSR_SR_COMMAND",
        "nvidia_vsr": "NVIDIA_VSR_SR_COMMAND",
        "maxine_vfx": "MAXINE_VFX_SR_COMMAND",
    }.get(backend, "")


def check_sr_backend_available(backend: str, external_command: str = "") -> Dict[str, Any]:
    backend = str(backend or "off").strip().lower()
    if backend in {"off", "none", "opencv"}:
        return {"backend": backend, "available": True, "reason": "builtin"}
    if backend not in SR_BACKENDS:
        return {"backend": backend, "available": False, "reason": "unknown_backend"}
    command = str(external_command or "").strip()
    env_name = sr_external_command_env(backend)
    if not command and env_name:
        command = os.environ.get(env_name, "").strip()
    if command:
        exe = command.split()[0].strip('"')
        exists = bool(shutil.which(exe) or Path(exe).exists())
        return {
            "backend": backend,
            "available": exists,
            "reason": "external_command" if exists else "external_command_not_found",
            "external_command_env": env_name,
            "external_command": command,
        }
    return {
        "backend": backend,
        "available": False,
        "reason": "missing_external_command",
        "external_command_env": env_name,
    }


def load_sr_cache_metadata(cache_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not cache_dir:
        return None
    cache_dir = Path(cache_dir)
    report = cache_dir / "sr_cache.report.json"
    if not report.exists():
        return None
    try:
        meta = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return None
    if str(meta.get("status", "")).lower() != "ok":
        return None
    frame_dir = Path(meta.get("sr_frame_dir") or (cache_dir / "frames"))
    if not frame_dir.is_absolute():
        frame_dir = (cache_dir / frame_dir).resolve()
    meta["sr_frame_dir"] = str(frame_dir)
    return meta


def read_sr_cache_frame(meta: Optional[Dict[str, Any]], frame_idx: int) -> Optional[Any]:
    if not meta:
        return None
    frame_dir = Path(str(meta.get("sr_frame_dir", "")))
    frame_path = frame_dir / f"{int(frame_idx):06d}.png"
    if not frame_path.exists():
        return None
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    return img


def crop_tile_from_sr_roi(sr_roi: Any, roi_bbox: Sequence[float], tile_bbox: Sequence[float], scale: float) -> Tuple[Any, float]:
    rx1, ry1, _, _ = [float(v) for v in roi_bbox]
    x1, y1, x2, y2 = [float(v) for v in tile_bbox]
    scale = max(1.0, float(scale))
    sx1 = int(max(0, round((x1 - rx1) * scale)))
    sy1 = int(max(0, round((y1 - ry1) * scale)))
    sx2 = int(min(sr_roi.shape[1], round((x2 - rx1) * scale)))
    sy2 = int(min(sr_roi.shape[0], round((y2 - ry1) * scale)))
    if sx2 <= sx1 or sy2 <= sy1:
        return sr_roi[0:0, 0:0], scale
    return sr_roi[sy1:sy2, sx1:sx2], scale


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def bbox_overlap_min(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, min(area_a, area_b))


def bbox_center(b: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in b]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _same_class(a: Dict[str, Any], b: Dict[str, Any], class_sensitive: bool) -> bool:
    if not class_sensitive:
        return True
    av = a.get("cls_id", a.get("label"))
    bv = b.get("cls_id", b.get("label"))
    return av == bv


def dedupe_detections(
    detections: Iterable[Dict[str, Any]],
    *,
    frame_shape: Sequence[int],
    class_sensitive: bool = False,
    iou_thres: float = 0.45,
    overlap_min_thres: float = 0.55,
    center_ratio_thres: float = 0.035,
) -> List[Dict[str, Any]]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    diag = math.hypot(w, h)
    center_thres = max(10.0, float(center_ratio_thres) * diag)
    sorted_dets = sorted(detections, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []
    for det in sorted_dets:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        duplicate = False
        cx, cy = bbox_center(bbox)
        for prev in kept:
            if not _same_class(det, prev, class_sensitive):
                continue
            pb = prev.get("bbox", [])
            pcx, pcy = bbox_center(pb)
            center_dist = math.hypot(cx - pcx, cy - pcy)
            if (
                bbox_iou(bbox, pb) >= float(iou_thres)
                or bbox_overlap_min(bbox, pb) >= float(overlap_min_thres)
                or center_dist <= center_thres
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def count_in_roi(detections: Iterable[Dict[str, Any]], roi_bbox: Sequence[float], conf_thres: float) -> int:
    rx1, ry1, rx2, ry2 = [float(v) for v in roi_bbox]
    count = 0
    for det in detections:
        if float(det.get("conf", 0.0)) < float(conf_thres):
            continue
        cx, cy = bbox_center(det.get("bbox", [0, 0, 0, 0]))
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            count += 1
    return count


def summarize_diagnostics(rows: List[Dict[str, Any]], *, mode: str, roi: Sequence[float]) -> Dict[str, Any]:
    if not rows:
        return {"mode": mode, "roi": list(roi), "frames": 0}
    return {
        "mode": mode,
        "roi": [round(float(v), 3) for v in roi],
        "frames": len(rows),
        "full_raw_total": int(sum(int(r.get("full_raw", 0)) for r in rows)),
        "tile_raw_total": int(sum(int(r.get("tile_raw", 0)) for r in rows)),
        "merged_total": int(sum(int(r.get("merged", 0)) for r in rows)),
        "rear_merged_total": int(sum(int(r.get("rear_merged", 0)) for r in rows)),
        "sample_frames": rows[:5],
    }


def write_diagnostics(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
