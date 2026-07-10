import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classroom_pipeline.utils.sliced_inference_utils import (
    apply_roi_preprocess,
    apply_sr,
    check_sr_backend_available,
    resolve_roi,
    sr_external_command_env,
)


def _resolve(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except OSError:
                pass
    return total


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_roi_frames(
    *,
    video_path: Path,
    out_dir: Path,
    roi: str,
    max_frames: int,
    write_lr: bool,
    backend: str,
    scale: float,
    preprocess: str,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    written = 0
    roi_bbox: List[float] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break
        roi_bbox = resolve_roi(frame.shape, roi)
        x1, y1, x2, y2 = [int(round(v)) for v in roi_bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            frame_idx += 1
            continue
        crop = apply_roi_preprocess(crop, mode=preprocess)
        if write_lr:
            image = crop
        else:
            image, _ = apply_sr(crop, backend=backend, scale=scale)
        cv2.imwrite(str(out_dir / f"{frame_idx:06d}.png"), image)
        written += 1
        frame_idx += 1
    cap.release()
    return {
        "fps": float(fps),
        "frames_seen": int(frame_idx),
        "frames_written": int(written),
        "roi": [round(float(v), 3) for v in roi_bbox],
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build rear-row ROI super-resolution frame cache.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--backend",
        choices=["off", "opencv", "realesrgan", "basicvsrpp", "realbasicvsr", "nvidia_vsr", "maxine_vfx"],
        default="off",
    )
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--roi", default="auto_rear")
    parser.add_argument("--preprocess", choices=["off", "denoise", "deblock", "deblur", "artifact_deblur", "clahe"], default="off")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--external_command", default="")
    parser.add_argument("--allow_unavailable", type=int, default=0)
    args = parser.parse_args()

    video_path = _resolve(base_dir, args.video)
    out_dir = _resolve(base_dir, args.out_dir)
    report_path = out_dir / "sr_cache.report.json"
    backend = str(args.backend).strip().lower()
    scale = max(1.0, float(args.scale))
    started = time.perf_counter()

    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    availability = check_sr_backend_available(backend, args.external_command)
    if backend in {"off", "none"}:
        payload = {
            "stage": "rear_roi_sr_cache",
            "status": "skipped",
            "backend": backend,
            "reason": "sr_backend_off",
            "input": str(video_path),
            "out_dir": str(out_dir),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        _write_report(report_path, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return

    sr_dir = out_dir / "frames"
    lr_dir = out_dir / "lr_frames"
    sr_dir.mkdir(parents=True, exist_ok=True)

    if backend == "opencv":
        stats = _extract_roi_frames(
            video_path=video_path,
            out_dir=sr_dir,
            roi=str(args.roi),
            max_frames=int(args.max_frames),
            write_lr=False,
            backend=backend,
            scale=scale,
            preprocess=str(args.preprocess),
        )
        payload = {
            "stage": "rear_roi_sr_cache",
            "status": "ok",
            "backend": backend,
            "input": str(video_path),
            "out_dir": str(out_dir),
            "sr_frame_dir": str(sr_dir),
            "scale": scale,
            "preprocess": str(args.preprocess),
            "roi": stats.get("roi", []),
            "fps": stats.get("fps"),
            "frames_seen": stats.get("frames_seen"),
            "frames_written": stats.get("frames_written"),
            "cache_bytes": _dir_bytes(out_dir),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        _write_report(report_path, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return

    command = str(args.external_command or "").strip()
    env_name = sr_external_command_env(backend)
    if not command and env_name:
        command = str(os.environ.get(env_name, "")).strip()

    if not availability.get("available") or not command:
        payload = {
            "stage": "rear_roi_sr_cache",
            "status": "unavailable",
            "backend": backend,
            "input": str(video_path),
            "out_dir": str(out_dir),
            "scale": scale,
            "availability": availability,
            "reason": availability.get("reason", "missing_external_command"),
            "required_env": env_name,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        _write_report(report_path, payload)
        print(json.dumps(payload, ensure_ascii=False))
        if int(args.allow_unavailable) == 1:
            return
        raise RuntimeError(f"SR backend unavailable: {backend}. Set {env_name} or use --external_command.")

    stats = _extract_roi_frames(
        video_path=video_path,
        out_dir=lr_dir,
        roi=str(args.roi),
        max_frames=int(args.max_frames),
        write_lr=True,
        backend="off",
        scale=1.0,
        preprocess=str(args.preprocess),
    )
    formatted = command.format(
        input=str(lr_dir),
        output=str(sr_dir),
        scale=str(scale),
        backend=backend,
        video=str(video_path),
    )
    sr_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(formatted, shell=True, check=False)
    frames_written = len(list(sr_dir.glob("*.png")))
    status = "ok" if completed.returncode == 0 and frames_written > 0 else "failed"
    payload = {
        "stage": "rear_roi_sr_cache",
        "status": status,
        "backend": backend,
        "input": str(video_path),
        "out_dir": str(out_dir),
        "lr_frame_dir": str(lr_dir),
        "sr_frame_dir": str(sr_dir),
        "scale": scale,
        "preprocess": str(args.preprocess),
        "roi": stats.get("roi", []),
        "fps": stats.get("fps"),
        "frames_seen": stats.get("frames_seen"),
        "lr_frames_written": stats.get("frames_written"),
        "frames_written": frames_written,
        "external_command": formatted,
        "external_returncode": int(completed.returncode),
        "cache_bytes": _dir_bytes(out_dir),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    _write_report(report_path, payload)
    print(json.dumps(payload, ensure_ascii=False))
    if status != "ok":
        raise RuntimeError(f"SR external command failed for {backend}: {formatted}")


if __name__ == "__main__":
    main()
