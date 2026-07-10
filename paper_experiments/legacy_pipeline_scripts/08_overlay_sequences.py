# scripts/08_overlay_sequences.py
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def load_mllm(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            return [x for x in obj["items"] if isinstance(x, dict)]
    except Exception:
        pass
    return []


def load_pose_centers(path: Path) -> Dict[int, Dict[int, Tuple[int, int]]]:
    out: Dict[int, Dict[int, Tuple[int, int]]] = {}
    for rec in load_jsonl(path):
        frame = rec.get("frame")
        persons = rec.get("persons", [])
        if frame is None or not isinstance(persons, list):
            continue
        row: Dict[int, Tuple[int, int]] = {}
        for p in persons:
            try:
                tid = int(p.get("track_id"))
            except Exception:
                continue
            bbox = p.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            cx = int((float(bbox[0]) + float(bbox[2])) * 0.5)
            cy = int((float(bbox[1]) + float(bbox[3])) * 0.5)
            row[tid] = (cx, cy)
        out[int(frame)] = row
    return out


def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], size: int = 24, color=(255, 255, 255)) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_path = "simhei.ttf"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/simhei.ttf"
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _safe_float(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d


def normalize_actions(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        label = str(r.get("action", r.get("label", ""))).strip()
        if not label:
            continue
        st = r.get("start_time", r.get("start", None))
        ed = r.get("end_time", r.get("end", None))
        t = r.get("time", r.get("t", None))
        if st is None and t is not None:
            st = t
        if ed is None and st is not None:
            ed = _safe_float(st) + 0.2
        if st is None and ed is None:
            continue
        stf = _safe_float(st, 0.0)
        edf = _safe_float(ed, stf + 0.2)
        if edf < stf:
            stf, edf = edf, stf
        out.append(
            {
                "track_id": r.get("track_id", r.get("id", None)),
                "label": label,
                "conf": _safe_float(r.get("confidence", r.get("conf", 0.0))),
                "start": stf,
                "end": edf,
            }
        )
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def normalize_transcript(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        text = str(r.get("text", "")).strip()
        if not text:
            continue
        st = _safe_float(r.get("start", r.get("time_start", 0.0)))
        ed = _safe_float(r.get("end", r.get("time_end", st + 0.1)))
        if ed < st:
            st, ed = ed, st
        out.append({"start": st, "end": ed, "text": text})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def active_actions(rows: List[dict], t: float, window: float = 0.25, max_lines: int = 6) -> List[dict]:
    picked = [r for r in rows if r["start"] - window <= t <= r["end"] + window]
    picked.sort(key=lambda x: x.get("conf", 0.0), reverse=True)
    return picked[:max_lines]


def active_transcript(rows: List[dict], t: float) -> Optional[str]:
    for r in rows:
        if r["start"] <= t <= r["end"]:
            return r["text"]
    return None


def active_group_event(rows: List[dict], t: float) -> Optional[dict]:
    for r in rows:
        st = _safe_float(r.get("start_time", r.get("start", 0.0)))
        ed = _safe_float(r.get("end_time", r.get("end", st)))
        if st <= t <= ed:
            return r
    return None


def active_mllm(rows: List[dict], t: float, topn: int = 3) -> List[dict]:
    out = []
    for r in rows:
        st = _safe_float(r.get("start_time", 0.0))
        ed = _safe_float(r.get("end_time", st))
        if st <= t <= ed:
            out.append(r)
    out.sort(key=lambda x: _safe_float(x.get("mllm_confidence", 0.0)), reverse=True)
    return out[:topn]


def _ffmpeg_supports_encoder(encoder_name: str) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode != 0:
            return False
        haystack = f"{proc.stdout}\n{proc.stderr}".lower()
        return encoder_name.lower() in haystack
    except Exception:
        return False


def main():
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--actions", type=str, required=True)
    parser.add_argument("--transcript", type=str, required=True)
    parser.add_argument("--group_src", type=str, default="")
    parser.add_argument("--mllm_src", type=str, default="")
    parser.add_argument("--pose_tracks", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="demo")
    parser.add_argument("--mux_audio", type=int, default=1)
    parser.add_argument("--out_video", type=str, default="")
    parser.add_argument("--video_codec", type=str, default="auto")
    parser.add_argument("--nvenc_preset", type=str, default="p4")
    parser.add_argument("--x264_preset", type=str, default="fast")
    parser.add_argument("--quality", type=str, default="23")
    args = parser.parse_args()

    video_path = Path(args.video)
    actions_path = Path(args.actions)
    transcript_path = Path(args.transcript)
    group_path = Path(args.group_src) if args.group_src else None
    mllm_path = Path(args.mllm_src) if args.mllm_src else None
    pose_tracks_path = Path(args.pose_tracks) if args.pose_tracks else None
    out_dir = Path(args.out_dir)

    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    if not actions_path.is_absolute():
        actions_path = (base_dir / actions_path).resolve()
    if not transcript_path.is_absolute():
        transcript_path = (base_dir / transcript_path).resolve()
    if group_path and not group_path.is_absolute():
        group_path = (base_dir / group_path).resolve()
    if mllm_path and not mllm_path.is_absolute():
        mllm_path = (base_dir / mllm_path).resolve()
    if pose_tracks_path and not pose_tracks_path.is_absolute():
        pose_tracks_path = (base_dir / pose_tracks_path).resolve()
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_video = Path(args.out_video) if args.out_video else (out_dir / f"{args.name}_overlay.mp4")
    if not out_video.is_absolute():
        out_video = (base_dir / out_video).resolve()
    out_video.parent.mkdir(parents=True, exist_ok=True)
    temp_video = out_dir / f"{args.name}_overlay_noaudio.mp4"

    actions = normalize_actions(load_jsonl(actions_path))
    transcript = normalize_transcript(load_jsonl(transcript_path))
    group_events = load_jsonl(group_path) if group_path and group_path.exists() else []
    mllm_items = load_mllm(mllm_path) if mllm_path and mllm_path.exists() else []
    pose_centers = load_pose_centers(pose_tracks_path) if pose_tracks_path and pose_tracks_path.exists() else {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        # HUD title
        cv2.rectangle(frame, (8, 8), (min(820, width - 8), 44), (0, 0, 0), -1)
        frame = draw_text(frame, f"t={t:.2f}s frame={frame_idx}", (16, 12), size=22, color=(180, 255, 180))

        # Active actions panel
        act = active_actions(actions, t=t)
        panel_h = 32 + 28 * max(1, len(act))
        cv2.rectangle(frame, (8, 52), (min(880, width - 8), 52 + panel_h), (0, 0, 0), -1)
        frame = draw_text(frame, "Actions", (16, 58), size=22, color=(255, 255, 255))
        if act:
            y = 86
            for a in act:
                tid = a.get("track_id")
                line = f"ID {tid}: {a['label']} ({a.get('conf', 0.0):.2f})"
                frame = draw_text(frame, line[:58], (16, y), size=20, color=(255, 240, 120))
                y += 28
        else:
            frame = draw_text(frame, "(none)", (16, 86), size=20, color=(180, 180, 180))

        # Group event + interaction pairs
        g = active_group_event(group_events, t=t)
        if g is not None:
            label = str(g.get("group_event", "unknown"))
            frame = draw_text(frame, f"Group: {label}", (16, min(height - 190, 260)), size=22, color=(120, 220, 255))
            pairs = g.get("interaction_pairs", [])
            y = min(height - 158, 288)
            for p in pairs[:4]:
                a_id = p.get("id_a")
                b_id = p.get("id_b")
                p_type = p.get("type", "link")
                p_score = _safe_float(p.get("score", 0.0))
                frame = draw_text(frame, f"Pair {a_id}-{b_id} {p_type} {p_score:.2f}", (16, y), size=20, color=(120, 255, 180))
                y += 26

            # Draw interaction lines if pose centers are available.
            centers = pose_centers.get(frame_idx, {})
            for p in pairs:
                try:
                    a_id = int(p.get("id_a"))
                    b_id = int(p.get("id_b"))
                except Exception:
                    continue
                if a_id not in centers or b_id not in centers:
                    continue
                p1 = centers[a_id]
                p2 = centers[b_id]
                score = float(np.clip(_safe_float(p.get("score", 0.5)), 0.0, 1.0))
                color = (0, int(255 * score), 255 - int(180 * score))
                thickness = 1 + int(3 * score)
                cv2.line(frame, p1, p2, color, thickness)
                cv2.circle(frame, p1, 3, color, -1)
                cv2.circle(frame, p2, 3, color, -1)

        # MLLM panel
        mllm_active = active_mllm(mllm_items, t=t)
        if mllm_active:
            x0 = max(width - 520, 8)
            cv2.rectangle(frame, (x0, 8), (width - 8, 130), (0, 0, 0), -1)
            frame = draw_text(frame, "MLLM", (x0 + 8, 12), size=22, color=(255, 200, 120))
            y = 40
            for item in mllm_active:
                line = f"ID {item.get('track_id')}: {item.get('mllm_label')} ({_safe_float(item.get('mllm_confidence')):.2f})"
                frame = draw_text(frame, line[:50], (x0 + 8, y), size=20, color=(255, 220, 160))
                y += 26

        # Transcript
        tr = active_transcript(transcript, t=t)
        if tr:
            tr = tr[:96]
            cv2.rectangle(frame, (0, height - 72), (width, height), (0, 0, 0), -1)
            frame = draw_text(frame, f"Speech: {tr}", (16, height - 62), size=24, color=(255, 255, 255))

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if int(args.mux_audio) == 1:
        try:
            selected_codec = str(args.video_codec or "auto").strip().lower()
            if selected_codec in {"", "auto"}:
                selected_codec = "h264_nvenc" if _ffmpeg_supports_encoder("h264_nvenc") else "libx264"
            if selected_codec not in {"libx264", "h264_nvenc"}:
                selected_codec = "libx264"

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_video),
                "-i",
                str(video_path),
            ]
            if selected_codec == "h264_nvenc":
                ffmpeg_cmd += [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    str(args.nvenc_preset),
                    "-cq",
                    str(args.quality),
                    "-b:v",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                ]
            else:
                ffmpeg_cmd += [
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(args.quality),
                    "-preset",
                    str(args.x264_preset),
                    "-pix_fmt",
                    "yuv420p",
                ]

            ffmpeg_cmd += [
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                str(out_video),
            ]
            print(f"[INFO] mux codec={selected_codec}")
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if temp_video.exists():
                temp_video.unlink()
        except Exception:
            if out_video.exists():
                out_video.unlink()
            temp_video.replace(out_video)
    else:
        if out_video.exists():
            out_video.unlink()
        temp_video.replace(out_video)

    print(f"[DONE] overlay saved to: {out_video}")


if __name__ == "__main__":
    main()
