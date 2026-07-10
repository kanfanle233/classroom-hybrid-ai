import argparse
import json
import math
import pickle
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


CLASS_LABELS = {
    0: "listen",
    1: "distract",
    2: "phone",
    3: "doze",
    4: "chat",
    5: "note",
    6: "raise_hand",
    7: "stand",
    8: "read",
}

ACTION_TO_CODE = {v: k for k, v in CLASS_LABELS.items()}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_tracks_by_frame(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    tracks: Dict[int, List[Dict[str, Any]]] = {}
    if not path.exists():
        return tracks
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            frame = row.get("frame")
            persons = row.get("persons", [])
            if isinstance(frame, int) and isinstance(persons, list):
                tracks[frame] = [p for p in persons if isinstance(p, dict)]
    return tracks


def _mid(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float]:
    return ((_safe_float(a.get("x")) + _safe_float(b.get("x"))) * 0.5, (_safe_float(a.get("y")) + _safe_float(b.get("y"))) * 0.5)


def _rule_action(kpts: List[Dict[str, Any]], bbox: List[Any], h_median: float) -> str:
    if len(kpts) < 13:
        return "listen"

    def _ok(i: int) -> bool:
        return i < len(kpts) and isinstance(kpts[i], dict)

    if not all(_ok(i) for i in [0, 5, 6, 9, 10, 11, 12]):
        return "listen"

    ls, rs = kpts[5], kpts[6]
    lw, rw = kpts[9], kpts[10]
    nose = kpts[0]
    lh, rh = kpts[11], kpts[12]

    sh = _mid(ls, rs)
    hp = _mid(lh, rh)
    torso = abs(hp[1] - sh[1]) + 1e-6

    # Raise hand
    if _safe_float(lw.get("y")) < _safe_float(ls.get("y")) - 0.15 * torso:
        return "raise_hand"
    if _safe_float(rw.get("y")) < _safe_float(rs.get("y")) - 0.15 * torso:
        return "raise_hand"

    # Head down / doze
    shoulder_mid_y = (_safe_float(ls.get("y")) + _safe_float(rs.get("y"))) * 0.5
    nose_y = _safe_float(nose.get("y"))
    if nose_y > shoulder_mid_y + 0.18 * torso:
        return "doze"
    if nose_y > shoulder_mid_y + 0.10 * torso:
        return "note"

    # Stand
    h = _safe_float(bbox[3]) - _safe_float(bbox[1]) if len(bbox) >= 4 else 0.0
    if h_median > 1.0 and h > 1.18 * h_median:
        return "stand"

    return "listen"


def _merge_actions(rows: List[Dict[str, Any]], gap_tol: float = 0.22) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    rows = sorted(rows, key=lambda x: (x["track_id"], x["start_time"], x["end_time"]))
    out: List[Dict[str, Any]] = [rows[0]]
    for row in rows[1:]:
        last = out[-1]
        same_track = row["track_id"] == last["track_id"]
        same_action = row["action"] == last["action"]
        close = row["start_time"] <= last["end_time"] + gap_tol
        if same_track and same_action and close:
            last["end_time"] = max(last["end_time"], row["end_time"])
            last["end_frame"] = max(last["end_frame"], row["end_frame"])
            last["duration"] = float(last["end_time"] - last["start_time"])
            last["conf"] = float((last["conf"] + row["conf"]) * 0.5)
        else:
            out.append(row)
    return out


def _export_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run_rules_backend(
    *,
    frame_tracks: Dict[int, List[Dict[str, Any]]],
    fps: float,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Any]]]:
    heights_by_tid: Dict[int, List[float]] = defaultdict(list)
    for _, persons in frame_tracks.items():
        for p in persons:
            tid = p.get("track_id")
            bbox = p.get("bbox", [])
            if not isinstance(tid, int) or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            h = _safe_float(bbox[3]) - _safe_float(bbox[1])
            if h > 0:
                heights_by_tid[tid].append(h)

    median_h: Dict[int, float] = {}
    for tid, hs in heights_by_tid.items():
        hs = sorted(hs)
        median_h[tid] = hs[len(hs) // 2] if hs else 0.0

    raw_rows: List[Dict[str, Any]] = []
    for frame in sorted(frame_tracks.keys()):
        persons = frame_tracks[frame]
        for p in persons:
            tid = p.get("track_id")
            bbox = p.get("bbox", [])
            kpts = p.get("keypoints", [])
            if not isinstance(tid, int) or not isinstance(bbox, list) or not isinstance(kpts, list):
                continue
            action = _rule_action(kpts, bbox, h_median=median_h.get(tid, 0.0))
            conf_vals = []
            for kp in kpts:
                if isinstance(kp, dict):
                    conf_vals.append(_safe_float(kp.get("c"), 0.0))
            conf = (sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0
            st = frame / fps
            ed = (frame + 1) / fps
            raw_rows.append(
                {
                    "track_id": tid,
                    "action": action,
                    "action_code": ACTION_TO_CODE.get(action, 0),
                    "conf": float(max(0.0, min(1.0, conf))),
                    "start_time": float(st),
                    "end_time": float(ed),
                    "start_frame": int(frame),
                    "end_frame": int(frame + 1),
                    "frame": int(frame),
                    "t": float(st),
                    "source": "rule_baseline",
                }
            )

    merged = _merge_actions(raw_rows)
    return merged, {}


def _run_slowfast_backend(
    *,
    video_path: Path,
    frame_tracks: Dict[int, List[Dict[str, Any]]],
    model_weight: str,
    stride: int,
    clip_duration: int,
    crop_size: int,
    device: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Any]]]:
    # Lazy imports so `--model_mode rules` works without heavy deps.
    import sys

    import torch
    import torchvision
    from pytorchvideo.models.hub import slowfast_r50
    from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import Compose, Lambda
    from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

    try:
        import torchvision.transforms.functional_tensor as F_t  # type: ignore
    except Exception:
        from torchvision.transforms import functional as F_t  # type: ignore

        sys.modules["torchvision.transforms.functional_tensor"] = F_t

    if device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = slowfast_r50(pretrained=True)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = torch.nn.Linear(in_features, len(CLASS_LABELS))
    if model_weight and Path(model_weight).exists():
        state = torch.load(model_weight, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    model = model.eval().to(device)

    current_embedding = {"vec": None}

    def hook_fn(module, module_input, module_output):
        if module_input and len(module_input) > 0:
            current_embedding["vec"] = module_input[0].detach().cpu().numpy()

    model.blocks[-1].proj.register_forward_hook(hook_fn)

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(clip_duration),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                ShortSideScale(size=256),
                CenterCropVideo(crop_size),
            ]
        ),
    )

    def _pack_pathways(video_tensor):
        fast_pathway = video_tensor
        slow_pathway = torch.index_select(
            video_tensor,
            1,
            torch.linspace(0, video_tensor.shape[1] - 1, video_tensor.shape[1] // 4).long(),
        )
        return [slow_pathway, fast_pathway]

    def _crop_person(frame, bbox, padding=0.2):
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w_box = x2 - x1
        h_box = y2 - y1
        cx, cy = x1 + w_box // 2, y1 + h_box // 2
        size = int(max(w_box, h_box) * (1 + padding))
        hs = size // 2
        x1n = max(0, cx - hs)
        y1n = max(0, cy - hs)
        x2n = min(w_img, cx + hs)
        y2n = min(h_img, cy + hs)
        crop = frame[y1n:y2n, x1n:x2n]
        if crop.size == 0:
            return cv2.resize(frame, (crop_size, crop_size))
        return cv2.resize(crop, (crop_size, crop_size))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    buffers = defaultdict(lambda: deque(maxlen=max(clip_duration * 2, clip_duration)))
    actions: List[Dict[str, Any]] = []
    embeddings = defaultdict(list)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for p in frame_tracks.get(frame_idx, []):
            tid = p.get("track_id")
            bbox = p.get("bbox")
            if not isinstance(tid, int) or not isinstance(bbox, list):
                continue
            crop = _crop_person(frame, bbox)
            buffers[tid].append(crop)

            if len(buffers[tid]) >= clip_duration and frame_idx % stride == 0:
                buffer = list(buffers[tid])
                idx = np.linspace(0, len(buffer) - 1, clip_duration).astype(int)
                frames = [cv2.cvtColor(buffer[i], cv2.COLOR_BGR2RGB) for i in idx]
                tensor = torch.from_numpy(np.array(frames)).float().permute(3, 0, 1, 2)
                video_data = transform({"video": tensor})["video"]
                inputs = [x.to(device)[None, ...] for x in _pack_pathways(video_data)]
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(model(inputs), dim=1)
                top_val, top_idx = torch.max(probs, dim=1)
                label_id = int(top_idx.item())
                conf = float(top_val.item())

                st = frame_idx / fps
                ed = (frame_idx + stride) / fps
                actions.append(
                    {
                        "track_id": tid,
                        "action": CLASS_LABELS.get(label_id, "listen"),
                        "action_code": label_id,
                        "conf": conf,
                        "start_time": st,
                        "end_time": ed,
                        "start_frame": frame_idx,
                        "end_frame": frame_idx + stride,
                        "frame": frame_idx,
                        "t": st,
                        "source": "slowfast",
                    }
                )

                emb = current_embedding.get("vec")
                if emb is not None:
                    embeddings[tid].append(([int(frame_idx), int(frame_idx + stride)], emb[0].astype(np.float32).reshape(-1).tolist()))

        frame_idx += 1

    cap.release()
    actions = _merge_actions(actions)
    return actions, dict(embeddings)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--pose", type=str, required=True, help="pose_tracks_smooth.jsonl")
    parser.add_argument("--out", type=str, required=True, help="actions.jsonl")
    parser.add_argument("--model_weight", type=str, default="", help="custom .pth for slowfast head")
    parser.add_argument("--emb_out", type=str, default="", help="output embeddings.pkl")
    parser.add_argument("--device", type=str, default="", help="cuda/cpu")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--clip_duration", type=int, default=32)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--save_keyframes", type=int, default=0)
    parser.add_argument("--keyframe_dir", type=str, default="")
    parser.add_argument("--model_mode", choices=["auto", "slowfast", "rules"], default="auto")
    args = parser.parse_args()

    video_path = Path(args.video)
    pose_path = Path(args.pose)
    out_path = Path(args.out)
    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    if not pose_path.is_absolute():
        pose_path = (base_dir / pose_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    emb_out = Path(args.emb_out) if args.emb_out else out_path.parent / "embeddings.pkl"
    if not emb_out.is_absolute():
        emb_out = (base_dir / emb_out).resolve()

    if not pose_path.exists():
        raise FileNotFoundError(f"pose tracks not found: {pose_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    frame_tracks = load_tracks_by_frame(pose_path)
    if not frame_tracks:
        _export_jsonl(out_path, [])
        with emb_out.open("wb") as f:
            pickle.dump({}, f)
        print(f"[WARN] empty pose tracks, wrote empty actions: {out_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    actions: List[Dict[str, Any]] = []
    embeddings: Dict[int, List[Any]] = {}

    mode = args.model_mode
    if mode == "rules":
        actions, embeddings = _run_rules_backend(frame_tracks=frame_tracks, fps=fps)
    else:
        try:
            actions, embeddings = _run_slowfast_backend(
                video_path=video_path,
                frame_tracks=frame_tracks,
                model_weight=args.model_weight,
                stride=int(args.stride),
                clip_duration=int(args.clip_duration),
                crop_size=int(args.crop_size),
                device=args.device,
            )
            print("[INFO] slowfast backend completed.")
        except Exception as exc:
            if mode == "slowfast":
                raise RuntimeError(f"slowfast backend failed (no fallback in strict mode): {exc}") from exc
            print(f"[WARN] slowfast backend unavailable, fallback to rules baseline: {exc}")
            actions, embeddings = _run_rules_backend(frame_tracks=frame_tracks, fps=fps)

    _export_jsonl(out_path, actions)
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    with emb_out.open("wb") as f:
        pickle.dump(embeddings, f)

    print(f"[DONE] actions: {out_path} ({len(actions)})")
    print(f"[DONE] embeddings: {emb_out}")
    print(f"[INFO] mode: {mode}")


if __name__ == "__main__":
    main()
