import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ====== COCO-Pose 关键点索引 ======
# 0 nose
# 5 left_shoulder, 6 right_shoulder
# 7 left_elbow, 8 right_elbow
# 9 left_wrist, 10 right_wrist
# 11 left_hip, 12 right_hip

NOSE = 0
LS, RS = 5, 6
LW, RW = 9, 10
LH, RH = 11, 12


def mid_point(a, b):
    return ((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0)


def dist_y(a, b):
    return a["y"] - b["y"]


def torso_length(kpts):
    """肩中点到髋中点的距离（用于尺度归一化）"""
    sh = mid_point(kpts[LS], kpts[RS])
    hp = mid_point(kpts[LH], kpts[RH])
    return abs(hp[1] - sh[1]) + 1e-6


def load_tracks(path):
    tracks = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            frame = rec["frame"]
            t = rec["t"]
            for p in rec["persons"]:
                tracks[p["track_id"]].append({
                    "frame": frame,
                    "t": t,
                    "bbox": p["bbox"],
                    "keypoints": p["keypoints"]
                })
    return tracks


def detect_actions(tracks, fps=25.0):
    events = []

    # ====== 规则参数（课堂经验值） ======
    raise_hand_frames = int(0.5 * fps)   # ≥0.5s
    head_down_frames = int(0.7 * fps)    # ≥0.7s
    stand_frames = int(0.7 * fps)

    for tid, seq in tracks.items():
        if len(seq) < 30:
            continue

        # 用 bbox 高度中位数作为“坐姿基线”
        heights = [(s["bbox"][3] - s["bbox"][1]) for s in seq]
        median_h = sorted(heights)[len(heights)//2]

        # 状态缓存
        rh_cnt = lh_cnt = hd_cnt = st_cnt = 0
        rh_start = lh_start = hd_start = st_start = None

        for i, s in enumerate(seq):
            k = s["keypoints"]
            frame = s["frame"]
            t = s["t"]
            torso = torso_length(k)

            # ====== 举手（左右分别） ======
            # wrist 高于 shoulder
            if dist_y(k[LW], k[LS]) < -0.15 * torso:
                if lh_cnt == 0:
                    lh_start = (frame, t)
                lh_cnt += 1
            else:
                if lh_cnt >= raise_hand_frames:
                    events.append({
                        "track_id": tid,
                        "action": "raise_hand",
                        "side": "left",
                        "start_frame": lh_start[0],
                        "end_frame": frame,
                        "start_time": lh_start[1],
                        "end_time": t,
                        "duration": t - lh_start[1],
                        "confidence": min(1.0, lh_cnt / (2 * raise_hand_frames))
                    })
                lh_cnt = 0

            if dist_y(k[RW], k[RS]) < -0.15 * torso:
                if rh_cnt == 0:
                    rh_start = (frame, t)
                rh_cnt += 1
            else:
                if rh_cnt >= raise_hand_frames:
                    events.append({
                        "track_id": tid,
                        "action": "raise_hand",
                        "side": "right",
                        "start_frame": rh_start[0],
                        "end_frame": frame,
                        "start_time": rh_start[1],
                        "end_time": t,
                        "duration": t - rh_start[1],
                        "confidence": min(1.0, rh_cnt / (2 * raise_hand_frames))
                    })
                rh_cnt = 0

            # ====== 低头 ======
            shoulder_mid_y = (k[LS]["y"] + k[RS]["y"]) / 2.0
            if k[NOSE]["y"] > shoulder_mid_y + 0.12 * torso:
                if hd_cnt == 0:
                    hd_start = (frame, t)
                hd_cnt += 1
            else:
                if hd_cnt >= head_down_frames:
                    events.append({
                        "track_id": tid,
                        "action": "head_down",
                        "start_frame": hd_start[0],
                        "end_frame": frame,
                        "start_time": hd_start[1],
                        "end_time": t,
                        "duration": t - hd_start[1],
                        "confidence": min(1.0, hd_cnt / (2 * head_down_frames))
                    })
                hd_cnt = 0

            # ====== 站立 ======
            h = s["bbox"][3] - s["bbox"][1]
            if h > 1.18 * median_h:
                if st_cnt == 0:
                    st_start = (frame, t)
                st_cnt += 1
            else:
                if st_cnt >= stand_frames:
                    events.append({
                        "track_id": tid,
                        "action": "stand",
                        "start_frame": st_start[0],
                        "end_frame": frame,
                        "start_time": st_start[1],
                        "end_time": t,
                        "duration": t - st_start[1],
                        "confidence": min(1.0, st_cnt / (2 * stand_frames))
                    })
                st_cnt = 0

    return events


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, default=os.path.join("output", "pose_tracks_smooth.jsonl"))
    parser.add_argument("--out", dest="out_path", type=str, default=os.path.join("output", "actions.jsonl"))
    parser.add_argument("--fps", dest="fps", type=float, default=25.0)
    args = parser.parse_args()

    in_path = os.path.join(base_dir, args.in_path)
    out_path = os.path.join(base_dir, args.out_path)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tracks = load_tracks(in_path)
    events = detect_actions(tracks, fps=float(args.fps))

    with open(out_path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("[DONE] actions.jsonl:", out_path)
    print("[INFO] total events:", len(events))


if __name__ == "__main__":
    main()
