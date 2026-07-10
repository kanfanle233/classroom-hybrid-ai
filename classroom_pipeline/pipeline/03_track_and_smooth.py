import os
import json
import math
import argparse
from collections import defaultdict
import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class KalmanBoxTracker:
    """Simple Kalman filter for bbox tracking with constant velocity model.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """

    def __init__(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[4:, 4:] *= 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        self._last_bbox = bbox
        self.hit_streak = 0
        self.time_since_update = 0

    def predict(self):
        self.time_since_update += 1
        pred = self.kf.predict()
        cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        self.kf.correct(np.array([[cx], [cy], [w], [h]], dtype=np.float32))
        self._last_bbox = bbox

    @property
    def bbox(self):
        return self._last_bbox


def write_json(path, obj):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ema_update(prev, cur, alpha=0.75):
    out = []
    for p, c in zip(prev, cur):
        if c["c"] is None or c["c"] < 0.30:
            out.append({"x": p["x"], "y": p["y"], "c": c["c"]})
        else:
            out.append({
                "x": alpha * c["x"] + (1 - alpha) * p["x"],
                "y": alpha * c["y"] + (1 - alpha) * p["y"],
                "c": c["c"],
            })
    return out


def normalize_keypoints_17(keypoints):
    """Ensure exactly 17 keypoints with x/y/c triplets."""
    out = []
    keypoints = keypoints or []
    for i in range(min(17, len(keypoints))):
        k = keypoints[i]
        if isinstance(k, dict):
            out.append(
                {
                    "x": float(k.get("x", 0.0)),
                    "y": float(k.get("y", 0.0)),
                    "c": float(k.get("c", 0.0) if k.get("c", None) is not None else 0.0),
                }
            )
        elif isinstance(k, (list, tuple)) and len(k) >= 2:
            c = float(k[2]) if len(k) > 2 and k[2] is not None else 0.0
            out.append({"x": float(k[0]), "y": float(k[1]), "c": c})
    while len(out) < 17:
        out.append({"x": 0.0, "y": 0.0, "c": 0.0})
    return out


def main():
    base_dir = Path(__file__).resolve().parents[2]

    # ===== 命令行参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, default=os.path.join("output", "pose_keypoints_v2.jsonl"))
    parser.add_argument("--out", dest="out_path", type=str, default=os.path.join("output", "pose_tracks_smooth.jsonl"))
    parser.add_argument("--video", dest="video_path", type=str, default=os.path.join("data", "videos", "demo1.mp4"))
    parser.add_argument("--person_conf_thres", type=float, default=0.20)
    parser.add_argument("--track_min_frames", type=int, default=75)
    parser.add_argument("--track_min_frames_ratio", type=float, default=0.0)
    parser.add_argument("--track_min_frames_min", type=int, default=10)
    parser.add_argument("--track_max_lost_frames", type=int, default=500)
    parser.add_argument("--track_iou_thres", type=float, default=0.05)
    parser.add_argument("--track_max_center_dist_ratio", type=float, default=0.15)
    parser.add_argument("--track_max_dx_ratio", type=float, default=0.05)
    parser.add_argument("--track_height_penalty", type=float, default=0.60)
    parser.add_argument("--seat_prior_mode", choices=["off", "x_anchor"], default="x_anchor")
    parser.add_argument("--track_match_mode", choices=["hungarian", "greedy"], default="hungarian")
    parser.add_argument("--track_motion_model", choices=["none", "kalman"], default="kalman")
    parser.add_argument("--report_out", type=str, default="")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.is_absolute():
        in_path = (base_dir / in_path).resolve()
    out_path = Path(args.out_path)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    video_path = Path(args.video_path)
    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    report_path = Path(args.report_out) if args.report_out else None
    if report_path is not None and not report_path.is_absolute():
        report_path = (base_dir / report_path).resolve()

    # ===== 用视频拿分辨率 =====
    cap = cv2.VideoCapture(video_path)
    fps = 25.0
    frame_count = 0
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps_raw if fps_raw > 0 else fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    else:
        w, h = 1920, 1080

    frame_diag = math.hypot(w, h)

    # ====== 核心优化参数 (针对 ID 爆炸问题) ======
    person_conf_thres = max(0.0, min(1.0, float(args.person_conf_thres)))
    alpha = 0.75

    # [修改点 1] 允许更长遮挡：从 150 改为 500
    # 500帧 / 25fps = 20秒。
    # 只要学生在 20 秒内重新被检测到，且位置没大变，就算同一个人。
    max_lost_frames = max(1, int(args.track_max_lost_frames))

    # 匹配更依赖位置：pose bbox 抖动大，IOU 要放宽
    iou_thres = max(0.0, min(1.0, float(args.track_iou_thres)))

    # [修改点 2] 稍微放宽匹配半径，从 0.10 改为 0.15
    # 防止学生大幅度动作（如站起）导致中心点漂移出匹配范围
    max_center_dist_ratio = max(0.0, float(args.track_max_center_dist_ratio))
    max_center_dist = max(1.0, max_center_dist_ratio * frame_diag)

    # ★课堂先验：同一学生横向位置基本不动（强约束）
    # x 方向允许漂移：画面宽度的 5% (保持不变，防止把同桌认错)
    seat_prior_mode = str(args.seat_prior_mode).strip().lower()
    max_dx_ratio = max(0.0, float(args.track_max_dx_ratio))
    max_dx = max_dx_ratio * w
    height_penalty = max(0.0, float(args.track_height_penalty))

    # [修改点 3] 过滤短暂出现的人：从 30 改为 75
    # 75帧 = 3秒。只有由于持续存在超过3秒的检测才会被记录。
    track_min_frames = max(1, int(args.track_min_frames))
    if float(args.track_min_frames_ratio) > 0.0 and frame_count > 0:
        adaptive_min = max(
            int(args.track_min_frames_min),
            int(frame_count * float(args.track_min_frames_ratio)),
        )
        track_min_frames = max(1, min(track_min_frames, adaptive_min))

    next_id = 1
    tracks = {}  # tid -> state
    track_lengths = defaultdict(int)

    if not in_path.exists():
        raise FileNotFoundError(f"Input jsonl not found: {in_path}")

    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ====== [新增] 临时文件：第一遍写盘，避免 temp_records 内存爆炸 ======
    tmp_path = str(out_path) + ".tmp.jsonl"

    print(f"[INFO] Tracking started using strict logic...")
    print(f"       person_conf_thres={person_conf_thres}")
    print(f"       max_lost_frames={max_lost_frames} (Memory ~20s)")
    print(f"       track_min_frames={track_min_frames} (configured={args.track_min_frames}, ratio={args.track_min_frames_ratio})")
    print(f"       seat_prior_mode={seat_prior_mode}, max_dx_ratio={max_dx_ratio}, max_center_dist_ratio={max_center_dist_ratio}")
    print(f"       track_match_mode={args.track_match_mode}, motion_model={args.track_motion_model}")
    print(f"       tmp_path={tmp_path}")

    # ====== 第一遍：跑 tracking + 直接写临时 jsonl ======
    with open(in_path, "r", encoding="utf-8") as f_in, open(tmp_path, "w", encoding="utf-8") as f_tmp:
        for line_idx, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # 输入 jsonl 某一行坏了就跳过，避免整条视频重跑
                continue

            frame = rec["frame"]
            t = rec.get("t", None)

            dets = []
            for p in rec.get("persons", []):
                if p.get("conf", 1.0) < person_conf_thres:
                    continue
                if "bbox" not in p:
                    continue
                dets.append(p)

            det_to_tid = [-1] * len(dets)
            track_match_mode = str(getattr(args, "track_match_mode", "hungarian")).strip().lower()
            motion_model = str(getattr(args, "track_motion_model", "kalman")).strip().lower()

            # ====== Kalman 预测（匹配前） ======
            for tid, st in tracks.items():
                if motion_model == "kalman" and "kalman" in st:
                    st["kalman"].predict()
                # 用于匹配的 bbox：Kalman 预测位置，或最后观测位置
                if motion_model == "kalman" and "kalman" in st and st["kalman"].time_since_update <= max_lost_frames:
                    st["match_bbox"] = st["kalman"].bbox
                else:
                    st["match_bbox"] = st["bbox"]

            # ====== 收集活跃 track ======
            active_tracks = [
                (tid, st) for tid, st in tracks.items()
                if frame - st["last_frame"] <= max_lost_frames
            ]

            if track_match_mode == "hungarian" and active_tracks and dets:
                # ====== 匈牙利算法全局最优匹配 ======
                n_dets = len(dets)
                n_tracks = len(active_tracks)
                cost = np.full((n_dets, n_tracks), 1e9, dtype=np.float64)

                for di, d in enumerate(dets):
                    bc = bbox_center(d["bbox"])
                    for ti, (tid, st) in enumerate(active_tracks):
                        match_bbox = st.get("match_bbox", st["bbox"])
                        prev_c = bbox_center(match_bbox)
                        # 课堂先验：横向漂移过大直接拒绝
                        if seat_prior_mode == "x_anchor" and abs(bc[0] - prev_c[0]) > max_dx:
                            continue
                        dc = dist(bc, prev_c)
                        if dc > max_center_dist:
                            continue
                        iou = iou_xyxy(d["bbox"], match_bbox)
                        h_cur = d["bbox"][3] - d["bbox"][1]
                        h_prev = match_bbox[3] - match_bbox[1]
                        h_ratio = abs(h_cur - h_prev) / max(h_prev, 1.0)
                        score = 1.5 * iou - 1.2 * (dc / max_center_dist) - height_penalty * h_ratio
                        # 拒绝极低质量匹配
                        if iou < iou_thres and dc > (0.06 * frame_diag):
                            continue
                        cost[di, ti] = -score  # Hungarian 最小化，取负

                det_indices, track_indices = linear_sum_assignment(cost)
                used_tracks = set()
                for di, ti in zip(det_indices, track_indices):
                    if cost[di, ti] >= 1e8:
                        continue
                    tid = active_tracks[ti][0]
                    if tid in used_tracks:
                        continue
                    det_to_tid[di] = tid
                    used_tracks.add(tid)
            else:
                # ====== 原有贪心匹配（legacy fallback）======
                for di, d in enumerate(dets):
                    bc = bbox_center(d["bbox"])
                    best_tid, best_score = None, -1e9
                    for tid, st in active_tracks:
                        prev_c = bbox_center(st["bbox"])
                        if seat_prior_mode == "x_anchor" and abs(bc[0] - prev_c[0]) > max_dx:
                            continue
                        dc = dist(bc, prev_c)
                        if dc > max_center_dist:
                            continue
                        iou = iou_xyxy(d["bbox"], st["bbox"])
                        h_cur = d["bbox"][3] - d["bbox"][1]
                        h_prev = st["bbox"][3] - st["bbox"][1]
                        h_ratio = abs(h_cur - h_prev) / max(h_prev, 1.0)
                        score = 1.5 * iou - 1.2 * (dc / max_center_dist) - height_penalty * h_ratio
                        if iou < iou_thres and dc > (0.06 * frame_diag):
                            continue
                        if score > best_score:
                            best_score, best_tid = score, tid
                    if best_tid is not None:
                        det_to_tid[di] = best_tid

                # 冲突解决：一个 track 只能匹配一个 det
                used = {}
                for di, tid in enumerate(det_to_tid):
                    if tid == -1:
                        continue
                    bc = bbox_center(dets[di]["bbox"])
                    prev_c = bbox_center(tracks[tid].get("match_bbox", tracks[tid]["bbox"]))
                    dc = dist(bc, prev_c)
                    iou = iou_xyxy(dets[di]["bbox"], tracks[tid].get("match_bbox", tracks[tid]["bbox"]))
                    score = 1.5 * iou - 1.2 * (dc / max_center_dist)
                    if tid not in used or score > used[tid][0]:
                        used[tid] = (score, di)
                det_to_tid = [-1] * len(dets)
                for tid, (_, di) in used.items():
                    det_to_tid[di] = tid

            # ====== 新建 track ======
            for di, d in enumerate(dets):
                if det_to_tid[di] == -1:
                    tid = next_id
                    next_id += 1
                    det_to_tid[di] = tid
                    st = {
                        "bbox": d["bbox"],
                        "kpts": normalize_keypoints_17(d.get("keypoints")),
                        "last_frame": frame,
                        "len": 0,
                    }
                    if motion_model == "kalman":
                        st["kalman"] = KalmanBoxTracker(d["bbox"])
                    tracks[tid] = st

            # ====== 更新状态 + EMA ======
            frame_out = []
            for di, d in enumerate(dets):
                tid = det_to_tid[di]
                st = tracks[tid]

                if st["len"] == 0:
                    smooth = normalize_keypoints_17(d.get("keypoints"))
                else:
                    smooth = ema_update(st["kpts"], normalize_keypoints_17(d.get("keypoints")), alpha)

                st["bbox"] = d["bbox"]
                st["kpts"] = smooth
                st["last_frame"] = frame
                st["len"] += 1
                track_lengths[tid] += 1

                if motion_model == "kalman" and "kalman" in st:
                    st["kalman"].update(d["bbox"])

                frame_out.append({
                    "track_id": tid,
                    "bbox": d["bbox"],
                    "conf": d.get("conf", None),
                    "keypoints": smooth
                })

            # ====== [修改核心] 不再 append 到 temp_records，直接写盘 ======
            if frame_out:
                f_tmp.write(json.dumps({
                    "frame": frame,
                    "t": t,
                    "persons": frame_out
                }, ensure_ascii=False) + "\n")

            # 可选：定期 flush，避免中途崩溃丢太多
            if line_idx % 200 == 0:
                f_tmp.flush()

    # ====== 过滤短轨迹 (关键步骤) ======
    valid_ids = {tid for tid, ln in track_lengths.items() if ln >= track_min_frames}

    # ====== 第二遍：读临时文件 -> 过滤 -> 写最终输出 ======
    emitted_rows = 0
    emitted_person_rows = 0
    per_track_frames = defaultdict(list)
    with open(tmp_path, "r", encoding="utf-8") as fi, open(out_path, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            ps = [p for p in rec.get("persons", []) if p.get("track_id") in valid_ids]
            if ps:
                rec["persons"] = ps
                emitted_rows += 1
                emitted_person_rows += len(ps)
                for p in ps:
                    per_track_frames[int(p.get("track_id"))].append(int(rec.get("frame", -1)))
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 清理临时文件（失败也不影响主结果）
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    gap_count = 0
    max_gap = 0
    for frames in per_track_frames.values():
        frames = sorted(frames)
        for a, b in zip(frames, frames[1:]):
            gap = int(b) - int(a)
            if gap > 1:
                gap_count += 1
                max_gap = max(max_gap, gap)
    if report_path is not None:
        write_json(
            report_path,
            {
                "stage": "pose_track_and_smooth",
                "status": "ok",
                "input": str(in_path),
                "output": str(out_path),
                "video": str(video_path),
                "frame_count": int(frame_count),
                "fps": float(fps),
                "raw_tracks": int(len(track_lengths)),
                "valid_tracks": int(len(valid_ids)),
                "dropped_tracks": int(max(0, len(track_lengths) - len(valid_ids))),
                "emitted_frame_rows": int(emitted_rows),
                "emitted_person_rows": int(emitted_person_rows),
                "track_gap_count_proxy": int(gap_count),
                "track_max_gap_proxy": int(max_gap),
                "params": {
                    "person_conf_thres": float(person_conf_thres),
                    "max_lost_frames": int(max_lost_frames),
                    "iou_thres": float(iou_thres),
                    "max_center_dist_ratio": float(max_center_dist_ratio),
                    "max_dx_ratio": float(max_dx_ratio),
                    "seat_prior_mode": str(seat_prior_mode),
                    "track_min_frames": int(track_min_frames),
                    "height_penalty": float(height_penalty),
                    "track_match_mode": str(getattr(args, "track_match_mode", "hungarian")),
                    "track_motion_model": str(getattr(args, "track_motion_model", "kalman")),
                },
            },
        )

    print("[DONE] pose_tracks_smooth.jsonl:", out_path)
    print(f"[INFO] Final valid IDs count: {len(valid_ids)} (Filtered out {next_id - 1 - len(valid_ids)} short tracks)")


if __name__ == "__main__":
    main()
