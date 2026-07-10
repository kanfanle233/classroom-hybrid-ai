import os
import json
import argparse
import math
from collections import defaultdict

LEGACY_NOTICE = (
    "[LEGACY/EXPERIMENTAL] scripts/04_complex_logic.py is not part of the formal fixed-schema pipeline. "
    "Use scripts/09_run_pipeline.py for default execution."
)

# ====== 1. 基础几何工具 ======
def is_close(box_person, box_obj, threshold=0.1):
    """
    判断物体是否属于这个人。
    逻辑：物体中心点在人的 BBox 内部，或者距离非常近。
    threshold: 相对人 BBox 对角线长度的容忍度
    """
    px1, py1, px2, py2 = box_person
    ox1, oy1, ox2, oy2 = box_obj

    # 计算中心点
    p_center = ((px1 + px2) / 2, (py1 + py2) / 2)
    o_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)

    # 1. 如果物体中心在人的框内 -> 判定为拥有
    if px1 < o_center[0] < px2 and py1 < o_center[1] < py2:
        return True

    # 2. 如果不在框内，计算距离是否足够近 (例如手拿着手机伸出去了)
    p_diag = math.hypot(px2 - px1, py2 - py1)
    dist = math.hypot(p_center[0] - o_center[0], p_center[1] - o_center[1])

    if dist < p_diag * threshold:
        return True

    return False


# ====== 2. 核心判定逻辑 ======
def determine_action(kpts, bbox, nearby_objects):
    """
    输入：17个关键点、人的BBox、身边的物品列表
    输出：动作名称
    """
    # COCO Keypoints 索引
    NOSE = 0
    LEAR, REAR = 3, 4
    LSH, RSH = 5, 6
    LWRI, RWRI = 9, 10

    # 提取关键点 (如果置信度过低则忽略)
    def get_y(idx):
        if kpts[idx]['c'] is None or kpts[idx]['c'] < 0.3: return None
        return kpts[idx]['y']

    nose_y = get_y(NOSE)
    lear_y = get_y(LEAR)
    rear_y = get_y(REAR)
    lwri_y = get_y(LWRI)
    rwri_y = get_y(RWRI)
    lsh_y = get_y(LSH)
    rsh_y = get_y(RSH)

    shoulder_avg_y = None
    if lsh_y is not None and rsh_y is not None:
        shoulder_avg_y = (lsh_y + rsh_y) / 2

    # --- 物品状态 ---
    has_phone = any(o['name'] == 'cell phone' for o in nearby_objects)
    has_book = any(o['name'] == 'book' for o in nearby_objects)

    # --- 规则 1: 举手 (Raise Hand) ---
    # 逻辑：手腕高于耳朵 (Y值更小)
    is_raising = False
    if lwri_y is not None and lear_y is not None and lwri_y < lear_y:
        is_raising = True
    if rwri_y is not None and rear_y is not None and rwri_y < rear_y:
        is_raising = True

    if is_raising:
        return "raising_hand"

    # --- 规则 2: 低头 (Head Down) ---
    # 逻辑：鼻子位置接近肩膀高度
    is_head_down = False
    if nose_y is not None and shoulder_avg_y is not None:
        box_h = bbox[3] - bbox[1]
        if nose_y > shoulder_avg_y - (box_h * 0.15):
            is_head_down = True

    if nose_y is not None and shoulder_avg_y is not None and nose_y > shoulder_avg_y:
        return "sleeping"

    # --- 综合判断 (结合物品) ---
    if is_head_down:
        if has_phone:
            return "playing_phone"  # 抓到了！低头且有手机
        elif has_book:
            return "reading_writing"  # 低头且有书
        else:
            return "writing"  # 低头无物品，默认为写字或发呆

    # 抬头状态但有书，可能在看黑板对照书
    if has_book:
        return "reading"

    # 默认状态
    return "listening"


# ====== 3. 主流程 ======
def main():
    print(LEGACY_NOTICE)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    # 接收来自 09 脚本的参数
    parser.add_argument("--pose", type=str, required=True, help="pose_tracks_smooth.jsonl")
    parser.add_argument("--obj", type=str, required=True, help="objects.jsonl")
    parser.add_argument("--out", type=str, required=True, help="output actions.jsonl")
    args = parser.parse_args()

    # 1. 预加载所有物品，建立 Frame 索引
    print(f"[INFO] Loading objects from {args.obj}...")
    frame_objects = defaultdict(list)
    if os.path.exists(args.obj):
        with open(args.obj, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "objects" in d:
                        frame_objects[d['frame']] = d['objects']
                except:
                    continue

    # 2. 处理姿态轨迹
    print(f"[INFO] Processing poses and matching objects...")
    actions = []

    if not os.path.exists(args.pose):
        print(f"[ERROR] Pose file not found: {args.pose}")
        return

    with open(args.pose, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except:
                continue

            frame_idx = d['frame']
            t = d['t']
            persons = d['persons']

            # 取出当前帧的所有物品
            current_objs = frame_objects.get(frame_idx, [])

            for p in persons:
                bbox = p['bbox']
                kpts = p['keypoints']
                tid = p['track_id']

                # 找到属于这个人的物品
                my_objs = [o for o in current_objs if is_close(bbox, o['bbox'])]

                # 判定动作
                act_label = determine_action(kpts, bbox, my_objs)

                # 构建输出
                # 为了兼容后续脚本，我们把每一帧都当作一个微小的动作片段
                # 后续可以用简单的合并逻辑或者直接可视化
                # fps 需要一个确定值（你这里默认 25fps 的假设已经写在注释里）
                FPS = 25.0
                FRAME_DT = 1.0 / FPS

                actions.append({
                    "track_id": tid,
                    "action": act_label,

                    # ---- 时间协议（保持你原来的设计） ----
                    "start_time": float(t),
                    "end_time": float(t) + FRAME_DT,

                    # ---- 帧协议（新增：让 overlay 能吃） ----
                    "start_frame": int(frame_idx),
                    "end_frame": int(frame_idx),

                    # 兼容字段：你原来就有
                    "frame": int(frame_idx),

                    "confidence": 1.0,
                    "objects_found": [o["name"] for o in my_objs]
                })

    # 3. 保存
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for a in actions:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"[DONE] Generated {len(actions)} action records to {args.out}")


if __name__ == "__main__":
    main()
