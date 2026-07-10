import os
import json
import cv2
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

from ultralytics import YOLO
from classroom_pipeline.model_resolver import resolve_model_reference

CLASSROOM_ALIASES = {
    "cell phone": "phone",
    "mobile phone": "phone",
    "book": "book",
    "textbook": "book",
    "pen": "pen",
    "pencil": "pen",
    "notebook": "notebook",
    "laptop": "laptop",
}


def safe_mkdir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def parse_classes(s: str):
    """
    "67,73,63" -> [67,73,63]
    "" -> None
    """
    s = (s or "").strip()
    if not s:
        return None
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def normalize_name(name: str) -> str:
    raw = (name or "").strip().lower()
    return CLASSROOM_ALIASES.get(raw, raw)


def main():
    base_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/yolo11x.pt")

    # 鉁?鍏抽敭锛氳浣犺兘澶嶇幇/璋冨弬
    parser.add_argument("--conf", type=float, default=0.10, help="confidence threshold (small objects寤鸿 0.08~0.15)")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="inference image size (small objects寤鸿 960/1280)")
    parser.add_argument("--device", type=str, default="", help="e.g. 0 or 'cpu'. empty=auto")

    # 鉁?鍏抽敭锛氬彧妫€娴嬫寚瀹氱被锛堝己鐑堝缓璁粯璁や笉鍚?person锛?
    # COCO 甯歌锛?7 cell phone, 73 book, 63 laptop, 62 tv, 64 mouse, etc锛堜互浣犳ā鍨?names 涓哄噯锛?
    parser.add_argument("--classes", type=str, default="67,73,63",
                        help="comma-separated class ids to keep, e.g. '67,73,63'. default excludes person.")
    parser.add_argument("--class_name_map", type=str, default="",
                        help="optional JSON path mapping class_id->class_name (for custom classroom detector)")

    # 鉁?杈撳嚭涓€涓?demo 瑙嗛锛堝儚01涓€鏍凤級锛屾柟渚胯倝鐪奸獙璇?
    parser.add_argument("--out_video", type=str, default="", help="optional: output demo video with boxes")

    # 鉁?鍙€夛細姣忛殧澶氬皯甯ф帹鐞嗕竴娆★紙鍔犻€熺敤锛岄粯璁ゆ瘡甯э級
    parser.add_argument("--stride", type=int, default=1, help="infer every N frames (1=every frame)")

    args = parser.parse_args()

    video_path = (base_dir / args.video).resolve()
    jsonl_path = (base_dir / args.out).resolve()
    safe_mkdir(jsonl_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model_ref = resolve_model_reference(args.model, project_root=PROJECT_ROOT, required=False)

    model = YOLO(model_ref)

    # 鉁?names 鏄犲皠锛屽悗闈㈢敤鏉ヤ繚璇?name 涓€瀹氭纭?
    names = model.model.names if hasattr(model, "model") else model.names
    if not isinstance(names, dict):
        # 鏈夌殑鐗堟湰鏄?list
        names = {i: n for i, n in enumerate(list(names))}
    if args.class_name_map:
        map_path = Path(args.class_name_map)
        if not map_path.is_absolute():
            map_path = (base_dir / map_path).resolve()
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as f:
                ext_map = json.load(f)
            if isinstance(ext_map, dict):
                for k, v in ext_map.items():
                    try:
                        names[int(k)] = str(v)
                    except Exception:
                        continue

    keep_classes = parse_classes(args.classes)  # None or [..]
    if keep_classes is not None:
        # 闃插尽锛氳繃婊ゆ帀瓒婄晫 id
        keep_classes = [cid for cid in keep_classes if cid in names]

    print(f"[INFO] video   = {video_path}")
    print(f"[INFO] out     = {jsonl_path}")
    print(f"[INFO] model   = {model_ref}")
    print(f"[INFO] conf/iou = {args.conf}/{args.iou}, imgsz={args.imgsz}, stride={args.stride}")
    print(f"[INFO] classes = {keep_classes} -> {[names[c] for c in keep_classes] if keep_classes else 'ALL'}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1e-6:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 瑙嗛 writer锛堝彲閫夛級
    writer = None
    out_video_path = None
    if args.out_video.strip():
        out_video_path = Path(args.out_video)
        if not out_video_path.is_absolute():
            out_video_path = (base_dir / out_video_path).resolve()
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"[INFO] demo video -> {out_video_path}")

    frame_idx = 0
    wrote_lines = 0

    with open(str(jsonl_path), "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            do_infer = (frame_idx % max(1, args.stride) == 0)

            detections = []
            if do_infer:
                # 鉁?ultralytics 鍘熺敓 classes 杩囨护锛氭渶鍏抽敭鐨勨€滈槻 person 姹℃煋鈥?
                results = model.predict(
                    frame,
                    verbose=False,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device if args.device else None,
                    classes=keep_classes  # 鉁?鍏抽敭锛氬彧瑕佹寚瀹氱被
                )
                r = results[0]

                if r.boxes is not None and len(r.boxes) > 0:
                    for b in r.boxes:
                        cls_id = int(b.cls[0])
                        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                        conf = float(b.conf[0])
                        name = normalize_name(str(names.get(cls_id, "unknown")))

                        # 鉁?schema 绋冲畾锛氭瘡鏉″繀鏈?name/conf/bbox
                        detections.append({
                            "cls_id": cls_id,
                            "name": name,
                            "conf": conf,
                            "bbox": [x1, y1, x2, y2]
                        })

                        # 鍙鍖?
                        if writer is not None:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                            cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), max(0, int(y1) - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 鉁?寤鸿锛氬嵆浣挎湰甯ф病妫€娴嬪埌锛屼篃鍙互涓嶅啓锛堢渷绌洪棿锛?
            # 浣嗕负浜嗗悗缁榻愭洿绋筹細浣犲彲浠ユ敼鎴愨€滄瘡甯ч兘鍐欌€濓紝杩欓噷鎸変綘涔嬪墠閫昏緫锛氭湁 det 鎵嶅啓
            if len(detections) > 0:
                rec = {
                    "frame": frame_idx,
                    "t": round(frame_idx / fps, 3),
                    "objects": detections
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote_lines += 1

            if writer is not None:
                cv2.putText(frame, f"t={frame_idx/fps:.2f}s frame={frame_idx}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                writer.write(frame)

            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"[INFO] processed {frame_idx} frames, wrote_lines={wrote_lines}, last dets={len(detections)}")

    cap.release()
    if writer is not None:
        writer.release()

    print(f"[DONE] objects jsonl saved: {jsonl_path} (lines={wrote_lines})")
    if out_video_path is not None:
        print(f"[DONE] objects demo video: {out_video_path}")


if __name__ == "__main__":
    main()
