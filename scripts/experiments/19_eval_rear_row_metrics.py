import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classroom_pipeline.utils.sliced_inference_utils import bbox_center, bbox_iou, resolve_roi


BBox = Sequence[float]


def _resolve(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _grab_frame_shape(video: Optional[Path], fallback_rows: Sequence[Dict[str, Any]]) -> Sequence[int]:
    if video and video.exists():
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame.shape
    for row in fallback_rows:
        roi = row.get("roi")
        if isinstance(roi, list) and len(roi) == 4:
            return [int(max(float(roi[3]), 1.0)), int(max(float(roi[2]), 1.0)), 3]
    return [1080, 1920, 3]


def _in_roi(bbox: BBox, roi: BBox) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    cx, cy = bbox_center(bbox)
    rx1, ry1, rx2, ry2 = [float(v) for v in roi]
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _valid_bbox(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in raw]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _filter_persons(row: Dict[str, Any], roi: BBox) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for person in row.get("persons", []):
        if not isinstance(person, dict):
            continue
        bbox = _valid_bbox(person.get("bbox"))
        if bbox is None or not _in_roi(bbox, roi):
            continue
        item = dict(person)
        item["bbox"] = bbox
        out.append(item)
    return out


def _rows_by_frame(path: Path, roi: BBox) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in _iter_jsonl(path):
        try:
            frame = int(row.get("frame", -1))
        except Exception:
            continue
        out[frame] = _filter_persons(row, roi)
    return out


def _gt_rows_by_frame(gt_jsonl: Path, roi: BBox) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    rows = list(_iter_jsonl(gt_jsonl))
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        try:
            frame = int(row.get("frame", -1))
        except Exception:
            continue
        out[frame] = _filter_persons(row, roi)
    return rows, out


def _match_greedy(
    gt_persons: Sequence[Dict[str, Any]],
    pred_persons: Sequence[Dict[str, Any]],
    iou_thres: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_persons):
        gt_bbox = gt.get("bbox", [])
        for pi, pred in enumerate(pred_persons):
            score = bbox_iou(gt_bbox, pred.get("bbox", []))
            if score >= iou_thres:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_gt = set()
    used_pred = set()
    matches: List[Tuple[int, int, float]] = []
    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, float(score)))
    unmatched_gt = [i for i in range(len(gt_persons)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_persons)) if i not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def _f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _parse_thresholds(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def _average_precision(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    if not recalls:
        return 0.0
    ap = 0.0
    for t in [i / 100.0 for i in range(101)]:
        p = 0.0
        for recall, precision in zip(recalls, precisions):
            if recall >= t:
                p = max(p, precision)
        ap += p / 101.0
    return ap


def _ap_for_iou(
    gt_by_frame: Dict[int, List[Dict[str, Any]]],
    pred_by_frame: Dict[int, List[Dict[str, Any]]],
    iou_thres: float,
) -> float:
    total_gt = sum(len(v) for v in gt_by_frame.values())
    if total_gt <= 0:
        return 0.0
    preds: List[Tuple[float, int, int, Dict[str, Any]]] = []
    for frame, items in pred_by_frame.items():
        for idx, pred in enumerate(items):
            try:
                conf = float(pred.get("conf", pred.get("score", 0.0)) or 0.0)
            except Exception:
                conf = 0.0
            preds.append((conf, int(frame), idx, pred))
    preds.sort(reverse=True, key=lambda x: x[0])
    used: Dict[int, set] = {frame: set() for frame in gt_by_frame}
    tp_list: List[int] = []
    fp_list: List[int] = []
    for _, frame, _, pred in preds:
        best_i = -1
        best_iou = 0.0
        for gi, gt in enumerate(gt_by_frame.get(frame, [])):
            if gi in used.setdefault(frame, set()):
                continue
            score = bbox_iou(gt.get("bbox", []), pred.get("bbox", []))
            if score > best_iou:
                best_iou = score
                best_i = gi
        if best_i >= 0 and best_iou >= iou_thres:
            used[frame].add(best_i)
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)
    cum_tp = 0
    cum_fp = 0
    recalls: List[float] = []
    precisions: List[float] = []
    for tp, fp in zip(tp_list, fp_list):
        cum_tp += tp
        cum_fp += fp
        recalls.append(cum_tp / max(1, total_gt))
        precisions.append(cum_tp / max(1, cum_tp + cum_fp))
    return round(_average_precision(recalls, precisions), 4)


def _detection_metrics(
    gt_by_frame: Dict[int, List[Dict[str, Any]]],
    pred_by_frame: Dict[int, List[Dict[str, Any]]],
    iou_thresholds: Sequence[float],
    map_thresholds: Sequence[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    first_counts: Optional[Tuple[int, int, int]] = None
    for thres in iou_thresholds:
        tp = fp = fn = 0
        for frame, gt_persons in gt_by_frame.items():
            pred_persons = pred_by_frame.get(frame, [])
            matches, unmatched_gt, unmatched_pred = _match_greedy(gt_persons, pred_persons, thres)
            tp += len(matches)
            fn += len(unmatched_gt)
            fp += len(unmatched_pred)
        precision, recall, f1 = _f1(tp, fp, fn)
        suffix = str(thres).replace(".", "_")
        out[f"person_precision_iou_{suffix}"] = precision
        out[f"person_recall_iou_{suffix}"] = recall
        out[f"person_f1_iou_{suffix}"] = f1
        if abs(float(thres) - 0.5) < 1e-6:
            out.update(
                {
                    "person_precision": precision,
                    "person_recall": recall,
                    "person_f1": f1,
                    "rear_person_recall": recall,
                    "gt_tp": tp,
                    "gt_fp": fp,
                    "gt_fn": fn,
                }
            )
            first_counts = (tp, fp, fn)
    if first_counts is None:
        out.update({"person_precision": None, "person_recall": None, "person_f1": None, "rear_person_recall": None})
    ap_values = {float(th): _ap_for_iou(gt_by_frame, pred_by_frame, float(th)) for th in map_thresholds}
    out["AP50"] = ap_values.get(0.5)
    out["AP75"] = ap_values.get(0.75)
    out["mAP50-95"] = round(sum(ap_values.values()) / max(1, len(ap_values)), 4) if ap_values else None
    out["mAP50_95"] = out["mAP50-95"]
    out["ap_iou_values"] = {str(k): v for k, v in ap_values.items()}
    return out


def _keypoint_xyc(raw: Any) -> Optional[Tuple[float, float, float]]:
    try:
        if isinstance(raw, dict):
            x = float(raw.get("x"))
            y = float(raw.get("y"))
            c = float(raw.get("c", raw.get("conf", raw.get("v", 1.0))))
            return x, y, c
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            x = float(raw[0])
            y = float(raw[1])
            c = float(raw[2]) if len(raw) >= 3 and raw[2] is not None else 1.0
            return x, y, c
    except Exception:
        return None
    return None


def _person_keypoints(person: Dict[str, Any]) -> List[Optional[Tuple[float, float, float]]]:
    raw = person.get("keypoints", [])
    if not isinstance(raw, list):
        return []
    return [_keypoint_xyc(kp) for kp in raw]


def _oks(gt: Dict[str, Any], pred: Dict[str, Any]) -> Optional[float]:
    gt_kps = _person_keypoints(gt)
    pred_kps = _person_keypoints(pred)
    if not gt_kps or not pred_kps:
        return None
    bbox = _valid_bbox(gt.get("bbox"))
    if not bbox:
        return None
    area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    scale = math.sqrt(area)
    sigma = 0.1
    values: List[float] = []
    for idx, gt_kp in enumerate(gt_kps):
        if idx >= len(pred_kps) or gt_kp is None or pred_kps[idx] is None:
            continue
        gx, gy, gv = gt_kp
        px, py, pc = pred_kps[idx]
        if gv <= 0.0 or pc <= 0.0:
            continue
        dist2 = (gx - px) ** 2 + (gy - py) ** 2
        values.append(math.exp(-dist2 / max(1e-9, 2.0 * (scale * sigma) ** 2)))
    if not values:
        return None
    return sum(values) / len(values)


def _pose_metrics(
    gt_by_frame: Dict[int, List[Dict[str, Any]]],
    pred_by_frame: Dict[int, List[Dict[str, Any]]],
    pck_thresholds: Sequence[float],
    keypoint_conf: float,
) -> Dict[str, Any]:
    total_visible = 0
    correct_by_thres = {float(th): 0 for th in pck_thresholds}
    pred_visible_counts: List[int] = []
    head_shoulder_ok = 0
    head_shoulder_total = 0
    oks_values: List[float] = []
    has_gt_keypoints = False
    for frame, gt_persons in gt_by_frame.items():
        pred_persons = pred_by_frame.get(frame, [])
        matches, _, _ = _match_greedy(gt_persons, pred_persons, 0.5)
        for gi, pi, _ in matches:
            gt = gt_persons[gi]
            pred = pred_persons[pi]
            gt_kps = _person_keypoints(gt)
            pred_kps = _person_keypoints(pred)
            if gt_kps:
                has_gt_keypoints = True
            visible_pred = 0
            for kp in pred_kps:
                if kp is not None and kp[2] >= keypoint_conf:
                    visible_pred += 1
            if pred_kps:
                pred_visible_counts.append(visible_pred)
                head_shoulder_total += 1
                visible_head_shoulder = 0
                for idx in [0, 5, 6]:
                    if idx < len(pred_kps) and pred_kps[idx] is not None and pred_kps[idx][2] >= keypoint_conf:
                        visible_head_shoulder += 1
                if visible_head_shoulder >= 2:
                    head_shoulder_ok += 1
            bbox = _valid_bbox(gt.get("bbox"))
            if bbox is None:
                continue
            norm = max(1.0, bbox[2] - bbox[0], bbox[3] - bbox[1])
            for idx, gt_kp in enumerate(gt_kps):
                if idx >= len(pred_kps) or gt_kp is None or pred_kps[idx] is None:
                    continue
                gx, gy, gv = gt_kp
                px, py, pc = pred_kps[idx]
                if gv <= 0.0:
                    continue
                total_visible += 1
                dist = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
                for thres in pck_thresholds:
                    if pc > 0.0 and dist <= float(thres) * norm:
                        correct_by_thres[float(thres)] += 1
            score = _oks(gt, pred)
            if score is not None:
                oks_values.append(score)
    out: Dict[str, Any] = {
        "pose_gt_status": "ok" if has_gt_keypoints else "missing_keypoints",
        "visible_keypoints_mean": round(sum(pred_visible_counts) / max(1, len(pred_visible_counts)), 4)
        if pred_visible_counts
        else None,
        "head_shoulder_visible_rate": round(head_shoulder_ok / max(1, head_shoulder_total), 4)
        if head_shoulder_total
        else None,
    }
    for thres in pck_thresholds:
        value = round(correct_by_thres[float(thres)] / max(1, total_visible), 4) if has_gt_keypoints else None
        pretty = f"PCK@{float(thres):.2f}"
        safe = "PCK_" + str(float(thres)).replace(".", "_")
        fixed_safe = "PCK_" + f"{float(thres):.2f}".replace(".", "_")
        out[pretty] = value
        out[safe] = value
        out[fixed_safe] = value
    if oks_values:
        oks_thresholds = [0.50 + 0.05 * i for i in range(10)]
        oks_ap = sum(sum(1 for v in oks_values if v >= th) / max(1, len(oks_values)) for th in oks_thresholds) / len(oks_thresholds)
        oks_ar = sum(1 for v in oks_values if v >= 0.5) / max(1, len(oks_values))
        out.update(
            {
                "OKS_mean": round(sum(oks_values) / len(oks_values), 4),
                "OKS-AP": round(oks_ap, 4),
                "OKS_AP": round(oks_ap, 4),
                "OKS-AR": round(oks_ar, 4),
                "OKS_AR": round(oks_ar, 4),
                "OKS_status": "proxy_approx",
            }
        )
    else:
        out.update({"OKS_mean": None, "OKS-AP": None, "OKS_AP": None, "OKS-AR": None, "OKS_AR": None, "OKS_status": "missing_keypoints"})
    return out


def _tracking_metrics(
    gt_by_frame: Dict[int, List[Dict[str, Any]]],
    pred_by_frame: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    idtp = idfp = idfn = 0
    idsw = 0
    gt_total = 0
    last_pred_for_gt: Dict[Any, Any] = {}
    matched_flags: Dict[Any, List[Tuple[int, bool]]] = {}
    missing_track_id = False
    for frame in sorted(gt_by_frame):
        gt_persons = gt_by_frame.get(frame, [])
        pred_persons = pred_by_frame.get(frame, [])
        gt_total += len(gt_persons)
        matches, unmatched_gt, unmatched_pred = _match_greedy(gt_persons, pred_persons, 0.5)
        idtp += len(matches)
        idfn += len(unmatched_gt)
        idfp += len(unmatched_pred)
        matched_gt_indexes = {gi for gi, _, _ in matches}
        for gi, gt in enumerate(gt_persons):
            gt_id = gt.get("track_id")
            if gt_id in [None, ""]:
                missing_track_id = True
                gt_id = f"frame{frame}_idx{gi}"
            matched_flags.setdefault(gt_id, []).append((int(frame), gi in matched_gt_indexes))
        for gi, pi, _ in matches:
            gt_id = gt_persons[gi].get("track_id")
            pred_id = pred_persons[pi].get("track_id")
            if gt_id in [None, ""] or pred_id in [None, ""]:
                missing_track_id = True
                continue
            if gt_id in last_pred_for_gt and last_pred_for_gt[gt_id] != pred_id:
                idsw += 1
                # Approximate IDF1 penalty for an identity discontinuity on this matched detection.
                idfp += 1
                idfn += 1
            last_pred_for_gt[gt_id] = pred_id
    idf1 = 2.0 * idtp / max(1, 2 * idtp + idfp + idfn)
    mota = 1.0 - (idfn + idfp + idsw) / max(1, gt_total)
    deta = idtp / max(1, idtp + idfp + idfn)
    assa = max(0.0, 1.0 - idsw / max(1, idtp))
    fragments = 0
    for flags in matched_flags.values():
        flags.sort(key=lambda x: x[0])
        seen_match = False
        in_gap = False
        for _, matched in flags:
            if matched:
                if seen_match and in_gap:
                    fragments += 1
                seen_match = True
                in_gap = False
            elif seen_match:
                in_gap = True
    return {
        "tracking_gt_status": "missing_track_id" if missing_track_id else "ok",
        "IDF1": round(idf1, 4),
        "IDSW": int(idsw),
        "MOTA": round(mota, 4),
        "HOTA": round(math.sqrt(max(0.0, deta * assa)), 4),
        "HOTA_status": "proxy_approx",
        "track_fragments": int(fragments),
        "idtp": int(idtp),
        "idfp": int(idfp),
        "idfn": int(idfn),
    }


def _match_behavior_pred(gt: Dict[str, Any], preds: Sequence[Dict[str, Any]]) -> Tuple[Optional[int], float]:
    gt_tid = gt.get("track_id")
    best_idx = None
    best_score = -1.0
    for idx, pred in enumerate(preds):
        pred_tid = pred.get("track_id")
        score = bbox_iou(gt.get("bbox", []), pred.get("bbox", []))
        if gt_tid not in [None, ""] and pred_tid == gt_tid:
            score += 1.0
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx is None:
        return None, 0.0
    return best_idx, best_score


def _macro_micro_f1(class_stats: Dict[str, Dict[str, int]]) -> Tuple[Optional[float], Optional[float], Dict[str, Dict[str, float]]]:
    per_class: Dict[str, Dict[str, float]] = {}
    macro_vals: List[float] = []
    total_tp = total_fp = total_fn = 0
    for label in sorted(class_stats):
        stats = class_stats[label]
        tp, fp, fn = int(stats.get("tp", 0)), int(stats.get("fp", 0)), int(stats.get("fn", 0))
        precision, recall, f1 = _f1(tp, fp, fn)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        macro_vals.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    if not class_stats:
        return None, None, {}
    _, _, micro_f1 = _f1(total_tp, total_fp, total_fn)
    return round(sum(macro_vals) / max(1, len(macro_vals)), 4), micro_f1, per_class


def _behavior_frame_ap(
    gt_by_frame: Dict[int, List[Dict[str, Any]]],
    pred_by_frame: Dict[int, List[Dict[str, Any]]],
    iou_thres: float,
) -> Optional[float]:
    labels = sorted(
        {
            str(p.get("behavior_code"))
            for persons in gt_by_frame.values()
            for p in persons
            if p.get("behavior_code") not in [None, ""]
        }
    )
    if not labels:
        return None
    aps: List[float] = []
    for label in labels:
        gt_label_by_frame = {
            frame: [p for p in persons if str(p.get("behavior_code")) == label]
            for frame, persons in gt_by_frame.items()
        }
        pred_label_by_frame = {
            frame: [p for p in pred_by_frame.get(frame, []) if str(p.get("behavior_code", p.get("action", ""))) == label]
            for frame in gt_by_frame
        }
        aps.append(_ap_for_iou(gt_label_by_frame, pred_label_by_frame, iou_thres))
    return round(sum(aps) / max(1, len(aps)), 4)


def _segments_from_gt_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for row in rows:
        for item in row.get("actions", []):
            if isinstance(item, dict):
                segments.append(item)
        for person in row.get("persons", []):
            if not isinstance(person, dict):
                continue
            for item in person.get("segments", []):
                if isinstance(item, dict):
                    seg = dict(item)
                    seg.setdefault("track_id", person.get("track_id"))
                    seg.setdefault("behavior_code", person.get("behavior_code"))
                    segments.append(seg)
    return segments


def _segment_interval(seg: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if "start_frame" in seg and "end_frame" in seg:
        try:
            return float(seg["start_frame"]), float(seg["end_frame"])
        except Exception:
            return None
    if "start_time" in seg and "end_time" in seg:
        try:
            return float(seg["start_time"]), float(seg["end_time"])
        except Exception:
            return None
    return None


def _segment_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ai = _segment_interval(a)
    bi = _segment_interval(b)
    if ai is None or bi is None:
        return 0.0
    start = max(ai[0], bi[0])
    end = min(ai[1], bi[1])
    inter = max(0.0, end - start)
    union = max(ai[1], bi[1]) - min(ai[0], bi[0])
    return inter / max(1e-9, union)


def _temporal_ap(gt_segments: Sequence[Dict[str, Any]], pred_segments: Sequence[Dict[str, Any]], thres: float) -> Optional[float]:
    if not gt_segments:
        return None
    preds = []
    for pred in pred_segments:
        try:
            conf = float(pred.get("conf", pred.get("confidence", 0.0)) or 0.0)
        except Exception:
            conf = 0.0
        preds.append((conf, pred))
    preds.sort(reverse=True, key=lambda x: x[0])
    used = set()
    tp_list: List[int] = []
    fp_list: List[int] = []
    for _, pred in preds:
        best_i = -1
        best_iou = 0.0
        for gi, gt in enumerate(gt_segments):
            if gi in used:
                continue
            if str(gt.get("behavior_code", gt.get("semantic_id", ""))) != str(pred.get("behavior_code", pred.get("semantic_id", ""))):
                continue
            if gt.get("track_id") not in [None, ""] and pred.get("track_id") not in [None, ""] and str(gt.get("track_id")) != str(pred.get("track_id")):
                continue
            score = _segment_iou(gt, pred)
            if score > best_iou:
                best_iou = score
                best_i = gi
        if best_i >= 0 and best_iou >= thres:
            used.add(best_i)
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)
    cum_tp = 0
    cum_fp = 0
    recalls: List[float] = []
    precisions: List[float] = []
    for tp, fp in zip(tp_list, fp_list):
        cum_tp += tp
        cum_fp += fp
        recalls.append(cum_tp / max(1, len(gt_segments)))
        precisions.append(cum_tp / max(1, cum_tp + cum_fp))
    return round(_average_precision(recalls, precisions), 4)


def _behavior_metrics(
    gt_rows: Sequence[Dict[str, Any]],
    gt_by_frame: Dict[int, List[Dict[str, Any]]],
    student_by_frame: Dict[int, List[Dict[str, Any]]],
    action_path: Path,
) -> Dict[str, Any]:
    has_behavior_gt = any(
        p.get("behavior_code") not in [None, ""]
        for persons in gt_by_frame.values()
        for p in persons
    )
    if not has_behavior_gt:
        return {
            "behavior_gt_status": "missing_behavior",
            "behavior_macro_f1": None,
            "behavior_micro_f1": None,
            "frame_mAP@IoU0.5": None,
            "frame_mAP_IoU0_5": None,
            "segment_iou": None,
            "temporal_mAP@0.3": None,
            "temporal_mAP@0.5": None,
            "temporal_mAP@0.7": None,
            "temporal_mAP_0_3": None,
            "temporal_mAP_0_5": None,
            "temporal_mAP_0_7": None,
            "behavior_segment_gt_status": "missing_segments",
            "per_class_behavior": {},
        }
    class_stats: Dict[str, Dict[str, int]] = {}
    for frame, gt_persons in gt_by_frame.items():
        preds = student_by_frame.get(frame, [])
        used_pred = set()
        for gt in gt_persons:
            gt_label = gt.get("behavior_code")
            if gt_label in [None, ""]:
                continue
            gt_label = str(gt_label)
            class_stats.setdefault(gt_label, {"tp": 0, "fp": 0, "fn": 0})
            pred_idx, score = _match_behavior_pred(gt, preds)
            pred_label = None
            if pred_idx is not None and pred_idx not in used_pred and score >= 0.5:
                pred = preds[pred_idx]
                pred_label = pred.get("behavior_code", pred.get("action"))
                used_pred.add(pred_idx)
            if pred_label is None or pred_label == "":
                class_stats[gt_label]["fn"] += 1
            elif str(pred_label) == gt_label:
                class_stats[gt_label]["tp"] += 1
            else:
                class_stats[gt_label]["fn"] += 1
                class_stats.setdefault(str(pred_label), {"tp": 0, "fp": 0, "fn": 0})
                class_stats[str(pred_label)]["fp"] += 1
        for idx, pred in enumerate(preds):
            if idx in used_pred:
                continue
            pred_label = pred.get("behavior_code", pred.get("action"))
            if pred_label not in [None, ""]:
                class_stats.setdefault(str(pred_label), {"tp": 0, "fp": 0, "fn": 0})
                class_stats[str(pred_label)]["fp"] += 1
    macro_f1, micro_f1, per_class = _macro_micro_f1(class_stats)
    frame_map = _behavior_frame_ap(gt_by_frame, student_by_frame, 0.5)

    gt_segments = _segments_from_gt_rows(gt_rows)
    pred_segments = [row for row in _iter_jsonl(action_path)]
    if gt_segments:
        best_ious: List[float] = []
        for gt in gt_segments:
            candidates = [
                _segment_iou(gt, pred)
                for pred in pred_segments
                if str(gt.get("behavior_code", gt.get("semantic_id", ""))) == str(pred.get("behavior_code", pred.get("semantic_id", "")))
            ]
            best_ious.append(max(candidates) if candidates else 0.0)
        segment_iou = round(sum(best_ious) / max(1, len(best_ious)), 4)
        tm03 = _temporal_ap(gt_segments, pred_segments, 0.3)
        tm05 = _temporal_ap(gt_segments, pred_segments, 0.5)
        tm07 = _temporal_ap(gt_segments, pred_segments, 0.7)
        segment_status = "ok"
    else:
        segment_iou = None
        tm03 = tm05 = tm07 = None
        segment_status = "missing_segments"
    return {
        "behavior_gt_status": "ok",
        "behavior_macro_f1": macro_f1,
        "behavior_micro_f1": micro_f1,
        "frame_mAP@IoU0.5": frame_map,
        "frame_mAP_IoU0_5": frame_map,
        "segment_iou": segment_iou,
        "temporal_mAP@0.3": tm03,
        "temporal_mAP@0.5": tm05,
        "temporal_mAP@0.7": tm07,
        "temporal_mAP_0_3": tm03,
        "temporal_mAP_0_5": tm05,
        "temporal_mAP_0_7": tm07,
        "behavior_segment_gt_status": segment_status,
        "per_class_behavior": per_class,
    }


def _engineering_metrics(case_dir: Path) -> Dict[str, Any]:
    runtime = _read_json(case_dir / "variant_runtime.json")
    pipeline = _read_json(case_dir / "pipeline_contract_v2_report.json")
    counts = pipeline.get("counts", {}) if isinstance(pipeline.get("counts"), dict) else {}
    frames_processed = sum(1 for _ in _iter_jsonl(case_dir / "pose_keypoints_v2.jsonl"))
    sr_reports = list((case_dir / "rear_roi_sr").glob("*/sr_cache.report.json")) if (case_dir / "rear_roi_sr").exists() else []
    sr_report = _read_json(sr_reports[0]) if sr_reports else {}
    stage_runtime_sec = runtime.get("elapsed_sec")
    effective_fps = None
    try:
        if stage_runtime_sec:
            effective_fps = round(float(frames_processed) / max(1e-9, float(stage_runtime_sec)), 4)
    except Exception:
        effective_fps = None
    return {
        "stage_runtime_sec": stage_runtime_sec,
        "effective_fps": effective_fps,
        "peak_vram_mb": None,
        "sr_cache_bytes": sr_report.get("cache_bytes"),
        "sr_elapsed_sec": sr_report.get("elapsed_sec"),
        "frames_processed": frames_processed or counts.get("pose_rows"),
        "pipeline_status": pipeline.get("status"),
        "pipeline_counts": counts,
    }


def _sr_quality_metrics() -> Dict[str, Any]:
    return {
        "sr_quality_status": "missing_hr_reference",
        "PSNR": None,
        "SSIM": None,
        "LPIPS": None,
        "NIQE": None,
        "BRISQUE": None,
    }


def _flat_without_nested(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        out[key] = value
    return out


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate rear-row detection, pose, tracking, behavior, SR, and runtime metrics.")
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--gt_jsonl", required=True)
    parser.add_argument("--video", default="")
    parser.add_argument("--roi", default="auto_rear")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--det_iou_thresholds", default="0.50,0.75")
    parser.add_argument("--map_iou_thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--pck_thresholds", default="0.05,0.10,0.20")
    parser.add_argument("--keypoint_conf", type=float, default=0.35)
    args = parser.parse_args()

    case_dir = _resolve(base_dir, args.case_dir)
    gt_jsonl = _resolve(base_dir, args.gt_jsonl)
    video = _resolve(base_dir, args.video) if args.video else None
    out_json = _resolve(base_dir, args.out_json) if args.out_json else case_dir / "rear_row_metrics.json"
    out_csv = _resolve(base_dir, args.out_csv) if args.out_csv else case_dir / "rear_row_metrics.csv"

    if not gt_jsonl.exists():
        payload = {
            "status": "missing_gt",
            "gt_status": "missing",
            "case_dir": str(case_dir),
            "gt_jsonl": str(gt_jsonl),
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(out_csv, _flat_without_nested(payload))
        print(json.dumps(payload, ensure_ascii=False))
        return

    gt_raw_rows = list(_iter_jsonl(gt_jsonl))
    frame_shape = _grab_frame_shape(video, gt_raw_rows)
    roi = resolve_roi(frame_shape, args.roi)
    gt_rows, gt_by_frame = _gt_rows_by_frame(gt_jsonl, roi)
    pose_by_frame = _rows_by_frame(case_dir / "pose_keypoints_v2.jsonl", roi)
    tracks_by_frame = _rows_by_frame(case_dir / "pose_tracks_smooth.jsonl", roi)
    student_by_frame = _rows_by_frame(case_dir / "student_tracks.jsonl", roi)

    det = _detection_metrics(
        gt_by_frame,
        pose_by_frame,
        _parse_thresholds(args.det_iou_thresholds),
        _parse_thresholds(args.map_iou_thresholds),
    )
    pose = _pose_metrics(gt_by_frame, pose_by_frame, _parse_thresholds(args.pck_thresholds), float(args.keypoint_conf))
    tracking = _tracking_metrics(gt_by_frame, tracks_by_frame)
    behavior = _behavior_metrics(gt_rows, gt_by_frame, student_by_frame, case_dir / "actions.behavior.semantic.jsonl")
    engineering = _engineering_metrics(case_dir)
    sr_quality = _sr_quality_metrics()

    payload: Dict[str, Any] = {
        "status": "ok",
        "gt_status": "ok",
        "case_dir": str(case_dir),
        "gt_jsonl": str(gt_jsonl),
        "roi": [round(float(v), 3) for v in roi],
        "gt_frames": len(gt_by_frame),
        "gt_persons": sum(len(v) for v in gt_by_frame.values()),
    }
    payload.update(det)
    payload.update(pose)
    payload.update(tracking)
    payload.update(behavior)
    payload.update(sr_quality)
    payload.update(engineering)
    payload["metric_groups"] = {
        "detection": det,
        "pose": pose,
        "tracking": tracking,
        "behavior": {k: v for k, v in behavior.items() if k != "per_class_behavior"},
        "sr_quality": sr_quality,
        "engineering": engineering,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_csv, _flat_without_nested(payload))
    print(json.dumps({"status": "ok", "out_json": str(out_json), "out_csv": str(out_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
