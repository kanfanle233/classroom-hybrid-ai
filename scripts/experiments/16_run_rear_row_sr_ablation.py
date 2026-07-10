import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classroom_pipeline.utils.sliced_inference_utils import bbox_center, bbox_iou, check_sr_backend_available, resolve_roi


VARIANTS = [
    {
        "id": "A0_full_no_sr",
        "sr_backend": "off",
        "pose_infer_mode": "full",
        "behavior_infer_mode": "full",
        "existing_dir": "output/codex_reports/front_002_full_pose020_hybrid",
    },
    {
        "id": "A1_full_sliced_no_sr",
        "sr_backend": "off",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
        "existing_dir": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid",
    },
    {
        "id": "A2_full_sliced_opencv",
        "sr_backend": "opencv",
        "sr_preprocess": "off",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
    },
    {
        "id": "A3_full_sliced_artifact_deblur_opencv",
        "sr_backend": "opencv",
        "sr_preprocess": "artifact_deblur",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
    },
    {
        "id": "A4_full_sliced_realesrgan",
        "sr_backend": "realesrgan",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
    },
    {
        "id": "A5_full_sliced_basicvsrpp",
        "sr_backend": "basicvsrpp",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
    },
    {
        "id": "A6_full_sliced_realbasicvsr",
        "sr_backend": "realbasicvsr",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
    },
    {
        "id": "A7_sliced_strong_seat_prior",
        "sr_backend": "off",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
        "pose_track_max_lost_frames": 750,
        "pose_track_max_center_dist_ratio": 0.12,
        "pose_track_max_dx_ratio": 0.035,
        "pose_track_seat_prior_mode": "x_anchor",
    },
    {
        "id": "A8_adaptive_sliced_artifact_deblur_opencv",
        "sr_backend": "opencv",
        "sr_preprocess": "artifact_deblur",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
        "slice_grid": "rear_adaptive",
        "slice_overlap": 0.35,
    },
    {
        "id": "A9_rear_dense_artifact_deblur_opencv",
        "sr_backend": "opencv",
        "sr_preprocess": "artifact_deblur",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
        "slice_grid": "rear_dense",
        "slice_overlap": 0.35,
    },
    {
        "id": "A10_full_sliced_nvidia_vsr",
        "sr_backend": "nvidia_vsr",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
    },
    {
        "id": "A11_full_sliced_maxine_vfx",
        "sr_backend": "maxine_vfx",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
    },
    {
        "id": "A12_deepstream_reid_placeholder",
        "sr_backend": "off",
        "pose_infer_mode": "full_sliced",
        "behavior_infer_mode": "full_sliced",
        "requires_external": "deepstream_reid_tracker",
    },
]


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


def _grab_frame_shape(video: Path) -> Sequence[int]:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read first frame from {video}")
    return frame.shape


def _in_roi(bbox: Sequence[float], roi: Sequence[float]) -> bool:
    cx, cy = bbox_center(bbox)
    rx1, ry1, rx2, ry2 = [float(v) for v in roi]
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _pose_proxy_metrics(case_dir: Path, video: Path, roi_name: str) -> Dict[str, Any]:
    pose_path = case_dir / "pose_keypoints_v2.jsonl"
    roi = resolve_roi(_grab_frame_shape(video), roi_name)
    total = 0
    rear = 0
    confs: List[float] = []
    heights: List[float] = []
    visible_counts: List[int] = []
    rear_heights: List[float] = []
    rear_visible_counts: List[int] = []
    rear_head_shoulder_ok = 0
    rear_total = 0
    for row in _iter_jsonl(pose_path):
        for person in row.get("persons", []):
            if not isinstance(person, dict):
                continue
            bbox = person.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            total += 1
            is_rear = _in_roi(bbox, roi)
            if is_rear:
                rear += 1
                rear_total += 1
                rear_heights.append(max(0.0, float(bbox[3]) - float(bbox[1])))
            confs.append(float(person.get("conf", 0.0)))
            heights.append(max(0.0, float(bbox[3]) - float(bbox[1])))
            keypoints = person.get("keypoints", [])
            visible = 0
            head_shoulder_visible = 0
            for kp in keypoints:
                if isinstance(kp, dict):
                    c = kp.get("c", kp.get("conf", 0.0))
                elif isinstance(kp, list) and len(kp) > 2:
                    c = kp[2]
                else:
                    c = 0.0
                try:
                    if c is not None and float(c) >= 0.35:
                        visible += 1
                except Exception:
                    pass
            for idx in [0, 5, 6]:
                if idx < len(keypoints):
                    kp = keypoints[idx]
                    if isinstance(kp, dict):
                        c = kp.get("c", kp.get("conf", 0.0))
                    elif isinstance(kp, list) and len(kp) > 2:
                        c = kp[2]
                    else:
                        c = 0.0
                    try:
                        if c is not None and float(c) >= 0.35:
                            head_shoulder_visible += 1
                    except Exception:
                        pass
            if keypoints:
                visible_counts.append(visible)
                if is_rear:
                    rear_visible_counts.append(visible)
                    if head_shoulder_visible >= 2:
                        rear_head_shoulder_ok += 1
    return {
        "pose_person_rows_proxy": total,
        "rear_pose_person_rows_proxy": rear,
        "avg_pose_conf": round(sum(confs) / max(1, len(confs)), 4),
        "avg_bbox_height": round(sum(heights) / max(1, len(heights)), 3),
        "rear_avg_bbox_height": round(sum(rear_heights) / max(1, len(rear_heights)), 3),
        "avg_visible_keypoints": round(sum(visible_counts) / max(1, len(visible_counts)), 3),
        "rear_avg_visible_keypoints": round(sum(rear_visible_counts) / max(1, len(rear_visible_counts)), 3),
        "rear_low_visible_keypoint_rate": round(
            sum(1 for v in rear_visible_counts if v < 5) / max(1, len(rear_visible_counts)),
            4,
        ),
        "rear_head_shoulder_visible_rate": round(rear_head_shoulder_ok / max(1, rear_total), 4),
        "roi": [round(float(v), 3) for v in roi],
    }


def _track_stability_metrics(case_dir: Path, video: Path, roi_name: str) -> Dict[str, Any]:
    tracks_path = case_dir / "pose_tracks_smooth.jsonl"
    if not tracks_path.exists():
        return {}
    frame_shape = _grab_frame_shape(video)
    roi = resolve_roi(frame_shape, roi_name)
    diag = (float(frame_shape[0]) ** 2 + float(frame_shape[1]) ** 2) ** 0.5
    by_tid: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}
    for row in _iter_jsonl(tracks_path):
        frame = int(row.get("frame", -1))
        for person in row.get("persons", []):
            if not isinstance(person, dict):
                continue
            bbox = person.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4 or not _in_roi(bbox, roi):
                continue
            tid = int(person.get("track_id", -1))
            if tid < 0:
                continue
            by_tid.setdefault(tid, []).append((frame, bbox_center(bbox)))

    gap_count = 0
    jitter_values: List[float] = []
    high_jitter = 0
    for items in by_tid.values():
        items.sort(key=lambda x: x[0])
        if len(items) < 2:
            continue
        frames = [f for f, _ in items]
        for a, b in zip(frames, frames[1:]):
            if b - a > 1:
                gap_count += 1
        cx = sorted(p[0] for _, p in items)[len(items) // 2]
        cy = sorted(p[1] for _, p in items)[len(items) // 2]
        distances = [((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 / max(1.0, diag) for _, p in items]
        jitter = sum(distances) / max(1, len(distances))
        jitter_values.append(jitter)
        if jitter > 0.035:
            high_jitter += 1
    return {
        "rear_track_count_proxy": len(by_tid),
        "track_gap_count_proxy": gap_count,
        "seat_anchor_jitter_mean": round(sum(jitter_values) / max(1, len(jitter_values)), 5),
        "seat_anchor_high_jitter_tracks": high_jitter,
        "seat_prior_candidate": high_jitter > 0 or gap_count > 0,
    }


def _gt_detection_metrics(case_dir: Path, gt_jsonl: Path, video: Path, roi_name: str, iou_thres: float) -> Dict[str, Any]:
    if not gt_jsonl.exists():
        return {"gt_status": "missing", "person_precision": None, "person_recall": None, "person_f1": None}
    roi = resolve_roi(_grab_frame_shape(video), roi_name)
    preds_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for row in _iter_jsonl(case_dir / "pose_keypoints_v2.jsonl"):
        preds = []
        for p in row.get("persons", []):
            bbox = p.get("bbox", []) if isinstance(p, dict) else []
            if isinstance(bbox, list) and len(bbox) == 4 and _in_roi(bbox, roi):
                preds.append(p)
        preds_by_frame[int(row.get("frame", -1))] = preds

    tp = fp = fn = 0
    for row in _iter_jsonl(gt_jsonl):
        frame = int(row.get("frame", -1))
        gt_boxes = []
        for p in row.get("persons", []):
            bbox = p.get("bbox", []) if isinstance(p, dict) else []
            if isinstance(bbox, list) and len(bbox) == 4 and _in_roi(bbox, roi):
                gt_boxes.append(bbox)
        preds = list(preds_by_frame.get(frame, []))
        used_pred = set()
        for gt_box in gt_boxes:
            best_i = -1
            best_iou = 0.0
            for i, pred in enumerate(preds):
                if i in used_pred:
                    continue
                score = bbox_iou(gt_box, pred.get("bbox", []))
                if score > best_iou:
                    best_iou = score
                    best_i = i
            if best_i >= 0 and best_iou >= iou_thres:
                tp += 1
                used_pred.add(best_i)
            else:
                fn += 1
        fp += max(0, len(preds) - len(used_pred))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "gt_status": "ok",
        "gt_iou_thres": iou_thres,
        "person_precision": round(precision, 4),
        "person_recall": round(recall, 4),
        "person_f1": round(f1, 4),
        "gt_tp": tp,
        "gt_fp": fp,
        "gt_fn": fn,
    }


def _collect_metrics(variant: Dict[str, Any], case_dir: Path, video: Path, roi: str, gt_jsonl: Path) -> Dict[str, Any]:
    pipeline = _read_json(case_dir / "pipeline_contract_v2_report.json")
    counts = pipeline.get("counts", {}) if isinstance(pipeline.get("counts", {}), dict) else {}
    pose_diag = _read_json(case_dir / "rear_row_pose_diagnostics.json")
    pose_summary = pose_diag.get("summary", {}) if isinstance(pose_diag.get("summary", {}), dict) else {}
    behavior_diag = _read_json(case_dir / "rear_row_behavior_diagnostics.json")
    behavior_summary = behavior_diag.get("summary", {}) if isinstance(behavior_diag.get("summary", {}), dict) else {}
    student_report = _read_json(case_dir / "student_tracks.report.json")
    student_stats = student_report.get("stats", {}) if isinstance(student_report.get("stats", {}), dict) else {}
    pose_track_report = _read_json(case_dir / "pose_tracks.report.json")
    pose_track_params = pose_track_report.get("params", {}) if isinstance(pose_track_report.get("params", {}), dict) else {}
    sr_report = _read_json(case_dir / "rear_roi_sr" / str(variant["sr_backend"]) / "sr_cache.report.json")
    variant_status = _read_json(case_dir / "variant_status.json")
    variant_runtime = _read_json(case_dir / "variant_runtime.json")
    official_metrics = _read_json(case_dir / "rear_row_metrics.json")
    proxy = _pose_proxy_metrics(case_dir, video, roi) if (case_dir / "pose_keypoints_v2.jsonl").exists() else {}
    track_proxy = _track_stability_metrics(case_dir, video, roi) if (case_dir / "pose_tracks_smooth.jsonl").exists() else {}
    if official_metrics.get("status") == "ok":
        gt = {k: v for k, v in official_metrics.items() if not isinstance(v, (dict, list))}
    else:
        gt = (
            _gt_detection_metrics(case_dir, gt_jsonl, video, roi, 0.5)
            if gt_jsonl.exists() and gt_jsonl.is_file()
            else {"gt_status": "missing", "person_precision": None, "person_recall": None, "person_f1": None}
        )

    row: Dict[str, Any] = {
        "variant": variant["id"],
        "case_dir": str(case_dir),
        "sr_backend": variant["sr_backend"],
        "sr_preprocess": variant.get("sr_preprocess", "off"),
        "pose_infer_mode": variant["pose_infer_mode"],
        "behavior_infer_mode": variant["behavior_infer_mode"],
        "variant_status": variant_status.get("status", "ok" if pipeline.get("status") else "missing"),
        "pipeline_status": pipeline.get("status", ""),
        "sr_status": sr_report.get("status", "off" if variant["sr_backend"] == "off" else variant_status.get("status", "")),
        "sr_reason": sr_report.get("reason", variant_status.get("reason", "")),
        "sr_report_preprocess": sr_report.get("preprocess", variant.get("sr_preprocess", "off")),
        "tracked_students": counts.get("tracked_students"),
        "student_track_students": counts.get("student_track_students"),
        "behavior_items": counts.get("behavior_items"),
        "actions_fusion_v2": counts.get("actions_fusion_v2"),
        "timeline_student_rows": counts.get("timeline_student_rows"),
        "unlinked_rows_ge_200000": counts.get("actions_fusion_v2_unlinked_rows_ge_200000"),
        "pose_full_raw_total": pose_summary.get("full_raw_total"),
        "pose_tile_raw_total": pose_summary.get("tile_raw_total"),
        "pose_merged_total": pose_summary.get("merged_total"),
        "pose_rear_merged_total": pose_summary.get("rear_merged_total"),
        "behavior_full_raw_total": behavior_summary.get("full_raw_total"),
        "behavior_tile_raw_total": behavior_summary.get("tile_raw_total"),
        "behavior_merged_total": behavior_summary.get("merged_total"),
        "behavior_rear_merged_total": behavior_summary.get("rear_merged_total"),
        "matched_behavior_items": student_stats.get("matched_behavior_items"),
        "unmatched_behavior_items": student_stats.get("unmatched_behavior_items"),
        "pose_person_rows": student_stats.get("pose_person_rows"),
        "pose_students_total": student_stats.get("pose_students_total"),
        "pose_students_with_behavior_match": student_stats.get("pose_students_with_behavior_match"),
        "sr_cache_bytes": sr_report.get("cache_bytes"),
        "sr_elapsed_sec": sr_report.get("elapsed_sec"),
        "stage_runtime_sec": variant_runtime.get("elapsed_sec"),
        "effective_fps": None,
        "peak_vram_mb": None,
        "pose_track_valid_tracks": pose_track_report.get("valid_tracks"),
        "pose_track_gap_count_proxy": pose_track_report.get("track_gap_count_proxy"),
        "pose_track_max_gap_proxy": pose_track_report.get("track_max_gap_proxy"),
        "pose_track_seat_prior_mode": pose_track_params.get("seat_prior_mode", variant.get("pose_track_seat_prior_mode")),
        "pose_track_max_lost_frames": pose_track_params.get("max_lost_frames", variant.get("pose_track_max_lost_frames")),
        "pose_track_max_dx_ratio": pose_track_params.get("max_dx_ratio", variant.get("pose_track_max_dx_ratio")),
        "pose_track_max_center_dist_ratio": pose_track_params.get(
            "max_center_dist_ratio",
            variant.get("pose_track_max_center_dist_ratio"),
        ),
        "IDSW": None,
        "IDF1": None,
        "HOTA": None,
    }
    row.update(proxy)
    row.update(track_proxy)
    row.update(gt)
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "variant",
        "sr_backend",
        "sr_preprocess",
        "pipeline_status",
        "sr_status",
        "tracked_students",
        "rear_pose_person_rows_proxy",
        "pose_rear_merged_total",
        "rear_avg_visible_keypoints",
        "rear_low_visible_keypoint_rate",
        "rear_head_shoulder_visible_rate",
        "rear_track_count_proxy",
        "track_gap_count_proxy",
        "seat_anchor_jitter_mean",
        "pose_track_gap_count_proxy",
        "pose_track_max_gap_proxy",
        "pose_track_seat_prior_mode",
        "pose_track_max_dx_ratio",
        "matched_behavior_items",
        "unmatched_behavior_items",
        "actions_fusion_v2",
        "timeline_student_rows",
        "person_recall",
        "person_precision",
        "person_f1",
        "AP50",
        "AP75",
        "mAP50_95",
        "PCK_0_10",
        "OKS_AP",
        "IDF1",
        "IDSW",
        "HOTA",
        "MOTA",
        "track_fragments",
        "behavior_macro_f1",
        "behavior_micro_f1",
        "segment_iou",
        "temporal_mAP_0_5",
        "frame_mAP_IoU0_5",
        "stage_runtime_sec",
        "effective_fps",
        "peak_vram_mb",
        "sr_cache_bytes",
        "gt_status",
        "sr_elapsed_sec",
    ]
    lines = [
        "# Rear-row ROI SR Ablation",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    lines += [
        "",
        "## Paper-use Notes",
        "- `person_recall/person_precision/person_f1`, AP/PCK/OKS/IDF1/HOTA/MOTA and behavior F1 are only valid when a rear-row GT JSONL is supplied.",
        "- Without GT, `rear_pose_person_rows_proxy`, `pose_rear_merged_total`, `tracked_students`, and `matched_behavior_items` are recall proxies, not accuracy claims.",
        "- `track_gap_count_proxy` and `seat_anchor_jitter_mean` are seat-prior diagnostics, not formal IDF1/HOTA.",
        "- DLSS should not be claimed directly; use `ROI video super-resolution` or `resolution-aware rear-row enhancement`.",
        "- `nvidia_vsr` and `maxine_vfx` rows are external-command adapters; they are unavailable until the SDK command is configured.",
        "- A publishable main claim needs GT annotations, ablation figures, runtime cost, and failure cases.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_pose_by_frame(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in _iter_jsonl(path):
        out[int(row.get("frame", -1))] = [p for p in row.get("persons", []) if isinstance(p, dict)]
    return out


def _draw_variant_frame(frame: Any, persons: List[Dict[str, Any]], title: str, roi: Sequence[float]) -> Any:
    out = frame.copy()
    rx1, ry1, rx2, ry2 = [int(round(float(v))) for v in roi]
    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2, cv2.LINE_AA)
    for p in persons:
        bbox = p.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        color = (0, 220, 255) if str(p.get("source", "")) == "tile" else (255, 140, 40)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(out, f"{title} persons={len(persons)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, f"{title} persons={len(persons)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _resize_h(img: Any, height: int) -> Any:
    scale = height / img.shape[0]
    return cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), height), interpolation=cv2.INTER_AREA)


def _write_contact_sheet(path: Path, video: Path, rows: List[Dict[str, Any]], frames: List[int], roi_name: str) -> None:
    available = [r for r in rows if r.get("pipeline_status") == "ok" and Path(str(r.get("case_dir", ""))).exists()]
    if not available:
        return
    pose_maps = {r["variant"]: _read_pose_by_frame(Path(str(r["case_dir"])) / "pose_keypoints_v2.jsonl") for r in available}
    roi = resolve_roi(_grab_frame_shape(video), roi_name)
    cap = cv2.VideoCapture(str(video))
    sheet_rows = []
    for frame_idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        cells = []
        for row in available:
            persons = pose_maps.get(row["variant"], {}).get(frame_idx, [])
            cells.append(_resize_h(_draw_variant_frame(frame, persons, row["variant"], roi), 260))
        if not cells:
            continue
        sep = np.full((cells[0].shape[0], 8, 3), 255, dtype=np.uint8)
        parts = []
        for cell in cells:
            if parts:
                parts.append(sep)
            parts.append(cell)
        sheet_rows.append(cv2.hconcat(parts))
    cap.release()
    if not sheet_rows:
        return
    width = max(r.shape[1] for r in sheet_rows)
    padded = []
    for row in sheet_rows:
        if row.shape[1] < width:
            row = cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.vconcat(padded))


def _variant_dir(base_dir: Path, out_root: Path, variant: Dict[str, Any], reuse_existing: bool) -> Path:
    # Historical A0/A1 baselines live outside the front_002 ablation folder. Do not reuse
    # those paths for other cases, or cross-case paper metrics become contaminated.
    allow_external_existing = "front_002" in str(out_root).lower()
    if reuse_existing and allow_external_existing and variant.get("existing_dir"):
        existing = _resolve(base_dir, str(variant["existing_dir"]))
        if (existing / "pipeline_contract_v2_report.json").exists():
            return existing
    return out_root / str(variant["id"])


def _run_pipeline(base_dir: Path, py: str, video: Path, out_dir: Path, variant: Dict[str, Any], args: argparse.Namespace) -> int:
    cmd = [
        py,
        str(base_dir / "scripts" / "09_run_pipeline.py"),
        "--video",
        str(video),
        "--out_dir",
        str(out_dir),
        "--py",
        py,
        "--track_backend",
        "hybrid",
        "--from_step",
        "1",
        "--to_step",
        "91",
        "--pose_conf",
        "0.20",
        "--pose_infer_mode",
        str(variant["pose_infer_mode"]),
        "--behavior_infer_mode",
        str(variant["behavior_infer_mode"]),
        "--sr_backend",
        str(variant["sr_backend"]),
        "--sr_scale",
        str(float(args.sr_scale)),
        "--sr_preprocess",
        str(variant.get("sr_preprocess", "off")),
        "--pose_slice_grid",
        str(variant.get("slice_grid", args.slice_grid)),
        "--pose_slice_overlap",
        str(float(variant.get("slice_overlap", args.slice_overlap))),
        "--pose_slice_roi",
        str(args.slice_roi),
        "--behavior_hybrid_match_mode",
        "no_prune",
        "--pose_behavior_video_show_unlinked_behavior_bbox",
        "1",
        "--pose_behavior_video_show_unlinked_behavior_label",
        "0",
        "--pose_behavior_video_show_unlinked_behavior_legend",
        "1",
    ]
    command = args.sr_external_command or os.environ.get(f"{str(variant['sr_backend']).upper()}_SR_COMMAND", "")
    if command:
        cmd += ["--sr_cache_external_command", command]
    cmd += [
        "--pose_track_max_lost_frames",
        str(int(variant.get("pose_track_max_lost_frames", args.pose_track_max_lost_frames))),
        "--pose_track_iou_thres",
        str(float(variant.get("pose_track_iou_thres", args.pose_track_iou_thres))),
        "--pose_track_max_center_dist_ratio",
        str(float(variant.get("pose_track_max_center_dist_ratio", args.pose_track_max_center_dist_ratio))),
        "--pose_track_max_dx_ratio",
        str(float(variant.get("pose_track_max_dx_ratio", args.pose_track_max_dx_ratio))),
        "--pose_track_height_penalty",
        str(float(variant.get("pose_track_height_penalty", args.pose_track_height_penalty))),
        "--pose_track_seat_prior_mode",
        str(variant.get("pose_track_seat_prior_mode", args.pose_track_seat_prior_mode)),
    ]
    if int(args.force) == 1:
        cmd.append("--force")
    started = time.perf_counter()
    print("[RUN]", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    (out_dir / "variant_runtime.json").write_text(
        json.dumps(
            {
                "variant": variant["id"],
                "returncode": int(completed.returncode),
                "elapsed_sec": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return int(completed.returncode)


def _run_official_eval(base_dir: Path, py: str, video: Path, case_dir: Path, gt_jsonl: Path, roi: str) -> int:
    if not gt_jsonl.exists() or not gt_jsonl.is_file():
        return 0
    cmd = [
        py,
        str(base_dir / "scripts" / "19_eval_rear_row_metrics.py"),
        "--case_dir",
        str(case_dir),
        "--gt_jsonl",
        str(gt_jsonl),
        "--video",
        str(video),
        "--roi",
        str(roi),
        "--out_json",
        str(case_dir / "rear_row_metrics.json"),
        "--out_csv",
        str(case_dir / "rear_row_metrics.csv"),
    ]
    print("[EVAL]", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run and collect rear-row ROI SR ablation metrics.")
    parser.add_argument("--video", default="data/智慧课堂学生行为数据集/正方视角/002.mp4")
    parser.add_argument("--out_root", default="output/codex_reports/front_002_sr_ablation")
    parser.add_argument("--py", default=sys.executable)
    parser.add_argument("--variants", default="all", help="Comma-separated variant ids, or all.")
    parser.add_argument("--reuse_existing", type=int, default=1)
    parser.add_argument("--run_pipeline", type=int, default=1)
    parser.add_argument("--run_heavy", type=int, default=0)
    parser.add_argument("--force", type=int, default=0)
    parser.add_argument("--slice_grid", default="2x2", help="NxM, auto/adaptive, rear_adaptive, or rear_dense.")
    parser.add_argument("--slice_overlap", type=float, default=0.25)
    parser.add_argument("--slice_roi", default="auto_rear")
    parser.add_argument("--sr_scale", type=float, default=2.0)
    parser.add_argument("--sr_external_command", default="")
    parser.add_argument("--pose_track_max_lost_frames", type=int, default=500)
    parser.add_argument("--pose_track_iou_thres", type=float, default=0.05)
    parser.add_argument("--pose_track_max_center_dist_ratio", type=float, default=0.15)
    parser.add_argument("--pose_track_max_dx_ratio", type=float, default=0.05)
    parser.add_argument("--pose_track_height_penalty", type=float, default=0.60)
    parser.add_argument("--pose_track_seat_prior_mode", choices=["off", "x_anchor"], default="x_anchor")
    parser.add_argument("--gt_jsonl", default="")
    parser.add_argument("--contact_frames", default="0,25,50,100")
    args = parser.parse_args()

    video = _resolve(base_dir, args.video)
    out_root = _resolve(base_dir, args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    selected = {x.strip() for x in str(args.variants).split(",") if x.strip()}
    variants = VARIANTS if "all" in selected else [v for v in VARIANTS if v["id"] in selected]
    gt_jsonl = _resolve(base_dir, args.gt_jsonl) if args.gt_jsonl else Path("")

    case_dirs: Dict[str, Path] = {}
    for variant in variants:
        out_dir = _variant_dir(base_dir, out_root, variant, bool(int(args.reuse_existing)))
        case_dirs[variant["id"]] = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / "pipeline_contract_v2_report.json"
        backend = str(variant["sr_backend"])
        availability = check_sr_backend_available(backend, args.sr_external_command)
        should_run = int(args.run_pipeline) == 1 and (int(args.force) == 1 or not report.exists())
        if variant.get("requires_external"):
            status = {
                "variant": variant["id"],
                "status": "unavailable",
                "sr_backend": backend,
                "reason": str(variant.get("requires_external")),
                "note": "This row is a planned method. It needs a dedicated implementation before execution.",
            }
            (out_dir / "variant_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[SKIP]", variant["id"], status["reason"])
            continue
        if backend in {"realesrgan", "basicvsrpp", "realbasicvsr", "nvidia_vsr", "maxine_vfx"} and not (
            int(args.run_heavy) == 1 and availability.get("available")
        ):
            status = {
                "variant": variant["id"],
                "status": "unavailable",
                "sr_backend": backend,
                "reason": availability.get("reason", "heavy_backend_disabled"),
                "availability": availability,
                "note": "Set --run_heavy 1 and provide the backend external command env to execute this variant.",
            }
            (out_dir / "variant_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[SKIP]", variant["id"], status["reason"])
            continue
        if should_run:
            rc = _run_pipeline(base_dir, str(args.py), video, out_dir, variant, args)
            if rc != 0:
                status = {
                    "variant": variant["id"],
                    "status": "failed",
                    "sr_backend": backend,
                    "returncode": rc,
                }
                (out_dir / "variant_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        if gt_jsonl.exists() and gt_jsonl.is_file() and (out_dir / "pose_keypoints_v2.jsonl").exists():
            _run_official_eval(base_dir, str(args.py), video, out_dir, gt_jsonl, str(args.slice_roi))

    rows = [_collect_metrics(variant, case_dirs[variant["id"]], video, str(args.slice_roi), gt_jsonl) for variant in variants]
    json_path = out_root / "sr_ablation_metrics.json"
    csv_path = out_root / "sr_ablation_metrics.csv"
    md_path = out_root / "sr_ablation_compare_table.md"
    sheet_path = out_root / "sr_ablation_contact_sheet.jpg"
    json_path.write_text(json.dumps({"video": str(video), "variants": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    frames = [int(x.strip()) for x in str(args.contact_frames).split(",") if x.strip()]
    _write_contact_sheet(sheet_path, video, rows, frames, str(args.slice_roi))
    failure_dir = out_root / "rear_row_failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)
    (failure_dir / "README.md").write_text(
        "# Rear-row Failure Cases\n\n"
        "Put manually selected missed detections, false positives, ID switches, and behavior-confusion frames here. "
        "These examples should be paired with the numeric SR ablation table before making a paper-level claim.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "out_root": str(out_root),
                "metrics_json": str(json_path),
                "metrics_csv": str(csv_path),
                "compare_table": str(md_path),
                "contact_sheet": str(sheet_path) if sheet_path.exists() else "",
                "failure_cases": str(failure_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
