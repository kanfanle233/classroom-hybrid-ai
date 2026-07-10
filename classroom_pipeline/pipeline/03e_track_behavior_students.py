import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, validate_jsonl_file, validate_pose_uq_record


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _resolve(base_dir: Path, raw: str) -> Optional[Path]:
    if not str(raw or "").strip():
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _bbox_iou(a: List[float], b: List[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _bbox_diag(bbox: List[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(1.0, math.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1)))


def _center_affinity(a: List[float], b: List[float]) -> float:
    acx, acy = _bbox_center(a)
    bcx, bcy = _bbox_center(b)
    dist = math.hypot(acx - bcx, acy - bcy)
    scale = max(_bbox_diag(a), _bbox_diag(b), 1.0)
    return _clamp01(1.0 - dist / (0.65 * scale))


def _upper_pose_bbox(bbox: List[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    h = max(1.0, y2 - y1)
    return [x1, y1, x2, y1 + (0.68 * h)]


def _expand_behavior_bbox(bbox: List[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return [
        x1 - (0.18 * w),
        y1 - (0.08 * h),
        x2 + (0.18 * w),
        y2 + (0.95 * h),
    ]


def _track_match_score(track_bbox: List[float], det_bbox: List[float]) -> float:
    iou = _bbox_iou(track_bbox, det_bbox)
    if iou > 0:
        return iou
    return 0.35 * _center_affinity(track_bbox, det_bbox)


def _hybrid_match_details(pose_bbox: List[float], det_bbox: List[float]) -> Dict[str, float]:
    raw_iou = _bbox_iou(pose_bbox, det_bbox)
    upper_pose = _upper_pose_bbox(pose_bbox)
    expanded_det = _expand_behavior_bbox(det_bbox)
    upper_iou = _bbox_iou(upper_pose, det_bbox)
    expanded_iou = _bbox_iou(pose_bbox, expanded_det)
    iou = max(raw_iou, upper_iou, expanded_iou)
    affinity = max(_center_affinity(pose_bbox, det_bbox), _center_affinity(upper_pose, det_bbox))
    score = (0.55 * iou) + (0.45 * affinity)
    return {
        "iou": round(iou, 6),
        "raw_iou": round(raw_iou, 6),
        "upper_iou": round(upper_iou, 6),
        "expanded_iou": round(expanded_iou, 6),
        "center_affinity": round(affinity, 6),
        "score": round(score, 6),
    }


def _round_bbox(raw: List[float]) -> List[float]:
    return [round(float(v), 3) for v in raw]


def _valid_bbox(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    vals = [_safe_float(v) for v in raw]
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def _semantic_id(det: Dict[str, Any]) -> str:
    return str(det.get("semantic_id", det.get("action", det.get("label", "")))).strip().lower()


def _behavior_code(det: Dict[str, Any]) -> str:
    return str(det.get("behavior_code", det.get("label", ""))).strip().lower()


@dataclass
class Observation:
    frame: int
    t: float
    bbox: List[float]
    conf: float
    item: Dict[str, Any]
    top_behaviors: List[Dict[str, Any]]
    behavior_track_id: int = -1
    output_track_id: int = -1
    linked_pose_track_id: Optional[int] = None
    link_score: float = 0.0


@dataclass
class TrackState:
    track_id: int
    bbox: List[float]
    last_frame: int
    conf: float
    frames_seen: int = 0
    observations: List[Observation] = field(default_factory=list)


def _cluster_frame_behaviors(
    row: Dict[str, Any],
    *,
    same_person_iou: float,
    track_low_thresh: float,
) -> List[Observation]:
    frame = int(row.get("frame", 0))
    t = _safe_float(row.get("t", 0.0), 0.0)
    behaviors = row.get("behaviors", row.get("behavior_detections", []))
    if not isinstance(behaviors, list):
        return []

    candidates: List[Dict[str, Any]] = []
    for raw in behaviors:
        if not isinstance(raw, dict):
            continue
        bbox = _valid_bbox(raw.get("bbox"))
        if bbox is None:
            continue
        conf = _clamp01(_safe_float(raw.get("conf", raw.get("confidence", 0.0)), 0.0))
        if conf < float(track_low_thresh):
            continue
        det = dict(raw)
        det["bbox"] = bbox
        det["conf"] = conf
        candidates.append(det)

    candidates.sort(key=lambda x: float(x.get("conf", 0.0)), reverse=True)
    clusters: List[List[Dict[str, Any]]] = []
    for det in candidates:
        bbox = det["bbox"]
        best_idx = -1
        best_score = 0.0
        for idx, cluster in enumerate(clusters):
            anchor = cluster[0]["bbox"]
            iou = _bbox_iou(bbox, anchor)
            score = max(iou, 0.50 * _center_affinity(bbox, anchor) if iou >= 0.05 else 0.0)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score >= float(same_person_iou):
            clusters[best_idx].append(det)
        else:
            clusters.append([det])

    observations: List[Observation] = []
    for cluster in clusters:
        cluster.sort(key=lambda x: float(x.get("conf", 0.0)), reverse=True)
        best = dict(cluster[0])
        top_behaviors: List[Dict[str, Any]] = []
        for det in cluster[:5]:
            top_behaviors.append(
                {
                    "label": str(det.get("label", "")).strip().lower(),
                    "action": str(det.get("action", "")).strip().lower(),
                    "behavior_code": _behavior_code(det),
                    "semantic_id": _semantic_id(det),
                    "conf": round(_clamp01(_safe_float(det.get("conf", 0.0), 0.0)), 6),
                }
            )
        observations.append(
            Observation(
                frame=frame,
                t=t,
                bbox=[float(v) for v in best["bbox"]],
                conf=_clamp01(_safe_float(best.get("conf", 0.0), 0.0)),
                item=best,
                top_behaviors=top_behaviors,
            )
        )
    return observations


def _track_observations(
    frame_observations: Dict[int, List[Observation]],
    *,
    iou_thres: float,
    track_buffer: int,
    new_track_thresh: float,
) -> Tuple[Dict[int, TrackState], Dict[str, Any]]:
    tracks: Dict[int, TrackState] = {}
    next_tid = 1
    raw_observation_count = 0
    matched_count = 0

    for frame in sorted(frame_observations):
        observations = sorted(frame_observations[frame], key=lambda x: x.conf, reverse=True)
        raw_observation_count += len(observations)
        used_tids: set[int] = set()

        for obs in observations:
            best_tid = -1
            best_score = -1.0
            for tid, track in tracks.items():
                if tid in used_tids:
                    continue
                gap = int(obs.frame) - int(track.last_frame)
                if gap < 0 or gap > int(track_buffer):
                    continue
                score = _track_match_score(track.bbox, obs.bbox)
                if score >= float(iou_thres) and score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid < 0:
                if obs.conf < float(new_track_thresh):
                    continue
                best_tid = next_tid
                next_tid += 1
                tracks[best_tid] = TrackState(
                    track_id=best_tid,
                    bbox=list(obs.bbox),
                    last_frame=int(obs.frame),
                    conf=float(obs.conf),
                )
            else:
                matched_count += 1

            track = tracks[best_tid]
            track.bbox = list(obs.bbox)
            track.last_frame = int(obs.frame)
            track.conf = float(obs.conf)
            track.frames_seen += 1
            obs.behavior_track_id = int(best_tid)
            track.observations.append(obs)
            used_tids.add(best_tid)

    stats = {
        "raw_observations": raw_observation_count,
        "matched_observations": matched_count,
        "created_tracks": len(tracks),
    }
    return tracks, stats


def _pose_frame_index(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        frame = row.get("frame")
        if not isinstance(frame, int):
            continue
        people = row.get("persons", row.get("people", []))
        if isinstance(people, dict):
            people = list(people.values())
        if not isinstance(people, list):
            continue
        valid: List[Dict[str, Any]] = []
        for person in people:
            if not isinstance(person, dict):
                continue
            tid = person.get("track_id")
            bbox = _valid_bbox(person.get("bbox"))
            if not isinstance(tid, int) or bbox is None:
                continue
            keypoints = person.get("keypoints", [])
            if not isinstance(keypoints, list):
                keypoints = []
            valid.append(
                {
                    "track_id": int(tid),
                    "bbox": bbox,
                    "conf": _clamp01(_safe_float(person.get("conf", 0.0), 0.0)),
                    "keypoints": keypoints,
                }
            )
        out[int(frame)] = {
            "frame": int(frame),
            "t": round(_safe_float(row.get("t", 0.0), 0.0), 6),
            "persons": valid,
        }
    return out


def _pose_index(path: Optional[Path]) -> Dict[int, List[Dict[str, Any]]]:
    pose_frames = _pose_frame_index(path)
    return {frame: list(data.get("persons", [])) for frame, data in pose_frames.items()}


def _link_to_pose_tracks(
    tracks: Dict[int, TrackState],
    pose_by_frame: Dict[int, List[Dict[str, Any]]],
    *,
    link_iou: float,
    unlinked_track_offset: int,
) -> Dict[int, Dict[str, Any]]:
    scores: Dict[Tuple[int, int], Dict[str, float]] = {}
    for behavior_tid, track in tracks.items():
        for obs in track.observations:
            for pose in pose_by_frame.get(int(obs.frame), []):
                iou = _bbox_iou(obs.bbox, pose["bbox"])
                if iou < float(link_iou):
                    continue
                key = (int(behavior_tid), int(pose["track_id"]))
                item = scores.setdefault(key, {"score": 0.0, "count": 0.0})
                item["score"] += float(iou) * max(0.1, float(obs.conf))
                item["count"] += 1.0

    candidates: List[Tuple[float, int, int, float]] = []
    for (behavior_tid, pose_tid), data in scores.items():
        count = max(1.0, data["count"])
        mean_score = data["score"] / count
        candidates.append((float(data["score"]), int(behavior_tid), int(pose_tid), float(mean_score)))
    candidates.sort(reverse=True)

    used_behavior: set[int] = set()
    used_pose: set[int] = set()
    mapping: Dict[int, Dict[str, Any]] = {}
    for total_score, behavior_tid, pose_tid, mean_score in candidates:
        if behavior_tid in used_behavior or pose_tid in used_pose:
            continue
        mapping[behavior_tid] = {
            "output_track_id": pose_tid,
            "linked_pose_track_id": pose_tid,
            "link_score": round(mean_score, 6),
            "link_total_score": round(total_score, 6),
            "track_link_source": "pose_iou",
        }
        used_behavior.add(behavior_tid)
        used_pose.add(pose_tid)

    for behavior_tid in tracks:
        if behavior_tid in mapping:
            continue
        mapping[behavior_tid] = {
            "output_track_id": int(unlinked_track_offset) + int(behavior_tid),
            "linked_pose_track_id": None,
            "link_score": 0.0,
            "link_total_score": 0.0,
            "track_link_source": "unlinked_behavior",
        }
    return mapping


def _passthrough_track_mapping(tracks: Dict[int, TrackState]) -> Dict[int, Dict[str, Any]]:
    return {
        int(tid): {
            "output_track_id": int(tid),
            "linked_pose_track_id": None,
            "link_score": 1.0,
            "link_total_score": 1.0,
            "track_link_source": "behavior_only",
        }
        for tid in tracks
    }


def _behavior_candidate_entry(obs: Observation, details: Dict[str, float]) -> Dict[str, Any]:
    return {
        "score": round(float(details["score"]), 6),
        "iou": round(float(details["iou"]), 6),
        "raw_iou": round(float(details.get("raw_iou", details["iou"])), 6),
        "upper_iou": round(float(details.get("upper_iou", 0.0)), 6),
        "expanded_iou": round(float(details.get("expanded_iou", 0.0)), 6),
        "center_affinity": round(float(details["center_affinity"]), 6),
        "bbox": _round_bbox(obs.bbox),
        "conf": round(float(obs.conf), 6),
        "semantic_id": _semantic_id(obs.item),
        "behavior_code": _behavior_code(obs.item),
        "label": str(obs.item.get("label", "")).strip().lower(),
    }


def _pose_candidate_entry(pose: Dict[str, Any], details: Dict[str, float]) -> Dict[str, Any]:
    return {
        "track_id": int(pose["track_id"]),
        "score": round(float(details["score"]), 6),
        "iou": round(float(details["iou"]), 6),
        "raw_iou": round(float(details.get("raw_iou", details["iou"])), 6),
        "upper_iou": round(float(details.get("upper_iou", 0.0)), 6),
        "expanded_iou": round(float(details.get("expanded_iou", 0.0)), 6),
        "center_affinity": round(float(details["center_affinity"]), 6),
        "bbox": _round_bbox(pose["bbox"]),
        "pose_conf": round(float(pose.get("conf", 0.0)), 6),
    }


def _is_viable_hybrid_match(details: Dict[str, float], score_thresh: float, match_mode: str) -> bool:
    iou = float(details["iou"])
    affinity = float(details["center_affinity"])
    score = float(details["score"])
    mode = str(match_mode or "relaxed").strip().lower()
    if mode == "no_prune":
        # ByteTrack-style: do not discard plausible boxes before association.
        # Keep only mathematically unrelated pairs out of the assignment pool.
        return score > 0.0 or iou > 0.0 or affinity > 0.0
    if mode == "strict":
        return score >= float(score_thresh) and (iou >= 0.05 or affinity >= 0.55)
    return score >= float(score_thresh) and (iou >= 0.025 or affinity >= 0.42)


def _person_from_observation(obs: Observation, *, track_frame_count: int) -> Dict[str, Any]:
    item = obs.item
    semantic_id = _semantic_id(item)
    behavior_code = _behavior_code(item)
    raw_action = str(item.get("raw_action", item.get("action", ""))).strip().lower()
    person = {
        "track_id": int(obs.output_track_id),
        "behavior_track_id": int(obs.behavior_track_id),
        "linked_pose_track_id": obs.linked_pose_track_id,
        "track_link_score": round(float(obs.link_score), 6),
        "track_link_source": str(item.get("track_link_source", "")),
        "bbox": _round_bbox(obs.bbox),
        "conf": round(float(obs.conf), 6),
        "det_conf": round(float(obs.conf), 6),
        "track_conf": round(float(obs.conf), 6),
        "action": semantic_id,
        "raw_action": raw_action,
        "behavior_code": behavior_code,
        "behavior_label_zh": str(item.get("behavior_label_zh", "")).strip(),
        "behavior_label_en": str(item.get("behavior_label_en", "")).strip(),
        "semantic_id": semantic_id,
        "semantic_label_zh": str(item.get("semantic_label_zh", "")).strip(),
        "semantic_label_en": str(item.get("semantic_label_en", "")).strip(),
        "taxonomy_version": str(item.get("taxonomy_version", "")).strip(),
        "cls_id": item.get("cls_id"),
        "label": str(item.get("label", "")).strip().lower(),
        "source": "behavior_student_tracker",
        "track_frame_count": int(track_frame_count),
        "top_behaviors": obs.top_behaviors,
        "behavior_match_status": "matched",
        "behavior_match_score": round(float(obs.link_score), 6) if obs.linked_pose_track_id is not None else 0.0,
        "behavior_bbox": _round_bbox(obs.bbox),
        "behavior_candidates_topk": [],
    }
    return person


def _person_from_pose_backbone(
    pose: Dict[str, Any],
    *,
    track_frame_count: int,
    match: Optional[Dict[str, Any]],
    candidate_topk: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pose_tid = int(pose["track_id"])
    pose_bbox = pose["bbox"]
    pose_conf = _clamp01(_safe_float(pose.get("conf", 0.0), 0.0))
    person = {
        "track_id": pose_tid,
        "behavior_track_id": None,
        "linked_pose_track_id": pose_tid,
        "track_link_score": 1.0,
        "track_link_source": "pose_backbone",
        "bbox": _round_bbox(pose_bbox),
        "pose_bbox": _round_bbox(pose_bbox),
        "behavior_bbox": [],
        "conf": round(float(pose_conf), 6),
        "det_conf": 0.0,
        "pose_conf": round(float(pose_conf), 6),
        "track_conf": round(float(pose_conf), 6),
        "action": "",
        "raw_action": "",
        "behavior_code": "",
        "behavior_label_zh": "",
        "behavior_label_en": "",
        "semantic_id": "",
        "semantic_label_zh": "",
        "semantic_label_en": "",
        "taxonomy_version": "",
        "cls_id": None,
        "label": "",
        "source": "behavior_student_tracker",
        "track_frame_count": int(track_frame_count),
        "top_behaviors": [],
        "behavior_match_status": "unmatched",
        "behavior_match_score": 0.0,
        "behavior_iou": 0.0,
        "behavior_center_affinity": 0.0,
        "behavior_candidates_topk": candidate_topk,
    }

    if not match:
        return person

    obs = match["obs"]
    item = obs.item
    semantic_id = _semantic_id(item)
    behavior_code = _behavior_code(item)
    details = match["details"]
    person.update(
        {
            "behavior_track_id": obs.behavior_track_id if obs.behavior_track_id >= 0 else None,
            "behavior_bbox": _round_bbox(obs.bbox),
            "conf": round(float(obs.conf), 6),
            "det_conf": round(float(obs.conf), 6),
            "action": semantic_id,
            "raw_action": str(item.get("raw_action", item.get("action", ""))).strip().lower(),
            "behavior_code": behavior_code,
            "behavior_label_zh": str(item.get("behavior_label_zh", "")).strip(),
            "behavior_label_en": str(item.get("behavior_label_en", "")).strip(),
            "semantic_id": semantic_id,
            "semantic_label_zh": str(item.get("semantic_label_zh", "")).strip(),
            "semantic_label_en": str(item.get("semantic_label_en", "")).strip(),
            "taxonomy_version": str(item.get("taxonomy_version", "")).strip(),
            "cls_id": item.get("cls_id"),
            "label": str(item.get("label", "")).strip().lower(),
            "top_behaviors": obs.top_behaviors,
            "behavior_match_status": "matched",
            "behavior_match_score": round(float(details["score"]), 6),
            "behavior_iou": round(float(details["iou"]), 6),
            "behavior_center_affinity": round(float(details["center_affinity"]), 6),
        }
    )
    return person


def _unmatched_behavior_row(
    *,
    frame: int,
    t: float,
    unmatched: List[Tuple[Observation, List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION + "+behavior_unmatched_v1",
        "frame": int(frame),
        "t": round(float(t), 6),
        "behaviors": [
            {
                "bbox": _round_bbox(obs.bbox),
                "conf": round(float(obs.conf), 6),
                "action": _semantic_id(obs.item),
                "raw_action": str(obs.item.get("raw_action", obs.item.get("action", ""))).strip().lower(),
                "behavior_code": _behavior_code(obs.item),
                "behavior_label_zh": str(obs.item.get("behavior_label_zh", "")).strip(),
                "behavior_label_en": str(obs.item.get("behavior_label_en", "")).strip(),
                "semantic_id": _semantic_id(obs.item),
                "semantic_label_zh": str(obs.item.get("semantic_label_zh", "")).strip(),
                "semantic_label_en": str(obs.item.get("semantic_label_en", "")).strip(),
                "taxonomy_version": str(obs.item.get("taxonomy_version", "")).strip(),
                "label": str(obs.item.get("label", "")).strip().lower(),
                "top_behaviors": obs.top_behaviors,
                "pose_candidates_topk": candidates[:3],
            }
            for obs, candidates in unmatched
        ],
    }


def _build_pose_backbone_outputs(
    pose_frames: Dict[int, Dict[str, Any]],
    frame_observations: Dict[int, List[Observation]],
    *,
    fps: float,
    match_score_thres: float,
    match_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    track_frame_counts: Dict[int, int] = {}
    pose_track_ids: set[int] = set()
    for frame_data in pose_frames.values():
        for pose in frame_data.get("persons", []):
            tid = int(pose["track_id"])
            pose_track_ids.add(tid)
            track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1

    student_rows: List[Dict[str, Any]] = []
    pose_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []
    matched_pose_track_ids: set[int] = set()
    matched_behavior_items = 0
    unmatched_behavior_items = 0
    pose_person_rows = 0
    pose_person_rows_matched = 0

    for frame in sorted(pose_frames):
        frame_data = pose_frames[frame]
        t = _safe_float(frame_data.get("t", frame / fps), frame / fps)
        pose_people = sorted(
            list(frame_data.get("persons", [])),
            key=lambda p: (float(p["bbox"][0]), float(p["bbox"][1]), int(p["track_id"])),
        )
        observations = list(frame_observations.get(frame, []))
        pose_candidates: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(len(pose_people))}
        obs_candidates: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(len(observations))}
        pair_candidates: List[Tuple[float, int, int, Dict[str, float]]] = []

        for pose_idx, pose in enumerate(pose_people):
            for obs_idx, obs in enumerate(observations):
                details = _hybrid_match_details(pose["bbox"], obs.bbox)
                pose_candidates[pose_idx].append(_behavior_candidate_entry(obs, details))
                obs_candidates[obs_idx].append(_pose_candidate_entry(pose, details))
                if _is_viable_hybrid_match(details, match_score_thres, match_mode):
                    pair_candidates.append((float(details["score"]), pose_idx, obs_idx, details))

        for candidates in pose_candidates.values():
            candidates.sort(key=lambda x: (float(x["score"]), float(x["iou"]), float(x["conf"])), reverse=True)
        for candidates in obs_candidates.values():
            candidates.sort(key=lambda x: (float(x["score"]), float(x["iou"]), float(x["pose_conf"])), reverse=True)

        pair_candidates.sort(key=lambda x: (x[0], float(x[3]["iou"]), float(x[3]["center_affinity"])), reverse=True)
        matched_pose: set[int] = set()
        matched_obs: set[int] = set()
        assignments: Dict[int, Dict[str, Any]] = {}
        for _, pose_idx, obs_idx, details in pair_candidates:
            if pose_idx in matched_pose or obs_idx in matched_obs:
                continue
            assignments[pose_idx] = {"obs": observations[obs_idx], "details": details}
            matched_pose.add(pose_idx)
            matched_obs.add(obs_idx)

        persons_out: List[Dict[str, Any]] = []
        pose_row_persons: List[Dict[str, Any]] = []
        for idx, pose in enumerate(pose_people):
            tid = int(pose["track_id"])
            person = _person_from_pose_backbone(
                pose,
                track_frame_count=int(track_frame_counts.get(tid, 0)),
                match=assignments.get(idx),
                candidate_topk=pose_candidates.get(idx, [])[:3],
            )
            pose_person_rows += 1
            if person["behavior_match_status"] == "matched":
                pose_person_rows_matched += 1
                matched_pose_track_ids.add(tid)
                matched_behavior_items += 1
            persons_out.append(person)

            pose_person = dict(person)
            keypoints = pose.get("keypoints", [])
            if not (isinstance(keypoints, list) and keypoints):
                keypoints = _synthetic_keypoints(pose["bbox"])
            pose_person["person_idx"] = idx
            pose_person["keypoints"] = keypoints
            pose_row_persons.append(pose_person)

        if persons_out:
            student_rows.append(
                {
                    "schema_version": SCHEMA_VERSION + "+student_tracks_v2",
                    "frame": int(frame),
                    "t": round(float(t), 6),
                    "persons": persons_out,
                }
            )
            pose_rows.append({"frame": int(frame), "t": round(float(t), 6), "persons": pose_row_persons})

        unmatched_payload: List[Tuple[Observation, List[Dict[str, Any]]]] = []
        for obs_idx, obs in enumerate(observations):
            if obs_idx in matched_obs:
                continue
            unmatched_behavior_items += 1
            unmatched_payload.append((obs, obs_candidates.get(obs_idx, [])[:3]))
        if unmatched_payload:
            unmatched_rows.append(_unmatched_behavior_row(frame=int(frame), t=float(t), unmatched=unmatched_payload))

    orphan_frames = sorted(set(frame_observations.keys()) - set(pose_frames.keys()))
    for frame in orphan_frames:
        observations = frame_observations.get(frame, [])
        if not observations:
            continue
        unmatched_behavior_items += len(observations)
        unmatched_rows.append(
            _unmatched_behavior_row(
                frame=int(frame),
                t=float(observations[0].t if observations else (frame / fps)),
                unmatched=[(obs, []) for obs in observations],
            )
        )

    stats = {
        "pose_students_total": len(pose_track_ids),
        "pose_students_with_behavior_match": len(matched_pose_track_ids),
        "matched_behavior_items": matched_behavior_items,
        "unmatched_behavior_items": unmatched_behavior_items,
        "pose_person_rows": pose_person_rows,
        "pose_person_rows_matched": pose_person_rows_matched,
        "pose_person_rows_unmatched": max(0, pose_person_rows - pose_person_rows_matched),
        "linked_tracks": len(matched_pose_track_ids),
        "student_track_rows": len(student_rows),
        "student_track_items": sum(len(r.get("persons", [])) for r in student_rows),
        "behavior_unmatched_rows": len(unmatched_rows),
        "orphan_behavior_frames": len(orphan_frames),
    }
    return student_rows, pose_rows, unmatched_rows, stats


def _synthetic_keypoints(bbox: List[float]) -> List[Dict[str, float]]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    rel = [
        (0.50, 0.12),
        (0.44, 0.10),
        (0.56, 0.10),
        (0.38, 0.13),
        (0.62, 0.13),
        (0.35, 0.32),
        (0.65, 0.32),
        (0.28, 0.50),
        (0.72, 0.50),
        (0.25, 0.68),
        (0.75, 0.68),
        (0.40, 0.68),
        (0.60, 0.68),
        (0.38, 0.86),
        (0.62, 0.86),
        (0.36, 0.98),
        (0.64, 0.98),
    ]
    points = []
    for rx, ry in rel:
        points.append({"x": round(x1 + rx * w, 3), "y": round(y1 + ry * h, 3), "c": 0.0})
    if not points:
        points.append({"x": round(cx, 3), "y": round(cy, 3), "c": 0.0})
    return points


def _build_behavior_outputs(
    tracks: Dict[int, TrackState],
    mapping: Dict[int, Dict[str, Any]],
    *,
    min_track_frames: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    frame_rows: Dict[int, Dict[str, Any]] = {}
    kept_tracks = 0
    dropped_tracks = 0
    linked_tracks = 0

    for behavior_tid, track in tracks.items():
        if int(track.frames_seen) < int(min_track_frames):
            dropped_tracks += 1
            continue
        kept_tracks += 1
        link = mapping.get(int(behavior_tid), {})
        if link.get("linked_pose_track_id") is not None:
            linked_tracks += 1
        for obs in track.observations:
            obs.output_track_id = int(link.get("output_track_id", behavior_tid))
            linked_pose = link.get("linked_pose_track_id")
            obs.linked_pose_track_id = int(linked_pose) if linked_pose is not None else None
            obs.link_score = _safe_float(link.get("link_score", 0.0), 0.0)
            obs.item["track_link_source"] = str(link.get("track_link_source", ""))
            row = frame_rows.setdefault(
                int(obs.frame),
                {
                    "schema_version": SCHEMA_VERSION + "+student_tracks_v2",
                    "frame": int(obs.frame),
                    "t": round(float(obs.t), 6),
                    "persons": [],
                },
            )
            row["persons"].append(_person_from_observation(obs, track_frame_count=int(track.frames_seen)))

    student_rows: List[Dict[str, Any]] = []
    pose_rows: List[Dict[str, Any]] = []
    for frame in sorted(frame_rows):
        row = frame_rows[frame]
        row["persons"].sort(key=lambda p: (float(p["bbox"][0]), float(p["bbox"][1]), int(p["track_id"])))
        student_rows.append(row)
        pose_persons: List[Dict[str, Any]] = []
        for idx, person in enumerate(row["persons"]):
            bbox = [float(v) for v in person["bbox"]]
            pose_person = dict(person)
            pose_person["person_idx"] = idx
            pose_person["keypoints"] = _synthetic_keypoints(bbox)
            pose_persons.append(pose_person)
        pose_rows.append({"frame": row["frame"], "t": row["t"], "persons": pose_persons})

    stats = {
        "kept_tracks": kept_tracks,
        "dropped_tracks": dropped_tracks,
        "linked_tracks": linked_tracks,
        "student_track_rows": len(student_rows),
        "student_track_items": sum(len(r.get("persons", [])) for r in student_rows),
    }
    return student_rows, pose_rows, stats


def _write_pose_uq(path: Path, pose_rows: List[Dict[str, Any]], *, validate: bool) -> int:
    prev_bbox: Dict[int, List[float]] = {}
    prev_frame: Dict[int, int] = {}
    out_rows: List[Dict[str, Any]] = []
    for row in pose_rows:
        frame = int(row.get("frame", 0))
        t = _safe_float(row.get("t", 0.0), 0.0)
        persons = row.get("persons", [])
        if not isinstance(persons, list):
            continue
        person_out: List[Dict[str, Any]] = []
        for person in persons:
            if not isinstance(person, dict):
                continue
            tid = person.get("track_id")
            bbox = _valid_bbox(person.get("bbox"))
            if not isinstance(tid, int) or bbox is None:
                continue
            conf_instability = 1.0 - _clamp01(_safe_float(person.get("conf", 0.5), 0.5))
            if tid in prev_bbox:
                iou = _bbox_iou(prev_bbox[tid], bbox)
                gap = max(1, frame - prev_frame.get(tid, frame - 1))
                motion = _clamp01((1.0 - iou) + max(0, gap - 1) * 0.05)
                bbox_shift = _clamp01(1.0 - iou)
            else:
                motion = 0.5
                bbox_shift = 0.5
            uq_track = _clamp01(0.45 * conf_instability + 0.35 * motion + 0.20 * bbox_shift)
            person_out.append(
                {
                    "track_id": int(tid),
                    "uq_track": round(uq_track, 6),
                    "uq_conf": round(conf_instability, 6),
                    "uq_motion": round(motion, 6),
                    "uq_kpt": round(bbox_shift, 6),
                    "log_sigma2": round(math.log(max(1e-6, uq_track + 1e-4)), 6),
                    "uq_score": round(uq_track, 6),
                    "uq_source": ["behavior_bbox_compat"],
                    "motion_stability": round(1.0 - motion, 6),
                    "bbox_stability": round(1.0 - bbox_shift, 6),
                }
            )
            prev_bbox[int(tid)] = bbox
            prev_frame[int(tid)] = frame

        if person_out:
            uq_frame = sum(float(p["uq_track"]) for p in person_out) / max(1, len(person_out))
            out_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "uq_type": "behavior_bbox_compat",
                    "uq_scope": "track_sequence",
                    "presence_aware": True,
                    "variance_head": False,
                    "frame": frame,
                    "t": round(t, 6),
                    "uq_frame": round(uq_frame, 6),
                    "persons": person_out,
                }
            )
    written = _write_jsonl(path, out_rows)
    if validate:
        ok, _, errors = validate_jsonl_file(path, validate_pose_uq_record)
        if not ok:
            first = errors[0] if errors else "unknown schema error"
            raise ValueError(f"invalid behavior-compatible UQ schema: {first}")
    return written


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Track students from semantic behavior detections.")
    parser.add_argument("--in", dest="in_path", required=True, type=str, help="behavior_det.semantic.jsonl")
    parser.add_argument("--out", required=True, type=str, help="student_tracks.jsonl")
    parser.add_argument("--track_backend", choices=["behavior", "hybrid"], default="behavior")
    parser.add_argument("--pose_tracks", default="", type=str, help="pose_tracks_smooth.jsonl for hybrid pose backbone")
    parser.add_argument("--behavior_unmatched_out", default="", type=str)
    parser.add_argument("--pose_keypoints_compat_out", default="", type=str)
    parser.add_argument("--pose_tracks_compat_out", default="", type=str)
    parser.add_argument("--pose_uq_out", default="", type=str)
    parser.add_argument("--report", default="", type=str)
    parser.add_argument("--fps", default=25.0, type=float)
    parser.add_argument("--tracker", choices=["bytetrack", "botsort", "simple"], default="bytetrack")
    parser.add_argument("--same_person_iou", default=0.55, type=float)
    parser.add_argument("--iou_thres", default=0.30, type=float)
    parser.add_argument("--track_low_thresh", default=0.10, type=float)
    parser.add_argument("--new_track_thresh", default=0.25, type=float)
    parser.add_argument("--track_buffer", default=30, type=int)
    parser.add_argument("--min_track_frames", default=3, type=int)
    parser.add_argument("--link_pose", default=1, type=int)
    parser.add_argument("--link_iou", default=0.20, type=float)
    parser.add_argument("--hybrid_match_mode", choices=["strict", "relaxed", "no_prune"], default="no_prune")
    parser.add_argument("--unlinked_track_offset", default=200000, type=int)
    parser.add_argument("--validate_uq", default=1, type=int)
    args = parser.parse_args()

    in_path = _resolve(base_dir, args.in_path)
    out_path = _resolve(base_dir, args.out)
    pose_tracks_path = _resolve(base_dir, args.pose_tracks)
    behavior_unmatched_out = _resolve(base_dir, args.behavior_unmatched_out)
    pose_keypoints_compat_out = _resolve(base_dir, args.pose_keypoints_compat_out)
    pose_tracks_compat_out = _resolve(base_dir, args.pose_tracks_compat_out)
    pose_uq_out = _resolve(base_dir, args.pose_uq_out)
    report_path = _resolve(base_dir, args.report) or (out_path.with_suffix(".report.json") if out_path else None)
    if in_path is None or out_path is None or report_path is None:
        raise ValueError("input, output, and report paths must resolve")
    if not in_path.exists():
        raise FileNotFoundError(f"behavior semantic detections not found: {in_path}")

    rows = _read_jsonl(in_path)
    frame_observations: Dict[int, List[Observation]] = {}
    for row in rows:
        if not isinstance(row.get("frame"), int):
            continue
        observations = _cluster_frame_behaviors(
            row,
            same_person_iou=max(0.0, min(1.0, float(args.same_person_iou))),
            track_low_thresh=max(0.0, min(1.0, float(args.track_low_thresh))),
        )
        if observations:
            frame_observations[int(row["frame"])] = observations

    track_backend = str(args.track_backend).strip().lower()
    pose_keypoints_written = 0
    pose_tracks_written = 0
    pose_uq_written = 0
    behavior_unmatched_written = 0

    if track_backend == "hybrid":
        if pose_tracks_path is None or not pose_tracks_path.exists():
            raise FileNotFoundError("track_backend=hybrid requires --pose_tracks to exist")
        pose_frames = _pose_frame_index(pose_tracks_path)
        if not pose_frames:
            raise ValueError("track_backend=hybrid requires non-empty pose_tracks input")

        student_rows, pose_rows, unmatched_rows, output_stats = _build_pose_backbone_outputs(
            pose_frames,
            frame_observations,
            fps=max(1.0, float(args.fps)),
            match_score_thres=max(0.0, min(1.0, float(args.link_iou))),
            match_mode=str(args.hybrid_match_mode),
        )
        track_stats = {
            "raw_observations": sum(len(items) for items in frame_observations.values()),
            "matched_observations": int(output_stats.get("matched_behavior_items", 0)),
            "created_tracks": 0,
        }
        if behavior_unmatched_out is not None:
            behavior_unmatched_written = _write_jsonl(behavior_unmatched_out, unmatched_rows)
    else:
        tracks, track_stats = _track_observations(
            frame_observations,
            iou_thres=max(0.0, min(1.0, float(args.iou_thres))),
            track_buffer=max(1, int(args.track_buffer)),
            new_track_thresh=max(0.0, min(1.0, float(args.new_track_thresh))),
        )
        pose_by_frame = _pose_index(pose_tracks_path) if int(args.link_pose) == 1 else {}
        if pose_by_frame:
            mapping = _link_to_pose_tracks(
                tracks,
                pose_by_frame,
                link_iou=max(0.0, min(1.0, float(args.link_iou))),
                unlinked_track_offset=int(args.unlinked_track_offset),
            )
        else:
            mapping = _passthrough_track_mapping(tracks)
        student_rows, pose_rows, output_stats = _build_behavior_outputs(
            tracks,
            mapping,
            min_track_frames=max(1, int(args.min_track_frames)),
        )

    student_written = _write_jsonl(out_path, student_rows)
    if pose_keypoints_compat_out is not None:
        pose_keypoints_written = _write_jsonl(pose_keypoints_compat_out, pose_rows)
    if pose_tracks_compat_out is not None:
        pose_tracks_written = _write_jsonl(pose_tracks_compat_out, pose_rows)
    if pose_uq_out is not None:
        pose_uq_written = _write_pose_uq(pose_uq_out, pose_rows, validate=bool(int(args.validate_uq)))

    tracker_impl = "pose_backbone_frame_matcher" if track_backend == "hybrid" else "byte_style_jsonl_iou_association"
    report = {
        "stage": "track_behavior_students",
        "input": str(in_path),
        "output": str(out_path),
        "track_backend": track_backend,
        "tracker_requested": str(args.tracker),
        "tracker_impl": tracker_impl,
        "pose_tracks": str(pose_tracks_path) if pose_tracks_path else "",
        "behavior_unmatched_output": str(behavior_unmatched_out) if behavior_unmatched_out else "",
        "rows_in": len(rows),
        "frames_with_observations": len(frame_observations),
        "student_rows_written": student_written,
        "pose_keypoints_compat_rows_written": pose_keypoints_written,
        "pose_tracks_compat_rows_written": pose_tracks_written,
        "pose_uq_rows_written": pose_uq_written,
        "behavior_unmatched_rows_written": behavior_unmatched_written,
        "params": {
            "same_person_iou": float(args.same_person_iou),
            "iou_thres": float(args.iou_thres),
            "track_low_thresh": float(args.track_low_thresh),
            "new_track_thresh": float(args.new_track_thresh),
            "track_buffer": int(args.track_buffer),
            "min_track_frames": int(args.min_track_frames),
            "link_pose": int(args.link_pose),
            "link_iou": float(args.link_iou),
            "hybrid_match_mode": str(args.hybrid_match_mode),
            "unlinked_track_offset": int(args.unlinked_track_offset),
        },
        "stats": {**track_stats, **output_stats},
        "status": "ok" if student_written > 0 else "failed",
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
