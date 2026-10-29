"""Versioned screenshot localization routes, caches and student confirmation.

YOLO detects people; it never proves that a person is a student. Only a Locate
student result or an uninterrupted, unambiguous association can carry that
confirmation into downstream action review. Raw people remain available to
detection evaluation even when no student action request can be scheduled.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
from PIL import Image

from .contracts import TrackObservation
from .hashing import canonical_sha256, sha256_file, uncached_hashes
from .locate_tracking import LocateStudentIdAssigner, STUDENT_LABELS, associate_locate_rows
from .storage import atomic_write_json
from .yolo_worker import YoloProcessClient


LOCALIZATION_SCHEMA = "classroom_hybrid_ai.localization.v1"
ROUTES = frozenset({"locate_only", "yolo_only", "yolo_locate_hybrid"})
_ASSOCIATION_DEFAULTS = {
    "max_gap_seconds": 2.0,
    "minimum_iou": 0.10,
    "maximum_center_distance": 0.75,
    "minimum_match_score": 0.30,
    "minimum_match_margin": 0.10,
}


class LocalizationError(RuntimeError):
    pass


def _ordered_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[int, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        index, pts = int(row["frame_index"]), float(row["pts_seconds"])
        if index < 0 or not math.isfinite(pts) or pts < 0:
            raise LocalizationError("invalid_localization_frame_or_pts")
        if index in unique:
            previous = unique[index]
            if str(row.get("path")) != str(previous.get("path")) or pts != float(previous["pts_seconds"]):
                raise LocalizationError(f"conflicting_localization_frame:{index}")
            continue
        unique[index] = row
    return [unique[index] for index in sorted(unique)]


def _scene_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    inferred = 0
    for index, row in enumerate(rows):
        explicit = row.get("scene_id", row.get("scene_index", row.get("shot_id")))
        if explicit is not None:
            scene = f"explicit:{explicit}"
        else:
            if index and str(row.get("anchor_source")) == "cpu_scene_change":
                inferred += 1
            scene = f"inferred:{inferred}"
        result.append(scene)
    return result


def _valid_yolo_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping) or not isinstance(result.get("detections"), list):
        raise LocalizationError("yolo_detection_result_malformed")
    detections = []
    for index, raw in enumerate(result["detections"], 1):
        if not isinstance(raw, Mapping):
            raise LocalizationError("yolo_detection_not_mapping")
        if raw.get("class_id") not in (None, 0):
            continue
        if str(raw.get("label", "person")).lower() not in STUDENT_LABELS:
            raise LocalizationError("yolo_person_class_metadata_mismatch")
        bbox = raw.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise LocalizationError("yolo_detection_bbox_malformed")
        bbox = [float(value) for value in bbox]
        if not all(math.isfinite(value) for value in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise LocalizationError("yolo_detection_bbox_invalid")
        score = raw.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise LocalizationError("yolo_detection_confidence_invalid")
        detection = dict(raw)
        detection.pop("student_id", None)
        detection.update({
            "label": "person", "bbox": bbox, "score": float(score), "source": "yolo",
            "detection_index": index, "raw_track_id": raw.get("raw_track_id", raw.get("track_id")),
            "student_confirmed": False, "student_confirmation": "unverified_person",
        })
        detection.pop("track_id", None)
        detections.append(detection)
    return {"detections": detections, "metadata": dict(result.get("metadata", {}))}


def run_yolo_screenshots(
    scenes: Sequence[Mapping[str, Any]], *, model_path: str | Path,
    cache_dir: str | Path, code_fingerprint: str, model_sha256: str | None = None,
    imgsz: int = 960, conf: float = 0.20, batch_size: int = 1, device: Any = 0,
    config_dir: str | Path | None = None, reuse_responses: bool = True,
    adapter_factory: Callable[..., Any] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Atomically cache each screenshot and close YOLO before returning.

    ``adapter_factory`` receives the same keyword arguments as
    YoloTrackerAdapter and is a CPU-test seam. Production always uses a spawn
    worker and verifies both its close ACK and exit code before Locate starts.
    """
    if type(batch_size) is not int or batch_size not in (1, 2, 4):
        raise ValueError("localization_batch_size_must_be_1_2_or_4")
    if type(imgsz) is not int or imgsz <= 0 or not math.isfinite(float(conf)) or not 0 < float(conf) <= 1:
        raise ValueError("invalid_localization_image_size_or_confidence")
    if not code_fingerprint:
        raise ValueError("localization_code_fingerprint_required")
    ordered = _ordered_rows(scenes)
    weights = Path(model_path).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"localization_requires_local_model:{weights}")
    with uncached_hashes():
        actual_model_sha = sha256_file(weights)
    if model_sha256 is not None and actual_model_sha != model_sha256:
        raise LocalizationError("localization_model_sha_mismatch")
    effective = {"imgsz": imgsz, "conf": float(conf), "batch_size": batch_size,
                 "device": str(device), "classes": [0], "operation": "predict", "persist": False}
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], dict[str, Any], Path, tuple[int, int]]] = []
    reused = 0
    for scene in ordered:
        source = Path(str(scene["path"])).resolve()
        with uncached_hashes():
            image_sha = sha256_file(source)
        with Image.open(source) as image:
            size = image.size
        signature = {"schema_version": LOCALIZATION_SCHEMA, "image_sha256": image_sha,
                     "model_sha256": actual_model_sha, "parameters": effective,
                     "code_fingerprint": code_fingerprint}
        key = canonical_sha256(signature)
        path = destination / f"{key}.json"
        cached = None
        if reuse_responses and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("invalid_cached_yolo_payload")
                body = payload.get("result")
                if payload.get("signature") == signature and payload.get("status") == "ok" and canonical_sha256(body) == payload.get("result_sha256"):
                    cached = _valid_yolo_result(body)
            except (ValueError, TypeError, KeyError, OSError, LocalizationError):
                pass
        if cached is not None:
            rows[int(scene["frame_index"])] = _yolo_row(scene, cached, signature, size, True)
            reused += 1
        else:
            pending.append((scene, signature, path, size))
    processed = reused
    if progress_callback is not None:
        progress_callback(processed, len(ordered))
    adapter = None
    requests = 0
    request_timings: list[float] = []
    performances: list[dict[str, Any]] = []
    lifecycle: dict[str, Any] = {"model_started": False, "status": "not_started", "exitcode": 0, "close_acknowledged": True}
    failure: BaseException | None = None
    try:
        if pending:
            kwargs = {"model_path": str(weights), "config_dir": str(config_dir) if config_dir else None, "device": device}
            adapter = adapter_factory(**kwargs) if adapter_factory else YoloProcessClient(kwargs)
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            started = time.perf_counter()
            requests += 1
            result = adapter.predict_batch([str(item[0]["path"]) for item in batch], imgsz=imgsz, conf=float(conf), classes=[0], batch=len(batch))
            elapsed = time.perf_counter() - started
            request_timings.append(elapsed)
            performances.append(dict(getattr(adapter, "last_performance", {})))
            if not isinstance(result, list) or len(result) != len(batch):
                raise LocalizationError("yolo_batch_result_count_mismatch")
            for (scene, signature, path, size), raw in zip(batch, result):
                body = _valid_yolo_result(raw)
                atomic_write_json(path, {"status": "ok", "signature": signature, "result": body,
                                         "result_sha256": canonical_sha256(body), "batch_wall_seconds": elapsed})
                rows[int(scene["frame_index"])] = _yolo_row(scene, body, signature, size, False)
                processed += 1
                if progress_callback is not None:
                    progress_callback(processed, len(ordered))
    except BaseException as exc:
        failure = exc
        atomic_write_json(destination / "last_failure.v1.json", {
            "status": "failed", "error": f"{type(exc).__name__}:{exc}"[:4000],
            "successful_screenshots": processed, "requested_screenshots": len(ordered),
        })
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
            lifecycle = dict(getattr(adapter, "receipt", {}))
            if adapter_factory is not None:
                lifecycle = {"owned_process": False, "model_started": True, "status": "injected_adapter",
                             "close_acknowledged": failure is None, **lifecycle}
        atomic_write_json(destination / "worker_lifecycle.v1.json", lifecycle)
    if failure is not None:
        raise failure
    if adapter_factory is None and (lifecycle.get("close_acknowledged") is not True or type(lifecycle.get("exitcode")) is not int or lifecycle.get("exitcode") != 0):
        raise LocalizationError("yolo_teardown_not_confirmed")
    return [rows[int(row["frame_index"])] for row in ordered], {
        "schema_version": LOCALIZATION_SCHEMA, "status": "ok", "source": "yolo",
        "screenshots": len(ordered), "detections": sum(len(row["detections"]) for row in rows.values()),
        "inference_calls": requests, "inferred_screenshots": len(pending), "cache_hits": reused,
        "model_sha256": actual_model_sha, "effective_parameters": effective,
        "requested_parameters": effective, "request_wall_seconds": request_timings,
        "worker_performance": performances, "lifecycle": lifecycle,
    }


def _yolo_row(scene: Mapping[str, Any], body: Mapping[str, Any], signature: Mapping[str, Any],
              size: tuple[int, int], cache_hit: bool) -> dict[str, Any]:
    return {**dict(scene), **dict(body), "schema_version": LOCALIZATION_SCHEMA, "status": "ok",
            "source": "yolo", "width": size[0], "height": size[1], "image_sha256": signature["image_sha256"],
            "model_sha256": signature["model_sha256"], "effective_parameters": signature["parameters"],
            "cache_hit": cache_hit, "cross_frame_ids": False}


def build_difficulty_queue(
    yolo_rows: Sequence[Mapping[str, Any]], *, audit_interval_seconds: float = 30.0,
    low_confidence: float = 0.28, association_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not math.isfinite(audit_interval_seconds) or audit_interval_seconds <= 0 or not 0 <= low_confidence <= 1:
        raise ValueError("invalid_localization_difficulty_threshold")
    ordered = _ordered_rows(yolo_rows)
    queue, audit = [], []
    prior_scene = None
    previous_ids: set[str] = set()
    last_anchor = 0.0
    assigner = None
    for row, scene in zip(ordered, _scene_ids(ordered)):
        if row.get("status") != "ok":
            raise LocalizationError(f"yolo_failed_frame:{row['frame_index']}")
        reasons: set[str] = set()
        pts = float(row["pts_seconds"])
        if scene != prior_scene:
            reasons.add("scene_first_screenshot")
            assigner = LocateStudentIdAssigner(**{**_ASSOCIATION_DEFAULTS, **dict(association_kwargs or {})})
            previous_ids = set()
            last_anchor = pts
        elif pts - last_anchor >= audit_interval_seconds:
            reasons.add("periodic_audit_anchor")
            last_anchor = pts
        assigned = assigner.update(frame_index=int(row["frame_index"]), pts_seconds=pts, detections=row.get("detections", []))
        current_ids = {str(item["student_id"]) for item in assigned}
        if not assigned:
            reasons.add("zero_person_detections")
        if any(float(item.get("score", 0.0)) < low_confidence for item in assigned):
            reasons.add("low_person_confidence")
        if any(item.get("association_method") == "new_track" for item in assigned):
            reasons.add("new_provisional_track")
        if any(item.get("association_ambiguous") for item in assigned):
            reasons.add("association_ambiguous")
        if previous_ids - current_ids or any(item.get("association_method") == "seat_anchor_reacquisition" for item in assigned):
            reasons.add("broken_provisional_track")
        if any(item.get("association_method") != "new_track" and str(item["student_id"]) not in previous_ids for item in assigned):
            reasons.add("unmatched_previous_screenshot")
        reason_list = sorted(reasons)
        if reasons:
            # _locate_pass copies its input scene into the response. Do not
            # carry YOLO-owned results/model receipts into a Locate request.
            screenshot = dict(row)
            for key in ("source", "detections", "metadata", "model_sha256", "effective_parameters",
                        "cache_hit", "cross_frame_ids", "schema_version", "localization_route"):
                screenshot.pop(key, None)
            queue.append({**screenshot, "difficulty_reasons": reason_list, "localization_scene_id": scene})
        audit.append({"frame_index": int(row["frame_index"]), "pts_seconds": pts, "scene_id": scene,
                      "requires_locate": bool(reasons), "reasons": reason_list,
                      "provisional_associations": [{"provisional_track_id": item["student_id"],
                           "detection_index": item["locate_detection_index"],
                           "method": item["association_method"], "ambiguous": item["association_ambiguous"]}
                           for item in assigned]})
        prior_scene, previous_ids = scene, current_ids
    return queue, {"schema_version": LOCALIZATION_SCHEMA, "policy_version": "difficulty.v1",
                   "screenshots": len(ordered), "locate_screenshots": len(queue), "truncated": False,
                   "audit_interval_seconds": audit_interval_seconds, "low_confidence": low_confidence,
                   "rows": audit}


def merge_localization_rows(
    yolo_rows: Sequence[Mapping[str, Any]], locate_rows: Sequence[Mapping[str, Any]],
    difficulty_queue: Sequence[Mapping[str, Any]], *, route: str,
) -> list[dict[str, Any]]:
    if route not in ROUTES:
        raise ValueError(f"unknown_localization_route:{route}")
    if route == "locate_only":
        return [dict(row) for row in locate_rows]
    ordered = _ordered_rows(yolo_rows)
    required = {int(row["frame_index"]): row for row in difficulty_queue} if route == "yolo_locate_hybrid" else {}
    located = {int(row["frame_index"]): row for row in _ordered_rows(locate_rows)}
    result = []
    for yolo in ordered:
        frame_index = int(yolo["frame_index"])
        if frame_index in required:
            if frame_index not in located or located[frame_index].get("status") != "ok":
                raise LocalizationError(f"required_locate_result_missing_or_failed:{frame_index}")
            raw = located[frame_index]
            if not isinstance(raw.get("detections"), list):
                raise LocalizationError(f"required_locate_result_malformed:{frame_index}")
            if raw.get("source") not in (None, "locate_anything") or any(
                not isinstance(item, Mapping) or item.get("source") not in (None, "locate_anything")
                for item in raw["detections"]
            ):
                raise LocalizationError(f"required_locate_result_wrong_source:{frame_index}")
            if str(raw.get("path")) != str(yolo.get("path")) or float(raw["pts_seconds"]) != float(yolo["pts_seconds"]):
                raise LocalizationError(f"required_locate_result_input_mismatch:{frame_index}")
            selected = {**dict(yolo), **dict(raw), "source": "locate_anything"}
            selected["detections"] = [{**dict(item), "source": "locate_anything"} for item in raw["detections"]]
            selected["difficulty_reasons"] = list(required[frame_index].get("difficulty_reasons", []))
            # YOLO-owned request/cache fields must not describe a Locate row.
            for key in ("model_sha256", "effective_parameters", "cache_hit"):
                if key not in raw:
                    selected.pop(key, None)
        else:
            if yolo.get("status") != "ok":
                raise LocalizationError(f"yolo_failed_frame:{frame_index}")
            selected = dict(yolo)
        selected.update({"schema_version": LOCALIZATION_SCHEMA, "localization_route": route})
        result.append(selected)
    if set(required) - {int(row["frame_index"]) for row in ordered}:
        raise LocalizationError("difficulty_queue_not_subset_of_yolo_schedule")
    return result


def associate_localization_rows(
    rows: Sequence[Mapping[str, Any]], *, route: str,
    max_gap_seconds: float = 2.0, minimum_iou: float = 0.10,
    maximum_center_distance: float = 0.75, minimum_match_score: float = 0.30,
    minimum_match_margin: float = 0.10,
) -> tuple[list[TrackObservation], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reuse legacy CPU matching, then enforce explicit student confirmation."""
    if route not in ROUTES:
        raise ValueError(f"unknown_localization_route:{route}")
    kwargs = {"max_gap_seconds": max_gap_seconds, "minimum_iou": minimum_iou,
              "maximum_center_distance": maximum_center_distance, "minimum_match_score": minimum_match_score,
              "minimum_match_margin": minimum_match_margin}
    if route == "locate_only":
        return associate_locate_rows(rows, **kwargs)
    ordered = _ordered_rows(rows)
    observations, enriched, boxes, stage = associate_locate_rows(ordered, **kwargs)
    detections_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    confirmed_previous: set[str] = set()
    previous_scene = None
    unverified = 0
    confirmed_count = 0
    confirmed_students: set[str] = set()
    all_people: set[str] = set()
    for row, scene in zip(enriched, _scene_ids(enriched)):
        if scene != previous_scene:
            confirmed_previous = set()
        confirmed_now: set[str] = set()
        for raw in row.get("detections", []):
            if not raw.get("student_id"):
                continue
            person_id = str(raw["student_id"])
            all_people.add(person_id)
            source = str(raw.get("source", row.get("source", "yolo")))
            direct = route == "yolo_locate_hybrid" and source == "locate_anything"
            inherited = (route == "yolo_locate_hybrid" and person_id in confirmed_previous
                         and not raw.get("association_ambiguous")
                         and raw.get("association_method") == "seat_anchor_match")
            confirmed = direct or inherited
            reason = "locate_confirmed" if direct else "unambiguous_inheritance" if inherited else "unverified_person"
            raw.update({"source": source, "person_id": person_id, "student_confirmed": confirmed,
                        "student_confirmation": reason, "detector_detection_index": raw.get("locate_detection_index"),
                        "label": "student" if confirmed else "person"})
            if confirmed:
                confirmed_count += 1
                confirmed_students.add(person_id)
                if not raw.get("association_ambiguous"):
                    confirmed_now.add(person_id)
            else:
                unverified += 1
                raw.pop("student_id", None)
            detections_by_key[(int(row["frame_index"]), person_id)] = raw
        row.update({"schema_version": LOCALIZATION_SCHEMA, "localization_route": route,
                    "identity_provider": "cpu_seat_association", "student_confirmation_required": True})
        confirmed_previous, previous_scene = confirmed_now, scene
    result_observations = []
    for observation in observations:
        detection = detections_by_key[(observation.frame_index, observation.track_id)]
        confirmed = bool(detection["student_confirmed"])
        if route == "yolo_locate_hybrid" and not confirmed:
            continue
        uncertain = bool(observation.metadata.get("id_uncertain")) or not confirmed
        attrs = {**dict(detection.get("attributes", {})), **dict(observation.detection.attributes), "student_confirmed": confirmed,
                 "student_confirmation": detection["student_confirmation"], "person_id": observation.track_id,
                 "id_uncertain": uncertain, "detector_detection_index": detection["detector_detection_index"]}
        if detection["source"] == "yolo":
            attrs.pop("locate_detection_index", None)
        metadata = {**dict(observation.metadata), "identity_provider": "cpu_seat_association",
                    "student_confirmed": confirmed, "id_uncertain": uncertain,
                    "localization_contract": LOCALIZATION_SCHEMA, "diagnostic_person_only": not confirmed}
        if detection["source"] == "yolo":
            metadata.pop("locate_frame_detection_id", None)
        result_observations.append(replace(observation,
            detection=replace(observation.detection, source=detection["source"],
                              label="student" if confirmed else "person", attributes=attrs),
            metadata=metadata,
        ))
    for box in boxes:
        person_id = str(box["student_id"])
        detection = detections_by_key[(int(box["frame_index"]), person_id)]
        box.update({"source": detection["source"], "person_id": person_id,
                    "student_confirmed": detection["student_confirmed"],
                    "student_confirmation": detection["student_confirmation"],
                    "label": detection["label"], "id_uncertain": bool(box.get("id_uncertain")) or not detection["student_confirmed"]})
        if not detection["student_confirmed"]:
            box.pop("student_id", None)
    uncertain_ids = set(stage.get("uncertain_student_ids", [])) & confirmed_students
    stage.update({"schema_version": LOCALIZATION_SCHEMA, "source": route,
                  "identity_provider": "cpu_seat_association", "input_boxes_source": route,
                  "yolo_model_loaded": False, "students": len(confirmed_students),
                  "people": len(all_people), "unverified_person_observations": unverified,
                  "confirmed_student_observations": confirmed_count,
                  "observations": len(result_observations), "reliable_students": len(confirmed_students - uncertain_ids),
                  "uncertain_students": len(uncertain_ids), "uncertain_student_ids": sorted(uncertain_ids),
                  "person_track_quality": stage.get("track_quality", {}),
                  "track_quality": {key: value for key, value in stage.get("track_quality", {}).items() if key in confirmed_students},
                  "diagnostic_only": route == "yolo_only", "student_confirmation_required": True})
    return result_observations, enriched, boxes, stage


def _label_rectangles(
    x1: int, y1: int, x2: int, y2: int, text_width: int, text_height: int, baseline: int,
    width: int, height: int,
) -> list[tuple[int, int, int, int, int]]:
    """Return up to two label-plate rects (above, then below) with their text baseline."""

    needed = text_height + baseline + 2
    candidates = [(max(needed, y1) - needed, max(needed, y1))]
    below = y2 + 2
    if below + needed < height:
        candidates.append((below, below + needed))
    rects = []
    for top, bottom in candidates:
        right = min(width - 1, x1 + text_width + 4)
        if top < 0 or bottom >= height or right <= x1:
            continue
        rects.append((x1, top, right, bottom, bottom - baseline - 1))
    return rects


def render_localization_annotations(
    rows: Sequence[Mapping[str, Any]], output_dir: str | Path, *, jpeg_quality: int = 95,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render actual detector provenance and visibly unverified people."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result = []
    rendered = failed = confirmed_boxes = unverified_boxes = suppressed_labels = 0
    # Dense scenes (dozens of people per frame) use compact identifiers and
    # suppress label plates that would overlap an already drawn plate.  Boxes
    # and colours are never dropped; ``annotation_labels`` stays complete for
    # audit even when a plate is hidden.
    dense_label_limit = 10
    for raw_row in rows:
        row = dict(raw_row)
        if row.get("status") != "ok":
            result.append(row)
            continue
        image = cv2.imread(str(row.get("path", "")))
        if image is None:
            result.append({**row, "annotation_status": "failed", "annotation_error": "screenshot_decode_failed"})
            failed += 1
            continue
        height, width = image.shape[:2]
        labels = []
        occupied: list[tuple[int, int, int, int]] = []
        detections = row.get("detections") or ()
        dense = len(detections) > dense_label_limit
        for index, detection in enumerate(detections, 1):
            bbox = detection.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
            y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
            if x1 >= x2 or y1 >= y2:
                continue
            confirmed = bool(detection.get("student_confirmed"))
            uncertain = bool(detection.get("association_ambiguous") or detection.get("id_uncertain"))
            identifier = str(detection.get("student_id") if confirmed else detection.get("person_id") or f"person-frame-{index}")
            source = str(detection.get("source", row.get("source", "unknown")))
            display_source = {"yolo": "YOLO", "locate_anything": "Locate"}.get(source, source)
            detection_index = detection.get("detector_detection_index", detection.get("detection_index", index))
            label = f"{identifier}{'?' if uncertain and confirmed else ''} | {display_source} D{detection_index}"
            if not confirmed:
                label += " | unverified person"
            labels.append(label)
            color = (128, 128, 128) if not confirmed else (0, 190, 255) if uncertain else (60, 220, 60)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if confirmed else 1)
            short = f"{identifier}{'?' if uncertain and confirmed else ''}" + ("" if confirmed else " | U")
            text = short if dense else label
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)
            for plate in _label_rectangles(x1, y1, x2, y2, text_width, text_height, baseline, width, height):
                left, top, right, bottom, text_y = plate
                if any(not (right <= other[0] or other[2] <= left or bottom <= other[1] or other[3] <= top) for other in occupied):
                    continue
                cv2.rectangle(image, (left, top), (right, bottom), color, -1)
                cv2.putText(image, text, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
                occupied.append((left, top, right, bottom))
                break
            else:
                suppressed_labels += 1
            confirmed_boxes += int(confirmed)
            unverified_boxes += int(not confirmed)
        target = destination / f"frame_{int(row['frame_index']):09d}.jpg"
        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
            row.update({"annotation_status": "failed", "annotation_error": "annotation_write_failed"})
            failed += 1
        else:
            row.update({"annotation_status": "ok", "annotated_path": str(target), "annotation_labels": labels})
            rendered += 1
        result.append(row)
    return result, {"schema_version": LOCALIZATION_SCHEMA, "status": "warning" if failed else "ok",
                    "rendered_screenshots": rendered, "failed_screenshots": failed,
                    "rendered_student_boxes": confirmed_boxes, "rendered_unverified_person_boxes": unverified_boxes,
                    "suppressed_label_count": suppressed_labels,
                    "artifact_dir": str(destination), "label_format": "student/person ID | actual detector D<frame-local index> | confirmation"}


__all__ = ["LOCALIZATION_SCHEMA", "ROUTES", "LocalizationError", "run_yolo_screenshots",
           "build_difficulty_queue", "merge_localization_rows", "associate_localization_rows",
           "render_localization_annotations"]
