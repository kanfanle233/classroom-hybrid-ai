"""Read-only evidence diagnosis and unlabelled human annotation preparation.

No prediction becomes gold, and a model's rationale remains a model claim.
The only files written are the explicitly requested diagnostic outputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


REASON_CATEGORIES = (
    "target_small_or_occluded", "missing_critical_parts", "identity_uncertain",
    "model_abstention", "parse_failure", "budget_uncovered",
)
VISUAL_REASONS = frozenset(REASON_CATEGORIES[:2])


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for ordinal, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"row_must_be_object:{path}:{ordinal}")
        result.append(value)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse replacement: independent audits must remain independently reviewable.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return _sha(path)


def _candidate_id(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate_event_id") or row.get("event_id") or "")


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _unknown(row: Mapping[str, Any]) -> bool:
    return str(row.get("behavior") or "UNKNOWN").upper() in {"UNKNOWN", "NULL", "NONE"} or row.get("decision") == "uncertain"


def _model_claims(rationale: str) -> list[str]:
    """Conservative search labels a claim, never a verified visual cause."""
    claims = []
    if any(token in rationale.lower() for token in ("遮挡", "太小", "过小", "occluded", "too small")):
        claims.append("target_small_or_occluded")
    if any(token in rationale.lower() for token in ("看不到手", "手部不可见", "手部缺失", "无书写接触", "看不清手", "hands not visible", "missing hand", "手部遮挡")):
        claims.append("missing_critical_parts")
    if any(token in rationale.lower() for token in ("身份不确定", "无法确定目标", "identity uncertain", "target ambiguous")):
        claims.append("identity_uncertain")
    return claims


def diagnose_unknown(run_dir: Path, *, output_path: Path, review_path: Path | None = None) -> dict[str, Any]:
    """Audit unique candidates from a normal S5 run or stability-run assets.

    Optional human review JSONL requires candidate_event_id, reason_categories,
    reviewer_id and evidence_note. It remains diagnosis, not action gold.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"run_directory_missing:{root}")
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    inputs: dict[str, str] = {}

    def rows(relative: str) -> list[dict[str, Any]]:
        path = root / relative
        if path.is_file():
            inputs[str(path)] = _sha(path)
        return _read_rows(path)

    selected = rows("artifacts/candidates.selected.jsonl")
    unselected = rows("artifacts/candidates.unselected.jsonl")
    queue = rows("artifacts/qwen_crop_queue.jsonl")
    screenshot_rows = rows("artifacts/screenshot_schedule.v1.jsonl")
    screenshot_by_frame = {row.get("frame_index"): row for row in screenshot_rows}
    localization_by_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows("artifacts/locate_student_boxes.jsonl"):
        localization_by_student[str(row.get("student_id") or "")].append(row)
    manifest_path = root / "run_manifest.v1.json"
    manifest = {}
    if manifest_path.is_file():
        inputs[str(manifest_path)] = _sha(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = {_candidate_id(row): row for row in rows("evidence/index.jsonl")}
    candidates = {_candidate_id(row): row for row in selected + unselected if _candidate_id(row)}
    queue_by_id = {_candidate_id(row): row for row in queue if _candidate_id(row)}
    candidates.update({key: row for key, row in queue_by_id.items() if key not in candidates})

    response_files = sorted((root / "responses" / "stability").glob("*/*.json"))
    if not response_files:
        active_path = root / "artifacts" / "s5_active_response_set.v1.json"
        if active_path.is_file():
            inputs[str(active_path)] = _sha(active_path)
            active = json.loads(active_path.read_text(encoding="utf-8"))
            prefix = str(active.get("response_prefix") or "")
            directory = (root / prefix).resolve()
            if not prefix or not directory.is_relative_to(root / "responses"):
                raise ValueError("active_response_prefix_outside_responses")
            response_files = sorted(directory.glob("*.json"))
        else:
            response_files = sorted((root / "responses" / "classification").glob("*.json"))
    normalized_reviews = {_candidate_id(row): row for row in rows("reviews.classification.jsonl")}
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    read_errors = []
    for path in response_files:
        inputs[str(path)] = _sha(path)
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(response, dict):
                raise ValueError("response_not_object")
        except (ValueError, UnicodeError) as exc:
            read_errors.append({"path": str(path), "error": str(exc)})
            response = {"candidate_event_id": path.stem, "schema_valid": False, "behavior": "UNKNOWN", "diagnostic_error": str(exc)}
        key = _candidate_id(response) or path.stem
        # Current S5 reviews carry schema/transport classifications that are
        # deliberately absent from some error response envelopes.
        if key in normalized_reviews and "stability" not in path.parts:
            response = {**normalized_reviews[key], **response}
        attempts[key].append({"path": str(path), "response": response})
    # Normal runs without response files still expose normalized review artifacts.
    if not response_files:
        for relative in ("reviews.classification.jsonl", "artifacts/qwen_reviews.jsonl"):
            review_rows = rows(relative)
            if review_rows:
                for row in review_rows:
                    key = _candidate_id(row)
                    if key:
                        attempts[key].append({"path": str(root / relative), "response": row})
                break

    human_reviews: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if review_path is not None:
        review_file = Path(review_path).resolve()
        inputs[str(review_file)] = _sha(review_file)
        for row in _read_rows(review_file):
            key = _candidate_id(row)
            categories = row.get("reason_categories")
            if not key or key not in set(candidates) | set(attempts):
                raise ValueError("human_review_unknown_candidate")
            if not isinstance(categories, list) or any(item not in REASON_CATEGORIES for item in categories):
                raise ValueError("human_review_invalid_reason_categories")
            if not str(row.get("reviewer_id") or "").strip() or not str(row.get("evidence_note") or "").strip():
                raise ValueError("human_review_requires_reviewer_and_evidence_note")
            human_reviews[key].append(row)

    image_cache: dict[Path, dict[str, Any]] = {}

    def image_ref(item: Mapping[str, Any]) -> dict[str, Any]:
        raw_path = str(item.get("path") or "")
        if not raw_path:
            return {"status": "unavailable", "reason": "image_path_missing"}
        path = Path(raw_path)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path not in image_cache:
            if path.is_file():
                digest = _sha(path)
                measured: dict[str, Any] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size, "status": "available"}
                try:
                    from PIL import Image
                    with Image.open(path) as img:
                        measured["size_px"] = list(img.size)
                except Exception as exc:
                    measured.update(status="unreadable", error=type(exc).__name__)
                image_cache[path] = measured
            else:
                image_cache[path] = {"path": str(path), "status": "missing"}
        result = dict(image_cache[path])
        expected = item.get("sha256")
        if expected:
            result["recorded_sha256"] = expected
            result["integrity"] = "matched" if result.get("sha256") == expected else "mismatch_or_missing"
        return result

    records = []
    reason_counts: Counter[str] = Counter()
    human_counts: Counter[str] = Counter()
    claimed_counts: Counter[str] = Counter()
    for key in sorted(set(candidates) | set(attempts)):
        candidate = candidates.get(key, {})
        packet = evidence.get(key, {})
        student_id = candidate.get("track_id") or candidate.get("student_id") or packet.get("track_id")
        pts = candidate.get("trigger_pts_seconds", candidate.get("pts_seconds"))
        localizations = localization_by_student.get(str(student_id), [])
        if isinstance(pts, (int, float)):
            localizations = sorted(localizations, key=lambda row: abs(float(row.get("pts_seconds", 0)) - pts))[:3]
        event_attempts = attempts.get(key, [])
        responses = [item["response"] for item in event_attempts]
        unknown_count = sum(_unknown(row) for row in responses)
        reasons = set()
        canonical_reason = str(candidate.get("canonical_reason") or candidate.get("selection_reason") or "")
        if canonical_reason == "budget_exhausted" or candidate.get("selection_reason") == "budget_exhausted":
            reasons.add("budget_uncovered")
        if _metadata(candidate).get("id_uncertain") is True or canonical_reason == "id_uncertain" or any(row.get("association_ambiguous") is True for row in localizations):
            reasons.add("identity_uncertain")
        if any(row.get("reason") == "qwen_call_budget_exhausted" or row.get("reviewer") == "pipeline_budget" for row in responses):
            reasons.add("budget_uncovered")
        if any(row.get("schema_valid") is False and row.get("reviewer") not in {"pipeline_error", "pipeline_budget"} for row in responses):
            reasons.add("parse_failure")
        if any(_unknown(row) and row.get("schema_valid") is True for row in responses):
            reasons.add("model_abstention")
        claim_categories = set()
        rationales = Counter(str(row.get("rationale") or "") for row in responses)
        for rationale in rationales:
            claim_categories.update(_model_claims(rationale))
        human_categories = {reason for row in human_reviews.get(key, []) for reason in row["reason_categories"]}
        reason_counts.update(reasons)
        human_counts.update(human_categories)
        claimed_counts.update(claim_categories)
        evidence_rows = packet.get("evidence") or []
        references = []
        for item in evidence_rows:
            if isinstance(item, dict):
                references.append({**{name: item[name] for name in ("kind", "frame_index", "actual_pts_seconds", "requested_pts_seconds", "bbox_xyxy", "localization_source") if name in item}, "image": image_ref(item)})
        composite = packet.get("qwen_composite") or {}
        if not composite and queue_by_id.get(key, {}).get("composite_path"):
            composite = {"path": queue_by_id[key]["composite_path"]}
        records.append({
            "candidate_event_id": key, "student_id": student_id,
            "pts_seconds": pts,
            "attempt_count": len(responses), "unknown_attempt_count": unknown_count,
            "distinct_outputs": len({(str(row.get("decision")), str(row.get("behavior"))) for row in responses}),
            "diagnostic_reasons": sorted(reasons), "model_claimed_reasons": sorted(claim_categories),
            "human_reviewed_reasons": sorted(human_categories), "human_reviews": human_reviews.get(key, []),
            "visual_cause_status": "human_reviewed" if human_categories & VISUAL_REASONS else "unverified",
            "rationales": [{"text": key, "attempt_count": count, "source": "model_response"} for key, count in sorted(rationales.items())],
            "responses": [{"path": item["path"], "schema_valid": item["response"].get("schema_valid"), "decision": item["response"].get("decision"), "behavior": item["response"].get("behavior")} for item in event_attempts],
            "request_failures": [{"path": item["path"], "error": item["response"].get("error") or _metadata(item["response"]).get("error")} for item in event_attempts if item["response"].get("reviewer") == "pipeline_error" or item["response"].get("status") == "error"],
            "evidence": references, "actual_composite": image_ref(composite),
            "localization_records": localizations,
            "original_screenshots": [image_ref(screenshot_by_frame[frame]) for frame in sorted({row.get("frame_index") for row in localizations if type(row.get("frame_index")) is int}) if frame in screenshot_by_frame],
            "visual_review_checklist": {"hands_visible": "pending", "desk_contact_visible": "pending", "interaction_target_visible": "pending", "target_identity_clear": "pending"},
            "absence_of_response": "no_saved_response" if not responses else None,
        })
    total = sum(row["attempt_count"] for row in records)
    result = {
        "schema_version": "classroom_hybrid_ai.unknown_diagnostics.v1", "evaluation_status": "not_evaluated",
        "reason": "diagnostic_prediction_statistics_without_independent_gold", "run_dir": str(root),
        "diagnostic_code_sha256": _sha(Path(__file__)), "source": manifest.get("source", {}),
        "deduplication_key": "candidate_event_id", "reason_categories": list(REASON_CATEGORIES),
        "summary": {"attempts": total, "unique_candidates": len(records), "unknown_attempts": sum(row["unknown_attempt_count"] for row in records), "unique_candidates_with_unknown": sum(row["unknown_attempt_count"] > 0 for row in records), "diagnostic_reason_candidate_counts": dict(reason_counts), "model_claimed_reason_candidate_counts": dict(claimed_counts), "human_reviewed_reason_candidate_counts": dict(human_counts)},
        "limitations": ["repeated_candidates_are_not_independent_quality_samples", "saved_responses_are_normalized_transport_responses; original_model_text_and_exact_prompt_not_retained_in_legacy_assets", "visual_causes_require_human_evidence_review; geometry_and_model_claims_are_not_causal_proof", "unselected_without_explicit_budget_reason_is_not_assumed_budget_exhaustion"],
        "input_sha256": inputs, "read_errors": read_errors, "candidates": records,
    }
    _write_json(Path(output_path), result)
    return result


def _video_frames(video_path: Path) -> list[dict[str, Any]]:
    import av
    frames = []
    previous = None
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            pts = float(frame.pts * stream.time_base) if frame.pts is not None else None
            valid = pts is not None and math.isfinite(pts) and pts >= 0 and (previous is None or pts > previous)
            frames.append({"frame_index": index, "pts_seconds": pts, "pts_status": "valid" if valid else "requires_manual_timeline_check", "width": frame.width, "height": frame.height})
            if pts is not None and math.isfinite(pts):
                previous = pts
    if not frames:
        raise ValueError("video_has_no_decodable_frames")
    return frames


def prepare_annotation_package(video_path: Path, *, output_dir: Path, run_dir: Path | None = None, classroom_id: str | None = None, lesson_id: str | None = None, session_id: str | None = None) -> dict[str, Any]:
    """Prepare a new unlabelled, two-reviewer video package with exact frame PTS.

    Candidate frame references may supplement a one-second coverage grid; model
    boxes, model identities and actions are never copied into annotation files.
    """
    video = Path(video_path).resolve(strict=True)
    directory = Path(output_dir).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    source_sha = _sha(video)
    ledger = _video_frames(video)
    chosen = {0, len(ledger) - 1}
    next_second = 0.0
    for row in ledger:
        pts = row["pts_seconds"]
        if row["pts_status"] == "valid" and pts >= next_second:
            chosen.add(row["frame_index"])
            next_second = math.floor(pts) + 1.0
    run_ref = None
    if run_dir is not None:
        root = Path(run_dir).resolve()
        from .paper_evaluation import _run_metadata, _run_source_sha
        meta, _ = _run_metadata(root)
        if _run_source_sha(root, meta) != source_sha:
            raise ValueError("annotation_run_source_sha256_mismatch")
        run_ref = str(root)
        for row in _read_rows(root / "artifacts" / "screenshot_schedule.v1.jsonl"):
            index = row.get("frame_index")
            if type(index) is int and 0 <= index < len(ledger):
                chosen.add(index)
    directory.mkdir(parents=True)
    hashes = {"frame_ledger.v1.jsonl": _write_rows(directory / "frame_ledger.v1.jsonl", ledger)}
    review_rows = [{"frame_index": index, "pts_seconds": ledger[index]["pts_seconds"], "pts_status": ledger[index]["pts_status"], "confirmed_empty": None, "review_status": "pending", "reviewers": [], "disagreement_resolution": "pending"} for index in sorted(chosen)]
    hashes["annotated_frames.template.jsonl"] = _write_rows(directory / "annotated_frames.template.jsonl", review_rows)
    for name in ("boxes.v1.jsonl", "actions.v1.jsonl", "stable_identities.v1.jsonl"):
        hashes[name] = _write_rows(directory / name, [])
    identity = {"classroom_id": classroom_id, "source_lesson_id": lesson_id, "capture_session_id": session_id}
    package = {
        "schema_version": "classroom_hybrid_ai.annotation_preparation.v1", "status": "pending_human_annotation",
        "evaluation_status": "not_evaluated", "source_video": str(video), "source_video_sha256": source_sha,
        **identity, "source_identity_status": "provided_pending_source_evidence" if all(identity.values()) else "unresolved",
        "source_identity_evidence": [], "split": None, "split_status": "pending_source_isolation_audit",
        "frame_count": len(ledger), "suggested_review_frames": len(chosen), "frame_selection": "first_last_and_one_second_grid_plus_optional_run_screenshots",
        "frame_selection_is_gold": False, "run_reference": run_ref, "artifact_sha256": hashes,
        "double_review_status": "pending", "reviewer_ids": [], "disagreement_resolution": "pending",
        "license_status": "unverified", "permitted_uses": [],
        "promotion_requirements": ["verify_source_lesson_session_classroom_with_evidence", "assign_isolated_development_or_test_split", "annotate_student_boxes_and_stable_human_person_ids", "annotate_only_observable_core8_action_intervals", "confirm_each_reviewed_frame_empty_or_fully_annotated", "two_distinct_reviewers_and_adjudication_complete", "write_new_gold_manifest_with_final_artifact_hashes"],
        "row_contracts": {
            "boxes": ["frame_index", "pts_seconds", "gold_person_id", "bbox_px"],
            "actions": ["gold_person_id", "start_pts_seconds", "end_pts_seconds", "action_id"],
            "stable_identities": ["gold_person_id", "role", "first_frame_index", "last_frame_index", "identity_evidence", "reviewers"],
            "annotated_frames": ["frame_index", "pts_seconds", "confirmed_empty", "review_status=complete", "reviewers (two distinct human IDs)"],
        },
    }
    _write_json(directory / "annotation_package.v1.json", package)
    _write_json(directory / "independent_splits.template.json", {"schema_version": "classroom_hybrid_ai.independent_split_preparation.v1", "status": "pending_source_identity", "existing_frozen_data_unchanged": True, "split_units": ["source_video_sha256", "source_lesson_id", "capture_session_id", "classroom_id"], "development": [], "test": [], "unassigned": [{"source_video_sha256": source_sha, **identity}], "requirements": "Resolve source identities with evidence before assigning any split; do not use frozen test images for tuning."})
    (directory / "ANNOTATION_README.md").write_text("# 人工视频标注包\n\n本目录尚无金标。frame_ledger 记录逐帧解码所得 PTS；标注模板只建议首尾和每秒帧，可按动作边界增加帧。\n\n复核原视频后填写学生框和人工稳定 ID，教师不得标为学生。仅记录可观察的 core8 动作区间，不把未观测时间填成 UNKNOWN 金标。每个已复核帧必须标明 confirmed_empty；有框时为 false，无学生时为 true。review_status 完成后须有两名不同复核人及分歧处理记录。\n\n先凭来源证据确认课堂、课次和拍摄会话，按来源分配独立 development/test。模板中的未决信息不得自动推断。核验许可、完成双人标注后，单独生成 gold_manifest.v1.json 并重新计算全部 JSONL SHA；当前 annotation_package 不能作为 gold_manifest 使用。\n", encoding="utf-8")
    return package
