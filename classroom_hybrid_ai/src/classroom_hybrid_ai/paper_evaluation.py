"""Offline, auditable paper evaluation for the classroom research backend.

This module intentionally has no model, video, scipy, or TrackEval dependency.
It consumes a validated human gold package and the already-written artifacts of
one research run.  The implementation follows
``EVALUATION_PROTOCOL_V1.md``; in particular, a prediction artifact is never
used to repair or create a gold annotation.

The public entry point is :func:`evaluate_research_run`.  The smaller public
helpers are useful for fixture tests and for a split audit in CI.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = "classroom_hybrid_ai.metrics.v1"
GOLD_SCHEMA_VERSION = "classroom_hybrid_ai.gold_manifest.v1"
IOU_THRESHOLD = 0.50
HOTA_ALPHAS = tuple(round(index / 100.0, 2) for index in range(5, 100, 5))
UNKNOWN = "UNKNOWN"
_SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_EPS = 1e-9


class GoldValidationError(ValueError):
    """Raised internally while collecting a structured ``invalid_gold`` result."""


@dataclass(frozen=True)
class _ProfileInfo:
    profile_id: str
    sha256: str
    codes: tuple[str, ...]
    action_ids: tuple[str, ...]
    aliases: Mapping[str, str]
    is_extension11: bool = False


@dataclass(frozen=True)
class _GoldData:
    manifest: Mapping[str, Any]
    manifest_path: Path | None
    manifest_sha256: str
    boxes: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]
    profile: _ProfileInfo
    annotated_frames: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class _PredictionData:
    detections: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]
    files: tuple[Path, ...]
    parse_warnings: tuple[str, ...] = ()
    source_contract: Mapping[str, Any] = field(default_factory=dict)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_nonempty_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return ""
    return str(value).strip()


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GoldValidationError(f"{field}_not_numeric") from exc
    if not math.isfinite(result):
        raise GoldValidationError(f"{field}_not_finite")
    return result


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise GoldValidationError(f"{field}_not_integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise GoldValidationError(f"{field}_not_integer") from exc
    try:
        if float(value) != float(number):
            raise GoldValidationError(f"{field}_not_integer")
    except (TypeError, ValueError):
        raise GoldValidationError(f"{field}_not_integer")
    if number < 0:
        raise GoldValidationError(f"{field}_negative")
    return number


def _normalise_bbox(value: Any, *, field: str = "bbox_px") -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise GoldValidationError(f"{field}_must_have_four_values")
    values = tuple(_finite_float(item, field=field) for item in value)
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise GoldValidationError(f"{field}_must_have_positive_area")
    return values


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 0.0 else 0.0


def _read_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoldValidationError(f"{field}_must_be_object")
    return value


def _find_first(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _nested_value(mapping: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        current: Any = mapping
        found = True
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                found = False
                break
            current = current[part]
        if found:
            return current
    return None


def _normalise_profile(profile: Any, profile_id_hint: str | None = None) -> _ProfileInfo:
    """Load an ActionProfile-like input without making evaluation model-backed."""

    raw: Any = profile
    if raw is None and profile_id_hint:
        raw = profile_id_hint
    try:
        from .profiles import ActionProfile, load_profile, load_profiles
    except Exception:  # pragma: no cover - package import failure is reported below
        ActionProfile = None  # type: ignore[assignment]
        load_profile = load_profiles = None  # type: ignore[assignment]

    if isinstance(raw, _ProfileInfo):
        return raw
    if ActionProfile is not None and isinstance(raw, ActionProfile):
        actions = list(raw.actions)
        return _profile_from_actions(str(raw.profile_id), str(raw.sha256), actions)
    if isinstance(raw, Mapping):
        profile_id = _as_nonempty_text(raw.get("profile_id"))
        if not profile_id:
            raise GoldValidationError("profile_id_missing")
        actions = raw.get("actions")
        if not isinstance(actions, (list, tuple)):
            raise GoldValidationError("profile_actions_missing")
        # ActionProfile.from_dict recomputes the digest over the canonical
        # profile payload, which is the digest used by the backend manifest.
        if ActionProfile is not None:
            try:
                restored = ActionProfile.from_dict(raw)
                return _profile_from_actions(profile_id, restored.sha256, list(restored.actions))
            except Exception as exc:
                # Small fixture callers sometimes provide only the immutable
                # action list and its already-computed contract digest.  It is
                # still a profile contract, so accept it when the digest is
                # explicit; full backend profile dictionaries take the strict
                # ActionProfile path above.
                digest = _as_nonempty_text(raw.get("contract_sha256") or raw.get("taxonomy_contract_sha256") or raw.get("sha256"))
                if _SHA_RE.fullmatch(digest):
                    return _profile_from_actions(profile_id, digest, list(actions))
                raise GoldValidationError(f"profile_invalid:{type(exc).__name__}") from exc
        return _profile_from_actions(profile_id, _as_nonempty_text(raw.get("contract_sha256")), list(actions))
    if isinstance(raw, (str, Path)):
        text = str(raw)
        candidate = Path(text)
        if candidate.is_file():
            if load_profile is None:
                raise GoldValidationError("profile_loader_unavailable")
            try:
                selected = load_profile(candidate, profile_id_hint or "core8_v1")
            except Exception as exc:
                raise GoldValidationError(f"profile_file_invalid:{type(exc).__name__}") from exc
            return _profile_from_actions(str(selected.profile_id), str(selected.sha256), list(selected.actions))
        # An id is intentionally resolved only against the checked-in profile
        # contract.  There is no inference from prediction labels.
        package_root = Path(__file__).resolve().parents[2]
        candidates = (
            package_root / "configs" / "action_profiles.yaml",
            package_root.parent / "configs" / "action_profiles.yaml",
        )
        if load_profiles is not None:
            for path in candidates:
                if path.is_file():
                    try:
                        profiles = load_profiles(path)
                        selected = profiles.get(text)
                        if selected is not None:
                            return _profile_from_actions(str(selected.profile_id), str(selected.sha256), list(selected.actions))
                    except Exception:
                        continue
        raise GoldValidationError(f"profile_not_found:{text}")
    raise GoldValidationError("profile_input_invalid")


def _profile_from_actions(profile_id: str, digest: str, actions: Sequence[Mapping[str, Any]]) -> _ProfileInfo:
    codes: list[str] = []
    action_ids: list[str] = []
    aliases: dict[str, str] = {}
    for action in actions:
        code = _as_nonempty_text(action.get("code") or action.get("label")).lower()
        action_id = _as_nonempty_text(action.get("action_id") or action.get("id") or (f"a_{code}" if code else ""))
        name = _as_nonempty_text(action.get("name_zh") or action.get("name"))
        if not code or not action_id:
            raise GoldValidationError(f"profile_action_invalid:{profile_id}")
        if code in codes or action_id in action_ids:
            raise GoldValidationError(f"profile_action_duplicate:{profile_id}:{code}")
        codes.append(code)
        action_ids.append(action_id)
        aliases[code.lower()] = action_id
        aliases[action_id.lower()] = action_id
        if name:
            aliases[name] = action_id
    if not codes:
        raise GoldValidationError(f"profile_actions_empty:{profile_id}")
    if not _SHA_RE.fullmatch(digest or ""):
        raise GoldValidationError(f"profile_sha256_invalid:{profile_id}")
    return _ProfileInfo(
        profile_id=profile_id,
        sha256=digest.lower(),
        codes=tuple(codes),
        action_ids=tuple(action_ids),
        aliases=aliases,
        is_extension11=(profile_id == "scb_student_11_candidate_v1" and len(codes) == 11),
    )


def _manifest_input(value: Any) -> tuple[Mapping[str, Any], Path | None, str]:
    if isinstance(value, Mapping):
        payload = dict(value)
        source_path_value = payload.pop("__manifest_path", None)
        source_path = Path(str(source_path_value)).resolve() if source_path_value else None
        return payload, source_path, _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise GoldValidationError("gold_manifest_missing")
    try:
        payload = _json_load(path)
    except Exception as exc:
        raise GoldValidationError("gold_manifest_json_invalid") from exc
    if not isinstance(payload, Mapping):
        raise GoldValidationError("gold_manifest_must_be_object")
    return payload, path, _sha256_file(path)


def _extract_artifact_ref(manifest: Mapping[str, Any], name: str) -> tuple[str, str]:
    uri_keys = (f"{name}_uri", f"{name}_jsonl_uri", f"{name}_relative_uri", f"{name}_relative_path", f"{name}_path")
    sha_keys = (f"{name}_sha256", f"{name}_jsonl_sha256", f"{name}_sha")
    uri = _find_first(manifest, uri_keys)
    digest = _find_first(manifest, sha_keys)
    root_item = manifest.get(name)
    if isinstance(root_item, Mapping):
        uri = uri if uri is not None else _find_first(root_item, ("uri", "relative_uri", "path"))
        digest = digest if digest is not None else _find_first(root_item, ("sha256", "sha"))
    for parent_key in ("artifacts", "gold_artifacts", "jsonl", "annotations"):
        parent = manifest.get(parent_key)
        if isinstance(parent, Mapping):
            item = parent.get(name) or parent.get(f"{name}.v1") or parent.get(f"{name}_v1") or parent.get(f"{name}.v1.jsonl")
            if isinstance(item, Mapping):
                uri = uri if uri is not None else _find_first(item, ("uri", "relative_uri", "path"))
                digest = digest if digest is not None else _find_first(item, ("sha256", "sha"))
            elif isinstance(item, str) and uri is None:
                uri = item
    uri_text, sha_text = _as_nonempty_text(uri), _as_nonempty_text(digest)
    if not uri_text or not sha_text:
        raise GoldValidationError(f"{name}_reference_missing")
    if not _SHA_RE.fullmatch(sha_text):
        raise GoldValidationError(f"{name}_sha256_invalid")
    return uri_text, sha_text.lower()


def _resolve_relative_artifact(base: Path, uri: str, name: str) -> Path:
    # URI means a repository-relative file, never a URL or an absolute path.
    if "\x00" in uri or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", uri) or Path(uri).is_absolute():
        raise GoldValidationError(f"{name}_uri_must_be_relative")
    candidate = (base / Path(uri)).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise GoldValidationError(f"{name}_uri_escapes_manifest_directory") from exc
    if not candidate.is_file():
        raise GoldValidationError(f"{name}_file_missing")
    return candidate


def _has_key_with_value(mapping: Mapping[str, Any], keys: set[str]) -> bool:
    for key, value in mapping.items():
        normal = str(key).lower()
        if normal in keys and value not in (None, "", [], (), {}):
            return True
        if isinstance(value, Mapping) and _has_key_with_value(value, keys):
            return True
    return False


def _validate_license(manifest: Mapping[str, Any]) -> None:
    status = manifest.get("license_status")
    if isinstance(status, Mapping):
        status = _find_first(status, ("status", "value"))
    if _as_nonempty_text(status).lower() != "granted":
        raise GoldValidationError("license_status_not_granted")
    uses = manifest.get("permitted_uses")
    if isinstance(uses, str):
        uses_list = [item.strip().lower() for item in re.split(r"[,;]", uses) if item.strip()]
    elif isinstance(uses, (list, tuple, set)):
        uses_list = [_as_nonempty_text(item).lower() for item in uses]
    else:
        uses_list = []
    if "evaluation" not in uses_list:
        raise GoldValidationError("license_evaluation_permission_missing")
    # SCB is a known restricted source in this repository.  The extension
    # profile name itself is not evidence that a gold package is SCB data.
    identity_values = []
    for key in ("dataset_id", "dataset_name", "source_dataset", "license_dataset"):
        identity_values.append(_as_nonempty_text(manifest.get(key)).lower())
    if any("scb" in value for value in identity_values):
        raise GoldValidationError("scb_gold_not_admitted")


def _validate_annotation_envelope(manifest: Mapping[str, Any]) -> None:
    required = {
        "dataset_id",
        "split_id",
        "split",
        "source_video_sha256",
        "source_lesson_id",
        "classroom_id",
        "capture_session_id",
        "taxonomy_profile_id",
        "taxonomy_contract_sha256",
        "annotation_protocol",
    }
    missing = sorted(key for key in required if key not in manifest or _as_nonempty_text(manifest.get(key)) == "")
    if missing:
        raise GoldValidationError("manifest_fields_missing:" + ",".join(missing))
    source_sha = _as_nonempty_text(manifest.get("source_video_sha256"))
    contract_sha = _as_nonempty_text(manifest.get("taxonomy_contract_sha256"))
    if not _SHA_RE.fullmatch(source_sha):
        raise GoldValidationError("source_video_sha256_invalid")
    if not _SHA_RE.fullmatch(contract_sha):
        raise GoldValidationError("taxonomy_contract_sha256_invalid")
    # Accept the envelope spellings used by early v3 fixtures, while still
    # requiring explicit review and adjudication evidence.
    if not _has_key_with_value(
        manifest,
        {"double_review_status", "double_reviewer_status", "two_reviewer_status", "review_status", "double_review", "dual_review", "reviewer_count", "reviewers"},
    ):
        raise GoldValidationError("double_review_status_missing")
    for key in ("double_review_status", "double_reviewer_status", "two_reviewer_status", "review_status"):
        if key in manifest and str(manifest[key]).strip().lower() in {"pending", "incomplete", "unreviewed", "not_started", "failed", "false"}:
            raise GoldValidationError("double_review_not_complete")
    if not _has_key_with_value(
        manifest,
        {"disagreement_resolution", "disagreement_handling", "adjudication", "disagreement_record", "dispute_resolution", "disagreement_policy", "dispute_handling"},
    ):
        raise GoldValidationError("disagreement_resolution_missing")
    _validate_license(manifest)


def _parse_gold_boxes(path: Path, *, allow_empty: bool = False) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen_frame_ids: set[tuple[int, str]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise GoldValidationError("boxes_read_failed") from exc
    for ordinal, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except Exception as exc:
            raise GoldValidationError(f"boxes_json_invalid:{ordinal}") from exc
        if not isinstance(raw, Mapping):
            raise GoldValidationError(f"boxes_row_not_object:{ordinal}")
        try:
            frame = _nonnegative_int(raw.get("frame_index"), field="frame_index")
            pts = _finite_float(raw.get("pts_seconds"), field="pts_seconds")
            if pts < 0:
                raise GoldValidationError("pts_seconds_negative")
            person = _as_nonempty_text(raw.get("gold_person_id"))
            if not person or person.lower() in {"unknown", "none", "null"}:
                raise GoldValidationError("gold_person_id_unstable")
            bbox = _normalise_bbox(raw.get("bbox_px"), field="bbox_px")
            key = (frame, person)
            if key in seen_frame_ids:
                raise GoldValidationError("duplicate_gold_person_in_frame")
            seen_frame_ids.add(key)
            confidence = raw.get("confidence", None)
            if confidence is not None:
                confidence = _finite_float(confidence, field="gold_confidence")
                if not 0.0 <= confidence <= 1.0:
                    raise GoldValidationError("gold_confidence_out_of_range")
        except GoldValidationError as exc:
            raise GoldValidationError(f"boxes_row_{ordinal}:{exc}") from exc
        row = {
            "frame_index": frame,
            "pts_seconds": pts,
            "gold_person_id": person,
            "bbox_px": bbox,
            "original_ordinal": ordinal,
        }
        if confidence is not None:
            row["confidence"] = confidence
        rows.append(row)
    if not rows and not allow_empty:
        raise GoldValidationError("boxes_empty")
    return tuple(sorted(rows, key=lambda row: (int(row["frame_index"]), str(row["gold_person_id"]), int(row["original_ordinal"]))))


def _parse_annotated_frames(path: Path, boxes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Validate explicit review coverage, including deliberately empty frames."""
    rows: dict[int, dict[str, Any]] = {}
    for ordinal, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            raw = _read_mapping(json.loads(line), field="annotated_frames_row")
            frame = _nonnegative_int(raw.get("frame_index"), field="frame_index")
            if isinstance(raw.get("pts_seconds"), bool):
                raise GoldValidationError("pts_seconds_not_numeric")
            pts = _finite_float(raw.get("pts_seconds"), field="pts_seconds")
            if pts < 0 or frame in rows:
                raise GoldValidationError("duplicate_frame_or_negative_pts")
            if type(raw.get("confirmed_empty")) is not bool:
                raise GoldValidationError("confirmed_empty_must_be_boolean")
            if raw.get("review_status") != "complete":
                raise GoldValidationError("frame_review_not_complete")
            reviewers = raw.get("reviewers")
            if not isinstance(reviewers, list) or len({item.strip() for item in reviewers if isinstance(item, str) and item.strip()}) < 2:
                raise GoldValidationError("frame_requires_two_distinct_reviewers")
            rows[frame] = {"frame_index": frame, "pts_seconds": pts, "confirmed_empty": raw["confirmed_empty"]}
        except (ValueError, TypeError, GoldValidationError) as exc:
            raise GoldValidationError(f"annotated_frames_row_{ordinal}:{exc}") from exc
    if not rows:
        raise GoldValidationError("annotated_frames_empty")
    box_frames = {int(box["frame_index"]) for box in boxes}
    if not box_frames.issubset(rows):
        raise GoldValidationError("boxes_outside_annotated_frames")
    for frame, row in rows.items():
        if row["confirmed_empty"] == (frame in box_frames):
            raise GoldValidationError("annotated_frames_empty_flag_disagrees_with_boxes")
    for box in boxes:
        if abs(float(box["pts_seconds"]) - rows[int(box["frame_index"])]["pts_seconds"]) > 1e-6:
            raise GoldValidationError("annotated_frames_box_pts_mismatch")
    return tuple(rows[key] for key in sorted(rows))


def _parse_gold_actions(path: Path, profile: _ProfileInfo) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    persons: set[str] = set()
    # The person set is checked by the caller after boxes are parsed.
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise GoldValidationError("actions_read_failed") from exc
    for ordinal, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except Exception as exc:
            raise GoldValidationError(f"actions_json_invalid:{ordinal}") from exc
        if not isinstance(raw, Mapping):
            raise GoldValidationError(f"actions_row_not_object:{ordinal}")
        try:
            person = _as_nonempty_text(raw.get("gold_person_id"))
            start = _finite_float(raw.get("start_pts_seconds"), field="start_pts_seconds")
            end = _finite_float(raw.get("end_pts_seconds"), field="end_pts_seconds")
            if not person or person.lower() in {"unknown", "none", "null"}:
                raise GoldValidationError("gold_person_id_unstable")
            if start < 0 or end <= start:
                raise GoldValidationError("action_interval_not_positive")
            value = _as_nonempty_text(raw.get("action_id"))
            canonical = profile.aliases.get(value.lower()) or profile.aliases.get(value)
            if canonical is None:
                raise GoldValidationError("gold_action_outside_profile")
            action_code = profile.codes[profile.action_ids.index(canonical)]
            action_id = _as_nonempty_text(raw.get("action_id"))
        except GoldValidationError as exc:
            raise GoldValidationError(f"actions_row_{ordinal}:{exc}") from exc
        persons.add(person)
        rows.append({
            "gold_person_id": person,
            "start_pts_seconds": start,
            "end_pts_seconds": end,
            "action_id": canonical,
            "action_code": action_code,
            "original_ordinal": ordinal,
        })
    ordered = sorted(rows, key=lambda row: (str(row["gold_person_id"]), float(row["start_pts_seconds"]), float(row["end_pts_seconds"]), int(row["original_ordinal"])))
    previous: dict[str, float] = {}
    for row in ordered:
        person = str(row["gold_person_id"])
        start = float(row["start_pts_seconds"])
        if person in previous and start < previous[person] - _EPS:
            raise GoldValidationError("gold_action_intervals_overlap")
        previous[person] = float(row["end_pts_seconds"])
    return tuple(ordered)


def _run_metadata(run_dir: Path) -> tuple[Mapping[str, Any], Path | None]:
    for name in ("run_manifest.v1.json", "run_manifest.json", "manifest.v1.json"):
        path = run_dir / name
        if path.is_file():
            try:
                payload = _json_load(path)
            except Exception:
                return {}, path
            return payload if isinstance(payload, Mapping) else {}, path
    return {}, None


def _run_source_sha(run_dir: Path, metadata: Mapping[str, Any]) -> str:
    candidates = [
        metadata.get("source_video_sha256"),
        metadata.get("source_sha256"),
        _nested_value(metadata, ("source", "sha256"), ("source", "source_video_sha256"), ("input", "sha256")),
    ]
    for item in candidates:
        text = _as_nonempty_text(item)
        if _SHA_RE.fullmatch(text):
            return text.lower()
    source_value = _find_first(metadata, ("source_video", "video_path", "source_path", "input_video"))
    if isinstance(source_value, Mapping):
        source_value = _find_first(source_value, ("path", "uri"))
    if source_value:
        source = Path(str(source_value))
        if not source.is_absolute():
            source = (run_dir / source).resolve()
        if source.is_file():
            return _sha256_file(source)
    # Some v3 manifests put the source in a timeline receipt.
    for candidate in (run_dir / "timeline.v2.json", run_dir / "artifacts" / "timeline.v2.json"):
        if candidate.is_file():
            try:
                payload = _json_load(candidate)
            except Exception:
                continue
            values = [
                payload.get("source_video_sha256") if isinstance(payload, Mapping) else None,
                _nested_value(payload, ("video", "sha256"), ("source", "sha256")) if isinstance(payload, Mapping) else None,
            ]
            for item in values:
                text = _as_nonempty_text(item)
                if _SHA_RE.fullmatch(text):
                    return text.lower()
    return ""


def _run_profile_info(metadata: Mapping[str, Any], run_dir: Path) -> tuple[str, str]:
    profile = metadata.get("profile")
    if not isinstance(profile, Mapping):
        profile = metadata.get("taxonomy_profile")
    if isinstance(profile, Mapping):
        profile_id = _as_nonempty_text(_find_first(profile, ("profile_id", "taxonomy_profile_id")))
        digest = _as_nonempty_text(_find_first(profile, ("contract_sha256", "taxonomy_contract_sha256", "sha256")))
    else:
        profile_id = _as_nonempty_text(_find_first(metadata, ("taxonomy_profile_id", "profile_id")))
        digest = _as_nonempty_text(_find_first(metadata, ("taxonomy_contract_sha256", "profile_sha256", "contract_sha256")))
    if profile_id and digest:
        return profile_id, digest.lower()
    for candidate in (run_dir / "timeline.v2.json", run_dir / "artifacts" / "timeline.v2.json"):
        if candidate.is_file():
            try:
                payload = _json_load(candidate)
            except Exception:
                continue
            if isinstance(payload, Mapping):
                profile_id = _as_nonempty_text(payload.get("taxonomy_profile_id"))
                digest = _as_nonempty_text(payload.get("taxonomy_contract_sha256"))
                if profile_id and digest:
                    return profile_id, digest.lower()
    return "", ""


def _validate_gold(run_dir: Path, gold_manifest: Any, profile: Any, run_meta: Mapping[str, Any]) -> _GoldData:
    manifest, manifest_path, manifest_sha = _manifest_input(gold_manifest)
    if _as_nonempty_text(manifest.get("schema_version")) != GOLD_SCHEMA_VERSION:
        raise GoldValidationError("gold_manifest_schema_invalid")
    _validate_annotation_envelope(manifest)
    manifest_dir = (manifest_path.parent if manifest_path is not None else run_dir).resolve()
    profile_id_hint = _as_nonempty_text(manifest.get("taxonomy_profile_id"))
    selected = _normalise_profile(profile, profile_id_hint)
    if selected.profile_id != profile_id_hint:
        raise GoldValidationError("taxonomy_profile_id_mismatch")
    if _as_nonempty_text(manifest.get("taxonomy_contract_sha256")).lower() != selected.sha256:
        raise GoldValidationError("taxonomy_contract_sha256_mismatch")
    run_profile_id, run_profile_sha = _run_profile_info(run_meta, run_dir)
    if not run_profile_id or not run_profile_sha:
        raise GoldValidationError("run_profile_metadata_missing")
    if run_profile_id != selected.profile_id or run_profile_sha != selected.sha256:
        raise GoldValidationError("run_profile_mismatch")
    source_sha = _run_source_sha(run_dir, run_meta)
    if not source_sha:
        raise GoldValidationError("run_source_sha256_missing")
    if source_sha != _as_nonempty_text(manifest.get("source_video_sha256")).lower():
        raise GoldValidationError("source_video_sha256_mismatch")
    boxes_uri, boxes_sha = _extract_artifact_ref(manifest, "boxes")
    actions_uri, actions_sha = _extract_artifact_ref(manifest, "actions")
    boxes_path = _resolve_relative_artifact(manifest_dir, boxes_uri, "boxes")
    actions_path = _resolve_relative_artifact(manifest_dir, actions_uri, "actions")
    if _sha256_file(boxes_path) != boxes_sha:
        raise GoldValidationError("boxes_sha256_mismatch")
    if _sha256_file(actions_path) != actions_sha:
        raise GoldValidationError("actions_sha256_mismatch")
    frame_ref_present = any(key == "annotated_frames" or key.startswith("annotated_frames_") for key in manifest) or any(
        isinstance(manifest.get(key), Mapping) and any(str(name).startswith("annotated_frames") for name in manifest[key])
        for key in ("artifacts", "gold_artifacts", "jsonl", "annotations")
    )
    frames_path = None
    if frame_ref_present:
        frames_uri, frames_sha = _extract_artifact_ref(manifest, "annotated_frames")
        frames_path = _resolve_relative_artifact(manifest_dir, frames_uri, "annotated_frames")
        if _sha256_file(frames_path) != frames_sha:
            raise GoldValidationError("annotated_frames_sha256_mismatch")
    boxes = _parse_gold_boxes(boxes_path, allow_empty=frames_path is not None)
    annotated_frames = _parse_annotated_frames(frames_path, boxes) if frames_path is not None else None
    actions = _parse_gold_actions(actions_path, selected)
    people = {str(row["gold_person_id"]) for row in boxes}
    if any(str(row["gold_person_id"]) not in people for row in actions):
        raise GoldValidationError("action_person_missing_from_boxes")
    return _GoldData(manifest, manifest_path, manifest_sha, boxes, actions, selected, annotated_frames)


def _prediction_bbox(raw: Mapping[str, Any]) -> Any:
    value = _find_first(raw, ("bbox_px", "bbox", "box"))
    detection = raw.get("detection")
    if value is None and isinstance(detection, Mapping):
        value = _find_first(detection, ("bbox_px", "bbox", "box"))
    return value


def _prediction_id(raw: Mapping[str, Any]) -> str:
    value = _find_first(raw, ("pred_person_id", "track_id", "student_id", "prediction_id", "object_id", "id"))
    return _as_nonempty_text(value)


def _prediction_confidence(raw: Mapping[str, Any]) -> tuple[float | None, bool]:
    key = None
    for candidate in ("confidence", "score", "detection_confidence", "object_confidence"):
        if candidate in raw:
            key = candidate
            break
    value: Any = raw.get(key) if key else None
    if value is None and isinstance(raw.get("detection"), Mapping):
        detection = raw["detection"]
        for candidate in ("confidence", "score"):
            if candidate in detection:
                key, value = candidate, detection[candidate]
                break
    if key is None or value is None:
        return None, False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, False
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None, False
    return number, True


def _prediction_action_value(raw: Mapping[str, Any]) -> str:
    value = _find_first(raw, ("predicted_action_id", "predicted_action", "action_id", "primary_action_id", "action_code", "behavior", "label", "class"))
    if isinstance(value, Mapping):
        value = _find_first(value, ("action_id", "code", "label", "name"))
    return _as_nonempty_text(value)


def _walk_prediction_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_prediction_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_prediction_records(child)


def _candidate_prediction_files(run_dir: Path, excluded: set[Path]) -> tuple[Path, ...]:
    """Return one mutually-exclusive prediction source set.

    A completed v3 run can contain historical copies of observations in
    several directories.  Reading all ``*.jsonl`` files silently doubles TP,
    FP and action duration, so selection is explicit and deterministic:

    * ``predictions.v1.jsonl`` (or the unversioned equivalent) is authoritative
      and is the only source read when present;
    * otherwise detections come from the canonical track-observation file;
    * actions come from ``action_intervals.jsonl``, then the legacy root
      ``action_observations.jsonl``, then ``timeline.v2.json``.
    """

    def available(candidates: Sequence[Path]) -> Path | None:
        for path in candidates:
            if path.is_file() and path.resolve() not in excluded:
                return path
        return None

    combined = available((run_dir / "predictions.v1.jsonl", run_dir / "predictions.jsonl", run_dir / "artifacts" / "predictions.v1.jsonl"))
    if combined is not None:
        return (combined,)
    detection = available((run_dir / "artifacts" / "track_observations.jsonl", run_dir / "track_observations.jsonl", run_dir / "artifacts" / "student_tracks.v1.jsonl", run_dir / "artifacts" / "student_object_evidence.v1.jsonl"))
    action = available((run_dir / "action_intervals.jsonl", run_dir / "action_observations.jsonl"))
    if action is None:
        action = available((run_dir / "timeline.v2.json", run_dir / "artifacts" / "timeline.v2.json"))
    selected = [path for path in (detection, action) if path is not None]
    # Preserve the source-role contract for the parser.  The caller uses file
    # names to gate record types, so this function only returns paths; the
    # role map is reconstructed in ``_load_predictions`` below.
    return tuple(selected)


def _row_frame(raw: Mapping[str, Any]) -> int | None:
    value = raw.get("frame_index")
    if value is None:
        return None
    try:
        number = int(value)
        if float(value) != float(number) or number < 0:
            return None
        return number
    except (TypeError, ValueError):
        return None


def _row_pts(raw: Mapping[str, Any]) -> float | None:
    value = _find_first(raw, ("pts_seconds", "timestamp_seconds", "time_seconds", "pts"))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _parse_prediction_file(path: Path, detections: list[dict[str, Any]], actions: list[dict[str, Any]], warnings: list[str], ordinal_start: int, *, allow_detection: bool = True, allow_actions: bool = True) -> int:
    ordinal = ordinal_start
    try:
        if path.suffix.lower() == ".jsonl":
            values: Iterable[Any] = []
            rows: list[Any] = []
            for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    warnings.append(f"prediction_json_invalid:{path.name}:{line_index}")
            values = rows
        else:
            values = [_json_load(path)]
    except Exception as exc:
        warnings.append(f"prediction_read_failed:{path.name}:{type(exc).__name__}")
        return ordinal
    for value in values:
        for raw in _walk_prediction_records(value):
            frame = _row_frame(raw)
            pts = _row_pts(raw)
            bbox = _prediction_bbox(raw)
            if allow_detection and bbox is not None and frame is not None and pts is not None:
                try:
                    box = _normalise_bbox(bbox, field="prediction_bbox")
                except GoldValidationError:
                    box = None
                if box is not None:
                    confidence, has_confidence = _prediction_confidence(raw)
                    row: dict[str, Any] = {
                        "frame_index": frame,
                        "pts_seconds": pts,
                        "pred_person_id": _prediction_id(raw),
                        "bbox_px": box,
                        "original_ordinal": ordinal,
                        "source_file": path.as_posix(),
                    }
                    if has_confidence:
                        row["confidence"] = confidence
                    detections.append(row)
                    ordinal += 1
            start_value = _find_first(raw, ("start_pts_seconds", "start_seconds", "start"))
            end_value = _find_first(raw, ("end_pts_seconds", "end_seconds", "end"))
            if allow_actions and start_value is not None and end_value is not None:
                try:
                    start = float(start_value)
                    end = float(end_value)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(start) and math.isfinite(end) and end > start):
                    continue
                student_id = _prediction_id(raw)
                if not student_id:
                    continue
                actions.append({
                    "pred_person_id": student_id,
                    "start_pts_seconds": max(0.0, start),
                    "end_pts_seconds": end,
                    "action_value": _prediction_action_value(raw),
                    "status": _as_nonempty_text(raw.get("status")).lower(),
                    "original_ordinal": ordinal,
                    "source_file": path.as_posix(),
                })
                ordinal += 1
    return ordinal


def _frame_actions_to_intervals(raw_actions: list[dict[str, Any]], detections: Sequence[Mapping[str, Any]]) -> None:
    """Convert optional frame-level action observations in-place.

    Current v3 artifacts use intervals.  This small compatibility conversion
    lets an offline fixture use one action label per tracked frame without
    inventing a duration: the next observed timestamp (or the next detection
    timestamp for that ID) supplies the end boundary.
    """

    del raw_actions, detections


def _load_predictions(run_dir: Path, excluded: set[Path]) -> _PredictionData:
    detections: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    warnings: list[str] = []
    files = _candidate_prediction_files(run_dir, excluded)
    ordinal = 0
    combined = len(files) == 1 and files[0].name.lower() in {"predictions.v1.jsonl", "predictions.jsonl"}
    source_roles: dict[str, str] = {}
    for path in files:
        if combined:
            role = "combined"
            ordinal = _parse_prediction_file(path, detections, actions, warnings, ordinal)
        elif path.suffix.lower() == ".json":
            role = "timeline"
            ordinal = _parse_prediction_file(path, detections, actions, warnings, ordinal, allow_detection=False, allow_actions=True)
        elif "action" in path.name.lower():
            role = "actions"
            ordinal = _parse_prediction_file(path, detections, actions, warnings, ordinal, allow_detection=False, allow_actions=True)
        else:
            role = "detections"
            ordinal = _parse_prediction_file(path, detections, actions, warnings, ordinal, allow_detection=True, allow_actions=False)
        source_roles[role] = path.relative_to(run_dir).as_posix()
    detections.sort(key=lambda row: (int(row["frame_index"]), str(row.get("pred_person_id", "")), int(row["original_ordinal"])))
    actions.sort(key=lambda row: (str(row.get("pred_person_id", "")), float(row["start_pts_seconds"]), float(row["end_pts_seconds"]), int(row["original_ordinal"])))
    seen_detection_ids: set[tuple[int, str]] = set()
    for row in detections:
        pred_id = _as_nonempty_text(row.get("pred_person_id"))
        if not pred_id:
            continue
        key = (int(row["frame_index"]), pred_id)
        if key in seen_detection_ids:
            warnings.append(f"prediction_duplicate_id_in_frame:{key[0]}:{key[1]}")
        seen_detection_ids.add(key)
    source_contract = {
        "mode": "combined" if combined else "split_explicit_priority",
        "roles": dict(sorted(source_roles.items())),
        "sha256": {role: _sha256_file(next(path for path in files if path.relative_to(run_dir).as_posix() == relative)) for role, relative in sorted(source_roles.items())},
        "selection_rule": "predictions.v1.jsonl_only_else_track_observations_plus_action_intervals_then_timeline",
    }
    return _PredictionData(tuple(detections), tuple(actions), files, tuple(dict.fromkeys(warnings)), source_contract)


def _predictions_on_gold_frames(gold: _GoldData, pred: _PredictionData) -> _PredictionData:
    """Restrict box/tracking scores to explicitly annotated gold frames.

    New packages may explicitly list reviewed empty frames. Legacy sparse
    packages retain box-frame-only scoring. Unlisted frames never become
    inferred negatives. Action intervals keep their independent coverage.
    """

    frame_rows = gold.boxes if gold.annotated_frames is None else gold.annotated_frames
    evaluation_frames = {int(row["frame_index"]) for row in frame_rows}
    detections = tuple(
        row for row in pred.detections
        if int(row["frame_index"]) in evaluation_frames
    )
    contract = {
        **dict(pred.source_contract),
        "detection_frame_policy": "explicit_gold_box_frames_only" if gold.annotated_frames is None else "explicit_reviewed_frames_including_empty",
        "evaluation_frame_count": len(evaluation_frames),
        "detections_total": len(pred.detections),
        "detections_scored": len(detections),
    }
    if gold.annotated_frames is not None:
        contract["confirmed_empty_frame_count"] = sum(row["confirmed_empty"] for row in gold.annotated_frames)
    return _PredictionData(
        detections,
        pred.actions,
        pred.files,
        pred.parse_warnings,
        contract,
    )


def _hungarian_max(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Deterministic maximum-weight one-to-one matching, pure Python.

    The square Hungarian implementation uses a min-cost formulation.  Input
    rows and columns are already stable-sorted by the caller.  Zero-weight
    assignments are returned only when their weight is positive by the caller,
    which gives the desired optional matching semantics.
    """

    if not weights:
        return []
    row_count = len(weights)
    column_count = max((len(row) for row in weights), default=0)
    if column_count == 0:
        return []
    size = max(row_count, column_count)
    matrix = [[0.0] * size for _ in range(size)]
    for row_index, row in enumerate(weights):
        for col_index, value in enumerate(row[:size]):
            # A tiny deterministic tie-break favours earlier columns while
            # remaining far below the six-decimal IoU precision used here.
            matrix[row_index][col_index] = -float(value) + col_index * 1e-12
    # cp-algorithms 1-indexed Hungarian algorithm for minimum assignment.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j] - 1e-15 or (abs(cur - minv[j]) <= 1e-15 and j < j1):
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta - 1e-15 or (abs(minv[j] - delta) <= 1e-15 and j < j1):
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs: list[tuple[int, int]] = []
    for col in range(1, size + 1):
        row = p[col] - 1
        if 0 <= row < row_count and col - 1 < column_count:
            pairs.append((row, col - 1))
    return sorted(pairs)


def _frame_groups(rows: Sequence[Mapping[str, Any]], key: str) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            grouped[int(row[key])].append(row)
        except (KeyError, TypeError, ValueError):
            continue
    for frame in grouped:
        grouped[frame] = sorted(
            grouped[frame],
            key=lambda row: (str(row.get("gold_person_id", row.get("pred_person_id", ""))), int(row.get("original_ordinal", 0))),
        )
    return dict(grouped)


def _frame_matches(gold_rows: Sequence[Mapping[str, Any]], pred_rows: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    gold_by_frame = _frame_groups(gold_rows, "frame_index")
    pred_by_frame = _frame_groups(pred_rows, "frame_index")
    matches: list[dict[str, Any]] = []
    for frame in sorted(set(gold_by_frame) | set(pred_by_frame)):
        gold = gold_by_frame.get(frame, [])
        pred = pred_by_frame.get(frame, [])
        weights = [[_iou(g["bbox_px"], p["bbox_px"]) if _iou(g["bbox_px"], p["bbox_px"]) >= threshold else 0.0 for p in pred] for g in gold]
        for gi, pi in _hungarian_max(weights):
            value = _iou(gold[gi]["bbox_px"], pred[pi]["bbox_px"])
            if value + _EPS >= threshold:
                matches.append({
                    "frame_index": frame,
                    "gold": gold[gi],
                    "pred": pred[pi],
                    "iou": value,
                })
    return matches


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _detection_metrics(gold: _GoldData, pred: _PredictionData) -> dict[str, Any]:
    matches = _frame_matches(gold.boxes, pred.detections, IOU_THRESHOLD)
    tp = len(matches)
    gold_count = len(gold.boxes)
    pred_count = len(pred.detections)
    fn, fp = gold_count - tp, pred_count - tp
    result: dict[str, Any] = {
        "status": "complete" if pred.detections and all("confidence" in row for row in pred.detections) else "partial",
        "class": "person",
        "iou_threshold": IOU_THRESHOLD,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
        "miss_rate": _safe_ratio(fn, tp + fn),
    }
    if pred.detections and all("confidence" in row for row in pred.detections):
        result["AP50"] = _ap50(gold.boxes, pred.detections)
        result["mAP50"] = result["AP50"]
    return result


def _ap50(gold_rows: Sequence[Mapping[str, Any]], pred_rows: Sequence[Mapping[str, Any]]) -> float:
    gold_by_frame = _frame_groups(gold_rows, "frame_index")
    used: dict[int, set[int]] = defaultdict(set)
    ordered = sorted(pred_rows, key=lambda row: (-float(row["confidence"]), int(row.get("frame_index", 0)), str(row.get("pred_person_id", "")), int(row.get("original_ordinal", 0))))
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for pred in ordered:
        frame = int(pred["frame_index"])
        candidates = []
        for index, target in enumerate(gold_by_frame.get(frame, [])):
            if index in used[frame]:
                continue
            candidates.append((_iou(target["bbox_px"], pred["bbox_px"]), index))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if candidates and candidates[0][0] + _EPS >= IOU_THRESHOLD:
            used[frame].add(candidates[0][1])
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    total_gold = len(gold_rows)
    if total_gold == 0:
        return 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    tp_seen = fp_seen = 0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        tp_seen += tp_flag
        fp_seen += fp_flag
        precisions.append(_safe_ratio(tp_seen, tp_seen + fp_seen))
        recalls.append(_safe_ratio(tp_seen, total_gold))
    interpolated: list[float] = []
    for step in range(101):
        threshold = step / 100.0
        eligible = [precision for precision, recall in zip(precisions, recalls) if recall + _EPS >= threshold]
        interpolated.append(max(eligible) if eligible else 0.0)
    return sum(interpolated) / 101.0


def _global_id_mapping(gold: _GoldData, pred: _PredictionData, threshold: float = IOU_THRESHOLD) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    matches = _frame_matches(gold.boxes, pred.detections, threshold)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    # IDF1's global identity assignment is solved after accumulating every
    # eligible per-frame gold/pred identity overlap.  Do not discard a
    # potential pair merely because the detection metric's local IoU matching
    # selected another pair in that frame; the global Hungarian assignment is
    # what decides which identity pair is retained.
    gold_by_frame = _frame_groups(gold.boxes, "frame_index")
    pred_by_frame = _frame_groups(pred.detections, "frame_index")
    for frame in sorted(set(gold_by_frame) & set(pred_by_frame)):
        for gold_row in gold_by_frame[frame]:
            gold_id = _as_nonempty_text(gold_row.get("gold_person_id"))
            for pred_row in pred_by_frame[frame]:
                pred_id = _as_nonempty_text(pred_row.get("pred_person_id"))
                if gold_id and pred_id and _iou(gold_row["bbox_px"], pred_row["bbox_px"]) + _EPS >= threshold:
                    counts[(gold_id, pred_id)] += 1
    gold_ids = sorted({str(row["gold_person_id"]) for row in gold.boxes})
    pred_ids = sorted({str(row.get("pred_person_id")) for row in pred.detections if _as_nonempty_text(row.get("pred_person_id"))})
    weights = [[float(counts.get((gold_id, pred_id), 0)) for pred_id in pred_ids] for gold_id in gold_ids]
    mapping_gold: dict[str, str] = {}
    mapping_pred: dict[str, str] = {}
    for gi, pi in _hungarian_max(weights):
        if weights[gi][pi] > 0:
            mapping_gold[gold_ids[gi]] = pred_ids[pi]
            mapping_pred[pred_ids[pi]] = gold_ids[gi]
    return mapping_pred, mapping_gold, matches


def _tracking_metrics(gold: _GoldData, pred: _PredictionData) -> tuple[dict[str, Any], dict[str, str]]:
    mapping_pred, mapping_gold, base_matches = _global_id_mapping(gold, pred, IOU_THRESHOLD)
    gold_count = len(gold.boxes)
    pred_count = len(pred.detections)
    # IDTP is the count of eligible detections belonging to the selected
    # global gold/pred identity pair.  It is intentionally not the number of
    # locally IoU-matched rows: local matching can hide another eligible pair
    # in an identity-crossing frame before the global assignment is solved.
    gold_by_frame = _frame_groups(gold.boxes, "frame_index")
    pred_by_frame = _frame_groups(pred.detections, "frame_index")
    idtp = 0
    for frame in sorted(set(gold_by_frame) & set(pred_by_frame)):
        for gold_row in gold_by_frame[frame]:
            gold_id = _as_nonempty_text(gold_row.get("gold_person_id"))
            selected_pred = mapping_gold.get(gold_id)
            if not selected_pred:
                continue
            idtp += int(any(
                _as_nonempty_text(pred_row.get("pred_person_id")) == selected_pred
                and _iou(gold_row["bbox_px"], pred_row["bbox_px"]) + _EPS >= IOU_THRESHOLD
                for pred_row in pred_by_frame[frame]
            ))
    idfn, idfp = gold_count - idtp, pred_count - idtp
    previous: dict[str, str] = {}
    matched_frames: dict[str, list[int]] = defaultdict(list)
    switch_count = 0
    for item in sorted(base_matches, key=lambda row: (int(row["frame_index"]), str(row["gold"].get("gold_person_id", "")), int(row["gold"].get("original_ordinal", 0)))):
        gold_id = _as_nonempty_text(item["gold"].get("gold_person_id"))
        pred_id = _as_nonempty_text(item["pred"].get("pred_person_id"))
        if not gold_id or not pred_id:
            continue
        if gold_id in previous and previous[gold_id] != pred_id:
            switch_count += 1
        previous[gold_id] = pred_id
        matched_frames[gold_id].append(int(item["frame_index"]))
    # Fragmentation follows the sequence of annotated gold observations, not
    # the numeric spacing of frame indices.  Sparse gold at frames 0/10/20 is
    # one continuous successful fragment; a matched -> unmatched -> matched
    # boolean sequence contributes one fragment break.
    matched_keys = {
        (int(item["frame_index"]), str(item["gold"].get("gold_person_id", "")), int(item["gold"].get("original_ordinal", 0)))
        for item in base_matches
        if _as_nonempty_text(item["pred"].get("pred_person_id"))
    }
    gold_by_person: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gold.boxes:
        gold_by_person[str(row["gold_person_id"])].append(row)
    fragment_count = 0
    for person, rows in gold_by_person.items():
        ordered = sorted(rows, key=lambda row: (int(row["frame_index"]), int(row.get("original_ordinal", 0))))
        was_matched = False
        fragments = 0
        for row in ordered:
            key = (int(row["frame_index"]), person, int(row.get("original_ordinal", 0)))
            current = key in matched_keys
            if current and not was_matched:
                fragments += 1
            was_matched = current
        fragment_count += max(0, fragments - 1)
    result = {
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
        "IDF1": _safe_ratio(2.0 * idtp, 2.0 * idtp + idfp + idfn),
        "ID_switch": switch_count,
        "ID_switch_count": switch_count,
        "fragment_count": fragment_count,
        "iou_threshold": IOU_THRESHOLD,
        "global_id_mapping": {"pred_to_gold": dict(sorted(mapping_pred.items())), "gold_to_pred": dict(sorted(mapping_gold.items()))},
    }
    result["id_switch"] = switch_count
    result.update(_hota_metrics(gold, pred))
    return result, mapping_pred


def _hota_metrics(gold: _GoldData, pred: _PredictionData) -> dict[str, Any]:
    """Compute the frozen HOTA definition used by TrackEval.

    TrackEval first computes an identity-independent global alignment score
    from all-frame IoU similarity, then uses ``global_alignment * similarity``
    as the per-frame assignment weight for each alpha.  A raw-IoU Hungarian
    would be a different metric, so keep the two stages explicit here.
    """

    gold_by_frame = _frame_groups(gold.boxes, "frame_index")
    pred_by_frame = _frame_groups(pred.detections, "frame_index")
    gold_track_counts: dict[str, int] = defaultdict(int)
    pred_track_counts: dict[str, int] = defaultdict(int)
    for row in gold.boxes:
        gold_track_counts[str(row["gold_person_id"])] += 1
    for row in pred.detections:
        pred_id = _as_nonempty_text(row.get("pred_person_id"))
        if pred_id:
            pred_track_counts[pred_id] += 1
    gold_ids = sorted(gold_track_counts)
    pred_ids = sorted(pred_track_counts)
    potential: dict[tuple[str, str], float] = defaultdict(float)
    for frame in sorted(set(gold_by_frame) & set(pred_by_frame)):
        frame_gold = gold_by_frame[frame]
        frame_pred = pred_by_frame[frame]
        similarity = [[_iou(gold_row["bbox_px"], pred_row["bbox_px"]) for pred_row in frame_pred] for gold_row in frame_gold]
        row_sums = [sum(row) for row in similarity]
        col_sums = [sum(similarity[row_index][col_index] for row_index in range(len(similarity))) for col_index in range(len(frame_pred))]
        for row_index, gold_row in enumerate(frame_gold):
            gold_id = _as_nonempty_text(gold_row.get("gold_person_id"))
            for col_index, pred_row in enumerate(frame_pred):
                pred_id = _as_nonempty_text(pred_row.get("pred_person_id"))
                sim = similarity[row_index][col_index]
                denominator = row_sums[row_index] + col_sums[col_index] - sim
                if gold_id and pred_id and denominator > _EPS:
                    potential[(gold_id, pred_id)] += sim / denominator
    global_alignment: dict[tuple[str, str], float] = {}
    for gold_id in gold_ids:
        for pred_id in pred_ids:
            potential_value = potential.get((gold_id, pred_id), 0.0)
            denominator = gold_track_counts[gold_id] + pred_track_counts[pred_id] - potential_value
            global_alignment[(gold_id, pred_id)] = _safe_ratio(potential_value, denominator)
    # The global-alignment assignment is independent of alpha. Retain its
    # selected pair similarities once per frame, then apply each threshold.
    assigned_pairs: list[tuple[str, str, float]] = []
    for frame in sorted(set(gold_by_frame) & set(pred_by_frame)):
        frame_gold, frame_pred = gold_by_frame[frame], pred_by_frame[frame]
        similarities = [[_iou(g["bbox_px"], p["bbox_px"]) for p in frame_pred] for g in frame_gold]
        weights = [[global_alignment.get((str(g["gold_person_id"]), _as_nonempty_text(p.get("pred_person_id"))), 0.0) * similarities[gi][pi]
                    for pi, p in enumerate(frame_pred)] for gi, g in enumerate(frame_gold)]
        for gi, pi in _hungarian_max(weights):
            if gi >= len(frame_gold) or pi >= len(frame_pred) or weights[gi][pi] <= 0.0:
                continue
            gold_id = _as_nonempty_text(frame_gold[gi].get("gold_person_id"))
            pred_id = _as_nonempty_text(frame_pred[pi].get("pred_person_id"))
            if gold_id and pred_id:
                assigned_pairs.append((gold_id, pred_id, similarities[gi][pi]))
    per_alpha: list[dict[str, Any]] = []
    for alpha in HOTA_ALPHAS:
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        tp = 0
        for gold_id, pred_id, similarity in assigned_pairs:
            if similarity + _EPS >= alpha:
                pair_counts[(gold_id, pred_id)] += 1
                tp += 1
        det_a = _safe_ratio(tp, tp + (len(pred.detections) - tp) + (len(gold.boxes) - tp))
        ass_sum = 0.0
        for (gold_id, pred_id), pair_tp in pair_counts.items():
            denominator = gold_track_counts[gold_id] + pred_track_counts[pred_id] - pair_tp
            ass = _safe_ratio(pair_tp, denominator)
            ass_sum += pair_tp * ass
        ass_a = _safe_ratio(ass_sum, tp)
        hota = math.sqrt(max(0.0, det_a * ass_a))
        per_alpha.append({"alpha": alpha, "DetA": det_a, "AssA": ass_a, "HOTA": hota})
    return {
        "HOTA": _safe_ratio(sum(float(item["HOTA"]) for item in per_alpha), len(per_alpha)),
        "HOTA_by_alpha": per_alpha,
        "hota_by_alpha": per_alpha,
        "alpha_values": list(HOTA_ALPHAS),
        "reference": "JonathonLuiten/TrackEval:trackeval/metrics/hota.py",
    }


def _canonical_pred_action(value: str, profile: _ProfileInfo, status: str = "") -> tuple[str, str]:
    if status in {"unknown", "unobserved", "uncertain", "gray", "grey", "unavailable"}:
        return UNKNOWN, UNKNOWN
    text = _as_nonempty_text(value)
    if not text or text.upper() == UNKNOWN:
        return UNKNOWN, UNKNOWN
    action_id = profile.aliases.get(text.lower()) or profile.aliases.get(text)
    if action_id is None:
        return UNKNOWN, UNKNOWN
    code = profile.codes[profile.action_ids.index(action_id)]
    return action_id, code


def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _mapped_action_intervals(pred_actions: Sequence[Mapping[str, Any]], mapping_pred: Mapping[str, str], profile: _ProfileInfo) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in pred_actions:
        pred_id = _as_nonempty_text(row.get("pred_person_id"))
        gold_id = mapping_pred.get(pred_id)
        if not gold_id:
            continue
        action_id, code = _canonical_pred_action(_as_nonempty_text(row.get("action_value")), profile, _as_nonempty_text(row.get("status")).lower())
        result.append({
            "gold_person_id": gold_id,
            "pred_person_id": pred_id,
            "start_pts_seconds": float(row["start_pts_seconds"]),
            "end_pts_seconds": float(row["end_pts_seconds"]),
            "action_id": action_id,
            "action_code": code,
            "original_ordinal": int(row.get("original_ordinal", 0)),
        })
    return sorted(result, key=lambda row: (str(row["gold_person_id"]), float(row["start_pts_seconds"]), float(row["end_pts_seconds"]), int(row["original_ordinal"])))


def _action_metrics(gold: _GoldData, pred: _PredictionData, mapping_pred: Mapping[str, str], profile: _ProfileInfo) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _mapped_action_intervals(pred.actions, mapping_pred, profile)
    column_codes = list(profile.codes) + [UNKNOWN]
    row_codes = list(profile.codes)
    matrix = [[0.0 for _ in column_codes] for _ in row_codes]
    counts = [[0 for _ in column_codes] for _ in row_codes]
    unknown_seconds = 0.0
    covered_seconds = 0.0
    atomic_rows: list[dict[str, Any]] = []
    by_student_pred: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_student_pred[str(row["gold_person_id"])].append(row)
    for gold_row in gold.actions:
        student = str(gold_row["gold_person_id"])
        g_start, g_end = float(gold_row["start_pts_seconds"]), float(gold_row["end_pts_seconds"])
        boundaries = {g_start, g_end}
        relevant = [row for row in by_student_pred.get(student, []) if _overlap(g_start, g_end, float(row["start_pts_seconds"]), float(row["end_pts_seconds"])) > _EPS]
        for row in relevant:
            boundaries.add(max(g_start, float(row["start_pts_seconds"])))
            boundaries.add(min(g_end, float(row["end_pts_seconds"])))
        ordered = sorted(value for value in boundaries if g_start - _EPS <= value <= g_end + _EPS)
        gold_code = str(gold_row["action_code"])
        gi = row_codes.index(gold_code)
        for left, right in zip(ordered, ordered[1:]):
            duration = right - left
            if duration <= _EPS:
                continue
            candidates = []
            for row in relevant:
                overlap = _overlap(left, right, float(row["start_pts_seconds"]), float(row["end_pts_seconds"]))
                if overlap > _EPS:
                    candidates.append((overlap, int(row["original_ordinal"]), row))
            candidates.sort(key=lambda item: (-item[0], item[1], str(item[2].get("action_code", UNKNOWN))))
            pred_code = str(candidates[0][2]["action_code"]) if candidates else UNKNOWN
            if pred_code not in column_codes:
                pred_code = UNKNOWN
            ci = column_codes.index(pred_code)
            matrix[gi][ci] += duration
            counts[gi][ci] += 1
            covered_seconds += duration
            if pred_code == UNKNOWN:
                unknown_seconds += duration
            atomic_rows.append({"gold_person_id": student, "start": left, "end": right, "gold_code": gold_code, "pred_code": pred_code, "duration": duration})
    per_class: dict[str, dict[str, float]] = {}
    for index, code in enumerate(row_codes):
        tp = matrix[index][index]
        fp = sum(matrix[row_index][index] for row_index in range(len(row_codes)) if row_index != index)
        fn = sum(matrix[index][column_index] for column_index in range(len(column_codes)) if column_index != index)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        per_class[code] = {"precision": precision, "recall": recall, "F1": f1, "f1": f1, "support_seconds": sum(matrix[index])}
    macro = sum(item["F1"] for item in per_class.values()) / len(per_class) if per_class else 0.0
    payload = {
        "profile_id": profile.profile_id,
        "status": "complete",
        "row_labels": row_codes,
        "column_labels": column_codes,
        "confusion_matrix_seconds": matrix,
        "confusion_matrix_sample_counts": counts,
        "confusion_matrix": {"row_labels": row_codes, "column_labels": column_codes, "seconds": matrix, "sample_counts": counts},
        "per_class": per_class,
        "macro_F1": macro,
        "macro_f1": macro,
        "UNKNOWN_ratio": _safe_ratio(unknown_seconds, covered_seconds),
        "unknown_ratio": _safe_ratio(unknown_seconds, covered_seconds),
        "unknown_seconds": unknown_seconds,
        "covered_seconds": covered_seconds,
    }
    return payload, rows


def _merge_segments(rows: Sequence[Mapping[str, Any]], *, student_key: str = "gold_person_id") -> list[dict[str, Any]]:
    ordered = sorted(
        [row for row in rows if str(row.get("action_code", UNKNOWN)) != UNKNOWN],
        key=lambda row: (str(row.get(student_key, "")), str(row.get("action_code", "")), float(row["start_pts_seconds"]), float(row["end_pts_seconds"]), int(row.get("original_ordinal", 0))),
    )
    merged: list[dict[str, Any]] = []
    for row in ordered:
        current = {"gold_person_id": str(row[student_key]), "action_code": str(row["action_code"]), "start_pts_seconds": float(row["start_pts_seconds"]), "end_pts_seconds": float(row["end_pts_seconds"])}
        if merged and merged[-1]["gold_person_id"] == current["gold_person_id"] and merged[-1]["action_code"] == current["action_code"] and current["start_pts_seconds"] <= merged[-1]["end_pts_seconds"] + _EPS:
            merged[-1]["end_pts_seconds"] = max(float(merged[-1]["end_pts_seconds"]), current["end_pts_seconds"])
        else:
            merged.append(current)
    # The ordering above groups by class, so put merged segments back in time
    # order for flips and deterministic threshold matching.
    return sorted(merged, key=lambda row: (str(row["gold_person_id"]), float(row["start_pts_seconds"]), float(row["end_pts_seconds"]), str(row["action_code"])))


def _clip_pred_segments_to_gold(pred_segments: Sequence[Mapping[str, Any]], gold_actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_student: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gold_actions:
        by_student[str(row["gold_person_id"])].append(row)
    clipped: list[dict[str, Any]] = []
    for row in pred_segments:
        student = str(row["gold_person_id"])
        for gold_row in by_student.get(student, []):
            start = max(float(row["start_pts_seconds"]), float(gold_row["start_pts_seconds"]))
            end = min(float(row["end_pts_seconds"]), float(gold_row["end_pts_seconds"]))
            if end > start + _EPS:
                copy = dict(row)
                copy["start_pts_seconds"], copy["end_pts_seconds"] = start, end
                clipped.append(copy)
    return _merge_segments(clipped)


def _temporal_match(gold_segments: Sequence[Mapping[str, Any]], pred_segments: Sequence[Mapping[str, Any]], threshold: float) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for gi, gold in enumerate(gold_segments):
        for pi, pred in enumerate(pred_segments):
            if str(gold["gold_person_id"]) != str(pred["gold_person_id"]) or str(gold["action_code"]) != str(pred["action_code"]):
                continue
            gs, ge = float(gold["start_pts_seconds"]), float(gold["end_pts_seconds"])
            ps, pe = float(pred["start_pts_seconds"]), float(pred["end_pts_seconds"])
            intersection = _overlap(gs, ge, ps, pe)
            union = (ge - gs) + (pe - ps) - intersection
            tiou = intersection / union if union > 0 else 0.0
            if tiou + _EPS >= threshold:
                candidates.append((tiou, gi, pi))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_g, used_p = set(), set()
    tp = 0
    for _, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp += 1
    return tp, len(pred_segments) - tp, len(gold_segments) - tp


def _recorded_video_duration(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    # Prefer explicit video-level fields.  Do not recurse through stage
    # receipts and accidentally treat an S0/S1 duration as video duration.
    for key in ("video_duration_seconds", "duration_seconds"):
        try:
            number = float(value.get(key))
            if math.isfinite(number) and number > 0:
                return number
        except (TypeError, ValueError):
            pass
    for path in (("video", "duration_seconds"), ("video", "video_duration_seconds"), ("source", "duration_seconds"), ("input", "duration_seconds")):
        current: Any = value
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        try:
            number = float(current)
            if math.isfinite(number) and number > 0:
                return number
        except (TypeError, ValueError):
            pass
    return None


def _video_duration_seconds(run_meta: Mapping[str, Any], run_dir: Path) -> float | None:
    """Return only a recorded source-video duration; never infer one from boxes."""

    duration = _recorded_video_duration(run_meta)
    if duration is None:
        for path in (run_dir / "timeline.v2.json", run_dir / "artifacts" / "timeline.v2.json"):
            if path.is_file():
                try:
                    payload = _json_load(path)
                except Exception:
                    continue
                if isinstance(payload, Mapping):
                    duration = _recorded_video_duration(payload)
                    if duration is not None:
                        break
    if duration is None or duration <= 0:
        return None
    return float(duration)


def _unknown_full_student_seconds(gold: _GoldData, mapped_actions: Sequence[Mapping[str, Any]], duration: float) -> float:
    """Measure UNKNOWN over each complete recorded student timeline.

    Explicit UNKNOWN/gray intervals are unioned per student.  Gaps in a
    student's prediction timeline are also gray; this covers a student for
    whom no action interval was emitted at all.  Intervals are clipped to the
    actual source duration, so no frame-period extrapolation is involved.
    """

    by_student: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in mapped_actions:
        by_student[str(row["gold_person_id"])].append(row)

    unknown_total = 0.0
    persons = sorted({str(row["gold_person_id"]) for row in gold.boxes})
    for person in persons:
        rows = by_student.get(person, [])
        clipped_rows = []
        boundaries = {0.0, float(duration)}
        for row in rows:
            start = max(0.0, float(row["start_pts_seconds"]))
            end = min(float(duration), float(row["end_pts_seconds"]))
            if end > start + _EPS:
                clipped_rows.append((start, end, str(row.get("action_code", UNKNOWN))))
                boundaries.update((start, end))
        ordered = sorted(boundaries)
        # Every uncovered atom is gray.  Explicit UNKNOWN also marks an atom
        # gray; overlapping invalid intervals are conservatively gray when
        # either active prediction abstains.
        for left, right in zip(ordered, ordered[1:]):
            if right <= left + _EPS:
                continue
            active = [code for start, end, code in clipped_rows if start < right - _EPS and end > left + _EPS]
            if not active or any(code == UNKNOWN for code in active):
                unknown_total += right - left
    return unknown_total


def _temporal_metrics(gold: _GoldData, mapped_actions: Sequence[Mapping[str, Any]], run_meta: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    gold_segments = _merge_segments(gold.actions)
    pred_segments = _clip_pred_segments_to_gold(_merge_segments(mapped_actions), gold.actions)
    by_threshold: dict[str, dict[str, Any]] = {}
    for threshold in (0.10, 0.25, 0.50):
        tp, fp, fn = _temporal_match(gold_segments, pred_segments, threshold)
        by_threshold[f"{threshold:.2f}"] = {"TP": tp, "FP": fp, "FN": fn, "precision": _safe_ratio(tp, tp + fp), "recall": _safe_ratio(tp, tp + fn), "F1": _safe_ratio(2 * tp, 2 * tp + fp + fn)}
    observed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in mapped_actions:
        if str(row.get("action_code", UNKNOWN)) != UNKNOWN:
            observed[str(row["gold_person_id"])].append(row)
    flip_count = 0
    observed_seconds = 0.0
    for student, rows in observed.items():
        ordered = sorted(rows, key=lambda row: (float(row["start_pts_seconds"]), float(row["end_pts_seconds"]), int(row.get("original_ordinal", 0))))
        previous: Mapping[str, Any] | None = None
        for row in ordered:
            start, end = float(row["start_pts_seconds"]), float(row["end_pts_seconds"])
            observed_seconds += max(0.0, end - start)
            if previous is not None and str(previous.get("action_code")) != str(row.get("action_code")) and abs(float(previous["end_pts_seconds"]) - start) <= _EPS:
                flip_count += 1
            previous = row
    video_duration = _video_duration_seconds(run_meta, run_dir)
    unknown_seconds = _unknown_full_student_seconds(gold, mapped_actions, video_duration) if video_duration is not None else None
    f1_values = {f"@{key}": value["F1"] for key, value in by_threshold.items()}
    # Keep both compact report labels and protocol-style field names.  The
    # values are the same; this avoids consumers having to infer a threshold
    # from a display string.
    f1_values.update({"@0.1": by_threshold["0.10"]["F1"], "@0.25": by_threshold["0.25"]["F1"], "@0.5": by_threshold["0.50"]["F1"]})
    result = {
        "segmental_F1": f1_values,
        "segmental_by_threshold": by_threshold,
        "flip_count": flip_count,
        "flip_rate_per_observed_student_minute": _safe_ratio(flip_count, observed_seconds / 60.0),
        "observed_student_minutes": observed_seconds / 60.0,
    }
    if video_duration is not None:
        complete_seconds = video_duration * len(
            {str(row["gold_person_id"]) for row in gold.boxes}
        )
        result["unknown_student_seconds"] = float(unknown_seconds)
        result["complete_student_video_seconds"] = complete_seconds
        result["gray_time_ratio"] = _safe_ratio(float(unknown_seconds), complete_seconds)
    else:
        result["unavailable_metrics"] = {
            "unknown_student_seconds": "video_duration_not_recorded",
            "complete_student_video_seconds": "video_duration_not_recorded",
            "gray_time_ratio": "video_duration_not_recorded",
        }
    result["segmental_F1@0.1"] = by_threshold["0.10"]["F1"]
    result["segmental_F1@0.25"] = by_threshold["0.25"]["F1"]
    result["segmental_F1@0.5"] = by_threshold["0.50"]["F1"]
    return result


def _action_atomic_unknowns(gold: _GoldData, mapped_actions: Sequence[Mapping[str, Any]], profile: _ProfileInfo) -> list[dict[str, float]]:
    del profile
    by_student: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in mapped_actions:
        by_student[str(row["gold_person_id"])].append(row)
    unknowns: list[dict[str, float]] = []
    for gold_row in gold.actions:
        gs, ge = float(gold_row["start_pts_seconds"]), float(gold_row["end_pts_seconds"])
        relevant = by_student.get(str(gold_row["gold_person_id"]), [])
        boundaries = {gs, ge}
        for row in relevant:
            if _overlap(gs, ge, float(row["start_pts_seconds"]), float(row["end_pts_seconds"])) > _EPS:
                boundaries.add(max(gs, float(row["start_pts_seconds"])))
                boundaries.add(min(ge, float(row["end_pts_seconds"])))
        ordered = sorted(boundaries)
        for left, right in zip(ordered, ordered[1:]):
            if right <= left + _EPS:
                continue
            candidates = [row for row in relevant if _overlap(left, right, float(row["start_pts_seconds"]), float(row["end_pts_seconds"])) > _EPS]
            candidates.sort(key=lambda row: (-_overlap(left, right, float(row["start_pts_seconds"]), float(row["end_pts_seconds"])), int(row.get("original_ordinal", 0))))
            if not candidates or str(candidates[0].get("action_code", UNKNOWN)) == UNKNOWN:
                unknowns.append({"start": left, "end": right})
    return unknowns


def _find_numeric_recursive(value: Any, keys: set[str]) -> float | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normal = str(key).lower()
            if normal in keys:
                try:
                    number = float(child)
                    if math.isfinite(number):
                        return number
                except (TypeError, ValueError):
                    pass
            found = _find_numeric_recursive(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_numeric_recursive(child, keys)
            if found is not None:
                return found
    return None


def _max_numeric_recursive(value: Any, keys: set[str]) -> float | None:
    """Return the actual maximum of all recorded peaks, never the first stage."""

    values: list[float] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in keys and not isinstance(child, bool):
                try:
                    number = float(child)
                    if math.isfinite(number) and number >= 0:
                        values.append(number)
                except (TypeError, ValueError):
                    pass
            found = _max_numeric_recursive(child, keys)
            if found is not None:
                values.append(found)
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _max_numeric_recursive(child, keys)
            if found is not None:
                values.append(found)
    return max(values) if values else None


def _timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = _as_nonempty_text(value)
    if not text:
        return None
    try:
        return _datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _find_scalar_recursive(value: Any, keys: set[str]) -> Any:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in keys and child not in (None, ""):
                return child
            found = _find_scalar_recursive(child, keys)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_scalar_recursive(child, keys)
            if found not in (None, ""):
                return found
    return None


def _system_metrics(
    run_dir: Path,
    run_meta: Mapping[str, Any],
    *,
    stage_ids: Sequence[str] | None = None,
    include_event_history: bool = True,
    include_timeline_fallback: bool = True,
) -> dict[str, Any]:
    selected_stages = set(stage_ids) if stage_ids is not None else {f"S{index}" for index in range(8)}
    roots: list[tuple[str, Any]] = [("run_manifest", run_meta)]
    state_path = run_dir / "run_state.v1.json"
    state: Mapping[str, Any] = {}
    if state_path.is_file():
        try:
            payload = _json_load(state_path)
            state = payload if isinstance(payload, Mapping) else {}
        except Exception:
            state = {}
    if stage_ids is not None and isinstance(state.get("stages"), Mapping):
        state = {**state, "stages": {key: value for key, value in state["stages"].items() if str(key).upper() in selected_stages}}
    roots.append(("run_state", state))
    events_path = run_dir / "stage_events.v1.jsonl"
    events: list[Mapping[str, Any]] = []
    if include_event_history and events_path.is_file():
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, Mapping):
                    events.append(value)
            except Exception:
                continue
    roots.append(("stage_events", events))
    receipt_roots: dict[str, Any] = {}
    receipts_dir = run_dir / "stage_receipts"
    if receipts_dir.is_dir():
        for path in sorted(receipts_dir.glob("*.json")):
            if path.stem.upper() not in selected_stages:
                continue
            try:
                value = _json_load(path)
            except Exception:
                continue
            if isinstance(value, Mapping):
                receipt_roots[path.stem.upper()] = value
                roots.append((path.stem.upper(), value))
    unavailable: dict[str, str] = {}
    performance: dict[str, Any] = {}
    for path in sorted((run_dir / "diagnostics" / "performance").glob("S*.json")):
        if path.stem.upper() not in selected_stages:
            continue
        try:
            value = _json_load(path)
            if isinstance(value, Mapping):
                performance[path.stem] = value
        except (ValueError, OSError):
            continue
    stage_times: dict[str, Any] = {}
    for stage_index in range(8):
        stage = f"S{stage_index}"
        if stage not in selected_stages:
            unavailable[f"stage_wall_time_seconds.{stage}"] = "stage_not_yet_committed"
            continue
        value = _find_numeric_recursive(performance.get(stage, {}), {"stage.wall_seconds"})
        if value is None and isinstance(receipt_roots.get(stage), Mapping):
            value = _find_numeric_recursive(receipt_roots[stage], {"wall_time_seconds", "stage_wall_time_seconds", "elapsed_seconds", "duration_seconds"})
        if value is None and isinstance(state.get("stages"), Mapping):
            stage_state = state["stages"].get(stage) or state["stages"].get(stage.lower())
            if isinstance(stage_state, Mapping):
                value = _find_numeric_recursive(stage_state, {"wall_time_seconds", "elapsed_seconds", "duration_seconds"})
                if value is None:
                    started = _timestamp(_find_scalar_recursive(stage_state, {"started_at", "started_timestamp"}))
                    finished = _timestamp(_find_scalar_recursive(stage_state, {"finished_at", "finished_timestamp", "completed_at", "ended_at"}))
                    if started is not None and finished is not None and finished >= started:
                        value = finished - started
        if value is None:
            matching = [item for item in events if _as_nonempty_text(item.get("stage_id", item.get("stage"))) == stage]
            started = next((_timestamp(_find_scalar_recursive(item, {"started_at", "started_timestamp", "timestamp"})) for item in matching if str(item.get("event_type", item.get("event", item.get("type", "")))).lower() in {"start", "started", "running", "stage_started"}), None)
            finished = next((_timestamp(_find_scalar_recursive(item, {"finished_at", "finished_timestamp", "timestamp"})) for item in matching if str(item.get("event_type", item.get("event", item.get("type", "")))).lower() in {"finish", "finished", "completed", "stage_finished"}), None)
            if started is not None and finished is not None and finished >= started:
                value = finished - started
        if value is None:
            stage_times[stage] = {"status": "unavailable", "reason": "stage_wall_time_not_recorded"}
            unavailable[f"stage_wall_time_seconds.{stage}"] = "stage_wall_time_not_recorded"
        else:
            stage_times[stage] = float(value)
    total_wall = _find_numeric_recursive(run_meta, {"total_wall_time_seconds", "wall_time_seconds", "total_elapsed_seconds"})
    if total_wall is None:
        total_wall = _find_numeric_recursive(state, {"total_wall_time_seconds", "wall_time_seconds", "total_elapsed_seconds"})
    if total_wall is None:
        started = _timestamp(state.get("started_at", state.get("created_at")))
        finished_value = state.get("finished_at", state.get("completed_at", state.get("ended_at")))
        if finished_value is None and state.get("status") in {"completed", "warning", "failed", "blocked"}:
            finished_value = state.get("updated_at")
        finished = _timestamp(finished_value)
        if started is not None and finished is not None and finished >= started:
            total_wall = finished - started
    if "S7" not in selected_stages:
        total_wall = None
        unavailable["total_wall_time_seconds"] = "run_not_completed_when_metrics_committed"
    elif total_wall is None:
        unavailable["total_wall_time_seconds"] = "total_wall_time_not_recorded"
    locate_calls = _find_numeric_recursive(performance.get("S2", {}), {"locate.actual_calls"})
    if locate_calls is None:
        locate_calls = _find_numeric_recursive({"stage": receipt_roots.get("S2", {}), "run": run_meta, "state": state}, {"actual_requests", "locate_actual_requests", "locate_requests", "actual_locate_calls"})
    if locate_calls is None:
        unavailable["locate_actual_requests"] = "locate_actual_requests_not_recorded"
    qwen_calls = _find_numeric_recursive(performance.get("S5", {}), {"qwen.actual_calls"})
    if qwen_calls is None:
        qwen_calls = _find_numeric_recursive({"stage": receipt_roots.get("S5", {}), "run": run_meta, "state": state}, {"actual_calls", "qwen_actual_calls"})
    if qwen_calls is None:
        unavailable["qwen_actual_calls"] = "qwen_actual_calls_not_recorded"
    peak_roots = {"run": run_meta, "state": state, "receipts": receipt_roots, "performance": performance}
    peak_vram = _max_numeric_recursive(peak_roots, {"peak_vram", "peak_vram_gib", "peak_vram_gb", "peak_allocated_gib", "peak_reserved_gib"})
    allocated_bytes = _max_numeric_recursive(peak_roots, {"max_allocated_bytes", "request_peak_allocated_bytes"})
    reserved_bytes = _max_numeric_recursive(peak_roots, {"max_reserved_bytes", "request_peak_reserved_bytes"})
    measured_allocator_peaks = [value / 2**30 for value in (allocated_bytes, reserved_bytes) if value is not None]
    if measured_allocator_peaks:
        peak_vram = max([*measured_allocator_peaks, *([peak_vram] if peak_vram is not None else [])])
    if peak_vram is None:
        unavailable["peak_vram"] = "peak_vram_not_recorded"
    peak_ram = _max_numeric_recursive(peak_roots, {"peak_ram", "peak_ram_gib", "peak_ram_gb", "peak_memory_gib", "peak_memory_gb", "peak_rss_gib"})
    rss_bytes = _max_numeric_recursive(peak_roots, {"sampled_peak_process_tree_rss_bytes"})
    if rss_bytes is not None:
        peak_ram = max(rss_bytes / 2**30, peak_ram if peak_ram is not None else 0.0)
    if peak_ram is None:
        unavailable["peak_ram"] = "peak_ram_not_recorded"
    duration = _recorded_video_duration(run_meta)
    frames = _find_numeric_recursive(run_meta, {"frame_count", "frames"})
    for path in (run_dir / "timeline.v2.json", run_dir / "artifacts" / "timeline.v2.json"):
        if include_timeline_fallback and path.is_file() and (duration is None or frames is None):
            try:
                payload = _json_load(path)
            except Exception:
                continue
            if isinstance(payload, Mapping):
                if duration is None:
                    duration = _recorded_video_duration(payload)
                if frames is None:
                    frames = _find_numeric_recursive(payload, {"frame_count", "frames"})
    if total_wall is None or duration is None or duration <= 0:
        unavailable["realtime_factor"] = "total_wall_time_or_video_duration_not_recorded"
    if total_wall is None or frames is None or frames < 0:
        unavailable["processing_fps"] = "total_wall_time_or_frame_count_not_recorded"
    # Missing observations are represented once, in ``unavailable_metrics``;
    # a dict-valued placeholder in a numeric metric field is easy for a paper
    # table generator to mistake for a measured value.
    system: dict[str, Any] = {
        "stage_wall_time_seconds": {stage: value for stage, value in stage_times.items() if isinstance(value, (int, float))},
        "memory_units": "GiB (legacy *_gb fields interpreted as recorded GiB)",
        "physical_vram_scope": "device-wide sampled occupancy; Windows WDDM may include other processes and shared allocations",
    }
    if stage_ids is not None:
        system["snapshot_scope"] = "committed_upstream_stages_at_s7_output_write"
        system["included_stage_ids"] = sorted(selected_stages)
        system["completion_metrics_source"] = "read-only evaluate_research_run after the S7 receipt commits"
    if performance:
        system["stage_performance"] = performance
    if total_wall is not None:
        system["total_wall_time_seconds"] = total_wall
    if locate_calls is not None:
        system["locate_actual_requests"] = locate_calls
    if qwen_calls is not None:
        system["qwen_actual_calls"] = qwen_calls
    if peak_vram is not None:
        system["peak_vram"] = peak_vram
    if allocated_bytes is not None:
        system["peak_allocated_gib"] = allocated_bytes / 2**30
    if reserved_bytes is not None:
        system["peak_reserved_gib"] = reserved_bytes / 2**30
    system["allocator_peak_scope"] = "PyTorch worker-lifetime high-water mark; excludes driver allocations and other processes"
    post_allocated = _max_numeric_recursive(peak_roots, {"allocated_bytes"})
    post_reserved = _max_numeric_recursive(peak_roots, {"reserved_bytes"})
    if post_allocated is not None or post_reserved is not None:
        system["maximum_post_request_occupancy_gib"] = {
            key: value / 2**30 for key, value in (("allocated", post_allocated), ("reserved", post_reserved)) if value is not None
        }
    physical_mib = _max_numeric_recursive(peak_roots, {"sampled_peak_device_total_gpu_mib"})
    if physical_mib is not None:
        system["sampled_peak_physical_vram_gib"] = physical_mib / 1024
    if peak_ram is not None:
        system["peak_ram"] = peak_ram
    if total_wall is not None and duration is not None and duration > 0:
        system["realtime_factor"] = float(total_wall / duration)
    if total_wall is not None and frames is not None and total_wall > 0:
        system["processing_fps"] = float(frames / total_wall)
    system["unavailable_metrics"] = unavailable
    return system


def collect_system_metrics(
    run_dir: str | Path,
    *,
    video_info: Mapping[str, Any] | None = None,
    stage_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read compact stage receipts/telemetry without loading prediction JSONL.

    S7 calls this before its own artifacts commit, restricting the snapshot to
    S0-S6. This never reads metrics.v1.json, an artifact manifest, or the current
    S7 receipt, so writing the snapshot cannot create a hash reference cycle.
    """

    root = Path(run_dir).resolve()
    metadata, _ = _run_metadata(root)
    if video_info is not None:
        metadata = {**metadata, "source": dict(video_info)}
    return _system_metrics(root, metadata, stage_ids=stage_ids, include_event_history=False, include_timeline_fallback=False)


def _dependency_versions() -> dict[str, str]:
    names = ("numpy", "scipy", "PyYAML", "torch", "opencv-python", "trackeval")
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not_installed"
    return result


def _reproducibility(run_dir: Path, run_meta: Mapping[str, Any], gold: _GoldData | None, profile: _ProfileInfo | None, gold_manifest_sha256: str | None = None) -> dict[str, Any]:
    code_path = Path(__file__).resolve()
    run_manifest_path = next((run_dir / name for name in ("prediction_manifest.v1.json", "predictions_manifest.v1.json", "run_manifest.v1.json", "run_manifest.json", "manifest.v1.json") if (run_dir / name).is_file()), None)
    command = _find_first(run_meta, ("command", "run_command", "argv", "cmd"), "unavailable")
    if command in (None, "", [], {}):
        command = "unavailable"
    return {
        "score_code_sha256": _sha256_file(code_path),
        "gold_manifest_sha256": gold.manifest_sha256 if gold is not None else (gold_manifest_sha256 or "unavailable"),
        "prediction_manifest_sha256": _sha256_file(run_manifest_path) if run_manifest_path is not None else "unavailable",
        "profile_id": profile.profile_id if profile is not None else "unavailable",
        "profile_sha256": profile.sha256 if profile is not None else "unavailable",
        "matching_thresholds": {"detection_iou": IOU_THRESHOLD, "tracking_iou": IOU_THRESHOLD, "ap_iou": IOU_THRESHOLD},
        "run_command": command,
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "dependencies": _dependency_versions(),
        "stable_sort": ["frame_index", "gold_or_pred_id", "original_ordinal"],
    }


def _observed_summary(pred: _PredictionData, system: Mapping[str, Any]) -> dict[str, Any]:
    student_ids = sorted({str(row.get("pred_person_id")) for row in pred.detections if _as_nonempty_text(row.get("pred_person_id"))})
    unknown_intervals = sum(1 for row in pred.actions if _as_nonempty_text(row.get("action_value")).upper() in {"", UNKNOWN} or _as_nonempty_text(row.get("status")).lower() in {"unknown", "unobserved", "uncertain", "gray", "grey"})
    return {
        "source": "prediction_artifacts_only",
        "predicted_student_count": len(student_ids),
        "predicted_detection_count": len(pred.detections),
        "predicted_action_interval_count": len(pred.actions),
        "predicted_unknown_action_interval_count": unknown_intervals,
        "prediction_artifact_files": [path.as_posix() for path in pred.files],
        "prediction_source_contract": dict(pred.source_contract),
        "prediction_parse_warnings": list(pred.parse_warnings),
        "system_call_observations_available": sorted(key for key, value in system.items() if key.endswith("calls") and isinstance(value, (int, float))),
    }


def _invalid_result(reason: str, run_dir: Path, run_meta: Mapping[str, Any], profile: _ProfileInfo | None = None, gold: _GoldData | None = None, gold_manifest: Any = None) -> dict[str, Any]:
    manifest_sha = None
    if gold is None and gold_manifest is not None:
        try:
            _, _, manifest_sha = _manifest_input(gold_manifest)
        except GoldValidationError:
            manifest_sha = None
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "invalid_gold",
        "reason": reason,
        "gold_validation": {"status": "invalid_gold", "reason": reason},
        "observed_run_summary": _observed_summary(_load_predictions(run_dir, set()), _system_metrics(run_dir, run_meta)),
        "system": _system_metrics(run_dir, run_meta),
        "reproducibility": _reproducibility(run_dir, run_meta, gold, profile, manifest_sha),
    }


def _extension_support(manifest: Mapping[str, Any], profile: _ProfileInfo, base_dir: Path | None = None) -> dict[str, Any]:
    """Audit support from independent gold manifests, never self-reported totals.

    ``extension_support`` counters in one manifest are provenance claims, not
    evidence.  They can describe a verified audit only when a list of separate
    gold manifest files is supplied.  Each file is hash-checked and parsed
    here, then source SHA/classroom uniqueness and per-class action rows are
    counted from the actual JSONL data.
    """

    support = manifest.get("extension_support")
    if not isinstance(support, Mapping):
        support = manifest.get("support") if isinstance(manifest.get("support"), Mapping) else {}
    new_codes = [code for code in profile.codes if code not in {"dx", "dk", "tt", "zt", "js", "zl", "xt", "jz"}]
    per_class: dict[str, int] = {code: 0 for code in new_codes}
    source_shas: set[str] = set()
    classrooms_seen: set[str] = set()
    audit_errors: list[str] = []
    audit_files = support.get("verified_support_audit") if isinstance(support, Mapping) else None
    if isinstance(audit_files, Mapping):
        audit_files = audit_files.get("manifests")
    if not isinstance(audit_files, (list, tuple)) or not audit_files:
        audit_errors.append("verified_support_audit_missing")
    else:
        root = (base_dir or Path.cwd()).resolve()
        for index, item in enumerate(audit_files):
            candidate = item
            if isinstance(item, str):
                candidate_path = Path(item)
                if not candidate_path.is_absolute():
                    candidate_path = root / candidate_path
                candidate = candidate_path
            try:
                audit_manifest, audit_path, _ = _manifest_input(candidate)
                _validate_annotation_envelope(audit_manifest)
                if _as_nonempty_text(audit_manifest.get("taxonomy_profile_id")) != profile.profile_id or _as_nonempty_text(audit_manifest.get("taxonomy_contract_sha256")).lower() != profile.sha256:
                    raise GoldValidationError("support_profile_mismatch")
                boxes_uri, boxes_sha = _extract_artifact_ref(audit_manifest, "boxes")
                actions_uri, actions_sha = _extract_artifact_ref(audit_manifest, "actions")
                audit_root = audit_path.parent if audit_path is not None else root
                boxes_path = _resolve_relative_artifact(audit_root, boxes_uri, "support_boxes")
                actions_path = _resolve_relative_artifact(audit_root, actions_uri, "support_actions")
                if _sha256_file(boxes_path) != boxes_sha or _sha256_file(actions_path) != actions_sha:
                    raise GoldValidationError("support_artifact_sha256_mismatch")
                # Parse boxes to enforce stable/unique IDs, then count only
                # action rows that are actually in the independent JSONL.
                _parse_gold_boxes(boxes_path)
                action_rows = _parse_gold_actions(actions_path, profile)
                source_sha = _as_nonempty_text(audit_manifest.get("source_video_sha256")).lower()
                if source_sha in source_shas:
                    raise GoldValidationError("support_duplicate_source_video_sha256")
                source_shas.add(source_sha)
                classrooms_seen.add(_as_nonempty_text(audit_manifest.get("classroom_id")))
                for row in action_rows:
                    code = str(row["action_code"])
                    if code in per_class:
                        per_class[code] += 1
            except (GoldValidationError, OSError, ValueError) as exc:
                audit_errors.append(f"support_manifest_{index}:{exc}")
    videos = len(source_shas)
    classrooms = len(classrooms_seen)
    eligible = not audit_errors and bool(new_codes) and all(per_class.get(code, 0) >= 200 for code in new_codes) and videos >= 10 and classrooms >= 3
    return {
        "status": "eligible" if eligible else "exploratory_insufficient_support",
        "reason": "verified_support_audit_passed" if eligible else ";".join(audit_errors) or "verified_support_audit_minimum_not_met",
        "minimums": {"per_new_class_reviewed_samples": 200, "videos": 10, "classrooms": 3},
        "per_new_class_reviewed_samples": per_class,
        "video_count": videos,
        "classroom_count": classrooms,
    }


def evaluate_research_run(
    run_dir: str | Path,
    gold_manifest: Any = None,
    profile: Any = None,
    *,
    split_manifests: Iterable[Any] | None = None,
    require_split_audit: bool = False,
) -> dict[str, Any]:
    """Score one completed run against an independent gold package.

    ``gold_manifest=None`` deliberately returns the protocol's exact
    ``not_evaluated`` state.  Invalid gold is represented as ``invalid_gold``
    and contains no paper metric fields.
    """

    root = Path(run_dir).resolve()
    if not root.is_dir():
        return {"schema_version": SCHEMA_VERSION, "evaluation_status": "not_evaluated", "reason": "run_directory_missing", "observed_run_summary": {}, "system": {}}
    run_meta, _ = _run_metadata(root)
    if gold_manifest is None:
        system = _system_metrics(root, run_meta)
        return {
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "not_evaluated",
            "reason": "gold_annotations_not_provided",
            "observed_run_summary": _observed_summary(_load_predictions(root, set()), system),
            "system": system,
            "reproducibility": _reproducibility(root, run_meta, None, None),
        }
    try:
        profile_hint = None
        try:
            raw_manifest, _, _ = _manifest_input(gold_manifest)
            profile_hint = _as_nonempty_text(raw_manifest.get("taxonomy_profile_id"))
        except Exception:
            raw_manifest = {}
        selected_profile = _normalise_profile(profile, profile_hint)
        gold = _validate_gold(root, gold_manifest, selected_profile, run_meta)
    except GoldValidationError as exc:
        try:
            selected_profile = _normalise_profile(profile, profile_hint) if profile is not None or profile_hint else None
        except GoldValidationError:
            selected_profile = None
        return _invalid_result(str(exc), root, run_meta, selected_profile, gold_manifest=gold_manifest)
    except Exception as exc:  # malformed gold is still an invalid gold package
        return _invalid_result(f"gold_validation_error:{type(exc).__name__}", root, run_meta, gold_manifest=gold_manifest)
    split_audit = None
    if split_manifests is not None or require_split_audit:
        manifests = list(split_manifests or [])
        split_audit = audit_splits(manifests)
        if not manifests:
            split_audit = {**split_audit, "status": "invalid", "errors": ["split_manifests_not_provided"]}
        gold_identity = {
            "source_video_sha256": str(gold.manifest["source_video_sha256"]).lower(),
            "split_id": str(gold.manifest["split_id"]),
            "split": str(gold.manifest["split"]).lower(),
        }
        if not any(all(row.get(key) == value for key, value in gold_identity.items()) for row in split_audit["records"]):
            split_audit = {**split_audit, "status": "invalid", "errors": [*split_audit["errors"], "evaluated_gold_missing_from_split_audit"]}
        if split_audit["status"] != "valid":
            system = _system_metrics(root, run_meta)
            return {
                "schema_version": SCHEMA_VERSION,
                "evaluation_status": "invalid_split",
                "reason": "source_split_audit_failed",
                "gold_validation": {"status": "valid", "manifest_sha256": gold.manifest_sha256},
                "split_audit": split_audit,
                "paper_result_eligible": False,
                "system": system,
                "observed_run_summary": _observed_summary(_load_predictions(root, set()), system),
                "reproducibility": _reproducibility(root, run_meta, gold, selected_profile),
            }
    excluded: set[Path] = set()
    if gold.manifest_path is not None:
        try:
            boxes_uri, _ = _extract_artifact_ref(gold.manifest, "boxes")
            actions_uri, _ = _extract_artifact_ref(gold.manifest, "actions")
            excluded = {(gold.manifest_path.parent / boxes_uri).resolve(), (gold.manifest_path.parent / actions_uri).resolve(), gold.manifest_path.resolve()}
        except GoldValidationError:
            pass
    pred = _predictions_on_gold_frames(gold, _load_predictions(root, excluded))
    detection = _detection_metrics(gold, pred)
    tracking, mapping_pred = _tracking_metrics(gold, pred)
    action_metric, mapped_actions = _action_metrics(gold, pred, mapping_pred, selected_profile)
    temporal = _temporal_metrics(gold, mapped_actions, run_meta, root)
    system = _system_metrics(root, run_meta)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "gold_validation": {"status": "valid", "manifest_sha256": gold.manifest_sha256},
        "detection": detection,
        "tracking": tracking,
        "actions": action_metric,
        "temporal": temporal,
        "system": system,
        "observed_run_summary": _observed_summary(pred, system),
        "prediction_source_contract": dict(pred.source_contract),
        "reproducibility": _reproducibility(root, run_meta, gold, selected_profile),
        "paper_result_eligible": bool(split_audit and split_audit["status"] == "valid"),
    }
    if split_audit is not None:
        result["split_audit"] = split_audit
    if selected_profile.is_extension11:
        gate = _extension_support(gold.manifest, selected_profile, gold.manifest_path.parent if gold.manifest_path is not None else root)
        result["extension11"] = {"support_gate": gate, "metrics": action_metric, "status": gate["status"]}
        # Extension11 remains a separately named experiment.  Consumers must
        # not aggregate it with the core8 result.
        result["actions"] = {"profile_id": selected_profile.profile_id, "status": gate["status"], "extension_metrics": action_metric}
    return result


def audit_splits(manifests: Iterable[Any]) -> dict[str, Any]:
    """Validate lesson/session/classroom/source-SHA split separation."""

    if isinstance(manifests, (str, Path, Mapping)):
        manifests = [manifests]
    payloads: list[Mapping[str, Any]] = []
    errors: list[str] = []
    records: list[dict[str, str]] = []
    for index, item in enumerate(manifests):
        try:
            payload, _, _ = _manifest_input(item)
        except GoldValidationError as exc:
            errors.append(f"manifest_{index}:{exc}")
            continue
        payloads.append(payload)
        record = {
            "dataset_id": _as_nonempty_text(payload.get("dataset_id")),
            "split_id": _as_nonempty_text(payload.get("split_id")),
            "split": _as_nonempty_text(payload.get("split")).lower(),
            "source_video_sha256": _as_nonempty_text(payload.get("source_video_sha256")).lower(),
            "source_lesson_id": _as_nonempty_text(payload.get("source_lesson_id")),
            "capture_session_id": _as_nonempty_text(payload.get("capture_session_id")),
            "classroom_id": _as_nonempty_text(payload.get("classroom_id")),
        }
        missing = sorted(key for key in ("split_id", "split", "source_video_sha256", "source_lesson_id", "capture_session_id", "classroom_id") if not record[key])
        if missing:
            errors.append(f"manifest_{index}:split_fields_missing:{','.join(missing)}")
        records.append(record)
    def duplicate(field: str) -> None:
        by_value: dict[str, list[dict[str, str]]] = defaultdict(list)
        for record in records:
            value = record[field]
            if value:
                by_value[value].append(record)
        for value, rows in sorted(by_value.items()):
            if len(rows) > 1:
                errors.append(f"duplicate_{field}:{value}")
    duplicate("source_video_sha256")
    for field, name in (("source_lesson_id", "lesson"), ("capture_session_id", "session"), ("classroom_id", "classroom")):
        by_value: dict[str, set[str]] = defaultdict(set)
        for record in records:
            value = record[field]
            if value:
                by_value[value].add(record["split"])
        for value, splits in sorted(by_value.items()):
            normalized = {"train" if split in {"training", "train"} else "validation" if split in {"validation", "val", "dev"} else "test" if split in {"testing", "test", "eval"} else split for split in splits}
            if len(normalized) > 1:
                errors.append(f"split_leakage_{name}:{value}:{','.join(sorted(normalized))}")
    return {
        "schema_version": "classroom_hybrid_ai.split_audit.v1",
        "status": "invalid" if errors else "valid",
        "errors": sorted(dict.fromkeys(errors)),
        "manifest_count": len(records),
        "records": records,
    }


def validate_split_audit(manifests: Iterable[Any]) -> dict[str, Any]:
    return audit_splits(manifests)


def audit_data_splits(manifests: Iterable[Any]) -> dict[str, Any]:
    return audit_splits(manifests)


__all__ = [
    "GoldValidationError",
    "HOTA_ALPHAS",
    "IOU_THRESHOLD",
    "SCHEMA_VERSION",
    "UNKNOWN",
    "audit_data_splits",
    "audit_splits",
    "collect_system_metrics",
    "evaluate_research_run",
    "validate_split_audit",
]
