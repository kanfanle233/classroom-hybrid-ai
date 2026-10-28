"""Persistent append-only candidate-memory registry for open classroom actions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .open_vocab import ActionDescriptor, ActionRegistrySnapshot, action_id, colour_for_index, normalize_phrase
from .storage import atomic_write_json


_REGISTRY_FILE = re.compile(r"^registry\.v(?P<version>\d+)\.json$")


def load_latest(root: Path) -> ActionRegistrySnapshot:
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("registry.v*.json"))
    if not files:
        return ActionRegistrySnapshot.create((), registry_version=0, created_at="1970-01-01T00:00:00+00:00")
    import json
    with files[-1].open("r", encoding="utf-8") as handle:
        return ActionRegistrySnapshot.from_dict(json.load(handle))


def audit_registry(root: Path) -> dict[str, Any]:
    """Verify the append-only action registry and its immutable history.

    A registry is allowed to be empty before the first course run.  Once a
    version exists, files must form a contiguous ``v0001 -> v0002`` chain,
    every snapshot hash must validate, each parent digest must point to the
    previous snapshot, and an existing action's identity fields and colour
    must never change.  The check is read-only and never creates the directory.
    """

    directory = Path(root).expanduser().resolve()
    if not directory.exists():
        return {
            "schema_version": "classroom_hybrid_ai.action_registry_audit.v1",
            "status": "empty",
            "root": str(directory),
            "versions": 0,
            "actions": 0,
            "errors": [],
        }
    if not directory.is_dir():
        return {
            "schema_version": "classroom_hybrid_ai.action_registry_audit.v1",
            "status": "failed",
            "root": str(directory),
            "versions": 0,
            "actions": 0,
            "errors": ["registry_root_not_directory"],
        }

    parsed: list[tuple[int, Path]] = []
    errors: list[str] = []
    for path in sorted(directory.glob("registry.v*.json")):
        match = _REGISTRY_FILE.fullmatch(path.name)
        if match is None:
            errors.append(f"invalid_filename:{path.name}")
            continue
        parsed.append((int(match.group("version")), path))
    parsed.sort(key=lambda item: (item[0], item[1].name))
    if not parsed:
        return {
            "schema_version": "classroom_hybrid_ai.action_registry_audit.v1",
            "status": "failed" if errors else "empty",
            "root": str(directory),
            "versions": 0,
            "actions": 0,
            "errors": errors,
        }

    versions = [version for version, _ in parsed]
    expected_versions = list(range(1, max(versions) + 1))
    if versions != expected_versions:
        errors.append(f"version_sequence_invalid:actual={versions}:expected={expected_versions}")
    if len(set(versions)) != len(versions):
        errors.append("duplicate_registry_version")

    snapshots: list[ActionRegistrySnapshot] = []
    for version, path in parsed:
        try:
            with path.open("r", encoding="utf-8") as handle:
                snapshot = ActionRegistrySnapshot.from_dict(json.load(handle))
            if snapshot.registry_version != version:
                errors.append(
                    f"filename_version_mismatch:{path.name}:payload={snapshot.registry_version}"
                )
            snapshots.append(snapshot)
        except Exception as exc:
            errors.append(f"snapshot_invalid:{path.name}:{type(exc).__name__}:{exc}")

    previous: ActionRegistrySnapshot | None = None
    empty_parent_sha = ActionRegistrySnapshot.create(
        (), registry_version=0, parent_sha256=None, created_at="1970-01-01T00:00:00+00:00"
    ).sha256
    immutable: dict[str, tuple[str, str, str, tuple[str, ...], str]] = {}
    for snapshot in snapshots:
        if previous is not None and snapshot.parent_sha256 != previous.sha256:
            errors.append(
                f"parent_digest_mismatch:v{snapshot.registry_version:04d}:"
                f"expected={previous.sha256}:actual={snapshot.parent_sha256}"
            )
        if previous is None and snapshot.parent_sha256 not in {None, "", empty_parent_sha}:
            errors.append("first_snapshot_parent_must_be_empty")
        for action in snapshot.actions:
            signature = (
                action.name_zh,
                action.definition,
                json.dumps(dict(action.attributes), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                tuple(action.aliases),
                action.color.upper(),
            )
            old = immutable.get(action.action_id)
            if old is not None and old != signature:
                errors.append(
                    f"immutable_action_changed:v{snapshot.registry_version:04d}:{action.action_id}"
                )
            immutable.setdefault(action.action_id, signature)
        previous = snapshot

    latest_actions = len(snapshots[-1].actions) if snapshots else 0
    return {
        "schema_version": "classroom_hybrid_ai.action_registry_audit.v1",
        "status": "ok" if not errors else "failed",
        "root": str(directory),
        "versions": len(snapshots),
        "latest_registry_version": snapshots[-1].registry_version if snapshots else 0,
        "actions": latest_actions,
        "errors": errors,
        "latest_sha256": snapshots[-1].sha256 if snapshots else None,
    }


def append_candidates(
    root: Path,
    existing: ActionRegistrySnapshot,
    candidates: Iterable[ActionDescriptor],
) -> ActionRegistrySnapshot:
    """Append only genuinely new actions; immutable history is never rewritten."""

    current = {item.action_id: item for item in existing.actions}
    next_colour = len(current)
    for item in candidates:
        if item.action_id in current:
            continue
        current[item.action_id] = ActionDescriptor(
            action_id=item.action_id,
            name_zh=item.name_zh,
            definition=item.definition,
            aliases=item.aliases,
            attributes=item.attributes,
            color=item.color or colour_for_index(next_colour),
            status="shared_candidate",
            support_windows=item.support_windows,
            support_students=item.support_students,
            evidence_quality=item.evidence_quality,
        )
        next_colour += 1
    if tuple(sorted(current)) == tuple(sorted(item.action_id for item in existing.actions)):
        return existing
    snapshot = ActionRegistrySnapshot.create(
        current.values(), registry_version=existing.registry_version + 1, parent_sha256=existing.sha256,
    )
    atomic_write_json(root / f"registry.v{snapshot.registry_version:04d}.json", snapshot.to_dict())
    return snapshot


def descriptors_from_discovery(
    rows: Sequence[dict],
    *,
    minimum_windows: int,
    same_student_gap_seconds: float,
    minimum_quality: float,
    initial_colour_index: int,
) -> tuple[list[ActionDescriptor], list[ActionDescriptor]]:
    """Turn discovery output into shared candidates and course-only candidates.

    The input rows are intentionally model-agnostic dictionaries produced after
    strict response validation.  Similarity/Qwen synonym adjudication occurs
    before this function, so this function only makes repeatability decisions.
    """

    groups: dict[str, list[dict]] = {}
    for row in rows:
        # A registry action selected by the semantic adjudicator is already
        # persistent.  Do not recreate it under a phrase-derived hash.
        if row.get("canonical_action_id"):
            continue
        phrase = normalize_phrase(str(row.get("canonical_phrase") or row.get("raw_phrase") or ""))
        if phrase:
            groups.setdefault(phrase, []).append(row)
    shared: list[ActionDescriptor] = []
    provisional: list[ActionDescriptor] = []
    colour = initial_colour_index
    for phrase, items in sorted(groups.items()):
        windows = sorted({round(float(item.get("pts_seconds", 0.0)), 3) for item in items})
        students = {str(item.get("student_id", "")) for item in items if item.get("student_id")}
        quality = sorted(float(item.get("evidence_quality", 0.0)) for item in items)
        median_quality = quality[len(quality) // 2] if quality else 0.0
        independent = len(windows) >= minimum_windows and _has_independent_support(items, same_student_gap_seconds)
        structural: dict[str, object] = {"primary_phrase": phrase}
        for key in ("posture", "movement_state", "head_direction", "hand_action", "visible_objects", "interaction_target"):
            values: list[str] = []
            for item in items:
                attributes = item.get("attributes")
                value = attributes.get(key) if isinstance(attributes, dict) else None
                if value is None:
                    continue
                encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if encoded not in values:
                    values.append(encoded)
            if values:
                decoded = [json.loads(value) for value in sorted(values)]
                structural[key] = decoded[0] if len(decoded) == 1 else decoded
        status = "shared_candidate" if independent and median_quality >= minimum_quality else "session_provisional"
        descriptor = ActionDescriptor(
            action_id=action_id(phrase, structural),
            name_zh=phrase,
            definition=f"画面中可见的{phrase}",
            aliases=(phrase,),
            attributes=structural,
            color=colour_for_index(colour),
            status=status,
            support_windows=len(windows),
            support_students=len(students),
            evidence_quality=median_quality,
        )
        colour += 1
        (shared if status == "shared_candidate" else provisional).append(descriptor)
    return shared, provisional


def _has_independent_support(items: Sequence[dict], gap: float) -> bool:
    students = {str(item.get("student_id", "")) for item in items if item.get("student_id")}
    if len(students) >= 2:
        return True
    times = sorted(float(item.get("pts_seconds", 0.0)) for item in items)
    return any(right - left >= gap for left, right in zip(times, times[1:]))


__all__ = ["append_candidates", "audit_registry", "descriptors_from_discovery", "load_latest"]
