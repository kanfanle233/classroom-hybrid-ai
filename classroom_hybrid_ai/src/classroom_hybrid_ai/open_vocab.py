"""Open-vocabulary action contracts and deterministic timeline construction.

This module deliberately contains no model calls.  It makes the output of a
vision-language model auditable and repeatable: a course freezes a vocabulary
before classification and every student receives a non-overlapping timeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


UNOBSERVED_ACTION_ID = "unobserved"
UNOBSERVED_COLOR = "#B8B8B8"
PALETTE = (
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
    "#B07AA1", "#FF9DA7", "#9C755F", "#17BECF", "#2F4B7C", "#A05195",
    "#D45087", "#F95D6A", "#FF7C43", "#7A8F32", "#5F9EA0", "#C44E52",
    "#8172B3", "#64B5CD", "#8CD17D", "#B6992D", "#499894", "#86BCB6",
)
_SPACE = re.compile(r"\s+")
_LEADING = re.compile(r"^(?:学生|该学生|此学生|画面中学生)(?:正在|在)?")
_TRAILING = re.compile(r"(?:中|状态|动作)$")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_phrase(value: str) -> str:
    """Normalise only presentation noise; do not erase meaningful action words."""

    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = _LEADING.sub("", text)
    text = _TRAILING.sub("", text)
    text = re.sub(r"[，。；、,.!！:：()（）\[\]【】]", "", text)
    return _SPACE.sub("", text)


def action_id(name_zh: str, attributes: Mapping[str, Any]) -> str:
    name = normalize_phrase(name_zh)
    if not name:
        raise ValueError("action_name_empty")
    return "a_" + canonical_sha256({"name_zh": name, "attributes": dict(attributes)})[:12]


def colour_for_index(index: int) -> str:
    if index < 0:
        raise ValueError("colour_index_negative")
    if index < len(PALETTE):
        return PALETTE[index]
    # Stable fallback.  The first 24 entries remain hand selected; later ones
    # use a golden-angle OKLCH sweep, which is more perceptually even than an
    # RGB/HSL hue rotation and remains deterministic across machines.
    hue = math.radians((index * 137.50776405003785) % 360.0)
    lightness = 0.68 + (index % 3) * 0.035
    return _oklch_to_hex(lightness, 0.14, hue)


def _oklch_to_hex(lightness: float, chroma: float, hue_radians: float) -> str:
    """Convert an OKLCH colour to clipped sRGB without a third-party package."""

    a = chroma * math.cos(hue_radians)
    b = chroma * math.sin(hue_radians)
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    red_linear = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    green_linear = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    blue_linear = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def compand(value: float) -> int:
        value = max(0.0, min(1.0, value))
        srgb = 12.92 * value if value <= 0.0031308 else 1.055 * (value ** (1 / 2.4)) - 0.055
        return round(max(0.0, min(1.0, srgb)) * 255)

    return "#{:02X}{:02X}{:02X}".format(
        compand(red_linear), compand(green_linear), compand(blue_linear)
    )


@dataclass(frozen=True)
class ActionDescriptor:
    action_id: str
    name_zh: str
    definition: str
    aliases: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    color: str = ""
    status: str = "shared_candidate"
    support_windows: int = 0
    support_students: int = 0
    evidence_quality: float = 0.0

    def __post_init__(self) -> None:
        if not self.action_id.startswith("a_"):
            raise ValueError("action_id_invalid")
        if not normalize_phrase(self.name_zh):
            raise ValueError("action_name_empty")
        if not self.definition.strip():
            raise ValueError("action_definition_empty")
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", self.color):
            raise ValueError("action_color_invalid")
        if self.status not in {"shared_candidate", "session_provisional"}:
            raise ValueError("action_status_invalid")
        if self.support_windows < 0 or self.support_students < 0:
            raise ValueError("action_support_negative")
        if not 0.0 <= float(self.evidence_quality) <= 1.0:
            raise ValueError("action_evidence_quality_invalid")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        value["attributes"] = dict(self.attributes)
        return value

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionDescriptor":
        data = dict(value)
        data["aliases"] = tuple(data.get("aliases", ()))
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "ActionDescriptor":
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise ValueError("action_descriptor_json_must_be_object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class ActionRegistrySnapshot:
    schema_version: str
    registry_version: int
    parent_sha256: str | None
    created_at: str
    actions: tuple[ActionDescriptor, ...]
    sha256: str

    @classmethod
    def create(
        cls,
        actions: Iterable[ActionDescriptor],
        *,
        registry_version: int = 1,
        parent_sha256: str | None = None,
        created_at: str | None = None,
    ) -> "ActionRegistrySnapshot":
        ordered = tuple(sorted(actions, key=lambda item: item.action_id))
        payload = {
            "schema_version": "classroom_hybrid_ai.action_registry.v1",
            "registry_version": registry_version,
            "parent_sha256": parent_sha256,
            "actions": [item.to_dict() for item in ordered],
        }
        return cls(
            schema_version=payload["schema_version"],
            registry_version=registry_version,
            parent_sha256=parent_sha256,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            actions=ordered,
            sha256=canonical_sha256(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "parent_sha256": self.parent_sha256,
            "created_at": self.created_at,
            "actions": [item.to_dict() for item in self.actions],
            "sha256": self.sha256,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionRegistrySnapshot":
        actions = tuple(ActionDescriptor.from_dict(item) for item in value.get("actions", ()))
        instance = cls(
            schema_version=str(value["schema_version"]),
            registry_version=int(value["registry_version"]),
            parent_sha256=value.get("parent_sha256"),
            created_at=str(value["created_at"]),
            actions=actions,
            sha256=str(value["sha256"]),
        )
        expected = canonical_sha256({
            "schema_version": instance.schema_version,
            "registry_version": instance.registry_version,
            "parent_sha256": instance.parent_sha256,
            "actions": [item.to_dict() for item in instance.actions],
        })
        if expected != instance.sha256:
            raise ValueError("action_registry_sha256_mismatch")
        return instance

    @classmethod
    def from_json(cls, payload: str) -> "ActionRegistrySnapshot":
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise ValueError("action_registry_json_must_be_object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class ActionObservation:
    student_id: str
    pts_seconds: float
    raw_phrase: str | None
    primary_action_id: str | None
    secondary_phrases: tuple[str, ...] = ()
    observability: str = "none"
    evidence_frame_indices: tuple[int, ...] = ()
    evidence_quality: float = 0.0
    source: str = "qwen"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unobserved_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.student_id:
            raise ValueError("student_id_empty")
        if not math.isfinite(float(self.pts_seconds)) or self.pts_seconds < 0:
            raise ValueError("observation_pts_invalid")
        if self.observability not in {"clear", "partial", "none"}:
            raise ValueError("observability_invalid")
        if self.observability == "none" and self.primary_action_id is not None:
            raise ValueError("unobservable_observation_has_action")
        if self.unobserved_reason is not None and (
            not isinstance(self.unobserved_reason, str) or not self.unobserved_reason.strip()
        ):
            raise ValueError("unobserved_reason_invalid")
        if not 0.0 <= float(self.evidence_quality) <= 1.0:
            raise ValueError("observation_evidence_quality_invalid")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["secondary_phrases"] = list(self.secondary_phrases)
        value["evidence_frame_indices"] = list(self.evidence_frame_indices)
        value["metadata"] = dict(self.metadata)
        return value

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionObservation":
        data = dict(value)
        data["secondary_phrases"] = tuple(data.get("secondary_phrases", ()))
        data["evidence_frame_indices"] = tuple(data.get("evidence_frame_indices", ()))
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "ActionObservation":
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise ValueError("action_observation_json_must_be_object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class ActionInterval:
    student_id: str
    start_pts_seconds: float
    end_pts_seconds: float
    action_id: str
    action_name_zh: str
    color: str
    status: str
    source: str
    unobserved_reason: str | None = None
    # A merged coloured span can be supported by more than one Qwen candidate.
    # Keep every originating candidate ID so both directions remain auditable:
    # interval -> Qwen response and Qwen response -> interval(s).
    evidence_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.student_id:
            raise ValueError("interval_student_empty")
        if not (0.0 <= self.start_pts_seconds <= self.end_pts_seconds):
            raise ValueError("interval_bounds_invalid")
        if self.status not in {"confirmed", "unobserved"}:
            raise ValueError("interval_status_invalid")
        if self.status == "unobserved":
            if self.action_id != UNOBSERVED_ACTION_ID or self.color.upper() != UNOBSERVED_COLOR:
                raise ValueError("unobserved_interval_must_use_reserved_action")
            if not self.unobserved_reason:
                raise ValueError("unobserved_reason_required")
        elif self.unobserved_reason is not None:
            raise ValueError("confirmed_interval_cannot_have_unobserved_reason")
        if any(not isinstance(item, str) for item in self.evidence_event_ids):
            raise ValueError("interval_evidence_event_id_invalid")
        evidence_ids = tuple(dict.fromkeys(item.strip() for item in self.evidence_event_ids))
        if any(not item for item in evidence_ids):
            raise ValueError("interval_evidence_event_id_invalid")
        if self.status == "unobserved" and evidence_ids:
            raise ValueError("unobserved_interval_cannot_have_evidence_event_ids")
        object.__setattr__(self, "evidence_event_ids", evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_event_ids"] = list(self.evidence_event_ids)
        return value

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionInterval":
        data = dict(value)
        ids = data.get("evidence_event_ids")
        legacy_id = data.pop("evidence_event_id", None)
        if ids is None:
            ids = () if legacy_id is None or legacy_id == "" else (legacy_id,)
        if isinstance(ids, str):
            ids = (ids,)
        if not isinstance(ids, (list, tuple)):
            raise ValueError("interval_evidence_event_ids_invalid")
        data["evidence_event_ids"] = tuple(ids)
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "ActionInterval":
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise ValueError("action_interval_json_must_be_object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class TimelineDocument:
    """Validated timeline envelope consumed by a future frontend renderer."""

    schema_version: str = "classroom_hybrid_ai.timeline.v1"
    video: Mapping[str, Any] = field(default_factory=dict)
    student_rows: tuple[str, ...] = ()
    session_vocabulary_sha256: str = ""
    registry_sha256: str = ""
    start_pts_seconds: float = 0.0
    end_pts_seconds: float = 0.0
    intervals: tuple[ActionInterval, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "classroom_hybrid_ai.timeline.v1":
            raise ValueError("timeline_schema_version_invalid")
        object.__setattr__(self, "video", dict(self.video))
        rows = tuple(str(item) for item in self.student_rows)
        if any(not item.strip() for item in rows) or len(set(rows)) != len(rows):
            raise ValueError("timeline_student_rows_invalid")
        object.__setattr__(self, "student_rows", rows)
        for name in ("session_vocabulary_sha256", "registry_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"timeline_{name}_invalid")
        start = float(self.start_pts_seconds)
        end = float(self.end_pts_seconds)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("timeline_time_range_invalid")
        object.__setattr__(self, "start_pts_seconds", start)
        object.__setattr__(self, "end_pts_seconds", end)
        intervals = tuple(self.intervals)
        if any(not isinstance(item, ActionInterval) for item in intervals):
            raise TypeError("timeline_intervals_must_be_action_intervals")
        order = {student: index for index, student in enumerate(rows)}
        previous_end: dict[str, float] = {}
        for item in sorted(intervals, key=lambda value: (order.get(value.student_id, len(order)), value.start_pts_seconds)):
            if item.student_id not in order:
                raise ValueError("timeline_interval_student_not_in_rows")
            if item.start_pts_seconds < start or item.end_pts_seconds > end:
                raise ValueError("timeline_interval_outside_time_range")
            if item.student_id in previous_end and item.start_pts_seconds < previous_end[item.student_id]:
                raise ValueError("timeline_intervals_overlap")
            previous_end[item.student_id] = item.end_pts_seconds
        object.__setattr__(self, "intervals", intervals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video": dict(self.video),
            "student_rows": list(self.student_rows),
            "session_vocabulary_sha256": self.session_vocabulary_sha256,
            "registry_sha256": self.registry_sha256,
            "start_pts_seconds": self.start_pts_seconds,
            "end_pts_seconds": self.end_pts_seconds,
            "duration_seconds": self.end_pts_seconds - self.start_pts_seconds,
            "intervals": [item.to_dict() for item in self.intervals],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimelineDocument":
        data = dict(value)
        duration = data.get("duration_seconds")
        start = data.get("start_pts_seconds", 0.0)
        end = data.get("end_pts_seconds", float(start) + float(duration or 0.0))
        data["start_pts_seconds"] = start
        data["end_pts_seconds"] = end
        data["student_rows"] = tuple(data.get("student_rows", ()))
        data["intervals"] = tuple(ActionInterval.from_dict(item) for item in data.get("intervals", ()))
        data.pop("duration_seconds", None)
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "TimelineDocument":
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise ValueError("timeline_json_must_be_object")
        return cls.from_dict(value)


def evidence_quality(*, visible_tiles: int, crop_short_side: float, mean_pose_confidence: float) -> float:
    return max(0.0, min(1.0,
        0.50 * min(3, max(0, visible_tiles)) / 3.0
        + 0.25 * min(1.0, max(0.0, crop_short_side) / 256.0)
        + 0.25 * min(1.0, max(0.0, mean_pose_confidence))
    ))


def session_vocabulary(
    shared: Sequence[ActionDescriptor],
    session: Sequence[ActionDescriptor],
    *,
    maximum_actions: int = 24,
) -> tuple[ActionDescriptor, ...]:
    """Freeze a course vocabulary without changing any existing colour/ID."""

    combined: dict[str, ActionDescriptor] = {item.action_id: item for item in shared}
    for item in session:
        combined.setdefault(item.action_id, item)
    ordered = sorted(
        combined.values(),
        key=lambda item: (-item.support_windows, -item.support_students, -item.evidence_quality, item.action_id),
    )
    return tuple(ordered[:maximum_actions])


def session_vocabulary_digest(
    vocabulary: Sequence[ActionDescriptor],
    *,
    registry_sha256: str,
) -> str:
    """Return the digest of the frozen vocabulary artifact.

    The returned value is the hash of the artifact payload before its own
    ``sha256`` member is added.  Keeping this calculation here prevents the
    prompt-bound vocabulary hash and the persisted JSON hash from diverging.
    """

    return canonical_sha256({
        "schema_version": "classroom_hybrid_ai.session_vocabulary.v1",
        "registry_sha256": str(registry_sha256),
        "actions": [item.to_dict() for item in vocabulary],
    })


def build_timeline(
    observations: Sequence[ActionObservation],
    vocabulary: Sequence[ActionDescriptor],
    *,
    student_ids: Sequence[str],
    duration_seconds: float,
    max_hold_seconds: float = 120.0,
    max_gap_seconds: float = 1.0,
) -> list[ActionInterval]:
    """Build one non-overlapping primary-action row per student.

    A clear observation with two cited temporal cells can seed a coloured
    interval.  A partial observation needs the same action at an immediately
    adjacent anchor; isolated partial evidence remains explicitly unobserved.
    """

    if duration_seconds < 0:
        raise ValueError("duration_seconds_negative")
    if max_hold_seconds <= 0 or max_gap_seconds < 0:
        raise ValueError("timeline_policy_invalid")
    descriptor = {item.action_id: item for item in vocabulary}
    output: list[ActionInterval] = []
    ordered_students = list(dict.fromkeys(str(student) for student in student_ids))
    by_student: dict[str, list[ActionObservation]] = {student: [] for student in ordered_students}
    for item in observations:
        by_student.setdefault(item.student_id, []).append(item)
    ordered_students.extend(sorted(set(by_student).difference(ordered_students)))
    for student_id in ordered_students:
        rows = sorted(by_student[student_id], key=lambda item: item.pts_seconds)
        eligible: list[ActionObservation] = []
        for index, row in enumerate(rows):
            if row.primary_action_id not in descriptor:
                continue
            if row.observability == "clear" and len(row.evidence_frame_indices) >= 2:
                eligible.append(row)
                continue
            if row.observability != "partial":
                continue
            neighbours = (
                rows[index - 1] if index else None,
                rows[index + 1] if index + 1 < len(rows) else None,
            )
            if any(
                neighbour is not None
                and neighbour.primary_action_id == row.primary_action_id
                and neighbour.observability in {"clear", "partial"}
                and abs(neighbour.pts_seconds - row.pts_seconds) <= max_gap_seconds
                and len(neighbour.evidence_frame_indices) >= 2
                for neighbour in neighbours
            ):
                eligible.append(row)

        # If two different actions cite the same temporal evidence, neither is
        # safe to paint as the primary action.  Keep the disputed span gray so
        # a later consumer can distinguish it from an ordinary unreviewed gap.
        conflict_times: list[float] = []
        conflict_indices: set[int] = set()
        for left_index, left in enumerate(eligible):
            left_frames = set(left.evidence_frame_indices)
            for right_index in range(left_index + 1, len(eligible)):
                right = eligible[right_index]
                if right.pts_seconds - left.pts_seconds > max_hold_seconds:
                    break
                if right.primary_action_id == left.primary_action_id:
                    continue
                same_anchor = abs(right.pts_seconds - left.pts_seconds) <= 1e-6
                shared_frame = bool(left_frames & set(right.evidence_frame_indices))
                if same_anchor or shared_frame:
                    conflict_times.append(min(left.pts_seconds, right.pts_seconds))
                    conflict_indices.update({left_index, right_index})
        accepted = [row for index, row in enumerate(eligible) if index not in conflict_indices]

        explicit_gap_markers = [
            (row.pts_seconds, row.unobserved_reason or "unreviewed")
            for row in rows
            if row.primary_action_id not in descriptor or row.observability == "none"
        ]
        # A short occlusion can carry the last confirmed action across the
        # gap, provided the next accepted anchor agrees and arrives within the
        # configured bridge window. Longer gaps remain gray and auditable.
        accepted_by_time = sorted(accepted, key=lambda item: item.pts_seconds)
        bridged_markers: list[tuple[float, str]] = []
        for marker_time, marker_reason in explicit_gap_markers:
            previous = next(
                (item for item in reversed(accepted_by_time) if item.pts_seconds < marker_time),
                None,
            )
            following = next(
                (item for item in accepted_by_time if item.pts_seconds > marker_time),
                None,
            )
            if (
                previous is not None
                and following is not None
                and previous.primary_action_id == following.primary_action_id
                and following.pts_seconds - marker_time <= max_gap_seconds
                and marker_time - previous.pts_seconds <= max_gap_seconds
            ):
                continue
            bridged_markers.append((marker_time, marker_reason))
        explicit_gap_markers = bridged_markers

        # Merge conflict points into bounded spans.  A conflict is held only
        # until the next accepted anchor or the normal action hold limit.
        conflict_spans: list[tuple[float, float]] = []
        accepted_times = sorted(row.pts_seconds for row in accepted)
        for start in sorted(set(conflict_times)):
            next_anchor = next((pts for pts in accepted_times if pts > start), duration_seconds)
            end = min(duration_seconds, max(start, min(start + max_hold_seconds, next_anchor)))
            if end <= start and duration_seconds > start:
                end = min(duration_seconds, start + max_hold_seconds)
            if end > start:
                if conflict_spans and start <= conflict_spans[-1][1]:
                    conflict_spans[-1] = (conflict_spans[-1][0], max(conflict_spans[-1][1], end))
                else:
                    conflict_spans.append((start, end))

        timeline_events: list[tuple[float, str, Any]] = [
            (row.pts_seconds, "action", row) for row in accepted
        ] + [
            (start, "conflict", (start, end)) for start, end in conflict_spans
        ] + [
            (
                start,
                "gap",
                (
                    start,
                    min(
                        duration_seconds,
                        start + max_hold_seconds,
                        next((pts for pts in accepted_times if pts > start), duration_seconds),
                    ),
                    reason,
                ),
            )
            for start, reason in explicit_gap_markers
            if 0.0 <= start < duration_seconds
        ]
        timeline_events.sort(
            key=lambda item: (
                item[0],
                0 if item[1] == "conflict" else 1 if item[1] == "gap" else 2,
            )
        )
        cursor = 0.0
        for event_index, (start, kind, payload) in enumerate(timeline_events):
            if start > cursor:
                output.append(_unobserved(student_id, cursor, min(start, duration_seconds), "unreviewed"))
            if kind in {"conflict", "gap"}:
                if kind == "conflict":
                    _, end = payload
                    reason = "conflict"
                else:
                    _, end, reason = payload
                gap_end = max(cursor, min(duration_seconds, float(end)))
                if gap_end > max(cursor, start):
                    output.append(_unobserved(student_id, max(cursor, start), gap_end, str(reason)))
                cursor = max(cursor, gap_end)
                continue
            row = payload
            row_start = max(cursor, row.pts_seconds)
            end = min(duration_seconds, row.pts_seconds + max_hold_seconds)
            for next_start, _, _ in timeline_events[event_index + 1:]:
                if next_start > row.pts_seconds:
                    end = min(end, next_start)
                    # An anchor gap longer than the configured bridge window
                    # cannot be painted as continuously observed action.  End
                    # the colour block at the bridge boundary and let the
                    # normal cursor fill the remainder gray.
                    if next_start - row.pts_seconds > max_gap_seconds:
                        end = min(end, row.pts_seconds + max_gap_seconds)
                    break
            if end > row_start:
                action = descriptor[row.primary_action_id]
                metadata = row.metadata if isinstance(row.metadata, Mapping) else {}
                candidate_event_id = metadata.get("candidate_event_id")
                evidence_event_ids = (
                    (str(candidate_event_id).strip(),)
                    if isinstance(candidate_event_id, str) and candidate_event_id.strip()
                    else ()
                )
                output.append(ActionInterval(
                    student_id=student_id,
                    start_pts_seconds=row_start,
                    end_pts_seconds=end,
                    action_id=action.action_id,
                    action_name_zh=action.name_zh,
                    color=action.color,
                    status="confirmed",
                    source=row.source,
                    evidence_event_ids=evidence_event_ids,
                ))
                cursor = max(cursor, end)
        if cursor < duration_seconds:
            output.append(_unobserved(student_id, cursor, duration_seconds, "unreviewed"))
    return _merge_intervals(output, max_gap_seconds=max_gap_seconds, student_order=ordered_students)


def _unobserved(student_id: str, start: float, end: float, reason: str) -> ActionInterval:
    return ActionInterval(
        student_id=student_id, start_pts_seconds=start, end_pts_seconds=end,
        action_id=UNOBSERVED_ACTION_ID, action_name_zh="未观测", color=UNOBSERVED_COLOR,
        status="unobserved", source="timeline", unobserved_reason=reason,
    )


def _merge_intervals(
    intervals: Sequence[ActionInterval],
    *,
    max_gap_seconds: float,
    student_order: Sequence[str] | None = None,
) -> list[ActionInterval]:
    output: list[ActionInterval] = []
    order = {student: index for index, student in enumerate(student_order or ())}
    for interval in sorted(
        intervals,
        key=lambda item: (order.get(item.student_id, len(order)), item.start_pts_seconds, item.end_pts_seconds, item.student_id),
    ):
        if interval.end_pts_seconds <= interval.start_pts_seconds:
            continue
        if output:
            previous = output[-1]
            if (
                previous.student_id == interval.student_id
                and previous.action_id == interval.action_id
                and previous.status == interval.status
                and interval.start_pts_seconds - previous.end_pts_seconds <= max_gap_seconds
                and previous.unobserved_reason == interval.unobserved_reason
            ):
                output[-1] = ActionInterval(
                    student_id=previous.student_id,
                    start_pts_seconds=previous.start_pts_seconds,
                    end_pts_seconds=interval.end_pts_seconds,
                    action_id=previous.action_id,
                    action_name_zh=previous.action_name_zh,
                    color=previous.color,
                    status=previous.status,
                    source=previous.source,
                    unobserved_reason=previous.unobserved_reason,
                    evidence_event_ids=tuple(dict.fromkeys(
                        (*previous.evidence_event_ids, *interval.evidence_event_ids)
                    )),
                )
                continue
        output.append(interval)
    return output


__all__ = [
    "ActionDescriptor", "ActionInterval", "ActionObservation", "ActionRegistrySnapshot", "TimelineDocument",
    "PALETTE", "UNOBSERVED_ACTION_ID", "UNOBSERVED_COLOR", "action_id", "build_timeline",
    "canonical_sha256", "colour_for_index", "evidence_quality", "normalize_phrase", "session_vocabulary",
    "session_vocabulary_digest",
]
