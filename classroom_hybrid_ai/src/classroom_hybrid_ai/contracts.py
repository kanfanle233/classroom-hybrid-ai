"""Stable, dependency-free data contracts for the hybrid video pipeline.

The contracts deliberately keep model output and pipeline decisions separate.  In
particular, ``ReviewDecision.self_reported_confidence`` is opaque provenance: it
is serialized, but no numeric meaning is assigned to it here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple, Type, TypeVar


TContract = TypeVar("TContract", bound="JsonContract")

DECISIONS = frozenset({"confirmed", "rejected", "uncertain"})
FINAL_STATUSES = DECISIONS


def _encode(value: Any) -> Any:
    """Convert nested contract values to JSON-compatible Python values."""

    if isinstance(value, JsonContract):
        return value.to_dict()
    if is_dataclass(value):
        return {item.name: _encode(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("contract mapping keys must be strings")
            result[key] = _encode(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    return value


def _as_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


def _canonical_json_value(value: Any) -> Any:
    """Normalize containers so dict/JSON round trips preserve equality."""

    if isinstance(value, JsonContract):
        return value.to_dict()
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("contract mapping keys must be strings")
            result[key] = _canonical_json_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_canonical_json_value(item) for item in value]
    return value


def _canonical_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    return _canonical_json_value(_as_mapping(value, field_name))


def _nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _unit_score(value: Any, field_name: str) -> float:
    result = _finite_number(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return result


def _frame_indices(value: Any, field_name: str) -> Tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of frame indices")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of frame indices") from exc

    normalized = []
    seen = set()
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{field_name} entries must be integers")
        if item < 0:
            raise ValueError(f"{field_name} entries must be non-negative")
        if item not in seen:
            normalized.append(item)
            seen.add(item)
    return tuple(normalized)


@dataclass(frozen=True)
class JsonContract:
    """Mixin for deterministic dict/JSON round trips."""

    schema_version: ClassVar[int] = 1

    def to_dict(self) -> Dict[str, Any]:
        value = {item.name: _encode(getattr(self, item.name)) for item in fields(self)}
        contract_schema = getattr(self, "contract_schema", None)
        if isinstance(contract_schema, str) and contract_schema:
            value.setdefault("schema_version", contract_schema)
        return value

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls: Type[TContract], data: Mapping[str, Any]) -> TContract:
        values = _as_mapping(data, cls.__name__)
        if "schema_version" in values and not any(item.name == "schema_version" for item in fields(cls)):
            values.pop("schema_version", None)
        return cls(**values)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls: Type[TContract], payload: str) -> TContract:
        if not isinstance(payload, str):
            raise TypeError("payload must be a JSON string")
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("contract JSON must decode to an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class Detection(JsonContract):
    """One spatial model detection in source-image coordinates."""

    label: str
    bbox: Tuple[float, float, float, float]
    score: Optional[float] = None
    source: str = "unknown"
    class_id: Optional[int] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_text(self.label, "label")
        _nonempty_text(self.source, "source")

        if isinstance(self.bbox, (str, bytes)) or len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly four coordinates")
        bbox = tuple(_finite_number(item, "bbox coordinate") for item in self.bbox)
        x1, y1, x2, y2 = bbox
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox must satisfy x2 >= x1 and y2 >= y1")
        object.__setattr__(self, "bbox", bbox)

        if self.score is not None:
            object.__setattr__(self, "score", _unit_score(self.score, "score"))
        if self.class_id is not None:
            if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
                raise TypeError("class_id must be an integer or None")
        object.__setattr__(
            self, "attributes", _canonical_mapping(self.attributes, "attributes")
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Detection":
        values = _as_mapping(data, "Detection")
        if "bbox" in values:
            values["bbox"] = tuple(values["bbox"])
        return cls(**values)


@dataclass(frozen=True)
class TrackObservation(JsonContract):
    """Detection and anomaly metadata associated with one persistent track."""

    track_id: str
    frame_index: int
    pts_seconds: float
    detection: Detection
    stream_id: str = "default"
    anomaly_scores: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_text(self.track_id, "track_id")
        _nonempty_text(self.stream_id, "stream_id")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        pts = _finite_number(self.pts_seconds, "pts_seconds")
        if pts < 0:
            raise ValueError("pts_seconds must be non-negative")
        object.__setattr__(self, "pts_seconds", pts)
        if not isinstance(self.detection, Detection):
            raise TypeError("detection must be a Detection")

        scores = _as_mapping(self.anomaly_scores, "anomaly_scores")
        normalized_scores: Dict[str, float] = {}
        for label, score in scores.items():
            _nonempty_text(label, "anomaly score label")
            normalized_scores[label] = _unit_score(score, f"anomaly_scores[{label!r}]")
        object.__setattr__(self, "anomaly_scores", normalized_scores)
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackObservation":
        values = _as_mapping(data, "TrackObservation")
        detection = values.get("detection")
        if isinstance(detection, Mapping):
            values["detection"] = Detection.from_dict(detection)
        return cls(**values)


@dataclass(frozen=True)
class CandidateEvent(JsonContract):
    """A small-model or rule-based candidate awaiting multimodal review."""

    event_id: str
    track_id: str
    event_type: str
    trigger_frame_index: int
    trigger_pts_seconds: float
    stream_id: str = "default"
    start_pts_seconds: Optional[float] = None
    end_pts_seconds: Optional[float] = None
    score: Optional[float] = None
    evidence_frame_indices: Tuple[int, ...] = ()
    source: str = "yolo"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_text(self.event_id, "event_id")
        _nonempty_text(self.track_id, "track_id")
        _nonempty_text(self.event_type, "event_type")
        _nonempty_text(self.stream_id, "stream_id")
        _nonempty_text(self.source, "source")
        if isinstance(self.trigger_frame_index, bool) or not isinstance(
            self.trigger_frame_index, int
        ):
            raise TypeError("trigger_frame_index must be an integer")
        if self.trigger_frame_index < 0:
            raise ValueError("trigger_frame_index must be non-negative")

        trigger = _finite_number(self.trigger_pts_seconds, "trigger_pts_seconds")
        if trigger < 0:
            raise ValueError("trigger_pts_seconds must be non-negative")
        object.__setattr__(self, "trigger_pts_seconds", trigger)

        start = trigger if self.start_pts_seconds is None else _finite_number(
            self.start_pts_seconds, "start_pts_seconds"
        )
        if start < 0 or start > trigger:
            raise ValueError("start_pts_seconds must be non-negative and no later than trigger")
        object.__setattr__(self, "start_pts_seconds", start)

        if self.end_pts_seconds is not None:
            end = _finite_number(self.end_pts_seconds, "end_pts_seconds")
            if end < trigger:
                raise ValueError("end_pts_seconds must be no earlier than trigger")
            object.__setattr__(self, "end_pts_seconds", end)

        if self.score is not None:
            object.__setattr__(self, "score", _unit_score(self.score, "score"))
        object.__setattr__(
            self,
            "evidence_frame_indices",
            _frame_indices(self.evidence_frame_indices, "evidence_frame_indices"),
        )
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata, "metadata"))

    @property
    def candidate_event_id(self) -> str:
        return self.event_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateEvent":
        values = _as_mapping(data, "CandidateEvent")
        if "evidence_frame_indices" in values:
            values["evidence_frame_indices"] = tuple(values["evidence_frame_indices"])
        allowed = {
            "event_id", "track_id", "event_type", "trigger_frame_index", "trigger_pts_seconds",
            "stream_id", "start_pts_seconds", "end_pts_seconds", "score", "evidence_frame_indices",
            "source", "metadata",
        }
        extras = {key: value for key, value in values.items() if key not in allowed}
        if extras:
            metadata = dict(values.get("metadata") or {})
            metadata["selection_audit"] = _canonical_json_value(extras)
            values["metadata"] = metadata
            values = {key: value for key, value in values.items() if key in allowed}
        return cls(**values)


@dataclass(frozen=True)
class ReviewDecision(JsonContract):
    """A schema-checked Qwen review of one candidate.

    ``self_reported_confidence`` is intentionally typed as ``Any``.  Language
    model confidence prose or numbers are retained for audit only and must not be
    interpreted as calibrated probability by fusion code.
    """

    candidate_event_id: str
    decision: str
    reviewed_event_type: Optional[str] = None
    behavior: Optional[str] = None
    rationale: str = ""
    evidence_frame_indices: Tuple[int, ...] = ()
    reviewer: str = "qwen3.5"
    schema_valid: bool = True
    self_reported_confidence: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_text(self.candidate_event_id, "candidate_event_id")
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        if self.reviewed_event_type is not None:
            _nonempty_text(self.reviewed_event_type, "reviewed_event_type")
        if self.behavior is not None:
            _nonempty_text(self.behavior, "behavior")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")
        _nonempty_text(self.reviewer, "reviewer")
        if not isinstance(self.schema_valid, bool):
            raise TypeError("schema_valid must be a boolean")
        object.__setattr__(
            self,
            "evidence_frame_indices",
            _frame_indices(self.evidence_frame_indices, "evidence_frame_indices"),
        )
        object.__setattr__(
            self,
            "self_reported_confidence",
            _canonical_json_value(self.self_reported_confidence),
        )
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewDecision":
        values = _as_mapping(data, "ReviewDecision")
        if "evidence_frame_indices" in values:
            values["evidence_frame_indices"] = tuple(values["evidence_frame_indices"])
        allowed = {
            "candidate_event_id", "decision", "reviewed_event_type", "behavior", "rationale",
            "evidence_frame_indices", "reviewer", "schema_valid", "self_reported_confidence", "metadata",
        }
        extras = {key: value for key, value in values.items() if key not in allowed}
        if extras:
            metadata = dict(values.get("metadata") or {})
            metadata["model_details"] = _canonical_json_value(extras)
            values["metadata"] = metadata
            values = {key: value for key, value in values.items() if key in allowed}
        return cls(**values)


@dataclass(frozen=True)
class FinalEvent(JsonContract):
    """Auditable fusion result preserving both candidate and review payloads."""

    event_id: str
    candidate: CandidateEvent
    status: str
    review: Optional[ReviewDecision] = None
    behavior: Optional[str] = None
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_text(self.event_id, "event_id")
        if not isinstance(self.candidate, CandidateEvent):
            raise TypeError("candidate must be a CandidateEvent")
        if self.status not in FINAL_STATUSES:
            raise ValueError(f"status must be one of {sorted(FINAL_STATUSES)}")
        if self.review is not None and not isinstance(self.review, ReviewDecision):
            raise TypeError("review must be a ReviewDecision or None")
        if self.behavior is not None:
            _nonempty_text(self.behavior, "behavior")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata, "metadata"))

    @property
    def candidate_event_id(self) -> str:
        return self.candidate.event_id

    @property
    def track_id(self) -> str:
        return self.candidate.track_id

    @property
    def event_type(self) -> str:
        return self.candidate.event_type

    @property
    def stream_id(self) -> str:
        return self.candidate.stream_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinalEvent":
        values = _as_mapping(data, "FinalEvent")
        candidate = values.get("candidate")
        review = values.get("review")
        if isinstance(candidate, Mapping):
            values["candidate"] = CandidateEvent.from_dict(candidate)
        if isinstance(review, Mapping):
            values["review"] = ReviewDecision.from_dict(review)
        return cls(**values)


__all__ = [
    "CandidateEvent",
    "DECISIONS",
    "Detection",
    "FINAL_STATUSES",
    "FinalEvent",
    "JsonContract",
    "ReviewDecision",
    "TrackObservation",
]

# Re-export the open-vocabulary contracts from the package's central public
# contract module as well as their native ``open_vocab`` module.  This keeps
# downstream consumers from depending on the internal file split.
from .open_vocab import (  # noqa: E402  (import after the dependency-free core definitions)
    ActionDescriptor,
    ActionInterval,
    ActionObservation,
    ActionRegistrySnapshot,
    TimelineDocument,
)

__all__ += [
    "ActionDescriptor",
    "ActionInterval",
    "ActionObservation",
    "ActionRegistrySnapshot",
    "TimelineDocument",
    "ArtifactRef",
    "FrontendIndex",
    "PreviewRef",
    "RESEARCH_VALIDITIES",
    "RunState",
    "RUN_STATUSES",
    "StageEvent",
    "StageState",
    "STAGE_EVENT_TYPES",
    "STAGE_STATUSES",
]


# ---------------------------------------------------------------------------
# v3 stage/runtime contracts
# ---------------------------------------------------------------------------
# These contracts intentionally live beside the model contracts.  A watcher,
# a recovery process, and a clean wheel installation must be able to read the
# run directory without importing OpenCV, torch, or either model adapter.

RUN_STATUSES = frozenset({"running", "completed", "warning", "failed", "skipped", "blocked"})
STAGE_STATUSES = RUN_STATUSES
STAGE_EVENT_TYPES = frozenset({"started", "progress", "completed", "warning", "failed", "skipped", "blocked"})
RESEARCH_VALIDITIES = frozenset({"valid", "pending", "invalid"})


def _optional_nonnegative(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, field_name)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


@dataclass(frozen=True)
class ArtifactRef(JsonContract):
    """Portable reference to an artifact; ``uri`` is relative to run_dir."""

    uri: str
    kind: str = "artifact"
    sha256: str | None = None
    bytes: int | None = None
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    contract_schema: ClassVar[str] = "classroom_hybrid_ai.artifact_ref.v1"

    def __post_init__(self) -> None:
        _nonempty_text(self.uri, "uri")
        _nonempty_text(self.kind, "kind")
        normalized_uri = self.uri.replace("\\", "/")
        if normalized_uri.startswith("/") or ":/" in normalized_uri or any(part == ".." for part in normalized_uri.split("/")):
            raise ValueError("artifact_uri_must_be_relative")
        if self.sha256 is not None:
            _nonempty_text(self.sha256, "sha256")
        if self.bytes is not None:
            if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
                raise ValueError("bytes_must_be_nonnegative_integer")
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata, "metadata"))


@dataclass(frozen=True)
class PreviewRef(ArtifactRef):
    """Portable reference to an image/video preview exposed to the watcher."""

    kind: str = "preview"
    stage_id: str = ""
    contract_schema: ClassVar[str] = "classroom_hybrid_ai.preview_ref.v1"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty_text(self.stage_id, "stage_id")


@dataclass(frozen=True)
class StageEvent(JsonContract):
    """One append-only event in ``stage_events.v1.jsonl``."""

    event_id: str
    stage_id: str
    event_type: str
    timestamp: str
    seq: int = 0
    epoch: int = 0
    progress: float = 0.0
    processed_units: int = 0
    total_units: int | None = None
    status: str = "running"
    message: str = ""
    preview_uris: Tuple[str, ...] = ()
    artifact_uris: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    contract_schema: ClassVar[str] = "classroom_hybrid_ai.stage_event.v1"

    def __post_init__(self) -> None:
        _nonempty_text(self.event_id, "event_id")
        _nonempty_text(self.stage_id, "stage_id")
        _nonempty_text(self.event_type, "event_type")
        _nonempty_text(self.timestamp, "timestamp")
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0:
            raise ValueError("stage_event_seq_must_be_nonnegative_integer")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError("stage_event_epoch_must_be_nonnegative_integer")
        if self.event_type not in STAGE_EVENT_TYPES:
            raise ValueError(f"stage_event_type_invalid:{self.event_type}")
        if self.status not in STAGE_STATUSES:
            raise ValueError(f"stage_event_status_invalid:{self.status}")
        progress = _finite_number(self.progress, "progress")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress_must_be_between_zero_and_one")
        object.__setattr__(self, "progress", progress)
        if isinstance(self.processed_units, bool) or not isinstance(self.processed_units, int) or self.processed_units < 0:
            raise ValueError("processed_units_must_be_nonnegative_integer")
        if self.total_units is not None:
            if isinstance(self.total_units, bool) or not isinstance(self.total_units, int) or self.total_units < 0:
                raise ValueError("total_units_must_be_nonnegative_integer")
        object.__setattr__(self, "preview_uris", tuple(str(item) for item in self.preview_uris))
        object.__setattr__(self, "artifact_uris", tuple(str(item) for item in self.artifact_uris))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata, "metadata"))


@dataclass(frozen=True)
class StageState(JsonContract):
    """Latest durable state for one of the fixed S0-S7 stages."""

    stage_id: str
    name: str
    weight: float
    status: str = "pending"
    progress: float = 0.0
    processed_units: int = 0
    total_units: int | None = None
    started_at: str | None = None
    updated_at: str | None = None
    ended_at: str | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    preview_uris: Tuple[str, ...] = ()
    artifact_uris: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    model_loaded: bool = False
    model_pid: int | None = None
    unload_ack: bool | None = None
    exit_code: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    epoch: int = 0
    contract_schema: ClassVar[str] = "classroom_hybrid_ai.stage_state.v1"

    def __post_init__(self) -> None:
        _nonempty_text(self.stage_id, "stage_id")
        _nonempty_text(self.name, "name")
        weight = _finite_number(self.weight, "weight")
        if weight < 0:
            raise ValueError("stage_weight_must_be_nonnegative")
        object.__setattr__(self, "weight", weight)
        if self.status not in STAGE_STATUSES and self.status != "pending":
            raise ValueError(f"stage_status_invalid:{self.status}")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError("stage_state_epoch_must_be_nonnegative_integer")
        progress = _finite_number(self.progress, "progress")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("stage_progress_must_be_between_zero_and_one")
        object.__setattr__(self, "progress", progress)
        if isinstance(self.processed_units, bool) or not isinstance(self.processed_units, int) or self.processed_units < 0:
            raise ValueError("processed_units_must_be_nonnegative_integer")
        if self.total_units is not None and (isinstance(self.total_units, bool) or not isinstance(self.total_units, int) or self.total_units < 0):
            raise ValueError("total_units_must_be_nonnegative_integer")
        for name in ("elapsed_seconds", "eta_seconds"):
            object.__setattr__(self, name, _optional_nonnegative(getattr(self, name), name))
        if self.model_pid is not None and (isinstance(self.model_pid, bool) or not isinstance(self.model_pid, int) or self.model_pid < 0):
            raise ValueError("model_pid_must_be_nonnegative_integer")
        object.__setattr__(self, "preview_uris", tuple(str(item) for item in self.preview_uris))
        object.__setattr__(self, "artifact_uris", tuple(str(item) for item in self.artifact_uris))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageState":
        values = _as_mapping(data, "StageState")
        values.pop("schema_version", None)
        for key in ("preview_uris", "artifact_uris", "warnings", "errors"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class RunState(JsonContract):
    """Atomic top-level state consumed by status/watch and recovery."""

    run_id: str
    video_id: str
    status: str = "running"
    current_stage: str = "S0"
    overall_progress: float | None = 0.0
    current_stage_progress: float | None = 0.0
    processed_units: int = 0
    total_units: int | None = None
    started_at: str = ""
    updated_at: str = ""
    eta_seconds: float | None = None
    stages: Mapping[str, StageState | Mapping[str, Any]] = field(default_factory=dict)
    previews: Tuple[str, ...] = ()
    artifacts: Tuple[ArtifactRef | Mapping[str, Any], ...] = ()
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    model_loaded: bool = False
    model_pid: int | None = None
    unload_ack: bool | None = None
    exit_code: int | None = None
    research_validity: str = "pending"
    epoch: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    contract_schema: ClassVar[str] = "classroom_hybrid_ai.run_state.v1"

    def __post_init__(self) -> None:
        _nonempty_text(self.run_id, "run_id")
        _nonempty_text(self.video_id, "video_id")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"run_status_invalid:{self.status}")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError("run_state_epoch_must_be_nonnegative_integer")
        if self.overall_progress is not None:
            progress = _finite_number(self.overall_progress, "overall_progress")
            if not 0.0 <= progress <= 100.0:
                raise ValueError("overall_progress_must_be_between_zero_and_hundred")
            object.__setattr__(self, "overall_progress", progress)
        if self.current_stage_progress is not None:
            progress = _finite_number(self.current_stage_progress, "current_stage_progress")
            if not 0.0 <= progress <= 100.0:
                raise ValueError("current_stage_progress_must_be_between_zero_and_hundred")
            object.__setattr__(self, "current_stage_progress", progress)
        if self.research_validity not in RESEARCH_VALIDITIES:
            raise ValueError(f"research_validity_invalid:{self.research_validity}")
        object.__setattr__(self, "stages", {
            str(key): value if isinstance(value, StageState) else StageState.from_dict(value)
            for key, value in self.stages.items()
        })
        object.__setattr__(self, "artifacts", tuple(
            item if isinstance(item, ArtifactRef) else ArtifactRef.from_dict(item)
            for item in self.artifacts
        ))
        object.__setattr__(self, "previews", tuple(str(item) for item in self.previews))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunState":
        values = _as_mapping(data, "RunState")
        values.pop("schema_version", None)
        return cls(**values)


@dataclass(frozen=True)
class FrontendIndex(JsonContract):
    """Read-only frontend entry point; all URIs are relative to the run root."""

    run_uri: str = "run_state.v1.json"
    stages_uri: str = "stage_events.v1.jsonl"
    students_uri: str = "students.jsonl"
    actions_uri: str = "action_observations.jsonl"
    intervals_uri: str = "action_intervals.jsonl"
    timeline_uri: str = "timeline.v2.json"
    metrics_uri: str = "metrics.v1.json"
    video_uris: Mapping[str, str] = field(default_factory=dict)
    preview_uris: Tuple[str, ...] = ()
    evidence_uri: str = "evidence/"
    overlay_chunks: Tuple[str, ...] = ()
    students: Tuple[Mapping[str, Any], ...] = ()
    actions: Tuple[Mapping[str, Any], ...] = ()
    endpoints: Mapping[str, str] = field(default_factory=dict)
    contract_sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    contract_schema: ClassVar[str] = "classroom_hybrid_ai.frontend_index.v1"

    def __post_init__(self) -> None:
        for name in ("run_uri", "stages_uri", "students_uri", "actions_uri", "intervals_uri", "timeline_uri", "metrics_uri", "evidence_uri"):
            value = getattr(self, name)
            _nonempty_text(value, name)
            if value.startswith("/") or ":\\" in value or ":/" in value:
                raise ValueError(f"{name}_must_be_relative")
        object.__setattr__(self, "video_uris", _canonical_mapping(self.video_uris, "video_uris"))
        object.__setattr__(self, "overlay_chunks", tuple(str(item) for item in self.overlay_chunks))
        object.__setattr__(self, "students", tuple(_canonical_mapping(item, "students") for item in self.students))
        object.__setattr__(self, "actions", tuple(_canonical_mapping(item, "actions") for item in self.actions))
        object.__setattr__(self, "endpoints", _canonical_mapping(self.endpoints, "endpoints"))
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrontendIndex":
        values = _as_mapping(data, "FrontendIndex")
        values.pop("schema_version", None)
        values["preview_uris"] = tuple(values.get("preview_uris", ()))
        values["overlay_chunks"] = tuple(values.get("overlay_chunks", ()))
        values["students"] = tuple(values.get("students", ()))
        values["actions"] = tuple(values.get("actions", ()))
        return cls(**values)
