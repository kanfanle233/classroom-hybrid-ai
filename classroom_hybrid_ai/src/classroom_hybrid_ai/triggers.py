"""Sampling, hysteresis, cooldown, and event de-duplication primitives."""

from __future__ import annotations

import math
import threading
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Hashable, Optional, Tuple

from .contracts import CandidateEvent, TrackObservation


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class StrideSampler:
    """Deterministic frame-index sampler.

    With ``stride=10`` and ``offset=0``, frames 0, 10, 20, ... are selected.
    """

    stride: int
    offset: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.stride, bool) or not isinstance(self.stride, int):
            raise TypeError("stride must be an integer")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise TypeError("offset must be an integer")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")

    def should_sample(self, frame_index: int) -> bool:
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index must be an integer")
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        return frame_index >= self.offset and (frame_index - self.offset) % self.stride == 0


class PTSSampler:
    """Presentation-timestamp sampler that does not accumulate scheduling drift.

    State is kept independently per stream.  A backwards PTS is treated as a
    seek/discontinuity and starts a new schedule for that stream.
    """

    def __init__(
        self,
        interval_seconds: float,
        *,
        sample_first: bool = True,
        reset_on_regression: bool = True,
        epsilon: float = 1e-9,
    ) -> None:
        interval = _finite(interval_seconds, "interval_seconds")
        if interval <= 0:
            raise ValueError("interval_seconds must be positive")
        if not isinstance(sample_first, bool):
            raise TypeError("sample_first must be a boolean")
        if not isinstance(reset_on_regression, bool):
            raise TypeError("reset_on_regression must be a boolean")
        epsilon_value = _finite(epsilon, "epsilon")
        if epsilon_value < 0:
            raise ValueError("epsilon must be non-negative")

        self.interval_seconds = interval
        self.sample_first = sample_first
        self.reset_on_regression = reset_on_regression
        self.epsilon = epsilon_value
        self._next_due: Dict[str, float] = {}
        self._last_pts: Dict[str, float] = {}
        self._lock = threading.RLock()

    def should_sample(self, pts_seconds: float, stream_id: str = "default") -> bool:
        pts = _finite(pts_seconds, "pts_seconds")
        if pts < 0:
            raise ValueError("pts_seconds must be non-negative")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stream_id must be a non-empty string")

        with self._lock:
            last = self._last_pts.get(stream_id)
            if last is not None and pts + self.epsilon < last:
                if not self.reset_on_regression:
                    raise ValueError("pts_seconds regressed for stream")
                self._next_due.pop(stream_id, None)

            self._last_pts[stream_id] = pts
            next_due = self._next_due.get(stream_id)
            if next_due is None:
                self._next_due[stream_id] = pts + self.interval_seconds
                return self.sample_first

            if pts + self.epsilon < next_due:
                return False

            # Advance from the existing deadline, rather than from the actual
            # arrival time, so late frames do not shift all future samples.
            missed = math.floor((pts - next_due + self.epsilon) / self.interval_seconds)
            self._next_due[stream_id] = next_due + (missed + 1) * self.interval_seconds
            return True

    def reset(self, stream_id: Optional[str] = None) -> None:
        with self._lock:
            if stream_id is None:
                self._next_due.clear()
                self._last_pts.clear()
            else:
                self._next_due.pop(stream_id, None)
                self._last_pts.pop(stream_id, None)


class PeriodicSampler:
    """Convenience wrapper for frame-stride or PTS-based periodic sampling.

    If a PTS interval is configured and a PTS is supplied, the PTS schedule is
    used.  The frame stride is a deterministic fallback for sources without PTS.
    """

    def __init__(
        self,
        *,
        frame_stride: Optional[int] = None,
        pts_interval_seconds: Optional[float] = None,
        frame_offset: int = 0,
        sample_first_pts: bool = True,
    ) -> None:
        if frame_stride is None and pts_interval_seconds is None:
            raise ValueError("configure frame_stride, pts_interval_seconds, or both")
        self.stride_sampler = (
            StrideSampler(frame_stride, frame_offset) if frame_stride is not None else None
        )
        self.pts_sampler = (
            PTSSampler(pts_interval_seconds, sample_first=sample_first_pts)
            if pts_interval_seconds is not None
            else None
        )

    def should_sample(
        self,
        frame_index: int,
        pts_seconds: Optional[float] = None,
        stream_id: str = "default",
    ) -> bool:
        if self.pts_sampler is not None and pts_seconds is not None:
            return self.pts_sampler.should_sample(pts_seconds, stream_id)
        if self.stride_sampler is None:
            raise ValueError("pts_seconds is required when no frame_stride fallback is configured")
        return self.stride_sampler.should_sample(frame_index)

    def reset(self, stream_id: Optional[str] = None) -> None:
        if self.pts_sampler is not None:
            self.pts_sampler.reset(stream_id)


@dataclass(frozen=True)
class ThresholdTransition:
    """Result of one k-of-m hysteresis update."""

    active: bool
    started: bool
    ended: bool
    high_votes: int
    low_votes: int
    observations: int
    score: float


class KOfMThresholdTrigger:
    """Per-key k-of-m hysteresis trigger.

    An inactive key starts after at least ``k`` of its latest ``m`` scores meet
    ``high_threshold``.  An active key ends after at least ``k`` scores meet
    ``low_threshold``.  Windows are cleared at transitions so stale low votes
    cannot immediately cancel a newly started event.
    """

    def __init__(
        self,
        *,
        k: int,
        m: int,
        high_threshold: float,
        low_threshold: float,
        require_full_window: bool = False,
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if isinstance(m, bool) or not isinstance(m, int):
            raise TypeError("m must be an integer")
        if m <= 0 or k <= 0 or k > m:
            raise ValueError("k and m must satisfy 1 <= k <= m")
        high = _unit(high_threshold, "high_threshold")
        low = _unit(low_threshold, "low_threshold")
        if low >= high:
            raise ValueError("low_threshold must be lower than high_threshold")
        if not isinstance(require_full_window, bool):
            raise TypeError("require_full_window must be a boolean")

        self.k = k
        self.m = m
        self.high_threshold = high
        self.low_threshold = low
        self.require_full_window = require_full_window
        self._windows: Dict[Hashable, Deque[float]] = {}
        self._active: Dict[Hashable, bool] = {}
        self._lock = threading.RLock()

    def update(self, key: Hashable, score: float) -> ThresholdTransition:
        normalized_score = _unit(score, "score")
        with self._lock:
            window = self._windows.setdefault(key, deque(maxlen=self.m))
            window.append(normalized_score)
            active = self._active.get(key, False)
            enough = len(window) == self.m if self.require_full_window else len(window) >= self.k
            high_votes = sum(item >= self.high_threshold for item in window)
            low_votes = sum(item <= self.low_threshold for item in window)
            started = bool(not active and enough and high_votes >= self.k)
            ended = bool(active and enough and low_votes >= self.k)

            observations = len(window)
            if started:
                active = True
                self._active[key] = True
                window.clear()
            elif ended:
                active = False
                self._active[key] = False
                window.clear()
            else:
                self._active[key] = active

            return ThresholdTransition(
                active=active,
                started=started,
                ended=ended,
                high_votes=high_votes,
                low_votes=low_votes,
                observations=observations,
                score=normalized_score,
            )

    def is_active(self, key: Hashable) -> bool:
        with self._lock:
            return self._active.get(key, False)

    def reset(self, key: Optional[Hashable] = None) -> None:
        with self._lock:
            if key is None:
                self._windows.clear()
                self._active.clear()
            else:
                self._windows.pop(key, None)
                self._active.pop(key, None)


class CooldownGate:
    """Allow at most one accepted trigger per key in a PTS interval."""

    def __init__(self, cooldown_seconds: float) -> None:
        cooldown = _finite(cooldown_seconds, "cooldown_seconds")
        if cooldown < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        self.cooldown_seconds = cooldown
        self._last_accepted: Dict[Hashable, float] = {}
        self._lock = threading.RLock()

    def allow(self, key: Hashable, pts_seconds: float) -> bool:
        pts = _finite(pts_seconds, "pts_seconds")
        if pts < 0:
            raise ValueError("pts_seconds must be non-negative")
        with self._lock:
            last = self._last_accepted.get(key)
            if last is not None and pts - last < self.cooldown_seconds:
                return False
            self._last_accepted[key] = pts
            return True

    def remaining(self, key: Hashable, pts_seconds: float) -> float:
        pts = _finite(pts_seconds, "pts_seconds")
        with self._lock:
            last = self._last_accepted.get(key)
            if last is None:
                return 0.0
            return max(0.0, self.cooldown_seconds - max(0.0, pts - last))

    def reset(self, key: Optional[Hashable] = None) -> None:
        with self._lock:
            if key is None:
                self._last_accepted.clear()
            else:
                self._last_accepted.pop(key, None)


@dataclass(frozen=True)
class DeduplicationResult:
    accepted: bool
    canonical_event_id: str
    reason: str


class EventDeduplicator:
    """Suppress repeated event IDs and nearby events for the same track/type."""

    def __init__(self, dedup_window_seconds: float, *, max_history: int = 10000) -> None:
        window = _finite(dedup_window_seconds, "dedup_window_seconds")
        if window < 0:
            raise ValueError("dedup_window_seconds must be non-negative")
        if isinstance(max_history, bool) or not isinstance(max_history, int):
            raise TypeError("max_history must be an integer")
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self.dedup_window_seconds = window
        self.max_history = max_history
        self._history: "OrderedDict[str, CandidateEvent]" = OrderedDict()
        self._latest_by_key: Dict[Tuple[str, str, str], CandidateEvent] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(event: CandidateEvent) -> Tuple[str, str, str]:
        return (event.stream_id, event.track_id, event.event_type)

    @staticmethod
    def _bounds(event: CandidateEvent) -> Tuple[float, float]:
        start = event.start_pts_seconds
        assert start is not None  # CandidateEvent normalizes it in __post_init__.
        end = event.end_pts_seconds
        if end is None:
            end = event.trigger_pts_seconds
        return start, end

    @classmethod
    def _gap(cls, left: CandidateEvent, right: CandidateEvent) -> float:
        left_start, left_end = cls._bounds(left)
        right_start, right_end = cls._bounds(right)
        if right_start > left_end:
            return right_start - left_end
        if left_start > right_end:
            return left_start - right_end
        return 0.0

    def accept(self, event: CandidateEvent) -> DeduplicationResult:
        if not isinstance(event, CandidateEvent):
            raise TypeError("event must be a CandidateEvent")
        with self._lock:
            repeated = self._history.get(event.event_id)
            if repeated is not None:
                return DeduplicationResult(False, repeated.event_id, "same_event_id")

            key = self._key(event)
            previous = self._latest_by_key.get(key)
            if previous is not None and self._gap(previous, event) <= self.dedup_window_seconds:
                return DeduplicationResult(False, previous.event_id, "same_track_type_window")

            self._history[event.event_id] = event
            self._latest_by_key[key] = event
            while len(self._history) > self.max_history:
                old_id, old_event = self._history.popitem(last=False)
                old_key = self._key(old_event)
                if self._latest_by_key.get(old_key) is old_event:
                    self._latest_by_key.pop(old_key, None)
                # ``old_id`` is named for readability in debugger snapshots.
                del old_id
            return DeduplicationResult(True, event.event_id, "accepted")

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._latest_by_key.clear()


EventIdFactory = Callable[[TrackObservation, str], str]


def _default_event_id(observation: TrackObservation, event_type: str) -> str:
    return f"{observation.stream_id}:{observation.track_id}:{event_type}:{uuid.uuid4().hex}"


class TriggerEngine:
    """Compose hysteresis, cooldown, and de-duplication into candidates."""

    def __init__(
        self,
        threshold: KOfMThresholdTrigger,
        *,
        cooldown: Optional[CooldownGate] = None,
        deduplicator: Optional[EventDeduplicator] = None,
        event_id_factory: Optional[EventIdFactory] = None,
    ) -> None:
        if not isinstance(threshold, KOfMThresholdTrigger):
            raise TypeError("threshold must be a KOfMThresholdTrigger")
        self.threshold = threshold
        self.cooldown = cooldown
        self.deduplicator = deduplicator
        self.event_id_factory = event_id_factory or _default_event_id
        self._evidence: Dict[
            Tuple[str, str, str], Deque[Tuple[int, float, float]]
        ] = {}
        self._lock = threading.RLock()

    def update(
        self,
        observation: TrackObservation,
        event_type: str,
        score: Optional[float] = None,
    ) -> Optional[CandidateEvent]:
        if not isinstance(observation, TrackObservation):
            raise TypeError("observation must be a TrackObservation")
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        if score is None:
            if event_type not in observation.anomaly_scores:
                raise KeyError(f"observation has no anomaly score for {event_type!r}")
            score = observation.anomaly_scores[event_type]
        normalized_score = _unit(score, "score")
        key = (observation.stream_id, observation.track_id, event_type)

        with self._lock:
            evidence = self._evidence.setdefault(key, deque(maxlen=self.threshold.m))
            evidence.append(
                (observation.frame_index, observation.pts_seconds, normalized_score)
            )
            transition = self.threshold.update(key, normalized_score)
            if transition.ended:
                evidence.clear()
                return None
            if not transition.started:
                return None

            high_evidence = [
                item for item in evidence if item[2] >= self.threshold.high_threshold
            ]
            if not high_evidence:
                high_evidence = [
                    (observation.frame_index, observation.pts_seconds, normalized_score)
                ]
            evidence.clear()

            if self.cooldown is not None and not self.cooldown.allow(
                key, observation.pts_seconds
            ):
                return None

            event_id = self.event_id_factory(observation, event_type)
            candidate = CandidateEvent(
                event_id=event_id,
                stream_id=observation.stream_id,
                track_id=observation.track_id,
                event_type=event_type,
                trigger_frame_index=observation.frame_index,
                trigger_pts_seconds=observation.pts_seconds,
                start_pts_seconds=min(item[1] for item in high_evidence),
                score=normalized_score,
                evidence_frame_indices=tuple(item[0] for item in high_evidence),
                source=observation.detection.source,
                metadata={
                    "trigger": "k_of_m",
                    "k": self.threshold.k,
                    "m": self.threshold.m,
                    "high_threshold": self.threshold.high_threshold,
                    "low_threshold": self.threshold.low_threshold,
                    "high_votes": transition.high_votes,
                },
            )
            if self.deduplicator is not None:
                result = self.deduplicator.accept(candidate)
                if not result.accepted:
                    return None
            return candidate

    def reset(self) -> None:
        with self._lock:
            self.threshold.reset()
            if self.cooldown is not None:
                self.cooldown.reset()
            if self.deduplicator is not None:
                self.deduplicator.reset()
            self._evidence.clear()

    def reset_track(self, stream_id: str, track_id: str) -> None:
        """Discard every partially filled window for one interrupted track."""

        with self._lock:
            matching = [
                key for key in self._evidence
                if key[0] == stream_id and key[1] == track_id
            ]
            for key in matching:
                self.threshold.reset(key)
                if self.cooldown is not None:
                    self.cooldown.reset(key)
                self._evidence.pop(key, None)


__all__ = [
    "CooldownGate",
    "DeduplicationResult",
    "EventDeduplicator",
    "KOfMThresholdTrigger",
    "PTSSampler",
    "PeriodicSampler",
    "StrideSampler",
    "ThresholdTransition",
    "TriggerEngine",
]
