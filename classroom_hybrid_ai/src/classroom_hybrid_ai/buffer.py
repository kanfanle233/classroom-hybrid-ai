"""Track-scoped metadata windows for pre/post-event review evidence."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from .contracts import CandidateEvent, TrackObservation


TrackKey = Tuple[str, str]


def _duration(value: float, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0 or (not allow_zero and result == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class TrackWindow:
    """A time-bounded, track-specific view over retained observations."""

    stream_id: str
    track_id: str
    center_pts_seconds: float
    start_pts_seconds: float
    end_pts_seconds: float
    observations: Tuple[TrackObservation, ...]
    history_complete: bool
    future_complete: bool

    @property
    def complete(self) -> bool:
        return self.history_complete and self.future_complete

    @property
    def frame_indices(self) -> Tuple[int, ...]:
        return tuple(item.frame_index for item in self.observations)


@dataclass(frozen=True)
class EventWindow:
    """A candidate paired with the retained observations sent for review."""

    candidate: CandidateEvent
    window: TrackWindow


class TrackWindowBuffer:
    """Bounded metadata buffer keyed by ``(stream_id, track_id)``.

    ``append`` records observations.  Call ``advance`` once after all track
    observations for a video timestamp have been appended; this explicit stream
    watermark prevents a pending event from being finalized halfway through a
    multi-track frame.
    """

    def __init__(
        self,
        *,
        before_seconds: float = 2.0,
        after_seconds: float = 2.0,
        retention_seconds: Optional[float] = None,
        max_observations_per_track: int = 4096,
        max_completed_events: int = 1000,
    ) -> None:
        before = _duration(before_seconds, "before_seconds")
        after = _duration(after_seconds, "after_seconds")
        minimum_retention = before + after
        if retention_seconds is None:
            retention = max(minimum_retention, 1.0) + 1.0
        else:
            retention = _duration(retention_seconds, "retention_seconds", allow_zero=False)
        if retention + 1e-12 < minimum_retention:
            raise ValueError(
                "retention_seconds must be at least before_seconds + after_seconds"
            )
        if isinstance(max_observations_per_track, bool) or not isinstance(
            max_observations_per_track, int
        ):
            raise TypeError("max_observations_per_track must be an integer")
        if max_observations_per_track <= 0:
            raise ValueError("max_observations_per_track must be positive")
        if isinstance(max_completed_events, bool) or not isinstance(
            max_completed_events, int
        ):
            raise TypeError("max_completed_events must be an integer")
        if max_completed_events <= 0:
            raise ValueError("max_completed_events must be positive")

        self.before_seconds = before
        self.after_seconds = after
        self.retention_seconds = retention
        self.max_observations_per_track = max_observations_per_track
        self.max_completed_events = max_completed_events

        self._tracks: Dict[TrackKey, Deque[TrackObservation]] = {}
        self._last_seen: Dict[TrackKey, Tuple[float, int]] = {}
        self._pruned_through: Dict[TrackKey, float] = {}
        self._watermarks: Dict[str, float] = {}
        self._pending: "OrderedDict[str, CandidateEvent]" = OrderedDict()
        self._completed: "OrderedDict[str, EventWindow]" = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _key(stream_id: str, track_id: str) -> TrackKey:
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stream_id must be a non-empty string")
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("track_id must be a non-empty string")
        return stream_id, track_id

    def append(self, observation: TrackObservation) -> None:
        if not isinstance(observation, TrackObservation):
            raise TypeError("observation must be a TrackObservation")
        key = self._key(observation.stream_id, observation.track_id)
        with self._lock:
            last_seen = self._last_seen.get(key)
            if last_seen is not None:
                previous_pts, previous_frame = last_seen
                if observation.pts_seconds < previous_pts:
                    raise ValueError("track observations must have non-decreasing PTS")
                if observation.frame_index <= previous_frame:
                    raise ValueError("track frame indices must be strictly increasing")
            items = self._tracks.setdefault(key, deque())
            items.append(observation)
            self._last_seen[key] = (observation.pts_seconds, observation.frame_index)
            while len(items) > self.max_observations_per_track:
                removed = items.popleft()
                self._record_pruned(key, removed.pts_seconds)
            self._prune_locked(observation.stream_id)

    # Familiar alias for callers that treat this as a ring buffer.
    add = append

    def advance(self, stream_id: str, pts_seconds: float) -> None:
        """Advance a stream watermark after one complete video timestamp/frame."""

        self._key(stream_id, "_watermark")
        pts = _duration(pts_seconds, "pts_seconds")
        with self._lock:
            previous = self._watermarks.get(stream_id)
            if previous is not None and pts < previous:
                raise ValueError("stream watermark must be non-decreasing")
            self._watermarks[stream_id] = pts
            self._finalize_ready_locked(stream_id)
            self._prune_locked(stream_id)

    def get_window(
        self,
        track_id: str,
        center_pts_seconds: float,
        *,
        stream_id: str = "default",
        before_seconds: Optional[float] = None,
        after_seconds: Optional[float] = None,
    ) -> TrackWindow:
        key = self._key(stream_id, track_id)
        center = _duration(center_pts_seconds, "center_pts_seconds")
        before = (
            self.before_seconds
            if before_seconds is None
            else _duration(before_seconds, "before_seconds")
        )
        after = (
            self.after_seconds
            if after_seconds is None
            else _duration(after_seconds, "after_seconds")
        )
        with self._lock:
            return self._window_locked(key, center, before, after)

    # Short alias used by some orchestration code.
    window = get_window

    def register_event(self, candidate: CandidateEvent) -> TrackWindow:
        """Register a candidate and return its currently available window."""

        if not isinstance(candidate, CandidateEvent):
            raise TypeError("candidate must be a CandidateEvent")
        with self._lock:
            completed = self._completed.get(candidate.event_id)
            if completed is not None:
                if completed.candidate != candidate:
                    raise ValueError("event_id is already registered with different content")
                return completed.window
            existing = self._pending.get(candidate.event_id)
            if existing is not None and existing != candidate:
                raise ValueError("event_id is already registered with different content")
            self._pending[candidate.event_id] = candidate
            self._finalize_ready_locked(candidate.stream_id)
            completed = self._completed.get(candidate.event_id)
            if completed is not None:
                return completed.window
            return self._window_locked(
                self._key(candidate.stream_id, candidate.track_id),
                candidate.trigger_pts_seconds,
                self.before_seconds,
                self.after_seconds,
            )

    def get_event_window(self, event_id: str) -> Optional[EventWindow]:
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be a non-empty string")
        with self._lock:
            completed = self._completed.get(event_id)
            if completed is not None:
                return completed
            candidate = self._pending.get(event_id)
            if candidate is None:
                return None
            return EventWindow(
                candidate,
                self._window_locked(
                    self._key(candidate.stream_id, candidate.track_id),
                    candidate.trigger_pts_seconds,
                    self.before_seconds,
                    self.after_seconds,
                ),
            )

    def pop_ready_windows(self) -> Tuple[EventWindow, ...]:
        with self._lock:
            ready = tuple(self._completed.values())
            self._completed.clear()
            return ready

    @property
    def pending_event_ids(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._pending.keys())

    def track_count(self) -> int:
        with self._lock:
            return len(self._tracks)

    def clear(self, *, stream_id: Optional[str] = None) -> None:
        with self._lock:
            if stream_id is None:
                self._tracks.clear()
                self._last_seen.clear()
                self._pruned_through.clear()
                self._watermarks.clear()
                self._pending.clear()
                self._completed.clear()
                return

            keys = [key for key in self._tracks if key[0] == stream_id]
            for key in keys:
                self._tracks.pop(key, None)
                self._last_seen.pop(key, None)
                self._pruned_through.pop(key, None)
            self._watermarks.pop(stream_id, None)
            for event_id, candidate in tuple(self._pending.items()):
                if candidate.stream_id == stream_id:
                    self._pending.pop(event_id, None)
            for event_id, event_window in tuple(self._completed.items()):
                if event_window.candidate.stream_id == stream_id:
                    self._completed.pop(event_id, None)

    def _window_locked(
        self,
        key: TrackKey,
        center: float,
        before: float,
        after: float,
    ) -> TrackWindow:
        start = max(0.0, center - before)
        end = center + after
        items = self._tracks.get(key, ())
        observations = tuple(
            item for item in items if start <= item.pts_seconds <= end
        )
        pruned_through = self._pruned_through.get(key)
        history_complete = pruned_through is None or pruned_through < start
        watermark = self._watermarks.get(key[0])
        future_complete = watermark is not None and watermark >= end
        return TrackWindow(
            stream_id=key[0],
            track_id=key[1],
            center_pts_seconds=center,
            start_pts_seconds=start,
            end_pts_seconds=end,
            observations=observations,
            history_complete=history_complete,
            future_complete=future_complete,
        )

    def _finalize_ready_locked(self, stream_id: str) -> None:
        watermark = self._watermarks.get(stream_id)
        if watermark is None:
            return
        for event_id, candidate in tuple(self._pending.items()):
            if candidate.stream_id != stream_id:
                continue
            if watermark < candidate.trigger_pts_seconds + self.after_seconds:
                continue
            event_window = EventWindow(
                candidate=candidate,
                window=self._window_locked(
                    self._key(candidate.stream_id, candidate.track_id),
                    candidate.trigger_pts_seconds,
                    self.before_seconds,
                    self.after_seconds,
                ),
            )
            self._completed[event_id] = event_window
            self._pending.pop(event_id, None)
            while len(self._completed) > self.max_completed_events:
                self._completed.popitem(last=False)

    def _record_pruned(self, key: TrackKey, pts_seconds: float) -> None:
        current = self._pruned_through.get(key)
        if current is None or pts_seconds > current:
            self._pruned_through[key] = pts_seconds

    def _prune_locked(self, stream_id: str) -> None:
        watermark = self._watermarks.get(stream_id)
        if watermark is None:
            return
        cutoff = watermark - self.retention_seconds
        if cutoff <= 0:
            return
        for key, items in tuple(self._tracks.items()):
            if key[0] != stream_id:
                continue
            while items and items[0].pts_seconds < cutoff:
                removed = items.popleft()
                self._record_pruned(key, removed.pts_seconds)
            if not items:
                self._tracks.pop(key, None)


# Explicit alias for users who prefer the data-structure name.
TrackRingBuffer = TrackWindowBuffer


__all__ = [
    "EventWindow",
    "TrackRingBuffer",
    "TrackWindow",
    "TrackWindowBuffer",
]
