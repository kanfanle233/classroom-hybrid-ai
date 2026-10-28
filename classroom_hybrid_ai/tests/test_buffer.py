from __future__ import annotations

import unittest

from classroom_hybrid_ai.buffer import TrackWindowBuffer
from classroom_hybrid_ai.contracts import CandidateEvent, Detection, TrackObservation


def observation(
    frame: int,
    pts: float,
    *,
    track_id: str = "student-1",
    stream_id: str = "camera-1",
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        stream_id=stream_id,
        frame_index=frame,
        pts_seconds=pts,
        detection=Detection(
            label="student",
            bbox=(frame, 0, frame + 10, 20),
            score=0.9,
            source="yolo26",
            attributes={"seat": "A1"},
        ),
        metadata={"source_frame": f"frame-{frame}"},
    )


def candidate(
    event_id: str = "event-1",
    *,
    track_id: str = "student-1",
    stream_id: str = "camera-1",
    pts: float = 3.0,
) -> CandidateEvent:
    return CandidateEvent(
        event_id=event_id,
        track_id=track_id,
        stream_id=stream_id,
        event_type="leave_seat",
        trigger_frame_index=int(pts),
        trigger_pts_seconds=pts,
    )


class TrackWindowBufferTests(unittest.TestCase):
    def test_pending_event_collects_pre_and_post_window_metadata(self) -> None:
        buffer = TrackWindowBuffer(before_seconds=2.0, after_seconds=2.0)
        for frame in range(4):
            buffer.append(observation(frame, float(frame)))
            buffer.advance("camera-1", float(frame))

        initial = buffer.register_event(candidate())
        self.assertEqual(initial.frame_indices, (1, 2, 3))
        self.assertFalse(initial.future_complete)
        self.assertEqual(buffer.pending_event_ids, ("event-1",))

        buffer.append(observation(4, 4.0))
        buffer.advance("camera-1", 4.0)
        self.assertEqual(buffer.pop_ready_windows(), ())

        buffer.append(observation(5, 5.0))
        buffer.advance("camera-1", 5.0)
        ready = buffer.pop_ready_windows()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].candidate.event_id, "event-1")
        self.assertEqual(ready[0].window.frame_indices, (1, 2, 3, 4, 5))
        self.assertTrue(ready[0].window.complete)
        self.assertEqual(
            ready[0].window.observations[-1].metadata["source_frame"], "frame-5"
        )

    def test_tracks_and_streams_are_isolated(self) -> None:
        buffer = TrackWindowBuffer(before_seconds=1.0, after_seconds=1.0)
        buffer.append(observation(10, 1.0, track_id="a", stream_id="cam-a"))
        buffer.append(observation(10, 1.0, track_id="b", stream_id="cam-a"))
        buffer.append(observation(10, 1.0, track_id="a", stream_id="cam-b"))
        buffer.advance("cam-a", 1.0)
        buffer.advance("cam-b", 1.0)

        window = buffer.get_window("a", 1.0, stream_id="cam-a")
        self.assertEqual(len(window.observations), 1)
        self.assertEqual(window.observations[0].track_id, "a")
        self.assertEqual(window.observations[0].stream_id, "cam-a")
        self.assertEqual(buffer.track_count(), 3)

    def test_explicit_watermark_waits_until_all_tracks_for_frame_are_added(self) -> None:
        buffer = TrackWindowBuffer(before_seconds=0.0, after_seconds=1.0)
        buffer.append(observation(1, 1.0, track_id="b"))
        buffer.advance("camera-1", 1.0)
        buffer.register_event(candidate(track_id="b", pts=1.0))

        buffer.append(observation(2, 2.0, track_id="a"))
        self.assertEqual(buffer.pop_ready_windows(), ())
        buffer.append(observation(2, 2.0, track_id="b"))
        buffer.advance("camera-1", 2.0)

        ready = buffer.pop_ready_windows()
        self.assertEqual(ready[0].window.frame_indices, (1, 2))

    def test_retention_pruning_marks_old_history_incomplete(self) -> None:
        buffer = TrackWindowBuffer(
            before_seconds=1.0,
            after_seconds=1.0,
            retention_seconds=2.0,
        )
        for frame in range(3):
            buffer.append(observation(frame, float(frame)))
        buffer.advance("camera-1", 5.0)

        old = buffer.get_window("student-1", 1.0, stream_id="camera-1")
        self.assertEqual(old.observations, ())
        self.assertFalse(old.history_complete)
        self.assertTrue(old.future_complete)

    def test_count_bound_preserves_latest_observations_and_reports_loss(self) -> None:
        buffer = TrackWindowBuffer(
            before_seconds=2.0,
            after_seconds=0.0,
            retention_seconds=2.0,
            max_observations_per_track=2,
        )
        for frame in range(3):
            buffer.append(observation(frame, float(frame)))
        buffer.advance("camera-1", 2.0)
        window = buffer.get_window("student-1", 2.0, stream_id="camera-1")
        self.assertEqual(window.frame_indices, (1, 2))
        self.assertFalse(window.history_complete)

    def test_rejects_out_of_order_track_observations_and_watermarks(self) -> None:
        buffer = TrackWindowBuffer()
        buffer.append(observation(2, 2.0))
        with self.assertRaises(ValueError):
            buffer.append(observation(1, 1.0))

        buffer.advance("camera-1", 3.0)
        with self.assertRaises(ValueError):
            buffer.advance("camera-1", 2.5)

        pruned = TrackWindowBuffer(
            before_seconds=0.0,
            after_seconds=0.0,
            retention_seconds=0.5,
        )
        pruned.append(observation(2, 2.0))
        pruned.advance("camera-1", 3.0)
        with self.assertRaises(ValueError):
            pruned.append(observation(1, 1.0))

    def test_event_registration_is_idempotent_but_rejects_id_collision(self) -> None:
        buffer = TrackWindowBuffer()
        event = candidate()
        first = buffer.register_event(event)
        second = buffer.register_event(event)
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            buffer.register_event(candidate(track_id="other"))


if __name__ == "__main__":
    unittest.main()
