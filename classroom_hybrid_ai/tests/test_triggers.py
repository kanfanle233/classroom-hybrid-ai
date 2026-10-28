from __future__ import annotations

import unittest

from classroom_hybrid_ai.contracts import CandidateEvent, Detection, TrackObservation
from classroom_hybrid_ai.triggers import (
    CooldownGate,
    EventDeduplicator,
    KOfMThresholdTrigger,
    PTSSampler,
    PeriodicSampler,
    StrideSampler,
    TriggerEngine,
)


def observation(frame: int, pts: float, score: float, track_id: str = "7") -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        frame_index=frame,
        pts_seconds=pts,
        detection=Detection(
            label="student",
            bbox=(10, 20, 30, 50),
            score=0.9,
            source="yolo26",
        ),
        anomaly_scores={"leave_seat": score},
    )


def candidate(
    event_id: str,
    pts: float,
    *,
    track_id: str = "7",
    event_type: str = "leave_seat",
) -> CandidateEvent:
    return CandidateEvent(
        event_id=event_id,
        track_id=track_id,
        event_type=event_type,
        trigger_frame_index=int(pts * 10),
        trigger_pts_seconds=pts,
    )


class SamplingTests(unittest.TestCase):
    def test_stride_sampler_honors_offset(self) -> None:
        sampler = StrideSampler(stride=10, offset=3)
        selected = [index for index in range(35) if sampler.should_sample(index)]
        self.assertEqual(selected, [3, 13, 23, 33])

    def test_pts_sampler_uses_deadlines_without_drift(self) -> None:
        sampler = PTSSampler(1.0)
        decisions = [sampler.should_sample(pts) for pts in (0.0, 0.4, 1.1, 1.9, 2.0)]
        self.assertEqual(decisions, [True, False, True, False, True])

    def test_pts_sampler_has_independent_stream_state_and_resets_on_seek(self) -> None:
        sampler = PTSSampler(2.0)
        self.assertTrue(sampler.should_sample(5.0, "a"))
        self.assertFalse(sampler.should_sample(5.5, "a"))
        self.assertTrue(sampler.should_sample(5.5, "b"))
        self.assertTrue(sampler.should_sample(1.0, "a"))

    def test_periodic_sampler_prefers_pts_and_falls_back_to_stride(self) -> None:
        sampler = PeriodicSampler(frame_stride=5, pts_interval_seconds=1.0)
        self.assertTrue(sampler.should_sample(0, None))
        self.assertFalse(sampler.should_sample(1, None))
        self.assertTrue(sampler.should_sample(1, 10.0))
        self.assertFalse(sampler.should_sample(5, 10.2))


class ThresholdTests(unittest.TestCase):
    def test_k_of_m_starts_and_low_threshold_ends_with_hysteresis(self) -> None:
        trigger = KOfMThresholdTrigger(
            k=2, m=3, high_threshold=0.8, low_threshold=0.2
        )
        key = ("stream", "track", "event")
        self.assertFalse(trigger.update(key, 0.81).started)
        started = trigger.update(key, 0.90)
        self.assertTrue(started.started)
        self.assertTrue(started.active)
        self.assertEqual(started.high_votes, 2)

        self.assertFalse(trigger.update(key, 0.50).ended)
        self.assertFalse(trigger.update(key, 0.10).ended)
        ended = trigger.update(key, 0.20)
        self.assertTrue(ended.ended)
        self.assertFalse(ended.active)

    def test_full_window_option_waits_for_m_observations(self) -> None:
        trigger = KOfMThresholdTrigger(
            k=2,
            m=3,
            high_threshold=0.8,
            low_threshold=0.2,
            require_full_window=True,
        )
        self.assertFalse(trigger.update("key", 0.9).started)
        self.assertFalse(trigger.update("key", 0.9).started)
        self.assertTrue(trigger.update("key", 0.4).started)


class GateAndDeduplicationTests(unittest.TestCase):
    def test_cooldown_is_scoped_by_key_and_marks_only_accepts(self) -> None:
        gate = CooldownGate(5.0)
        self.assertTrue(gate.allow("a", 10.0))
        self.assertFalse(gate.allow("a", 14.9))
        self.assertAlmostEqual(gate.remaining("a", 14.9), 0.1)
        self.assertTrue(gate.allow("b", 11.0))
        self.assertTrue(gate.allow("a", 15.0))

    def test_deduplicates_by_id_and_nearby_track_event(self) -> None:
        dedup = EventDeduplicator(2.0)
        first = candidate("e1", 10.0)
        self.assertTrue(dedup.accept(first).accepted)

        same_id = dedup.accept(first)
        self.assertFalse(same_id.accepted)
        self.assertEqual(same_id.reason, "same_event_id")

        nearby = dedup.accept(candidate("e2", 11.0))
        self.assertFalse(nearby.accepted)
        self.assertEqual(nearby.canonical_event_id, "e1")

        self.assertTrue(dedup.accept(candidate("e3", 11.0, track_id="8")).accepted)
        self.assertTrue(dedup.accept(candidate("e4", 12.01)).accepted)


class TriggerEngineTests(unittest.TestCase):
    def test_engine_builds_candidate_from_high_vote_evidence(self) -> None:
        ids = iter(("event-1", "event-2"))
        engine = TriggerEngine(
            KOfMThresholdTrigger(k=2, m=3, high_threshold=0.8, low_threshold=0.2),
            cooldown=CooldownGate(5.0),
            deduplicator=EventDeduplicator(1.0),
            event_id_factory=lambda _observation, _event_type: next(ids),
        )

        self.assertIsNone(engine.update(observation(10, 1.0, 0.85), "leave_seat"))
        event = engine.update(observation(11, 1.1, 0.90), "leave_seat")
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.event_id, "event-1")
        self.assertEqual(event.evidence_frame_indices, (10, 11))
        self.assertEqual(event.start_pts_seconds, 1.0)
        self.assertEqual(event.source, "yolo26")

        # It remains active until k low votes arrive, so another high score does
        # not create an event storm.
        self.assertIsNone(engine.update(observation(12, 1.2, 0.99), "leave_seat"))
        self.assertIsNone(engine.update(observation(13, 1.3, 0.1), "leave_seat"))
        self.assertIsNone(engine.update(observation(14, 1.4, 0.1), "leave_seat"))

        # A new transition inside cooldown is suppressed.
        self.assertIsNone(engine.update(observation(15, 2.0, 0.9), "leave_seat"))
        self.assertIsNone(engine.update(observation(16, 2.1, 0.9), "leave_seat"))

    def test_engine_reads_named_score_from_observation(self) -> None:
        engine = TriggerEngine(
            KOfMThresholdTrigger(k=1, m=1, high_threshold=0.8, low_threshold=0.2),
            event_id_factory=lambda _observation, _event_type: "named-score",
        )
        event = engine.update(observation(1, 0.1, 0.9), "leave_seat")
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.score, 0.9)

    def test_track_reset_discards_sparse_votes_before_reconnection(self) -> None:
        engine = TriggerEngine(
            KOfMThresholdTrigger(k=2, m=3, high_threshold=0.8, low_threshold=0.2),
            event_id_factory=lambda _observation, _event_type: "unexpected",
        )
        self.assertIsNone(engine.update(observation(1, 0.0, 0.9), "leave_seat"))
        engine.reset_track("default", "7")
        self.assertIsNone(engine.update(observation(100, 20.0, 0.9), "leave_seat"))


if __name__ == "__main__":
    unittest.main()
