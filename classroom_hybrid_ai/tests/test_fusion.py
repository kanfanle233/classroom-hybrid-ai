from __future__ import annotations

import unittest

from classroom_hybrid_ai.contracts import (
    CandidateEvent,
    Detection,
    FinalEvent,
    ReviewDecision,
    TrackObservation,
)
from classroom_hybrid_ai.fusion import StrictFusionPolicy, fuse_candidate


def candidate() -> CandidateEvent:
    return CandidateEvent(
        event_id="event-42",
        stream_id="camera-1",
        track_id="student-7",
        event_type="leave_seat",
        trigger_frame_index=102,
        trigger_pts_seconds=4.08,
        start_pts_seconds=4.00,
        score=0.91,
        evidence_frame_indices=(100, 101, 102),
        source="yolo26",
        metadata={"seat": "B3"},
    )


def review(
    decision: str = "confirmed",
    *,
    confidence: object = "high",
    **changes: object,
) -> ReviewDecision:
    values = {
        "candidate_event_id": "event-42",
        "decision": decision,
        "reviewed_event_type": "leave_seat",
        "behavior": "leave_seat" if decision == "confirmed" else None,
        "rationale": "The tracked student moves fully away from the assigned seat.",
        "evidence_frame_indices": (100, 102),
        "self_reported_confidence": confidence,
    }
    values.update(changes)
    return ReviewDecision(**values)  # type: ignore[arg-type]


class ContractRoundTripTests(unittest.TestCase):
    def test_nested_contracts_round_trip_through_dict_and_json(self) -> None:
        detection = Detection(
            label="student",
            bbox=(1, 2, 30, 40),
            score=0.8,
            source="locateanything",
            attributes={"prompt": "student"},
        )
        observation = TrackObservation(
            track_id="student-7",
            frame_index=100,
            pts_seconds=4.0,
            detection=detection,
            anomaly_scores={"leave_seat": 0.9},
        )
        self.assertEqual(TrackObservation.from_dict(observation.to_dict()), observation)
        self.assertEqual(TrackObservation.from_json(observation.to_json()), observation)

        final = fuse_candidate(candidate(), review(confidence={"claim": 0.99}))
        from_dict = FinalEvent.from_dict(final.to_dict())
        from_json = FinalEvent.from_json(final.to_json())
        self.assertEqual(from_dict, final)
        self.assertEqual(from_json, final)
        self.assertIsInstance(from_json.candidate, CandidateEvent)
        self.assertIsInstance(from_json.review, ReviewDecision)

    def test_bbox_validation_rejects_inverted_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            Detection(label="student", bbox=(10, 10, 5, 20))

    def test_review_rejects_unknown_decision(self) -> None:
        with self.assertRaises(ValueError):
            review("maybe")


class StrictFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = StrictFusionPolicy()
        self.candidate = candidate()

    def test_valid_confirmation_and_rejection(self) -> None:
        confirmed = self.policy.fuse(self.candidate, review("confirmed"))
        self.assertEqual(confirmed.status, "confirmed")
        self.assertEqual(confirmed.behavior, "leave_seat")

        rejected = self.policy.fuse(self.candidate, review("rejected"))
        self.assertEqual(rejected.status, "rejected")
        self.assertIsNone(rejected.behavior)

    def test_self_reported_confidence_never_changes_outcome(self) -> None:
        low_claim = self.policy.fuse(
            self.candidate, review("confirmed", confidence=0.01)
        )
        high_claim = self.policy.fuse(
            self.candidate, review("confirmed", confidence=0.999999)
        )
        prose_claim = self.policy.fuse(
            self.candidate, review("confirmed", confidence="absolutely certain")
        )
        self.assertEqual(
            (low_claim.status, high_claim.status, prose_claim.status),
            ("confirmed", "confirmed", "confirmed"),
        )
        self.assertFalse(low_claim.metadata["self_reported_confidence_used"])

    def test_declared_uncertainty_stays_uncertain_even_with_high_claim(self) -> None:
        result = self.policy.fuse(
            self.candidate,
            review(
                "uncertain",
                confidence=1.0,
                behavior=None,
                rationale="The crop is occluded.",
            ),
        )
        self.assertEqual(result.status, "uncertain")
        self.assertIn("review_declared_uncertain", result.metadata["validation_reasons"])

    def test_missing_review_is_uncertain(self) -> None:
        result = self.policy.fuse(self.candidate, None)
        self.assertEqual(result.status, "uncertain")
        self.assertEqual(result.metadata["validation_reasons"], ["missing_review"])

    def test_invalid_schema_cannot_confirm(self) -> None:
        result = self.policy.fuse(
            self.candidate,
            review("confirmed", confidence=1.0, schema_valid=False),
        )
        self.assertEqual(result.status, "uncertain")
        self.assertIn("review_schema_invalid", result.metadata["validation_reasons"])

    def test_mismatched_candidate_or_event_type_is_uncertain(self) -> None:
        wrong_id = self.policy.fuse(
            self.candidate,
            review("confirmed", candidate_event_id="event-other"),
        )
        self.assertEqual(wrong_id.status, "uncertain")
        self.assertIn("candidate_id_mismatch", wrong_id.metadata["validation_reasons"])

        wrong_type = self.policy.fuse(
            self.candidate,
            review("confirmed", reviewed_event_type="phone_use"),
        )
        self.assertEqual(wrong_type.status, "uncertain")
        self.assertIn("event_type_mismatch", wrong_type.metadata["validation_reasons"])

    def test_missing_evidence_rationale_or_behavior_is_uncertain(self) -> None:
        missing_evidence = self.policy.fuse(
            self.candidate, review("confirmed", evidence_frame_indices=())
        )
        self.assertIn(
            "insufficient_evidence_frames",
            missing_evidence.metadata["validation_reasons"],
        )

        missing_rationale = self.policy.fuse(
            self.candidate, review("confirmed", rationale="")
        )
        self.assertIn("missing_rationale", missing_rationale.metadata["validation_reasons"])

        missing_behavior = self.policy.fuse(
            self.candidate, review("confirmed", behavior=None)
        )
        self.assertIn(
            "missing_confirmed_behavior",
            missing_behavior.metadata["validation_reasons"],
        )
        self.assertEqual(
            (missing_evidence.status, missing_rationale.status, missing_behavior.status),
            ("uncertain", "uncertain", "uncertain"),
        )

    def test_review_preserves_model_details_without_relaxing_contract(self) -> None:
        decision = ReviewDecision.from_dict({
            "candidate_event_id": "event", "decision": "confirmed", "reviewed_event_type": "review",
            "behavior": "写字", "rationale": "visible", "evidence_frame_indices": [1, 2],
            "posture": "坐姿", "secondary_behaviors": ["与同桌交谈"],
        })
        assert decision.metadata["model_details"]["posture"] == "坐姿"

    def test_policy_can_require_multiple_evidence_frames(self) -> None:
        policy = StrictFusionPolicy(min_evidence_frames=3)
        result = policy.fuse(self.candidate, review("rejected"))
        self.assertEqual(result.status, "uncertain")


if __name__ == "__main__":
    unittest.main()
