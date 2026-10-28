"""Strict fusion of small-model candidates and Qwen review decisions."""

from __future__ import annotations

from typing import List, Optional

from .contracts import CandidateEvent, FinalEvent, ReviewDecision


class StrictFusionPolicy:
    """Validate explicit review evidence before accepting a final decision.

    This policy performs no confidence arithmetic.  In particular,
    ``ReviewDecision.self_reported_confidence`` is never inspected when choosing
    ``confirmed``, ``rejected``, or ``uncertain``.  A language model's confidence
    wording is not a calibrated probability.
    """

    policy_name = "strict_qwen_review_v1"

    def __init__(
        self,
        *,
        min_evidence_frames: int = 1,
        require_rationale: bool = True,
        require_event_type_match: bool = True,
        require_behavior_for_confirmation: bool = True,
    ) -> None:
        if isinstance(min_evidence_frames, bool) or not isinstance(
            min_evidence_frames, int
        ):
            raise TypeError("min_evidence_frames must be an integer")
        if min_evidence_frames < 0:
            raise ValueError("min_evidence_frames must be non-negative")
        for name, value in (
            ("require_rationale", require_rationale),
            ("require_event_type_match", require_event_type_match),
            ("require_behavior_for_confirmation", require_behavior_for_confirmation),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")

        self.min_evidence_frames = min_evidence_frames
        self.require_rationale = require_rationale
        self.require_event_type_match = require_event_type_match
        self.require_behavior_for_confirmation = require_behavior_for_confirmation

    def fuse(
        self,
        candidate: CandidateEvent,
        review: Optional[ReviewDecision],
    ) -> FinalEvent:
        if not isinstance(candidate, CandidateEvent):
            raise TypeError("candidate must be a CandidateEvent")
        if review is not None and not isinstance(review, ReviewDecision):
            raise TypeError("review must be a ReviewDecision or None")

        if review is None:
            return self._uncertain(candidate, None, ["missing_review"])

        reasons = self._validation_failures(candidate, review)

        # An explicit uncertainty is terminal.  It cannot be promoted by a high
        # self-reported confidence value or by the YOLO candidate score.
        if review.decision == "uncertain":
            reasons.append("review_declared_uncertain")
            return self._uncertain(candidate, review, reasons)

        if reasons:
            return self._uncertain(candidate, review, reasons)

        return FinalEvent(
            event_id=candidate.event_id,
            candidate=candidate,
            review=review,
            status=review.decision,
            behavior=review.behavior,
            rationale=review.rationale.strip(),
            metadata=self._metadata(()),
        )

    def _validation_failures(
        self, candidate: CandidateEvent, review: ReviewDecision
    ) -> List[str]:
        reasons: List[str] = []
        if not review.schema_valid:
            reasons.append("review_schema_invalid")
        if review.candidate_event_id != candidate.event_id:
            reasons.append("candidate_id_mismatch")
        if self.require_event_type_match:
            if review.reviewed_event_type is None:
                reasons.append("missing_reviewed_event_type")
            elif review.reviewed_event_type != candidate.event_type:
                reasons.append("event_type_mismatch")
        if self.require_rationale and not review.rationale.strip():
            reasons.append("missing_rationale")
        if len(review.evidence_frame_indices) < self.min_evidence_frames:
            reasons.append("insufficient_evidence_frames")
        if (
            review.decision == "confirmed"
            and self.require_behavior_for_confirmation
            and review.behavior is None
        ):
            reasons.append("missing_confirmed_behavior")
        return reasons

    def _uncertain(
        self,
        candidate: CandidateEvent,
        review: Optional[ReviewDecision],
        reasons: List[str],
    ) -> FinalEvent:
        rationale = "fusion could not make a strict decision: " + ", ".join(reasons)
        if review is not None and review.rationale.strip():
            rationale = f"{rationale}; reviewer rationale: {review.rationale.strip()}"
        return FinalEvent(
            event_id=candidate.event_id,
            candidate=candidate,
            review=review,
            status="uncertain",
            behavior=review.behavior if review is not None else None,
            rationale=rationale,
            metadata=self._metadata(reasons),
        )

    def _metadata(self, validation_reasons: object) -> dict:
        return {
            "fusion_policy": self.policy_name,
            "validation_reasons": tuple(validation_reasons),
            "self_reported_confidence_used": False,
            "candidate_score_used_as_review_probability": False,
        }


def fuse_candidate(
    candidate: CandidateEvent,
    review: Optional[ReviewDecision],
    *,
    policy: Optional[StrictFusionPolicy] = None,
) -> FinalEvent:
    """Fuse one candidate using the default strict policy."""

    return (policy or StrictFusionPolicy()).fuse(candidate, review)


# Concise compatibility alias for orchestration code.
FusionPolicy = StrictFusionPolicy


__all__ = ["FusionPolicy", "StrictFusionPolicy", "fuse_candidate"]
