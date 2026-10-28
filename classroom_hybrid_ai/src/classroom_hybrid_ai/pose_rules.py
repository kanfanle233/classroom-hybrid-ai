"""Observable pose proxies that create review candidates, never final diagnoses."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _point(keypoints: Sequence[Sequence[float]], index: int) -> tuple[float, float, float] | None:
    if index >= len(keypoints) or len(keypoints[index]) < 2:
        return None
    row = keypoints[index]
    confidence = float(row[2]) if len(row) > 2 else 1.0
    if confidence < 0.15:
        return None
    return float(row[0]), float(row[1]), confidence


def pose_proxy_scores(
    keypoints_xyc: Sequence[Sequence[float]] | None,
    bbox_xyxy: Sequence[float],
    detection_confidence: float,
) -> dict[str, float]:
    """Return conservative scores for a few visible pose states.

    These scores are routing signals. They are not calibrated probabilities and
    are not written as final actions without multimodal review.
    """

    scores = {"low_detection_confidence": max(0.0, min(1.0, 1.0 - detection_confidence))}
    if not keypoints_xyc or len(bbox_xyxy) != 4:
        return scores
    box_height = max(1.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))
    nose = _point(keypoints_xyc, 0)
    left_shoulder = _point(keypoints_xyc, 5)
    right_shoulder = _point(keypoints_xyc, 6)
    left_wrist = _point(keypoints_xyc, 9)
    right_wrist = _point(keypoints_xyc, 10)
    left_hip = _point(keypoints_xyc, 11)
    right_hip = _point(keypoints_xyc, 12)
    left_ankle = _point(keypoints_xyc, 15)
    right_ankle = _point(keypoints_xyc, 16)

    shoulders = [point for point in (left_shoulder, right_shoulder) if point]
    if nose and shoulders:
        shoulder_y = sum(point[1] for point in shoulders) / len(shoulders)
        normalized_drop = (nose[1] - shoulder_y) / box_height
        scores["head_down"] = max(0.0, min(1.0, (normalized_drop + 0.02) / 0.18))

    raised_votes: list[float] = []
    for wrist, shoulder in ((left_wrist, left_shoulder), (right_wrist, right_shoulder)):
        if wrist and shoulder:
            vertical = (shoulder[1] - wrist[1]) / box_height
            raised_votes.append(max(0.0, min(1.0, vertical / 0.22)))
    if raised_votes:
        scores["raise_hand"] = max(raised_votes)

    hips = [point for point in (left_hip, right_hip) if point]
    ankles = [point for point in (left_ankle, right_ankle) if point]
    if shoulders and hips and ankles:
        shoulder_y = sum(point[1] for point in shoulders) / len(shoulders)
        hip_y = sum(point[1] for point in hips) / len(hips)
        ankle_y = sum(point[1] for point in ankles) / len(ankles)
        torso = max(1.0, hip_y - shoulder_y)
        leg_ratio = (ankle_y - hip_y) / torso
        scores["stand"] = max(0.0, min(1.0, (leg_ratio - 0.65) / 0.9))
    return scores


def keypoint_displacement_score(
    previous_keypoints: Sequence[Sequence[float]] | None,
    previous_bbox: Sequence[float] | None,
    current_keypoints: Sequence[Sequence[float]] | None,
    current_bbox: Sequence[float] | None,
    *,
    threshold: float = 0.15,
) -> float:
    """Score normalized pose movement between two observations.

    Coordinates are normalized independently by each person's bounding box, so
    a camera pan or a small change in scale does not become a movement event.
    The returned value reaches ``1`` when the mean common-keypoint displacement
    reaches ``threshold``; missing/low-confidence keypoints return zero.
    """

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if not previous_keypoints or not current_keypoints:
        return 0.0
    if previous_bbox is None or current_bbox is None:
        return 0.0
    if len(previous_bbox) != 4 or len(current_bbox) != 4:
        return 0.0

    def normalize(points: Sequence[Sequence[float]], bbox: Sequence[float]) -> dict[int, tuple[float, float]]:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        output: dict[int, tuple[float, float]] = {}
        for index, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            confidence = float(point[2]) if len(point) > 2 else 1.0
            if confidence < 0.15:
                continue
            output[index] = ((float(point[0]) - x1) / width, (float(point[1]) - y1) / height)
        return output

    previous = normalize(previous_keypoints, previous_bbox)
    current = normalize(current_keypoints, current_bbox)
    common = sorted(set(previous) & set(current))
    if not common:
        return 0.0
    displacement = sum(
        ((previous[index][0] - current[index][0]) ** 2 +
         (previous[index][1] - current[index][1]) ** 2) ** 0.5
        for index in common
    ) / len(common)
    return max(0.0, min(1.0, displacement / float(threshold)))


def extract_keypoints(result: Mapping[str, Any]) -> Sequence[Sequence[float]] | None:
    for key in ("keypoints_xyc", "keypoints", "pose_keypoints"):
        value = result.get(key)
        if isinstance(value, list):
            return value
    return None
