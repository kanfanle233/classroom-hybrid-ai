from classroom_hybrid_ai.pose_rules import keypoint_displacement_score, pose_proxy_scores


def test_raised_wrist_creates_candidate_score_only() -> None:
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    keypoints[5] = [30, 50, 0.9]
    keypoints[9] = [30, 20, 0.9]
    scores = pose_proxy_scores(keypoints, [0, 0, 100, 200], 0.9)
    assert scores["raise_hand"] > 0
    assert all(0 <= value <= 1 for value in scores.values())


def test_keypoint_displacement_is_normalized_by_each_bbox() -> None:
    previous = [[50.0, 100.0, 0.9]] + [[0.0, 0.0, 0.0] for _ in range(16)]
    current = [[65.0, 100.0, 0.9]] + [[0.0, 0.0, 0.0] for _ in range(16)]
    scaled_current = [[100.0, 200.0, 0.9]] + [[0.0, 0.0, 0.0] for _ in range(16)]
    assert keypoint_displacement_score(
        previous, [0, 0, 100, 200], current, [0, 0, 100, 200], threshold=0.15
    ) == 1.0
    assert keypoint_displacement_score(
        previous, [0, 0, 100, 200], scaled_current, [0, 0, 200, 400], threshold=0.15
    ) == 0.0
