from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from classroom_hybrid_ai.cli import main
from classroom_hybrid_ai.profiles import ActionProfile, load_profiles
from classroom_hybrid_ai.watch_server import serve_watch


ROOT = Path(__file__).resolve().parents[1]


def _tiny_video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
    for index in range(12):
        writer.write(np.full((32, 32, 3), index * 10, dtype=np.uint8))
    writer.release()
    return path


def test_action_profiles_round_trip_has_stable_digest() -> None:
    profiles = load_profiles(ROOT / "configs" / "action_profiles.yaml")
    assert set(profiles) == {"core8_v1", "scb_student_11_candidate_v1"}
    for profile in profiles.values():
        encoded = profile.to_dict()
        restored = ActionProfile.from_dict(encoded)
        assert restored.to_dict() == encoded
        assert restored.sha256 == profile.sha256 == encoded["contract_sha256"]
        encoded["contract_sha256"] = "tampered"
        assert ActionProfile.from_dict(encoded).sha256 == profile.sha256


@pytest.mark.parametrize("profile_id", ["core8_v1", "scb_student_11_candidate_v1"])
def test_cli_plan_supports_both_profiles(tmp_path: Path, capsys, profile_id: str) -> None:
    video = _tiny_video(tmp_path / "lesson.avi")
    rc = main([
        "--config", str(ROOT / "configs" / "default.yaml"),
        "plan", str(video), "--profile", profile_id,
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["profile_id"] == profile_id
    assert payload["profile"]["contract_sha256"]


def test_cli_runtime_error_is_json_and_nonzero(tmp_path: Path, capsys) -> None:
    rc = main([
        "--config", str(ROOT / "configs" / "default.yaml"),
        "plan", str(tmp_path / "missing.mp4"),
    ])
    assert rc != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["command"] == "plan"
    assert payload["error"]["code"] == "video_not_found"


def test_watch_server_is_loopback_only(tmp_path: Path) -> None:
    server = serve_watch(tmp_path, port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
    with pytest.raises(ValueError, match="watch_host_must_be_127.0.0.1"):
        serve_watch(tmp_path, host="0.0.0.0", port=0)
