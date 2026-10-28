from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from classroom_hybrid_ai.outputs_v3 import build_v3_outputs
from classroom_hybrid_ai.profiles import load_profile
from classroom_hybrid_ai.stage_runtime import AdaptiveScreenshotScheduler, STAGE_IDS, StageLedger, weighted_progress


ROOT = Path(__file__).resolve().parents[1]


def test_stage_ledger_events_and_progress_are_monotonic(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path, run_id="run-1", video_id="video-1", total_units=10)
    ledger.begin("S0", total_units=1)
    ledger.progress("S0", 1, 1)
    ledger.finish("S0")
    ledger.begin("S1", total_units=10)
    ledger.progress("S1", 3, 10)
    ledger.progress("S1", 2, 10)  # stale producer update cannot go backwards
    state = ledger.state
    assert state.overall_progress is not None
    assert state.overall_progress >= 5.0
    events = [json.loads(line) for line in (tmp_path / "stage_events.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["event_type"] == "started"
    assert all(float(row["progress"]) >= 0 for row in events)
    assert (tmp_path / "run_state.v1.json").is_file()


def test_stage_ledger_resume_replaces_current_run_fingerprints(tmp_path: Path) -> None:
    ledger = StageLedger(
        tmp_path,
        run_id="run-1",
        video_id="video-1",
        metadata={"code_sha256": "old-code", "config_sha256": "old-config"},
    )
    ledger.prepare_resume(
        "S0",
        invalidated_by=("input_fingerprint_mismatch",),
        metadata={"code_sha256": "new-code", "config_sha256": "new-config"},
    )
    assert ledger.state.metadata["code_sha256"] == "new-code"
    assert ledger.state.metadata["config_sha256"] == "new-config"
    assert ledger.state.metadata["resume_invalidated_by"] == ["input_fingerprint_mismatch"]


def test_adaptive_and_exhaustive_scheduler_caps() -> None:
    adaptive = AdaptiveScreenshotScheduler(fps=10, frame_count=36000, duration_seconds=3600, mode="production_adaptive")
    assert adaptive.base_interval_seconds == 3.0
    assert len(adaptive.schedule()) <= 1500
    exhaustive = AdaptiveScreenshotScheduler(fps=25, frame_count=10000, duration_seconds=400, mode="exhaustive_validation")
    rows = exhaustive.schedule()
    assert len(rows) == 300
    assert rows[1]["frame_index"] - rows[0]["frame_index"] == 10


def test_v3_output_contract_contains_timeline_overlay_and_videos(tmp_path: Path) -> None:
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    for index in range(5):
        writer.write(np.full((64, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    rows = []
    for index in range(5):
        rows.append({"frame_index": index, "pts_seconds": index / 10, "student_id": "student-001", "detection": {"label": "student", "bbox": [5, 5, 25, 50], "source": "locate_anything"}})
    (run_dir / "artifacts" / "track_observations.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    profile = load_profile(ROOT / "configs" / "action_profiles.yaml")
    result = build_v3_outputs(run_dir, source=video, info={"path": str(video), "fps": 10.0, "frame_count": 5, "duration_seconds": 0.5, "width": 64, "height": 64}, profile=profile, config_digest="config")
    assert result["students"] == 1
    assert (run_dir / "timeline.v2.json").is_file()
    assert (run_dir / "frontend" / "index.v1.json").is_file()
    assert (run_dir / "frontend" / "overlay_chunks" / "000000.json").is_file()
    assert all((run_dir / "videos" / name).is_file() for name in ("tracking_preview.mp4", "action_preview.mp4", "annotated_final.mp4"))
    timeline = json.loads((run_dir / "timeline.v2.json").read_text(encoding="utf-8"))
    assert timeline["coverage"]["complete"] is True
    assert {item["action_code"] for item in timeline["intervals"]} == {"UNKNOWN"}
    frontend = json.loads((run_dir / "frontend" / "index.v1.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert timeline["taxonomy_profile_id"] == profile.profile_id
    assert timeline["taxonomy_contract_sha256"] == profile.sha256
    assert frontend["taxonomy_profile_id"] == profile.profile_id
    assert frontend["taxonomy_contract_sha256"] == profile.sha256
    assert manifest["profile_sha256"] == profile.sha256
