from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from classroom_hybrid_ai.contracts import CandidateEvent, StageEvent
from classroom_hybrid_ai.locate_tracking import materialize_scheduled_screenshots
from classroom_hybrid_ai.pipeline import fair_candidate_selection
from classroom_hybrid_ai.stage_runtime import AdaptiveScreenshotScheduler, StageLedger
from classroom_hybrid_ai.video import detect_cpu_change_anchors
from classroom_hybrid_ai.watch_server import _status_payload, serve_watch


def _nested(payload: dict[str, object], path: str) -> object:
    value: object = payload
    for part in path.split("."):
        assert isinstance(value, dict)
        value = value[part]
    return value


def test_t5_defaults_match_source_and_packaged_configs() -> None:
    project = Path(__file__).resolve().parents[1]
    workspace = yaml.safe_load((project / "configs" / "default.yaml").read_text(encoding="utf-8"))
    packaged = yaml.safe_load(
        (project / "src" / "classroom_hybrid_ai" / "configs" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    # The two files intentionally use different repository-root resolution
    # tokens. T5 defaults must remain identical in both distribution paths.
    for path in (
        "locate_first.maximum_screenshots",
        "locate_first.qwen_every_screenshot",
        "locate_first.locate_cap",
        "locate_first.reserved_change_events",
        "locate_first.change_anchor_threshold",
        "locate_first.change_anchor_sample_stride",
        "locate_first.change_anchor_min_gap_seconds",
        "locate_first.exhaustive_max_screenshots",
    ):
        assert _nested(workspace, path) == _nested(packaged, path)


def test_adaptive_change_budget_and_exhaustive_caps_cover_long_video() -> None:
    scheduler = AdaptiveScreenshotScheduler(
        fps=10.0,
        frame_count=100_000,
        duration_seconds=10_000.0,
        mode="production_adaptive",
        locate_cap=5_000,
        reserved_change_events=5_000,
        max_screenshots=9_000,
    )
    rows = scheduler.schedule(
        change_anchors=[
            {"pts_seconds": float(index), "change_score": 1.0, "anchor_source": "cpu_scene_change"}
            for index in range(0, 10_000, 2)
        ]
    )
    assert scheduler.cap == 1_500
    assert scheduler.reserved_change_events == 300
    assert len(rows) <= 1_500
    assert {0, 50_000, 99_999}.issubset({int(row["frame_index"]) for row in rows})
    change_rows = [row for row in rows if row.get("anchor_source") == "cpu_scene_change"]
    assert len(change_rows) <= 300
    assert all(
        any(region_start <= float(row["pts_seconds"]) < region_end for row in change_rows)
        for region_start, region_end in (
            (0.0, 10_000.0 / 3.0),
            (10_000.0 / 3.0, 20_000.0 / 3.0),
            (20_000.0 / 3.0, 10_000.0),
        )
    )

    exhaustive = AdaptiveScreenshotScheduler(
        fps=25.0,
        frame_count=10_000,
        duration_seconds=400.0,
        mode="exhaustive_validation",
        max_screenshots=9_000,
    ).schedule()
    assert len(exhaustive) == 300
    assert all(
        right["frame_index"] - left["frame_index"] == 10
        for left, right in zip(exhaustive, exhaustive[1:])
    )


def test_s1_decode_and_screenshot_helpers_report_real_monotonic_units(tmp_path: Path) -> None:
    video = tmp_path / "progress.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (32, 32))
    assert writer.isOpened()
    for index in range(6):
        writer.write(np.full((32, 32, 3), index * 30, dtype=np.uint8))
    writer.release()

    scan_progress: list[tuple[int, int | None]] = []
    detect_cpu_change_anchors(
        video,
        fps=2.0,
        frame_count=6,
        duration_seconds=3.0,
        sample_stride=1,
        threshold=0.0,
        progress_callback=lambda processed, total: scan_progress.append((processed, total)),
    )
    assert scan_progress[0] == (0, 6)
    assert scan_progress[-1] == (6, 6)
    assert [row[0] for row in scan_progress] == sorted(row[0] for row in scan_progress)

    screenshot_progress: list[tuple[int, int]] = []
    rows, receipt = materialize_scheduled_screenshots(
        video,
        tmp_path / "screenshots",
        [
            {"frame_index": 0, "pts_seconds": 0.0},
            {"frame_index": 5, "pts_seconds": 2.5},
        ],
        progress_callback=lambda processed, total: screenshot_progress.append((processed, total)),
    )
    assert receipt["screenshots"] == len(rows) == 2
    assert screenshot_progress == [(0, 2), (1, 2), (2, 2)]


def test_fair_budget_covers_signals_late_video_and_students_with_reasons() -> None:
    candidates: list[CandidateEvent] = []
    ordinal = 0
    for student in range(1, 5):
        for timestamp in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0):
            ordinal += 1
            candidates.append(CandidateEvent(
                event_id=f"event-{ordinal}",
                track_id=f"student-{student:03d}",
                event_type="snapshot",
                trigger_frame_index=int(timestamp),
                trigger_pts_seconds=timestamp,
                metadata={
                    "edge_seat": student == 1 and timestamp == 0.0,
                    "new_student_id": timestamp == 0.0,
                    "action_change": timestamp == 30.0,
                    "anomaly": student == 2 and timestamp == 40.0,
                },
            ))
    candidates.append(CandidateEvent(
        event_id="id-uncertain",
        track_id="student-999",
        event_type="snapshot",
        trigger_frame_index=55,
        trigger_pts_seconds=55.0,
        metadata={"id_uncertain": True},
    ))
    selected, audit = fair_candidate_selection(candidates, limit=8, bucket_seconds=10.0)
    counts = Counter(item.track_id for item in selected)
    assert len(selected) == 8
    assert max(counts.values()) - min(counts.values()) <= 1
    assert any(item.trigger_pts_seconds >= 30.0 for item in selected)
    assert any(item.metadata.get("anomaly") for item in selected)
    assert any(item.metadata.get("action_change") for item in selected)
    assert any(item.metadata.get("edge_seat") for item in selected)
    assert any(item.metadata.get("new_student_id") for item in selected)
    assert {row["selection_status"] for row in audit} == {"selected", "unselected"}
    assert all(str(row.get("selection_reason", "")).strip() for row in audit)
    assert next(row for row in audit if row["event_id"] == "id-uncertain")["canonical_reason"] == "id_uncertain"
    assert {row["canonical_reason"] for row in audit if row["selection_status"] == "selected"} == {"scheduled"}
    assert {
        row["canonical_reason"]
        for row in audit
        if row["selection_status"] == "unselected"
    } <= {"budget_exhausted", "id_uncertain"}


def test_fair_budget_fills_when_a_less_served_student_has_no_rows_left() -> None:
    candidates = [
        CandidateEvent(
            event_id="student-a-only",
            track_id="student-001",
            event_type="snapshot",
            trigger_frame_index=0,
            trigger_pts_seconds=0.0,
        ),
        *[
            CandidateEvent(
                event_id=f"student-b-{index}",
                track_id="student-002",
                event_type="snapshot",
                trigger_frame_index=index,
                trigger_pts_seconds=float(index),
            )
            for index in range(1, 4)
        ],
    ]

    selected, audit = fair_candidate_selection(candidates, limit=4, bucket_seconds=10.0)

    assert {item.event_id for item in selected} == {item.event_id for item in candidates}
    assert len(selected) == 4
    assert all(row["selection_status"] == "selected" for row in audit)


def test_stage_events_100k_append_is_monotonic_and_resume_epoch_is_explicit(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path, run_id="run", video_id="video")
    for sequence in range(1, 100_001):
        ledger._append_event(StageEvent(
            event_id=f"event-{sequence}",
            stage_id="S0",
            event_type="progress",
            timestamp="2026-01-01T00:00:00.000Z",
            seq=sequence,
            epoch=0,
        ))
    ledger.close()
    rows = [json.loads(line) for line in (tmp_path / "stage_events.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 100_000
    assert [row["seq"] for row in rows[:3]] == [1, 2, 3]
    assert rows[-1]["seq"] == 100_000

    resumed = StageLedger.open_existing(tmp_path)
    resumed.prepare_resume("S0", invalidated_by=("test",))
    resumed.begin("S0", total_units=2)
    resumed.progress("S0", 1, 2)
    resumed.close()
    tail = [json.loads(line) for line in (tmp_path / "stage_events.v1.jsonl").read_text(encoding="utf-8").splitlines()[-3:]]
    assert tail[-1]["epoch"] == 1
    assert tail[-1]["seq"] > 100_000
    state = json.loads((tmp_path / "run_state.v1.json").read_text(encoding="utf-8"))
    assert state["epoch"] == 1


def test_watch_status_exposes_eta_and_terminal_zero(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path, run_id="run", video_id="video")
    ledger.begin("S0", total_units=2)
    ledger.progress("S0", 1, 2, preview_uris=("previews/S0/preview.json",))
    running = _status_payload(tmp_path)
    assert running["state"]["eta_seconds"] is not None
    assert running["state"]["eta_seconds"] >= 0.0
    assert running["state"]["previews"] == ["previews/S0/preview.json"]

    ledger.complete_run(status="completed")
    completed = _status_payload(tmp_path)
    assert completed["state"]["eta_seconds"] == 0.0
    ledger.close()


def test_new_stage_total_is_persisted_immediately_at_zero_progress(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path, run_id="run", video_id="video", total_units=51)
    ledger.begin("S5", total_units=None)
    ledger.progress("S5", 0, 3, message="S5 已发现 3 个候选")
    state = json.loads((tmp_path / "run_state.v1.json").read_text(encoding="utf-8"))
    assert state["stages"]["S5"]["processed_units"] == 0
    assert state["stages"]["S5"]["total_units"] == 3
    assert state["current_stage_progress"] == 0.0
    events = [
        json.loads(line)
        for line in (tmp_path / "stage_events.v1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["stage_id"] == "S5"
    assert events[-1]["total_units"] == 3
    ledger.close()


def test_watch_sse_reads_new_bytes_after_cursor_and_blocks_traversal(tmp_path: Path) -> None:
    (tmp_path / "run_state.v1.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    events = tmp_path / "stage_events.v1.jsonl"
    events.write_text(json.dumps({"seq": 1, "epoch": 0}) + "\n", encoding="utf-8")
    server = serve_watch(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/events?since=0"
        first = urllib.request.urlopen(endpoint)
        first_body = first.read().decode("utf-8")
        offset = int(first.headers["X-Events-Next-Offset"])
        assert "id: 1" in first_body

        with events.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"seq": 2, "epoch": 1}) + "\n")
        second = urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/events?since=1&offset={offset}"
        )
        second_body = second.read().decode("utf-8")
        assert "id: 2" in second_body
        assert "id: 1" not in second_body

        # The same ``since=0`` poll is incremental after the initial full
        # history read; it must not rescan or replay the old rows.
        repeat = urllib.request.urlopen(endpoint)
        assert repeat.read().decode("utf-8") == ""
        with events.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"seq": 3, "epoch": 1}) + "\n")
        appended = urllib.request.urlopen(endpoint)
        appended_body = appended.read().decode("utf-8")
        assert "id: 3" in appended_body
        assert "id: 1" not in appended_body
        assert "id: 2" not in appended_body

        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_address[1]}/artifact/../run_state.v1.json"
            )
        except urllib.error.HTTPError as error:
            assert error.code in {400, 404}
        else:
            raise AssertionError("path traversal unexpectedly served an artifact")
    finally:
        server.shutdown()
        server.server_close()
