import json
import re
from pathlib import Path

from classroom_pipeline.cli import _option_value
from classroom_pipeline.contracts.schemas import SCHEMA_VERSION
from classroom_pipeline.orchestrator_legacy import STEP_ORDER
from classroom_pipeline.stage_registry import STAGES
from classroom_pipeline.validate import validate_case
from server.build_frontend_data_bundle import _git_head_commit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _valid_case(case_dir: Path) -> None:
    case_dir.mkdir()
    (case_dir / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "case_1",
                "video_id": "front__case_1",
                "schema_version": SCHEMA_VERSION,
                "artifacts": {},
                "config_snapshot": {},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(case_dir / "actions.fusion_v2.jsonl", [{"action": "listen"}])
    _write_jsonl(
        case_dir / "event_queries.fusion_v2.jsonl",
        [
            {
                "event_id": "e1",
                "schema_version": SCHEMA_VERSION,
                "query_text": "please listen",
                "event_type": "listen",
                "trigger_words": ["listen"],
                "timestamp": 1.0,
                "start": 0.5,
                "end": 1.5,
                "confidence": 0.9,
                "source": "asr",
            }
        ],
    )
    _write_jsonl(
        case_dir / "verified_events.jsonl",
        [
            {
                "event_id": "e1",
                "track_id": 1,
                "event_type": "listen",
                "query_text": "please listen",
                "query_time": 1.0,
                "window_start": 0.5,
                "window_end": 1.5,
                "p_match": 0.8,
                "p_mismatch": 0.2,
                "reliability_score": 0.75,
                "uncertainty": 0.25,
                "label": "match",
                "threshold_source": "test",
                "model_version": "test",
            }
        ],
    )


def test_registered_stage_paths_are_unique_and_exist():
    names = [stage.name for stage in STAGES]
    paths = [path for stage in STAGES for path in stage.required_abs()]
    assert len(names) == len(set(names))
    assert len(paths) == len(set(paths))
    assert all(path.exists() for path in paths)


def test_coherence_stage_is_part_of_main_execution_order():
    assert 73 in STEP_ORDER
    coherence = next(stage for stage in STAGES if stage.name == "coherence_diagnosis")
    assert coherence.required_paths == ("classroom_pipeline/pipeline/08_coherence_diagnosis.py",)


def test_run_option_value_supports_both_argparse_forms():
    assert _option_value(["--video", "a.mp4"], "--video") == "a.mp4"
    assert _option_value(["--video=b.mp4"], "--video") == "b.mp4"
    assert _option_value([], "--video") == ""


def test_validate_case_rejects_missing_required_artifacts(tmp_path):
    case_dir = tmp_path / "partial"
    case_dir.mkdir()
    assert validate_case(case_dir) == 2


def test_validate_case_accepts_contract_valid_core_artifacts(tmp_path):
    case_dir = tmp_path / "complete"
    _valid_case(case_dir)
    assert validate_case(case_dir) == 0


def test_validate_case_rejects_malformed_jsonl(tmp_path):
    case_dir = tmp_path / "malformed"
    _valid_case(case_dir)
    (case_dir / "actions.fusion_v2.jsonl").write_text("not-json\n", encoding="utf-8")
    assert validate_case(case_dir) == 2


def test_frontend_bundle_reads_commit_from_repository_root():
    assert re.fullmatch(r"[0-9a-fA-F]{7,40}", _git_head_commit())
