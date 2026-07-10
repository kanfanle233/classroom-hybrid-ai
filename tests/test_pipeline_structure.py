import importlib
import json
import re
from pathlib import Path

from classroom_pipeline.cli import _normalize_argv, _option_value
from classroom_pipeline.contracts.schemas import SCHEMA_VERSION
from classroom_pipeline.orchestrator_legacy import STEP_ORDER
from classroom_pipeline.stage_registry import STAGES
from classroom_pipeline.validate import validate_case
from scripts.utils.build_verifier_training_samples import _assign_split
from server.build_frontend_data_bundle import _git_head_commit


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


def test_empty_cli_shows_help_instead_of_using_missing_demo_video():
    assert _normalize_argv([]) == ["--help"]


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


def test_paper_alignment_index_only_references_existing_scripts():
    module = importlib.import_module("scripts.paper.33_build_repo_report_alignment")
    assert module.ROOT == Path(__file__).resolve().parents[1]
    assert all((module.ROOT / row["script_path"]).exists() for row in module.SCRIPT_CATEGORY_SPECS)


def test_verifier_sample_split_is_deterministic_and_non_degenerate():
    rows = [
        {"sample_id": f"s_{index:07d}", "event_id": f"e_{index // 3:04d}"}
        for index in range(100)
    ]
    _assign_split(rows, 0.8)
    train_count = sum(row["split"] == "train" for row in rows)
    assert 65 <= train_count <= 95
    assert any(row["split"] == "val" for row in rows)
    by_event: dict[str, set[str]] = {}
    for row in rows:
        by_event.setdefault(row["event_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_event.values())

    repeated = [
        {"sample_id": row["sample_id"], "event_id": row["event_id"]}
        for row in rows
    ]
    _assign_split(repeated, 0.8)
    assert [row["split"] for row in repeated] == [row["split"] for row in rows]
