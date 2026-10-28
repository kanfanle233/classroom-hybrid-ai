"""CPU-only acceptance of locked whole-video experiment orchestration."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import yaml

from classroom_hybrid_ai import performance_study as study


def make_manifest(tmp_path):
    project = tmp_path / "project"
    package = project / "src/classroom_hybrid_ai"
    package.mkdir(parents=True)
    (package / "research_backend.py").write_text("# source\n", encoding="utf-8")
    (project / "scripts").mkdir()
    (project / "configs").mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"locked weights")
    (model / "tokenizer.json").write_bytes(b"{}")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"locked video")
    config = project / "configs/default.yaml"
    config.write_text(yaml.safe_dump({"qwen": {"enabled": True, "model_dir": str(model)},
                                     "locate_anything": {"model_dir": str(model)}}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    study.write_study_manifest(manifest, repository_root=tmp_path,
        videos=[{"id": "lesson", "path": str(video)}],
        variants=[{"id": name, "source_path": str(project / "src"), "config": str(config)} for name in ("original", "candidate")],
        comparison_type="engineering_equivalence", baseline_variant="original")
    return manifest


def _save_manifest(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def no_decode(monkeypatch):
    import classroom_hybrid_ai.video as video
    monkeypatch.setattr(video, "probe_video", lambda _: SimpleNamespace(duration_seconds=60., frame_count=1200))


def mock_trials(monkeypatch, *, stop_after=None):
    calls = []
    def launch(job, assets, python):
        calls.append(dict(job))
        run = Path(job["trial_dir"]) / "runs/real"
        run.mkdir(parents=True)
        (run / "evidence.jpg").write_bytes(b"immutable evidence")
        if stop_after and len(calls) == stop_after:
            (assets.parent / "STOP").touch()
        return {"status": "ok", "component": "pipeline", "run_dir": str(run), "semantic_sha256": "same",
                "semantic_validation": {"status": "passed"}, "seed_control": {"status": "passed"},
                "wall_seconds": 10 if "original" in job["name"] else 7,
                "request_observations": [{"model": "qwen", "wall_seconds": 1.}]}
    monkeypatch.setattr(study, "_launch", launch)
    return calls


def test_manifest_hashes_model_tokenizer_and_source(tmp_path):
    manifest = make_manifest(tmp_path)
    assert study.load_study(manifest)["seed"] == 0
    (tmp_path / "model/tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset_sha_mismatch"):
        study.load_study(manifest)


@pytest.mark.parametrize("field,value", [("max_frames", None), ("max_requests", 1), ("run_dir", "fixture"),
                                         ("skip_qwen", False), ("skip_stages", [])])
def test_bounded_or_skipped_studies_rejected_even_false(tmp_path, field, value):
    manifest = make_manifest(tmp_path)
    payload = study._read(manifest)
    payload[field] = value
    _save_manifest(manifest, payload)
    with pytest.raises(ValueError, match="forbidden"):
        study.load_study(manifest)


@pytest.mark.parametrize("field,value,match", [("repeats", True, "three_repeats"), ("repeats", 2, "three_repeats"),
    ("seed", True, "uint32"), ("seed", -1, "uint32"), ("seed", 2**32, "uint32"),
    ("delivery", "no_video", "full_delivery"), ("comparison_type", "faster", "comparison_type")])
def test_invalid_study_controls(tmp_path, field, value, match):
    manifest = make_manifest(tmp_path)
    payload = study._read(manifest)
    payload[field] = value
    _save_manifest(manifest, payload)
    with pytest.raises(ValueError, match=match):
        study.load_study(manifest)


def test_dry_run_no_model_no_output_and_unknown_time(tmp_path, monkeypatch, no_decode):
    manifest = make_manifest(tmp_path)
    monkeypatch.setattr(study, "_launch", lambda *a, **k: pytest.fail("dry-run launched model"))
    result = study.run_study(manifest, output=tmp_path / "out", dry_run=True)
    assert result["status"] == "planned" and result["estimates"]["trials"] == 6
    assert result["estimates"]["per_trial"][0]["estimated_wall_seconds_range"] is None
    assert not (tmp_path / "out").exists()


def test_rotating_full_pipeline_cold_trials_and_no_gold_no_promotion(tmp_path, monkeypatch, no_decode):
    manifest = make_manifest(tmp_path)
    calls = mock_trials(monkeypatch)
    result = study.run_study(manifest, output=tmp_path / "out", python=Path(sys.executable))
    assert [row["name"].split("-")[1] for row in calls] == ["original", "candidate", "candidate", "original", "original", "candidate"]
    assert all(row["component"] == "pipeline" and row["max_frames"] is None and row["max_requests"] is None and row["run_dir"] is None and row["seed"] == 0 for row in calls)
    comparison = result["comparisons"]["candidate"]
    assert result["status"] == "ok" and comparison["outputs_equivalent"] is True
    assert comparison["median_reduction_fraction"] == pytest.approx(.30)
    assert comparison["quality"]["status"] == "not_evaluated"
    assert comparison["default_promotion_eligible"] is False
    # Independent audit rechecks and reuses completed trials without reruns.
    second = study.run_study(manifest, output=tmp_path / "out")
    assert len(calls) == 6 and len(second["resumed_verified_trials"]) == 6


def test_resume_detects_artifact_content_change(tmp_path, monkeypatch, no_decode):
    manifest = make_manifest(tmp_path)
    mock_trials(monkeypatch)
    report = study.run_study(manifest, output=tmp_path / "out")
    (Path(report["trials"][0]["run_dir"]) / "evidence.jpg").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact_changed"):
        study.run_study(manifest, output=tmp_path / "out")


def test_cooperative_stop_and_verified_trial_resume(tmp_path, monkeypatch, no_decode):
    manifest = make_manifest(tmp_path)
    calls = mock_trials(monkeypatch, stop_after=2)
    report = study.run_study(manifest, output=tmp_path / "out")
    assert report["status"] == "paused_by_user" and len(calls) == 2
    (tmp_path / "out/STOP").unlink()
    report = study.run_study(manifest, output=tmp_path / "out")
    assert report["status"] == "ok" and len(calls) == 6


def test_unreceipted_interrupted_trial_never_reexecuted(tmp_path, monkeypatch, no_decode):
    manifest = make_manifest(tmp_path)
    calls = mock_trials(monkeypatch)
    target = tmp_path / "out/trials"
    target.mkdir(parents=True)
    (target / "001-original-lesson.job.json").write_text("{}", encoding="utf-8")
    report = study.run_study(manifest, output=tmp_path / "out")
    assert report["status"] == "needs_attention" and calls == []


def test_atomic_receipt_recovers_crash_before_progress_commit(tmp_path, monkeypatch, no_decode):
    manifest = make_manifest(tmp_path)
    calls = mock_trials(monkeypatch)
    original = study.run_study(manifest, output=tmp_path / "out")
    progress = deepcopy(original)
    lost = progress["trials"].pop(0)
    study._write(tmp_path / "out/progress.json", progress)
    result = study.run_study(manifest, output=tmp_path / "out")
    assert len(calls) == 6 and len(result["trials"]) == 6
    assert lost in result["trials"]


def test_source_edit_invalidates_before_launch(tmp_path, monkeypatch, no_decode):
    manifest = make_manifest(tmp_path)
    (tmp_path / "project/src/classroom_hybrid_ai/research_backend.py").write_text("# changed", encoding="utf-8")
    calls = mock_trials(monkeypatch)
    with pytest.raises(ValueError, match="source_sha_mismatch"):
        study.run_study(manifest, output=tmp_path / "out")
    assert calls == []


def _quality_fixture(tmp_path):
    basic = {name: .8 for name in study._METRICS}
    basic.update(per_class={code: {"f1": .8, "support_seconds": 20} for code in study.CORE8_CODES},
                 per_lesson_metrics=[{"lesson_id": name, **{metric: .8 for metric in study._METRICS}} for name in ("x", "y")],
                 uncertainty={"method": "lesson_bootstrap", "confidence_level": .95, "intervals": {metric: [.7, .9] for metric in study._METRICS}})
    trials = [{"trial_id": "original-x", "semantic_sha256": "a"}, {"trial_id": "candidate-x", "semantic_sha256": "b"}]
    payload = {"schema_version": "performance_study_quality.v1", "status": "evaluated",
               "independent_gold": True, "provenance_review_status": "passed", "source_video_sha256": {"x": "video"},
               "trial_semantic_sha256": {row["trial_id"]: row["semantic_sha256"] for row in trials},
               "variants": {"original": deepcopy(basic), "candidate": deepcopy(basic)}}
    assets = [{"id": name, "source_sha256": "source", "config_sha256": "config", "models": [{"id": "model", "sha256": "weights"}]} for name in ("original", "candidate")]
    payload["variant_asset_sha256"] = {row["id"]: {"source": "source", "config": "config", "models": {"model": "weights"}} for row in assets}
    path = tmp_path / "quality.json"
    study._write(path, payload)
    spec = {"baseline_variant": "original", "videos": [{"id": "x", "sha256": "video"}], "variants": assets,
            "quality_report": {"path": str(path), "sha256": study._sha(path)}}
    return payload, path, spec, trials


@pytest.mark.parametrize("metric", list(study._METRICS))
def test_each_quality_regression_blocks(tmp_path, metric):
    payload, path, spec, trials = _quality_fixture(tmp_path)
    payload["variants"]["candidate"][metric] += .011 if metric == "unknown_time_fraction" else -.011
    study._write(path, payload)
    spec["quality_report"]["sha256"] = study._sha(path)
    result = study._quality_gate(spec, trials, "candidate")
    assert f"quality_regression:{metric}" in result["errors"]


def test_quality_class_absence_and_support_mismatch_block(tmp_path):
    payload, path, spec, trials = _quality_fixture(tmp_path)
    assert study._quality_gate(spec, trials, "candidate")["status"] == "passed"
    payload["variants"]["candidate"]["per_class"]["dx"]["support_seconds"] = 0
    study._write(path, payload)
    spec["quality_report"]["sha256"] = study._sha(path)
    assert "quality_class_support_or_f1_invalid:dx" in study._quality_gate(spec, trials, "candidate")["errors"]


@pytest.mark.parametrize("field,value", [("source_video_sha256", {}), ("trial_semantic_sha256", {}),
                                         ("independent_gold", False), ("provenance_review_status", "pending")])
def test_quality_claims_require_actual_trial_source_provenance_bindings(tmp_path, field, value):
    payload, path, spec, trials = _quality_fixture(tmp_path)
    payload[field] = value
    study._write(path, payload)
    spec["quality_report"]["sha256"] = study._sha(path)
    assert study._quality_gate(spec, trials, "candidate")["status"] == "failed"


def test_missing_uncertainty_intervals_and_single_lesson_never_pass(tmp_path):
    payload, path, spec, trials = _quality_fixture(tmp_path)
    payload["variants"]["candidate"]["per_lesson_metrics"].pop()
    payload["variants"]["candidate"]["uncertainty"]["intervals"].pop("hota")
    study._write(path, payload)
    spec["quality_report"]["sha256"] = study._sha(path)
    result = study._quality_gate(spec, trials, "candidate")
    assert "quality_per_lesson_metrics_missing:candidate" in result["errors"]
    assert "quality_uncertainty_interval_invalid:candidate:hota" in result["errors"]


def test_posthoc_quality_audit_does_not_rerun_or_edit_original_study(tmp_path, monkeypatch, no_decode):
    manifest = make_manifest(tmp_path)
    calls = mock_trials(monkeypatch)
    report = study.run_study(manifest, output=tmp_path / "out")
    before = (tmp_path / "out/study.json").read_bytes()
    quality = tmp_path / "not_yet_gold.json"
    study._write(quality, {"schema_version": "performance_study_quality.v1", "status": "not_evaluated"})
    result = study.assess_study_quality(tmp_path / "out", quality)
    assert result["status"] == "not_passed" and len(calls) == 6
    assert (tmp_path / "out/study.json").read_bytes() == before


def test_seed_receipt_covers_locate_qwen_and_controller(tmp_path):
    directory = tmp_path / "seed"
    environment = study.prepare_seed_control(directory, 0)
    assert environment["PYTHONHASHSEED"] == "0"
    trial = {"child_pid": 10, "run_dir": str(tmp_path / "run"),
             "request_observations": [{"model": "qwen", "worker_pid": 11}, {"model": "locate"}, {"model": "yolo", "worker_pid": 13}]}
    study._write(tmp_path / "run/diagnostics/locate_service_receipt.v1.json", {"owned_process": True, "pid": 12})
    for pid in (10, 11):
        study._write(directory / "audit" / f"{pid}.json", {"pid": pid, "seed": 0, "status": "ok"})
    assert study.verify_seed_control(directory, 0, trial)["status"] == "failed"
    study._write(directory / "audit/12.json", {"pid": 12, "seed": 0, "status": "ok"})
    assert study.verify_seed_control(directory, 0, trial)["status"] == "failed"
    study._write(directory / "audit/13.json", {"pid": 13, "seed": 0, "status": "ok"})
    assert study.verify_seed_control(directory, 0, trial)["status"] == "passed"


def test_engineering_four_variants_modify_real_executable_source(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    original = repository / ".runtime/performance_baselines/20260909T151631Z/classroom_hybrid_ai"
    donor = repository / ".runtime/performance_candidates/20260913T062453Z/classroom_hybrid_ai"
    if not original.exists() or not donor.exists():
        pytest.skip("historical source snapshots unavailable outside research workspace")
    result = study.build_engineering_variants(original, donor, output=tmp_path / "variants")
    assert result["status"] == "built_not_performance_validated"
    assert len({row["source_sha256"] for row in result["variants"]}) == 4
    request = (tmp_path / "variants/request_overhead/src/classroom_hybrid_ai/adapters/locate_anything.py").read_text(encoding="utf-8")
    node = study._method_span(request, "_local_answer")
    import ast
    final_calls = [n for block in ast.walk(node) if isinstance(block, ast.Try) for n in block.finalbody
                   if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "_empty_cuda_cache"]
    assert final_calls == []
    assert "except BaseException:" in request
    combined = (tmp_path / "variants/combined/src/classroom_hybrid_ai/research_stages.py").read_text(encoding="utf-8")
    assert "preprocess_workers=2" in combined and "scan_and_materialize_screenshots(" in combined
