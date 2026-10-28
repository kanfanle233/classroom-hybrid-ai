"""CPU contracts for fair isolated benchmarks and truthful measurements."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from classroom_hybrid_ai import performance_benchmark as bench
from classroom_hybrid_ai.performance import (
    PerformanceRecorder, ResourceSampler, record_model_performance, recording,
)


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_trial_comparison_requires_paired_complete_runs_and_equal_semantics():
    baseline = {"variant": "baseline", "status": "ok", "wall_seconds": 10., "semantic_sha256": "same"}
    candidate = {"variant": "candidate", "status": "ok", "wall_seconds": 7., "semantic_sha256": "same"}
    assert bench.summarize_trials([baseline, candidate])["comparison"]["status"] == "target_met"
    assert bench.summarize_trials([baseline, candidate, baseline])["comparison"]["status"] == "incomplete_or_failed"
    result = bench.summarize_trials([baseline, {**candidate, "semantic_sha256": "different"}])
    assert result["comparison"]["outputs_equivalent_on_benchmark"] is False
    assert result["comparison"]["status"] == "target_not_met"


def test_repeated_semantic_instability_never_passes_equivalence_gate():
    trials = [{"variant": variant, "status": "ok", "wall_seconds": seconds, "semantic_sha256": value}
              for variant, seconds in (("baseline", 10), ("candidate", 5)) for value in ("a", "b")]
    assert bench.summarize_trials(trials)["comparison"]["outputs_equivalent_on_benchmark"] is False


def test_fixture_copy_relocates_every_nested_path_preserves_source_and_skips_responses(tmp_path):
    source, target = tmp_path / "frozen", tmp_path / "trial"
    source.mkdir()
    video = source / "original.mp4"
    video.write_bytes(b"source")
    image = source / "evidence/event/frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"pixels")
    evidence = [{"images": [str(image)], "nested": {"output_image": str(image), "source_video": str(video)}}]
    _jsonl(source / "evidence/index.jsonl", evidence)
    _jsonl(source / "responses/classification/cached.jsonl", [{"cached": True}])
    bench._write(source / "run_manifest.v1.json", {"source": {"path": str(video), "sha256": bench._sha(video)}})
    original = (source / "evidence/index.jsonl").read_bytes()
    bench._copy_fixture(source, target)
    copied = bench._rows(target / "evidence/index.jsonl")[0]
    assert copied["images"] == [str(target / "evidence/event/frame.jpg")]
    assert copied["nested"]["output_image"] == str(target / "evidence/event/frame.jpg")
    assert copied["nested"]["source_video"] == str(video)
    assert (source / "evidence/index.jsonl").read_bytes() == original
    assert not (target / "responses").exists()
    assert bench._read(target / "run_manifest.v1.json")["source"]["path"] == str(video)


def test_fixture_copy_refuses_source_or_descendant_destination(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        bench._copy_fixture(tmp_path, tmp_path / "child")


def test_qwen_bounding_uses_selected_order_and_same_ready_ids(tmp_path):
    _jsonl(tmp_path / "artifacts/candidates.selected.jsonl", [{"event_id": value} for value in ("b", "bad", "a", "c")])
    _jsonl(tmp_path / "artifacts/qwen_crop_queue.jsonl", [
        {"candidate_event_id": value, "status": "blocked" if value == "bad" else "ready"}
        for value in ("c", "bad", "a", "b")])
    bench._bound_qwen_fixture(tmp_path, 2)
    assert [row["event_id"] for row in bench._rows(tmp_path / "artifacts/candidates.selected.jsonl")] == ["b", "a"]
    assert {row["candidate_event_id"] for row in bench._rows(tmp_path / "artifacts/qwen_crop_queue.jsonl")} == {"a", "b"}


def test_qwen_bounding_refuses_empty_ready_workload(tmp_path):
    with pytest.raises(ValueError, match="no_ready"):
        bench._bound_qwen_fixture(tmp_path, 3)


def test_semantic_hash_ignores_trial_image_path_but_catches_pixel_and_candidate_changes(tmp_path):
    roots = [tmp_path / "baseline", tmp_path / "candidate"]
    for root in roots:
        image = root / "evidence/event/frame.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"same pixels")
        _jsonl(root / "evidence/index.jsonl", [{"images": [str(image)], "created_at": str(root)}])
        _jsonl(root / "artifacts/candidates.selected.jsonl", [{"event_id": "1", "track_id": "student-001"}])
        _jsonl(root / "reviews.classification.jsonl", [{"candidate_event_id": "1", "decision": "confirmed"}])
    before = [bench._canonical(bench._semantic_outputs(root, "qwen")) for root in roots]
    assert before[0] == before[1]
    (roots[1] / "evidence/event/frame.jpg").write_bytes(b"changed pixels")
    assert bench._canonical(bench._semantic_outputs(roots[1], "qwen")) != before[0]
    (roots[1] / "evidence/event/frame.jpg").write_bytes(b"same pixels")
    _jsonl(roots[1] / "artifacts/candidates.selected.jsonl", [{"event_id": "2", "track_id": "student-001"}])
    assert bench._canonical(bench._semantic_outputs(roots[1], "qwen")) != before[0]


def test_target_project_requires_matching_service_and_configs(tmp_path):
    source = tmp_path / "snapshot/src"
    (source / "classroom_hybrid_ai").mkdir(parents=True)
    with pytest.raises(ValueError, match="matching_configs"):
        bench._target_project(source)
    (source.parent / "configs").mkdir()
    (source.parent / "scripts").mkdir()
    assert bench._target_project(source) == source.parent


def test_actual_calls_use_model_telemetry_instead_of_attempt_count():
    recorder = PerformanceRecorder("test")
    with recording(recorder):
        record_model_performance("qwen", {"counts": {"inference_calls": 0}}, wall_seconds=1.)
        record_model_performance("qwen", {"counts": {"inference_calls": 2}, "wall_seconds": -1}, wall_seconds=3.)
        record_model_performance("qwen", None, wall_seconds=2.)
    assert recorder.counts["qwen.actual_calls"] == 2
    assert recorder.counts["qwen.request_attempts"] == 3
    assert recorder.counts["qwen.requests_without_inference_count"] == 1
    assert recorder.model_requests[1]["wall_seconds"] == 3.


def test_downsampling_preserves_observed_peak_and_io_deltas():
    sampler = ResourceSampler()
    for index in range(18001):
        sampler._observe({"process_tree_rss_bytes": 1000 if index == 2 else 1,
                          "parent_read_bytes": 2 * index, "parent_write_bytes": index})
    result = sampler.payload()
    assert len(sampler.samples) < 18001
    assert result["sampled_peak_process_tree_rss_bytes"] == 1000
    assert result["observed_delta_parent_read_bytes"] == 36000
    assert result["observed_delta_parent_write_bytes"] == 18000


@pytest.mark.parametrize("failure", ["invalid_json", "launch_error", "missing_result"])
def test_launch_failure_is_a_structured_trial(tmp_path, monkeypatch, failure):
    def run(*_args, **_kwargs):
        if failure == "launch_error":
            raise OSError("unavailable executable")
        if failure == "invalid_json":
            (tmp_path / "trial.result.json").write_text("broken", encoding="utf-8")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(bench.subprocess, "run", run)
    value = bench._launch({"name": "trial", "repository_root": str(tmp_path)}, tmp_path, Path(sys.executable))
    assert value["status"] == "failed"
    assert value["wall_seconds"] >= 0


def test_launch_uses_full_cold_process_time_and_rejects_stale_result(tmp_path, monkeypatch):
    clock = iter((100., 110.))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(clock))
    def run(*_args, **_kwargs):
        bench._write(tmp_path / "trial.result.json", {"status": "ok", "wall_seconds": 2., "stage_execution_seconds": 2.})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(bench.subprocess, "run", run)
    job = {"name": "trial", "repository_root": str(tmp_path)}
    value = bench._launch(job, tmp_path, Path(sys.executable))
    assert value["wall_seconds"] == 10.
    assert value["stage_execution_seconds"] == 2.
    assert bench._launch(job, tmp_path, Path(sys.executable))["status"] == "failed"


def test_benchmark_alternates_variants_and_saves_failed_trial(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test input")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("unused")
    config = SimpleNamespace(source_path=config_file, repository_root=tmp_path, python_executable=Path(sys.executable))
    names = []
    def launch(job, _assets, _python):
        names.append(job["name"])
        return {"status": "failed" if job["name"] == "002-candidate" else "ok",
                "wall_seconds": 10 if "baseline" in job["name"] else 5, "semantic_sha256": "same"}
    monkeypatch.setattr(bench, "_launch", launch)
    output = tmp_path / "report.json"
    report = bench.run_benchmark(config, video, component="io", repeats=3, output=output, baseline_config=config_file)
    assert names == ["001-baseline", "001-candidate", "002-baseline", "002-candidate"]
    assert report["status"] == "failed"
    assert bench._read(output)["comparison"]["status"] == "incomplete_or_failed"
    assert bench._read(output.with_name(output.name + ".progress.json"))["status"] == "failed"
    assert video.read_bytes() == b"test input"


def test_fixture_mismatch_is_rejected_before_creating_output_assets(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"input")
    fixture = tmp_path / "fixture"
    bench._write(fixture / "run_manifest.v1.json", {"source": {"sha256": "wrong"}})
    config = SimpleNamespace(source_path=tmp_path / "config.yaml", repository_root=tmp_path)
    with pytest.raises(ValueError, match="source_mismatch"):
        bench.run_benchmark(config, video, component="qwen", repeats=1, output=tmp_path / "report.json", run_dir=fixture)
    assert not (tmp_path / "report.assets").exists()


def test_child_imports_historical_source_and_does_not_mark_warning_as_success(tmp_path):
    """A real fresh process proves old source needs no new telemetry module."""
    project = tmp_path / "historical"
    source = project / "src"
    package = source / "classroom_hybrid_ai"
    package.mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "scripts").mkdir()
    (package / "__init__.py").write_text("")
    (package / "config.py").write_text('''
from pathlib import Path
class ProjectConfig:
    def __init__(self, source_path, project_root, repository_root, payload):
        self.source_path, self.project_root, self.repository_root, self.payload = source_path, project_root, repository_root, payload
    def get(self, key, default=None):
        value = self.payload
        for part in key.split('.'):
            if not isinstance(value, dict): return default
            value = value.get(part, default)
        return value
    @property
    def output_root(self): return Path(self.payload['paths']['output_root'])
def load_config(path):
    path = Path(path)
    return ProjectConfig(path, path.parent.parent, path.parent.parent, {})
''', encoding="utf-8")
    (package / "research_backend.py").write_text('''
from types import SimpleNamespace
def _effective_config(config, **kwargs): return config
class ResearchBackend:
    def __init__(self, config): self.config = config
    def profile(self): return SimpleNamespace(sha256='old-profile')
    def _build_spec(self, **kwargs):
        return SimpleNamespace(**kwargs, locate_model_sha256='locate', qwen_model_sha256='qwen')
''', encoding="utf-8")
    (package / "storage.py").write_text('''
class RunStore:
    def __init__(self, output, video, digest, unique_run=False):
        self.run_dir = output / video / 'historical-run'
        self.run_dir.mkdir(parents=True)
''', encoding="utf-8")
    (package / "research_stages.py").write_text('''
from types import SimpleNamespace
class ResearchStageExecutor:
    def __init__(self, spec): self.spec = spec
    def execute(self, stage, **kwargs):
        return SimpleNamespace(status='warning', errors=(), warnings=('historical-stage-incomplete',), stats={})
''', encoding="utf-8")
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video: historical stage does not decode")
    config = project / "configs/default.yaml"
    config.write_text("schema_version: unused")
    assets = tmp_path / "assets"
    assets.mkdir()
    result = bench._launch({
        "name": "historical", "source_path": str(source), "config": str(config),
        "repository_root": str(tmp_path), "video": str(video), "component": "io",
        "trial_dir": str(tmp_path / "trial"),
    }, assets, Path(sys.executable))
    assert result.get("error") is None, result
    assert result["status"] == "failed"
    assert result["stage_results"]["S0"]["warnings"] == ["historical-stage-incomplete"]
    assert "S1" not in result["stage_results"]
    assert result["target_project_root"] == str(project)
    assert result["source_changed_during_trial"] is False
    assert result["process_exitcode"] == 1


def test_lifecycle_summary_deduplicates_cumulative_loading_across_reviews():
    lifecycle = {"worker_pid": 42, "timings_seconds": {"model_load": 4.0}, "counts": {"model_load_attempts": 1}}
    observations = [{"performance": {"worker_lifecycle": lifecycle}} for _ in range(100)]
    summary = bench._collect_model_lifecycles(observations, {"model_requests": observations})
    assert len(summary) == 1
    assert summary[0]["timings_seconds"]["model_load"] == 4.0
    assert summary[0]["counts"]["model_load_attempts"] == 1


@pytest.mark.parametrize("field,value", [("repeats", True), ("repeats", 1.5), ("max_frames", False), ("max_frames", 2.5)])
def test_invalid_limits_fail_before_creating_outputs(tmp_path, field, value):
    with pytest.raises(ValueError):
        bench.run_benchmark(SimpleNamespace(), tmp_path / "unused.mp4", output=tmp_path / "report.json", **{field: value})
    assert list(tmp_path.iterdir()) == []


def test_output_inside_frozen_fixture_is_rejected_without_writes(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"original")
    fixture = tmp_path / "fixture"
    bench._write(fixture / "run_manifest.v1.json", {"source": {"sha256": bench._sha(video)}})
    config = SimpleNamespace(source_path=tmp_path / "config.yaml", repository_root=tmp_path)
    before = {path.relative_to(fixture): path.read_bytes() for path in fixture.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="outside_fixture"):
        bench.run_benchmark(config, video, component="qwen", run_dir=fixture, output=fixture / "nested/report.json")
    assert {path.relative_to(fixture): path.read_bytes() for path in fixture.rglob("*") if path.is_file()} == before
    assert not (fixture / "nested").exists()


def test_transparent_request_observer_preserves_arguments_result_and_error(monkeypatch):
    from classroom_hybrid_ai.adapters.locate_anything import LocateAnythingAdapter
    sentinel = object()
    calls = []
    def original(self, *args, **kwargs):
        calls.append((args, kwargs))
        if kwargs.get("fail"):
            raise RuntimeError("original error")
        return sentinel
    monkeypatch.setattr(LocateAnythingAdapter, "locate", original)
    adapter = LocateAnythingAdapter()
    observations = []
    with bench._observe_requests(observations):
        assert adapter.locate("image", "prompt", max_new_tokens=256) is sentinel
        with pytest.raises(RuntimeError, match="original error"):
            adapter.locate("image", "prompt", fail=True)
    assert LocateAnythingAdapter.locate is original
    assert calls == [(("image", "prompt"), {"max_new_tokens": 256}), (("image", "prompt"), {"fail": True})]
    assert [row["status"] for row in observations] == ["ok", "failed"]
    assert all(row["wall_seconds"] >= 0 for row in observations)


def test_old_source_request_percentiles_use_shared_host_observer():
    trials = [{"variant": variant, "status": "ok", "wall_seconds": 10., "semantic_sha256": "same", "requests": [],
               "request_observations": [{"wall_seconds": 1.}, {"wall_seconds": 3.}]}
              for variant in ("baseline", "candidate")]
    summary = bench.summarize_trials(trials)
    assert summary["variants"]["baseline"]["request_p95_seconds"] == pytest.approx(2.9)
    assert summary["variants"]["candidate"]["request_p95_seconds"] == pytest.approx(2.9)


def _complete_pipeline_fixture(root):
    """Old v3.1 paths, intentionally zero students and zero candidates."""
    for relative in bench._PIPELINE_JSONL:
        _jsonl(root / relative, [])
    for relative in bench._PIPELINE_JSON:
        bench._write(root / relative, {})
    bench._write(root / "timeline.v2.json", {"intervals": [], "student_rows": [], "duration_seconds": .3})
    receipts = {}
    for name in bench._PIPELINE_VIDEOS:
        path = root / f"videos/{name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub media")
        receipts[name] = {
            "uri": f"videos/{name}.mp4", "path": str(path), "sha256": bench._sha(path),
            "frames": 3, "fps": 10., "semantic_id": name, "semantic": {"kind": name},
            "clip_ranges": [], "frame_ranges": [], "audio_preserved": False, "audio_status": "unavailable",
        }
    bench._write(root / "manifest.json", {"videos": receipts, "config_sha256": str(root), "runtime": {}})
    bench._write(root / "frontend/index.v1.json", {
        "video_uris": {name: receipt["uri"] for name, receipt in receipts.items()},
        "video_metadata": {name: {"frames": 3, "semantic_id": name} for name in receipts},
        "overlay_chunks": ["frontend/overlay_chunks/000000.json"],
        "students": [],
    })
    bench._write(root / "frontend/overlay_chunks/000000.json", {"rows": [], "start_pts_seconds": 0., "end_pts_seconds": 60.})


def _verified_stub_media(path):
    return {"status": "passed", "video": {
        "frames": 3, "fps": 10., "shape": [16, 16, 3], "decoded_rgb_sha256": "pixels",
        "pts_sha256": "pts", "first_pts_seconds": 0., "last_pts_seconds": .2,
    }, "audio": {"stream_count": 0, "streams": []}}


def test_pipeline_zero_students_is_complete_and_runtime_fields_do_not_change_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "_media_signature", _verified_stub_media)
    roots = [tmp_path / "baseline", tmp_path / "candidate"]
    for root in roots:
        _complete_pipeline_fixture(root)
    _jsonl(roots[0] / "action_observations.jsonl", [{"action_id": "UNKNOWN", "created_at": "old", "performance": {"seconds": 99}}])
    _jsonl(roots[1] / "action_observations.jsonl", [{"action_id": "UNKNOWN", "created_at": "new", "performance": {"seconds": 1}}])
    results = [bench._pipeline_semantics(root) for root in roots]
    assert all(result["validation"]["status"] == "passed" for result in results)
    assert bench._canonical(results[0]) == bench._canonical(results[1])


@pytest.mark.parametrize("relative", [
    "action_observations.jsonl", "artifacts/track_observations.jsonl",
    "artifacts/locate_student_boxes.jsonl", "artifacts/overlay_rows.v3.jsonl",
    "frontend/action_observations_display.v1.jsonl", "reviews.classification.jsonl",
])
def test_pipeline_semantics_detect_previously_uncompared_structured_changes(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(bench, "_media_signature", _verified_stub_media)
    _complete_pipeline_fixture(tmp_path)
    before = bench._canonical(bench._pipeline_semantics(tmp_path))
    _jsonl(tmp_path / relative, [{"student_id": "student-002", "primary_action_id": "UNKNOWN", "observability": "none"}])
    assert bench._canonical(bench._pipeline_semantics(tmp_path)) != before


@pytest.mark.parametrize("relative", [
    "action_observations.jsonl", "timeline.v2.json", "artifacts/track_observations.jsonl",
    "frontend/overlay_chunks/000000.json", "videos/tracking_preview.mp4",
    "videos/action_preview.mp4", "videos/annotated_final.mp4",
])
def test_pipeline_missing_required_output_never_verifies(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(bench, "_media_signature", _verified_stub_media)
    _complete_pipeline_fixture(tmp_path)
    (tmp_path / relative).unlink()
    result = bench._pipeline_semantics(tmp_path)
    assert result["validation"]["status"] == "failed"
    assert result["validation"]["errors"]


def test_pipeline_timeline_and_overlay_frontend_changes_are_compared(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "_media_signature", _verified_stub_media)
    _complete_pipeline_fixture(tmp_path)
    before = bench._canonical(bench._pipeline_semantics(tmp_path))
    bench._write(tmp_path / "frontend/overlay_chunks/000000.json", {"rows": [{"color": "#FF0000"}]})
    assert bench._canonical(bench._pipeline_semantics(tmp_path)) != before
    bench._write(tmp_path / "timeline.v2.json", {"intervals": [{"action_id": "UNKNOWN"}], "student_rows": []})
    assert "timeline_intervals_disagree_with_action_intervals" in bench._pipeline_semantics(tmp_path)["validation"]["errors"]


@pytest.mark.parametrize("validation", [None, {}, {"schema_version": "pipeline_semantic_validation.v1", "status": "failed"}])
def test_pipeline_summary_requires_explicit_complete_new_semantic_audit(validation):
    rows = [{"variant": variant, "component": "pipeline", "status": "ok", "wall_seconds": seconds,
             "semantic_sha256": "same", "semantic_validation": validation}
            for variant, seconds in (("baseline", 10.), ("candidate", 5.))]
    assert bench.summarize_trials(rows)["comparison"]["status"] == "incomplete_or_failed"
    for row in rows:
        row["semantic_validation"] = {"schema_version": "pipeline_semantic_validation.v1", "status": "passed"}
    assert bench.summarize_trials(rows)["comparison"]["status"] == "target_met"


def test_pipeline_legacy_encoder_receipt_is_opencv_and_nvenc_is_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "_media_signature", _verified_stub_media)
    _complete_pipeline_fixture(tmp_path)
    old = bench._pipeline_semantics(tmp_path)
    manifest = bench._read(tmp_path / "manifest.json")
    for receipt in manifest["videos"].values():
        receipt["encoder"] = {"requested": "opencv", "effective": "opencv", "source_decode_passes": 1}
    bench._write(tmp_path / "manifest.json", manifest)
    assert bench._canonical(bench._pipeline_semantics(tmp_path)) == bench._canonical(old)
    manifest["videos"]["annotated_final"]["encoder"]["requested"] = "nvenc"
    bench._write(tmp_path / "manifest.json", manifest)
    assert bench._canonical(bench._pipeline_semantics(tmp_path)) != bench._canonical(old)


def test_pipeline_unreadable_video_and_frame_count_mismatch_cannot_pass(tmp_path, monkeypatch):
    _complete_pipeline_fixture(tmp_path)
    monkeypatch.setattr(bench, "_media_signature", lambda _: {"status": "failed", "reason": "decoder_missing"})
    assert bench._pipeline_semantics(tmp_path)["validation"]["status"] == "failed"
    def truncated(path):
        value = _verified_stub_media(path)
        value["video"]["frames"] = 2
        return value
    monkeypatch.setattr(bench, "_media_signature", truncated)
    assert "video_frame_count_mismatch:annotated_final" in bench._pipeline_semantics(tmp_path)["validation"]["errors"]


def _real_media(path, *, pixel_offset=0, audio_value=None):
    av = pytest.importorskip("av")
    import numpy as np
    from fractions import Fraction
    with av.open(str(path), "w") as output:
        video = output.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 32, 24, "yuv420p"
        audio = output.add_stream("pcm_s16le", rate=48000) if audio_value is not None else None
        if audio is not None:
            audio.codec_context.layout = "mono"
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 30 + pixel_offset, np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
        if audio is not None:
            for index in range(3):
                frame = av.AudioFrame.from_ndarray(np.full((1, 4800), audio_value, np.int16), format="s16", layout="mono")
                frame.sample_rate, frame.pts, frame.time_base = 48000, index * 4800, Fraction(1, 48000)
                for packet in audio.encode(frame):
                    output.mux(packet)
            for packet in audio.encode():
                output.mux(packet)


def test_media_digest_decodes_all_pixels_pts_and_audio_content(tmp_path):
    paths = [tmp_path / f"{name}.mkv" for name in ("same1", "same2", "pixels", "audio", "silent")]
    for path, offset, value in zip(paths, (0, 0, 10, 0, 0), (100, 100, 100, 200, None)):
        _real_media(path, pixel_offset=offset, audio_value=value)
    values = [bench._media_signature(path) for path in paths]
    assert all(value["status"] == "passed" for value in values), values
    assert values[0] == values[1]
    assert values[0]["video"]["frames"] == 3
    assert values[0]["video"]["first_pts_seconds"] == 0
    assert values[0]["video"]["last_pts_seconds"] == pytest.approx(.2)
    assert values[0]["video"]["decoded_rgb_sha256"] != values[2]["video"]["decoded_rgb_sha256"]
    assert values[0]["audio"]["streams"][0]["decoded_samples_and_pts_sha256"] != values[3]["audio"]["streams"][0]["decoded_samples_and_pts_sha256"]
    assert values[4]["audio"] == {"stream_count": 0, "streams": []}
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    assert bench._media_signature(corrupt)["status"] == "failed"


@pytest.mark.parametrize("change", ["pixels", "pts", "lost_audio"])
def test_pipeline_equality_gate_detects_media_regressions(tmp_path, monkeypatch, change):
    roots = [tmp_path / "baseline", tmp_path / "candidate"]
    for root in roots:
        _complete_pipeline_fixture(root)
    def media(path):
        value = _verified_stub_media(path)
        if change == "lost_audio" and "baseline" in path.parts:
            value["audio"] = {"stream_count": 1, "streams": [{"decoded_samples_and_pts_sha256": "audio"}]}
        if "candidate" in path.parts:
            if change == "pixels":
                value["video"]["decoded_rgb_sha256"] = "changed"
            elif change == "pts":
                value["video"]["pts_sha256"] = "shifted"
        return value
    monkeypatch.setattr(bench, "_media_signature", media)
    rows = []
    for root in roots:
        semantics = bench._pipeline_semantics(root)
        assert semantics["validation"]["status"] == "passed"
        rows.append({"variant": root.name, "component": "pipeline", "status": "ok",
                     "wall_seconds": 10. if root.name == "baseline" else 5.,
                     "semantic_sha256": bench._canonical(semantics), "semantic_validation": semantics["validation"]})
    result = bench.summarize_trials(rows)["comparison"]
    assert result["status"] == "target_not_met"
    assert result["outputs_equivalent_on_benchmark"] is False


@pytest.mark.parametrize("pts", [(None, 1), (0, float("nan")), (0, 0), (1, 0)])
def test_media_invalid_pts_cannot_be_hidden_by_nominal_fps(tmp_path, monkeypatch, pts):
    from contextlib import nullcontext
    import numpy as np
    frames = [SimpleNamespace(pts=value, time_base=.1, is_corrupt=False,
                              to_ndarray=lambda **kwargs: np.zeros((16, 16, 3), dtype=np.uint8)) for value in pts]
    stream = SimpleNamespace(average_rate=10, frames=2, codec_context=SimpleNamespace(name="test"))
    container = SimpleNamespace(streams=SimpleNamespace(video=[stream], audio=[]), decode=lambda **kwargs: iter(frames))
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda *_args: nullcontext(container)))
    assert bench._media_signature(tmp_path / "test.mp4")["status"] == "failed"


def test_media_missing_pyav_is_unverified_not_silent_opencv_fallback(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)
    result = bench._media_signature(tmp_path / "test.mp4")
    assert result["status"] == "failed"
    assert "ModuleNotFoundError" in result["reason"]
