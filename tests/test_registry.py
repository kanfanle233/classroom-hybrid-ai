"""
Unit tests for server/registry.py — CaseRegistry, ArtifactResolver, GoldLoader, evaluator.
"""
import json
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.registry import (
    ArtifactResolver,
    ArtifactStatus,
    CaseManifestV1,
    CaseRegistry,
    GoldEventV1,
    GoldLoader,
    GoldBackedEvaluator,
    VerifierEvalReportV1,
    BaselineResult,
    ARTIFACT_KEYS,
    _load_jsonl,
    _load_json,
    _validate_gold,
    _eval_baseline,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def gold_dir():
    return PROJECT_ROOT / "paper_experiments" / "gold"


@pytest.fixture
def gold_loader(gold_dir):
    loader = GoldLoader(gold_dir)
    loader.load()
    return loader


@pytest.fixture
def case_registry(project_root):
    return CaseRegistry(project_root)


@pytest.fixture
def output_case_dir(project_root):
    """Find the first available case dir in output/."""
    output = project_root / "output"
    for d in output.rglob("*"):
        if d.is_dir() and (d / "verified_events.jsonl").exists():
            return d
    pytest.skip("No case dir with verified_events.jsonl found in output/")


# ── ArtifactResolver tests ────────────────────────────────────────

class TestArtifactResolver:
    def test_resolve_all_returns_status_for_each_key(self, output_case_dir):
        resolver = ArtifactResolver(output_case_dir)
        statuses = resolver.resolve_all()
        assert len(statuses) == len(ARTIFACT_KEYS)
        found_keys = {s.key for s in statuses if s.found}
        assert "verified_events" in found_keys
        assert "align_multimodal" in found_keys

    def test_missing_artifact_has_no_path(self, output_case_dir):
        resolver = ArtifactResolver(output_case_dir)
        statuses = resolver.resolve_all()
        for s in statuses:
            if not s.found:
                assert s.file_path is None
                assert s.size_bytes is None

    def test_found_artifact_has_path_and_size(self, output_case_dir):
        resolver = ArtifactResolver(output_case_dir)
        ve_status = next(s for s in resolver.resolve_all() if s.key == "verified_events")
        assert ve_status.found is True
        assert ve_status.file_path is not None
        assert ve_status.size_bytes is not None
        assert ve_status.size_bytes > 0

    def test_get_path_returns_path_for_existing(self, output_case_dir):
        resolver = ArtifactResolver(output_case_dir)
        p = resolver.get_path("verified_events")
        assert p is not None
        assert p.exists()

    def test_get_path_returns_none_for_missing(self, output_case_dir):
        resolver = ArtifactResolver(output_case_dir)
        p = resolver.get_path("verifier_model")
        assert p is None

    def test_read_jsonl_returns_list(self, output_case_dir):
        resolver = ArtifactResolver(output_case_dir)
        rows = resolver.read_jsonl("verified_events")
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_read_jsonl_empty_for_missing(self, output_case_dir):
        resolver = ArtifactResolver(output_case_dir)
        rows = resolver.read_jsonl("verifier_model")
        assert rows == []

    def test_read_json_returns_dict(self, output_case_dir):
        resolver = ArtifactResolver(output_case_dir)
        data = resolver.read_json("align_multimodal")
        assert isinstance(data, list)


class TestArtifactResolverMissingDir:
    def test_nonexistent_dir_returns_all_missing(self, tmp_path):
        resolver = ArtifactResolver(tmp_path / "nonexistent")
        statuses = resolver.resolve_all()
        assert all(not s.found for s in statuses)


# ── GoldLoader tests ──────────────────────────────────────────────

class TestGoldLoader:
    def test_load_defaults_to_active_gold_file(self, gold_loader):
        events = gold_loader.load()
        assert gold_loader.gold_path.name == "gold_events.official.visual_223.jsonl"
        assert len(events) == gold_loader.summary("official")["official_total"]
        assert all(ev.review_status in ("peer_reviewed", "adjudicated", "final") for ev in events)

    def test_load_defaults_to_official_events(self, gold_loader):
        events = gold_loader.load()
        assert len(events) == gold_loader.summary("official")["official_total"]
        assert all(ev.review_status in ("peer_reviewed", "adjudicated", "final") for ev in events)

    def test_load_all_includes_non_official_candidates(self, gold_loader):
        official = gold_loader.load("official")
        all_events = gold_loader.load("all")
        assert len(all_events) >= len(official)
        assert len(official) == sum(
            1 for ev in all_events if ev.review_status in ("peer_reviewed", "adjudicated", "final")
        )

    def test_validates_cleanly(self, gold_loader):
        assert gold_loader.is_valid()
        assert gold_loader.validation_errors == []

    def test_by_case_groups_correctly(self, gold_loader):
        by_case = gold_loader.by_case()
        events = gold_loader.load("official")
        assert set(by_case) == {ev.case_id for ev in events}
        for case_id, case_events in by_case.items():
            assert len(case_events) == sum(1 for ev in events if ev.case_id == case_id)

    def test_summary_has_correct_totals(self, gold_loader):
        summary = gold_loader.summary()
        assert summary["quality_filter"] == "official"
        assert summary["total"] == summary["official_total"]
        assert summary["included_total"] == summary["total"]
        assert summary["excluded_total"] == summary["total_loaded"] - summary["included_total"]
        assert "excluded_reason_distribution" in summary
        assert "excluded_by_review_status" in summary
        assert summary["total_loaded"] >= summary["total"]
        assert summary["total"] == sum(summary["label_distribution"].values())
        assert summary["official_visual_gold"] == summary["official_total"]

    def test_gold_event_fields(self, gold_loader):
        ev = gold_loader.load()[0]
        assert isinstance(ev, GoldEventV1)
        assert ev.case_id != ""
        assert ev.suggested_event_id != ""
        assert ev.match_label in ("match", "mismatch", "uncertain")
        assert hasattr(ev, "batch_id")
        assert hasattr(ev, "review_round")
        assert hasattr(ev, "adjudicator")

    def test_summary_reports_excluded_reasons_for_filtered_quality(self, tmp_path):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        rows = [
            {
                "case_id": "case_official", "view_code": "front", "suggested_event_id": "e1",
                "suggested_track_id": 1, "suggested_action": "stand",
                "suggested_time_window": {"start": 0, "end": 1},
                "suggested_label": "match", "gold_track_id": "1",
                "gold_action": "stand", "gold_start": 0, "gold_end": 1,
                "match_label": "match", "noise_type": "none",
                "occlusion_level": "low", "asr_quality": "high",
                "annotator": "test", "review_status": "adjudicated", "notes": "",
            },
            {
                "case_id": "case_draft", "view_code": "front", "suggested_event_id": "e2",
                "suggested_track_id": 1, "suggested_action": "stand",
                "suggested_time_window": {"start": 0, "end": 1},
                "suggested_label": "match", "gold_track_id": "1",
                "gold_action": "stand", "gold_start": 0, "gold_end": 1,
                "match_label": "match", "noise_type": "none",
                "occlusion_level": "low", "asr_quality": "high",
                "annotator": "test", "review_status": "draft", "notes": "",
            },
            {
                "case_id": "case_rejected", "view_code": "front", "suggested_event_id": "e3",
                "suggested_track_id": 1, "suggested_action": "stand",
                "suggested_time_window": {"start": 0, "end": 1},
                "suggested_label": "match", "gold_track_id": "1",
                "gold_action": "stand", "gold_start": 0, "gold_end": 1,
                "match_label": "match", "noise_type": "none",
                "occlusion_level": "low", "asr_quality": "high",
                "annotator": "test", "review_status": "rejected", "notes": "",
            },
        ]
        with (gold_dir / "gold_events.real.proposed_223.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        official_summary = GoldLoader(gold_dir).summary("official")
        assert official_summary["included_total"] == 1
        assert official_summary["excluded_total"] == 2
        assert official_summary["excluded_reason_distribution"] == {
            "non_official_review_status:draft": 1,
            "non_official_review_status:rejected": 1,
        }

        draft_summary = GoldLoader(gold_dir).summary("draft")
        assert draft_summary["included_total"] == 1
        assert draft_summary["excluded_reason_distribution"] == {
            "not_draft_review_status:adjudicated": 1,
            "not_draft_review_status:rejected": 1,
        }

        all_summary = GoldLoader(gold_dir).summary("all")
        assert all_summary["included_total"] == 3
        assert all_summary["excluded_total"] == 0
        assert all_summary["excluded_reason_distribution"] == {}


class TestGoldValidation:
    def test_valid_gold_passes(self):
        events = [
            GoldEventV1(
                case_id="test", view_code="front", suggested_event_id="e1",
                suggested_track_id=1, suggested_action="stand",
                suggested_time_window={"start": 0, "end": 1},
                suggested_label="match", gold_track_id="1",
                gold_action="stand", gold_start=0, gold_end=1,
                match_label="match", noise_type="none",
                occlusion_level="low", asr_quality="high",
                annotator="test", review_status="peer_reviewed", notes="",
            )
        ]
        ok, errs = _validate_gold(events)
        assert ok
        assert errs == []

    def test_draft_status_is_valid_lifecycle_but_not_official(self):
        events = [
            GoldEventV1(
                case_id="test", view_code="front", suggested_event_id="e1",
                suggested_track_id=1, suggested_action="stand",
                suggested_time_window={"start": 0, "end": 1},
                suggested_label="match", gold_track_id="1",
                gold_action="stand", gold_start=0, gold_end=1,
                match_label="match", noise_type="none",
                occlusion_level="low", asr_quality="high",
                annotator="test", review_status="draft", notes="",
            )
        ]
        ok, errs = _validate_gold(events)
        assert ok
        assert errs == []
        official_ok, official_errs = _validate_gold(events, require_official=True)
        assert not official_ok
        assert official_errs

    def test_unknown_review_status_fails(self):
        events = [
            GoldEventV1(
                case_id="test", view_code="front", suggested_event_id="e1",
                suggested_track_id=1, suggested_action="stand",
                suggested_time_window={"start": 0, "end": 1},
                suggested_label="match", gold_track_id="1",
                gold_action="stand", gold_start=0, gold_end=1,
                match_label="match", noise_type="none",
                occlusion_level="low", asr_quality="high",
                annotator="test", review_status="not_a_status", notes="",
            )
        ]
        ok, errs = _validate_gold(events)
        assert not ok
        assert errs

    def test_invalid_match_label_fails(self):
        events = [
            GoldEventV1(
                case_id="test", view_code="front", suggested_event_id="e1",
                suggested_track_id=1, suggested_action="stand",
                suggested_time_window={"start": 0, "end": 1},
                suggested_label="match", gold_track_id="1",
                gold_action="stand", gold_start=0, gold_end=1,
                match_label="invalid_label", noise_type="none",
                occlusion_level="low", asr_quality="high",
                annotator="test", review_status="peer_reviewed", notes="",
            )
        ]
        ok, errs = _validate_gold(events)
        assert not ok


# ── CaseRegistry tests ────────────────────────────────────────────

class TestCaseRegistry:
    def test_list_cases_returns_manifests(self, case_registry):
        cases = case_registry.list_cases()
        assert len(cases) >= 5
        assert all(isinstance(c, CaseManifestV1) for c in cases)

    def test_each_case_has_artifacts(self, case_registry):
        for m in case_registry.list_cases():
            assert len(m.artifacts) == len(ARTIFACT_KEYS)

    def test_case_has_warnings_when_missing(self, case_registry):
        for m in case_registry.list_cases():
            if "verified_events" in m.missing_artifacts():
                assert any("verified_events" in w for w in m.warnings)

    def test_get_case_returns_manifest(self, case_registry):
        cases = case_registry.list_cases()
        if cases:
            m = case_registry.get_case(cases[0].case_id)
            assert m is not None
            assert m.case_id == cases[0].case_id

    def test_get_case_returns_none_for_unknown(self, case_registry):
        m = case_registry.get_case("nonexistent_case_xyz")
        assert m is None

    def test_manifest_has_gold_counts(self, case_registry):
        for m in case_registry.list_cases():
            assert m.gold_event_count >= 0

    def test_artifact_summary_dict(self, case_registry):
        for m in case_registry.list_cases():
            summary = m.artifact_summary()
            assert isinstance(summary, dict)
            assert len(summary) == len(ARTIFACT_KEYS)
            assert all(isinstance(v, bool) for v in summary.values())


# ── GoldBackedEvaluator tests ─────────────────────────────────────

class TestGoldBackedEvaluator:
    def test_evaluate_returns_report(self, project_root, gold_dir):
        evaluator = GoldBackedEvaluator(gold_dir, project_root)
        report = evaluator.evaluate()
        assert isinstance(report, VerifierEvalReportV1)
        assert report.gold_event_count_requested == len(GoldLoader(gold_dir).load("official"))
        assert report.sample_count + report.skipped_gold_event_count == report.gold_event_count_requested
        assert report.quality_tier == "official"

    def test_report_has_all_metrics(self, project_root, gold_dir):
        evaluator = GoldBackedEvaluator(gold_dir, project_root)
        report = evaluator.evaluate()
        assert 0 <= report.accuracy <= 1
        assert 0 <= report.precision <= 1
        assert 0 <= report.recall <= 1
        assert 0 <= report.f1 <= 1
        assert 0 <= report.uncertain_ratio <= 1
        assert 0 <= report.candidate_recall <= 1
        assert report.ece >= 0
        assert report.brier >= 0

    def test_report_has_baselines(self, project_root, gold_dir):
        evaluator = GoldBackedEvaluator(gold_dir, project_root)
        report = evaluator.evaluate()
        assert len(report.baselines) == 4
        baseline_names = {b.name for b in report.baselines}
        assert "visual_only" in baseline_names
        assert "fixed_window" in baseline_names
        assert "without_uq" in baseline_names
        assert "always_uncertain" in baseline_names

    def test_report_has_per_case_breakdown(self, project_root, gold_dir):
        evaluator = GoldBackedEvaluator(gold_dir, project_root)
        report = evaluator.evaluate()
        assert report.per_case_breakdown
        first = report.per_case_breakdown[0]
        expected = {
            "case_id", "view_code", "gold_event_count", "sample_count",
            "accuracy", "label_distribution", "predicted_label_distribution",
            "candidate_recall", "uncertain_ratio", "missing_prediction_count",
            "output_status", "source_dir", "warnings",
        }
        assert expected <= set(first)

    def test_missing_output_cases_are_structured(self, tmp_path):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        row = {
            "case_id": "case_without_output", "view_code": "front", "suggested_event_id": "e1",
            "suggested_track_id": 1, "suggested_action": "stand",
            "suggested_time_window": {"start": 0, "end": 1},
            "suggested_label": "match", "gold_track_id": "1",
            "gold_action": "stand", "gold_start": 0, "gold_end": 1,
            "match_label": "match", "noise_type": "none",
            "occlusion_level": "low", "asr_quality": "high",
            "annotator": "test", "review_status": "peer_reviewed", "notes": "",
        }
        with (gold_dir / "gold_events.real.proposed_223.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (tmp_path / "output").mkdir()

        report = GoldBackedEvaluator(gold_dir, tmp_path).evaluate()
        assert report.gold_event_count_requested == 1
        assert report.sample_count == 0
        assert report.skipped_gold_event_count == 1
        assert report.missing_output_cases == [{
            "case_id": "case_without_output",
            "view_code": "front",
            "gold_event_count": 1,
            "gold_event_ids": ["e1"],
            "missing_reason": "output_dir_not_found",
        }]
        assert report.per_case_breakdown[0]["output_status"] == "missing_output"

    def test_always_uncertain_baseline_is_reported(self, project_root, gold_dir):
        evaluator = GoldBackedEvaluator(gold_dir, project_root)
        report = evaluator.evaluate()
        always_uq = next(b for b in report.baselines if b.name == "always_uncertain")
        assert 0 <= always_uq.accuracy <= 1
        assert always_uq.uncertain_ratio == 1.0
        assert report.sample_count == report.pipeline_evaluable_gold_count

    def test_per_sample_has_all_fields(self, project_root, gold_dir):
        evaluator = GoldBackedEvaluator(gold_dir, project_root)
        report = evaluator.evaluate()
        for s in report.per_sample:
            assert s.gold is not None
            assert s.predicted_label in ("match", "mismatch", "uncertain", "missing")
            assert isinstance(s.confidence_match, float)
            assert isinstance(s.reliability, float)

    def test_filter_by_case_ids(self, project_root, gold_dir):
        evaluator = GoldBackedEvaluator(gold_dir, project_root)
        report = evaluator.evaluate(case_ids=["front__001"])
        assert report.sample_count == 1
        assert report.per_sample[0].gold.case_id == "front__001"

    def test_draft_quality_does_not_enter_default_eval(self, project_root, gold_dir):
        evaluator = GoldBackedEvaluator(gold_dir, project_root)
        official_report = evaluator.evaluate()
        all_report = evaluator.evaluate(quality="all")
        assert official_report.sample_count <= all_report.sample_count
        assert official_report.quality_tier == "official"
        assert all_report.quality_tier == "all"
        assert any("not paper-eligible" in w for w in all_report.warnings)

    def test_empty_case_ids_returns_all(self, project_root, gold_dir):
        evaluator = GoldBackedEvaluator(gold_dir, project_root)
        report_all = evaluator.evaluate()
        report_filtered = evaluator.evaluate(case_ids=None)
        assert report_all.sample_count == report_filtered.sample_count


# ── Baseline evaluation tests ─────────────────────────────────────

class TestBaselineEvaluation:
    def test_eval_baseline_correct_predictions(self):
        from server.registry import GoldEventV1
        gold = GoldEventV1(
            case_id="test", view_code="front", suggested_event_id="e1",
            suggested_track_id=1, suggested_action="stand",
            suggested_time_window={"start": 0, "end": 1},
            suggested_label="match", gold_track_id="1",
            gold_action="stand", gold_start=0, gold_end=1,
            match_label="match", noise_type="none",
            occlusion_level="low", asr_quality="high",
            annotator="test", review_status="peer_reviewed", notes="",
        )
        from server.registry import EvalSample
        sample = EvalSample(
            gold=gold, predicted_label="match", predicted_track_id=1,
            predicted_action="stand", confidence_match=0.8,
            confidence_mismatch=0.2, reliability=0.7, uncertainty=0.3,
            top_k_candidate_count=3, gold_predicted_match=True,
            gold_predicted_action_match=True, gold_candidate_in_top_k=True,
        )
        result = _eval_baseline("test", ["match"], [sample], "test desc")
        assert result.accuracy == 1.0
        assert result.name == "test"

    def test_eval_baseline_wrong_predictions(self):
        from server.registry import GoldEventV1, EvalSample
        gold = GoldEventV1(
            case_id="test", view_code="front", suggested_event_id="e1",
            suggested_track_id=1, suggested_action="stand",
            suggested_time_window={"start": 0, "end": 1},
            suggested_label="match", gold_track_id="1",
            gold_action="stand", gold_start=0, gold_end=1,
            match_label="match", noise_type="none",
            occlusion_level="low", asr_quality="high",
            annotator="test", review_status="peer_reviewed", notes="",
        )
        sample = EvalSample(
            gold=gold, predicted_label="match", predicted_track_id=1,
            predicted_action="stand", confidence_match=0.8,
            confidence_mismatch=0.2, reliability=0.7, uncertainty=0.3,
            top_k_candidate_count=3, gold_predicted_match=True,
            gold_predicted_action_match=True, gold_candidate_in_top_k=True,
        )
        result = _eval_baseline("test", ["mismatch"], [sample], "test desc")
        assert result.accuracy == 0.0


# ── Low-level helper tests ────────────────────────────────────────

class TestHelpers:
    def test_load_jsonl_valid(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        rows = _load_jsonl(path)
        assert len(rows) == 2
        assert rows[0]["a"] == 1

    def test_load_jsonl_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        rows = _load_jsonl(path)
        assert rows == []

    def test_load_jsonl_nonexistent(self, tmp_path):
        rows = _load_jsonl(tmp_path / "nonexistent.jsonl")
        assert rows == []

    def test_load_json_valid(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        data = _load_json(path)
        assert data == {"key": "value"}

    def test_load_json_nonexistent(self, tmp_path):
        data = _load_json(tmp_path / "nonexistent.json")
        assert data is None

    def test_load_json_invalid(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        data = _load_json(path)
        assert data is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
