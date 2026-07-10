import os
from pathlib import Path

from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLOV11_CASE_FIXTURE_ROOT", str(_PROJECT_ROOT / "tests" / "fixtures" / "server_cases"))

from server.app import app


client = TestClient(app)


def test_gold_tasks_uses_csv_as_active_source():
    response = client.get("/api/gold/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["source_files"]["tasks"].endswith("tasks.csv")
    assert data["progress"]["exported"] <= data["progress"]["total"]
    assert data["tasks"][0]["manual_review_status"] == "peer_reviewed"


def test_canonical_evidence_review_mode_exposes_review_fields():
    response = client.get("/api/v2/cases/front__001/events/e_000000_00/evidence")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["mode"] == "review"
    assert "asr_quality" in data
    assert "pipeline_step_status" in data
    assert "gold_label_status" in data
    assert data["selected"] is not None
    assert "label" in data["selected"]
    assert "coherence" in data
    assert "observation_card" in data
    assert data["observation_card"]["policy"] == "neutral_observation_only"


def test_canonical_evidence_blinded_mode_hides_outcome_fields():
    response = client.get("/api/v2/cases/front__001/events/e_000000_00/evidence?mode=blinded")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["mode"] == "blinded"
    assert data["selected"] == {"hidden": True, "reason": "blinded_mode"}
    assert data["coherence"] == {"hidden": True, "reason": "blinded_mode"}
    assert data["observation_card"] == {"hidden": True, "reason": "blinded_mode"}
    assert data["gold_label_status"]["hidden"] is True
    assert "match_label" not in data["gold_label_status"]
    for candidate in data.get("align_candidates", []):
        assert "is_selected" not in candidate
        assert "composite_score" not in candidate


def test_visual_gold_bridge_report_endpoint():
    response = client.get("/api/eval/visual-gold-bridge")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["sample_size"] == 204
    assert data["verifier_evaluable_count"] == 40
    assert data["verifier_blocked_count"] == 164
    assert data["excluded_reason_distribution"]
    assert data["source_files"]["event_bridge_map"].endswith("agent300_event_bridge_map.jsonl")


def test_agent300_visual_gold_report_endpoint():
    response = client.get("/api/eval/agent300-visual-gold-report")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["sample_size"] == 204
    assert data["scope"] == "visual_gold_quality_and_verifier_readiness_not_full_accuracy"
    assert data["verifier_metric_eligible"] is False
    assert data["verifier_evaluable_count"] == 40
    assert data["verifier_blocked_count"] == 164
    assert data["verifier_readiness"]["excluded_from_verifier"] == 164
    assert data["source_files"]["event_bridge_map"].endswith("agent300_event_bridge_map.jsonl")


def test_visual_gold_report_endpoint_layers_gold_and_verifier_scope():
    response = client.get("/api/eval/visual-gold-report")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["official_visual_gold_count"] == 223
    assert data["legacy_official_gold_count"] == 19
    assert data["agent300_visual_gold_count"] == 204
    assert data["pipeline_evaluable_count"] == 59
    assert data["excluded_from_verifier"] == 164


def test_reviewer_response_report_endpoint_is_paper_ready():
    response = client.get("/api/eval/reviewer-response-report")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["official_visual_gold_count"] == 223
    assert data["pipeline_evaluable_count"] == 59
    assert data["excluded_from_verifier"] == 164
    assert data["verifier_metrics"]["sample_count"] == 59
    assert "markdown" in data
