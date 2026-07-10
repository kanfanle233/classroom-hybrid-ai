"""
Regression / contract tests for the paper evidence API (Wave 1-2).

Covers 3 canonical cases:
  - front_45618_full   (raw codex report case)
  - front_45618_sliced  (frontend bundle case)
  - run_full_paper_mainline_001 (full integration case)

Usage:
  python -m server.test_regression
  pytest server/test_regression.py -v
"""

import json
import os
import re
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("YOLOV11_CASE_FIXTURE_ROOT", str(_PROJECT_ROOT / "tests" / "fixtures" / "server_cases"))

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

# ── Canonical case IDs ───────────────────────────────────────────────
RAW_CASE = "front_45618_full"
BUNDLE_CASE = "front_45618_sliced"
INTEGRATION_CASE = "run_full_paper_mainline_001"
CANONICAL_CASES = [RAW_CASE, BUNDLE_CASE, INTEGRATION_CASE]

# At least one event that must exist in every known case
KNOWN_EVENT_ID = "e_000000_00"

_REPORTS_DIR = _PROJECT_ROOT / "output" / "codex_reports"
_BUNDLE_DIR = _PROJECT_ROOT / "output" / "frontend_bundle"


def _discover_case_ids(root: Path, matcher) -> list[str]:
    if not root.exists():
        return []
    case_ids: list[str] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if matcher(child.name):
            case_ids.append(child.name)
    return case_ids


EXTRA_FULL_CASES = [
    cid for cid in _discover_case_ids(
        _REPORTS_DIR,
        lambda name: bool(re.match(r"^front_.*_full$", name)) and name != RAW_CASE,
    )
][:2]

EXTRA_SLICED_CASES = [
    cid for cid in _discover_case_ids(
        _BUNDLE_DIR,
        lambda name: bool(re.match(r"^front_.*_sliced$", name)) and name != BUNDLE_CASE,
    )
][:2]

GENERALIZATION_CASES = list(dict.fromkeys(CANONICAL_CASES + EXTRA_FULL_CASES + EXTRA_SLICED_CASES))

# ── Helpers ──────────────────────────────────────────────────────────

def _ok(r, expected_status=200):
    """Assert status code and return JSON body."""
    assert r.status_code == expected_status, (
        f"Expected {expected_status}, got {r.status_code}: {r.text[:300]}")
    return r.json()


def _has_keys(obj, *keys):
    """Assert all keys are present in a dict."""
    missing = [k for k in keys if k not in obj]
    assert not missing, f"Missing keys: {missing}"


# ── Test helpers ─────────────────────────────────────────────────────

def _find_existing_event_id(case_id):
    """Find a real event_id from the case's verified events, or fall back to KNOWN_EVENT_ID."""
    # Hit summary to get verified count, then try evidence list
    r = client.get(f"/api/case/{case_id}/summary")
    if r.status_code != 200:
        return KNOWN_EVENT_ID
    # Try the VSumVis detail endpoint to get event list
    r2 = client.get(f"/api/v2/vsumvis/case/{case_id}")
    if r2.status_code != 200:
        return KNOWN_EVENT_ID
    data = r2.json().get("data", {})
    events = data.get("verified_events", [])
    if events:
        return events[0].get("event_id") or events[0].get("query_id") or KNOWN_EVENT_ID
    return KNOWN_EVENT_ID


# ══════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════

# ── /api/cases ───────────────────────────────────────────────────────

def test_cases_list():
    """GET /api/cases returns non-empty list with case_kind and data_source."""
    r = _ok(client.get("/api/cases"))
    assert r["status"] == "success"
    assert isinstance(r["cases"], list)
    assert len(r["cases"]) > 0
    assert r["total"] == len(r["cases"])
    for case in r["cases"]:
        _has_keys(case, "case_id", "case_kind", "data_source")


# ── /api/case/{case_id}/summary ──────────────────────────────────────

def test_summary_for_all_cases():
    """GET /api/case/{case_id}/summary returns 200 for canonical + sampled front cases."""
    for case_id in GENERALIZATION_CASES:
        r = _ok(client.get(f"/api/case/{case_id}/summary"))
        assert r["status"] == "success"
        assert r["case_id"] == case_id
        _has_keys(r, "label_distribution", "asr_status", "contract_status",
                  "verified_event_count", "query_event_count", "student_count",
                  "available_files")


def test_summary_label_distribution():
    """Summary label_distribution has match/uncertain/mismatch keys."""
    r = _ok(client.get(f"/api/case/{RAW_CASE}/summary"))
    ld = r["label_distribution"]
    for k in ("match", "uncertain", "mismatch", "unverified"):
        assert k in ld, f"Missing label key: {k}"
    total = sum(ld.values())
    assert total == r["verified_event_count"]


def test_summary_llm_student_v4_status():
    """Summary exposes V4 student benchmark status without claiming gold labels."""
    r = _ok(client.get(f"/api/case/{RAW_CASE}/summary"))
    status = r.get("llm_distilled_student_v4")
    assert isinstance(status, dict), "Missing llm_distilled_student_v4 summary"
    if status.get("available"):
        assert status.get("fusion_mode") == "llm_distilled_student_v4"
        assert status.get("teacher_source") == "claude_agent"
        assert status.get("teacher_dataset") == "llm_adjudicated_dataset_v4"
        assert status.get("student_model_name") == "student_judge_v4_best.joblib"
        assert status.get("evaluation_kind") == "pseudo_label_benchmark"
        assert status.get("silver_gold_boundary") == "silver_labels_not_human_gold"


# ── /api/case/{case_id}/contract ─────────────────────────────────────

def test_contract_for_all_cases():
    """GET /api/case/{case_id}/contract returns 200 for canonical + sampled front cases."""
    for case_id in GENERALIZATION_CASES:
        r = _ok(client.get(f"/api/case/{case_id}/contract"))
        assert r["status"] == "success"
        _has_keys(r, "contract", "missing_files", "error_count", "warning_count")


# ── /api/case/{case_id}/asr-quality ──────────────────────────────────

def test_asr_quality_for_all_cases():
    """GET /api/case/{case_id}/asr-quality returns 200 with asr_available flag."""
    for case_id in GENERALIZATION_CASES:
        r = _ok(client.get(f"/api/case/{case_id}/asr-quality"))
        assert r["status"] == "success"
        assert "asr_available" in r


def test_asr_quality_returns_200_even_when_missing():
    """ASR quality returns 200 with asr_available=False for missing data, not 404."""
    # Use a case that likely has no ASR report
    r = _ok(client.get(f"/api/case/{BUNDLE_CASE}/asr-quality"))
    # Must not 404; already asserted via _ok(200). Check structure.
    assert "asr_available" in r


# ── /api/case/{case_id}/evidence/{event_id} ──────────────────────────

def test_evidence_api_for_all_cases():
    """GET /api/case/{case_id}/evidence/{event_id} returns 200 with query, selected, candidates."""
    for case_id in CANONICAL_CASES:
        event_id = _find_existing_event_id(case_id)
        r = _ok(client.get(f"/api/case/{case_id}/evidence/{event_id}"))
        assert r["status"] == "success", f"Evidence API failed for {case_id}/{event_id}: {r}"
        _has_keys(r, "query", "selected", "align_candidates", "media", "source_files")


def test_evidence_query_block():
    """Evidence query block has query_source, is_visual_fallback, source_conflict."""
    event_id = _find_existing_event_id(RAW_CASE)
    r = _ok(client.get(f"/api/case/{RAW_CASE}/evidence/{event_id}"))
    q = r["query"]
    assert q is not None, f"No query block for {RAW_CASE}/{event_id}"
    _has_keys(q, "event_id", "query_text", "query_source", "is_visual_fallback",
              "source_conflict", "asr_inferred_source", "query_time",
              "window_start", "window_end")
    assert q["query_source"] in ("asr", "visual_fallback", "unknown")
    assert isinstance(q["is_visual_fallback"], bool)
    assert isinstance(q["source_conflict"], bool)


def test_evidence_selected_block():
    """Evidence selected block has rank_metric and selected_by fields."""
    event_id = _find_existing_event_id(RAW_CASE)
    r = _ok(client.get(f"/api/case/{RAW_CASE}/evidence/{event_id}"))
    s = r["selected"]
    assert s is not None, f"No selected block for {RAW_CASE}/{event_id}"
    _has_keys(s, "track_id", "label", "p_match", "p_mismatch",
              "reliability_score", "uncertainty", "visual_score", "text_score",
              "uq_score", "selected_candidate_rank", "candidate_count",
              "rank_metric", "selected_by", "fusion_mode", "distillation")
    assert s["selected_by"] == "track_id_match"
    assert isinstance(s["rank_metric"], str) and len(s["rank_metric"]) > 0
    assert isinstance(s["distillation"], dict)
    _has_keys(s["distillation"], "enabled", "fusion_mode", "student_model_name",
              "student_feature_version", "teacher_source", "teacher_dataset",
              "evaluation_kind", "silver_gold_boundary")


def test_evidence_candidates_block():
    """Evidence align_candidates has rank, is_selected, composite_score per candidate."""
    event_id = _find_existing_event_id(RAW_CASE)
    r = _ok(client.get(f"/api/case/{RAW_CASE}/evidence/{event_id}"))
    candidates = r.get("align_candidates", [])
    # May be empty if no alignment data
    for c in candidates:
        _has_keys(c, "track_id", "rank", "is_selected", "overlap",
                  "action_confidence", "uq_track", "composite_score")


def test_evidence_media_block():
    """Evidence media block has video_url and playback range."""
    event_id = _find_existing_event_id(RAW_CASE)
    r = _ok(client.get(f"/api/case/{RAW_CASE}/evidence/{event_id}"))
    m = r["media"]
    if m is not None:
        _has_keys(m, "video_url", "start_sec", "end_sec")


def test_evidence_event_not_found_returns_200():
    """Missing event returns 200 with status 'event_not_found', not 404."""
    r = _ok(client.get(f"/api/case/{RAW_CASE}/evidence/nonexistent_event_999"))
    assert r["status"] == "event_not_found"
    assert r["query"] is None
    assert r["selected"] is None


# ── /api/case/{case_id}/alignment/{event_id} ─────────────────────────

def test_alignment_api_for_all_cases():
    """GET /api/case/{case_id}/alignment/{event_id} returns 200 with candidates."""
    for case_id in CANONICAL_CASES:
        event_id = _find_existing_event_id(case_id)
        r = _ok(client.get(f"/api/case/{case_id}/alignment/{event_id}"))
        assert r["status"] in ("success", "alignment_not_found"), (
            f"Unexpected status for {case_id}/{event_id}: {r.get('status')}")
        if r["status"] == "success":
            _has_keys(r, "selected_track_id", "selected_candidate_rank",
                      "candidate_count", "rank_metric", "selected_by", "candidates")


def test_alignment_missing_returns_200():
    """Missing alignment returns 200 with status 'alignment_not_found', not 404."""
    r = _ok(client.get(f"/api/case/{RAW_CASE}/alignment/nonexistent_event_999"))
    assert r["status"] == "alignment_not_found"
    assert r["candidate_count"] == 0
    assert r["candidates"] == []


def test_vsumvis_case_detail_generalization():
    """GET /api/v2/vsumvis/case/{case_id} returns legal detail for sampled front cases."""
    for case_id in GENERALIZATION_CASES:
        r = _ok(client.get(f"/api/v2/vsumvis/case/{case_id}"))
        assert r["status"] == "success"
        data = r.get("data", {})
        _has_keys(data, "case_info", "timeline_segments", "verified_events", "event_queries")


def test_front_vsumvis_page_contract():
    """Paper front page loads and still contains evidence API hooks."""
    r = client.get("/paper/front-vsumvis")
    assert r.status_code == 200
    body = r.text
    assert "evidence-api-section" in body
    assert "/api/case/" in body
    assert "fetchEvidenceAPI" in body


# ── /api/paper/metrics ───────────────────────────────────────────────

def test_paper_metrics():
    """GET /api/paper/metrics returns 200 with metrics, figures, provenance."""
    r = _ok(client.get("/api/paper/metrics"))
    assert r["status"] == "success"
    _has_keys(r, "metrics", "figures", "quality_gates", "provenance")


# ── /api/ablation ────────────────────────────────────────────────────

def test_ablation_list():
    """GET /api/ablation returns available dimensions."""
    r = _ok(client.get("/api/ablation"))
    assert r["status"] == "success"
    dims = {d["dimension"] for d in r["dimensions"]}
    assert "sr" in dims


def test_ablation_sr():
    """GET /api/ablation/sr returns 200 with ablations list."""
    r = _ok(client.get("/api/ablation/sr"))
    assert r["status"] == "success"
    assert "ablations" in r


def test_ablation_unknown_dimension_returns_400():
    """Unknown ablation dimension returns 400."""
    r = client.get("/api/ablation/unknown_dim_xyz")
    assert r.status_code == 400


# ── /api/cases (canonical list) ──────────────────────────────────────

def test_cases_includes_sampled_cases():
    """The unified case list includes canonical and sampled front cases."""
    r = _ok(client.get("/api/cases"))
    case_ids = {c["case_id"] for c in r["cases"]}
    for cid in GENERALIZATION_CASES:
        assert cid in case_ids, f"Canonical case {cid} missing from /api/cases"


# ── Label filter on VSumVis detail ───────────────────────────────────

def test_vsumvis_label_filter():
    """GET /api/v2/vsumvis/case/{case_id}?label=mismatch filters events."""
    r = _ok(client.get(f"/api/v2/vsumvis/case/{RAW_CASE}?label=mismatch"))
    events = r.get("data", {}).get("verified_events", [])
    for ve in events:
        lbl = ve.get("match_label") or ve.get("label") or ""
        assert lbl == "mismatch", f"Expected mismatch, got {lbl}"


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors: list[str] = []

    tests = [
        test_cases_list,
        test_summary_for_all_cases,
        test_summary_label_distribution,
        test_summary_llm_student_v4_status,
        test_contract_for_all_cases,
        test_asr_quality_for_all_cases,
        test_asr_quality_returns_200_even_when_missing,
        test_evidence_api_for_all_cases,
        test_evidence_query_block,
        test_evidence_selected_block,
        test_evidence_candidates_block,
        test_evidence_media_block,
        test_evidence_event_not_found_returns_200,
        test_alignment_api_for_all_cases,
        test_alignment_missing_returns_200,
        test_paper_metrics,
        test_ablation_list,
        test_ablation_sr,
        test_ablation_unknown_dimension_returns_400,
        test_vsumvis_case_detail_generalization,
        test_front_vsumvis_page_contract,
        test_cases_includes_sampled_cases,
        test_vsumvis_label_filter,
    ]

    for t in tests:
        name = t.__name__
        try:
            t()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            msg = f"{name}: {e}"
            errors.append(msg)
            print(f"  FAIL  {msg}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")

    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All regression tests passed.")
