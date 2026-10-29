from __future__ import annotations

import json
from pathlib import Path

import pytest

from classroom_hybrid_ai.pipeline import PipelineError, _classification_prompt, _validate_classification_response
from classroom_hybrid_ai.profiles import load_profiles


ROOT = Path(__file__).resolve().parents[1]


def _response(behavior: str | None) -> dict[str, object]:
    return {
        "candidate_event_id": "event-1",
        "decision": "confirmed" if behavior else "uncertain",
        "reviewed_event_type": "action",
        "behavior": behavior,
        "rationale": "可见",
        "evidence_frame_indices": [1, 2],
        "self_reported_confidence": None,
        "reviewer": "qwen3.5",
        "schema_valid": True,
        "vocabulary_sha256": "vocabulary-digest",
        "observability": "clear",
    }


def test_prompt_contains_the_selected_profile_identity_and_extension_has_11_actions() -> None:
    profiles = load_profiles(ROOT / "configs" / "action_profiles.yaml")
    extension = profiles["scb_student_11_candidate_v1"]
    prompt = _classification_prompt((), "vocabulary-digest", action_profile=extension)

    assert f"Action profile={extension.profile_id}" in prompt
    assert extension.sha256 in prompt
    # The full contract JSON is deliberately not echoed: it duplicated the
    # frozen vocabulary and pushed every request past the 8 GiB prefill budget.
    envelope = json.dumps(extension.prompt_payload(), ensure_ascii=False, separators=(",", ":"))
    assert envelope not in prompt
    assert "Profile contract=" not in prompt
    assert prompt.count('"action_id":"a_') >= 11
    assert all(str(action["name_zh"]) in prompt for action in extension.actions)
    assert all(str(action["name_en"]).isascii() for action in extension.actions)
    assert all(str(action["color"]) in prompt for action in extension.actions)
    assert 'behavior="UNKNOWN"' in prompt
    assert "behavior=null" not in prompt
    assert 'the string "null"' in prompt


def test_core_profile_accepts_only_the_eight_profile_actions() -> None:
    core = load_profiles(ROOT / "configs" / "action_profiles.yaml")["core8_v1"]
    assert core.is_core8
    assert len(core.actions) == 8
    accepted = _validate_classification_response(
        _response("dx"), [1, 2], action_profile=core, target_highlighted=True
    )
    assert accepted["behavior"] == "a_dx"
    with pytest.raises(PipelineError, match="model_schema_error:behavior_outside_profile"):
        _validate_classification_response(
            _response("a_bs"), [1, 2], action_profile=core, target_highlighted=True
        )


@pytest.mark.parametrize("code", ["bs", "zs", "hd"])
def test_extension_actions_require_target_highlighted_global_context(code: str) -> None:
    extension = load_profiles(ROOT / "configs" / "action_profiles.yaml")["scb_student_11_candidate_v1"]
    with pytest.raises(PipelineError, match="model_schema_error:target_highlight_required"):
        _validate_classification_response(
            _response(code), [1, 2], action_profile=extension, target_highlighted=False
        )
    accepted = _validate_classification_response(
        _response(code), [1, 2], action_profile=extension, target_highlighted=True
    )
    assert accepted["behavior"] == f"a_{code}"


def test_unknown_profile_action_is_not_silently_accepted() -> None:
    extension = load_profiles(ROOT / "configs" / "action_profiles.yaml")["scb_student_11_candidate_v1"]
    with pytest.raises(PipelineError, match="model_schema_error:behavior_outside_profile"):
        _validate_classification_response(
            _response("phone_use"), [1, 2], action_profile=extension, target_highlighted=True
        )


def test_unknown_is_the_only_explicit_abstention() -> None:
    extension = load_profiles(ROOT / "configs" / "action_profiles.yaml")["scb_student_11_candidate_v1"]
    accepted = _validate_classification_response(
        _response("UNKNOWN"), [1, 2], action_profile=extension, target_highlighted=False
    )
    assert accepted["behavior"] == "UNKNOWN"
