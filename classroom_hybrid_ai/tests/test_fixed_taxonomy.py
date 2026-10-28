from __future__ import annotations

import importlib.util
from pathlib import Path

from classroom_hybrid_ai.taxonomy import (
    EIGHT_CODES,
    decode_fixed_action,
    fixed_registry,
    load_fixed_taxonomy,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_frozen_qwen.py"
SPEC = importlib.util.spec_from_file_location("run_frozen_qwen_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
frozen_qwen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen_qwen)


def test_fixed_taxonomy_has_exact_codes_names_ids_and_unique_colors() -> None:
    actions = load_fixed_taxonomy(ROOT / "configs" / "behaviors.yaml")
    assert tuple(action.code for action in actions) == EIGHT_CODES
    assert tuple(action.name_zh for action in actions) == (
        "低头写字", "低头看书", "抬头听课", "转头",
        "举手", "站立", "小组讨论", "教师指导",
    )
    assert tuple(action.action_id for action in actions) == tuple(f"a_{code}" for code in EIGHT_CODES)
    assert len({action.color for action in actions}) == 8
    registry = fixed_registry(actions)
    assert {item.attributes["fixed_eight_code"] for item in registry.actions} == set(EIGHT_CODES)


def test_fixed_action_decoder_and_prediction_writer_do_not_turn_valid_dk_unknown() -> None:
    actions = load_fixed_taxonomy(ROOT / "configs" / "behaviors.yaml")
    assert decode_fixed_action("a_dk", actions).code == "dk"
    assert decode_fixed_action("低头看书", actions).code == "dk"
    label, name = frozen_qwen._prediction_fields(
        "a_dk", response_valid=True, actions=actions, phrase_map={}
    )
    assert (label, name) == ("dk", "低头看书")
    assert frozen_qwen._prediction_fields(
        "a_dk", response_valid=False, actions=actions, phrase_map={}
    ) == ("UNKNOWN", None)


def test_fixed_prompt_lists_all_actions_and_stratified_selector_is_balanced() -> None:
    actions = load_fixed_taxonomy(ROOT / "configs" / "behaviors.yaml")
    prompt = frozen_qwen._classification_prompt(actions)
    assert all(action.action_id in prompt and action.name_zh in prompt for action in actions)
    inputs = [
        {
            "_gold_label": code,
            "ordinal": ordinal,
            "sample_id": f"{code}-{ordinal}",
            "box_index": ordinal,
            "source_group": f"lesson-{ordinal}",
        }
        for ordinal in range(3)
        for code in EIGHT_CODES
    ]
    selected = frozen_qwen._stratified_inputs(inputs, 2)
    assert len(selected) == 16
    assert {code: sum(item["_gold_label"] == code for item in selected) for code in EIGHT_CODES} == {
        code: 2 for code in EIGHT_CODES
    }
