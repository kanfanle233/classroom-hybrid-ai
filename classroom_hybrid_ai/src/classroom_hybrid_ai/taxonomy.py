"""Closed eight-action taxonomy used by Qwen classification and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
import warnings

from .open_vocab import ActionDescriptor, ActionRegistrySnapshot, colour_for_index, normalize_phrase


# ``CORE8_CODES`` is the canonical name.  Keep the old export for third-party
# scripts, but make migration visible to callers that enable deprecation
# warnings.  It remains an eight-class constant and never includes extension11
# labels.
CORE8_CODES = ("dx", "dk", "tt", "zt", "js", "zl", "xt", "jz")


def __getattr__(name: str):
    """Resolve the legacy alias only when a caller actually asks for it."""

    if name == "EIGHT_CODES":
        warnings.warn(
            "taxonomy.EIGHT_CODES is deprecated; use CORE8_CODES or an ActionProfile",
            DeprecationWarning,
            stacklevel=2,
        )
        return CORE8_CODES
    raise AttributeError(name)
PROMPT_PRIORITY_CODES = ("js", "zl", "jz", "xt", "zt", "dx", "dk", "tt")

FIXED_EIGHT_VISUAL_RULES = (
    "Apply this visible-evidence checklist in order before choosing a class: "
    "js only when a hand or forearm is visibly raised above the target's shoulder; "
    "zl only when the target's torso and pelvis are clearly upright rather than seated; "
    "jz only when a visible teacher is close to and directly interacting with or pointing to the target; "
    "xt only when the target visibly faces, gestures to, talks with, or exchanges an object with a peer; "
    "zt only when the head or upper torso is visibly turned sideways or backward relative to the desk/teaching direction; "
    "dx only when a pen/pencil hand is visibly contacting paper in a writing posture; "
    "dk only when gaze is directed at an open book/sheet without visible writing contact; "
    "tt only when the head/face is up toward the teaching area and none of the other actions is visible. "
    "A lowered head, desk, paper, or book by itself never proves dx or dk. Context objects belonging to nearby students "
    "must not be attributed to the target. If the required cue is too small, blurred, occluded, or absent, return uncertain. "
)


@dataclass(frozen=True, slots=True)
class FixedAction:
    code: str
    name_zh: str
    definition: str
    color: str

    @property
    def action_id(self) -> str:
        # ActionDescriptor keeps the ``a_`` namespace for timeline IDs.  The
        # short dataset code remains an explicit attribute and output field.
        return f"a_{self.code}"

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "action_id": self.action_id,
            "code": self.code,
            "name_zh": self.name_zh,
            "definition": self.definition,
        }


def load_fixed_taxonomy(path: str | Path) -> tuple[FixedAction, ...]:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("behavior_taxonomy_root_must_be_mapping")
    rows = payload.get("primary_actions")
    if not isinstance(rows, list):
        raise ValueError("behavior_taxonomy_primary_actions_missing")

    by_code: dict[str, FixedAction] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"behavior_taxonomy_row_invalid:{index}")
        code = str(row.get("label", "")).strip().lower()
        name = normalize_phrase(str(row.get("name_zh", "")))
        definition = str(row.get("definition", "")).strip()
        color = str(row.get("color") or colour_for_index(index)).strip().upper()
        if code not in CORE8_CODES:
            raise ValueError(f"behavior_taxonomy_code_invalid:{code}")
        if code in by_code:
            raise ValueError(f"behavior_taxonomy_code_duplicate:{code}")
        if not name:
            raise ValueError(f"behavior_taxonomy_name_missing:{code}")
        if not definition:
            raise ValueError(f"behavior_taxonomy_definition_missing:{code}")
        if len(color) != 7 or not color.startswith("#"):
            raise ValueError(f"behavior_taxonomy_color_invalid:{code}")
        try:
            int(color[1:], 16)
        except ValueError as exc:
            raise ValueError(f"behavior_taxonomy_color_invalid:{code}") from exc
        by_code[code] = FixedAction(code, name, definition, color)

    missing = [code for code in CORE8_CODES if code not in by_code]
    if missing or len(by_code) != len(CORE8_CODES):
        raise ValueError(f"behavior_taxonomy_must_contain_exact_eight:{','.join(missing)}")
    actions = tuple(by_code[code] for code in CORE8_CODES)
    names = [normalize_phrase(action.name_zh) for action in actions]
    if len(set(names)) != len(names):
        raise ValueError("behavior_taxonomy_names_must_be_unique")
    colors = [action.color for action in actions]
    if len(set(colors)) != len(colors):
        raise ValueError("behavior_taxonomy_colors_must_be_unique")
    return actions


def fixed_descriptors(actions: Sequence[FixedAction]) -> tuple[ActionDescriptor, ...]:
    return tuple(
        ActionDescriptor(
            action_id=action.action_id,
            name_zh=action.name_zh,
            definition=action.definition,
            aliases=(action.code,),
            attributes={"fixed_eight_code": action.code},
            color=action.color,
            status="shared_candidate",
        )
        for action in actions
    )


def fixed_registry(actions: Sequence[FixedAction]) -> ActionRegistrySnapshot:
    return ActionRegistrySnapshot.create(fixed_descriptors(actions))


def decode_fixed_action(value: Any, actions: Sequence[FixedAction]) -> FixedAction | None:
    """Resolve an exact code, action ID, or official detailed Chinese name."""

    normalized = normalize_phrase(str(value or ""))
    if not normalized or normalized.upper() == "UNKNOWN":
        return None
    lowered = normalized.lower()
    for action in actions:
        if lowered in {action.code, action.action_id.lower()}:
            return action
        if normalized == normalize_phrase(action.name_zh):
            return action
    return None


__all__ = [
    "CORE8_CODES",
    "EIGHT_CODES",
    "FIXED_EIGHT_VISUAL_RULES",
    "FixedAction",
    "PROMPT_PRIORITY_CODES",
    "decode_fixed_action",
    "fixed_descriptors",
    "fixed_registry",
    "load_fixed_taxonomy",
]
