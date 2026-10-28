"""Versioned closed action profiles for reproducible classroom experiments.

The profile digest is deliberately derived from a payload which does not
contain the digest itself.  This matters for manifests: adding or re-writing
``contract_sha256`` must never change the identity of the action contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .hashing import canonical_sha256


_PROFILE_SCHEMA = "classroom_hybrid_ai.taxonomy_profile.v1"

# These are deliberately kept here instead of being inferred from a profile
# length.  The frozen eight-class benchmark must never accidentally start
# scoring an extension profile just because both happen to be loaded in one
# Python process.
CORE8_PROFILE_ID = "core8_v1"
EXTENSION11_PROFILE_ID = "scb_student_11_candidate_v1"
TARGET_HIGHLIGHT_REQUIRED_CODES = frozenset({"bs", "zs", "hd"})


class ActionProfileSchemaError(ValueError):
    """A model response cannot be represented by the selected profile."""

    code = "model_schema_error"


def _normalise_actions(actions: Iterable[Mapping[str, Any]], profile_id: str) -> tuple[dict[str, Any], ...]:
    """Validate and canonicalise action rows used by a profile.

    YAML mappings are copied before they enter the frozen dataclass.  The copy
    prevents a caller mutating a parsed YAML object from silently changing a
    profile after its SHA has been recorded.
    """

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError(f"action_profile_action_invalid:{profile_id}")
        code = str(action.get("code", "")).strip().lower()
        name = str(action.get("name_zh", "")).strip()
        # ``name_zh`` remains the canonical research taxonomy wording.  The
        # renderer and frontend use the deterministic ASCII display name so
        # OpenCV never substitutes unsupported CJK glyphs with question marks.
        name_en = str(action.get("name_en") or code.upper()).strip()
        definition = str(action.get("definition", "")).strip()
        color = str(action.get("color", "#B8B8B8")).strip().upper()
        if not code or code in seen or not name or not name_en or not definition:
            raise ValueError(f"action_profile_action_fields_invalid:{profile_id}:{code}")
        if len(color) != 7 or not color.startswith("#"):
            raise ValueError(f"action_profile_color_invalid:{profile_id}:{code}")
        # Keep the public payload small and deterministic.  Unknown YAML
        # fields do not become accidental parts of the profile contract.
        seen.add(code)
        result.append({
            "code": code,
            "action_id": str(action.get("action_id") or f"a_{code}"),
            "name_zh": name,
            "name_en": name_en,
            "definition": definition,
            "color": color,
        })
    if not result:
        raise ValueError(f"action_profile_actions_missing:{profile_id}")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ActionProfile:
    profile_id: str
    version: str
    purpose: str
    actions: tuple[dict[str, Any], ...]
    excluded: tuple[str, ...] = ()
    audio_required_for: tuple[str, ...] = ()
    source_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        if not profile_id:
            raise ValueError("action_profile_id_missing")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "version", str(self.version).strip() or "1")
        object.__setattr__(self, "purpose", str(self.purpose).strip())
        object.__setattr__(self, "actions", _normalise_actions(self.actions, profile_id))
        if profile_id == CORE8_PROFILE_ID and len(self.actions) != 8:
            raise ValueError("core8_profile_must_contain_exactly_8_actions")
        if profile_id == EXTENSION11_PROFILE_ID and len(self.actions) != 11:
            raise ValueError("extension11_profile_must_contain_exactly_11_actions")
        object.__setattr__(self, "excluded", tuple(str(item) for item in self.excluded))
        object.__setattr__(self, "audio_required_for", tuple(str(item) for item in self.audio_required_for))
        object.__setattr__(self, "source_notes", tuple(str(item) for item in self.source_notes))

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(str(item["code"]) for item in self.actions)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(str(item["action_id"]) for item in self.actions)

    @property
    def is_core8(self) -> bool:
        return self.profile_id == CORE8_PROFILE_ID and len(self.actions) == 8

    @property
    def is_extension11(self) -> bool:
        return self.profile_id == EXTENSION11_PROFILE_ID and len(self.actions) == 11

    def action_for(self, value: Any) -> dict[str, Any] | None:
        """Resolve a model label to one immutable profile action.

        Model transport accepts the short code, canonical ``action_id`` or the
        detailed Chinese name.  The returned dictionary is a copy so callers
        cannot mutate the profile that was bound at run start.
        """

        normalized = str(value or "").strip()
        if not normalized or normalized.upper() == "UNKNOWN":
            return None
        lowered = normalized.lower()
        for action in self.actions:
            if lowered in {str(action["code"]).lower(), str(action["action_id"]).lower()}:
                return dict(action)
            if normalized in {str(action["name_zh"]).strip(), str(action["name_en"]).strip()}:
                return dict(action)
        return None

    def canonical_action_id(self, value: Any, *, unknown: str = "UNKNOWN") -> str:
        action = self.action_for(value)
        return str(action["action_id"]) if action is not None else unknown

    def prompt_payload(self) -> dict[str, Any]:
        """The exact profile envelope sent to a Qwen classification prompt."""

        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "contract_sha256": self.sha256,
            "actions": [dict(item) for item in self.actions],
            "unknown_action": {"code": "UNKNOWN", "name_zh": "证据不足", "name_en": "UNKNOWN", "color": "#B8B8B8"},
        }

    def requires_target_highlight(self, value: Any) -> bool:
        action = self.action_for(value)
        return action is not None and str(action["code"]).lower() in TARGET_HIGHLIGHT_REQUIRED_CODES

    def normalize_model_behavior(self, value: Any) -> tuple[str, dict[str, Any] | None]:
        """Return a canonical action id, or UNKNOWN plus a schema error.

        ``UNKNOWN`` is a valid abstention.  Any other out-of-profile label is
        retained as an auditable model-schema failure; it is never silently
        treated as a valid action.
        """

        text = str(value or "").strip()
        if not text or text.upper() == "UNKNOWN":
            return "UNKNOWN", None
        action = self.action_for(text)
        if action is None:
            raise ActionProfileSchemaError(
                f"model_schema_error:behavior_outside_profile:{self.profile_id}:{text[:80]}"
            )
        return str(action["action_id"]), action

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload())

    @property
    def contract_sha256(self) -> str:
        """Alias used by manifests and the JSON profile contract."""

        return self.sha256

    def _payload(self) -> dict[str, Any]:
        # English display labels are intentionally excluded from the model
        # contract digest.  They change only portable rendering/JSON display,
        # never the frozen classifier choices, definitions, or Qwen prompt.
        contract_actions = [
            {key: value for key, value in item.items() if key != "name_en"}
            for item in self.actions
        ]
        return {
            "schema_version": _PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "version": self.version,
            "purpose": self.purpose,
            "actions": contract_actions,
            "excluded": list(self.excluded),
            "audio_required_for": list(self.audio_required_for),
            "source_notes": list(self.source_notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "contract_sha256": self.sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionProfile":
        """Reconstruct a profile from ``to_dict`` output.

        ``schema_version`` and ``contract_sha256`` are envelope fields.  They
        are intentionally ignored while reconstructing the object so a
        profile's identity remains the hash of its canonical payload only.
        A caller can therefore round-trip a manifest without creating a
        self-referential hash calculation.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("action_profile_payload_must_be_mapping")
        values = dict(payload)
        schema = values.pop("schema_version", None)
        if schema not in (None, _PROFILE_SCHEMA, "classroom_hybrid_ai.action_profiles.v1"):
            raise ValueError(f"action_profile_schema_invalid:{schema}")
        # The supplied digest is provenance only; recompute it from the
        # canonical payload and expose the resulting value through sha256.
        values.pop("contract_sha256", None)
        values.pop("sha256", None)
        required = {"profile_id", "version", "purpose", "actions"}
        missing = sorted(required.difference(values))
        if missing:
            raise ValueError(f"action_profile_fields_missing:{','.join(missing)}")
        return cls(
            profile_id=str(values["profile_id"]),
            version=str(values["version"]),
            purpose=str(values["purpose"]),
            actions=tuple(values["actions"]),
            excluded=tuple(values.get("excluded", ())),
            audio_required_for=tuple(values.get("audio_required_for", ())),
            source_notes=tuple(values.get("source_notes", ())),
        )


def load_profiles(path: str | Path) -> dict[str, ActionProfile]:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("action_profiles_root_must_be_mapping")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("action_profiles_missing")
    result: dict[str, ActionProfile] = {}
    for profile_id, raw in raw_profiles.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"action_profile_invalid:{profile_id}")
        actions = raw.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"action_profile_actions_missing:{profile_id}")
        result[str(profile_id)] = ActionProfile(
            profile_id=str(profile_id),
            version=str(raw.get("version", "1")),
            purpose=str(raw.get("purpose", "")),
            actions=tuple(actions),
            excluded=tuple(str(item) for item in raw.get("excluded", ())),
            audio_required_for=tuple(str(item) for item in raw.get("audio_required_for", ())),
            source_notes=tuple(str(item) for item in raw.get("source_notes", ())),
        )
    return result


def load_profile(path: str | Path, profile_id: str = "core8_v1") -> ActionProfile:
    profiles = load_profiles(path)
    try:
        return profiles[profile_id]
    except KeyError as exc:
        raise ValueError(f"action_profile_not_found:{profile_id}") from exc


__all__ = [
    "ActionProfile",
    "ActionProfileSchemaError",
    "CORE8_PROFILE_ID",
    "EXTENSION11_PROFILE_ID",
    "TARGET_HIGHLIGHT_REQUIRED_CODES",
    "load_profile",
    "load_profiles",
]
