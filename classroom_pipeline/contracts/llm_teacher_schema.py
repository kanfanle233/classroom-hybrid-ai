"""LLM Teacher output schema — structured late-fusion judge labels.

Each record represents the LLM teacher's judgment on one (query, candidate)
pair, produced from structured multimodal evidence (text/JSON only).

Field lineage:
  - schema_version, case_id, event_id    → pipeline metadata
  - model_name, prompt_version           → LLM provenance
  - generated_at                         → timestamp
  - input_signature                      → hash of input evidence for dedup
  - llm_label                            → "match" | "mismatch" | "uncertain"
  - llm_match_score / llm_mismatch_score / llm_uncertain_score  → softmax-like scores
  - llm_confidence                       → LLM's self-reported confidence [0,1]
  - llm_rationale                        → free-text reasoning
  - llm_raw_response                     → original LLM response (for debugging)
  - provider_mode                        → "real" | "simulate"
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ── Constants ───────────────────────────────────────────────────────────

SCHEMA_VERSION = "2026-04-01"
PROMPT_VERSION = "llm_teacher_fusion_v1"

VALID_LABELS = {"match", "mismatch", "uncertain"}
VALID_PROVIDER_MODES = {"real", "simulate"}


# ── Internal helpers (mirror contracts/schemas.py) ──────────────────────

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _in_01(value: Any) -> bool:
    return _is_number(value) and 0.0 <= float(value) <= 1.0


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _schema_version_ok(value: Any) -> bool:
    if not _is_non_empty_str(value):
        return False
    v = str(value).strip()
    return v == SCHEMA_VERSION or v.startswith(SCHEMA_VERSION + "+")


def _require_keys(obj: Dict[str, Any], keys: Sequence[str]) -> Tuple[bool, str]:
    for key in keys:
        if key not in obj:
            return False, f"missing key: {key}"
    return True, ""


# ── Input signature helper ──────────────────────────────────────────────

def compute_input_signature(event_id: str, track_id: int, query_text: str, behavior_code: str) -> str:
    """Deterministic hash of the input evidence for deduplication and provenance."""
    raw = f"{event_id}|{track_id}|{query_text}|{behavior_code}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ── Record builder ──────────────────────────────────────────────────────

def make_teacher_record(
    *,
    case_id: str,
    event_id: str,
    track_id: int,
    query_text: str,
    behavior_code: str,
    llm_label: str,
    llm_match_score: float,
    llm_mismatch_score: float,
    llm_uncertain_score: float,
    llm_confidence: float,
    llm_rationale: str,
    llm_raw_response: str,
    model_name: str,
    provider_mode: str,
) -> Dict[str, Any]:
    """Build a validated LLM teacher output record."""
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "event_id": event_id,
        "track_id": track_id,
        "model_name": model_name,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_signature": compute_input_signature(event_id, track_id, query_text, behavior_code),
        "llm_label": llm_label,
        "llm_match_score": round(float(llm_match_score), 6),
        "llm_mismatch_score": round(float(llm_mismatch_score), 6),
        "llm_uncertain_score": round(float(llm_uncertain_score), 6),
        "llm_confidence": round(float(llm_confidence), 6),
        "llm_rationale": llm_rationale,
        "llm_raw_response": llm_raw_response,
        "provider_mode": provider_mode,
    }


# ── Schema validation ───────────────────────────────────────────────────

def validate_llm_teacher_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a single LLM teacher output record.

    Returns (is_valid, error_message).
    """
    required = [
        "schema_version",
        "case_id",
        "event_id",
        "track_id",
        "model_name",
        "prompt_version",
        "generated_at",
        "input_signature",
        "llm_label",
        "llm_match_score",
        "llm_mismatch_score",
        "llm_uncertain_score",
        "llm_confidence",
        "llm_rationale",
        "llm_raw_response",
        "provider_mode",
    ]
    ok, msg = _require_keys(record, required)
    if not ok:
        return False, msg

    # ── Type and value checks ──
    if not _schema_version_ok(record["schema_version"]):
        return False, f"schema_version must be '{SCHEMA_VERSION}' or '{SCHEMA_VERSION}+*'"

    if not _is_non_empty_str(record.get("case_id", "")):
        return False, "case_id must be non-empty string"
    if not _is_non_empty_str(record.get("event_id", "")):
        return False, "event_id must be non-empty string"
    if not isinstance(record.get("track_id"), int):
        return False, "track_id must be int"

    if not _is_non_empty_str(record.get("model_name", "")):
        return False, "model_name must be non-empty string"
    if not _is_non_empty_str(record.get("prompt_version", "")):
        return False, "prompt_version must be non-empty string"
    if not _is_non_empty_str(record.get("generated_at", "")):
        return False, "generated_at must be non-empty string"
    if not _is_non_empty_str(record.get("input_signature", "")):
        return False, "input_signature must be non-empty string"

    label = record.get("llm_label", "")
    if label not in VALID_LABELS:
        return False, f"llm_label must be one of {sorted(VALID_LABELS)}, got '{label}'"

    for score_key in ("llm_match_score", "llm_mismatch_score", "llm_uncertain_score"):
        val = record.get(score_key, -1)
        if not _in_01(val):
            return False, f"{score_key} must be in [0,1], got {val}"

    if not _in_01(record.get("llm_confidence", -1)):
        return False, f"llm_confidence must be in [0,1], got {record.get('llm_confidence')}"

    if not isinstance(record.get("llm_rationale"), str):
        return False, "llm_rationale must be string"

    provider = record.get("provider_mode", "")
    if provider not in VALID_PROVIDER_MODES:
        return False, f"provider_mode must be one of {sorted(VALID_PROVIDER_MODES)}, got '{provider}'"

    # ── Score consistency: the label must correspond to the highest score ──
    label_to_key = {
        "match": "llm_match_score",
        "mismatch": "llm_mismatch_score",
        "uncertain": "llm_uncertain_score",
    }
    label_key = label_to_key.get(label)
    if label_key:
        label_score = float(record[label_key])
        other_scores = [
            float(record[k]) for k in ("llm_match_score", "llm_mismatch_score", "llm_uncertain_score")
            if k != label_key
        ]
        # The label's score should be at least as high as any other score
        if label_score < max(other_scores) - 1e-9:
            return False, (
                f"label '{label}' score ({label_score}) not dominant over "
                f"other scores {[(k, record[k]) for k in ('llm_match_score', 'llm_mismatch_score', 'llm_uncertain_score') if k != label_key]}"
            )

    # ── Score sum should be approximately 1.0 ──
    total = (
        float(record["llm_match_score"])
        + float(record["llm_mismatch_score"])
        + float(record["llm_uncertain_score"])
    )
    if abs(total - 1.0) > 0.05:
        return False, f"score sum {total:.4f} deviates from 1.0 by more than 0.05"

    return True, ""


# ── Bulk validation ─────────────────────────────────────────────────────

def validate_llm_teacher_jsonl(path: Path) -> Tuple[bool, int, List[str]]:
    """Validate an entire JSONL file against the LLM teacher schema."""
    errors: List[str] = []
    count = 0
    if not path.exists():
        return False, 0, [f"file not found: {path}"]
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no}: json decode error: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"{path.name}:{line_no}: record is not a dict")
                continue
            ok, msg = validate_llm_teacher_record(record)
            if not ok:
                errors.append(f"{path.name}:{line_no}: {msg}")
    return len(errors) == 0, count, errors


# ── JSONL writer ────────────────────────────────────────────────────────

def write_teacher_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write LLM teacher records to a JSONL file after schema validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            # Validate before writing
            ok, msg = validate_llm_teacher_record(record)
            if not ok:
                raise ValueError(f"invalid teacher record: {msg}")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
