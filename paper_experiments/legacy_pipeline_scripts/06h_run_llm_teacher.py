#!/usr/bin/env python3
"""06h — LLM Teacher: produce silver labels from multimodal evidence.

This step is part of the LLM judge pipeline. It:
1. Reads pipeline intermediate artifacts (event queries, alignment, actions, uncertainty)
2. Builds structured multimodal evidence (text/JSON only — no video)
3. Calls an LLM teacher (simulate or real API) to produce late-fusion judgments
4. Validates and writes silver labels to output/llm_judge_pipeline/teacher_labels/

Reference: Demirel et al., "Using LLMs for Late Multimodal Sensor Fusion
for Activity Recognition", arXiv:2509.10729

Model requirement: TEXT-ONLY LLM (GPT-4, Claude, Qwen, DeepSeek, etc.)
No vision model needed — the LLM reasons about text descriptions, not video frames.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from classroom_pipeline.contracts.llm_teacher_schema import (
    PROMPT_VERSION,
    make_teacher_record,
    validate_llm_teacher_record,
    validate_llm_teacher_jsonl,
    write_teacher_jsonl,
)
from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, write_json
from classroom_pipeline.verifier.model import action_match_score


# ══════════════════════════════════════════════════════════════════════════
# Evidence Builder
# ══════════════════════════════════════════════════════════════════════════

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_uq_index(path: Optional[Path]) -> Dict[int, float]:
    if path is None or not path.exists():
        return {}
    rows = _load_jsonl(path)
    by_track: Dict[int, List[float]] = {}
    for row in rows:
        persons = row.get("persons")
        if isinstance(persons, list):
            for person in persons:
                if not isinstance(person, dict):
                    continue
                tid = person.get("track_id")
                uq = person.get("uq_track", person.get("uq_score"))
                if isinstance(tid, int) and isinstance(uq, (int, float)):
                    by_track.setdefault(tid, []).append(float(uq))
            continue
        tid = row.get("track_id")
        uq = row.get("uq_score", row.get("uq_track"))
        if isinstance(tid, int) and isinstance(uq, (int, float)):
            by_track.setdefault(tid, []).append(float(uq))
    return {tid: (sum(vals) / len(vals) if vals else 0.5) for tid, vals in by_track.items()}


def build_evidence_pairs(
    *,
    event_queries_path: Path,
    aligned_path: Path,
    actions_path: Optional[Path] = None,
    pose_uq_path: Optional[Path] = None,
    case_id: str = "",
) -> List[Dict[str, Any]]:
    """Build (query, candidate) evidence pairs for the LLM teacher.

    Each pair is a self-contained dict with all structured evidence needed
    for the LLM to make a late-fusion judgment. No video frames.
    """
    queries = _load_jsonl(event_queries_path)
    q_index = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}

    aligned_obj = _load_json(aligned_path)
    aligned = aligned_obj if isinstance(aligned_obj, list) else []

    uq_index: Dict[int, float] = {}
    if pose_uq_path is not None:
        uq_index = _load_uq_index(pose_uq_path)

    pairs: List[Dict[str, Any]] = []

    for block in aligned:
        if not isinstance(block, dict):
            continue
        query_id = str(block.get("query_id", block.get("event_id", "")))
        query = q_index.get(query_id, {})
        event_type = str(query.get("event_type", block.get("event_type", "unknown")))
        query_text = str(query.get("query_text", block.get("query_text", "")))
        query_source = str(query.get("source", block.get("source", "unknown")))
        audio_confidence = _safe_float(query.get("confidence", 0.0), 0.0)

        window = block.get("window", {})
        if not isinstance(window, dict):
            window = {
                "start": _safe_float(block.get("window_start", 0.0), 0.0),
                "end": _safe_float(block.get("window_end", 0.0), 0.0),
            }
        w_start = _safe_float(window.get("start", 0.0), 0.0)
        w_end = _safe_float(window.get("end", max(w_start, 0.0)), max(w_start, 0.0))
        if w_end < w_start:
            w_start, w_end = w_end, w_start

        candidates = block.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        total_candidates = len(candidates)

        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            track_id = int(cand.get("track_id", -1))
            overlap = _safe_float(cand.get("overlap", 0.0), 0.0)
            action_conf = _safe_float(
                cand.get("action_confidence", cand.get("confidence", cand.get("conf", 0.0))), 0.0
            )
            uq = _safe_float(
                cand.get("uq_score", cand.get("uq_track", uq_index.get(track_id, 0.5))), 0.5
            )
            action_label = str(cand.get("semantic_id", cand.get("action", ""))).strip().lower()
            behavior_code = str(cand.get("behavior_code", "")).strip().lower()
            behavior_label_zh = str(cand.get("behavior_label_zh", "")).strip()
            behavior_label_en = str(cand.get("behavior_label_en", "")).strip()
            semantic_label_zh = str(cand.get("semantic_label_zh", "")).strip()
            semantic_label_en = str(cand.get("semantic_label_en", "")).strip()

            # Compute text_score for reference
            text_score = action_match_score(event_type, query_text, action_label)

            pair = {
                "case_id": case_id,
                "event_id": query_id,
                "track_id": track_id,
                "query_text": query_text,
                "event_type": event_type,
                "query_source": query_source,
                "audio_confidence": audio_confidence,
                "window_start": round(w_start, 3),
                "window_end": round(w_end, 3),
                "action_label": action_label,
                "behavior_code": behavior_code,
                "behavior_label_zh": behavior_label_zh,
                "behavior_label_en": behavior_label_en,
                "semantic_label_zh": semantic_label_zh,
                "semantic_label_en": semantic_label_en,
                "overlap": overlap,
                "action_confidence": action_conf,
                "uq_score": uq,
                "text_score": text_score,
                "total_candidates_for_event": total_candidates,
            }
            pairs.append(pair)

    return pairs


# ══════════════════════════════════════════════════════════════════════════
# Prompt Templates
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a **late multimodal fusion judge** for classroom event verification.

Your role:
- You receive **structured text evidence** from multiple classroom sensing modalities:
  • ASR (teacher speech transcription + confidence)
  • Visual behavior detection (student action + confidence)
  • Temporal alignment (time overlap between speech and behavior)
  • Tracking uncertainty (how stable the visual detection is)
- You do NOT see raw video, audio waveforms, or images.
- Your job is to perform **late fusion**: judge whether the detected student behavior MATCHES, MISMATCHES, or is UNCERTAIN relative to the teacher's instruction.

Behavior taxonomy (8-class classroom coding):
- tt (head-up listening): looking at teacher/board
- dx (writing): taking notes, writing
- dk (reading): reading a book or material
- zt (turning head): looking around, possibly distracted
- xt (group discussion): discussing with classmates
- js (raising hand): hand raised to answer or ask
- zl (standing): standing up
- jz (teacher interaction): interacting with teacher

Scoring rules:
- Output THREE scores (match, mismatch, uncertain) that sum to 1.0 (softmax-like).
- The highest score determines the label.
- Be conservative: reserve extreme scores (e.g. 0.95+) for clear cases.
- Consider Chinese classroom context: 讨论 → discussion expected, 派代表 → someone should speak/stand.

Output format — valid JSON only, no surrounding text:
{"match_score": float, "mismatch_score": float, "uncertain_score": float, "label": "match"|"mismatch"|"uncertain", "confidence": float, "rationale": str}"""


USER_PROMPT_TEMPLATE = """=== Multimodal Evidence ===

[Teacher Instruction]
  Text: {query_text}
  Type: {event_type}
  Source: {query_source}
  ASR Confidence: {audio_confidence:.2f}
  Window: [{window_start:.1f}s – {window_end:.1f}s]

[Student Behavior]
  Behavior: {behavior_label_zh} ({behavior_code}) — {behavior_label_en}
  Action Label: {action_label}
  Visual Confidence: {action_confidence:.2f}
  Temporal Overlap with Instruction: {overlap_pct:.0f}%
  Tracking Uncertainty: {uq_score:.2f} (lower = more stable)

[Event Context]
  Total candidates for this event: {total_candidates}

Instruction: Based on the above structured evidence, determine whether the student's behavior matches, mismatches, or is uncertain relative to the teacher's instruction. Output JSON only."""


def format_evidence_text(pair: Dict[str, Any]) -> str:
    """Format an evidence pair as structured text for the LLM prompt."""
    overlap_pct = _clamp01(pair["overlap"]) * 100.0
    return USER_PROMPT_TEMPLATE.format(
        query_text=pair["query_text"],
        event_type=pair["event_type"],
        query_source=pair["query_source"],
        audio_confidence=pair["audio_confidence"],
        window_start=pair["window_start"],
        window_end=pair["window_end"],
        behavior_label_zh=pair["behavior_label_zh"],
        behavior_code=pair["behavior_code"],
        behavior_label_en=pair["behavior_label_en"],
        action_label=pair["action_label"],
        action_confidence=pair["action_confidence"],
        overlap_pct=overlap_pct,
        uq_score=pair["uq_score"],
        total_candidates=pair["total_candidates_for_event"],
    )


# ══════════════════════════════════════════════════════════════════════════
# LLM Response Parsing (with retry)
# ══════════════════════════════════════════════════════════════════════════

def _parse_llm_response(raw: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON response with retry/recovery.

    Handles common issues:
    - JSON embedded in markdown code blocks
    - Trailing commas
    - Extra text before/after JSON
    """
    text = raw.strip()

    for attempt in range(max_retries + 1):
        # Strategy 1: try direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract from markdown code block
        block_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if block_match:
            try:
                result = json.loads(block_match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Strategy 3: find JSON object with braces
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                # Strategy 4: fix common issues
                fixed = re.sub(r',(\s*[}\]])', r'\1', candidate)  # trailing commas
                try:
                    result = json.loads(fixed)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass

        if attempt < max_retries:
            return None  # caller can retry

    return None


# ══════════════════════════════════════════════════════════════════════════
# Simulate Teacher
# ══════════════════════════════════════════════════════════════════════════

def _simulate_teacher(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based simulation of the LLM teacher.

    Uses action_match_score as base and applies keyword adjustments.
    Produces the same output schema as the real LLM.
    """
    event_type = pair["event_type"]
    query_text = pair["query_text"]
    action_label = pair["action_label"]
    behavior_code = pair["behavior_code"]
    overlap = pair["overlap"]
    action_conf = pair["action_confidence"]
    uq = pair["uq_score"]

    # Base score from taxonomic match
    base = action_match_score(event_type, query_text, action_label)

    # Keyword adjustments
    q = query_text.lower()
    adjustment = 0.0

    # Supportive patterns
    if behavior_code in ("tt", "listen") and any(w in q for w in ["听", "看黑板", "看前面", "坐好", "安静"]):
        adjustment = 0.20
    elif behavior_code in ("dx", "write") and any(w in q for w in ["写", "画", "记笔记", "笔记", "填空"]):
        adjustment = 0.20
    elif behavior_code in ("dk", "read") and any(w in q for w in ["读", "看第", "翻开", "看书", "阅读", "朗读"]):
        adjustment = 0.20
    elif behavior_code in ("xt", "group_discussion") and any(w in q for w in ["讨论", "交流", "小组", "商量"]):
        adjustment = 0.25
    elif behavior_code in ("js", "raise_hand", "zl", "stand") and any(w in q for w in ["举手", "回答", "派代表", "发言"]):
        adjustment = 0.20
    elif behavior_code in ("jz", "teacher_interaction") and any(w in q for w in ["互动", "上台", "展示", "演示"]):
        adjustment = 0.18

    # Contradictory patterns
    if behavior_code in ("dx", "write", "dk", "read") and any(w in q for w in ["停笔", "不要写", "看黑板", "坐好"]):
        adjustment = -0.30
    elif behavior_code in ("zt", "turn_head", "phone") and any(w in q for w in ["听", "看黑板", "看前面", "坐好"]):
        adjustment = -0.25
    elif behavior_code in ("tt", "listen") and any(w in q for w in ["讨论", "交流", "小组"]):
        adjustment = -0.15

    # Combine and adjust
    raw_score = base + adjustment
    raw_score = raw_score * (0.6 + 0.4 * overlap)
    raw_score = raw_score * (0.8 + 0.2 * action_conf)
    raw_score = max(0.0, min(1.0, raw_score))

    # Deterministic noise from evidence_id for diversity
    import hashlib
    seed = f"{pair['event_id']}_{pair['track_id']}"
    noise_seed = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    noise = ((noise_seed % 21) - 10) / 200.0
    raw_score = max(0.0, min(1.0, raw_score + noise))

    # Produce three softmax-like scores
    if raw_score >= 0.6:
        match_s = raw_score
        uncertain_s = (1.0 - raw_score) * 0.6
        mismatch_s = 1.0 - match_s - uncertain_s
        label = "match"
    elif raw_score <= 0.4:
        mismatch_s = 1.0 - raw_score
        uncertain_s = raw_score * 0.6
        match_s = 1.0 - mismatch_s - uncertain_s
        label = "mismatch"
    else:
        uncertain_s = 0.6
        match_s = raw_score * 0.5
        mismatch_s = 1.0 - uncertain_s - match_s
        label = "uncertain"

    total = match_s + mismatch_s + uncertain_s
    match_s /= total
    mismatch_s /= total
    uncertain_s /= total

    confidence = 0.5 + 0.5 * abs(raw_score - 0.5) * 2.0
    confidence = _clamp01(confidence * (0.7 + 0.3 * overlap) * (0.7 + 0.3 * action_conf))

    # Rationale
    if adjustment > 0.1:
        rationale = f"Instruction supports behavior '{behavior_code}' (adj={adjustment:+.2f})."
    elif adjustment < -0.1:
        rationale = f"Instruction contradicts behavior '{behavior_code}' (adj={adjustment:+.2f})."
    elif base >= 0.6:
        rationale = f"Behavior '{behavior_code}' is taxonomically consistent with event type '{event_type}'."
    elif base <= 0.3:
        rationale = f"Behavior '{behavior_code}' is not typically associated with '{event_type}'."
    else:
        rationale = "No clear semantic link between instruction and detected behavior."

    return {
        "match_score": round(match_s, 6),
        "mismatch_score": round(mismatch_s, 6),
        "uncertain_score": round(uncertain_s, 6),
        "label": label,
        "confidence": round(confidence, 6),
        "rationale": rationale,
    }


# ══════════════════════════════════════════════════════════════════════════
# Real LLM API (OpenAI-compatible, non-blocking when missing)
# ══════════════════════════════════════════════════════════════════════════

def _api_available(model_name: str) -> bool:
    """Check if the real API is available without raising."""
    if model_name == "simulate":
        return False
    try:
        if model_name.startswith("openai:"):
            import openai
            return True
        elif model_name.startswith("claude:"):
            import anthropic
            return True
        return False
    except ImportError:
        return False


def _call_openai_compatible(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Call an OpenAI-compatible API."""
    try:
        import openai
    except ImportError:
        print("  [WARN] openai SDK not installed, falling back to simulate", file=sys.stderr)
        return None

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = openai.OpenAI(**client_kwargs)
    actual_model = model_name.split(":")[-1] if ":" in model_name else model_name

    try:
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = response.choices[0].message.content.strip()
        parsed = _parse_llm_response(content)
        return parsed
    except Exception as e:
        print(f"  [WARN] API call failed: {e}", file=sys.stderr)
        return None


def _call_claude(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    api_key: str,
    temperature: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        print("  [WARN] anthropic SDK not installed, falling back to simulate", file=sys.stderr)
        return None

    client = anthropic.Anthropic(api_key=api_key)
    actual_model = model_name.split(":")[-1]

    try:
        response = client.messages.create(
            model=actual_model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=500,
            temperature=temperature,
        )
        content = response.content[0].text.strip()
        parsed = _parse_llm_response(content)
        return parsed
    except Exception as e:
        print(f"  [WARN] Claude API call failed: {e}", file=sys.stderr)
        return None


# ══════════════════════════════════════════════════════════════════════════
# Teacher Orchestrator
# ══════════════════════════════════════════════════════════════════════════

def run_llm_teacher(
    pairs: List[Dict[str, Any]],
    *,
    model_name: str,
    api_key: str = "",
    api_base_url: str = "",
    temperature: float = 0.1,
    max_records: int = 0,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run the LLM teacher on a list of evidence pairs.

    Args:
        pairs: evidence pairs from build_evidence_pairs()
        model_name: "simulate", "openai:gpt-4o", "openai:gpt-4o-mini",
                    "claude:sonnet-4-20250506", or any OpenAI-compatible name
        api_key: API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)
        api_base_url: custom base URL for OpenAI-compatible API
        temperature: LLM temperature
        max_records: if > 0, only process this many
        verbose: print progress

    Returns: list of validated teacher output records
    """
    if max_records > 0:
        pairs = pairs[:max_records]

    provider_mode = "simulate" if model_name == "simulate" else "real"

    records: List[Dict[str, Any]] = []
    parse_failures = 0
    api_fallbacks = 0

    for i, pair in enumerate(pairs):
        user_text = format_evidence_text(pair)

        if model_name == "simulate":
            result = _simulate_teacher(pair)
            used_model = "simulate"
            raw_resp = json.dumps(result, ensure_ascii=False)
        else:
            # Try real API
            if model_name.startswith("claude:"):
                api_key_resolved = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key_resolved:
                    if verbose:
                        print(f"  [WARN] No ANTHROPIC_API_KEY, falling back to simulate for record {i}",
                              file=sys.stderr)
                    result = _simulate_teacher(pair)
                    used_model = "simulate"
                    raw_resp = json.dumps(result, ensure_ascii=False)
                    api_fallbacks += 1
                else:
                    llm_result = _call_claude(SYSTEM_PROMPT, user_text, model_name, api_key_resolved, temperature)
                    if llm_result is None:
                        if verbose:
                            print(f"  [WARN] Claude call failed, falling back to simulate for record {i}",
                                  file=sys.stderr)
                        result = _simulate_teacher(pair)
                        used_model = "simulate"
                        api_fallbacks += 1
                    else:
                        result = llm_result
                        used_model = model_name
                        raw_resp = json.dumps(llm_result, ensure_ascii=False)
            else:
                # OpenAI-compatible
                api_key_resolved = api_key or os.environ.get("OPENAI_API_KEY", "")
                if not api_key_resolved:
                    if verbose:
                        print(f"  [WARN] No OPENAI_API_KEY, falling back to simulate for record {i}",
                              file=sys.stderr)
                    result = _simulate_teacher(pair)
                    used_model = "simulate"
                    raw_resp = json.dumps(result, ensure_ascii=False)
                    api_fallbacks += 1
                else:
                    llm_result = _call_openai_compatible(
                        SYSTEM_PROMPT, user_text, model_name, api_key_resolved,
                        base_url=api_base_url if api_base_url else None,
                        temperature=temperature,
                    )
                    if llm_result is None:
                        if verbose:
                            print(f"  [WARN] API call failed, falling back to simulate for record {i}",
                                  file=sys.stderr)
                        result = _simulate_teacher(pair)
                        used_model = "simulate"
                        api_fallbacks += 1
                    else:
                        result = llm_result
                        used_model = model_name
                        raw_resp = json.dumps(llm_result, ensure_ascii=False)

        # Validate LLM response structure
        label = str(result.get("label", "uncertain")).strip().lower()
        if label not in ("match", "mismatch", "uncertain"):
            label = "uncertain"

        for key in ("match_score", "mismatch_score", "uncertain_score"):
            if key not in result or not isinstance(result.get(key), (int, float)):
                result[key] = 0.333333

        confidence = _safe_float(result.get("confidence", 0.5), 0.5)
        confidence = _clamp01(confidence)

        rationale = str(result.get("rationale", ""))

        # Determine effective provider mode: if we fell back to simulate
        effective_provider = "real" if used_model != "simulate" else "simulate"

        record = make_teacher_record(
            case_id=pair["case_id"],
            event_id=pair["event_id"],
            track_id=pair["track_id"],
            query_text=pair["query_text"],
            behavior_code=pair["behavior_code"],
            llm_label=label,
            llm_match_score=result["match_score"],
            llm_mismatch_score=result["mismatch_score"],
            llm_uncertain_score=result["uncertain_score"],
            llm_confidence=confidence,
            llm_rationale=rationale,
            llm_raw_response=raw_resp,
            model_name=used_model,
            provider_mode=effective_provider,
        )

        # Validate before appending
        ok, msg = validate_llm_teacher_record(record)
        if not ok:
            if verbose:
                print(f"  [ERROR] Schema validation failed for record {i}: {msg}", file=sys.stderr)
            parse_failures += 1
            continue

        records.append(record)

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] processed ({effective_provider})")

    if verbose:
        print(f"  [DONE] {len(records)} records, {parse_failures} schema failures, "
              f"{api_fallbacks} API fallbacks")

    return records


# ══════════════════════════════════════════════════════════════════════════
# Smoke Test
# ══════════════════════════════════════════════════════════════════════════

def run_smoke_test(
    records: List[Dict[str, Any]],
    output_dir: Path,
    case_id: str,
) -> Dict[str, Any]:
    """Validate all records and produce a summary."""
    total = len(records)
    labels = Counter(r["llm_label"] for r in records)
    provider_modes = Counter(r["provider_mode"] for r in records)

    match_scores = [r["llm_match_score"] for r in records]
    confidence = [r["llm_confidence"] for r in records]

    # Full schema validation on the output file
    sample_path = output_dir / "llm_teacher_output.sample.jsonl"
    write_teacher_jsonl(sample_path, records)

    ok, count, errors = validate_llm_teacher_jsonl(sample_path)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "total_records": total,
        "schema_validation": {
            "passed": ok,
            "valid_count": count,
            "error_count": len(errors),
            "first_error": errors[0] if errors else None,
        },
        "label_distribution": dict(labels),
        "provider_modes": dict(provider_modes),
        "mean_match_score": round(sum(match_scores) / total, 4) if total else 0.0,
        "mean_confidence": round(sum(confidence) / total, 4) if total else 0.0,
        "output_file": str(sample_path),
    }

    # Write summary
    summary_path = output_dir / "silver_label_summary.json"
    write_json(summary_path, summary)

    print(f"\n[SMOKE] Records: {total}")
    print(f"[SMOKE] Schema: {'PASS' if ok else 'FAIL'} ({count} valid, {len(errors)} errors)")
    print(f"[SMOKE] Labels: {dict(labels)}")
    print(f"[SMOKE] Provider modes: {dict(provider_modes)}")
    print(f"[SMOKE] Mean match score: {summary['mean_match_score']:.4f}")
    print(f"[SMOKE] Mean confidence: {summary['mean_confidence']:.4f}")
    print(f"[SMOKE] Output: {sample_path}")
    print(f"[SMOKE] Summary: {summary_path}")

    return summary


# ══════════════════════════════════════════════════════════════════════════
# Main CLI
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="06h — LLM Teacher: produce silver labels from multimodal evidence"
    )
    # Inputs
    parser.add_argument("--event_queries", required=True, type=str,
                        help="event_queries.fusion_v2.jsonl")
    parser.add_argument("--aligned", required=True, type=str,
                        help="align_multimodal.json")
    parser.add_argument("--actions", default="", type=str,
                        help="actions.fusion_v2.jsonl (optional)")
    parser.add_argument("--pose_uq", default="", type=str,
                        help="pose_tracks_smooth_uq.jsonl (optional)")
    parser.add_argument("--case_id", default="unknown", type=str,
                        help="case identifier for output records")

    # Output
    parser.add_argument("--out_dir", default="output/llm_judge_pipeline/teacher_labels",
                        type=str, help="output directory")

    # Teacher config
    parser.add_argument("--model", default="simulate", type=str,
                        help='LLM model: simulate, openai:gpt-4o-mini, '
                             'claude:sonnet-4-20250506, or any OpenAI-compatible name')
    parser.add_argument("--api_key", default="", type=str,
                        help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    parser.add_argument("--api_base_url", default="", type=str,
                        help="Custom base URL for OpenAI-compatible API")
    parser.add_argument("--temperature", type=float, default=0.1)

    # Control
    parser.add_argument("--max_records", type=int, default=0,
                        help="limit for testing (0 = all)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="run smoke test after generation (max 30 records)")
    args = parser.parse_args()

    # Resolve paths
    event_queries_path = Path(args.event_queries)
    if not event_queries_path.is_absolute():
        event_queries_path = (_PROJECT_ROOT / event_queries_path).resolve()
    aligned_path = Path(args.aligned)
    if not aligned_path.is_absolute():
        aligned_path = (_PROJECT_ROOT / aligned_path).resolve()
    actions_path = Path(args.actions).resolve() if args.actions else None
    pose_uq_path = Path(args.pose_uq).resolve() if args.pose_uq else None

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (_PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Limit for smoke test
    max_records = args.max_records
    if args.smoke_test:
        if max_records <= 0 or max_records > 30:
            max_records = 30

    case_id = args.case_id

    # ── Step 1: Build evidence pairs ──
    print("Building evidence pairs...")
    pairs = build_evidence_pairs(
        event_queries_path=event_queries_path,
        aligned_path=aligned_path,
        actions_path=actions_path,
        pose_uq_path=pose_uq_path,
        case_id=case_id,
    )
    print(f"  {len(pairs)} evidence pairs built")

    if not pairs:
        print("[ERROR] No evidence pairs generated", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Run LLM teacher ──
    print(f"Running LLM teacher (model={args.model})...")
    records = run_llm_teacher(
        pairs,
        model_name=args.model,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        temperature=args.temperature,
        max_records=max_records,
    )

    if not records:
        print("[ERROR] No records produced by LLM teacher", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Smoke test (if requested) ──
    if args.smoke_test:
        print("\n" + "=" * 50)
        print("Running smoke test...")
        print("=" * 50)

        summary = run_smoke_test(records, out_dir, case_id)

        if not summary["schema_validation"]["passed"]:
            print(f"\n[FAIL] Smoke test failed — schema errors: "
                  f"{summary['schema_validation']['first_error']}", file=sys.stderr)
            sys.exit(1)
        print(f"\n[PASS] Smoke test passed")
    else:
        # Write full output without smoke test
        output_path = out_dir / f"llm_teacher_output.{case_id}.jsonl"
        write_teacher_jsonl(output_path, records)
        print(f"\n[DONE] Output: {output_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
