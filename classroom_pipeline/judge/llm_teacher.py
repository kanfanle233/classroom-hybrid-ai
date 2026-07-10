"""LLM Teacher: late-fusion semantic judge for classroom event verification.

Each (query, candidate) evidence record is presented to the LLM as a text-only
structured prompt. The LLM outputs a silver label: match / uncertain / mismatch.

Supports:
  - simulate: rule-based simulation for development (no API cost)
  - openai:gpt-4o / openai:gpt-4o-mini
  - claude:sonnet-4-20250506 / claude:haiku-3-5

Reference: Demirel et al., "Using LLMs for Late Multimodal Sensor Fusion
for Activity Recognition", arXiv:2509.10729
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, write_jsonl


# ── Behavior taxonomy (8-class, matches verifier/model.py EVENT_TO_ACTIONS) ──

BEHAVIOR_TAXONOMY = {
    "tt": "抬头听课 (head-up listening) — student looking at teacher/board",
    "dx": "低头写字 (writing) — student writing or taking notes",
    "dk": "低头看书 (reading) — student reading a book or material",
    "zt": "转头 (turning head) — student turning head, possibly distracted",
    "xt": "小组讨论 (group discussion) — student engaged in group discussion",
    "js": "举手 (raising hand) — student raising hand to answer or ask",
    "zl": "站立 (standing) — student standing up",
    "jz": "教师互动 (teacher interaction) — student interacting with teacher",
}


# ── Prompt templates ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a classroom behavior analysis expert. Your task is to judge whether a student's detected behavior matches a teacher's spoken instruction.

You receive:
1. TEACHER INSTRUCTION — Chinese text from classroom ASR
2. STUDENT BEHAVIOR — detected student action with confidence
3. TIME CONTEXT — temporal alignment between instruction and behavior

Behavior taxonomy:
- tt (head-up listening): looking at teacher/board
- dx (writing): taking notes, writing
- dk (reading): reading a book or material
- zt (turning head): looking around, possibly distracted
- xt (group discussion): discussing with classmates
- js (raising hand): hand raised to answer or ask
- zl (standing): standing up
- jz (teacher interaction): interacting with teacher

Rules:
- A behavior MATCHES if it is a reasonable response to the instruction
- A behavior MISMATCHES if it clearly contradicts the instruction
- Use UNCERTAIN when evidence is ambiguous
- Consider Chinese classroom context: e.g. "讨论" → discussion expected
- Be conservative: strong judgments only when evidence is clear

Respond ONLY with valid JSON:
{"judgment": "match"|"mismatch"|"uncertain", "p_match": float [0.0-1.0], "reasoning": str, "confidence": float [0.0-1.0]}"""


USER_PROMPT_TEMPLATE = """TEACHER INSTRUCTION: {query_text}
INSTRUCTION TYPE: {event_type}
AUDIO CONFIDENCE: {audio_confidence:.2f}
TIME WINDOW: [{window_start:.1f}s, {window_end:.1f}s]

STUDENT BEHAVIOR: {behavior_label_zh} ({behavior_code}) — {behavior_label_en}
VISUAL CONFIDENCE: {action_confidence:.2f}
TIME OVERLAP WITH WINDOW: {overlap_pct:.0f}%
TRACKING UNCERTAINTY: {uq_score:.2f} (lower = more stable)

EVENT TYPE: {event_type_desc}

QUESTION: Does this student's behavior MATCH, MISMATCH, or is it UNCERTAIN relative to the teacher's instruction?
Output JSON only: {{"judgment": "match"|"mismatch"|"uncertain", "p_match": float [0-1], "reasoning": str, "confidence": float [0-1]}}"""


# ── Config ───────────────────────────────────────────────────────────────

@dataclass
class LLMTeacherConfig:
    model: str = "simulate"
    api_key: str = ""
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.1
    max_retries: int = 3
    concurrency: int = 1


# ── Evidence formatting ──────────────────────────────────────────────────

def _format_user_prompt(record: Dict[str, Any]) -> str:
    q = record["query"]
    c = record["candidate"]
    query_text = q.get("query_text", "")
    event_type = q.get("event_type", "unknown")
    audio_conf = float(q.get("audio_confidence", 0.0))
    w_start = float(q.get("window_start", 0.0))
    w_end = float(q.get("window_end", 0.0))
    behavior_zh = c.get("behavior_label_zh", "?")
    behavior_code = c.get("behavior_code", "?")
    behavior_en = c.get("behavior_label_en", "?")
    action_conf = float(c.get("action_confidence", 0.0))
    overlap = float(c.get("overlap", 0.0))
    uq = float(c.get("uq_score", 0.5))

    # Pretty overlap as percentage
    overlap_pct = overlap * 100.0

    # Describe event type
    event_type_desc = event_type if event_type != "unknown" else "could not be classified by rule-based detector"

    return USER_PROMPT_TEMPLATE.format(
        query_text=query_text,
        event_type=event_type,
        audio_confidence=audio_conf,
        window_start=w_start,
        window_end=w_end,
        behavior_label_zh=behavior_zh,
        behavior_code=behavior_code,
        behavior_label_en=behavior_en,
        action_confidence=action_conf,
        overlap_pct=overlap_pct,
        uq_score=uq,
        event_type_desc=event_type_desc,
    )


# ── Simulation (rule-based, zero API cost) ──────────────────────────────

def _simulate_judgment(record: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based simulation using behavior taxonomy + Chinese keyword matching.

    Uses action_match_score for known event types, and Chinese keyword
    matching between query text and behavior labels when event_type is unknown.
    Ensures diverse label distribution across all three classes.
    """
    from classroom_pipeline.verifier.model import action_match_score

    q = record["query"]
    c = record["candidate"]
    query_text = str(q.get("query_text", ""))
    event_type = str(q.get("event_type", "unknown"))
    behavior_code = c.get("behavior_code", "").lower()
    action_label = c.get("action", "").lower()
    behavior_label_zh = str(c.get("behavior_label_zh", "")).lower()
    behavior_label_en = str(c.get("behavior_label_en", "")).lower()
    overlap = float(c.get("overlap", 0.0))
    action_conf = float(c.get("action_confidence", 0.0))
    uq = float(c.get("uq_score", 0.5))

    # Base score
    if event_type != "unknown":
        base_score = action_match_score(event_type, query_text, action_label)
    else:
        # When event_type is unknown, compute a text similarity between
        # query_text (Chinese) and behavior_label_zh (Chinese)
        base_score = _chinese_text_score(query_text, behavior_label_zh, behavior_label_en, behavior_code)

    # Apply keyword-based semantic adjustments
    adjustment = _keyword_adjustment(query_text, behavior_code)

    # Combine
    score = base_score + adjustment

    # Adjust by overlap and confidence
    score = score * (0.6 + 0.4 * overlap)
    score = score * (0.8 + 0.2 * action_conf)

    # Deterministic noise from evidence_id for diversity
    import hashlib
    seed_str = str(record.get("evidence_id", ""))
    noise_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    noise = ((noise_seed % 21) - 10) / 200.0  # ±0.05
    score = score + noise
    score = max(0.0, min(1.0, score))

    # Determine judgment from score
    if score >= 0.6:
        judgment = "match"
    elif score <= 0.4:
        judgment = "mismatch"
    else:
        judgment = "uncertain"

    # Confidence
    confidence = 0.5 + 0.5 * abs(score - 0.5) * 2
    confidence = min(1.0, max(0.0, confidence))
    confidence *= (0.7 + 0.3 * overlap) * (0.7 + 0.3 * action_conf)
    confidence = min(1.0, max(0.0, confidence))

    # Reasoning
    reasoning_parts = []
    if adjustment > 0.1:
        reasoning_parts.append(f"Instruction supports behavior '{behavior_code}'.")
    elif adjustment < -0.1:
        reasoning_parts.append(f"Instruction contradicts behavior '{behavior_code}'.")
    else:
        if base_score >= 0.6:
            reasoning_parts.append(f"Behavior '{behavior_code}' is generally consistent.")
        elif base_score <= 0.4:
            reasoning_parts.append(f"Behavior '{behavior_code}' is generally inconsistent.")
        else:
            reasoning_parts.append("No clear semantic link detected.")
    if overlap < 0.3:
        reasoning_parts.append("Low temporal overlap.")
    if action_conf < 0.5:
        reasoning_parts.append("Low visual confidence.")

    return {
        "judgment": judgment,
        "p_match": round(score, 4),
        "reasoning": " ".join(reasoning_parts),
        "confidence": round(confidence, 4),
        "_simulated": True,
    }


def _chinese_text_score(
    query_text: str,
    behavior_label_zh: str,
    behavior_label_en: str,
    behavior_code: str,
) -> float:
    """Compute a semantic similarity score between Chinese query text and behavior labels.

    Uses keyword matching against Chinese and English behavior descriptions.
    Returns float in [0, 1].
    """
    q = query_text.lower()

    # Map behavior codes to keywords that would appear in teacher instructions
    behavior_keywords = {
        "tt": {"听", "看", "look", "listen", "前面", "黑板", "坐好", "安静", "注意"},
        "listen": {"听", "look", "listen", "前面", "黑板", "坐好", "安静", "注意"},
        "dx": {"写", "画", "记", "note", "write", "draw", "笔记", "填空", "练习"},
        "write": {"写", "画", "记", "note", "write", "draw", "笔记", "填空", "练习"},
        "dk": {"读", "看", "read", "book", "书", "阅读", "朗读", "翻开", "翻到"},
        "read": {"读", "看", "read", "book", "书", "阅读", "朗读", "翻开", "翻到"},
        "xt": {"讨论", "交流", "小组", "discuss", "group", "商量", "合作"},
        "group_discussion": {"讨论", "交流", "小组", "discuss", "group", "商量", "合作"},
        "js": {"举手", "回答", "raise", "answer", "发言", "提问", "谁说", "派代表"},
        "raise_hand": {"举手", "回答", "raise", "answer", "发言", "提问", "谁说", "派代表"},
        "zl": {"站", "stand", "上台", "演示", "展示", "present"},
        "stand": {"站", "stand", "上台", "演示", "展示", "present"},
        "zt": {"转", "turn", "看", "回头"},
        "turn_head": {"转", "turn", "看", "回头"},
        "jz": {"互动", "上台", "展示", "present", "interact", "演示", "回答"},
        "teacher_interaction": {"互动", "上台", "展示", "present", "interact", "演示", "回答"},
    }

    keywords = behavior_keywords.get(behavior_code, set())
    if not keywords and behavior_label_en:
        en_words = set(behavior_label_en.lower().split())
        keywords = en_words

    if not keywords:
        return 0.5

    # Count matches
    matches = sum(1 for kw in keywords if kw in q)
    ratio = matches / len(keywords) if keywords else 0.0

    # Map to score: higher ratio → higher match score
    if ratio >= 0.3:
        return 0.7 + 0.3 * min(1.0, ratio)
    elif ratio >= 0.1:
        return 0.5 + 0.2 * (ratio / 0.3)
    else:
        return 0.3 + 0.2 * (ratio / 0.1)


def _keyword_adjustment(query_text: str, behavior_code: str) -> float:
    """Apply keyword-based adjustment for known classroom patterns."""
    q = query_text.lower()
    adjustment = 0.0

    # Strong supportive patterns
    if behavior_code in ("tt", "listen") and any(
        w in q for w in ["听", "看黑板", "看前面", "坐好", "安静", "look at"]
    ):
        adjustment = 0.20
    elif behavior_code in ("dx", "write") and any(
        w in q for w in ["写", "画", "记笔记", "笔记", "填空题", "write", "draw"]
    ):
        adjustment = 0.20
    elif behavior_code in ("dk", "read") and any(
        w in q for w in ["读", "看第", "翻开", "看书", "阅读", "朗读", "翻到", "read", "book"]
    ):
        adjustment = 0.20
    elif behavior_code in ("xt", "group_discussion") and any(
        w in q for w in ["讨论", "交流", "小组", "商量", "合作", "discuss", "group"]
    ):
        adjustment = 0.25
    elif behavior_code in ("js", "raise_hand", "zl", "stand") and any(
        w in q for w in ["举手", "回答", "派代表", "谁说", "raise", "answer", "发言", "提问"]
    ):
        adjustment = 0.20
    elif behavior_code in ("jz", "teacher_interaction") and any(
        w in q for w in ["互动", "上台", "展示", "interact", "present", "演示", "回答"]
    ):
        adjustment = 0.18

    # Contradictory patterns
    if behavior_code in ("dx", "write", "dk", "read") and any(
        w in q for w in ["停笔", "不要写", "把手放", "stop writing", "看黑板", "看前面", "坐好", "安静"]
    ):
        adjustment = -0.30
    elif behavior_code in ("dx", "write") and any(
        w in q for w in ["派代表", "介绍", "上台", "present", "演示", "展示", "发言"]
    ):
        adjustment = -0.20
    elif behavior_code in ("zt", "turn_head", "phone") and any(
        w in q for w in ["听", "看黑板", "看前面", "坐好", "安静", "注意", "listen"]
    ):
        adjustment = -0.25
    elif behavior_code in ("tt", "listen") and any(
        w in q for w in ["讨论", "交流", "小组", "商量", "合作", "discuss"]
    ):
        adjustment = -0.15
    elif behavior_code in ("js", "raise_hand") and any(
        w in q for w in ["写", "画", "记笔记", "笔记", "write", "draw"]
    ):
        adjustment = -0.15

    return adjustment


# ── Real LLM calls ───────────────────────────────────────────────────────

def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
) -> Dict[str, Any]:
    """Call OpenAI API and parse JSON response."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai SDK not installed; use model='simulate' or pip install openai")

    client = openai.OpenAI(api_key=api_key)
    model_name = model.split(":")[-1]  # strip "openai:" prefix

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )

    content = response.choices[0].message.content.strip()
    result = json.loads(content)
    result["_simulated"] = False
    result["_provider"] = "openai"
    result["_model"] = model_name
    result["_prompt_tokens"] = response.usage.prompt_tokens if response.usage else 0
    result["_completion_tokens"] = response.usage.completion_tokens if response.usage else 0
    return result


def _call_claude(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
) -> Dict[str, Any]:
    """Call Anthropic Claude API and parse JSON response."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic SDK not installed; use model='simulate' or pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    model_name = model.split(":")[-1]

    response = client.messages.create(
        model=model_name,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=500,
        temperature=temperature,
    )

    content = response.content[0].text.strip()
    result = json.loads(content)
    result["_simulated"] = False
    result["_provider"] = "anthropic"
    result["_model"] = model_name
    result["_prompt_tokens"] = response.usage.input_tokens if response.usage else 0
    result["_completion_tokens"] = response.usage.output_tokens if response.usage else 0
    return result


# ── Teacher API ──────────────────────────────────────────────────────────

def call_llm_teacher(
    record: Dict[str, Any],
    config: LLMTeacherConfig,
) -> Dict[str, Any]:
    """Call LLM teacher on a single evidence record.

    Returns a dict with silver_label, silver_p_match, reasoning, etc.
    """
    user_prompt = _format_user_prompt(record)

    if config.model == "simulate":
        response = _simulate_judgment(record)
    elif config.model.startswith("openai:"):
        response = _call_openai(
            SYSTEM_PROMPT, user_prompt, config.model,
            config.api_key or os.environ.get(config.api_key_env, ""),
            config.temperature,
        )
    elif config.model.startswith("claude:"):
        response = _call_claude(
            SYSTEM_PROMPT, user_prompt, config.model,
            config.api_key or os.environ.get(config.api_key_env, ""),
            config.temperature,
        )
    else:
        raise ValueError(f"Unknown teacher model: {config.model}")

    # Normalize response to silver label format
    judgment = str(response.get("judgment", "uncertain")).strip().lower()
    if judgment not in ("match", "mismatch", "uncertain"):
        judgment = "uncertain"

    p_match = float(response.get("p_match", 0.5))
    p_match = max(0.0, min(1.0, p_match))

    reasoning = str(response.get("reasoning", ""))
    llm_confidence = float(response.get("confidence", 0.5))
    llm_confidence = max(0.0, min(1.0, llm_confidence))

    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": str(record.get("evidence_id", "")),
        "query_id": str(record.get("query_id", "")),
        "track_id": int(record.get("track_id", -1)),
        "teacher_model": config.model,
        "teacher_temperature": config.temperature,
        "silver_label": judgment,
        "silver_p_match": round(p_match, 4),
        "silver_confidence": round(llm_confidence, 4),
        "llm_reasoning": reasoning,
        "prompt_tokens": int(response.get("_prompt_tokens", 0)),
        "completion_tokens": int(response.get("_completion_tokens", 0)),
        "is_simulated": bool(response.get("_simulated", True)),
    }


def run_llm_teacher_batch(
    evidence_path: Path,
    output_path: Path,
    config: LLMTeacherConfig,
    max_records: int = 0,
) -> List[Dict[str, Any]]:
    """Process all evidence records through the LLM teacher.

    Args:
        evidence_path: Path to llm_evidence.jsonl
        output_path: Path for output llm_silver_labels.jsonl
        config: LLMTeacherConfig
        max_records: If > 0, only process this many records (for testing)
    """
    evidence_list: List[Dict[str, Any]] = []
    with evidence_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evidence_list.append(json.loads(line))
            except Exception:
                continue

    if max_records > 0:
        evidence_list = evidence_list[:max_records]

    silver_labels: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, record in enumerate(evidence_list):
        try:
            result = call_llm_teacher(record, config)
            silver_labels.append(result)
            total_prompt_tokens += result.get("prompt_tokens", 0)
            total_completion_tokens += result.get("completion_tokens", 0)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(evidence_list)}] processed, "
                      f"tokens ~{total_prompt_tokens + total_completion_tokens}")
        except Exception as e:
            print(f"  [ERROR] record {i} ({record.get('evidence_id', '?')}): {e}", file=sys.stderr)
            # Still write a fallback entry so we don't lose alignment
            silver_labels.append({
                "schema_version": SCHEMA_VERSION,
                "evidence_id": str(record.get("evidence_id", "")),
                "query_id": str(record.get("query_id", "")),
                "track_id": int(record.get("track_id", -1)),
                "teacher_model": config.model,
                "teacher_temperature": config.temperature,
                "silver_label": "uncertain",
                "silver_p_match": 0.5,
                "silver_confidence": 0.0,
                "llm_reasoning": f"ERROR: {e}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "is_simulated": True,
            })

        # Rate limiting for real API calls
        if not config.model.startswith("simulate") and (i + 1) % 5 == 0:
            time.sleep(0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, silver_labels)

    print(f"[DONE] LLM silver labels: {output_path} ({len(silver_labels)} records)")
    print(f"[INFO] Model: {config.model}, simulated={config.model == 'simulate'}")
    print(f"[INFO] Total prompt tokens: {total_prompt_tokens}, completion: {total_completion_tokens}")
    if total_prompt_tokens + total_completion_tokens > 0:
        cost = total_prompt_tokens * 0.00001 + total_completion_tokens * 0.00003
        print(f"[INFO] Estimated cost: ${cost:.4f}")

    return silver_labels


# ── Smoke check ──────────────────────────────────────────────────────────

def smoke_check_silver_labels(
    silver_labels: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate silver label quality and distribution."""
    if not silver_labels:
        return {"passed": False, "detail": "No silver labels generated"}

    total = len(silver_labels)
    labels = [s.get("silver_label", "") for s in silver_labels]
    p_matches = [float(s.get("silver_p_match", 0.5)) for s in silver_labels]
    confs = [float(s.get("silver_confidence", 0.0)) for s in silver_labels]

    # All labels must be valid
    valid_labels = all(l in ("match", "uncertain", "mismatch") for l in labels)
    # All p_match in [0, 1]
    valid_pmatch = all(0.0 <= p <= 1.0 for p in p_matches)
    # All confidence in [0, 1]
    valid_conf = all(0.0 <= c <= 1.0 for c in confs)

    # Distribution
    from collections import Counter
    dist = Counter(labels)
    coverage_ok = all(
        dist.get(label, 0) / total > 0.05
        for label in ("match", "uncertain", "mismatch")
    )

    # No missing evidence_ids
    all_have_ids = all(bool(s.get("evidence_id", "")) for s in silver_labels)

    errors = []
    if not valid_labels:
        errors.append("invalid labels found")
    if not valid_pmatch:
        errors.append("p_match out of [0,1]")
    if not valid_conf:
        errors.append("confidence out of [0,1]")
    if not coverage_ok:
        errors.append(f"label distribution lacks coverage: {dict(dist)}")
    if not all_have_ids:
        errors.append("missing evidence_ids")

    passed = len(errors) == 0
    return {
        "passed": passed,
        "total": total,
        "label_distribution": dict(dist),
        "mean_p_match": round(sum(p_matches) / total, 4),
        "mean_confidence": round(sum(confs) / total, 4),
        "all_have_ids": all_have_ids,
        "errors": errors,
        "detail": "OK" if passed else "; ".join(errors),
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: LLM Teacher — produce silver labels from evidence"
    )
    parser.add_argument("--evidence", required=True, type=str,
                        help="llm_evidence.jsonl from step 1")
    parser.add_argument("--out", required=True, type=str,
                        help="llm_silver_labels.jsonl")
    parser.add_argument("--model", default="simulate",
                        help="LLM model: simulate, openai:gpt-4o, claude:sonnet-4-20250506")
    parser.add_argument("--api_key", default="",
                        help="API key (or set LLM_API_KEY env var)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_records", type=int, default=0,
                        help="limit for testing (0 = all)")
    parser.add_argument("--smoke_report", default="", type=str)
    args = parser.parse_args()

    config = LLMTeacherConfig(
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    labels = run_llm_teacher_batch(
        evidence_path=Path(args.evidence),
        output_path=Path(args.out),
        config=config,
        max_records=args.max_records,
    )

    check = smoke_check_silver_labels(labels)
    if args.smoke_report:
        smoke_path = Path(args.smoke_report)
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        with smoke_path.open("w", encoding="utf-8") as f:
            json.dump(check, f, ensure_ascii=False, indent=2)

    if not check["passed"]:
        print(f"[SMOKE FAIL] {check['detail']}")
        sys.exit(1)
    else:
        print(f"[SMOKE PASS] {check['detail']}")
        print(f"[INFO] Distribution: {check['label_distribution']}")


if __name__ == "__main__":
    main()
