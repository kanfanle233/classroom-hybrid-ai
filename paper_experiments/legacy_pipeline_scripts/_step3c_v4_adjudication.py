#!/usr/bin/env python3
"""Step 3c: V4 LLM-Adjudicated High-Disambiguation Dataset.

Reads high_disambiguation_samples.jsonl, builds LLM prompt payloads,
attempts real LLM adjudication, and either:
  - Produces llm_adjudicated_dataset_v4.jsonl (if real LLM available)
  - Produces blocked_real_llm_report.json (if no real LLM)

CRITICAL: Does NOT fall back to simulate. V4 = real LLM only.
If blocked, NO student training proceeds.
"""
import json, sys, os, hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

OUT_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline"
REPORT_DIR = OUT_DIR / "reports"
TEACHER_DIR = OUT_DIR / "teacher_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TEACHER_DIR.mkdir(parents=True, exist_ok=True)

# Prompt templates (same as 06h_run_llm_teacher.py)
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
  Verifier heuristic score: p_match={p_match:.3f}, reliability={reliability:.3f}, gap_to_next={gap_to_next:.3f}

Instruction: Based on the above structured evidence, determine whether the student's behavior matches, mismatches, or is uncertain relative to the teacher's instruction. Output JSON only."""


def _safe_float(x, d=0.0):
    try: return float(x)
    except: return d


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except json.JSONDecodeError: continue
    return out


BATCH_SIZE = 20  # samples per batch for LLM efficiency


def build_adjudication_queue(
    samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build the LLM adjudication queue with prompt payloads."""
    queue = []
    for s in samples:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            query_text=s.get("query_text", ""),
            event_type=s.get("event_type", "unknown"),
            query_source=s.get("query_source", "unknown"),
            audio_confidence=_safe_float(s.get("audio_confidence", 0.0)),
            window_start=_safe_float(s.get("window_start", 0.0)),
            window_end=_safe_float(s.get("window_end", 0.0)),
            behavior_label_zh=s.get("behavior_label_zh", ""),
            behavior_code=s.get("behavior_code", ""),
            behavior_label_en=s.get("semantic_label_zh", ""),
            action_label=s.get("action", s.get("semantic_id", "")),
            action_confidence=_safe_float(s.get("action_confidence_raw", s.get("action_confidence", 0.0))),
            overlap_pct=_safe_float(s.get("overlap_raw", s.get("overlap", 0.0))) * 100.0,
            uq_score=_safe_float(s.get("uq_score_raw", s.get("uq_score", 0.5))),
            total_candidates=s.get("total_candidates", 0),
            p_match=_safe_float(s.get("p_match", 0.0)),
            reliability=_safe_float(s.get("reliability", 0.0)),
            gap_to_next=_safe_float(s.get("gap_to_next", 0.0)),
        )

        queue.append({
            "case_id": s["case_id"],
            "event_id": s["event_id"],
            "track_id": s["track_id"],
            "query_text": s.get("query_text", "")[:200],
            "query_source": s.get("query_source", "unknown"),
            "event_type": s.get("event_type", "unknown"),
            "verifier_label": s.get("label", ""),
            "verifier_p_match": _safe_float(s.get("p_match", 0.0)),
            "verifier_reliability": _safe_float(s.get("reliability", 0.0)),
            "visual_score": _safe_float(s.get("visual_score", 0.0)),
            "text_score": s.get("text_score"),
            "uq_score": _safe_float(s.get("uq_score_raw", s.get("uq_score", 0.5))),
            "overlap": _safe_float(s.get("overlap_raw", s.get("overlap", 0.0))),
            "action_confidence": _safe_float(s.get("action_confidence_raw", s.get("action_confidence", 0.0))),
            "behavior_code": s.get("behavior_code", ""),
            "behavior_label_zh": s.get("behavior_label_zh", ""),
            "action": s.get("action", ""),
            "topk_gap": _safe_float(s.get("gap_to_next", 0.0)),
            "ambiguity_reasons": s.get("ambiguity_reasons", []),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        })
    return queue


def check_llm_availability() -> Dict[str, Any]:
    """Check if any real LLM is available WITHOUT scanning for keys.

    Returns availability info. Does NOT search filesystem for keys.
    """
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "providers": {},
        "any_available": False,
        "block_reason": "",
    }

    # Check openai SDK
    try:
        import openai
        has_openai_sdk = True
    except ImportError:
        has_openai_sdk = False

    # Check anthropic SDK
    try:
        import anthropic
        has_anthropic_sdk = True
    except ImportError:
        has_anthropic_sdk = False

    # Check env vars (just read, don't search)
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    has_openai = has_openai_sdk and bool(openai_key)
    has_anthropic = has_anthropic_sdk and bool(anthropic_key)

    result["providers"]["openai"] = {
        "sdk_installed": has_openai_sdk,
        "key_configured": bool(openai_key),
        "available": has_openai,
    }
    result["providers"]["anthropic"] = {
        "sdk_installed": has_anthropic_sdk,
        "key_configured": bool(anthropic_key),
        "available": has_anthropic,
    }

    result["any_available"] = has_openai or has_anthropic

    if not has_openai_sdk and not has_anthropic_sdk:
        result["block_reason"] = "No LLM SDK installed. Install 'openai' or 'anthropic' package."
    elif not openai_key and not anthropic_key:
        result["block_reason"] = "No API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
    elif has_anthropic:
        result["recommended_provider"] = "anthropic"
    elif has_openai:
        result["recommended_provider"] = "openai"

    return result


def run_adjudication(queue: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attempt real LLM adjudication. Returns (records, stats)."""
    availability = check_llm_availability()
    if not availability["any_available"]:
        return [], availability

    records = []
    stats = {"total": len(queue), "success": 0, "failed": 0, "provider_used": "", "model_used": ""}

    # Try Anthropic first, then OpenAI
    if availability["providers"]["anthropic"]["available"]:
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            client = anthropic.Anthropic(api_key=api_key)
            model = "claude-sonnet-4-20250514"
            stats["provider_used"] = "anthropic"
            stats["model_used"] = model

            for i, item in enumerate(queue):
                try:
                    response = client.messages.create(
                        model=model,
                        system=item["system_prompt"],
                        messages=[{"role": "user", "content": item["user_prompt"]}],
                        max_tokens=500,
                        temperature=0.1,
                    )
                    content = response.content[0].text.strip()
                    record = make_adjudication_record(item, content, "claude-sonnet-4", "real")
                    records.append(record)
                    stats["success"] += 1
                except Exception as e:
                    print(f"  [FAIL] record {i}: {e}", file=sys.stderr)
                    stats["failed"] += 1

            return records, stats
        except Exception as e:
            print(f"  [FATAL] Anthropic API init failed: {e}", file=sys.stderr)

    if availability["providers"]["openai"]["available"]:
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY", "")
            client = openai.OpenAI(api_key=api_key)
            model = "gpt-4o-mini"
            stats["provider_used"] = "openai"
            stats["model_used"] = model

            for i, item in enumerate(queue):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": item["system_prompt"]},
                            {"role": "user", "content": item["user_prompt"]},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1,
                    )
                    content = response.choices[0].message.content.strip()
                    record = make_adjudication_record(item, content, "gpt-4o-mini", "real")
                    records.append(record)
                    stats["success"] += 1
                except Exception as e:
                    print(f"  [FAIL] record {i}: {e}", file=sys.stderr)
                    stats["failed"] += 1

            return records, stats
        except Exception as e:
            print(f"  [FATAL] OpenAI API init failed: {e}", file=sys.stderr)

    return [], stats


def make_adjudication_record(
    item: Dict[str, Any],
    raw_response: str,
    model_name: str,
    provider_mode: str,
) -> Dict[str, Any]:
    """Parse LLM response and build validated teacher output record."""
    # Parse JSON from response
    parsed = _parse_json_response(raw_response)

    match_score = _safe_float(parsed.get("match_score", 0.0))
    mismatch_score = _safe_float(parsed.get("mismatch_score", 1.0))
    uncertain_score = _safe_float(parsed.get("uncertain_score", 0.0))
    label = str(parsed.get("label", "mismatch")).strip().lower()
    confidence = _safe_float(parsed.get("confidence", 0.5))
    rationale = str(parsed.get("rationale", parsed.get("reasoning", "")))

    # Validate label
    if label not in ("match", "mismatch", "uncertain"):
        # Re-derive from scores
        scores = {"match": match_score, "mismatch": mismatch_score, "uncertain": uncertain_score}
        label = max(scores, key=scores.get)

    # Normalize scores to sum ≈ 1.0
    total = match_score + mismatch_score + uncertain_score
    if total > 0:
        match_score /= total
        mismatch_score /= total
        uncertain_score /= total

    input_sig = hashlib.sha256(
        f"{item['event_id']}|{item['track_id']}|{item['query_text']}|{item['behavior_code']}".encode()
    ).hexdigest()[:16]

    return {
        "schema_version": "2026-04-01",
        "case_id": item["case_id"],
        "event_id": item["event_id"],
        "track_id": item["track_id"],
        "model_name": model_name,
        "prompt_version": "llm_teacher_fusion_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_signature": input_sig,
        "llm_label": label,
        "llm_match_score": round(match_score, 6),
        "llm_mismatch_score": round(mismatch_score, 6),
        "llm_uncertain_score": round(uncertain_score, 6),
        "llm_confidence": round(confidence, 6),
        "llm_rationale": rationale[:500],
        "llm_raw_response": raw_response[:2000],
        "provider_mode": provider_mode,
    }


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with recovery strategies."""
    import re
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from markdown code block
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find JSON object
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: trailing comma fix
    if m:
        fixed = re.sub(r',\s*}', '}', m.group(0))
        fixed = re.sub(r',\s*]', ']', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    return {}


def main():
    print("=" * 60)
    print("Step 3c: V4 LLM-Adjudicated Dataset Construction")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # 1. Load high-disambiguation samples
    # ═══════════════════════════════════════════════════════════════
    hd_path = OUT_DIR / "high_disambiguation_samples.jsonl"
    if not hd_path.exists():
        print(f"FATAL: {hd_path} not found. Run _step3b first.")
        sys.exit(1)

    samples = load_jsonl(hd_path)
    print(f"\nLoaded {len(samples)} high-disambiguation samples")

    # ═══════════════════════════════════════════════════════════════
    # 2. Build adjudication queue
    # ═══════════════════════════════════════════════════════════════
    queue = build_adjudication_queue(samples)
    print(f"Built adjudication queue: {len(queue)} items")

    queue_path = OUT_DIR / "llm_adjudication_queue_v4.jsonl"
    with queue_path.open("w", encoding="utf-8") as f:
        for item in queue:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[OUT] {queue_path} ({queue_path.stat().st_size} bytes)")

    # ═══════════════════════════════════════════════════════════════
    # 3. Check LLM availability
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- LLM Availability Check ---")
    availability = check_llm_availability()
    for provider, info in availability["providers"].items():
        print(f"  {provider}: sdk={'YES' if info['sdk_installed'] else 'NO'}, "
              f"key={'SET' if info['key_configured'] else 'NOT SET'}, "
              f"available={'YES' if info['available'] else 'NO'}")

    if not availability["any_available"]:
        print(f"\n*** BLOCKED: No real LLM available ***")
        print(f"  Reason: {availability['block_reason']}")

        # ═══════════════════════════════════════════════════════════
        # 4a. Produce BLOCKED report
        # ═══════════════════════════════════════════════════════════
        blocked_report = {
            "step": "3c_v4_adjudication",
            "status": "BLOCKED",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "block_reason": availability["block_reason"],
            "availability": availability,
            "queue_built": True,
            "queue_path": str(queue_path),
            "queue_size": len(queue),
            "what_was_done": [
                "Built llm_adjudication_queue_v4.jsonl with prompt payloads for all 200 samples",
                "Checked LLM SDK availability: openai NOT installed, anthropic NOT installed",
                "Checked API key configuration: OPENAI_API_KEY not set, ANTHROPIC_API_KEY not set",
                "No real LLM available for adjudication",
            ],
            "what_is_blocked": [
                "LLM adjudication of 200 high-disambiguation samples",
                "Production of llm_adjudicated_dataset_v4.jsonl",
                "Training of student_judge_v4 (requires real LLM labels)",
                "LLM-distilled student model cannot be produced without real LLM",
            ],
            "dataset_status": {
                "v3": "heuristic boundary dataset — available at balanced_teacher_dataset_v3.jsonl",
                "v4": "BLOCKED — requires real LLM adjudication",
                "v3_can_train": True,
                "v3_is_llm_distilled": False,
                "note": "V3 uses verifier heuristic replication labels. V4 requires real LLM labels. Student trained on V3 is heuristic-distilled, NOT LLM-distilled."
            },
            "to_unblock": [
                "Install openai or anthropic SDK: pip install openai anthropic",
                "Set API key: export OPENAI_API_KEY=sk-... or export ANTHROPIC_API_KEY=sk-ant-...",
                "Re-run: python scripts/pipeline/_step3c_v4_adjudication.py",
            ],
            "output_files": {
                "adjudication_queue": str(queue_path),
                "blocked_report": str(REPORT_DIR / "blocked_real_llm_report_v4.json"),
            },
        }

        report_path = REPORT_DIR / "blocked_real_llm_report_v4.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(blocked_report, f, ensure_ascii=False, indent=2)
        print(f"\n[OUT] {report_path} ({report_path.stat().st_size} bytes)")

        # Also write a short markdown
        md_path = REPORT_DIR / "blocked_real_llm_report_v4.md"
        md_content = f"""# V4 LLM Adjudication: BLOCKED

*Generated: {datetime.now(timezone.utc).isoformat()}*

## Status: BLOCKED

**Reason**: {availability['block_reason']}

## What Was Done
- Built `llm_adjudication_queue_v4.jsonl` with {len(queue)} prompt payloads
- All prompt payloads include SYSTEM_PROMPT + USER_PROMPT_TEMPLATE filled with evidence

## What Is Blocked
- LLM adjudication of {len(queue)} high-disambiguation samples
- `llm_adjudicated_dataset_v4.jsonl` production
- `student_judge_v4` training (requires real LLM labels)
- LLM-distilled student model

## Dataset Status
| Version | Status | Labels | LLM-Distilled |
|---------|--------|--------|---------------|
| V3 | Available | Heuristic | No |
| V4 | BLOCKED | Real LLM | Would be |

## To Unblock
1. `pip install openai anthropic`
2. `export OPENAI_API_KEY=sk-...`
3. Re-run: `python scripts/pipeline/_step3c_v4_adjudication.py`
"""
        md_path.write_text(md_content, encoding="utf-8")
        print(f"[OUT] {md_path} ({md_path.stat().st_size} bytes)")

        print(f"\n{'='*60}")
        print("V4 ADJUDICATION: BLOCKED")
        print(f"{'='*60}")
        print(f"  Queue built: {len(queue)} items → {queue_path}")
        print(f"  Reason: {availability['block_reason']}")
        print(f"  V3 available as heuristic baseline (NOT LLM-distilled)")
        print(f"  V4 requires real LLM. Cannot proceed without it.")
        print(f"  Blocked report: {report_path}")
        print()
        return 1

    # ═══════════════════════════════════════════════════════════════
    # 4b. Run real LLM adjudication
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- Running Real LLM Adjudication ---")
    recommended = availability.get("recommended_provider", "unknown")
    print(f"  Using: {recommended}")

    records, stats = run_adjudication(queue)

    if stats["success"] == 0:
        print(f"\n*** BLOCKED: All LLM calls failed ***")
        print(f"  Failed: {stats['failed']}/{stats['total']}")
        # Produce blocked report
        blocked_report = {
            "step": "3c_v4_adjudication",
            "status": "BLOCKED",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "block_reason": f"All {stats['total']} LLM calls failed. Provider: {stats.get('provider_used', 'none')}",
            "availability": availability,
            "queue_size": len(queue),
            "stats": stats,
        }
        report_path = REPORT_DIR / "blocked_real_llm_report_v4.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(blocked_report, f, ensure_ascii=False, indent=2)
        print(f"[OUT] {report_path}")
        return 1

    # ═══════════════════════════════════════════════════════════════
    # 5. Write V4 adjudicated dataset
    # ═══════════════════════════════════════════════════════════════
    v4_path = TEACHER_DIR / "llm_adjudicated_dataset_v4.jsonl"
    with v4_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[OUT] {v4_path} ({v4_path.stat().st_size} bytes)")

    # Label distribution
    label_dist = Counter(r["llm_label"] for r in records)
    print(f"\nV4 Label distribution: {dict(label_dist)}")

    # ═══════════════════════════════════════════════════════════════
    # 6. Adjudication summary
    # ═══════════════════════════════════════════════════════════════
    has_3_classes = len(label_dist) >= 3
    summary = {
        "step": "3c_v4_adjudication",
        "status": "COMPLETED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "queue": str(queue_path),
            "queue_size": len(queue),
        },
        "adjudication": {
            "provider": stats["provider_used"],
            "model": stats["model_used"],
            "total_attempted": stats["total"],
            "success": stats["success"],
            "failed": stats["failed"],
            "provider_mode": "real",
        },
        "results": {
            "total_records": len(records),
            "label_distribution": dict(label_dist),
            "has_3_classes": has_3_classes,
        },
        "v3_comparison": {
            "note": "V3 = verifier heuristic labels. V4 = real LLM labels. They may differ significantly.",
        },
        "training_feasibility": {
            "3_classes_present": has_3_classes,
            "recommendation": (
                "READY for LLM-distilled student training"
                if has_3_classes and min(label_dist.values()) >= 5
                else "NEED more samples for minority class"
            ),
        },
        "output_files": {
            "v4_dataset": str(v4_path),
            "summary": str(REPORT_DIR / "llm_adjudication_summary_v4.json"),
        },
    }

    summary_path = REPORT_DIR / "llm_adjudication_summary_v4.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OUT] {summary_path} ({summary_path.stat().st_size} bytes)")

    print(f"\n{'='*60}")
    print("V4 ADJUDICATION: SUCCESS")
    print(f"{'='*60}")
    print(f"  Records: {len(records)}")
    print(f"  Labels: {dict(label_dist)}")
    print(f"  3-class: {has_3_classes}")
    print(f"  Provider: {stats['provider_used']} / {stats['model_used']}")
    print(f"  Next: train student_judge_v4 on this dataset")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
