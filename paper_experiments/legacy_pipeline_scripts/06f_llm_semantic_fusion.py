"""LLM-based Semantic Decision Fusion for Audio-Visual Classroom Behavior Analysis.

Reference: Demirel et al. (Apple/MIT, 2025) — "Using LLMs for Late Multimodal Sensor Fusion"

This module:
1. Takes instruction_context.json (teacher instructions from ASR) + actions data (visual behaviors)
2. Formats semantic fusion prompts for an LLM
3. Returns LLM-judged confidence adjustments

Model requirement: TEXT-ONLY LLM (GPT-4, Claude, Qwen, DeepSeek, etc.)
No vision model needed — the LLM reasons about text descriptions, not video frames.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# LLM Prompt Templates
# ============================================================

SYSTEM_PROMPT = """You are an expert in classroom behavior analysis. Your task is to evaluate whether a teacher's spoken instruction supports or contradicts a student's visually detected behavior.

You will receive:
1. TEACHER INSTRUCTION: Text transcribed from classroom audio (Chinese)
2. STUDENT BEHAVIOR: A visually detected student action with confidence score
3. TIME CONTEXT: The time window in seconds

Your job:
- If the instruction SUPPORTS the detected behavior (e.g., teacher says "open your books" and student is detected "reading"), output a positive boost (0.0 to +0.25).
- If the instruction CONTRADICTS the detected behavior (e.g., teacher says "stop writing" and student is detected "writing"), output a negative boost (-0.25 to 0.0).
- If the instruction is UNRELATED (e.g., teacher says "good morning" and student is detected "writing"), output 0.0.
- If UNCERTAIN, output a small value near 0.0 and explain why.

IMPORTANT RULES:
- Only adjust by ±0.25 maximum. Be conservative.
- Consider Chinese language context carefully (e.g., "派代表" = send a representative → student should LISTEN/STAND, not WRITE)
- Consider that some students may ignore the instruction (only give strong boosts/dampens if the link is clear)
- Output MUST be valid JSON with exactly these fields: {"boost": float, "reasoning": str, "confidence": float}
"""

USER_PROMPT_TEMPLATE = """TEACHER INSTRUCTION (Chinese): {instruction_text}
INSTRUCTION TIME: {instruction_start}s - {instruction_end}s
INSTRUCTION TYPE (auto-detected): {instruction_type}

STUDENT BEHAVIOR: {behavior_label_zh} ({behavior_code})
VISUAL CONFIDENCE: {visual_confidence}
BEHAVIOR TIME: {behavior_time}s

TIME OVERLAP: The student behavior occurs within the instruction's effective window.

Evaluate whether the teacher's instruction supports, contradicts, or is unrelated to this student's visually detected behavior. Output JSON only."""


# ============================================================
# Behavior taxonomy for LLM context
# ============================================================

BEHAVIOR_DESCRIPTIONS = {
    "tt": "抬头听课 (head-up listening) — student looking at teacher/board",
    "dx": "低头写字 (writing) — student writing or taking notes",
    "dk": "低头看书 (reading) — student reading a book or material",
    "zt": "转头 (turning head) — student turning head, possibly distracted or looking at neighbor",
    "xt": "小组讨论 (group discussion) — student engaged in group discussion",
    "js": "举手 (raising hand) — student raising hand to answer or ask",
    "zl": "站立 (standing) — student standing up",
    "jz": "教师互动 (teacher interaction) — student interacting with teacher",
}


def _format_llm_prompt(
    instruction: Dict[str, Any],
    action: Dict[str, Any],
) -> str:
    """Format a single fusion prompt for the LLM."""
    inst_type = instruction.get("event_type", "unknown")
    inst_label = instruction.get("label", inst_type)
    behavior_code = action.get("behavior_code", action.get("action", "unknown"))
    behavior_label = action.get("behavior_label_zh", BEHAVIOR_DESCRIPTIONS.get(behavior_code, behavior_code))

    return USER_PROMPT_TEMPLATE.format(
        instruction_text=instruction.get("trigger_text", ""),
        instruction_start=instruction.get("start_time", 0),
        instruction_end=instruction.get("end_time", 0),
        instruction_type=f"{inst_type} ({inst_label})",
        behavior_label_zh=behavior_label,
        behavior_code=behavior_code,
        visual_confidence=action.get("conf", action.get("confidence", 0.5)),
        behavior_time=action.get("t", action.get("start_time", 0)),
    )


def build_fusion_batch(
    instruction_context_path: Path,
    actions_path: Optional[Path] = None,
    max_cases: int = 30,
) -> List[Dict[str, Any]]:
    """Build a batch of LLM fusion prompts from instruction context and actions."""
    ctx = json.loads(instruction_context_path.read_text(encoding="utf-8"))
    instructions = ctx.get("instruction_events", [])

    actions = []
    if actions_path and actions_path.exists():
        if actions_path.suffix == ".jsonl":
            with actions_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        actions.append(json.loads(line))
                    except Exception:
                        continue
        elif actions_path.suffix == ".json":
            data = json.loads(actions_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                actions = data
            elif isinstance(data, dict):
                actions = data.get("actions_with_context", data.get("actions", []))

    batch = []
    for inst in instructions[:max_cases]:
        inst_start = float(inst.get("start_time", 0))
        inst_end = float(inst.get("end_time", inst_start + 8.0))

        matched_actions = []
        for act in actions:
            act_time = float(act.get("t", act.get("start_time", 0)))
            if inst_start <= act_time <= inst_end:
                matched_actions.append(act)

        if not matched_actions:
            # Include at least the instruction with no matched actions (edge case)
            batch.append({
                "instruction": inst,
                "matched_actions": [],
                "prompt": _format_llm_prompt(inst, {
                    "behavior_code": "none",
                    "behavior_label_zh": "无匹配行为",
                    "conf": 0.0,
                    "t": inst_start,
                }),
            })
            continue

        for act in matched_actions[:3]:  # Max 3 actions per instruction
            batch.append({
                "instruction": inst,
                "action": act,
                "prompt": _format_llm_prompt(inst, act),
            })

    return batch


def simulate_llm_response(prompt: str) -> Dict[str, Any]:
    """Simulate LLM response for testing without API access.

    Uses simple rule-based heuristics that mimic what an LLM would reason.
    In production, replace with actual LLM API call.
    """
    # Simple heuristics for demonstration
    boost = 0.0
    reasoning = "No clear semantic link detected."

    prompt_lower = prompt.lower()

    # Support cases: instruction suggests behavior
    if any(w in prompt_lower for w in ["翻开", "打开", "看书", "看第", "阅读", "read", "朗读"]) and \
       any(w in prompt_lower for w in ['"dk"', '"read"', "看书", "阅读"]):
        boost = 0.12
        reasoning = "Teacher instruction ('open book'/'read') supports reading behavior."
    elif any(w in prompt_lower for w in ["写", "画", "记笔记", "write", "draw"]) and \
         any(w in prompt_lower for w in ['"dx"', '"write"', "写字"]):
        boost = 0.12
        reasoning = "Teacher instruction ('write'/'draw') supports writing behavior."
    elif any(w in prompt_lower for w in ["举手", "回答", "来说", "谁说", "派代表"]) and \
         any(w in prompt_lower for w in ['"js"', '"zl"', '"stand"', "举手", "站立"]):
        boost = 0.15
        reasoning = "Teacher instruction ('raise hand'/'who wants to answer') supports raising-hand/standing."
    elif any(w in prompt_lower for w in ["讨论", "交流", "小组", "商量"]) and \
         any(w in prompt_lower for w in ['"xt"', "讨论", "group"]):
        boost = 0.15
        reasoning = "Teacher instruction ('discuss') supports group discussion behavior."
    elif any(w in prompt_lower for w in ["看黑板", "听", "看前面", "坐好", "安静"]) and \
         any(w in prompt_lower for w in ['"tt"', '"listen"', "听课"]):
        boost = 0.10
        reasoning = "Teacher instruction ('look at board'/'listen'/'quiet') supports listening behavior."

    # Contradiction cases
    elif any(w in prompt_lower for w in ["停笔", "不要写", "把手放", "别写"]) and \
         any(w in prompt_lower for w in ['"dx"', '"write"', "写字"]):
        boost = -0.15
        reasoning = "Teacher instruction ('stop writing') contradicts writing behavior."
    elif any(w in prompt_lower for w in ["派代表", "介绍", "上来", "打招呼"]) and \
         any(w in prompt_lower for w in ['"dx"', '"write"', "写字"]):
        boost = -0.08
        reasoning = "Teacher is organizing presentations — writing behavior is less likely during this phase."
    elif any(w in prompt_lower for w in ["派代表", "介绍"]) and \
         any(w in prompt_lower for w in ['"tt"', '"listen"', "听课"]):
        boost = 0.10
        reasoning = "Teacher is organizing presentations — listening behavior is expected."

    return {
        "boost": round(boost, 4),
        "reasoning": reasoning,
        "confidence": 0.75 if abs(boost) > 0.05 else 0.5,
        "_simulated": True,
    }


def call_llm(prompt: str, model: str = "simulate", api_key: str = "") -> Dict[str, Any]:
    """Call LLM API or simulation.

    Args:
        prompt: The formatted user prompt
        model: "simulate", "openai:gpt-4o", "openai:gpt-4o-mini", "claude:sonnet", etc.
        api_key: API key (if using real LLM)
    """
    if model == "simulate" or not api_key:
        return simulate_llm_response(prompt)

    # Real LLM call placeholder
    # Example with OpenAI:
    # import openai
    # client = openai.OpenAI(api_key=api_key)
    # response = client.chat.completions.create(
    #     model=model.split(":")[-1],
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": prompt},
    #     ],
    #     response_format={"type": "json_object"},
    #     temperature=0.1,
    # )
    # return json.loads(response.choices[0].message.content)

    raise NotImplementedError(
        f"Real LLM call not implemented for model={model}. "
        "Use model='simulate' for rule-based testing, or provide api_key for real API."
    )


def main():
    parser = argparse.ArgumentParser(description="LLM Semantic Fusion for audio-visual classroom analysis.")
    parser.add_argument("--instruction_context", required=True, help="instruction_context.json from 06e")
    parser.add_argument("--actions", default="", help="actions JSONL/JSON for matched behaviors")
    parser.add_argument("--out", required=True, help="output llm_fusion_results.json")
    parser.add_argument("--model", default="simulate", help="LLM model: simulate, openai:gpt-4o, claude:sonnet, etc.")
    parser.add_argument("--api_key", default="", help="API key for real LLM call")
    parser.add_argument("--max_cases", type=int, default=30)
    args = parser.parse_args()

    ic_path = Path(args.instruction_context)
    actions_path = Path(args.actions) if args.actions else None
    out_path = Path(args.out)

    if not ic_path.is_absolute():
        ic_path = Path(__file__).resolve().parents[2] / ic_path
    if actions_path and not actions_path.is_absolute():
        actions_path = Path(__file__).resolve().parents[2] / actions_path
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[2] / out_path

    batch = build_fusion_batch(ic_path, actions_path, max_cases=int(args.max_cases))

    results = []
    for item in batch:
        response = call_llm(item["prompt"], model=args.model, api_key=args.api_key)
        results.append({
            "instruction_type": item["instruction"].get("event_type", "?"),
            "instruction_text": item["instruction"].get("trigger_text", ""),
            "instruction_time": item["instruction"].get("start_time", 0),
            "behavior_code": item.get("action", {}).get("behavior_code", item.get("action", {}).get("action", "none")),
            "behavior_conf": item.get("action", {}).get("conf", 0),
            "llm_boost": response["boost"],
            "llm_reasoning": response["reasoning"],
            "llm_confidence": response["confidence"],
            "prompt": item["prompt"][:200],
        })

    output = {
        "model": args.model,
        "total_cases": len(batch),
        "positive_boosts": sum(1 for r in results if r["llm_boost"] > 0.01),
        "negative_boosts": sum(1 for r in results if r["llm_boost"] < -0.01),
        "neutral": sum(1 for r in results if abs(r["llm_boost"]) <= 0.01),
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] LLM fusion results: {out_path}")
    print(f"[INFO] Cases: {len(batch)}, +boost={output['positive_boosts']}, -boost={output['negative_boosts']}, neutral={output['neutral']}")


if __name__ == "__main__":
    main()
