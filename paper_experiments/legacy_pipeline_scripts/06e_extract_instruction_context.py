"""Extract teacher instruction events from ASR transcript for context-aware behavior enhancement.

Replaces the old "dual verification" text_score approach with a context-aware model:
- Teacher says "派代表/举手" → raise_hand/stand behavior confidence boost
- Teacher says "翻开课本" → read behavior confidence boost
- Teacher says "写下来" → write behavior confidence boost
- Teacher says "看黑板" → listen behavior confidence boost
- Teacher says "停笔/讨论" → write dampen / discuss boost

Reference: Stanford EDS late-stage fusion model (2025)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# Teacher instruction patterns → behavior adjustments
INSTRUCTION_PATTERNS: List[Dict[str, Any]] = [
    # [pattern, event_type, boost_target, boost_amount, dampen_target, dampen_amount]
    {"pattern": r"翻[开開].*(?:课[本體]|书|書)|打开.*(?:课[本體]|书|書)|看.*(?:课[本體]|书|書)第|看第.*[段節]",
     "event": "open_book", "boost": ["dk", "read"], "boost_amt": 0.12, "label": "翻开课本→阅读+12%"},
    {"pattern": r"写[下個來]|做笔记|记[下個來]|写[在個]|开始写|记笔记|画直[线線]|画[线線]|用直[纸紙尺]",
     "event": "start_write", "boost": ["dx", "write"], "boost_amt": 0.12, "label": "开始写/画线→写字+12%"},
    {"pattern": r"看黑板|看[前這]面|看屏幕|抬头|看这[里裡]|你[们們]看",
     "event": "look_board", "boost": ["tt", "listen"], "boost_amt": 0.10, "label": "看黑板→听课+10%"},
    {"pattern": r"谁[來来]?说|谁[來来]回答|举手|哪位同学|请.*回答|派.*代表|你[來来]说|.*说[一這]说",
     "event": "raise_hand", "boost": ["js", "raise_hand", "zl", "stand"], "boost_amt": 0.15, "label": "举手→举手/站立+15%"},
    {"pattern": r"一起读|朗读|读[一這]遍|跟[着著]读|读课文",
     "event": "read_aloud", "boost": ["dk", "read", "tt", "listen"], "boost_amt": 0.10, "label": "朗读→阅读+10%"},
    {"pattern": r"讨论|交流|同桌|小组|商量|组内|互相讨论",
     "event": "discuss", "boost": ["xt", "group_discussion", "zt", "turn_head"], "boost_amt": 0.15, "label": "讨论→讨论+15%"},
    {"pattern": r"停笔|停[下個來]|放下笔|不要写|别写|把手放[下個]",
     "event": "stop_action", "dampen": ["dx", "write", "js", "raise_hand"], "dampen_amt": 0.15, "label": "停止动作→写字/举手-15%"},
    {"pattern": r"介绍|做介绍|打招呼|说.*介绍|给大[家個]",
     "event": "introduce", "boost": ["listen", "tt", "zl", "stand"], "boost_amt": 0.10, "label": "介绍→听课/站立+10%"},
    {"pattern": r"坐好|安静|不要说话|别说话|先坐好",
     "event": "quiet_down", "boost": ["tt", "listen"], "dampen": ["xt", "group_discussion", "zt"], "boost_amt": 0.08, "dampen_amt": 0.10, "label": "安静→听课+8%/讨论-10%"},
    {"pattern": r"听到了吗|听清楚|听懂|听到了",
     "event": "check_listen", "boost": ["tt", "listen"], "boost_amt": 0.08, "label": "确认听讲→听课+8%"},
]


def _extract_instructions(segments: List[Dict[str, Any]], time_window: float = 8.0) -> List[Dict[str, Any]]:
    """Extract teacher instruction events from ASR segments."""
    instructions = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text or seg.get("is_placeholder"):
            continue
        quality = seg.get("asr_quality_status", "accepted")
        confidence = 1.0 if quality == "accepted" else 0.6

        for rule in INSTRUCTION_PATTERNS:
            if not re.search(rule["pattern"], text):
                continue
            inst = {
                "event_type": rule["event"],
                "label": rule["label"],
                "trigger_text": text,
                "start_time": seg.get("start", 0),
                "end_time": seg.get("end", seg.get("start", 0) + time_window),
                "window_start": seg.get("start", 0),
                "window_end": seg.get("start", 0) + time_window,
                "confidence": min(1.0, confidence),
                "source": "asr_instruction",
            }
            if "boost" in rule:
                inst["boost"] = {
                    "behaviors": rule["boost"],
                    "amount": rule["boost_amt"],
                }
            if "dampen" in rule:
                inst["dampen"] = {
                    "behaviors": rule["dampen"],
                    "amount": rule["dampen_amt"],
                }
            instructions.append(inst)
            break  # One event per segment
    return instructions


def _apply_context_boost(
    actions: List[Dict[str, Any]],
    instructions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply instruction context to adjust behavior confidence."""
    for action in actions:
        action_time = action.get("t", action.get("start_time", 0))
        behavior = str(action.get("behavior_code", action.get("action", ""))).strip()
        if not behavior:
            continue

        boost_total = 0.0
        triggers = []
        for inst in instructions:
            if inst["window_start"] <= action_time <= inst["window_end"]:
                if "boost" in inst and behavior in inst["boost"]["behaviors"]:
                    boost_total += inst["boost"]["amount"]
                    triggers.append(inst["event_type"])
                if "dampen" in inst and behavior in inst["dampen"]["behaviors"]:
                    boost_total -= inst["dampen"]["amount"]
                    triggers.append(f"-{inst['event_type']}")

        base_conf = float(action.get("conf", action.get("confidence", 0.5)))
        adjusted = max(0.0, min(1.0, base_conf + boost_total))
        action["audio_context_boost"] = round(boost_total, 4)
        action["audio_context_triggers"] = triggers
        action["confidence_adjusted"] = round(adjusted, 4)
        action["context_source"] = "audio_visual" if triggers else "visual_only"

    return actions


def main():
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Extract teacher instruction context from ASR transcript.")
    parser.add_argument("--transcript", required=True, help="transcript.jsonl from ASR step")
    parser.add_argument("--actions", default="", help="actions.jsonl to apply context boost to")
    parser.add_argument("--out", required=True, help="output instruction_context.jsonl")
    parser.add_argument("--time_window", type=float, default=8.0, help="window in seconds for context effect")
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    out_path = Path(args.out)
    if not transcript_path.is_absolute():
        transcript_path = (base_dir / transcript_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    # Read transcript
    segments = []
    if transcript_path.exists():
        with transcript_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    segments.append(json.loads(line))
                except Exception:
                    continue

    # Extract instructions
    instructions = _extract_instructions(segments, time_window=float(args.time_window))

    # Optionally apply to actions
    actions_out = []
    if args.actions:
        actions_path = Path(args.actions)
        if not actions_path.is_absolute():
            actions_path = (base_dir / actions_path).resolve()
        actions_in = []
        if actions_path.exists():
            with actions_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        actions_in.append(json.loads(line))
                    except Exception:
                        continue
        actions_out = _apply_context_boost(actions_in, instructions)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "source": "asr_instruction_context",
        "segments_total": len(segments),
        "instructions_extracted": len(instructions),
        "instruction_events": instructions,
        "actions_with_context": actions_out if args.actions else [],
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] instruction_context: {out_path}")
    print(f"[INFO] segments={len(segments)} instructions={len(instructions)}")
    for inst in instructions[:5]:
        print(f"  [{inst['start_time']:.0f}s] {inst['event_type']}: {inst['trigger_text'][:50]}")
    if actions_out:
        boosted = sum(1 for a in actions_out if a.get("context_source") == "audio_visual")
        print(f"[INFO] actions boosted: {boosted}/{len(actions_out)}")
    print(f"[INFO] time_window={args.time_window}s")


if __name__ == "__main__":
    main()
