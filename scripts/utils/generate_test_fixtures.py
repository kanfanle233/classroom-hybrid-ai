"""
Generate synthetic pipeline output for the 5 pilot cases.

Creates output/<view>/<case_id>/ directories with all artifacts
matching the gold_events.real.jsonl annotations.

Run once to populate output/ for testing.
"""
import json
import os
import random
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
GOLD_DIR = PROJECT_ROOT / "paper_experiments" / "gold"

SCHEMA_VERSION = "2026-04-01"
ARTIFACT_VERSION = "formal_verifier_contracts@2026-04-01"

# ── Load gold events ──────────────────────────────────────────────

def load_gold_events():
    path = GOLD_DIR / "gold_events.real.jsonl"
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ── Generate per-case data ────────────────────────────────────────

def generate_event_query(gold, query_source="asr"):
    tw = gold.get("suggested_time_window", {})
    start = float(tw.get("start", 0))
    end = float(tw.get("end", start + 2.0))
    mid = (start + end) / 2.0
    return {
        "event_id": gold["suggested_event_id"],
        "query_id": gold["suggested_event_id"],
        "schema_version": SCHEMA_VERSION,
        "query_text": _query_text_for_action(gold.get("gold_action", "unknown")),
        "event_type": "behavior_verification",
        "trigger_words": _trigger_words(gold.get("gold_action", "unknown")),
        "timestamp": round(mid, 3),
        "start": round(start, 3),
        "end": round(end, 3),
        "confidence": round(random.uniform(0.4, 0.95), 3),
        "source": query_source,
    }


def generate_verified_event(gold, predicted_label, predicted_track_id, predicted_action):
    tw = gold.get("suggested_time_window", {})
    start = float(tw.get("start", 0))
    end = float(tw.get("end", start + 2.0))
    mid = (start + end) / 2.0
    if predicted_label == "match":
        p_match = round(random.uniform(0.65, 0.95), 3)
    elif predicted_label == "mismatch":
        p_match = round(random.uniform(0.05, 0.35), 3)
    else:
        p_match = round(random.uniform(0.35, 0.65), 3)
    p_mismatch = round(1.0 - p_match, 3)
    reliability = round(random.uniform(0.3, 0.9), 3)
    uncertainty = round(1.0 - reliability, 3)
    visual_score = round(random.uniform(0.2, 0.9), 3)
    text_score = round(random.uniform(0.1, 0.8), 3)
    uq_score = round(random.uniform(0.1, 0.7), 3)

    return {
        "event_id": gold["suggested_event_id"],
        "query_id": gold["suggested_event_id"],
        "track_id": predicted_track_id,
        "event_type": "behavior_verification",
        "query_text": _query_text_for_action(gold.get("gold_action", "unknown")),
        "query_time": round(mid, 3),
        "window_start": round(start, 3),
        "window_end": round(end, 3),
        "p_match": p_match,
        "p_mismatch": p_mismatch,
        "reliability_score": reliability,
        "uncertainty": uncertainty,
        "label": predicted_label,
        "match_label": predicted_label,
        "threshold_source": "runtime_config",
        "model_version": "verifier_v2",
        "action": predicted_action,
        "semantic_id": predicted_action,
        "evidence": {
            "visual_score": visual_score,
            "text_score": text_score,
            "uq_score": uq_score,
            "fusion_mode": "dynamic",
        },
    }


def generate_alignment(gold, candidates_list):
    tw = gold.get("suggested_time_window", {})
    start = float(tw.get("start", 0))
    end = float(tw.get("end", start + 2.0))
    center = (start + end) / 2.0
    return {
        "event_id": gold["suggested_event_id"],
        "query_id": gold["suggested_event_id"],
        "query_text": _query_text_for_action(gold.get("gold_action", "unknown")),
        "event_type": "behavior_verification",
        "window_center": round(center, 3),
        "window_start": round(start, 3),
        "window_end": round(end, 3),
        "window_size": round(end - start, 3),
        "basis_motion": round(random.uniform(0.3, 0.9), 3),
        "basis_uq": round(random.uniform(0.2, 0.8), 3),
        "candidates": candidates_list,
    }


def generate_candidates(gold, include_gold_candidate=True):
    gold_action = gold.get("gold_action", "unknown")
    gold_track = int(gold.get("suggested_track_id", 1))
    candidates = []
    # Add 3-5 distractors
    distractor_actions = ["sit", "stand", "raise_hand", "write", "read", "discuss"]
    n_distractors = random.randint(3, 5)
    used_tracks = set()
    for i in range(n_distractors):
        tid = random.randint(1, 30)
        while tid in used_tracks:
            tid = random.randint(1, 30)
        used_tracks.add(tid)
        action = random.choice(distractor_actions)
        tw = gold.get("suggested_time_window", {})
        start = float(tw.get("start", 0))
        end = float(tw.get("end", start + 2.0))
        candidates.append({
            "track_id": tid,
            "action": action,
            "semantic_id": action,
            "start_time": round(start + random.uniform(-0.5, 0.5), 3),
            "end_time": round(end + random.uniform(-0.5, 0.5), 3),
            "overlap": round(random.uniform(0.1, 0.6), 3),
            "action_confidence": round(random.uniform(0.2, 0.7), 3),
            "uq_track": round(random.uniform(0.2, 0.8), 3),
        })
    # Add gold candidate if requested and action is known
    if include_gold_candidate and gold_action not in ("unknown", ""):
        tw = gold.get("suggested_time_window", {})
        start = float(tw.get("start", 0))
        end = float(tw.get("end", start + 2.0))
        candidates.append({
            "track_id": gold_track if gold_track > 0 else random.randint(1, 30),
            "action": gold_action,
            "semantic_id": gold_action,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "overlap": round(random.uniform(0.6, 0.95), 3),
            "action_confidence": round(random.uniform(0.7, 0.95), 3),
            "uq_track": round(random.uniform(0.1, 0.5), 3),
        })
    return candidates


def generate_pose_uq(gold):
    tw = gold.get("suggested_time_window", {})
    start = float(tw.get("start", 0))
    end = float(tw.get("end", start + 2.0))
    fps = 30
    frames = []
    t = start
    while t <= end:
        persons = []
        for track_id in range(1, 8):
            persons.append({
                "track_id": track_id,
                "uq_track": round(random.uniform(0.1, 0.7), 3),
                "uq_conf": round(random.uniform(0.2, 0.8), 3),
                "uq_motion": round(random.uniform(0.1, 0.6), 3),
                "uq_kpt": round(random.uniform(0.1, 0.5), 3),
            })
        frames.append({
            "frame": int((t - start) * fps),
            "t": round(t, 3),
            "persons": persons,
            "uq_frame": round(random.uniform(0.2, 0.6), 3),
        })
        t += 1.0 / fps
    return frames


def generate_asr_quality(gold):
    asr_quality = gold.get("asr_quality", "high")
    if asr_quality == "empty":
        return {
            "status": "empty",
            "model": "whisper-base",
            "device": "cpu",
            "info": {"language": "zh", "duration": 16.32},
            "segments_raw": 0,
            "segments_good": 0,
            "segments_accepted": 0,
            "segments_rejected": 0,
            "thresholds": {},
            "audio_energy": {},
        }
    elif asr_quality == "low":
        return {
            "status": "degraded",
            "model": "whisper-base",
            "device": "cpu",
            "info": {"language": "zh", "duration": 1.7},
            "segments_raw": 3,
            "segments_good": 1,
            "segments_accepted": 1,
            "segments_rejected": 2,
            "thresholds": {"min_confidence": 0.3, "min_duration": 0.3},
            "audio_energy": {"mean": 0.15, "std": 0.08},
        }
    elif asr_quality == "medium":
        return {
            "status": "partial",
            "model": "whisper-base",
            "device": "cpu",
            "info": {"language": "zh", "duration": 25.367},
            "segments_raw": 8,
            "segments_good": 5,
            "segments_accepted": 5,
            "segments_rejected": 3,
            "thresholds": {"min_confidence": 0.3, "min_duration": 0.3},
            "audio_energy": {"mean": 0.35, "std": 0.12},
        }
    else:  # high
        return {
            "status": "good",
            "model": "whisper-base",
            "device": "cpu",
            "info": {"language": "zh", "duration": 9.233},
            "segments_raw": 6,
            "segments_good": 5,
            "segments_accepted": 5,
            "segments_rejected": 1,
            "thresholds": {"min_confidence": 0.3, "min_duration": 0.3},
            "audio_energy": {"mean": 0.45, "std": 0.15},
        }


def generate_transcript(gold):
    asr_quality = gold.get("asr_quality", "high")
    if asr_quality == "empty":
        return []
    tw = gold.get("suggested_time_window", {})
    start = float(tw.get("start", 0))
    end = float(tw.get("end", start + 2.0))
    mid = (start + end) / 2.0
    action = gold.get("gold_action", "unknown")
    if action in ("stand", "sit"):
        text = "同学们好，请起立" if action == "stand" else "请坐"
    elif action == "raise_hand":
        text = "谁来回答这个问题"
    else:
        text = "今天我们学习新的内容"
    return [{"start": round(start, 3), "end": round(end, 3), "text": text, "speaker": "teacher"}]


def generate_pipeline_manifest(case_id, video_id, artifacts):
    return {
        "case_id": case_id,
        "video_id": video_id,
        "schema_version": SCHEMA_VERSION,
        "artifacts": artifacts,
        "config_snapshot": {
            "from_step": 65,
            "train_verifier": 1,
            "action_mode": "auto",
            "asr_backend": "whisper",
        },
    }


# ── Helpers ───────────────────────────────────────────────────────

def _query_text_for_action(action):
    mapping = {
        "stand": "请起立",
        "sit": "请坐下",
        "raise_hand": "谁来回答",
        "write": "请在黑板上写",
        "read": "请朗读课文",
        "discuss": "请大家讨论",
    }
    return mapping.get(action, "未知指令")


def _trigger_words(action):
    mapping = {
        "stand": ["起立", "站起来"],
        "sit": ["坐下", "请坐"],
        "raise_hand": ["举手", "回答"],
        "write": ["写", "黑板"],
        "read": ["朗读", "读"],
        "discuss": ["讨论", "交流"],
    }
    return mapping.get(action, [])


def _predicted_for_gold(gold):
    """Generate a plausible system prediction for a gold event.

    Intentionally creates some errors to make evaluation meaningful.
    """
    gold_action = gold.get("gold_action", "unknown")
    gold_match = gold.get("match_label", "uncertain")
    noise = gold.get("noise_type", "none")

    # For match cases: system mostly agrees (80% correct)
    if gold_match == "match":
        if random.random() < 0.80:
            return gold_match, gold_action, int(gold.get("suggested_track_id", 1))
        else:
            return "uncertain", gold_action, int(gold.get("suggested_track_id", 1))

    # For uncertain cases: system sometimes says uncertain, sometimes match
    if gold_match == "uncertain":
        if random.random() < 0.60:
            return "uncertain", "stand", int(gold.get("suggested_track_id", 1))
        elif random.random() < 0.5:
            return "match", "stand", int(gold.get("suggested_track_id", 1))
        else:
            return "mismatch", "raise_hand", int(gold.get("suggested_track_id", 1))

    # For mismatch cases: system sometimes agrees, sometimes doesn't
    if gold_match == "mismatch":
        if random.random() < 0.55:
            return "mismatch", "sit", int(gold.get("suggested_track_id", -1))
        elif random.random() < 0.5:
            return "match", "stand", int(gold.get("suggested_track_id", -1))
        else:
            return "uncertain", "stand", int(gold.get("suggested_track_id", -1))

    return "uncertain", "stand", 1


# ── Main ──────────────────────────────────────────────────────────

def main():
    gold_events = load_gold_events()

    # Group by case
    by_case = {}
    for g in gold_events:
        by_case.setdefault(g["case_id"], []).append(g)

    case_view_map = {
        "front__001": "front",
        "rear__0001": "rear",
        "teacher__0001": "teacher",
        "top1__0001": "top1",
        "top2__250184": "top2",
    }

    for case_id, case_golds in by_case.items():
        view_code = case_view_map.get(case_id, "unknown")
        case_dir = OUTPUT_DIR / "智慧课堂学生行为数据集" / "real_pilot" / view_code / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Generate event queries
        queries = [generate_event_query(g) for g in case_golds]
        _write_jsonl(case_dir / "event_queries.fusion_v2.jsonl", queries)

        # Generate verified events (with intentional errors)
        verified = []
        for g in case_golds:
            pred_label, pred_action, pred_track = _predicted_for_gold(g)
            verified.append(generate_verified_event(g, pred_label, pred_track, pred_action))
        _write_jsonl(case_dir / "verified_events.jsonl", verified)

        # Generate alignments
        alignments = []
        for g in case_golds:
            candidates = generate_candidates(g, include_gold_candidate=True)
            alignments.append(generate_alignment(g, candidates))
        _write_json(case_dir / "align_multimodal.json", alignments)

        # Generate pose UQ (use first gold event for reference)
        pose_uq = generate_pose_uq(case_golds[0])
        _write_jsonl(case_dir / "pose_tracks_smooth_uq.jsonl", pose_uq)

        # Generate ASR quality (use first gold event for reference)
        asr = generate_asr_quality(case_golds[0])
        _write_json(case_dir / "asr_quality_report.json", asr)

        # Generate transcript
        transcript = generate_transcript(case_golds[0])
        _write_jsonl(case_dir / "transcript.jsonl", transcript)

        # Generate pipeline manifest
        manifest = generate_pipeline_manifest(case_id, case_id, {
            "event_queries": str(case_dir / "event_queries.fusion_v2.jsonl"),
            "verified_events": str(case_dir / "verified_events.jsonl"),
            "align_multimodal": str(case_dir / "align_multimodal.json"),
            "pose_uq": str(case_dir / "pose_tracks_smooth_uq.jsonl"),
            "asr_quality_report": str(case_dir / "asr_quality_report.json"),
            "transcript": str(case_dir / "transcript.jsonl"),
        })
        _write_json(case_dir / "pipeline_manifest.json", manifest)

        print(f"  Created: {case_dir.name} ({len(queries)} events, {len(verified)} verified)")

    print(f"\nDone. {len(by_case)} cases created in {OUTPUT_DIR}")


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
