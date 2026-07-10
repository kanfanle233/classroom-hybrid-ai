import json
from pathlib import Path
from typing import Any, Dict, List

from classroom_pipeline.contracts.schemas import write_jsonl
from classroom_pipeline.verifier.model import action_match_score


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


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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
                rows.append(obj)
    return rows


def _normalize_aligned(aligned_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(aligned_obj, list):
        return [x for x in aligned_obj if isinstance(x, dict)]
    return []


def build_training_samples(
    *,
    event_queries_path: Path,
    aligned_path: Path,
    actions_path: Path,
    max_semantic_neg_per_query: int = 2,
) -> List[Dict[str, Any]]:
    queries = _load_jsonl(event_queries_path)
    query_by_id = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}
    aligned = _normalize_aligned(_load_json(aligned_path))
    actions = _load_jsonl(actions_path)

    samples: List[Dict[str, Any]] = []
    sid = 0

    for row in aligned:
        query_id = str(row.get("query_id", row.get("event_id", "")))
        q = query_by_id.get(query_id, {})
        event_type = str(q.get("event_type", row.get("event_type", "unknown")))
        query_text = str(q.get("query_text", row.get("query_text", "")))
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        if not candidates:
            continue

        # Score candidates by semantic*overlap to pick one positive anchor.
        ranked = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            action = str(c.get("semantic_id", c.get("action", "")))
            overlap = _safe_float(c.get("overlap", 0.0), 0.0)
            conf = _safe_float(c.get("action_confidence", c.get("confidence", c.get("conf", 0.0))), 0.0)
            uq = _safe_float(c.get("uq_score", c.get("uq_track", 0.5)), 0.5)
            text_s = action_match_score(event_type, query_text, action)
            ranked.append((text_s * (0.5 + 0.5 * overlap), text_s, overlap, conf, uq, c))
        ranked.sort(key=lambda x: x[0], reverse=True)
        best = ranked[0]
        _, text_s, overlap, conf, uq, cand = best
        track_id = int(cand.get("track_id", -1))
        action = str(cand.get("action", ""))

        # Positive sample
        samples.append(
            {
                "sample_id": f"s_{sid:07d}",
                "query_id": query_id,
                "event_type": event_type,
                "query_text": query_text,
                "track_id": track_id,
                "action_label": action,
                "overlap": _clamp01(overlap),
                "action_confidence": _clamp01(conf),
                "uq_score": _clamp01(uq),
                "text_score": _clamp01(text_s),
                "sample_type": "positive",
                "target": 1,
            }
        )
        sid += 1

        # Temporal negative: same action but shifted/weak overlap.
        temporal_overlap = _clamp01(max(0.0, overlap - 0.6))
        samples.append(
            {
                "sample_id": f"s_{sid:07d}",
                "query_id": query_id,
                "event_type": event_type,
                "query_text": query_text,
                "track_id": track_id,
                "action_label": action,
                "overlap": temporal_overlap,
                "action_confidence": _clamp01(conf),
                "uq_score": _clamp01(min(1.0, uq + 0.15)),
                "text_score": _clamp01(text_s),
                "sample_type": "temporal_shift",
                "target": 0,
            }
        )
        sid += 1

        # Semantic negatives: keep overlap high but mismatch query semantics.
        sem_neg_count = 0
        for _, neg_text_s, neg_overlap, neg_conf, neg_uq, neg_c in ranked[1:]:
            if sem_neg_count >= max_semantic_neg_per_query:
                break
            if neg_text_s > 0.45:
                continue
            samples.append(
                {
                    "sample_id": f"s_{sid:07d}",
                    "query_id": query_id,
                    "event_type": event_type,
                    "query_text": query_text,
                    "track_id": int(neg_c.get("track_id", -1)),
                    "action_label": str(neg_c.get("action", "")),
                    "overlap": _clamp01(neg_overlap),
                    "action_confidence": _clamp01(neg_conf),
                    "uq_score": _clamp01(neg_uq),
                    "text_score": _clamp01(neg_text_s),
                    "sample_type": "semantic_mismatch",
                    "target": 0,
                }
            )
            sid += 1
            sem_neg_count += 1

    # Fallback when aligned is empty: weak pseudo samples from actions.
    if not samples:
        for q in queries[:50]:
            qid = str(q.get("query_id", q.get("event_id", "")))
            et = str(q.get("event_type", "unknown"))
            qt = str(q.get("query_text", ""))
            for a in actions[:100]:
                tid = a.get("track_id")
                if not isinstance(tid, int):
                    continue
                action = str(a.get("action", ""))
                conf = _safe_float(a.get("conf", a.get("confidence", 0.0)), 0.0)
                text_s = action_match_score(et, qt, action)
                target = 1 if text_s >= 0.75 else 0
                stype = "positive" if target == 1 else "semantic_mismatch"
                samples.append(
                    {
                        "sample_id": f"s_{sid:07d}",
                        "query_id": qid,
                        "event_type": et,
                        "query_text": qt,
                        "track_id": int(tid),
                        "action_label": action,
                        "overlap": 0.5 if target == 1 else 0.2,
                        "action_confidence": _clamp01(conf),
                        "uq_score": 0.5,
                        "text_score": _clamp01(text_s),
                        "sample_type": stype,
                        "target": int(target),
                    }
                )
                sid += 1

    return samples


def convert_to_contract_samples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        event_id = str(row.get("event_id", row.get("query_id", "")))
        target = int(row.get("target", 0))
        sample_type = str(row.get("sample_type", "semantic_mismatch"))
        converted.append(
            {
                "sample_id": str(row.get("sample_id", f"s_{index:07d}")),
                "event_id": event_id,
                "sample_type": sample_type,
                "query_text": str(row.get("query_text", "")),
                "event_type": str(row.get("event_type", "unknown")),
                "track_id": int(row.get("track_id", -1)),
                "clip_start": float(row.get("clip_start", row.get("window_start", 0.0))),
                "clip_end": float(row.get("clip_end", row.get("window_end", row.get("clip_start", 0.0)))),
                "target_label": "match" if target == 1 else "mismatch",
                "negative_kind": "" if sample_type == "positive" else sample_type,
                "provenance": {
                    "source": "verifier.dataset",
                    "legacy_fields": {
                        "overlap": float(row.get("overlap", 0.0)),
                        "action_confidence": float(row.get("action_confidence", 0.0)),
                        "uq_score": float(row.get("uq_score", 0.0)),
                        "text_score": float(row.get("text_score", 0.0)),
                    },
                },
            }
        )
    return converted


def save_training_samples(path: Path, rows: List[Dict[str, Any]]) -> None:
    write_jsonl(path, rows)

