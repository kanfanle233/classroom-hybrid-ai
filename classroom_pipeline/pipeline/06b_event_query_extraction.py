import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, validate_event_query_record, validate_jsonl_file


EventPattern = Tuple[str, str, Sequence[str]]


# Keep keyword rules lightweight and auditable. Unknown segments still become queries.
EVENT_PATTERNS: List[EventPattern] = [
    ("raise_hand", "raise hand", ("raise hand", "hand up", "举手")),
    ("head_down", "head down", ("head down", "look down", "低头", "sleep", "doze")),
    ("discussion", "group discussion", ("discussion", "group work", "小组讨论")),
    ("respond_call", "respond to teacher call", ("answer", "who can", "点名", "回答")),
    ("teacher_instruction", "teacher instruction", ("listen", "be quiet", "stop", "注意", "安静")),
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_text(row: Dict[str, Any]) -> str:
    for key in ("text", "sentence", "utterance", "content"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _pick_time(row: Dict[str, Any]) -> Tuple[float, float]:
    start = _safe_float(row.get("start", row.get("start_time", row.get("t", 0.0))), 0.0)
    end = _safe_float(row.get("end", row.get("end_time", start)), start)
    if end < start:
        start, end = end, start
    if end <= start:
        end = start + 0.4
    return start, end


def _load_transcript(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            text = _safe_text(row)
            if not text:
                continue
            start, end = _pick_time(row)
            rows.append({"start": start, "end": end, "text": text})
    rows.sort(key=lambda r: (r["start"], r["end"]))
    return rows


def _make_query(
    *,
    event_id: str,
    event_type: str,
    query_text: str,
    trigger_text: str,
    trigger_words: Sequence[str],
    start: float,
    end: float,
    confidence: float,
) -> Dict[str, Any]:
    t_center = 0.5 * (start + end)
    return {
        "event_id": event_id,
        "query_id": event_id,
        "schema_version": SCHEMA_VERSION,
        "event_type": event_type,
        "query_text": query_text,
        "trigger_text": trigger_text,
        "trigger_words": list(trigger_words),
        "timestamp": round(t_center, 3),
        "t_center": round(t_center, 3),
        "start": round(start, 3),
        "end": round(end, 3),
        "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
        "source": "asr",
    }


def _extract_events(segment: Dict[str, Any], q_index: int) -> List[Dict[str, Any]]:
    text = str(segment["text"])
    text_norm = _normalize_text(text)
    start = float(segment["start"])
    end = float(segment["end"])
    out: List[Dict[str, Any]] = []

    for event_type, query_text, keywords in EVENT_PATTERNS:
        matched = [kw for kw in keywords if kw.lower() in text_norm]
        if not matched:
            continue
        confidence = min(0.95, 0.55 + 0.1 * len(matched))
        out.append(
            _make_query(
                event_id=f"e_{q_index:06d}_{len(out):02d}",
                event_type=event_type,
                query_text=query_text,
                trigger_text=text,
                trigger_words=matched,
                start=start,
                end=end,
                confidence=confidence,
            )
        )

    # No keyword match: still produce a generic query so downstream align/verifier
    # always receives temporally grounded supervision candidates.
    if not out:
        normalized = _normalize_text(text)
        preview = normalized[:48] if normalized else "asr segment"
        out.append(
            _make_query(
                event_id=f"e_{q_index:06d}_00",
                event_type="unknown",
                query_text=preview,
                trigger_text=text,
                trigger_words=[],
                start=start,
                end=end,
                confidence=0.25,
            )
        )

    return out


def _placeholder_query() -> Dict[str, Any]:
    return _make_query(
        event_id="e_000000_00",
        event_type="unknown",
        query_text="unknown event",
        trigger_text="[ASR_EMPTY]",
        trigger_words=[],
        start=0.0,
        end=0.5,
        confidence=0.1,
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Extract structured event queries from transcript.")
    parser.add_argument("--transcript", required=True, type=str, help="transcript.jsonl")
    parser.add_argument("--out", required=True, type=str, help="event_queries.jsonl")
    parser.add_argument("--validate", type=int, default=1)
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    out_path = Path(args.out)
    if not transcript_path.is_absolute():
        transcript_path = (base_dir / transcript_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    segments = _load_transcript(transcript_path)
    queries: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        queries.extend(_extract_events(seg, idx))
    if not queries:
        queries.append(_placeholder_query())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    if int(args.validate) == 1:
        ok, _, errors = validate_jsonl_file(out_path, validate_event_query_record)
        if not ok:
            first_error = errors[0] if errors else "unknown schema error"
            raise ValueError(f"invalid event query schema: {first_error}")

    print(f"[DONE] event queries exported: {out_path}")
    print(f"[INFO] total event queries: {len(queries)}")


if __name__ == "__main__":
    main()
