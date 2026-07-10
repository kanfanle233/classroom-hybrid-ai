import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import validate_verifier_sample_record, write_jsonl
from classroom_pipeline.verifier.dataset import build_training_samples, convert_to_contract_samples


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _validate_rows(rows: List[Dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        ok, msg = validate_verifier_sample_record(row)
        if not ok:
            raise ValueError(f"invalid training sample at idx={i}: {msg}")


def _assign_split(rows: List[Dict[str, Any]], train_ratio: float) -> None:
    """Assign a deterministic train/val split without changing sample order."""
    ratio = max(0.0, min(1.0, float(train_ratio)))
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        bucket = sum((index + 1) * ord(char) for index, char in enumerate(sample_id)) % 10_000
        row["split"] = "train" if bucket < int(ratio * 10_000) else "val"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verifier training samples for exp-c negative sampling.")
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--aligned", required=True, type=str)
    parser.add_argument("--actions", required=True, type=str)
    parser.add_argument("--out", required=True, type=str, help="train_samples.jsonl")
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--shift_1", default=1.0, type=float, help="temporal shift seconds")
    parser.add_argument("--shift_2", default=2.0, type=float, help="temporal shift seconds")
    args = parser.parse_args()

    event_queries = Path(args.event_queries).resolve()
    aligned = Path(args.aligned).resolve()
    actions = Path(args.actions).resolve()
    out = Path(args.out).resolve()

    raw_rows = build_training_samples(
        event_queries_path=event_queries,
        aligned_path=aligned,
        actions_path=actions,
    )
    rows = convert_to_contract_samples(raw_rows)
    if not rows:
        raise RuntimeError("no samples generated")

    _assign_split(rows, _safe_float(args.train_ratio, 0.8))
    _validate_rows(rows)
    write_jsonl(out, rows)

    if (float(args.shift_1), float(args.shift_2)) != (1.0, 2.0):
        print(
            "[WARN] --shift_1/--shift_2 are retained for CLI compatibility; "
            "the current contract generator derives temporal negatives from aligned overlap."
        )

    neg_counter = Counter(str(r.get("negative_kind", "positive")) or "positive" for r in rows)
    split_counter = Counter(str(r.get("split", "unknown")) for r in rows)
    print(f"[DONE] samples: {out}")
    print(f"[INFO] total={len(rows)} by_negative_type={dict(neg_counter)} by_split={dict(split_counter)}")


if __name__ == "__main__":
    main()
