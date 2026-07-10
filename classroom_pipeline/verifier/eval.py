import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import ARTIFACT_VERSION, validate_verifier_eval_report, write_json
from classroom_pipeline.verifier.metrics import (
    LABELS,
    build_matrix_list,
    confusion_matrix,
    label_distribution,
    metrics_from_confusion_matrix,
    parse_predicted_label,
    parse_reference_label,
    parse_score,
    pick_best_sweep,
    probability_consistency,
    threshold_sweep,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _build_threshold_values(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        step = 0.05
    if end < start:
        start, end = end, start
    values: List[float] = []
    cur = float(start)
    while cur <= end + 1e-9:
        values.append(round(cur, 4))
        cur += step
    if not values:
        values = [0.60]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate verified_events.jsonl and export verifier_eval_report.json")
    parser.add_argument("--verified", required=True, type=str, help="verified_events.jsonl")
    parser.add_argument("--out", required=True, type=str, help="verifier_eval_report.json")
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--target_field", default="auto", type=str, help="ground-truth label field or auto")
    parser.add_argument("--score_field", default="reliability_score", type=str)
    parser.add_argument("--uncertain_margin", default=0.20, type=float)
    parser.add_argument("--sweep_start", default=0.30, type=float)
    parser.add_argument("--sweep_end", default=0.90, type=float)
    parser.add_argument("--sweep_step", default=0.05, type=float)
    args = parser.parse_args()

    verified_path = Path(args.verified).resolve()
    out_path = Path(args.out).resolve()
    rows = _load_jsonl(verified_path)

    y_true: List[str] = []
    y_pred: List[str] = []
    scores: List[float] = []
    ref_source_counts: Dict[str, int] = {}
    for row in rows:
        target_label, source = parse_reference_label(row, target_field=str(args.target_field))
        pred_label = parse_predicted_label(row)
        score = parse_score(row, score_field=str(args.score_field))
        y_true.append(target_label)
        y_pred.append(pred_label)
        scores.append(score)
        ref_source_counts[source] = ref_source_counts.get(source, 0) + 1

    cm = confusion_matrix(y_true, y_pred)
    metrics = metrics_from_confusion_matrix(cm)
    threshold_values = _build_threshold_values(float(args.sweep_start), float(args.sweep_end), float(args.sweep_step))
    sweep = threshold_sweep(
        y_true=y_true,
        scores=scores,
        thresholds=threshold_values,
        uncertain_margin=float(args.uncertain_margin),
    )
    best = pick_best_sweep(sweep, key="f1")
    prob_check = probability_consistency(rows)

    report: Dict[str, Any] = {
        "split": str(args.split),
        "counts": {
            "total": len(rows),
            "reference_source": ref_source_counts,
            "probability_consistency": prob_check,
        },
        "metrics": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_label": metrics["per_label"],
        },
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": build_matrix_list(cm),
            "as_dict": cm,
        },
        "threshold_sweep": sweep,
        "label_distribution": {
            "reference": label_distribution(y_true),
            "predicted": label_distribution(y_pred),
        },
        "config": {
            "score_field": str(args.score_field),
            "target_field": str(args.target_field),
            "uncertain_margin": float(args.uncertain_margin),
            "threshold_source": "val_f1_best",
            "best_threshold": {
                "match_threshold": float(best.get("match_threshold", 0.60)),
                "uncertain_threshold": float(best.get("uncertain_threshold", 0.40)),
                "f1": float(best.get("f1", 0.0)),
            },
            "source_file": str(verified_path),
        },
        "artifact_version": ARTIFACT_VERSION,
    }

    ok, msg = validate_verifier_eval_report(report)
    if not ok:
        raise ValueError(f"invalid eval report payload: {msg}")
    write_json(out_path, report)
    print(f"[DONE] eval report: {out_path}")
    print(f"[INFO] total={len(rows)} f1={report['metrics']['f1']:.4f} e_prob={prob_check['mean_abs_error']:.4f}")


if __name__ == "__main__":
    main()
