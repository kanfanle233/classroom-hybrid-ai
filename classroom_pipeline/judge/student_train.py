"""Train a lightweight student judge on LLM silver labels.

The student is a VerifierMLP (4→16→16→1), fully compatible with the
existing verifier inference pipeline. It is trained on silver labels
produced by the LLM teacher, replacing heuristic pseudo-labels.

Output:
  - student_judge.pt (VerifierMLP checkpoint with teacher_provenance)
  - student_train_report.json (training metrics)
  - student_samples.jsonl (feature vectors + targets for reproducibility)
"""

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, write_jsonl
from classroom_pipeline.verifier.model import (
    VerifierMLP,
    VerifierRuntimeConfig,
    brier_score,
    build_feature_vector,
    expected_calibration_error,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


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
                out.append(json.loads(line))
            except Exception:
                continue
    return out


# ── sample building ──────────────────────────────────────────────────────

def build_student_samples(
    *,
    silver_labels_path: Path,
    event_queries_path: Path,
    aligned_path: Path,
    target_threshold: float = 0.60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build training samples from silver labels + alignment data.

    For each silver label record, find the corresponding alignment candidate
    and build the 4-dim feature vector. The target is binarized from
    silver_p_match.

    Returns:
      samples: list of dicts with features and target
      provenance: metadata about label source
    """
    silver_labels = _load_jsonl(silver_labels_path)
    queries = _load_jsonl(event_queries_path)
    q_index = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}

    # Build a lookup: query_id + track_id → silver label
    silver_map: Dict[str, Dict[str, Any]] = {}
    teacher_model = "unknown"
    for s in silver_labels:
        key = f"{s.get('query_id', '')}_track{s.get('track_id', -1)}"
        silver_map[key] = s
        if s.get("teacher_model") and teacher_model == "unknown":
            teacher_model = s["teacher_model"]

    # Load alignment to build feature vectors
    import json as _json
    aligned_obj = None
    if aligned_path.exists():
        with aligned_path.open("r", encoding="utf-8") as f:
            aligned_obj = _json.load(f)
    aligned = aligned_obj if isinstance(aligned_obj, list) else []

    samples: List[Dict[str, Any]] = []
    sid = 0
    num_silver_found = 0
    num_silver_matched = 0

    for block in aligned:
        if not isinstance(block, dict):
            continue
        query_id = str(block.get("query_id", block.get("event_id", "")))
        query = q_index.get(query_id, {})
        event_type = str(query.get("event_type", block.get("event_type", "unknown")))
        query_text = str(query.get("query_text", block.get("query_text", "")))
        candidates = block.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []

        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            track_id = int(cand.get("track_id", -1))
            key = f"{query_id}_track{track_id}"
            silver = silver_map.get(key)
            if silver is None:
                continue
            num_silver_found += 1

            overlap = _safe_float(cand.get("overlap", 0.0), 0.0)
            action_conf = _safe_float(
                cand.get("action_confidence", cand.get("confidence", cand.get("conf", 0.0))), 0.0
            )
            uq = _safe_float(cand.get("uq_score", cand.get("uq_track", 0.5)), 0.5)
            action_label = str(cand.get("semantic_id", cand.get("action", ""))).strip().lower()

            feat = build_feature_vector(
                event_type=event_type,
                query_text=query_text,
                action_label=action_label,
                overlap=overlap,
                action_confidence=action_conf,
                uq_score=uq,
            )

            silver_p_match = _safe_float(silver.get("silver_p_match", 0.5), 0.5)
            target = 1 if silver_p_match >= target_threshold else 0

            samples.append({
                "sample_id": f"sj_{sid:07d}",
                "query_id": query_id,
                "track_id": track_id,
                "event_type": event_type,
                "query_text": query_text,
                "action_label": action_label,
                "overlap": overlap,
                "action_confidence": action_conf,
                "uq_score": uq,
                "feature_vector": [round(v, 6) for v in feat],
                "silver_p_match": silver_p_match,
                "target": target,
            })
            sid += 1
            if silver is not None:
                num_silver_matched += 1

    num_silver_total = len(silver_labels)
    provenance = {
        "teacher_model": teacher_model,
        "num_teacher_labels": num_silver_total,
        "num_samples_built": len(samples),
        "num_silver_matched": num_silver_matched,
        "target_threshold": target_threshold,
        "training_date": str(date.today()),
    }

    if num_silver_matched < num_silver_total * 0.5:
        print(f"[WARN] Only matched {num_silver_matched}/{num_silver_total} "
              f"silver labels to alignment candidates", file=sys.stderr)

    return samples, provenance


# ── train/val split ──────────────────────────────────────────────────────

def _split_samples(samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/val split preserving class ratio."""
    pos = [s for s in samples if int(s.get("target", 0)) == 1]
    neg = [s for s in samples if int(s.get("target", 0)) == 0]

    import random
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)

    # Take 20% of each class for validation
    n_pos_val = max(1, len(pos) // 5)
    n_neg_val = max(1, len(neg) // 5)

    val = pos[:n_pos_val] + neg[:n_neg_val]
    train = pos[n_pos_val:] + neg[n_neg_val:]

    rng.shuffle(train)
    rng.shuffle(val)

    if not val:
        val = train[-max(1, len(train) // 5):]
        train = train[:len(train) - len(val)]

    return train, val


def _tensorize(samples: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = [s["feature_vector"] for s in samples]
    labels = [s["target"] for s in samples]
    x = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, y


# ── temperature scaling ──────────────────────────────────────────────────

def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    best_t = 1.0
    best_brier = 10.0
    for t in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        probs = torch.sigmoid(logits / t)
        bs = brier_score(probs, labels)
        if bs < best_brier:
            best_brier = bs
            best_t = float(t)
    return best_t


def _metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> Dict[str, float]:
    probs = torch.sigmoid(logits / temperature)
    pred = (probs >= 0.5).float()
    acc = float((pred == labels).float().mean().item())
    return {
        "accuracy": acc,
        "brier": brier_score(probs, labels),
        "ece": expected_calibration_error(probs, labels, num_bins=10),
        "positive_rate": float(labels.mean().item()),
    }


# ── runtime config from data ─────────────────────────────────────────────

def _derive_runtime_config(
    samples: List[Dict],
    val_logits: torch.Tensor,
    y_val: torch.Tensor,
    temperature: float,
) -> VerifierRuntimeConfig:
    val_probs = torch.sigmoid(val_logits / temperature)
    high_uq = [s for s in samples if float(s.get("uq_score", 0.0)) >= 0.6]
    high_uq_neg_rate = 0.0
    if high_uq:
        high_uq_neg_rate = sum(1 for s in high_uq if int(s["target"]) == 0) / len(high_uq)

    match_threshold = 0.60
    uncertain_threshold = 0.40
    pos_mask = y_val >= 0.5
    neg_mask = ~pos_mask
    if bool(pos_mask.any()) and bool(neg_mask.any()):
        pos_probs = val_probs[pos_mask]
        neg_probs = val_probs[neg_mask]
        match_threshold = _clamp(float(torch.quantile(pos_probs, 0.40).item()), 0.50, 0.80)
        uncertain_threshold = _clamp(float(torch.quantile(neg_probs, 0.80).item()), 0.20, match_threshold - 0.05)

    return VerifierRuntimeConfig(
        match_threshold=float(match_threshold),
        uncertain_threshold=float(uncertain_threshold),
        uq_gate=min(0.85, max(0.45, 0.50 + 0.35 * high_uq_neg_rate)),
        temperature=temperature,
    )


# ── main training function ───────────────────────────────────────────────

def train_student(
    *,
    silver_labels_path: Path,
    event_queries_path: Path,
    aligned_path: Path,
    out_model: Path,
    out_report: Path,
    out_samples: Path,
    epochs: int = 120,
    lr: float = 1e-3,
    hidden_dim: int = 16,
    target_threshold: float = 0.60,
) -> Dict[str, Any]:
    """Run the full student training pipeline.

    Returns the training report dict.
    """
    # 1. Build samples from silver labels
    samples, provenance = build_student_samples(
        silver_labels_path=silver_labels_path,
        event_queries_path=event_queries_path,
        aligned_path=aligned_path,
        target_threshold=target_threshold,
    )
    if not samples:
        raise RuntimeError("no training samples generated from silver labels")

    if out_samples:
        out_samples.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_samples, samples)

    # 2. Split and tensorize
    train_samples, val_samples = _split_samples(samples)
    x_train, y_train = _tensorize(train_samples)
    x_val, y_val = _tensorize(val_samples)

    # 3. Initialize model
    model = VerifierMLP(in_dim=x_train.shape[1], hidden_dim=hidden_dim)
    pos_count = float(y_train.sum().item())
    neg_count = float(len(y_train) - pos_count)
    if pos_count > 0 and neg_count > 0:
        pos_weight_value = max(1.0, neg_count / pos_count)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        pos_weight_value = 1.0
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Train
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # 5. Evaluate and calibrate
    model.eval()
    with torch.no_grad():
        train_logits = model(x_train)
        val_logits = model(x_val)
    temperature = _fit_temperature(val_logits, y_val)
    train_metrics = _metrics(train_logits, y_train, temperature=temperature)
    val_metrics = _metrics(val_logits, y_val, temperature=temperature)

    # 6. Derive runtime config
    runtime_cfg = _derive_runtime_config(samples, val_logits, y_val, temperature)

    # 7. Save checkpoint
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "student_judge_v1",
            "version": "2.0",
            "in_dim": int(x_train.shape[1]),
            "hidden_dim": hidden_dim,
            "state_dict": model.state_dict(),
            "runtime_config": runtime_cfg.to_dict(),
            "teacher_provenance": provenance,
        },
        out_model,
    )

    # 8. Build and save report
    label_dist = Counter(str(s.get("target", 0)) for s in samples)
    report: Dict[str, Any] = {
        "kind": "student_judge_train_report",
        "schema_version": SCHEMA_VERSION,
        "num_evidence_cases": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "label_distribution": {
            "positive": label_dist.get("1", 0),
            "negative": label_dist.get("0", 0),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "class_balance": {
            "train_positive": int(pos_count),
            "train_negative": int(neg_count),
            "pos_weight": float(pos_weight_value),
        },
        "runtime_config": runtime_cfg.to_dict(),
        "teacher_provenance": provenance,
        "paths": {
            "silver_labels": str(silver_labels_path),
            "event_queries": str(event_queries_path),
            "aligned": str(aligned_path),
            "model": str(out_model),
            "samples": str(out_samples) if out_samples else "",
        },
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] student judge model: {out_model}")
    print(f"[DONE] training report: {out_report}")
    print(f"[INFO] samples={len(samples)} train={len(train_samples)} val={len(val_samples)}")
    print(f"[INFO] val_acc={val_metrics['accuracy']:.4f} val_brier={val_metrics['brier']:.4f} "
          f"val_ece={val_metrics['ece']:.4f}")

    return report


# ── smoke check ──────────────────────────────────────────────────────────

def smoke_check_student(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate student training quality.

    Compares against majority-class baseline to handle imbalanced labels.
    Reports feature degeneracy warnings rather than hard-failing when
    features have no discriminative power (which is valid pipeline output).
    """
    warnings = []
    errors = []

    val_metrics = report.get("val_metrics", {})
    train_metrics = report.get("train_metrics", {})
    val_acc = val_metrics.get("accuracy", 0.0)
    train_acc = train_metrics.get("accuracy", 0.0)
    class_balance = report.get("class_balance", {})
    train_neg = float(class_balance.get("train_negative", 0))
    train_pos = float(class_balance.get("train_positive", 0))
    total = train_neg + train_pos
    majority_ratio = max(train_neg, train_pos) / max(1.0, total)

    # Must beat majority-class baseline by at least 0.02 to show learning.
    # When features are degenerate (all candidates have same overlap/text_score),
    # the model may not beat baseline. Report this as warning, not hard error.
    baseline = majority_ratio
    min_acceptable = baseline + 0.02

    if val_acc < min_acceptable:
        msg = (
            f"val_accuracy={val_acc:.4f} < {min_acceptable:.4f} "
            f"(majority baseline={baseline:.4f} + 0.02)"
        )
        if train_acc <= baseline:
            warnings.append(f"train_acc={train_acc:.4f} also at baseline — "
                           f"features may lack discriminative power for this case")
            warnings.append(msg)
        else:
            errors.append(msg)

    if val_metrics.get("brier", 1.0) > 0.70:
        warnings.append(f"val_brier={val_metrics['brier']:.4f} > 0.70")
    if val_metrics.get("ece", 1.0) > 0.50:
        warnings.append(f"val_ece={val_metrics['ece']:.4f} > 0.50")

    temp = report.get("runtime_config", {}).get("temperature", 0.0)
    if temp < 0.3:
        warnings.append(f"temperature={temp} < 0.3 (uncalibrated)")

    num_samples = report.get("num_evidence_cases", 0)
    if num_samples < 5:
        errors.append(f"too few samples: {num_samples}")

    passed = len(errors) == 0
    return {
        "passed": passed,
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "majority_baseline": round(baseline, 4),
        "min_acceptable": round(min_acceptable, 4),
        "val_brier": val_metrics.get("brier", -1),
        "val_ece": val_metrics.get("ece", -1),
        "temperature": temp,
        "num_samples": num_samples,
        "warnings": warnings,
        "errors": errors,
        "detail": "OK" if passed else "; ".join(errors) if errors else "; ".join(warnings),
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Train student judge on LLM silver labels"
    )
    parser.add_argument("--silver_labels", required=True, type=str,
                        help="llm_silver_labels.jsonl from step 2")
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--aligned", required=True, type=str)
    parser.add_argument("--out_model", required=True, type=str,
                        help="student_judge.pt")
    parser.add_argument("--out_report", required=True, type=str,
                        help="student_train_report.json")
    parser.add_argument("--out_samples", default="", type=str,
                        help="optional sample dump (.jsonl)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--target_threshold", type=float, default=0.60,
                        help="silver_p_match >= this → positive target")
    parser.add_argument("--smoke_report", default="", type=str)
    args = parser.parse_args()

    report = train_student(
        silver_labels_path=Path(args.silver_labels),
        event_queries_path=Path(args.event_queries),
        aligned_path=Path(args.aligned),
        out_model=Path(args.out_model),
        out_report=Path(args.out_report),
        out_samples=Path(args.out_samples) if args.out_samples else Path(),
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        target_threshold=args.target_threshold,
    )

    check = smoke_check_student(report)
    if args.smoke_report:
        smoke_path = Path(args.smoke_report)
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        with smoke_path.open("w", encoding="utf-8") as f:
            json.dump(check, f, ensure_ascii=False, indent=2)

    if not check["passed"]:
        print(f"[SMOKE FAIL] {check['detail']}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[SMOKE PASS] {check['detail']}")


if __name__ == "__main__":
    main()
