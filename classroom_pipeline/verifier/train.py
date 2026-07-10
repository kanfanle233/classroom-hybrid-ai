import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.verifier.dataset import build_training_samples, save_training_samples
from classroom_pipeline.verifier.model import (
    VerifierMLP,
    VerifierRuntimeConfig,
    brier_score,
    build_feature_vector,
    expected_calibration_error,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _split_samples(samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    train, val = [], []
    for s in samples:
        qid = str(s.get("query_id", ""))
        h = sum(ord(ch) for ch in qid) % 10
        if h < 8:
            train.append(s)
        else:
            val.append(s)
    if not val and train:
        val = train[-max(1, len(train) // 5) :]
        train = train[: len(train) - len(val)]
    return train, val


def _tensorize(samples: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[List[float]] = []
    labels: List[int] = []
    for s in samples:
        feats.append(
            build_feature_vector(
                event_type=str(s["event_type"]),
                query_text=str(s["query_text"]),
                action_label=str(s["action_label"]),
                overlap=float(s["overlap"]),
                action_confidence=float(s["action_confidence"]),
                uq_score=float(s["uq_score"]),
            )
        )
        labels.append(int(s["target"]))
    x = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, y


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train verifier (match vs mismatch) and export checkpoint.")
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--aligned", required=True, type=str)
    parser.add_argument("--actions", required=True, type=str)
    parser.add_argument("--out_model", required=True, type=str, help="verifier.pt")
    parser.add_argument("--out_report", required=True, type=str, help="verifier_report.json")
    parser.add_argument("--out_samples", default="", type=str, help="optional training sample jsonl")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=16)
    args = parser.parse_args()

    event_queries = Path(args.event_queries).resolve()
    aligned = Path(args.aligned).resolve()
    actions = Path(args.actions).resolve()
    out_model = Path(args.out_model).resolve()
    out_report = Path(args.out_report).resolve()
    out_samples = Path(args.out_samples).resolve() if args.out_samples else None

    samples = build_training_samples(
        event_queries_path=event_queries,
        aligned_path=aligned,
        actions_path=actions,
    )
    if not samples:
        raise RuntimeError("no training samples generated for verifier")
    if out_samples:
        out_samples.parent.mkdir(parents=True, exist_ok=True)
        save_training_samples(out_samples, samples)

    train_samples, val_samples = _split_samples(samples)
    x_train, y_train = _tensorize(train_samples)
    x_val, y_val = _tensorize(val_samples)

    model = VerifierMLP(in_dim=x_train.shape[1], hidden_dim=int(args.hidden_dim))
    pos_count = float(y_train.sum().item())
    neg_count = float(len(y_train) - pos_count)
    if pos_count > 0 and neg_count > 0:
        pos_weight_value = max(1.0, neg_count / pos_count)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        pos_weight_value = 1.0
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    model.train()
    for _ in range(int(args.epochs)):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(x_train)
        val_logits = model(x_val)
    temperature = _fit_temperature(val_logits, y_val)
    train_metrics = _metrics(train_logits, y_train, temperature=temperature)
    val_metrics = _metrics(val_logits, y_val, temperature=temperature)
    val_probs = torch.sigmoid(val_logits / temperature)

    # Slightly more conservative gating when high-UQ negatives are frequent.
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

    runtime_cfg = VerifierRuntimeConfig(
        match_threshold=float(match_threshold),
        uncertain_threshold=float(uncertain_threshold),
        uq_gate=min(0.85, max(0.45, 0.50 + 0.35 * high_uq_neg_rate)),
        temperature=temperature,
    )

    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "trainable_verifier",
            "version": "2.0",
            "in_dim": int(x_train.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "state_dict": model.state_dict(),
            "runtime_config": runtime_cfg.to_dict(),
        },
        out_model,
    )

    sample_type_counter = Counter(str(s.get("sample_type", "unknown")) for s in samples)
    report = {
        "kind": "verifier_train_report",
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "sample_types": dict(sample_type_counter),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "class_balance": {
            "train_positive": int(pos_count),
            "train_negative": int(neg_count),
            "pos_weight": float(pos_weight_value),
        },
        "runtime_config": runtime_cfg.to_dict(),
        "paths": {
            "event_queries": str(event_queries),
            "aligned": str(aligned),
            "actions": str(actions),
            "model": str(out_model),
            "samples": str(out_samples) if out_samples else "",
        },
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved verifier model: {out_model}")
    print(f"[DONE] saved training report: {out_report}")
    print(f"[INFO] samples={len(samples)} train={len(train_samples)} val={len(val_samples)}")
    print(f"[INFO] val_acc={val_metrics['accuracy']:.4f} val_brier={val_metrics['brier']:.4f} val_ece={val_metrics['ece']:.4f}")


if __name__ == "__main__":
    main()
