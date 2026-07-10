#!/usr/bin/env python3
"""06i — Train a lightweight student judge on LLM silver labels.

Trains sklearn / lightgbm / torch MLP models as multiclass classifiers
(match / mismatch / uncertain). Uses deterministic split by (case_id, event_id).

Output:
  output/llm_judge_pipeline/models/   — trained model files (.joblib / .pt)
  output/llm_judge_pipeline/metrics/  — training_report.json, per-class metrics
"""

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Ensure project root ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Models ──────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import evidence builder from 06h
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "llm_teacher_step",
    _PROJECT_ROOT / "scripts" / "pipeline" / "06h_run_llm_teacher.py",
)
_llm_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_llm_mod)
build_evidence_pairs = _llm_mod.build_evidence_pairs

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION, write_json


# ══════════════════════════════════════════════════════════════════════
# Feature allowlist — ONLY these may be used as student input
# ══════════════════════════════════════════════════════════════════════

FEATURE_ALLOWLIST = {
    # Numeric
    "overlap": {"type": "numeric", "source": "candidate"},
    "action_confidence": {"type": "numeric", "source": "candidate"},
    "uq_score": {"type": "numeric", "source": "candidate"},
    "text_score": {"type": "numeric", "source": "derived (action_match_score)"},
    "audio_confidence": {"type": "numeric", "source": "query"},
    "stability_score": {"type": "numeric", "source": "derived (1 - uq_score)"},
    # Categorical
    "behavior_code": {"type": "categorical_onehot", "source": "candidate", "categories": [
        "tt", "dx", "dk", "zt", "xt", "js", "zl", "jz"
    ]},
    "event_type": {"type": "categorical_onehot", "source": "query"},
    "query_source": {"type": "categorical_onehot", "source": "query", "categories": ["asr", "visual_fallback"]},
}

EXCLUDED_FEATURES = [
    # Target leakage
    "llm_label", "llm_match_score", "llm_mismatch_score", "llm_uncertain_score",
    "llm_confidence", "llm_rationale", "llm_raw_response",
    # Identity fields (not semantic)
    "event_id", "track_id", "case_id", "input_signature",
    # Raw text (not structured features)
    "query_text", "behavior_label_zh", "behavior_label_en",
    "semantic_label_zh", "semantic_label_en", "action_label", "raw_action",
    # Metadata
    "schema_version", "model_name", "prompt_version", "generated_at", "provider_mode",
]

LABEL_MAP = {"match": 0, "mismatch": 1, "uncertain": 2}
LABEL_INV = {0: "match", 1: "mismatch", 2: "uncertain"}


# ══════════════════════════════════════════════════════════════════════
# Data loading and feature building
# ══════════════════════════════════════════════════════════════════════

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
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


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_teacher_labels(teacher_dir: Path) -> List[Dict[str, Any]]:
    """Load all teacher labels from the teacher_labels directory, dedup by input_signature."""
    records: Dict[str, Dict[str, Any]] = {}
    for f in sorted(teacher_dir.glob("llm_teacher_output.front_*.jsonl")):
        if "sample" in f.name:
            continue
        for row in _load_jsonl(f):
            sig = row.get("input_signature", "")
            if sig:
                records[sig] = row
    return list(records.values())


def build_feature_matrix(
    labels: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Build feature matrix and target vector from teacher labels.

    Returns:
        X: feature matrix (n_samples, n_features)
        y: target vector (n_samples,) with values 0=match, 1=mismatch, 2=uncertain
        feature_names: list of feature column names
        metadata: dict with used_features, excluded_features, class_distribution
    """
    # Each label record has: case_id, event_id, track_id, input_signature, llm_label
    # We need to match these back to evidence features.
    # The input_signature = hash(event_id|track_id|query_text|behavior_code)
    # We rebuild evidence pairs for each case and match by (event_id, track_id).

    # Group labels by case_id
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for rec in labels:
        by_case.setdefault(rec["case_id"], []).append(rec)

    # Case configs (same as _run_teacher_batch.py)
    CASE_CONFIGS = {
        "front_002_full_pose020_hybrid": {
            "event_queries": "output/codex_reports/front_002_full_pose020_hybrid/event_queries.fusion_v2.jsonl",
            "aligned": "output/codex_reports/front_002_full_pose020_hybrid/align_multimodal.json",
            "actions": "output/codex_reports/front_002_full_pose020_hybrid/actions.fusion_v2.jsonl",
            "pose_uq": "output/codex_reports/front_002_full_pose020_hybrid/pose_tracks_smooth_uq.jsonl",
        },
        "front_002_rear_row_sliced_pose020_hybrid": {
            "event_queries": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/event_queries.fusion_v2.jsonl",
            "aligned": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/align_multimodal.json",
            "actions": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/actions.fusion_v2.jsonl",
            "pose_uq": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/pose_tracks_smooth_uq.jsonl",
        },
        "front_1885_full": {
            "event_queries": "output/codex_reports/front_1885_full/event_queries.fusion_v2.jsonl",
            "aligned": "output/codex_reports/front_1885_full/align_multimodal.json",
            "actions": "output/codex_reports/front_1885_full/actions.fusion_v2.jsonl",
            "pose_uq": "output/codex_reports/front_1885_full/pose_tracks_smooth_uq.jsonl",
        },
        "front_22259_full": {
            "event_queries": "output/codex_reports/front_22259_full/event_queries.fusion_v2.jsonl",
            "aligned": "output/codex_reports/front_22259_full/align_multimodal.json",
            "actions": "output/codex_reports/front_22259_full/actions.fusion_v2.jsonl",
            "pose_uq": "output/codex_reports/front_22259_full/pose_tracks_smooth_uq.jsonl",
        },
        "front_26729_full": {
            "event_queries": "output/codex_reports/front_26729_full/event_queries.fusion_v2.jsonl",
            "aligned": "output/codex_reports/front_26729_full/align_multimodal.json",
            "actions": "output/codex_reports/front_26729_full/actions.fusion_v2.jsonl",
            "pose_uq": "output/codex_reports/front_26729_full/pose_tracks_smooth_uq.jsonl",
        },
        "front_45618_full": {
            "event_queries": "output/codex_reports/front_45618_full/event_queries.fusion_v2.jsonl",
            "aligned": "output/codex_reports/front_45618_full/align_multimodal.json",
            "actions": "output/codex_reports/front_45618_full/actions.fusion_v2.jsonl",
            "pose_uq": "output/codex_reports/front_45618_full/pose_tracks_smooth_uq.jsonl",
        },
    }

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()

    # Build evidence pair lookup: (case_id, event_id, track_id) -> evidence pair
    evidence_lookup: Dict[str, Dict[str, Any]] = {}

    for case_name, cfg in CASE_CONFIGS.items():
        pairs = build_evidence_pairs(
            event_queries_path=_resolve(cfg["event_queries"]),
            aligned_path=_resolve(cfg["aligned"]),
            actions_path=_resolve(cfg["actions"]) if cfg.get("actions") else None,
            pose_uq_path=_resolve(cfg["pose_uq"]) if cfg.get("pose_uq") else None,
            case_id=case_name,
        )
        for pair in pairs:
            # Use input_signature as key (matches teacher label)
            sig = hashlib.sha256(
                f"{pair['event_id']}|{pair['track_id']}|{pair['query_text']}|{pair['behavior_code']}"
                .encode("utf-8")
            ).hexdigest()[:16]
            evidence_lookup[sig] = pair

    # Build feature rows
    X_rows: List[List[float]] = []
    y_list: List[int] = []
    matched = 0
    unmatched = 0
    missing_sigs: List[str] = []

    for rec in labels:
        sig = rec.get("input_signature", "")
        pair = evidence_lookup.get(sig)

        if pair is None:
            unmatched += 1
            missing_sigs.append(sig)
            continue
        matched += 1

        # Target
        label = rec["llm_label"]
        y_list.append(LABEL_MAP.get(label, 1))  # default mismatch

        # ── Feature vector ──
        row = []

        # Numeric features (in fixed order)
        row.append(_safe_float(pair.get("overlap", 0.0)))
        row.append(_safe_float(pair.get("action_confidence", 0.0)))
        row.append(_safe_float(pair.get("uq_score", 0.5)))
        row.append(_safe_float(pair.get("text_score", 0.0)))
        row.append(_safe_float(pair.get("audio_confidence", 0.0)))
        # stability_score = 1 - uq
        row.append(max(0.0, min(1.0, 1.0 - _safe_float(pair.get("uq_score", 0.5)))))

        # Categorical: behavior_code one-hot (8 classes)
        bc = pair.get("behavior_code", "").strip().lower()
        for cat in FEATURE_ALLOWLIST["behavior_code"]["categories"]:
            row.append(1.0 if bc == cat else 0.0)

        # Categorical: event_type one-hot
        et = pair.get("event_type", "unknown").strip().lower()
        # Collect unique event types across the dataset
        # We'll handle unknown dynamically
        # For now, just encode known vs unknown
        row.append(0.0 if et == "unknown" else 1.0)

        # Categorical: query_source
        qs = pair.get("query_source", "").strip().lower()
        row.append(1.0 if qs == "asr" else 0.0)

        X_rows.append(row)

    feature_names = [
        # Numeric (6)
        "overlap",
        "action_confidence",
        "uq_score",
        "text_score",
        "audio_confidence",
        "stability_score",
        # behavior_code one-hot (8)
        "behavior_code_tt",
        "behavior_code_dx",
        "behavior_code_dk",
        "behavior_code_zt",
        "behavior_code_xt",
        "behavior_code_js",
        "behavior_code_zl",
        "behavior_code_jz",
        # event_type
        "event_type_known",
        # query_source
        "query_source_asr",
    ]

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    class_counts = Counter(LABEL_INV.get(int(v), "unknown") for v in y)

    metadata = {
        "n_total_teacher_labels": len(labels),
        "n_matched_to_evidence": matched,
        "n_unmatched": unmatched,
        "n_unmatched_samples": missing_sigs[:5] if missing_sigs else [],
        "used_features": feature_names,
        "used_feature_sources": {
            name: FEATURE_ALLOWLIST.get(name.split("_")[0], {}).get("source", "derived")
            for name in feature_names
        },
        "excluded_features": EXCLUDED_FEATURES,
        "class_distribution": dict(class_counts),
        "feature_dim": len(feature_names),
    }

    return X, y, feature_names, metadata


def deterministic_split(
    X: np.ndarray,
    y: np.ndarray,
    labels: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           List[int], List[int], List[int]]:
    """Deterministic split by (case_id, event_id) hash.

    Ensures the same event never appears in both train and val/test.
    """
    # Group indices by split key
    key_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(labels):
        key = f"{rec.get('case_id', '')}|{rec.get('event_id', '')}"
        key_to_indices[key].append(i)

    # Deterministic assignment using hash
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for key, indices in key_to_indices.items():
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
        if h < train_ratio * 100:
            train_idx.extend(indices)
        elif h < (train_ratio + val_ratio) * 100:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    # Filter to indices that exist in our matched data
    n = X.shape[0]
    train_idx = [i for i in train_idx if i < n]
    val_idx = [i for i in val_idx if i < n]
    test_idx = [i for i in test_idx if i < n]

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
        train_idx, val_idx, test_idx,
    )


# ══════════════════════════════════════════════════════════════════════
# Model training
# ══════════════════════════════════════════════════════════════════════

def _clamp_probs(probs: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.clip(probs, eps, 1.0 - eps)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    split_name: str,
) -> Dict[str, Any]:
    """Compute per-class and aggregate metrics."""
    labels_sorted = sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred))
    label_names = [LABEL_INV.get(i, str(i)) for i in labels_sorted]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    cm_list = cm.tolist()

    # Per-class metrics
    per_class = {}
    report_dict = classification_report(
        y_true, y_pred, labels=labels_sorted,
        target_names=label_names, output_dict=True, zero_division=0
    )
    for name in label_names:
        if name in report_dict:
            per_class[name] = {
                "precision": round(float(report_dict[name]["precision"]), 4),
                "recall": round(float(report_dict[name]["recall"]), 4),
                "f1": round(float(report_dict[name]["f1-score"]), 4),
                "support": int(report_dict[name]["support"]),
            }

    # Log loss
    ll = float("nan")
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        y_onehot = np.zeros((len(y_true), n_classes))
        y_onehot[np.arange(len(y_true)), y_true] = 1.0
        ll = float(log_loss(y_onehot, _clamp_probs(y_proba)))

    return {
        "split": split_name,
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "log_loss": round(ll, 4) if not np.isnan(ll) else None,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": label_names,
            "matrix": cm_list,
        },
        "n_samples": int(len(y_true)),
        "class_distribution": dict(Counter(LABEL_INV.get(int(v), "?") for v in y_true)),
    }


ModelObjects = Tuple[Any, Any, Any]  # (lr, rf, hgb)


def train_sklearn_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Tuple[Dict[str, Any], ModelObjects]:
    """Train multiple sklearn models and return consolidated results + model objects."""
    results: Dict[str, Any] = {}
    best_name = None
    best_f1 = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ── 1. LogisticRegression (multinomial) ──
        print("  Training LogisticRegression...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            random_state=42,
            class_weight="balanced",
        )
        lr.fit(X_train_scaled, y_train)
        lr_val_pred = lr.predict(X_val_scaled)
        lr_test_pred = lr.predict(X_test_scaled)
        lr_val_proba = lr.predict_proba(X_val_scaled)
        lr_test_proba = lr.predict_proba(X_test_scaled)

        lr_val_metrics = compute_metrics(y_val, lr_val_pred, lr_val_proba, "val")
        lr_test_metrics = compute_metrics(y_test, lr_test_pred, lr_test_proba, "test")

        results["LogisticRegression"] = {
            "val_metrics": lr_val_metrics,
            "test_metrics": lr_test_metrics,
            "coefficients": {
                name: [round(float(c), 6) for c in coef]
                for name, coef in zip(
                    [LABEL_INV.get(i, str(i)) for i in range(lr.coef_.shape[0])],
                    lr.coef_
                )
            },
            "n_features": X_train.shape[1],
            "feature_names": feature_names,
            "classes": [LABEL_INV.get(i, str(i)) for i in lr.classes_],
            "model_type": "LogisticRegression",
        }

        f1_lr = lr_val_metrics["f1_macro"]
        if f1_lr > best_f1:
            best_f1 = f1_lr
            best_name = "LogisticRegression"

        # ── 2. RandomForestClassifier ──
        print("  Training RandomForest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_val_pred = rf.predict(X_val)
        rf_test_pred = rf.predict(X_test)
        rf_val_proba = rf.predict_proba(X_val)
        rf_test_proba = rf.predict_proba(X_test)

        rf_val_metrics = compute_metrics(y_val, rf_val_pred, rf_val_proba, "val")
        rf_test_metrics = compute_metrics(y_test, rf_test_pred, rf_test_proba, "test")

        rf_importances = sorted(
            zip(feature_names, rf.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )

        results["RandomForest"] = {
            "val_metrics": rf_val_metrics,
            "test_metrics": rf_test_metrics,
            "feature_importances": [
                {"feature": name, "importance": round(float(imp), 6)}
                for name, imp in rf_importances
            ],
            "n_features": X_train.shape[1],
            "feature_names": feature_names,
            "classes": [LABEL_INV.get(i, str(i)) for i in rf.classes_],
            "model_type": "RandomForest",
        }

        f1_rf = rf_val_metrics["f1_macro"]
        if f1_rf > best_f1:
            best_f1 = f1_rf
            best_name = "RandomForest"

        # ── 3. HistGradientBoostingClassifier ──
        print("  Training HistGradientBoosting...")
        hgb = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            min_samples_leaf=10,
            learning_rate=0.1,
            random_state=42,
            categorical_features=None,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        hgb.fit(X_train, y_train)
        hgb_val_pred = hgb.predict(X_val)
        hgb_test_pred = hgb.predict(X_test)
        hgb_val_proba = hgb.predict_proba(X_val)
        hgb_test_proba = hgb.predict_proba(X_test)

        hgb_val_metrics = compute_metrics(y_val, hgb_val_pred, hgb_val_proba, "val")
        hgb_test_metrics = compute_metrics(y_test, hgb_test_pred, hgb_test_proba, "test")

        from sklearn.inspection import permutation_importance
        perm_imp = permutation_importance(
            hgb, X_val, y_val,
            n_repeats=5, random_state=42, n_jobs=-1,
        )
        hgb_importances = sorted(
            zip(feature_names, perm_imp.importances_mean),
            key=lambda x: x[1], reverse=True,
        )

        results["HistGradientBoosting"] = {
            "val_metrics": hgb_val_metrics,
            "test_metrics": hgb_test_metrics,
            "feature_importances": [
                {"feature": name, "importance": round(float(imp), 6)}
                for name, imp in hgb_importances
            ],
            "n_features": X_train.shape[1],
            "feature_names": feature_names,
            "classes": [LABEL_INV.get(i, str(i)) for i in hgb.classes_],
            "model_type": "HistGradientBoosting",
        }

        f1_hgb = hgb_val_metrics["f1_macro"]
        if f1_hgb > best_f1:
            best_f1 = f1_hgb
            best_name = "HistGradientBoosting"

    return results, (lr, rf, hgb, scaler)


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Tuple[Optional[Dict[str, Any]], Any]:
    """Train LightGBM classifier if available. Returns (metrics_dict, model)."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None, None

    print("  Training LightGBM...")
    n_classes = len(set(int(v) for v in y_train))
    if n_classes < 2:
        return None, None

    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        min_child_samples=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )

    lgb_val_pred = lgb_model.predict(X_val)
    lgb_test_pred = lgb_model.predict(X_test)
    lgb_val_proba = lgb_model.predict_proba(X_val)
    lgb_test_proba = lgb_model.predict_proba(X_test)

    val_metrics = compute_metrics(y_val, lgb_val_pred, lgb_val_proba, "val")
    test_metrics = compute_metrics(y_test, lgb_test_pred, lgb_test_proba, "test")

    lgb_importances = sorted(
        zip(feature_names, lgb_model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )

    result = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importances": [
            {"feature": name, "importance": int(imp)}
            for name, imp in lgb_importances
        ],
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
        "classes": [LABEL_INV.get(i, str(i)) for i in range(n_classes)],
        "best_iteration": lgb_model.best_iteration_ if hasattr(lgb_model, "best_iteration_") else None,
    }
    return result, lgb_model


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    use_cuda: bool = True,
) -> Optional[Dict[str, Any]]:
    """Train a PyTorch MLP with CUDA if available."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return None

    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")
    print(f"  Training MLP (device={device})...")

    n_features = X_train.shape[1]
    n_classes = len(set(int(v) for v in y_train))
    if n_classes < 2:
        return None

    # Normalize
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_val_n = scaler.transform(X_val)
    X_test_n = scaler.transform(X_test)

    # Convert to tensors
    dtype = torch.float32
    x_t = torch.tensor(X_train_n, dtype=dtype).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    x_v = torch.tensor(X_val_n, dtype=dtype).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)
    x_te = torch.tensor(X_test_n, dtype=dtype).to(device)
    y_te = torch.tensor(y_test, dtype=torch.long).to(device)

    # MLP architecture: n_features → 64 → 32 → n_classes
    class MLP(nn.Module):
        def __init__(self, n_in: int, n_out: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, n_out),
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(n_features, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training
    best_val_loss = float("inf")
    best_state = None
    patience = 30
    no_improve = 0

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        logits = model(x_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(x_v)
            val_loss = criterion(val_logits, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Restore best
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate
    model.eval()
    with torch.no_grad():
        val_logits = model(x_v)
        test_logits = model(x_te)
        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        val_pred = np.argmax(val_probs, axis=1)
        test_pred = np.argmax(test_probs, axis=1)

    val_metrics = compute_metrics(y_val, val_pred, val_probs, "val")
    test_metrics = compute_metrics(y_test, test_pred, test_probs, "test")

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_features": n_features,
        "feature_names": feature_names,
        "device": str(device),
        "epochs_trained": epoch + 1,
        "best_val_loss": round(float(best_val_loss), 4),
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="06i — Train small student judge on LLM silver labels"
    )
    parser.add_argument("--teacher_dir",
                        default="output/llm_judge_pipeline/teacher_labels",
                        type=str, help="directory with llm_teacher_output.*.jsonl")
    parser.add_argument("--out_models",
                        default="output/llm_judge_pipeline/models",
                        type=str, help="output directory for model files")
    parser.add_argument("--out_metrics",
                        default="output/llm_judge_pipeline/metrics",
                        type=str, help="output directory for metrics")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="use CUDA for PyTorch MLP")
    parser.add_argument("--no_cuda", action="store_false", dest="cuda")
    args = parser.parse_args()

    t_start = time.time()

    teacher_dir = Path(args.teacher_dir)
    if not teacher_dir.is_absolute():
        teacher_dir = (_PROJECT_ROOT / teacher_dir).resolve()
    out_models = Path(args.out_models)
    if not out_models.is_absolute():
        out_models = (_PROJECT_ROOT / out_models).resolve()
    out_metrics = Path(args.out_metrics)
    if not out_metrics.is_absolute():
        out_metrics = (_PROJECT_ROOT / out_metrics).resolve()

    out_models.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ──
    print("Loading teacher labels...")
    labels = load_teacher_labels(teacher_dir)
    print(f"  {len(labels)} unique teacher labels loaded")

    if not labels:
        print("[ERROR] No teacher labels found", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Build feature matrix ──
    print("Building feature matrix...")
    X, y, feature_names, data_meta = build_feature_matrix(labels)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: {data_meta['class_distribution']}")

    if X.shape[0] < 10:
        print(f"[ERROR] Too few samples ({X.shape[0]})", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Split ──
    print("Splitting (case_id+event_id hash)...")
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = \
        deterministic_split(X, y, labels)

    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"  Train classes: {dict(Counter(LABEL_INV.get(int(v),'?') for v in y_train))}")

    # ── Step 4: Train models ──
    print("\nTraining sklearn models...")
    sklearn_results, (lr_model, rf_model, hgb_model, scaler_model) = train_sklearn_models(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names,
    )

    print("\nTraining LightGBM...")
    lgb_result, lgb_model = train_lightgbm(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names,
    )

    print("\nTraining PyTorch MLP...")
    mlp_result = train_torch_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names,
        use_cuda=args.cuda,
    )

    # ── Step 5: Build report ──
    flat_models: Dict[str, Any] = {
        "LogisticRegression": sklearn_results.get("LogisticRegression", {}),
        "RandomForest": sklearn_results.get("RandomForest", {}),
        "HistGradientBoosting": sklearn_results.get("HistGradientBoosting", {}),
    }
    if lgb_result is not None:
        flat_models["LightGBM"] = lgb_result
    if mlp_result is not None:
        flat_models["PyTorchMLP"] = mlp_result

    # Determine best model by val f1_macro
    best_model_name = "LogisticRegression"
    best_val_f1 = 0.0
    for name, info in flat_models.items():
        f1m = info.get("val_metrics", {}).get("f1_macro", 0.0)
        if f1m > best_val_f1:
            best_val_f1 = f1m
            best_model_name = name

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_kind": "pseudo_label_benchmark",
        "evaluation_note": (
            "Metrics are computed against LLM silver labels (pseudo-labels), "
            "not human gold labels. Performance against human gold may differ."
        ),
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "data": data_meta,
        "split": {
            "method": "deterministic_hash_by_case_id_and_event_id",
            "hash_key": "sha256(case_id|event_id) % 100",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "train_classes": dict(Counter(LABEL_INV.get(int(v), "?") for v in y_train)),
            "val_classes": dict(Counter(LABEL_INV.get(int(v), "?") for v in y_val)),
            "test_classes": dict(Counter(LABEL_INV.get(int(v), "?") for v in y_test)),
        },
        "models": flat_models,
        "best_model": best_model_name,
        "best_val_f1_macro": round(best_val_f1, 4),
        "all_feature_names": feature_names,
    }

    # ── Step 6: Save artifacts ──
    # Training report
    report_path = out_metrics / "training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Training report: {report_path}")

    # Save models with joblib
    try:
        import joblib as _jl

        # Save best model
        best_path = out_models / f"student_judge_{best_model_name}.joblib"
        model_map = {
            "LogisticRegression": lr_model,
            "RandomForest": rf_model,
            "HistGradientBoosting": hgb_model,
        }
        best_obj = model_map.get(best_model_name, lr_model)
        _jl.dump({"model": best_obj, "model_name": best_model_name, "feature_names": feature_names,
                   "scaler": scaler_model, "classes": ["match", "mismatch", "uncertain"]}, best_path)
        print(f"[DONE] Best model saved: {best_path}")

        # Save all models
        for mname, mobj in model_map.items():
            mpath = out_models / f"student_judge_{mname}.joblib"
            _jl.dump({"model": mobj, "model_name": mname, "feature_names": feature_names,
                       "scaler": scaler_model if mname == "LogisticRegression" else None,
                       "classes": ["match", "mismatch", "uncertain"]}, mpath)
            print(f"[DONE] Model saved: {mpath}")

        # Save LightGBM model if available
        if lgb_result is not None:
            # lgb_model is still in scope from train_lightgbm return
            import lightgbm as _lgb
            lgb_path = out_models / "student_judge_LightGBM.joblib"
            _jl.dump({"model": lgb_model, "model_name": "LightGBM", "feature_names": feature_names,
                       "classes": ["match", "mismatch", "uncertain"]}, lgb_path)
            print(f"[DONE] Model saved: {lgb_path}")

    except ImportError:
        print("[WARN] joblib not installed, models not saved")

    print(f"[INFO] Best model: {best_model_name} (val_f1_macro={best_val_f1:.4f})")

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[DONE] Training complete in {elapsed:.1f}s")
    print(f"  Samples: {X.shape[0]} ({X_train.shape[0]} train / {X_val.shape[0]} val / {X_test.shape[0]} test)")
    print(f"  Features: {X.shape[1]}")
    print(f"  Best model: {report['best_model']} (val_f1_macro={report['best_val_f1_macro']:.4f})")
    print(f"  Models: {out_models}")
    print(f"  Metrics: {out_metrics}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
