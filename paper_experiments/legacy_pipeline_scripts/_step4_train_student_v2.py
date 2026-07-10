#!/usr/bin/env python3
"""Step 4: Train student v2 on balanced dataset with case-aware splits and per-class metrics."""
import json, sys, hashlib, warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, log_loss, brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV


def _safe_float(x, d=0.0):
    try: return float(x)
    except: return d


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except: continue
    return out


def load_json(path: Path) -> Any:
    if not path.exists(): return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Feature building ───────────────────────────────────────────
FEATURE_ALLOWLIST = [
    "overlap", "action_confidence", "uq_score", "text_score",
    "audio_confidence", "stability_score",
    "behavior_code_tt", "behavior_code_dx", "behavior_code_dk",
    "behavior_code_zt", "behavior_code_xt", "behavior_code_js",
    "behavior_code_zl", "behavior_code_jz",
    "event_type_known", "query_source_asr",
]
BEHAVIOR_CODES = ["tt", "dx", "dk", "zt", "xt", "js", "zl", "jz"]

EXCLUDED = [
    "llm_label", "llm_match_score", "llm_mismatch_score", "llm_uncertain_score",
    "llm_confidence", "llm_rationale", "llm_raw_response",
    "case_id", "event_id", "track_id", "input_signature",
    "provider_mode", "model_name",
]


def compute_metrics_full(y_true, y_pred, y_proba, split_name, label_names):
    n_classes = len(label_names)
    n_samples = len(y_true)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_list = cm.tolist()

    # Single-class check (must be before metrics that depend on it)
    present_classes = sorted(set(int(v) for v in y_true))
    single_class = len(present_classes) < 2

    # Per-class
    per_class = {}
    for i, name in enumerate(label_names):
        tp = cm[i, i] if i < cm.shape[0] else 0
        fp = cm[:, i].sum() - tp if i < cm.shape[0] else 0
        fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
        tn = cm.sum() - tp - fp - fn
        pre = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * pre * rec / max(1e-9, pre + rec)
        per_class[name] = {
            "precision": round(float(pre), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "support": int(tp + fn),
        }

    # Brier
    brier = float("nan")
    if y_proba is not None and not single_class:
        try:
            if n_classes == 2:
                brier = float(brier_score_loss(y_true, y_proba[:, 1]))
            else:
                y_onehot = np.zeros((n_samples, n_classes))
                y_onehot[np.arange(n_samples), y_true] = 1
                brier = float(np.mean(np.sum((y_proba - y_onehot)**2, axis=1)))
        except Exception:
            pass

    # Log loss — requires multi-class in y_true
    ll = float("nan")
    if y_proba is not None and not single_class:
        try:
            ll = float(log_loss(y_true, np.clip(y_proba, 1e-15, 1-1e-15), labels=list(range(n_classes))))
        except Exception:
            pass

    return {
        "split": split_name,
        "n_samples": n_samples,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "brier": round(brier, 4) if not np.isnan(brier) else None,
        "log_loss": round(ll, 4) if not np.isnan(ll) else None,
        "per_class": per_class,
        "confusion_matrix": {"labels": label_names, "matrix": cm_list},
        "class_distribution": dict(Counter(int(v) for v in y_true)),
        "single_class_warning": single_class,
        "warning": "Single-class split — metrics are degenerate." if single_class else None,
    }


def deterministic_split_cases(records, train_frac=0.7, val_frac=0.15):
    """Split by case_id+event_id hash."""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        key = f"{r['case_id']}|{r['event_id']}"
        groups[key].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for key, indices in groups.items():
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
        if h < train_frac * 100:
            train_idx.extend(indices)
        elif h < (train_frac + val_frac) * 100:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)
    return train_idx, val_idx, test_idx


def leave_one_case_out(records):
    """LOOCV by case_id."""
    cases = sorted(set(r["case_id"] for r in records))
    results = {}
    for held_out in cases:
        train = [r for r in records if r["case_id"] != held_out]
        test = [r for r in records if r["case_id"] == held_out]
        if len(train) < 5 or len(test) < 2:
            results[held_out] = {"n_train": len(train), "n_test": len(test), "skipped": True}
            continue
        results[held_out] = {"n_train": len(train), "n_test": len(test), "skipped": False}
    return results


def main() -> None:
    REPORT_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "reports"
    MODEL_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "models"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load balanced dataset ───────────────────────────────────
    ds_path = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "balanced_teacher_dataset_v2.jsonl"
    records = load_jsonl(ds_path)
    if not records:
        print("[ERROR] No records in balanced dataset", file=sys.stderr)
        sys.exit(1)

    # ── Load evidence pairs for features ────────────────────────
    CASE_CONFIGS = {
        "front_002_full_pose020_hybrid": "output/codex_reports/front_002_full_pose020_hybrid",
        "front_002_rear_row_sliced_pose020_hybrid": "output/codex_reports/front_002_rear_row_sliced_pose020_hybrid",
        "front_1885_full": "output/codex_reports/front_1885_full",
        "front_22259_full": "output/codex_reports/front_22259_full",
        "front_26729_full": "output/codex_reports/front_26729_full",
        "front_45618_full": "output/codex_reports/front_45618_full",
    }

    evidence_map = {}
    for case_name, rel_path in CASE_CONFIGS.items():
        cd = _PROJECT_ROOT / rel_path
        eq_path = cd / "event_queries.fusion_v2.jsonl"
        al_path = cd / "align_multimodal.json"

        if not eq_path.exists() or not al_path.exists():
            continue

        queries = load_jsonl(eq_path)
        q_index = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}
        aligned = load_json(al_path)
        aligned = aligned if isinstance(aligned, list) else []

        for block in aligned:
            if not isinstance(block, dict): continue
            q_id = str(block.get("query_id", block.get("event_id", "")))
            q = q_index.get(q_id, {})
            for cand in block.get("candidates", []):
                if not isinstance(cand, dict): continue
                tid = int(cand.get("track_id", -1))
                eid = q_id
                q_text = str(q.get("query_text", block.get("query_text", "")))
                bc = str(cand.get("behavior_code", "")).strip().lower()

                sig = hashlib.sha256(
                    f"{eid}|{tid}|{q_text}|{bc}".encode("utf-8")
                ).hexdigest()[:16]
                evidence_map[sig] = {"case_name": case_name, "query": q, "candidate": cand}

    # ── Build feature matrix ────────────────────────────────────
    X_rows = []
    y_list = []
    matched = 0
    unmatched_sigs = []
    metadata_rows = []

    for r in records:
        sig = r.get("input_signature", "")
        ev = evidence_map.get(sig)
        if ev is None:
            unmatched_sigs.append(sig)
            continue
        matched += 1

        q = ev["query"]
        cand = ev["candidate"]

        overlap = _safe_float(cand.get("overlap", 0))
        action_conf = _safe_float(cand.get("action_confidence", cand.get("confidence", 0)))
        uq = _safe_float(cand.get("uq_score", cand.get("uq_track", 0.5)))
        text_score = _safe_float(0.0)  # action_match_score returns 0 for all "unknown"
        audio_conf = _safe_float(q.get("confidence", 0))
        stability = max(0, min(1, 1 - uq))
        bc = str(cand.get("behavior_code", "")).strip().lower()

        row = [
            overlap, action_conf, uq, text_score, audio_conf, stability,
            *[1.0 if bc == code else 0.0 for code in BEHAVIOR_CODES],
            0.0 if str(q.get("event_type", "unknown")).strip().lower() in ("", "unknown") else 1.0,
            1.0 if str(q.get("source", "")).strip().lower() == "asr" else 0.0,
        ]
        X_rows.append(row)
        y_list.append(1 if r["llm_label"] == "match" else 0)  # binary: match=1, mismatch=0
        metadata_rows.append(r)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    label_names = ["mismatch", "match"]

    print(f"Records: {len(records)}, matched: {matched}, unmatched: {len(unmatched_sigs)}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution: {dict(Counter(int(v) for v in y))}")

    # ── Deterministic split ─────────────────────────────────────
    train_idx, val_idx, test_idx = deterministic_split_cases(metadata_rows)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Train classes: {dict(Counter(int(v) for v in y_train))}")
    print(f"Val classes: {dict(Counter(int(v) for v in y_val))}")
    print(f"Test classes: {dict(Counter(int(v) for v in y_test))}")

    # Check single-class splits
    val_single = len(set(int(v) for v in y_val)) < 2
    test_single = len(set(int(v) for v in y_test)) < 2

    # ── Train models ────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    models_results = {}

    # ---- LogisticRegression ----
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)
    lr_val_pred = lr.predict(X_val_s)
    lr_val_proba = lr.predict_proba(X_val_s)
    lr_test_pred = lr.predict(X_test_s)
    lr_test_proba = lr.predict_proba(X_test_s)

    models_results["LogisticRegression"] = {
        "val_metrics": compute_metrics_full(y_val, lr_val_pred, lr_val_proba, "val", label_names),
        "test_metrics": compute_metrics_full(y_test, lr_test_pred, lr_test_proba, "test", label_names),
        "coefficients": {label_names[i]: [round(float(c), 6) for c in lr.coef_[i]] for i in range(lr.coef_.shape[0])},
    }

    # ---- RandomForest ----
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2,
                                 class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    rf_val_proba = rf.predict_proba(X_val)
    rf_test_pred = rf.predict(X_test)
    rf_test_proba = rf.predict_proba(X_test)

    rf_importances = sorted(zip(FEATURE_ALLOWLIST, rf.feature_importances_),
                            key=lambda x: x[1], reverse=True)
    models_results["RandomForest"] = {
        "val_metrics": compute_metrics_full(y_val, rf_val_pred, rf_val_proba, "val", label_names),
        "test_metrics": compute_metrics_full(y_test, rf_test_pred, rf_test_proba, "test", label_names),
        "feature_importances": [{"feature": n, "importance": round(float(i), 6)} for n, i in rf_importances],
    }

    # ---- HistGradientBoosting ----
    hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=6, min_samples_leaf=5,
                                          learning_rate=0.1, random_state=42)
    hgb.fit(X_train, y_train)
    hgb_val_pred = hgb.predict(X_val)
    hgb_val_proba = hgb.predict_proba(X_val)
    hgb_test_pred = hgb.predict(X_test)
    hgb_test_proba = hgb.predict_proba(X_test)
    models_results["HistGradientBoosting"] = {
        "val_metrics": compute_metrics_full(y_val, hgb_val_pred, hgb_val_proba, "val", label_names),
        "test_metrics": compute_metrics_full(y_test, hgb_test_pred, hgb_test_proba, "test", label_names),
    }

    # ---- LightGBM ----
    lgb_result = None
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                   class_weight="balanced", random_state=42, verbose=-1)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                 eval_metric="binary_logloss",
                 callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        lgb_val_pred = lgbm.predict(X_val)
        lgb_val_proba = lgbm.predict_proba(X_val)
        lgb_test_pred = lgbm.predict(X_test)
        lgb_test_proba = lgbm.predict_proba(X_test)
        lgb_importances = sorted(zip(FEATURE_ALLOWLIST, lgbm.feature_importances_),
                                 key=lambda x: x[1], reverse=True)
        lgb_result = {
            "val_metrics": compute_metrics_full(y_val, lgb_val_pred, lgb_val_proba, "val", label_names),
            "test_metrics": compute_metrics_full(y_test, lgb_test_pred, lgb_test_proba, "test", label_names),
            "feature_importances": [{"feature": n, "importance": int(i)} for n, i in lgb_importances],
        }
    except ImportError:
        pass

    # ── LOOCV ───────────────────────────────────────────────────
    loocv_cases = leave_one_case_out(metadata_rows)
    loocv_results = {}
    for case, cinfo in loocv_cases.items():
        if cinfo["skipped"]:
            loocv_results[case] = cinfo
            continue
        train_r = [r for r in metadata_rows if r["case_id"] != case]
        test_r = [r for r in metadata_rows if r["case_id"] == case]

        train_idx_cv = [metadata_rows.index(r) for r in train_r]
        test_idx_cv = [metadata_rows.index(r) for r in test_r]

        Xt, yt = X[train_idx_cv], y[train_idx_cv]
        Xte, yte = X[test_idx_cv], y[test_idx_cv]

        try:
            lr_cv = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
            Xt_s = StandardScaler().fit_transform(Xt)
            Xte_s = StandardScaler().fit(Xt).transform(Xte)
            lr_cv.fit(Xt_s, yt)
            pred = lr_cv.predict(Xte_s)
            proba = lr_cv.predict_proba(Xte_s)
            acc = accuracy_score(yte, pred)
            f1m = f1_score(yte, pred, average="macro", zero_division=0)
            loocv_results[case] = {
                "n_train": len(train_r), "n_test": len(test_r),
                "accuracy": round(float(acc), 4), "f1_macro": round(float(f1m), 4),
                "test_classes": dict(Counter(int(v) for v in yte)),
            }
        except Exception as e:
            loocv_results[case] = {"n_train": len(train_r), "n_test": len(test_r), "error": str(e)}

    # ── Save best model ─────────────────────────────────────────
    best_path = MODEL_DIR / "student_judge_v2_best.joblib"
    try:
        import joblib
        joblib.dump({"model": lr, "scaler": scaler, "model_name": "LogisticRegression",
                     "feature_names": FEATURE_ALLOWLIST,
                     "classes": label_names}, best_path)
        print(f"[DONE] Best model saved: {best_path}")
    except ImportError:
        print("[WARN] joblib not installed, model not saved")

    # ── Build report ────────────────────────────────────────────
    report = {
        "step": "4_student_v2_training",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PILOT — insufficient data for production training",

        "data": {
            "balanced_dataset": len(records),
            "matched_to_evidence": matched,
            "unmatched": len(unmatched_sigs),
            "features": len(FEATURE_ALLOWLIST),
            "feature_names": FEATURE_ALLOWLIST,
            "excluded_features": EXCLUDED,
            "class_distribution": dict(Counter(int(v) for v in y)),
            "labels": label_names,
        },

        "split": {
            "method": "deterministic_hash_by_case_id_event_id",
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "train_classes": dict(Counter(int(v) for v in y_train)),
            "val_classes": dict(Counter(int(v) for v in y_val)),
            "test_classes": dict(Counter(int(v) for v in y_test)),
            "val_single_class": val_single,
            "test_single_class": test_single,
        },

        "models": models_results,
        "lightgbm": lgb_result,
        "loocv": loocv_results,

        "evaluation_kind": "pseudo_label_benchmark",
        "note": "This is a PILOT study. Only 38 samples (8 match, 30 mismatch, 0 uncertain). "
                "All silver labels are simulate mode. Metrics are against pseudo-labels, "
                "not human gold. val/test may be single-class due to small sample size. "
                "Do NOT present accuracy as performance conclusion.",
        "warnings": [
            "uncertain=0 — binary classification only",
            "8 match samples — well below 30 minimum",
            "38 total samples — too few for meaningful generalization",
            "All simulate labels — not real LLM",
        ] if len(dict(Counter(int(v) for v in y))) < 3 else [],
    }

    report_path = REPORT_DIR / "step4_training_report_v2.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Feature importance
    fi_path = REPORT_DIR / "feature_importance_v2.json"
    with fi_path.open("w", encoding="utf-8") as f:
        json.dump({"RandomForest": models_results["RandomForest"]["feature_importances"]}, f, ensure_ascii=False, indent=2)

    # LOOCV
    loocv_path = REPORT_DIR / "loocv_report_v2.json"
    with loocv_path.open("w", encoding="utf-8") as f:
        json.dump(loocv_results, f, ensure_ascii=False, indent=2)

    # ── Print ───────────────────────────────────────────────────
    print(f"\n=== Step 4: Student v2 Training ===")
    print(f"PILOT status — 38 samples, 8 match, 30 mismatch, 0 uncertain")
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"Val single-class: {val_single}, Test single-class: {test_single}")
    for mname, mres in models_results.items():
        vm = mres["val_metrics"]
        acc = vm.get('accuracy')
        bal = vm.get('balanced_accuracy')
        brier = vm.get('brier')
        print(f"  {mname:25s} val_acc={acc if acc else 'nan'} "
              f"val_bal={bal if bal else 'nan'} "
              f"warn={vm.get('single_class_warning', False)}")
    print(f"[DONE] {report_path} ({report_path.stat().st_size} bytes)")
    print(f"[DONE] {fi_path} ({fi_path.stat().st_size} bytes)")
    print(f"[DONE] {loocv_path} ({loocv_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
