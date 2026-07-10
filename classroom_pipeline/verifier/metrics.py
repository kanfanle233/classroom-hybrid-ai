import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


LABELS: Tuple[str, str, str] = ("match", "uncertain", "mismatch")


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _label_from_value(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "match" if value else "mismatch"
    if isinstance(value, (int, float)):
        if float(value) >= 0.5:
            return "match"
        return "mismatch"
    if isinstance(value, str):
        v = value.strip().lower()
        alias = {
            "match": "match",
            "matched": "match",
            "positive": "match",
            "pos": "match",
            "1": "match",
            "true": "match",
            "uncertain": "uncertain",
            "unknown": "uncertain",
            "mismatch": "mismatch",
            "negative": "mismatch",
            "neg": "mismatch",
            "0": "mismatch",
            "false": "mismatch",
            "not_match": "mismatch",
            "non_match": "mismatch",
        }
        if v in alias:
            return alias[v]
    return None


def normalize_label(value: Any, default: str = "mismatch") -> str:
    parsed = _label_from_value(value)
    if parsed is not None:
        return parsed
    return default


def parse_reference_label(row: Dict[str, Any], target_field: str = "auto") -> Tuple[str, str]:
    if target_field != "auto":
        parsed = _label_from_value(row.get(target_field))
        if parsed is not None:
            return parsed, target_field
        return normalize_label(row.get("label")), f"{target_field}:fallback_label"

    preferred = [
        "target_label",
        "ground_truth_label",
        "gt_label",
        "label_gt",
        "reference_label",
        "truth_label",
        "y_true",
    ]
    for key in preferred:
        parsed = _label_from_value(row.get(key))
        if parsed is not None:
            return parsed, key
    return normalize_label(row.get("label")), "self_label_fallback"


def parse_predicted_label(row: Dict[str, Any]) -> str:
    return normalize_label(row.get("label", row.get("match_label", "mismatch")))


def parse_score(row: Dict[str, Any], score_field: str = "reliability_score") -> float:
    if score_field in row:
        return clamp01(row.get(score_field), default=0.0)
    return clamp01(row.get("p_match"), default=0.0)


def parse_prob(row: Dict[str, Any], field: str = "p_match") -> float:
    return clamp01(row.get(field), default=0.0)


def label_distribution(labels: Sequence[str]) -> Dict[str, int]:
    out = {k: 0 for k in LABELS}
    for lbl in labels:
        out[normalize_label(lbl)] += 1
    return out


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Dict[str, int]]:
    cm = {t: {p: 0 for p in LABELS} for t in LABELS}
    n = min(len(y_true), len(y_pred))
    for i in range(n):
        t = normalize_label(y_true[i])
        p = normalize_label(y_pred[i])
        cm[t][p] += 1
    return cm


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def metrics_from_confusion_matrix(cm: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    total = 0
    correct = 0
    per_label: Dict[str, Dict[str, float]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in LABELS:
        tp = float(cm[label][label])
        fp = float(sum(cm[other][label] for other in LABELS if other != label))
        fn = float(sum(cm[label][other] for other in LABELS if other != label))
        support = int(sum(cm[label][other] for other in LABELS))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        total += support
        correct += int(tp)

    n_labels = float(len(LABELS))
    return {
        "precision": macro_precision / n_labels,
        "recall": macro_recall / n_labels,
        "f1": macro_f1 / n_labels,
        "macro_f1": macro_f1 / n_labels,
        "accuracy": _safe_div(float(correct), float(max(total, 1))),
        "per_label": per_label,
    }


def build_matrix_list(cm: Dict[str, Dict[str, int]]) -> List[List[int]]:
    return [[int(cm[t][p]) for p in LABELS] for t in LABELS]


def score_to_label(score: float, match_threshold: float, uncertain_threshold: float) -> str:
    if score >= match_threshold:
        return "match"
    if score >= uncertain_threshold:
        return "uncertain"
    return "mismatch"


def threshold_sweep(
    *,
    y_true: Sequence[str],
    scores: Sequence[float],
    thresholds: Sequence[float],
    uncertain_margin: float = 0.20,
) -> List[Dict[str, Any]]:
    sweep: List[Dict[str, Any]] = []
    if not thresholds:
        thresholds = [i / 20.0 for i in range(6, 19)]  # [0.30, 0.90]
    n = min(len(y_true), len(scores))
    y_true_norm = [normalize_label(y_true[i]) for i in range(n)]
    scores_norm = [clamp01(scores[i], default=0.0) for i in range(n)]

    for t in thresholds:
        t_match = clamp01(t, default=0.60)
        t_uncertain = clamp01(t_match - uncertain_margin, default=0.40)
        y_pred = [score_to_label(s, t_match, t_uncertain) for s in scores_norm]
        cm = confusion_matrix(y_true_norm, y_pred)
        m = metrics_from_confusion_matrix(cm)
        sweep.append(
            {
                "match_threshold": t_match,
                "uncertain_threshold": t_uncertain,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "label_distribution": label_distribution(y_pred),
            }
        )
    return sweep


def pick_best_sweep(sweep: Sequence[Dict[str, Any]], key: str = "f1") -> Dict[str, Any]:
    if not sweep:
        return {"match_threshold": 0.60, "uncertain_threshold": 0.40, "f1": 0.0}
    best = max(sweep, key=lambda x: float(x.get(key, 0.0)))
    return dict(best)


def probability_consistency(
    rows: Sequence[Dict[str, Any]],
    *,
    p_match_field: str = "p_match",
    p_mismatch_field: str = "p_mismatch",
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    if not rows:
        return {
            "mean_abs_error": 0.0,
            "max_abs_error": 0.0,
            "violations": 0,
            "tolerance": float(tolerance),
        }
    errs: List[float] = []
    violations = 0
    for row in rows:
        p_match = parse_prob(row, p_match_field)
        p_mismatch = parse_prob(row, p_mismatch_field)
        err = abs((p_match + p_mismatch) - 1.0)
        errs.append(err)
        if err > tolerance:
            violations += 1
    return {
        "mean_abs_error": float(sum(errs) / len(errs)),
        "max_abs_error": float(max(errs)),
        "violations": int(violations),
        "tolerance": float(tolerance),
    }


def binary_target_from_label(label: str) -> int:
    return 1 if normalize_label(label) == "match" else 0


def brier_score(probs: Sequence[float], targets: Sequence[int]) -> float:
    if not probs:
        return 0.0
    n = min(len(probs), len(targets))
    if n <= 0:
        return 0.0
    loss = 0.0
    for i in range(n):
        p = clamp01(probs[i], default=0.0)
        y = 1.0 if int(targets[i]) == 1 else 0.0
        loss += (p - y) ** 2
    return float(loss / n)


def ece_and_bins(
    probs: Sequence[float],
    targets: Sequence[int],
    *,
    num_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    if num_bins <= 0:
        num_bins = 10
    n = min(len(probs), len(targets))
    if n <= 0:
        return 0.0, []

    bins: List[List[Tuple[float, int]]] = [[] for _ in range(num_bins)]
    for i in range(n):
        p = clamp01(probs[i], default=0.0)
        y = 1 if int(targets[i]) == 1 else 0
        idx = min(num_bins - 1, int(p * num_bins))
        bins[idx].append((p, y))

    ece = 0.0
    bin_stats: List[Dict[str, Any]] = []
    for i in range(num_bins):
        lo = i / num_bins
        hi = (i + 1) / num_bins
        items = bins[i]
        if not items:
            bin_stats.append(
                {
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": 0,
                    "acc": 0.0,
                    "conf": 0.0,
                }
            )
            continue
        conf = float(sum(p for p, _ in items) / len(items))
        acc = float(sum(y for _, y in items) / len(items))
        frac = len(items) / n
        ece += abs(conf - acc) * frac
        bin_stats.append(
            {
                "bin_left": lo,
                "bin_right": hi,
                "count": int(len(items)),
                "acc": acc,
                "conf": conf,
            }
        )
    return float(ece), bin_stats


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return float(math.log(p / (1.0 - p)))


def apply_temperature(probs: Sequence[float], temperature: float) -> List[float]:
    t = max(1e-6, float(temperature))
    out: List[float] = []
    for p in probs:
        logit = _logit(clamp01(p, default=0.0))
        out.append(_sigmoid(logit / t))
    return out


def fit_temperature_brier(
    probs: Sequence[float],
    targets: Sequence[int],
    *,
    candidates: Optional[Sequence[float]] = None,
) -> Tuple[float, List[float]]:
    if not probs:
        return 1.0, []
    if candidates is None:
        candidates = [0.50 + 0.05 * i for i in range(0, 51)]  # [0.50, 3.00]
    best_t = 1.0
    best_probs = list(probs)
    best_brier = brier_score(best_probs, targets)
    for t in candidates:
        scaled = apply_temperature(probs, float(t))
        bs = brier_score(scaled, targets)
        if bs < best_brier:
            best_brier = bs
            best_t = float(t)
            best_probs = scaled
    return best_t, best_probs
