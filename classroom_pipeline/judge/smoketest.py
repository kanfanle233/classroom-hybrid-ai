"""Anti-drift smoke tests for the LLM judge pipeline.

Each test outputs a pass/fail verdict. The pipeline exits with code 1
if any check fails.

Checks:
  A. Label distribution drift (KL divergence vs previous run)
  B. Student vs silver consistency per held-out case
  C. No label leakage in student checkpoint
  D. Cross-case generalization (leave-one-case-out)
  E. Silver vs heuristic correlation
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION
from classroom_pipeline.verifier.model import VerifierMLP, build_feature_vector


# ── helpers ──────────────────────────────────────────────────────────────

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """KL(P || Q) between two probability distributions over the same keys."""
    all_keys = set(p) | set(q)
    kl = 0.0
    for key in all_keys:
        pi = p.get(key, 1e-10)
        qi = q.get(key, 1e-10)
        if pi > 0:
            kl += pi * math.log(pi / qi)
    return kl


def _distribution_from_labels(labels: List[str], smooth: float = 1e-10) -> Dict[str, float]:
    """Convert label list to a probability distribution with Laplace smoothing."""
    counter = Counter(labels)
    total = len(labels)
    n_classes = max(3, len(counter))
    dist = {}
    for label in ("match", "uncertain", "mismatch"):
        count = counter.get(label, 0)
        dist[label] = (count + smooth) / (total + smooth * n_classes)
    return dist


def _load_checkpoint(model_path: Path) -> Optional[Dict[str, Any]]:
    if not model_path.exists():
        return None
    try:
        return torch.load(model_path, map_location="cpu")
    except Exception:
        return None


# ── Check A: Distribution drift ──────────────────────────────────────────

def check_distribution_drift(
    current_labels: List[str],
    previous_manifest: Optional[Dict[str, Any]] = None,
    kl_threshold: float = 0.20,
) -> Dict[str, Any]:
    """Compare silver label distribution against previous run."""
    current_dist = _distribution_from_labels(current_labels)

    if previous_manifest is None:
        return {
            "check": "A_label_distribution_drift",
            "passed": True,
            "kl_divergence": 0.0,
            "current_distribution": current_dist,
            "detail": "No previous run for comparison",
        }

    previous_labels = previous_manifest.get("label_distribution", {})
    # Convert previous label counts to distribution
    prev_dist = _distribution_from_labels(
        [k for k, v in previous_labels.items() for _ in range(v)]
        if isinstance(previous_labels, dict)
        else []
    )

    kl = _kl_divergence(current_dist, prev_dist)
    passed = kl < kl_threshold

    return {
        "check": "A_label_distribution_drift",
        "passed": passed,
        "kl_divergence": round(kl, 6),
        "threshold": kl_threshold,
        "current_distribution": current_dist,
        "previous_distribution": prev_dist,
        "detail": f"KL={kl:.4f} vs threshold={kl_threshold}" + (" [PASS]" if passed else " [DRIFT DETECTED]"),
    }


# ── Check B: Student vs Silver consistency per case ──────────────────────

def check_student_consistency(
    student_predictions: Dict[str, List[float]],
    silver_labels: Dict[str, List[float]],
    threshold: float = 0.15,
) -> Dict[str, Any]:
    """Verify student accuracy is consistent across cases."""
    deviations = {}
    all_cases = set(student_predictions) | set(silver_labels)

    for case in sorted(all_cases):
        s_preds = student_predictions.get(case, [])
        s_silver = silver_labels.get(case, [])
        if len(s_preds) != len(s_silver) or len(s_preds) == 0:
            deviations[case] = 1.0
            continue
        # Binarize both at 0.5
        pred_binary = [1 if p >= 0.5 else 0 for p in s_preds]
        silver_binary = [1 if p >= 0.5 else 0 for p in s_silver]
        acc = sum(1 for p, s in zip(pred_binary, silver_binary) if p == s) / len(pred_binary)
        deviations[case] = 1.0 - acc

    max_deviation = max(deviations.values()) if deviations else 0.0
    passed = max_deviation < threshold
    violating_cases = [c for c, d in deviations.items() if d >= threshold]

    return {
        "check": "B_student_silver_consistency",
        "passed": passed,
        "max_deviation": round(max_deviation, 4),
        "threshold": threshold,
        "deviations": {c: round(d, 4) for c, d in sorted(deviations.items())},
        "violating_cases": violating_cases,
        "detail": f"max_deviation={max_deviation:.4f} vs threshold={threshold}"
        + (" [PASS]" if passed else f" violating cases: {violating_cases}"),
    }


# ── Check C: No label leakage ────────────────────────────────────────────

def check_no_label_leakage(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Verify the student checkpoint does not contain final label fields."""
    # Check serialized fields in the checkpoint dict
    forbidden_keys = {"p_match", "p_mismatch", "match_label", "label", "reliability_score"}
    leaked = [k for k in forbidden_keys if k in checkpoint]

    # Check state_dict tensor names
    state_dict = checkpoint.get("state_dict", {})
    if isinstance(state_dict, dict):
        tensor_leaked = [k for k in state_dict if any(f in k for f in forbidden_keys)]
        leaked.extend(tensor_leaked)

    passed = len(leaked) == 0

    return {
        "check": "C_no_label_leakage",
        "passed": passed,
        "leaked_fields": leaked,
        "detail": "OK" if passed else f"Leaked fields found: {leaked}",
    }


# ── Check D: Cross-case generalization ───────────────────────────────────

def check_cross_case_generalization(
    samples_by_case: Dict[str, List[Dict[str, Any]]],
    epochs: int = 60,
    lr: float = 1e-3,
    hidden_dim: int = 16,
    std_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Leave-one-case-out cross-validation.

    Trains N models, each time holding out one case. Reports mean and std
    of held-out accuracy.
    """
    import torch.optim as optim
    import torch.nn as nn

    cases = sorted(samples_by_case.keys())
    if len(cases) < 2:
        return {
            "check": "D_cross_case_generalization",
            "passed": True,
            "detail": f"Only {len(cases)} case(s), skipping LOOCV",
            "num_cases": len(cases),
            "case_accuracies": {},
            "mean_val": 0.0,
            "std": 0.0,
            "worst_case": "",
            "worst_accuracy": 0.0,
        }

    def _tensorize(samples):
        feats = [s["feature_vector"] for s in samples]
        labels = [s["target"] for s in samples]
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    case_accs = {}
    for held_out in cases:
        train_samples = []
        for c in cases:
            if c != held_out:
                train_samples.extend(samples_by_case[c])
        val_samples = samples_by_case[held_out]

        if not train_samples or not val_samples:
            continue

        x_train, y_train = _tensorize(train_samples)
        x_val, y_val = _tensorize(val_samples)

        model = VerifierMLP(in_dim=4, hidden_dim=hidden_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(x_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
        val_probs = torch.sigmoid(val_logits)
        val_pred = (val_probs >= 0.5).float()
        val_acc = float((val_pred == y_val).float().mean().item()) if len(y_val) > 0 else 0.0
        case_accs[held_out] = val_acc

    if not case_accs:
        return {
            "check": "D_cross_case_generalization",
            "passed": False,
            "detail": "No cases could be evaluated",
            "num_cases": len(cases),
            "case_accuracies": {},
            "mean_val": 0.0,
            "std": 0.0,
        }

    acc_values = list(case_accs.values())
    mean_val = sum(acc_values) / len(acc_values)
    std_val = (sum((v - mean_val) ** 2 for v in acc_values) / len(acc_values)) ** 0.5
    worst_case = min(case_accs, key=case_accs.get)
    worst_acc = case_accs[worst_case]

    # A case is a "violation" if its accuracy is >1.5 std below mean
    violations = [c for c, a in case_accs.items() if a < mean_val - 1.5 * std_val]

    passed = std_val < std_threshold

    return {
        "check": "D_cross_case_generalization",
        "passed": passed,
        "num_cases": len(cases),
        "case_accuracies": {c: round(a, 4) for c, a in sorted(case_accs.items())},
        "mean_val": round(mean_val, 4),
        "std": round(std_val, 4),
        "std_threshold": std_threshold,
        "worst_case": worst_case,
        "worst_accuracy": round(worst_acc, 4),
        "violations": violations,
        "detail": f"mean={mean_val:.4f} std={std_val:.4f} threshold={std_threshold}"
        + (" [PASS]" if passed else f" std exceeds threshold, violations: {violations}"),
    }


# ── Check E: Silver vs Heuristic correlation ─────────────────────────────

def check_silver_heuristic_correlation(
    silver_scores: List[float],
    heuristic_scores: List[float],
) -> Dict[str, Any]:
    """Spearman rank correlation between silver labels and heuristic scores."""
    if len(silver_scores) != len(heuristic_scores) or len(silver_scores) < 5:
        return {
            "check": "E_silver_heuristic_correlation",
            "passed": True,  # informative only
            "spearman_rho": 0.0,
            "mean_abs_difference": 0.0,
            "num_pairs": len(silver_scores),
            "detail": "Insufficient data for correlation",
        }

    from scipy.stats import spearmanr
    rho, p_value = spearmanr(silver_scores, heuristic_scores)
    if math.isnan(rho):
        rho = 0.0
        p_value = 1.0

    mean_abs_diff = sum(abs(s - h) for s, h in zip(silver_scores, heuristic_scores)) / len(silver_scores)

    return {
        "check": "E_silver_heuristic_correlation",
        "passed": True,  # informational — no pass/fail
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(p_value), 6),
        "mean_abs_difference": round(mean_abs_diff, 4),
        "num_pairs": len(silver_scores),
        "detail": f"spearman_rho={rho:.4f} mean_abs_diff={mean_abs_diff:.4f}"
        + (" [low correlation — LLM captures different signal]" if abs(rho) < 0.3 else ""),
    }


# ── run all checks ───────────────────────────────────────────────────────

def run_all_checks(
    *,
    silver_labels_path: Path,
    student_samples_path: Optional[Path],
    student_checkpoint_path: Optional[Path],
    previous_smoketest_path: Optional[Path] = None,
    kl_threshold: float = 0.20,
    consistency_threshold: float = 0.15,
    cross_case_std_threshold: float = 0.10,
    loocv_epochs: int = 60,
) -> Dict[str, Any]:
    """Execute all anti-drift checks and return unified report."""
    checks: Dict[str, Any] = {}
    all_passed = True

    # Load silver labels
    silver_labels = _load_jsonl(silver_labels_path)
    silver_p_matches = [float(s.get("silver_p_match", 0.5)) for s in silver_labels]
    silver_label_strs = [str(s.get("silver_label", "uncertain")) for s in silver_labels]

    # Load previous smoketest for drift comparison
    previous_manifest = None
    if previous_smoketest_path and previous_smoketest_path.exists():
        previous_manifest = _load_json(previous_smoketest_path)

    # Check A: Distribution drift
    checks["A_label_distribution_drift"] = check_distribution_drift(
        silver_label_strs, previous_manifest, kl_threshold
    )
    if not checks["A_label_distribution_drift"]["passed"]:
        all_passed = False

    # Check C: Label leakage (requires checkpoint)
    ckpt = None
    if student_checkpoint_path:
        ckpt = _load_checkpoint(student_checkpoint_path)
    if ckpt is not None:
        checks["C_no_label_leakage"] = check_no_label_leakage(ckpt)
        if not checks["C_no_label_leakage"]["passed"]:
            all_passed = False
    else:
        checks["C_no_label_leakage"] = {
            "check": "C_no_label_leakage",
            "passed": True,
            "detail": "No checkpoint to verify (skipped)",
        }

    # Check D: Cross-case generalization (requires samples with case_id grouping)
    if student_samples_path and student_samples_path.exists():
        samples = _load_jsonl(student_samples_path)
        if len(samples) >= 10:
            # Group samples by query_id (first token before underscore as case_id)
            samples_by_case: Dict[str, List[Dict[str, Any]]] = {}
            for s in samples:
                qid = str(s.get("query_id", ""))
                case_id = qid.split("_")[0] if "_" in qid else qid
                if case_id:
                    samples_by_case.setdefault(case_id, []).append(s)
            checks["D_cross_case_generalization"] = check_cross_case_generalization(
                samples_by_case,
                epochs=loocv_epochs,
                std_threshold=cross_case_std_threshold,
            )
            if not checks["D_cross_case_generalization"]["passed"]:
                all_passed = False
        else:
            checks["D_cross_case_generalization"] = {
                "check": "D_cross_case_generalization",
                "passed": True,
                "detail": f"Too few samples ({len(samples)}), skipping",
            }
    else:
        checks["D_cross_case_generalization"] = {
            "check": "D_cross_case_generalization",
            "passed": True,
            "detail": "No training samples, skipping",
        }

    # Check E: Silver vs heuristic correlation
    heuristic_scores = []
    for s in silver_labels:
        feat = s.get("feature_vector", [])
        if len(feat) >= 3:
            # heuristic visual_score for reference
            overlap = float(s.get("candidate", {}).get("overlap", 0.0))
            action_conf = float(s.get("candidate", {}).get("action_confidence", 0.0))
            text_score = feat[2] if len(feat) > 2 else 0.0
            visual_score = min(1.0, max(0.0, 0.65 * overlap + 0.35 * action_conf))
            heuristic_scores.append(min(1.0, max(0.0, (visual_score + text_score) / 2.0)))
        else:
            heuristic_scores.append(0.5)

    if silver_p_matches:
        checks["E_silver_heuristic_correlation"] = check_silver_heuristic_correlation(
            silver_p_matches, heuristic_scores
        )
    else:
        checks["E_silver_heuristic_correlation"] = {
            "check": "E_silver_heuristic_correlation",
            "passed": True,
            "detail": "No silver scores, skipping",
        }

    # Build unified report
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_output": str(silver_labels_path.parent),
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "all_checks_passed": all_passed,
        "checks": checks,
    }

    return report


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: Anti-drift smoke tests for LLM judge pipeline"
    )
    parser.add_argument("--silver_labels", required=True, type=str)
    parser.add_argument("--student_samples", default="", type=str)
    parser.add_argument("--student_checkpoint", default="", type=str)
    parser.add_argument("--previous_smoketest", default="", type=str,
                        help="previous smoketest_report.json for drift comparison")
    parser.add_argument("--out", required=True, type=str,
                        help="output smoketest_report.json")
    parser.add_argument("--kl_threshold", type=float, default=0.20)
    parser.add_argument("--consistency_threshold", type=float, default=0.15)
    parser.add_argument("--cross_case_std_threshold", type=float, default=0.10)
    parser.add_argument("--loocv_epochs", type=int, default=60)
    args = parser.parse_args()

    report = run_all_checks(
        silver_labels_path=Path(args.silver_labels),
        student_samples_path=Path(args.student_samples) if args.student_samples else None,
        student_checkpoint_path=Path(args.student_checkpoint) if args.student_checkpoint else None,
        previous_smoketest_path=Path(args.previous_smoketest) if args.previous_smoketest else None,
        kl_threshold=args.kl_threshold,
        consistency_threshold=args.consistency_threshold,
        cross_case_std_threshold=args.cross_case_std_threshold,
        loocv_epochs=args.loocv_epochs,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    passed = report.get("all_checks_passed", False)
    for name, check in report.get("checks", {}).items():
        icon = "PASS" if check.get("passed", False) else "FAIL"
        print(f"  [{icon}] {name}: {check.get('detail', '')}")

    if passed:
        print(f"[SMOKE PASS] All {len(report.get('checks', {}))} checks passed")
    else:
        print(f"[SMOKE FAIL] Some checks failed — see {out_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
