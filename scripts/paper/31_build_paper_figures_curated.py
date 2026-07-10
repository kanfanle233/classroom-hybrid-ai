import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


@dataclass
class DeviceContext:
    requested: str
    resolved: str
    torch_available: bool
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str
    reason: str = ""
    fallback_reasons: List[str] = field(default_factory=list)
    op_devices: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def record_op(self, op: str, device: str, detail: str = "") -> None:
        self.op_devices[op] = {"device": device, "detail": detail}

    def record_fallback(self, op: str, reason: str) -> None:
        self.fallback_reasons.append(f"{op}: {reason}")
        self.op_devices[op] = {"device": "cpu_fallback", "detail": reason}


@dataclass
class ManifestEntry:
    artifact: str
    kind: str
    status: str
    output_path: str
    input_files: str
    sample_size: int
    quality_tier: str
    notes: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _choose_device(requested: str) -> DeviceContext:
    requested = requested.strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        requested = "auto"

    if not TORCH_AVAILABLE:
        return DeviceContext(
            requested=requested,
            resolved="cpu",
            torch_available=False,
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_name="",
            reason="torch_unavailable",
        )

    cuda_available = bool(torch.cuda.is_available())  # type: ignore[union-attr]
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0  # type: ignore[union-attr]
    cuda_name = ""
    if cuda_available and cuda_count > 0:
        try:
            cuda_name = str(torch.cuda.get_device_name(0))  # type: ignore[union-attr]
        except Exception:
            cuda_name = ""

    if requested == "cpu":
        return DeviceContext(
            requested=requested,
            resolved="cpu",
            torch_available=True,
            cuda_available=cuda_available,
            cuda_device_count=cuda_count,
            cuda_device_name=cuda_name,
            reason="user_forced_cpu",
        )

    if cuda_available and cuda_count > 0:
        return DeviceContext(
            requested=requested,
            resolved="cuda",
            torch_available=True,
            cuda_available=True,
            cuda_device_count=cuda_count,
            cuda_device_name=cuda_name,
            reason="cuda_available",
        )

    return DeviceContext(
        requested=requested,
        resolved="cpu",
        torch_available=True,
        cuda_available=False,
        cuda_device_count=0,
        cuda_device_name="",
        reason="cuda_unavailable",
        fallback_reasons=["cuda_unavailable"] if requested == "cuda" else [],
    )


def _gpu_histogram(
    values: np.ndarray,
    bins: int,
    vmin: float,
    vmax: float,
    device_ctx: DeviceContext,
    op_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    edges = np.linspace(vmin, vmax, bins + 1, dtype=np.float32)
    if arr.size == 0:
        device_ctx.record_op(op_name, "cpu", "empty")
        return np.zeros((bins,), dtype=np.float32), edges

    if device_ctx.resolved == "cuda" and TORCH_AVAILABLE:
        try:
            t = torch.as_tensor(arr, dtype=torch.float32, device="cuda")  # type: ignore[union-attr]
            counts = torch.histc(t, bins=bins, min=vmin, max=vmax).detach().cpu().numpy()  # type: ignore[union-attr]
            device_ctx.record_op(op_name, "cuda", f"n={arr.size}")
            return counts.astype(np.float32), edges
        except Exception as exc:
            device_ctx.record_fallback(op_name, f"torch_histc_failed: {exc}")

    counts, _ = np.histogram(arr, bins=bins, range=(vmin, vmax))
    device_ctx.record_op(op_name, "cpu", f"n={arr.size}")
    return counts.astype(np.float32), edges


def _gpu_scale_matrix(
    values: np.ndarray,
    device_ctx: DeviceContext,
    op_name: str,
    gpu_chunk_size: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        device_ctx.record_op(op_name, "cpu", "empty")
        return arr
    if device_ctx.resolved == "cuda" and TORCH_AVAILABLE:
        try:
            t = torch.as_tensor(arr, dtype=torch.float32, device="cuda")  # type: ignore[union-attr]
            m = t.shape[0]
            out = torch.empty_like(t)
            for s in range(0, m, max(1, gpu_chunk_size)):
                e = min(m, s + max(1, gpu_chunk_size))
                block = t[s:e]
                denom = torch.max(torch.abs(block), dim=0, keepdim=True).values
                denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
                out[s:e] = torch.clamp(block / denom, -1.0, 1.0)
            device_ctx.record_op(op_name, "cuda", f"shape={tuple(arr.shape)}")
            return out.detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            device_ctx.record_fallback(op_name, f"torch_scale_failed: {exc}")
    denom = np.max(np.abs(arr), axis=0, keepdims=True)
    denom[denom < 1e-8] = 1.0
    device_ctx.record_op(op_name, "cpu", f"shape={tuple(arr.shape)}")
    return np.clip(arr / denom, -1.0, 1.0).astype(np.float32)


def _set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.size"] = 11


def _bootstrap_ci(values: Sequence[float], n_boot: int = 2000, ci: float = 95.0) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(42)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    samples = arr[idx].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(samples, alpha))
    hi = float(np.percentile(samples, 100.0 - alpha))
    return mean, lo, hi


def _collect_run_frames(root: Path, paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        df = _read_csv(path)
        if df is not None:
            out[name] = df
    return out


def _build_run_metric_table(run_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_name, df in run_frames.items():
        local = df.copy()
        status_ok = local["status"].astype(str).str.lower().eq("ok") if "status" in local.columns else pd.Series([True] * len(local))
        local = local[status_ok].copy()
        if local.empty:
            rows.append(
                {
                    "run_name": run_name,
                    "n_cases_ok": 0,
                    "success_rate": 0.0,
                    "metric": "verified_p_match_mean",
                    "mean": 0.0,
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                }
            )
            continue

        metrics = [
            "verified_p_match_mean",
            "align_avg_candidates",
            "elapsed_sec",
            "event_queries_count",
        ]
        for m in metrics:
            if m not in local.columns:
                continue
            vals = pd.to_numeric(local[m], errors="coerce").dropna().to_numpy(dtype=np.float64)
            mean, lo, hi = _bootstrap_ci(vals)
            rows.append(
                {
                    "run_name": run_name,
                    "n_cases_ok": int(len(local)),
                    "success_rate": float(len(local)) / float(len(df)) if len(df) else 0.0,
                    "metric": m,
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["metric", "mean"], ascending=[True, False])
    return out


def _plot_batch_ci(metric_df: pd.DataFrame, out_path: Path) -> int:
    if metric_df.empty:
        return 0
    wanted = ["success_rate", "verified_p_match_mean", "align_avg_candidates", "elapsed_sec"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for ax, metric in zip(axes.flat, wanted):
        local = metric_df[metric_df["metric"] == metric].copy()
        if local.empty:
            ax.axis("off")
            continue
        local = local.sort_values(by="mean", ascending=False)
        x = np.arange(len(local))
        y = local["mean"].to_numpy(dtype=float)
        yerr_lo = y - local["ci95_low"].to_numpy(dtype=float)
        yerr_hi = local["ci95_high"].to_numpy(dtype=float) - y
        ax.bar(x, y, color=sns.color_palette("Set2", n_colors=len(local)), edgecolor="#333333", linewidth=0.5)
        ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="#222222", capsize=4, lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(local["run_name"].tolist(), rotation=20, ha="right")
        ax.set_title(metric)
        ax.set_ylabel("value")
        if metric != "elapsed_sec":
            ax.set_ylim(bottom=0.0)
        for i, v in enumerate(y):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Batch Comparison With 95% Bootstrap CI", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return int(metric_df["run_name"].nunique())


def _plot_verified_hist_pair(
    main_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    out_path: Path,
    device_ctx: DeviceContext,
) -> int:
    col = "verified_p_match_mean"
    if col not in main_df.columns or col not in behavior_df.columns:
        return 0
    a = pd.to_numeric(main_df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
    b = pd.to_numeric(behavior_df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if a.size == 0 and b.size == 0:
        return 0

    bins = 15
    c1, e1 = _gpu_histogram(a, bins=bins, vmin=0.0, vmax=1.0, device_ctx=device_ctx, op_name="hist_mainline_v3")
    c2, e2 = _gpu_histogram(b, bins=bins, vmin=0.0, vmax=1.0, device_ctx=device_ctx, op_name="hist_behavior_aug_v1")
    ctr1 = (e1[:-1] + e1[1:]) / 2
    ctr2 = (e2[:-1] + e2[1:]) / 2
    d1 = c1 / max(1.0, float(c1.sum()))
    d2 = c2 / max(1.0, float(c2.sum()))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(ctr1, d1, marker="o", linewidth=2.0, label="mainline_v3")
    ax.fill_between(ctr1, d1, alpha=0.15)
    ax.plot(ctr2, d2, marker="o", linewidth=2.0, label="behavior_aug_v1")
    ax.fill_between(ctr2, d2, alpha=0.15)
    ax.set_title("Histogram Density of verified_p_match_mean")
    ax.set_xlabel("verified_p_match_mean")
    ax.set_ylabel("density")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return int(a.size + b.size)


def _build_case_delta(main_df: pd.DataFrame, behavior_df: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in main_df.columns or "case_id" not in behavior_df.columns:
        return pd.DataFrame()
    m = main_df.copy().add_suffix("_main")
    b = behavior_df.copy().add_suffix("_beh")
    merged = pd.merge(m, b, left_on="case_id_main", right_on="case_id_beh", how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["case_id"] = merged["case_id_main"]
    pairs = [
        ("verified_p_match_mean", "d_verified_p_match_mean"),
        ("align_avg_candidates", "d_align_avg_candidates"),
        ("elapsed_sec", "d_elapsed_sec"),
        ("event_queries_count", "d_event_queries_count"),
        ("verified_count", "d_verified_count"),
    ]
    for src, dst in pairs:
        mcol = f"{src}_main"
        bcol = f"{src}_beh"
        if mcol in merged.columns and bcol in merged.columns:
            vm = pd.to_numeric(merged[mcol], errors="coerce")
            vb = pd.to_numeric(merged[bcol], errors="coerce")
            merged[dst] = vb - vm
    out_cols = ["case_id"] + [x[1] for x in pairs if x[1] in merged.columns]
    out = merged[out_cols].copy()
    out = out.sort_values(by="d_verified_p_match_mean", ascending=False) if "d_verified_p_match_mean" in out.columns else out
    return out


def _plot_case_delta_heatmap_curated(
    delta_df: pd.DataFrame,
    out_path: Path,
    device_ctx: DeviceContext,
    gpu_chunk_size: int,
) -> int:
    if delta_df.empty:
        return 0
    metric_cols = [c for c in delta_df.columns if c.startswith("d_")]
    if not metric_cols:
        return 0
    nz_cols: List[str] = []
    for c in metric_cols:
        vals = pd.to_numeric(delta_df[c], errors="coerce").fillna(0.0)
        if float((vals != 0).mean()) >= 0.1:
            nz_cols.append(c)
    if not nz_cols:
        return 0

    local = delta_df[["case_id"] + nz_cols].copy()
    if "d_verified_p_match_mean" in local.columns:
        local["rank_abs"] = pd.to_numeric(local["d_verified_p_match_mean"], errors="coerce").abs()
        local = local.sort_values(by="rank_abs", ascending=False).drop(columns=["rank_abs"])
    local = local.head(20).copy()
    for c in nz_cols:
        local[c] = pd.to_numeric(local[c], errors="coerce").fillna(0.0)
    mat_raw = local[nz_cols].to_numpy(dtype=np.float32)
    mat = _gpu_scale_matrix(mat_raw, device_ctx=device_ctx, op_name="heatmap_delta_scale", gpu_chunk_size=gpu_chunk_size)

    fig, ax = plt.subplots(figsize=(10.6, 8.4))
    sns.heatmap(mat, cmap="RdBu_r", center=0.0, cbar_kws={"label": "scaled delta"}, ax=ax)
    ax.set_title("Top-20 Case Delta Heatmap (BehaviorAug - Mainline)")
    xlabels = [x.replace("d_", "").replace("_", "\n") for x in nz_cols]
    ax.set_xticklabels(xlabels, rotation=0, ha="center")
    ax.set_yticklabels(local["case_id"].tolist(), fontsize=8, rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return int(len(local))


def _plot_epoch_three_lines(epoch_frames: Dict[str, pd.DataFrame], out_path: Path) -> int:
    valid: List[Tuple[str, pd.DataFrame]] = []
    required = ["epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"]
    for run_name, df in epoch_frames.items():
        if not all(c in df.columns for c in required):
            continue
        t = df.copy()
        for c in required:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        t = t.dropna(subset=required)
        if not t.empty:
            valid.append((run_name, t))
    if not valid:
        return 0
    fig, axes = plt.subplots(len(valid), 1, figsize=(12.0, max(4.2, len(valid) * 3.2)))
    if len(valid) == 1:
        axes = [axes]
    for ax, (name, t) in zip(axes, valid):
        ax.plot(t["epoch"], t["metrics/precision(B)"], label="Precision", linewidth=1.8)
        ax.plot(t["epoch"], t["metrics/recall(B)"], label="Recall", linewidth=1.8)
        ax.plot(t["epoch"], t["metrics/mAP50(B)"], label="mAP50", linewidth=2.0)
        ax.set_title(f"Epoch Curves ({name})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="lower right")
    fig.suptitle("Epoch Three-Line Curves", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return int(sum(len(x[1]) for x in valid))


def _aggregate_confusion_from_reports(output_root: Path, run_key: str) -> Tuple[np.ndarray, int, int]:
    agg = np.zeros((3, 3), dtype=np.int64)
    n_files = 0
    n_samples = 0
    for p in output_root.rglob("verifier_eval_report.json"):
        sp = str(p).replace("\\", "/")
        if f"/{run_key}/" not in sp:
            continue
        obj = _read_json(p)
        if not obj:
            continue
        cm = obj.get("confusion_matrix", {})
        mat = cm.get("matrix") if isinstance(cm, dict) else None
        if not (isinstance(mat, list) and len(mat) == 3 and all(isinstance(r, list) and len(r) == 3 for r in mat)):
            continue
        n_files += 1
        for i in range(3):
            for j in range(3):
                v = int(mat[i][j])
                agg[i, j] += v
                n_samples += v
    return agg, n_files, n_samples


def _plot_confusion_grid(confusions: Dict[str, Tuple[np.ndarray, int, int]], out_path: Path) -> int:
    valid = {k: v for k, v in confusions.items() if int(v[2]) > 0}
    if not valid:
        return 0
    labels = ["match", "uncertain", "mismatch"]
    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6), squeeze=False)
    for ax, (run_name, (mat, n_files, n_samples)) in zip(axes[0], valid.items()):
        row_sum = mat.sum(axis=1, keepdims=True).astype(np.float64)
        row_sum[row_sum == 0] = 1.0
        norm = mat / row_sum
        sns.heatmap(norm, vmin=0.0, vmax=1.0, cmap="Blues", annot=mat, fmt="d", cbar=False, ax=ax)
        ax.set_title(f"{run_name}\nfiles={n_files}, samples={n_samples}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_yticklabels(labels, rotation=0)
    fig.suptitle("Aggregated Confusion Matrix (row-normalized, count-annotated)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return int(sum(v[2] for v in valid.values()))


def _plot_ablation_facets(
    exp_c: Optional[pd.DataFrame],
    exp_d: Optional[pd.DataFrame],
    out_path: Path,
) -> int:
    has_c = exp_c is not None and not exp_c.empty
    has_d = exp_d is not None and not exp_d.empty
    if not has_c and not has_d:
        return 0
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    used = 0
    if has_c:
        t = exp_c.copy()
        rename = {
            "positive_only": "pos_only",
            "positive_plus_temporal_shift": "pos+temp_shift",
            "positive_plus_semantic_mismatch": "pos+sem_mis",
            "positive_plus_both": "pos+both",
        }
        if "setting" in t.columns:
            t["setting"] = t["setting"].astype(str).map(lambda x: rename.get(x, x))
        t["F1"] = pd.to_numeric(t["F1"], errors="coerce")
        t = t.dropna(subset=["F1"])
        sns.barplot(data=t, x="setting", y="F1", ax=axes[0], color="#4C78A8")
        axes[0].tick_params(axis="x", rotation=28)
        axes[0].set_title("Exp-C Negative Sampling (F1)")
        axes[0].set_xlabel("")
        axes[0].set_ylim(0.0, max(1.0, float(t["F1"].max()) * 1.2))
        for i, v in enumerate(t["F1"].tolist()):
            axes[0].text(i, float(v), f"{float(v):.3f}", ha="center", va="bottom", fontsize=8)
        used += len(t)
    else:
        axes[0].axis("off")

    if has_d:
        t = exp_d.copy()
        if "mode" in t.columns and "F1" in t.columns:
            t["F1"] = pd.to_numeric(t["F1"], errors="coerce")
            t = t.dropna(subset=["F1"])
            sns.barplot(data=t, x="mode", y="F1", ax=axes[1], color="#59A14F")
            axes[1].set_title("Exp-D Semantic Score Mode (F1)")
            axes[1].set_xlabel("")
            axes[1].set_ylim(0.0, max(1.0, float(t["F1"].max()) * 1.2))
            for i, v in enumerate(t["F1"].tolist()):
                axes[1].text(i, float(v), f"{float(v):.3f}", ha="center", va="bottom", fontsize=8)
            used += len(t)
        else:
            axes[1].axis("off")
    else:
        axes[1].axis("off")
    fig.suptitle("Ablation Bars (appendix-ready)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return int(used)


def _collect_data_quality_rows(
    run_metric_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    exp_bins: Optional[pd.DataFrame],
    confusion_samples: Dict[str, int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_run = int(run_metric_df["run_name"].nunique()) if not run_metric_df.empty else 0
    rows.append(
        {
            "artifact": "batch_ci_compare",
            "sample_size": n_run,
            "quality_tier": "main_text" if n_run >= 3 else "appendix_only",
            "reason": "run-level comparison across multiple runs",
        }
    )
    n_delta = int(len(delta_df))
    rows.append(
        {
            "artifact": "case_delta_heatmap_top20",
            "sample_size": n_delta,
            "quality_tier": "main_text" if n_delta >= 20 else "appendix_only",
            "reason": "paired case-level deltas",
        }
    )
    if exp_bins is not None and not exp_bins.empty and "count" in exp_bins.columns:
        cmin = int(pd.to_numeric(exp_bins["count"], errors="coerce").fillna(0).min())
        rows.append(
            {
                "artifact": "reliability_bins_curve",
                "sample_size": int(pd.to_numeric(exp_bins["count"], errors="coerce").fillna(0).sum()),
                "quality_tier": "appendix_only" if cmin < 5 else "main_text",
                "reason": f"min bin count={cmin}",
            }
        )
    for name, n in confusion_samples.items():
        rows.append(
            {
                "artifact": f"confusion_{name}",
                "sample_size": int(n),
                "quality_tier": "main_text" if n >= 100 else "appendix_only",
                "reason": "aggregated from per-case verifier_eval_report.json",
            }
        )
    return pd.DataFrame(rows)


def _write_manifest(path: Path, rows: List[ManifestEntry]) -> None:
    pd.DataFrame([asdict(r) for r in rows]).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    started_at = _now_iso()
    parser = argparse.ArgumentParser(description="Build curated paper figures after full-repo traversal audit.")
    parser.add_argument("--root", default=".", type=str)
    parser.add_argument("--out_chart_dir", default="docs/assets/charts/paper_curated", type=str)
    parser.add_argument("--out_table_dir", default="docs/assets/tables/paper_curated", type=str)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu_chunk_size", default=4096, type=int)
    parser.add_argument("--force_cpu_for_matplotlib", action="store_true")
    parser.add_argument(
        "--epoch_csvs",
        default="runs/detect/case_yolo_train/results.csv,runs/detect/train3/results.csv,runs/detect/train/results.csv",
        type=str,
    )
    args = parser.parse_args()

    try:
        global sns
        import seaborn as sns
    except ImportError as exc:
        parser.error(f"curated paper figure generation requires seaborn: {exc}")

    root = Path(args.root).resolve()
    chart_dir = _resolve_path(root, args.out_chart_dir)
    table_dir = _resolve_path(root, args.out_table_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    device_ctx = _choose_device(args.device)

    manifest: List[ManifestEntry] = []
    missing_inputs: List[str] = []

    run_paths = {
        "mainline_v3": _resolve_path(root, "paper_experiments/real_cases/random6_20260420_mainline_v3_case_metrics.csv"),
        "behavior_aug_v1": _resolve_path(root, "paper_experiments/real_cases/random6_20260420_behavior_aug_v1_case_metrics.csv"),
        "random6_baseline": _resolve_path(root, "paper_experiments/run_logs/random6_20260420_case_metrics.csv"),
        "yolo11x_object": _resolve_path(root, "paper_experiments/run_logs/random6_20260420_yolo11x_object_case_metrics.csv"),
    }
    run_frames = _collect_run_frames(root, run_paths)
    for n, p in run_paths.items():
        if n not in run_frames:
            missing_inputs.append(str(p))

    run_metric_df = _build_run_metric_table(run_frames)
    tbl_run = table_dir / "tbl01_run_metric_ci.csv"
    run_metric_df.to_csv(tbl_run, index=False, encoding="utf-8-sig")
    manifest.append(
        ManifestEntry(
            artifact="tbl01_run_metric_ci",
            kind="table",
            status="ok",
            output_path=str(tbl_run),
            input_files="; ".join(str(p) for p in run_paths.values()),
            sample_size=int(run_metric_df["run_name"].nunique()) if not run_metric_df.empty else 0,
            quality_tier="main_text",
            notes="run-level mean and bootstrap CI",
        )
    )

    fig_batch = chart_dir / "fig01_batch_compare_with_ci.png"
    n_batch = _plot_batch_ci(run_metric_df, fig_batch)
    manifest.append(
        ManifestEntry(
            artifact="fig01_batch_compare_with_ci",
            kind="figure",
            status="ok" if n_batch > 0 else "skipped",
            output_path=str(fig_batch),
            input_files=str(tbl_run),
            sample_size=n_batch,
            quality_tier="main_text" if n_batch >= 3 else "appendix_only",
            notes="drop misleading zero-fill, use CI",
        )
    )

    main_df = run_frames.get("mainline_v3")
    beh_df = run_frames.get("behavior_aug_v1")

    fig_hist = chart_dir / "fig02_verified_hist_mainline_vs_behavior.png"
    n_hist = 0
    if main_df is not None and beh_df is not None:
        n_hist = _plot_verified_hist_pair(main_df, beh_df, fig_hist, device_ctx=device_ctx)
    manifest.append(
        ManifestEntry(
            artifact="fig02_verified_hist_mainline_vs_behavior",
            kind="figure",
            status="ok" if n_hist > 0 else "skipped",
            output_path=str(fig_hist),
            input_files=f"{run_paths['mainline_v3']}; {run_paths['behavior_aug_v1']}",
            sample_size=n_hist,
            quality_tier="main_text" if n_hist >= 40 else "appendix_only",
            notes="histogram density for two key runs",
        )
    )

    delta_df = pd.DataFrame()
    if main_df is not None and beh_df is not None:
        delta_df = _build_case_delta(main_df, beh_df)
    tbl_delta = table_dir / "tbl02_case_delta_mainline_vs_behavior.csv"
    delta_df.to_csv(tbl_delta, index=False, encoding="utf-8-sig")
    manifest.append(
        ManifestEntry(
            artifact="tbl02_case_delta_mainline_vs_behavior",
            kind="table",
            status="ok" if not delta_df.empty else "skipped",
            output_path=str(tbl_delta),
            input_files=f"{run_paths['mainline_v3']}; {run_paths['behavior_aug_v1']}",
            sample_size=int(len(delta_df)),
            quality_tier="main_text" if len(delta_df) >= 20 else "appendix_only",
            notes="paired case deltas",
        )
    )

    fig_heat = chart_dir / "fig03_case_delta_heatmap_top20.png"
    n_heat = _plot_case_delta_heatmap_curated(
        delta_df=delta_df,
        out_path=fig_heat,
        device_ctx=device_ctx,
        gpu_chunk_size=int(args.gpu_chunk_size),
    )
    manifest.append(
        ManifestEntry(
            artifact="fig03_case_delta_heatmap_top20",
            kind="figure",
            status="ok" if n_heat > 0 else "skipped",
            output_path=str(fig_heat),
            input_files=str(tbl_delta),
            sample_size=n_heat,
            quality_tier="main_text" if n_heat >= 20 else "appendix_only",
            notes="top absolute delta cases only",
        )
    )

    epoch_frames: Dict[str, pd.DataFrame] = {}
    for raw in [x.strip() for x in args.epoch_csvs.split(",") if x.strip()]:
        p = _resolve_path(root, raw)
        df = _read_csv(p)
        if df is None:
            missing_inputs.append(str(p))
            continue
        epoch_frames[p.parent.name if p.parent.name else p.stem] = df
    fig_epoch = chart_dir / "fig04_epoch_three_lines_curated.png"
    n_epoch = _plot_epoch_three_lines(epoch_frames, fig_epoch)
    manifest.append(
        ManifestEntry(
            artifact="fig04_epoch_three_lines_curated",
            kind="figure",
            status="ok" if n_epoch > 0 else "skipped",
            output_path=str(fig_epoch),
            input_files="; ".join(args.epoch_csvs.split(",")),
            sample_size=n_epoch,
            quality_tier="main_text",
            notes="epoch precision/recall/mAP50 curves",
        )
    )

    exp_c = _read_csv(_resolve_path(root, "output/paper_experiments/exp_c_negative_sampling/metrics_compare.csv"))
    exp_d = _read_csv(_resolve_path(root, "output/paper_experiments/exp_d_semantic_embedding/text_score_compare.csv"))
    fig_ablation = chart_dir / "fig05_ablation_facets.png"
    n_ablation = _plot_ablation_facets(exp_c=exp_c, exp_d=exp_d, out_path=fig_ablation)
    manifest.append(
        ManifestEntry(
            artifact="fig05_ablation_facets",
            kind="figure",
            status="ok" if n_ablation > 0 else "skipped",
            output_path=str(fig_ablation),
            input_files="output/paper_experiments/exp_c_negative_sampling/metrics_compare.csv; output/paper_experiments/exp_d_semantic_embedding/text_score_compare.csv",
            sample_size=n_ablation,
            quality_tier="appendix_only",
            notes="small-sample ablation goes appendix",
        )
    )

    output_root = _resolve_path(root, "output")
    run_keys = [
        "random6_20260420_mainline_v3",
        "random6_20260420_behavior_aug_v1",
        "random6_20260420_yolo11x_object",
        "random6_20260420",
    ]
    confusions: Dict[str, Tuple[np.ndarray, int, int]] = {}
    conf_rows: List[Dict[str, Any]] = []
    for k in run_keys:
        mat, n_files, n_samples = _aggregate_confusion_from_reports(output_root=output_root, run_key=k)
        confusions[k] = (mat, n_files, n_samples)
        conf_rows.append(
            {
                "run_key": k,
                "n_case_reports": n_files,
                "n_samples": n_samples,
                "m00_match_match": int(mat[0, 0]),
                "m11_uncertain_uncertain": int(mat[1, 1]),
                "m22_mismatch_mismatch": int(mat[2, 2]),
                "m_all": int(mat.sum()),
            }
        )
    tbl_conf = table_dir / "tbl03_confusion_aggregated.csv"
    pd.DataFrame(conf_rows).to_csv(tbl_conf, index=False, encoding="utf-8-sig")
    manifest.append(
        ManifestEntry(
            artifact="tbl03_confusion_aggregated",
            kind="table",
            status="ok",
            output_path=str(tbl_conf),
            input_files="output/**/verifier_eval_report.json",
            sample_size=int(sum(r["n_samples"] for r in conf_rows)),
            quality_tier="main_text",
            notes="aggregate confusion from 30-case run directories",
        )
    )

    fig_conf = chart_dir / "fig06_confusion_aggregate_grid.png"
    n_conf = _plot_confusion_grid(confusions=confusions, out_path=fig_conf)
    manifest.append(
        ManifestEntry(
            artifact="fig06_confusion_aggregate_grid",
            kind="figure",
            status="ok" if n_conf > 0 else "skipped",
            output_path=str(fig_conf),
            input_files=str(tbl_conf),
            sample_size=n_conf,
            quality_tier="main_text" if n_conf >= 300 else "appendix_only",
            notes="row-normalized confusion + count annotations",
        )
    )

    reliability_bins = _read_csv(_resolve_path(root, "output/paper_experiments/exp_b_reliability_calibration/reliability_bins.csv"))
    quality_df = _collect_data_quality_rows(
        run_metric_df=run_metric_df,
        delta_df=delta_df,
        exp_bins=reliability_bins,
        confusion_samples={k: v[2] for k, v in confusions.items()},
    )
    tbl_quality = table_dir / "tbl04_data_quality_gate.csv"
    quality_df.to_csv(tbl_quality, index=False, encoding="utf-8-sig")
    manifest.append(
        ManifestEntry(
            artifact="tbl04_data_quality_gate",
            kind="table",
            status="ok",
            output_path=str(tbl_quality),
            input_files=f"{tbl_run}; {tbl_delta}; {tbl_conf}",
            sample_size=int(len(quality_df)),
            quality_tier="main_text",
            notes="main-text vs appendix recommendation",
        )
    )

    tbl_manifest = table_dir / "tbl05_artifact_manifest.csv"
    _write_manifest(tbl_manifest, manifest)

    log_path = table_dir / "plot_build_log_curated.json"
    build_log = {
        "started_at": started_at,
        "ended_at": _now_iso(),
        "root": str(root),
        "requested_device": args.device,
        "resolved_device": device_ctx.resolved,
        "device_context": asdict(device_ctx),
        "force_cpu_for_matplotlib": bool(args.force_cpu_for_matplotlib),
        "gpu_chunk_size": int(args.gpu_chunk_size),
        "out_chart_dir": str(chart_dir),
        "out_table_dir": str(table_dir),
        "missing_inputs": sorted(set(missing_inputs)),
        "manifest_csv": str(tbl_manifest),
        "generated_artifact_count": int(sum(1 for x in manifest if x.status == "ok")),
        "generated_artifacts": [x.output_path for x in manifest if x.status == "ok"],
    }
    log_path.write_text(json.dumps(build_log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] charts dir: {chart_dir}")
    print(f"[DONE] tables dir: {table_dir}")
    print(f"[DONE] manifest: {tbl_manifest}")
    print(f"[DONE] build log: {log_path}")
    print(f"[INFO] resolved device: {device_ctx.resolved} (requested={args.device})")
    print(f"[INFO] generated artifacts: {build_log['generated_artifact_count']}")
    if missing_inputs:
        print(f"[WARN] missing inputs: {len(set(missing_inputs))}")


if __name__ == "__main__":
    main()
