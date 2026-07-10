import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
        msg = f"{op}: {reason}"
        self.fallback_reasons.append(msg)
        self.op_devices[op] = {"device": "cpu_fallback", "detail": reason}


@dataclass
class ManifestEntry:
    artifact_name: str
    artifact_type: str
    status: str
    output_path: str
    input_files: str
    row_count: int
    notes: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _as_bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


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
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return None


_REQUIRED_SELECTION_ROWS: Dict[str, Dict[str, str]] = {
    "paper_fig01": {"preferred_family": "bar"},
    "paper_fig02": {"preferred_family": "line"},
    "paper_fig03": {"preferred_family": "bar"},
    "paper_fig04": {"preferred_family": "heatmap"},
    "paper_fig05": {"preferred_family": "line"},
    "paper_fig08": {"preferred_family": "scatter"},
}


def _validate_chart_selection_matrix(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path),
        "status": "ok",
        "issues": [],
        "checked_rows": sorted(_REQUIRED_SELECTION_ROWS.keys()),
    }
    if not path.exists():
        result["status"] = "missing"
        result["issues"].append("selection_matrix_not_found")
        return result

    df = _read_csv(path)
    if df is None or df.empty:
        result["status"] = "invalid"
        result["issues"].append("selection_matrix_empty")
        return result

    required_cols = {"chart_id", "preferred_family", "d3_family", "d3_url", "input_data"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        result["status"] = "invalid"
        result["issues"].append(f"missing_columns:{','.join(missing_cols)}")
        return result

    row_map = {}
    for _, row in df.iterrows():
        cid = str(row.get("chart_id", "")).strip()
        if cid:
            row_map[cid] = row

    for chart_id, spec in _REQUIRED_SELECTION_ROWS.items():
        row = row_map.get(chart_id)
        if row is None:
            result["issues"].append(f"{chart_id}:missing_row")
            continue
        preferred = str(row.get("preferred_family", "")).strip().lower()
        d3_family = str(row.get("d3_family", "")).strip().lower()
        d3_url = str(row.get("d3_url", "")).strip()
        input_data = str(row.get("input_data", "")).strip()
        expected_family = str(spec["preferred_family"]).strip().lower()
        if preferred != expected_family:
            result["issues"].append(f"{chart_id}:preferred_family={preferred} expected={expected_family}")
        if d3_family and d3_family != expected_family:
            result["issues"].append(f"{chart_id}:d3_family={d3_family} expected={expected_family}")
        if not d3_url:
            result["issues"].append(f"{chart_id}:empty_d3_url")
        if not input_data:
            result["issues"].append(f"{chart_id}:empty_input_data")

    if result["issues"]:
        result["status"] = "invalid"
    return result


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

    fallback_reason = "cuda_unavailable"
    return DeviceContext(
        requested=requested,
        resolved="cpu",
        torch_available=True,
        cuda_available=False,
        cuda_device_count=0,
        cuda_device_name="",
        reason=fallback_reason,
        fallback_reasons=[fallback_reason] if requested == "cuda" else [],
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
    if arr.size == 0:
        edges = np.linspace(vmin, vmax, bins + 1, dtype=np.float32)
        counts = np.zeros((bins,), dtype=np.float32)
        device_ctx.record_op(op_name, "cpu", "empty_input")
        return counts, edges

    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin >= vmax:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 1.0
        if vmin >= vmax:
            vmax = vmin + 1.0

    if device_ctx.resolved == "cuda" and TORCH_AVAILABLE:
        try:
            t = torch.as_tensor(arr, dtype=torch.float32, device="cuda")  # type: ignore[union-attr]
            counts = torch.histc(t, bins=bins, min=vmin, max=vmax).detach().cpu().numpy()  # type: ignore[union-attr]
            edges = np.linspace(vmin, vmax, bins + 1, dtype=np.float32)
            device_ctx.record_op(op_name, "cuda", f"n={int(arr.size)} bins={bins}")
            return counts, edges
        except Exception as exc:
            device_ctx.record_fallback(op_name, f"torch_histc_failed: {exc}")

    counts, edges = np.histogram(arr, bins=bins, range=(vmin, vmax))
    device_ctx.record_op(op_name, "cpu", f"n={int(arr.size)} bins={bins}")
    return counts.astype(np.float32), edges.astype(np.float32)


def _gpu_scale_matrix(
    values: np.ndarray,
    device_ctx: DeviceContext,
    gpu_chunk_size: int,
    op_name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        device_ctx.record_op(op_name, "cpu", "empty_matrix")
        return arr

    if device_ctx.resolved == "cuda" and TORCH_AVAILABLE:
        try:
            t = torch.as_tensor(arr, dtype=torch.float32, device="cuda")  # type: ignore[union-attr]
            n_rows, n_cols = t.shape
            max_abs = torch.zeros((n_cols,), dtype=torch.float32, device="cuda")  # type: ignore[union-attr]
            step = max(1, int(gpu_chunk_size))
            for start in range(0, int(n_rows), step):
                chunk = t[start : start + step].abs()
                if chunk.numel() == 0:
                    continue
                max_abs = torch.maximum(max_abs, chunk.max(dim=0).values)  # type: ignore[union-attr]
            max_abs = torch.where(max_abs < 1e-6, torch.ones_like(max_abs), max_abs)  # type: ignore[union-attr]
            scaled = torch.clamp(t / max_abs, -1.0, 1.0)  # type: ignore[union-attr]
            out = scaled.detach().cpu().numpy()
            device_ctx.record_op(op_name, "cuda", f"rows={n_rows} cols={n_cols} chunk={step}")
            return out
        except Exception as exc:
            device_ctx.record_fallback(op_name, f"gpu_scale_failed: {exc}")

    max_abs_cpu = np.max(np.abs(arr), axis=0, keepdims=True)
    max_abs_cpu[max_abs_cpu < 1e-6] = 1.0
    out_cpu = np.clip(arr / max_abs_cpu, -1.0, 1.0)
    device_ctx.record_op(op_name, "cpu", f"rows={arr.shape[0]} cols={arr.shape[1]}")
    return out_cpu


def _write_manifest_csv(path: Path, entries: Sequence[ManifestEntry]) -> None:
    df = pd.DataFrame([asdict(x) for x in entries])
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _add_manifest(
    entries: List[ManifestEntry],
    artifact_name: str,
    artifact_type: str,
    status: str,
    output_path: Path,
    input_files: Sequence[Path],
    row_count: int = 0,
    notes: str = "",
) -> None:
    entries.append(
        ManifestEntry(
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            status=status,
            output_path=str(output_path),
            input_files=";".join(str(p) for p in input_files),
            row_count=int(row_count),
            notes=notes,
        )
    )


def _collect_run_summary(
    run_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_name, df in run_frames.items():
        total = int(len(df))
        if total == 0:
            continue
        ok_count = total
        if "status" in df.columns:
            ok_count = int(df["status"].astype(str).str.lower().eq("ok").sum())
        success_rate = float(ok_count / max(1, total))

        def _mean_col(col: str) -> float:
            if col not in df.columns:
                return float("nan")
            return float(pd.to_numeric(df[col], errors="coerce").mean())

        rows.append(
            {
                "run_name": run_name,
                "num_cases": total,
                "ok_cases": ok_count,
                "success_rate": success_rate,
                "mean_verified_p_match_mean": _mean_col("verified_p_match_mean"),
                "mean_align_avg_candidates": _mean_col("align_avg_candidates"),
                "mean_elapsed_sec": _mean_col("elapsed_sec"),
                "mean_actions_count": _mean_col("actions_count"),
                "mean_pose_track_count": _mean_col("pose_track_count"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["success_rate", "mean_verified_p_match_mean"], ascending=[False, False])
    return out


def _build_case_delta(mainline_df: pd.DataFrame, behavior_df: pd.DataFrame) -> pd.DataFrame:
    left = mainline_df.copy()
    right = behavior_df.copy()
    left = left.rename(columns={c: f"{c}_mainline" for c in left.columns if c != "case_id"})
    right = right.rename(columns={c: f"{c}_behavior" for c in right.columns if c != "case_id"})
    merged = pd.merge(left, right, on="case_id", how="inner")

    delta_pairs = [
        ("verified_p_match_mean", "d_verified_p_match_mean"),
        ("align_avg_candidates", "d_align_avg_candidates"),
        ("actions_count", "d_actions_count"),
        ("event_queries_count", "d_event_queries_count"),
        ("verified_count", "d_verified_count"),
        ("pose_track_count", "d_pose_track_count"),
        ("elapsed_sec", "d_elapsed_sec"),
    ]
    for base_col, out_col in delta_pairs:
        col_m = f"{base_col}_mainline"
        col_b = f"{base_col}_behavior"
        if col_m in merged.columns and col_b in merged.columns:
            v_m = pd.to_numeric(merged[col_m], errors="coerce")
            v_b = pd.to_numeric(merged[col_b], errors="coerce")
            merged[out_col] = v_b - v_m
        else:
            merged[out_col] = np.nan

    if "status_mainline" in merged.columns and "status_behavior" in merged.columns:
        merged["status_changed"] = merged["status_mainline"].astype(str) != merged["status_behavior"].astype(str)
    else:
        merged["status_changed"] = False

    if "view_code_behavior" in merged.columns:
        merged["view_code"] = merged["view_code_behavior"]
    elif "view_code_mainline" in merged.columns:
        merged["view_code"] = merged["view_code_mainline"]
    else:
        merged["view_code"] = ""
    merged = merged.sort_values(by=["view_code", "case_id"])
    return merged


def _plot_batch_compare_bar(df: pd.DataFrame, out_path: Path) -> int:
    if df.empty:
        return 0
    plot_df = df.copy()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        ("success_rate", "Success Rate"),
        ("mean_verified_p_match_mean", "Mean Verified p_match"),
        ("mean_align_avg_candidates", "Mean Align Candidates"),
        ("mean_elapsed_sec", "Mean Elapsed (s)"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics):
        vals = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)
        local_df = pd.DataFrame({"run_name": plot_df["run_name"], "value": vals})
        sns.barplot(
            data=local_df,
            x="run_name",
            y="value",
            hue="run_name",
            dodge=False,
            legend=False,
            ax=ax,
            palette="tab10",
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        if col != "mean_elapsed_sec":
            ax.set_ylim(bottom=0.0)
        for idx, v in enumerate(vals):
            if math.isnan(float(v)):
                continue
            ax.text(idx, float(v), f"{float(v):.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Batch-Level Comparison Across Runs", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return int(len(plot_df))


def _plot_verified_score_hist(
    run_frames: Dict[str, pd.DataFrame],
    out_path: Path,
    device_ctx: DeviceContext,
) -> int:
    bins = 20
    all_rows = 0
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    for run_name, df in run_frames.items():
        if "verified_p_match_mean" not in df.columns:
            continue
        values = pd.to_numeric(df["verified_p_match_mean"], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if values.size == 0:
            continue
        counts, edges = _gpu_histogram(
            values=values,
            bins=bins,
            vmin=0.0,
            vmax=1.0,
            device_ctx=device_ctx,
            op_name=f"verified_hist_{run_name}",
        )
        centers = (edges[:-1] + edges[1:]) / 2.0
        density = counts / max(1.0, float(counts.sum()))
        ax.plot(centers, density, marker="o", linewidth=1.5, label=run_name)
        all_rows += int(values.size)
    ax.set_title("Distribution of Verified p_match Mean")
    ax.set_xlabel("verified_p_match_mean")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return all_rows


def _plot_case_delta_heatmap(
    delta_df: pd.DataFrame,
    out_path: Path,
    device_ctx: DeviceContext,
    gpu_chunk_size: int,
) -> int:
    def _blur_matrix_rows(arr: np.ndarray, passes: int = 2) -> np.ndarray:
        if arr.ndim != 2 or arr.shape[0] < 3:
            return arr
        out = np.asarray(arr, dtype=np.float32)
        kernel = np.asarray([0.25, 0.50, 0.25], dtype=np.float32)
        num_pass = max(1, int(passes))
        for _ in range(num_pass):
            pad = np.pad(out, ((1, 1), (0, 0)), mode="edge")
            out = kernel[0] * pad[:-2, :] + kernel[1] * pad[1:-1, :] + kernel[2] * pad[2:, :]
        return out

    metric_cols = [
        "d_verified_p_match_mean",
        "d_align_avg_candidates",
        "d_actions_count",
        "d_event_queries_count",
        "d_verified_count",
        "d_pose_track_count",
    ]
    existing_cols = [c for c in metric_cols if c in delta_df.columns]
    if delta_df.empty or not existing_cols:
        return 0

    plot_df = delta_df[["case_id"] + existing_cols].copy()
    if "d_verified_p_match_mean" in plot_df.columns:
        abs_score = pd.to_numeric(plot_df["d_verified_p_match_mean"], errors="coerce").abs().fillna(0.0)
        plot_df = plot_df.assign(_abs_verified=abs_score).sort_values(by="_abs_verified", ascending=False).drop(columns=["_abs_verified"])
    for c in existing_cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce").fillna(0.0)
    matrix_raw = plot_df[existing_cols].to_numpy(dtype=np.float32)
    matrix_scaled = _gpu_scale_matrix(
        values=matrix_raw,
        device_ctx=device_ctx,
        gpu_chunk_size=gpu_chunk_size,
        op_name="case_delta_heatmap_scale",
    )
    matrix_blur = _blur_matrix_rows(matrix_scaled, passes=2)

    n_rows = int(matrix_blur.shape[0])
    height = max(7.0, min(16.0, 0.24 * float(n_rows) + 4.0))
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(13, height))
    vmax = float(np.nanpercentile(np.abs(matrix_blur), 95)) if matrix_blur.size else 1.0
    if not np.isfinite(vmax) or vmax <= 1e-6:
        vmax = 1.0
    im = ax.imshow(
        matrix_blur,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Scaled Delta (blurred)")

    ax.set_title("Blurred Matrix Heatmap: BehaviorAug - Mainline (Case Delta)")
    ax.set_xticks(np.arange(len(existing_cols)))
    ax.set_xticklabels(existing_cols, rotation=20, ha="right")
    max_labels = 14
    step = max(1, int(math.ceil(max(1, n_rows) / max_labels)))
    y_ticks = np.arange(0, n_rows, step, dtype=int)
    case_labels = plot_df["case_id"].astype(str).tolist()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([case_labels[i] for i in y_ticks], fontsize=8)
    ax.set_xlabel("Delta metrics")
    ax.set_ylabel("Case ID (sampled labels)")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return int(len(plot_df))


def _parse_epoch_csvs(root: Path, raw: str) -> List[Path]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return [_resolve_path(root, v) for v in values]


def _build_epoch_table(epoch_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_name, df in epoch_frames.items():
        if df.empty:
            continue
        ep = pd.to_numeric(df.get("epoch"), errors="coerce")
        p = pd.to_numeric(df.get("metrics/precision(B)"), errors="coerce")
        r = pd.to_numeric(df.get("metrics/recall(B)"), errors="coerce")
        m = pd.to_numeric(df.get("metrics/mAP50(B)"), errors="coerce")
        if ep.isna().all():
            continue

        def _best(series: pd.Series) -> Tuple[float, int]:
            if series.isna().all():
                return float("nan"), -1
            idx = int(series.idxmax())
            val = float(series.loc[idx])
            epoch_val = _safe_int(ep.loc[idx], -1)
            return val, epoch_val

        p_val, p_ep = _best(p)
        r_val, r_ep = _best(r)
        m_val, m_ep = _best(m)
        rows.append(
            {
                "run_name": run_name,
                "num_epochs": int(ep.notna().sum()),
                "best_precision": p_val,
                "best_precision_epoch": p_ep,
                "best_recall": r_val,
                "best_recall_epoch": r_ep,
                "best_mAP50": m_val,
                "best_mAP50_epoch": m_ep,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["best_mAP50", "best_precision"], ascending=[False, False])
    return out


def _plot_epoch_three_lines(epoch_frames: Dict[str, pd.DataFrame], out_path: Path) -> int:
    if not epoch_frames:
        return 0
    valid_items = []
    for run_name, df in epoch_frames.items():
        required = ["epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"]
        if not all(col in df.columns for col in required):
            continue
        local = df.copy()
        for c in required:
            local[c] = pd.to_numeric(local[c], errors="coerce")
        local = local.dropna(subset=required)
        if local.empty:
            continue
        valid_items.append((run_name, local))
    if not valid_items:
        return 0

    n = len(valid_items)
    fig, axes = plt.subplots(n, 1, figsize=(13, max(4.5, n * 3.8)), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (run_name, df) in zip(axes, valid_items):
        ax.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", linewidth=1.5)
        ax.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", linewidth=1.5)
        ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", linewidth=1.8)
        ax.set_title(f"Epoch Curves ({run_name})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Epoch Three-Line Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return int(sum(len(x[1]) for x in valid_items))


def _collect_experiment_summary(
    exp_a_metrics: Optional[Dict[str, Any]],
    exp_b_metrics: Optional[Dict[str, Any]],
    exp_c_csv: Optional[pd.DataFrame],
    exp_d_metrics: Optional[Dict[str, Any]],
    exp_e_metrics: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if exp_a_metrics:
        bc = exp_a_metrics.get("baseline_comparison", {})
        for mode in ("fixed", "adaptive_uq"):
            m = bc.get(mode, {})
            rows.append(
                {
                    "experiment": "exp_a_uq_align",
                    "setting": mode,
                    "precision": _safe_float(m.get("alignment_precision")),
                    "recall": _safe_float(m.get("alignment_recall_at_1")),
                    "f1": float("nan"),
                    "ece": float("nan"),
                    "brier": float("nan"),
                    "auroc": float("nan"),
                    "alignment_recall_at_1": _safe_float(m.get("alignment_recall_at_1")),
                    "alignment_precision": _safe_float(m.get("alignment_precision")),
                    "notes": str(exp_a_metrics.get("data_mode", "")),
                }
            )

    if exp_b_metrics:
        mm = exp_b_metrics.get("metrics", {})
        for setting in ("no_uq_gate", "uq_gate", "calibrated_uq_gate"):
            m = mm.get(setting, {})
            rows.append(
                {
                    "experiment": "exp_b_reliability_calibration",
                    "setting": setting,
                    "precision": _safe_float(m.get("Precision")),
                    "recall": _safe_float(m.get("Recall")),
                    "f1": _safe_float(m.get("F1")),
                    "ece": _safe_float(m.get("ECE")),
                    "brier": _safe_float(m.get("Brier")),
                    "auroc": float("nan"),
                    "alignment_recall_at_1": float("nan"),
                    "alignment_precision": float("nan"),
                    "notes": str(exp_b_metrics.get("data_mode", "")),
                }
            )

    if exp_c_csv is not None and not exp_c_csv.empty:
        tmp = exp_c_csv.copy()
        for _, row in tmp.iterrows():
            rows.append(
                {
                    "experiment": "exp_c_negative_sampling",
                    "setting": str(row.get("setting", "")),
                    "precision": _safe_float(row.get("Precision")),
                    "recall": _safe_float(row.get("Recall")),
                    "f1": _safe_float(row.get("F1")),
                    "ece": _safe_float(row.get("ECE")),
                    "brier": _safe_float(row.get("Brier")),
                    "auroc": _safe_float(row.get("AUROC")),
                    "alignment_recall_at_1": float("nan"),
                    "alignment_precision": float("nan"),
                    "notes": "csv",
                }
            )

    if exp_d_metrics:
        mm = exp_d_metrics.get("metrics", {})
        for setting in ("rule", "embedding", "hybrid"):
            m = mm.get(setting, {})
            rows.append(
                {
                    "experiment": "exp_d_semantic_embedding",
                    "setting": setting,
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": _safe_float(m.get("F1")),
                    "ece": _safe_float(m.get("ECE")),
                    "brier": _safe_float(m.get("Brier")),
                    "auroc": float("nan"),
                    "alignment_recall_at_1": float("nan"),
                    "alignment_precision": float("nan"),
                    "notes": str(exp_d_metrics.get("data_mode", "")),
                }
            )

    if exp_e_metrics:
        mm = exp_e_metrics.get("metrics", {})
        for setting, key in (
            ("action_only", "action_only"),
            ("action_plus_object_evidence", "action_plus_object_evidence"),
        ):
            m = mm.get(key, {})
            rows.append(
                {
                    "experiment": "exp_e_object_evidence",
                    "setting": setting,
                    "precision": _safe_float(m.get("ambiguity_subset_precision")),
                    "recall": _safe_float(m.get("ambiguity_subset_recall")),
                    "f1": _safe_float(m.get("ambiguity_subset_f1")),
                    "ece": float("nan"),
                    "brier": float("nan"),
                    "auroc": float("nan"),
                    "alignment_recall_at_1": float("nan"),
                    "alignment_precision": float("nan"),
                    "notes": str(exp_e_metrics.get("data_mode", "")),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["experiment", "setting"])
    return out


def _plot_experiment_compare_bar(exp_df: pd.DataFrame, out_path: Path) -> int:
    if exp_df.empty:
        return 0
    plot_df = exp_df.copy()
    plot_df["label"] = plot_df["experiment"].astype(str) + ":" + plot_df["setting"].astype(str)
    metric = "f1"
    if pd.to_numeric(plot_df["f1"], errors="coerce").notna().sum() < 2:
        metric = "alignment_recall_at_1"
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        return 0

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(12, 0.75 * len(plot_df)), 6))
    sns.barplot(data=plot_df, x="label", y=metric, hue="experiment", dodge=False, ax=ax)
    ax.set_title(f"Experiment Comparison ({metric})")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, max(1.0, float(plot_df[metric].max()) * 1.1))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="best", fontsize=8)
    for i, v in enumerate(plot_df[metric].tolist()):
        ax.text(i, float(v), f"{float(v):.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return int(len(plot_df))


def _plot_alignment_noise_curve(df: pd.DataFrame, out_path: Path) -> int:
    if df.empty:
        return 0
    required = {"mode", "offset_sec", "alignment_recall_at_1"}
    if not required.issubset(set(df.columns)):
        return 0
    plot_df = df.copy()
    plot_df["offset_sec"] = pd.to_numeric(plot_df["offset_sec"], errors="coerce")
    plot_df["alignment_recall_at_1"] = pd.to_numeric(plot_df["alignment_recall_at_1"], errors="coerce")
    if "mean_temporal_overlap" in plot_df.columns:
        plot_df["mean_temporal_overlap"] = pd.to_numeric(plot_df["mean_temporal_overlap"], errors="coerce")
    plot_df = plot_df.dropna(subset=["offset_sec", "alignment_recall_at_1"])
    if plot_df.empty:
        return 0

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.lineplot(
        data=plot_df,
        x="offset_sec",
        y="alignment_recall_at_1",
        hue="mode",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Alignment Recall@1 vs Time Offset")
    axes[0].set_xlabel("offset_sec")
    axes[0].set_ylabel("alignment_recall_at_1")
    axes[0].set_ylim(0.0, 1.05)

    if "mean_temporal_overlap" in plot_df.columns:
        sns.lineplot(
            data=plot_df,
            x="offset_sec",
            y="mean_temporal_overlap",
            hue="mode",
            marker="o",
            ax=axes[1],
            legend=False,
        )
    else:
        axes[1].axis("off")
    axes[1].set_title("Mean Temporal Overlap vs Time Offset")
    axes[1].set_xlabel("offset_sec")
    axes[1].set_ylabel("mean_temporal_overlap")
    axes[1].set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return int(len(plot_df))


def _plot_reliability_bins(df: pd.DataFrame, out_path: Path) -> int:
    if df.empty:
        return 0
    required = {"method", "uq_bin"}
    if not required.issubset(set(df.columns)):
        return 0
    plot_df = df.copy()
    plot_df["uq_bin"] = (
        plot_df["uq_bin"]
        .astype(str)
        .str.replace("low_uq", "low")
        .str.replace("mid_uq", "mid")
        .str.replace("high_uq", "high")
    )
    plot_df["uq_bin"] = pd.Categorical(plot_df["uq_bin"], categories=["low", "mid", "high"], ordered=True)
    for col in ("F1", "ECE", "Brier", "Precision", "Recall"):
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.sort_values(by=["method", "uq_bin"])
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if "F1" in plot_df.columns and plot_df["F1"].notna().any():
        sns.lineplot(data=plot_df, x="uq_bin", y="F1", hue="method", marker="o", ax=axes[0])
        axes[0].set_title("F1 Across UQ Bins")
        axes[0].set_ylim(0.0, 1.05)
    else:
        axes[0].axis("off")

    if "ECE" in plot_df.columns and plot_df["ECE"].notna().any():
        sns.lineplot(data=plot_df, x="uq_bin", y="ECE", hue="method", marker="o", ax=axes[1])
        axes[1].set_title("ECE Across UQ Bins")
        axes[1].set_ylim(bottom=0.0)
    else:
        axes[1].axis("off")

    for ax in axes:
        if ax.has_data():
            ax.set_xlabel("uq_bin")
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return int(len(plot_df))


def _plot_confusion_heatmap(confusion_obj: Dict[str, Any], out_path: Path) -> int:
    cm = confusion_obj.get("confusion_matrix", {})
    labels = cm.get("labels", [])
    matrix = cm.get("matrix", [])
    if not isinstance(labels, list) or not isinstance(matrix, list):
        return 0
    if not labels or not matrix:
        return 0
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        return 0
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        arr,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_title("Verifier Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return int(arr.size)


def _set_plot_style() -> None:
    sns.set_palette("Set2")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#2a2a2a"
    plt.rcParams["axes.titleweight"] = "bold"


def main() -> None:
    build_started_at = _now_iso()
    parser = argparse.ArgumentParser(description="Build paper figures/tables from existing experiment outputs.")
    parser.add_argument("--root", default=".", type=str)
    parser.add_argument("--out_chart_dir", default="docs/assets/charts/paper_auto", type=str)
    parser.add_argument("--out_table_dir", default="docs/assets/tables/paper_auto", type=str)

    parser.add_argument(
        "--mainline_csv",
        default="paper_experiments/real_cases/random6_20260420_mainline_v3_case_metrics.csv",
        type=str,
    )
    parser.add_argument(
        "--behavior_csv",
        default="paper_experiments/real_cases/random6_20260420_behavior_aug_v1_case_metrics.csv",
        type=str,
    )
    parser.add_argument(
        "--yolo_csv",
        default="paper_experiments/run_logs/random6_20260420_yolo11x_object_case_metrics.csv",
        type=str,
    )
    parser.add_argument(
        "--baseline_csv",
        default="paper_experiments/run_logs/random6_20260420_case_metrics.csv",
        type=str,
    )
    parser.add_argument(
        "--mainline_v2_csv",
        default="paper_experiments/run_logs/random6_20260420_mainline_v2_case_metrics.csv",
        type=str,
    )

    parser.add_argument("--exp_a_noise_csv", default="output/paper_experiments/exp_a_uq_align/alignment_noise_curve.csv", type=str)
    parser.add_argument("--exp_b_reliability_bins_csv", default="output/paper_experiments/exp_b_reliability_calibration/reliability_bins.csv", type=str)
    parser.add_argument("--exp_c_metrics_csv", default="output/paper_experiments/exp_c_negative_sampling/metrics_compare.csv", type=str)
    parser.add_argument("--exp_d_scores_csv", default="output/paper_experiments/exp_d_semantic_embedding/text_score_compare.csv", type=str)
    parser.add_argument("--exp_e_pairs_csv", default="output/paper_experiments/exp_e_object_evidence/ambiguity_pairs_report.csv", type=str)

    parser.add_argument("--exp_a_metrics_json", default="output/paper_experiments/exp_a_uq_align/metrics.json", type=str)
    parser.add_argument("--exp_b_metrics_json", default="output/paper_experiments/exp_b_reliability_calibration/metrics.json", type=str)
    parser.add_argument("--exp_d_metrics_json", default="output/paper_experiments/exp_d_semantic_embedding/metrics.json", type=str)
    parser.add_argument("--exp_e_metrics_json", default="output/paper_experiments/exp_e_object_evidence/metrics.json", type=str)
    parser.add_argument(
        "--confusion_json",
        default="output/paper_experiments/exp_dual_verifier_reliability/verifier_eval_report.json",
        type=str,
    )
    parser.add_argument(
        "--epoch_csvs",
        default="runs/detect/case_yolo_train/results.csv,runs/detect/train3/results.csv,runs/detect/train/results.csv",
        type=str,
    )

    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu_chunk_size", default=4096, type=int)
    parser.add_argument("--force_cpu_for_matplotlib", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail when any required input is missing.")
    parser.add_argument(
        "--selection_matrix_csv",
        default="docs/assets/tables/paper_d3_selected/chart_selection_matrix.csv",
        type=str,
        help="Selection matrix used to validate paper chart family alignment.",
    )
    parser.add_argument(
        "--enforce_selection_matrix",
        action="store_true",
        help="Fail fast when chart_selection_matrix consistency check fails.",
    )
    args = parser.parse_args()

    try:
        global sns
        import seaborn as sns
    except ImportError as exc:
        parser.error(f"paper figure generation requires seaborn: {exc}")

    root = Path(args.root).resolve()
    chart_dir = _resolve_path(root, args.out_chart_dir)
    table_dir = _resolve_path(root, args.out_table_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if args.force_cpu_for_matplotlib:
        # matplotlib is already rendered with Agg backend on CPU.
        pass

    selection_matrix_path = _resolve_path(root, args.selection_matrix_csv)
    selection_validation = _validate_chart_selection_matrix(selection_matrix_path)
    selection_invalid = selection_validation.get("status") != "ok"
    if selection_invalid and (args.strict or args.enforce_selection_matrix):
        raise ValueError(f"chart_selection_matrix validation failed: {selection_validation}")

    _set_plot_style()
    device_ctx = _choose_device(args.device)
    manifest_entries: List[ManifestEntry] = []
    missing_inputs: List[str] = []
    if selection_invalid:
        missing_inputs.append(str(selection_matrix_path))
    _add_manifest(
        manifest_entries,
        artifact_name="validation_chart_selection_matrix",
        artifact_type="validation",
        status=str(selection_validation.get("status", "unknown")),
        output_path=selection_matrix_path,
        input_files=[selection_matrix_path],
        row_count=int(len(selection_validation.get("checked_rows", []))),
        notes=";".join(str(x) for x in selection_validation.get("issues", [])),
    )

    run_sources = {
        "mainline_v3": _resolve_path(root, args.mainline_csv),
        "behavior_aug_v1": _resolve_path(root, args.behavior_csv),
        "yolo11x_object": _resolve_path(root, args.yolo_csv),
        "random6_baseline": _resolve_path(root, args.baseline_csv),
        "mainline_v2_failed": _resolve_path(root, args.mainline_v2_csv),
    }

    run_frames: Dict[str, pd.DataFrame] = {}
    for name, path in run_sources.items():
        df = _read_csv(path)
        if df is None:
            missing_inputs.append(str(path))
            _add_manifest(
                manifest_entries,
                artifact_name=f"input_run_{name}",
                artifact_type="input",
                status="missing",
                output_path=path,
                input_files=[path],
                row_count=0,
                notes="run csv not found",
            )
            continue
        run_frames[name] = df
        _add_manifest(
            manifest_entries,
            artifact_name=f"input_run_{name}",
            artifact_type="input",
            status="ok",
            output_path=path,
            input_files=[path],
            row_count=int(len(df)),
            notes="loaded",
        )

    if args.strict and missing_inputs:
        raise FileNotFoundError(f"strict mode missing inputs: {missing_inputs}")

    # tbl01 + fig01
    tbl01 = _collect_run_summary(run_frames)
    tbl01_path = table_dir / "tbl01_run_summary.csv"
    tbl01.to_csv(tbl01_path, index=False, encoding="utf-8-sig")
    _add_manifest(manifest_entries, "tbl01_run_summary", "table", "ok", tbl01_path, list(run_sources.values()), len(tbl01))

    fig01_path = chart_dir / "fig01_batch_compare_bar.png"
    fig01_rows = _plot_batch_compare_bar(tbl01, fig01_path)
    _add_manifest(
        manifest_entries,
        "fig01_batch_compare_bar",
        "figure",
        "ok" if fig01_rows > 0 else "skipped",
        fig01_path,
        [tbl01_path],
        fig01_rows,
        "requires non-empty run summary",
    )

    # fig02 verified histogram
    fig02_path = chart_dir / "fig02_verified_score_hist.png"
    fig02_rows = _plot_verified_score_hist(run_frames, fig02_path, device_ctx)
    _add_manifest(
        manifest_entries,
        "fig02_verified_score_hist",
        "figure",
        "ok" if fig02_rows > 0 else "skipped",
        fig02_path,
        list(run_sources.values()),
        fig02_rows,
        "uses verified_p_match_mean",
    )

    # tbl02 + fig03
    mainline_df = run_frames.get("mainline_v3")
    behavior_df = run_frames.get("behavior_aug_v1")
    tbl02_path = table_dir / "tbl02_case_delta_mainline_vs_behavior.csv"
    fig03_path = chart_dir / "fig03_case_delta_heatmap.png"
    if mainline_df is not None and behavior_df is not None:
        delta_df = _build_case_delta(mainline_df, behavior_df)
        delta_df.to_csv(tbl02_path, index=False, encoding="utf-8-sig")
        _add_manifest(
            manifest_entries,
            "tbl02_case_delta_mainline_vs_behavior",
            "table",
            "ok",
            tbl02_path,
            [run_sources["mainline_v3"], run_sources["behavior_aug_v1"]],
            len(delta_df),
        )
        fig03_rows = _plot_case_delta_heatmap(
            delta_df=delta_df,
            out_path=fig03_path,
            device_ctx=device_ctx,
            gpu_chunk_size=int(args.gpu_chunk_size),
        )
        _add_manifest(
            manifest_entries,
            "fig03_case_delta_heatmap",
            "figure",
            "ok" if fig03_rows > 0 else "skipped",
            fig03_path,
            [tbl02_path],
            fig03_rows,
        )
    else:
        _add_manifest(
            manifest_entries,
            "tbl02_case_delta_mainline_vs_behavior",
            "table",
            "skipped",
            tbl02_path,
            [run_sources["mainline_v3"], run_sources["behavior_aug_v1"]],
            0,
            "missing mainline/behavior input",
        )
        _add_manifest(
            manifest_entries,
            "fig03_case_delta_heatmap",
            "figure",
            "skipped",
            fig03_path,
            [tbl02_path],
            0,
            "missing delta table",
        )

    # epoch curves + table
    epoch_paths = _parse_epoch_csvs(root, args.epoch_csvs)
    epoch_frames: Dict[str, pd.DataFrame] = {}
    for p in epoch_paths:
        df = _read_csv(p)
        if df is None:
            missing_inputs.append(str(p))
            _add_manifest(
                manifest_entries,
                artifact_name=f"input_epoch_{p.name}",
                artifact_type="input",
                status="missing",
                output_path=p,
                input_files=[p],
                row_count=0,
                notes="epoch csv missing",
            )
            continue
        run_name = p.parent.name if p.parent.name else p.stem
        epoch_frames[run_name] = df
        _add_manifest(
            manifest_entries,
            artifact_name=f"input_epoch_{run_name}",
            artifact_type="input",
            status="ok",
            output_path=p,
            input_files=[p],
            row_count=len(df),
            notes="epoch csv loaded",
        )

    tbl04 = _build_epoch_table(epoch_frames)
    tbl04_path = table_dir / "tbl04_epoch_keypoints.csv"
    tbl04.to_csv(tbl04_path, index=False, encoding="utf-8-sig")
    _add_manifest(manifest_entries, "tbl04_epoch_keypoints", "table", "ok", tbl04_path, epoch_paths, len(tbl04))

    fig04_path = chart_dir / "fig04_epoch_three_lines.png"
    fig04_rows = _plot_epoch_three_lines(epoch_frames, fig04_path)
    _add_manifest(
        manifest_entries,
        "fig04_epoch_three_lines",
        "figure",
        "ok" if fig04_rows > 0 else "skipped",
        fig04_path,
        epoch_paths,
        fig04_rows,
    )

    # experiment summary table + compare fig
    exp_a_metrics_path = _resolve_path(root, args.exp_a_metrics_json)
    exp_b_metrics_path = _resolve_path(root, args.exp_b_metrics_json)
    exp_d_metrics_path = _resolve_path(root, args.exp_d_metrics_json)
    exp_e_metrics_path = _resolve_path(root, args.exp_e_metrics_json)
    exp_c_csv_path = _resolve_path(root, args.exp_c_metrics_csv)
    exp_d_scores_path = _resolve_path(root, args.exp_d_scores_csv)
    exp_e_pairs_path = _resolve_path(root, args.exp_e_pairs_csv)

    exp_a_metrics = _read_json(exp_a_metrics_path)
    exp_b_metrics = _read_json(exp_b_metrics_path)
    exp_d_metrics = _read_json(exp_d_metrics_path)
    exp_e_metrics = _read_json(exp_e_metrics_path)
    exp_c_df = _read_csv(exp_c_csv_path)
    exp_d_scores_df = _read_csv(exp_d_scores_path)
    exp_e_pairs_df = _read_csv(exp_e_pairs_path)

    for src in [
        exp_a_metrics_path,
        exp_b_metrics_path,
        exp_c_csv_path,
        exp_d_metrics_path,
        exp_d_scores_path,
        exp_e_metrics_path,
        exp_e_pairs_path,
    ]:
        if not src.exists():
            missing_inputs.append(str(src))

    tbl03 = _collect_experiment_summary(
        exp_a_metrics=exp_a_metrics,
        exp_b_metrics=exp_b_metrics,
        exp_c_csv=exp_c_df,
        exp_d_metrics=exp_d_metrics,
        exp_e_metrics=exp_e_metrics,
    )
    if exp_d_scores_df is not None and not exp_d_scores_df.empty:
        exp_d_scores_df = exp_d_scores_df.copy()
        exp_d_scores_df["experiment"] = "exp_d_semantic_embedding"
    if exp_e_pairs_df is not None and not exp_e_pairs_df.empty:
        exp_e_pairs_df = exp_e_pairs_df.copy()
        exp_e_pairs_df["experiment"] = "exp_e_object_evidence"
    tbl03_path = table_dir / "tbl03_experiment_summary.csv"
    tbl03.to_csv(tbl03_path, index=False, encoding="utf-8-sig")
    _add_manifest(
        manifest_entries,
        "tbl03_experiment_summary",
        "table",
        "ok",
        tbl03_path,
        [
            exp_a_metrics_path,
            exp_b_metrics_path,
            exp_c_csv_path,
            exp_d_metrics_path,
            exp_e_metrics_path,
        ],
        len(tbl03),
    )

    fig05_path = chart_dir / "fig05_exp_ablation_compare_bar.png"
    fig05_rows = _plot_experiment_compare_bar(tbl03, fig05_path)
    _add_manifest(
        manifest_entries,
        "fig05_exp_ablation_compare_bar",
        "figure",
        "ok" if fig05_rows > 0 else "skipped",
        fig05_path,
        [tbl03_path],
        fig05_rows,
    )

    # fig06 alignment curve
    exp_a_noise_path = _resolve_path(root, args.exp_a_noise_csv)
    exp_a_noise_df = _read_csv(exp_a_noise_path)
    fig06_path = chart_dir / "fig06_alignment_noise_curve.png"
    fig06_rows = 0
    if exp_a_noise_df is not None:
        fig06_rows = _plot_alignment_noise_curve(exp_a_noise_df, fig06_path)
    else:
        missing_inputs.append(str(exp_a_noise_path))
    _add_manifest(
        manifest_entries,
        "fig06_alignment_noise_curve",
        "figure",
        "ok" if fig06_rows > 0 else "skipped",
        fig06_path,
        [exp_a_noise_path],
        fig06_rows,
    )

    # fig07 reliability bins
    exp_b_bins_path = _resolve_path(root, args.exp_b_reliability_bins_csv)
    exp_b_bins_df = _read_csv(exp_b_bins_path)
    fig07_path = chart_dir / "fig07_reliability_bins_curve.png"
    fig07_rows = 0
    if exp_b_bins_df is not None:
        fig07_rows = _plot_reliability_bins(exp_b_bins_df, fig07_path)
    else:
        missing_inputs.append(str(exp_b_bins_path))
    _add_manifest(
        manifest_entries,
        "fig07_reliability_bins_curve",
        "figure",
        "ok" if fig07_rows > 0 else "skipped",
        fig07_path,
        [exp_b_bins_path],
        fig07_rows,
    )

    # fig08 confusion heatmap
    confusion_path = _resolve_path(root, args.confusion_json)
    confusion_obj = _read_json(confusion_path)
    fig08_path = chart_dir / "fig08_confusion_heatmap.png"
    fig08_rows = 0
    if confusion_obj is not None:
        fig08_rows = _plot_confusion_heatmap(confusion_obj, fig08_path)
    else:
        missing_inputs.append(str(confusion_path))
    _add_manifest(
        manifest_entries,
        "fig08_confusion_heatmap",
        "figure",
        "ok" if fig08_rows > 0 else "skipped",
        fig08_path,
        [confusion_path],
        fig08_rows,
    )

    # tbl05 plot data manifest
    tbl05_path = table_dir / "tbl05_plot_data_manifest.csv"
    _write_manifest_csv(tbl05_path, manifest_entries)

    # build log json
    output_files = [m.output_path for m in manifest_entries if m.artifact_type in {"figure", "table"} and m.status == "ok"]
    build_log = {
        "started_at": build_started_at,
        "root": str(root),
        "requested_device": args.device,
        "resolved_device": device_ctx.resolved,
        "device_context": asdict(device_ctx),
        "force_cpu_for_matplotlib": bool(args.force_cpu_for_matplotlib),
        "gpu_chunk_size": int(args.gpu_chunk_size),
        "strict_mode": bool(args.strict),
        "enforce_selection_matrix": bool(args.enforce_selection_matrix),
        "output_chart_dir": str(chart_dir),
        "output_table_dir": str(table_dir),
        "missing_inputs": sorted(set(missing_inputs)),
        "selection_matrix_validation": selection_validation,
        "manifest_csv": str(tbl05_path),
        "generated_artifact_count": len(output_files),
        "generated_artifacts": output_files,
        "ended_at": _now_iso(),
    }
    log_path = table_dir / "plot_build_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(build_log, f, ensure_ascii=False, indent=2)

    print(f"[DONE] charts dir: {chart_dir}")
    print(f"[DONE] tables dir: {table_dir}")
    print(f"[DONE] manifest: {tbl05_path}")
    print(f"[DONE] build log: {log_path}")
    print(f"[INFO] resolved device: {device_ctx.resolved} (requested={args.device})")
    print(f"[INFO] generated artifacts: {len(output_files)}")
    if missing_inputs:
        print(f"[WARN] missing inputs: {len(set(missing_inputs))}")


if __name__ == "__main__":
    main()
