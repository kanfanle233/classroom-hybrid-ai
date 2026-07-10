import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


FEATURE_NAMES: Tuple[str, ...] = (
    "visible_ratio",
    "mean_conf",
    "prev_motion",
    "next_motion",
    "motion_delta",
    "bbox_shift",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


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


def _bbox_diag(bbox: Sequence[Any]) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 1.0
    x1, y1, x2, y2 = [_safe_float(v, 0.0) for v in bbox[:4]]
    return max(1.0, math.hypot(x2 - x1, y2 - y1))


def _bbox_center(bbox: Sequence[Any]) -> Tuple[float, float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0.0, 0.0
    x1, y1, x2, y2 = [_safe_float(v, 0.0) for v in bbox[:4]]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _iter_track_sequences(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    tracks: Dict[int, List[Dict[str, Any]]] = {}
    for row in _load_jsonl(path):
        frame_idx = int(row.get("frame", 0))
        t_sec = _safe_float(row.get("t", 0.0), 0.0)
        persons = row.get("persons", [])
        if not isinstance(persons, list):
            continue
        for person in persons:
            if not isinstance(person, dict):
                continue
            track_id = person.get("track_id")
            if not isinstance(track_id, int):
                continue
            tracks.setdefault(track_id, []).append(
                {
                    "frame": frame_idx,
                    "t": t_sec,
                    "track_id": track_id,
                    "bbox": person.get("bbox", [0, 0, 1, 1]),
                    "keypoints": person.get("keypoints", []),
                    "conf": _safe_float(person.get("conf", 0.0), 0.0),
                }
            )
    for seq in tracks.values():
        seq.sort(key=lambda x: int(x["frame"]))
    return tracks


def _joint_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    dx = _safe_float(a.get("x", 0.0), 0.0) - _safe_float(b.get("x", 0.0), 0.0)
    dy = _safe_float(a.get("y", 0.0), 0.0) - _safe_float(b.get("y", 0.0), 0.0)
    return math.hypot(dx, dy)


def _mean_joint_motion(a_kpts: Sequence[Any], b_kpts: Sequence[Any], diag: float, conf_th: float = 0.15) -> float:
    if not isinstance(a_kpts, list) or not isinstance(b_kpts, list) or len(a_kpts) != len(b_kpts):
        return 0.0
    vals: List[float] = []
    for a, b in zip(a_kpts, b_kpts):
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        if _safe_float(a.get("c", 0.0), 0.0) < conf_th and _safe_float(b.get("c", 0.0), 0.0) < conf_th:
            continue
        vals.append(_joint_distance(a, b) / max(1.0, diag))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _variance_target(prev_row: Dict[str, Any], cur_row: Dict[str, Any], next_row: Dict[str, Any], conf_th: float = 0.15) -> Optional[float]:
    prev_kpts = prev_row.get("keypoints", [])
    cur_kpts = cur_row.get("keypoints", [])
    next_kpts = next_row.get("keypoints", [])
    if not isinstance(prev_kpts, list) or not isinstance(cur_kpts, list) or not isinstance(next_kpts, list):
        return None
    if len(prev_kpts) != len(cur_kpts) or len(cur_kpts) != len(next_kpts):
        return None

    diag = _bbox_diag(cur_row.get("bbox", [0, 0, 1, 1]))
    errs: List[float] = []
    for prev_kp, cur_kp, next_kp in zip(prev_kpts, cur_kpts, next_kpts):
        if not isinstance(prev_kp, dict) or not isinstance(cur_kp, dict) or not isinstance(next_kp, dict):
            continue
        if min(
            _safe_float(prev_kp.get("c", 0.0), 0.0),
            _safe_float(cur_kp.get("c", 0.0), 0.0),
            _safe_float(next_kp.get("c", 0.0), 0.0),
        ) < conf_th:
            continue
        mu_x = 0.5 * (_safe_float(prev_kp.get("x", 0.0), 0.0) + _safe_float(next_kp.get("x", 0.0), 0.0))
        mu_y = 0.5 * (_safe_float(prev_kp.get("y", 0.0), 0.0) + _safe_float(next_kp.get("y", 0.0), 0.0))
        err_x = (_safe_float(cur_kp.get("x", 0.0), 0.0) - mu_x) / diag
        err_y = (_safe_float(cur_kp.get("y", 0.0), 0.0) - mu_y) / diag
        errs.append(err_x * err_x + err_y * err_y)
    if not errs:
        return None
    return float(max(1e-6, sum(errs) / len(errs)))


def _build_feature_row(prev_row: Dict[str, Any], cur_row: Dict[str, Any], next_row: Dict[str, Any], conf_th: float = 0.15) -> Optional[Dict[str, float]]:
    cur_kpts = cur_row.get("keypoints", [])
    if not isinstance(cur_kpts, list) or not cur_kpts:
        return None
    diag = _bbox_diag(cur_row.get("bbox", [0, 0, 1, 1]))
    visible = 0
    confs: List[float] = []
    for kp in cur_kpts:
        if not isinstance(kp, dict):
            continue
        conf = _safe_float(kp.get("c", 0.0), 0.0)
        confs.append(conf)
        if conf >= conf_th:
            visible += 1

    prev_motion = _mean_joint_motion(prev_row.get("keypoints", []), cur_kpts, diag, conf_th=conf_th)
    next_motion = _mean_joint_motion(cur_kpts, next_row.get("keypoints", []), diag, conf_th=conf_th)
    p_cx, p_cy = _bbox_center(prev_row.get("bbox", [0, 0, 1, 1]))
    c_cx, c_cy = _bbox_center(cur_row.get("bbox", [0, 0, 1, 1]))
    n_cx, n_cy = _bbox_center(next_row.get("bbox", [0, 0, 1, 1]))
    bbox_shift = 0.5 * (
        math.hypot(c_cx - p_cx, c_cy - p_cy) / max(1.0, diag)
        + math.hypot(n_cx - c_cx, n_cy - c_cy) / max(1.0, diag)
    )

    return {
        "visible_ratio": float(visible / max(1, len(cur_kpts))),
        "mean_conf": float(sum(confs) / max(1, len(confs))),
        "prev_motion": float(prev_motion),
        "next_motion": float(next_motion),
        "motion_delta": float(abs(prev_motion - next_motion)),
        "bbox_shift": float(bbox_shift),
    }


def build_variance_samples(
    pose_tracks_path: Path,
    *,
    conf_th: float = 0.15,
    max_frame_gap: int = 3,
) -> List[Dict[str, Any]]:
    tracks = _iter_track_sequences(pose_tracks_path)
    samples: List[Dict[str, Any]] = []
    sample_id = 0
    for track_id, seq in tracks.items():
        if len(seq) < 3:
            continue
        for index in range(1, len(seq) - 1):
            prev_row = seq[index - 1]
            cur_row = seq[index]
            next_row = seq[index + 1]
            if int(cur_row["frame"]) - int(prev_row["frame"]) > max_frame_gap:
                continue
            if int(next_row["frame"]) - int(cur_row["frame"]) > max_frame_gap:
                continue
            target_var = _variance_target(prev_row, cur_row, next_row, conf_th=conf_th)
            if target_var is None:
                continue
            features = _build_feature_row(prev_row, cur_row, next_row, conf_th=conf_th)
            if features is None:
                continue
            samples.append(
                {
                    "sample_id": f"var_{sample_id:07d}",
                    "track_id": int(track_id),
                    "frame": int(cur_row["frame"]),
                    "t": _safe_float(cur_row.get("t", 0.0), 0.0),
                    "target_var": float(target_var),
                    "features": [float(features[name]) for name in FEATURE_NAMES],
                    "feature_dict": features,
                }
            )
            sample_id += 1
    return samples


class VarianceHeadMLP(nn.Module):
    def __init__(self, in_dim: int = len(FEATURE_NAMES), hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def gaussian_nll_loss(pred_log_sigma2: torch.Tensor, target_var: torch.Tensor) -> torch.Tensor:
    pred_log_sigma2 = torch.clamp(pred_log_sigma2, min=-12.0, max=4.0)
    pred_var = torch.exp(pred_log_sigma2)
    target_var = torch.clamp(target_var, min=1e-6)
    return torch.mean(0.5 * (target_var / pred_var + pred_log_sigma2))


def _split_samples(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for sample in samples:
        track_id = int(sample.get("track_id", -1))
        bucket = track_id % 10
        if bucket < 8:
            train.append(sample)
        else:
            val.append(sample)
    if not val and train:
        pivot = max(1, len(train) // 5)
        val = train[-pivot:]
        train = train[:-pivot]
    return train, val


def _tensorize(samples: Sequence[Dict[str, Any]], target_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([sample["features"] for sample in samples], dtype=torch.float32)
    scale = max(1e-6, float(target_scale))
    y = torch.tensor([sample["target_var"] / scale for sample in samples], dtype=torch.float32)
    return x, y


def _stats(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    sorted_vals = sorted(vals)
    mid = sorted_vals[len(sorted_vals) // 2]
    return {
        "count": float(len(vals)),
        "mean": float(sum(vals) / len(vals)),
        "median": float(mid),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
    }


def _evaluate(model: VarianceHeadMLP, x: torch.Tensor, y: torch.Tensor, target_scale: float = 1.0) -> Dict[str, float]:
    if x.numel() == 0:
        return {"nll": 0.0, "mae_var": 0.0, "rmse_var": 0.0}
    model.eval()
    with torch.no_grad():
        pred_log_sigma2 = torch.clamp(model(x), min=-12.0, max=4.0)
        pred_var = torch.exp(pred_log_sigma2) * max(1e-6, float(target_scale))
        target_var = y * max(1e-6, float(target_scale))
        loss = gaussian_nll_loss(pred_log_sigma2, y)
        mae = torch.mean(torch.abs(pred_var - target_var)).item()
        rmse = torch.sqrt(torch.mean((pred_var - target_var) ** 2)).item()
    return {
        "nll": float(loss.item()),
        "mae_var": float(mae),
        "rmse_var": float(rmse),
    }


def train_variance_head(
    *,
    pose_tracks_path: Path,
    epochs: int = 60,
    lr: float = 1e-3,
    hidden_dim: int = 16,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    samples = build_variance_samples(pose_tracks_path)
    if len(samples) < 16:
        raise RuntimeError("not enough variance samples generated from pose tracks")

    target_stats = _stats([sample["target_var"] for sample in samples])
    target_scale = max(1e-6, float(target_stats.get("mean", 1e-6)))
    train_samples, val_samples = _split_samples(samples)
    x_train, y_train = _tensorize(train_samples, target_scale=target_scale)
    x_val, y_val = _tensorize(val_samples, target_scale=target_scale)

    model = VarianceHeadMLP(in_dim=x_train.shape[1], hidden_dim=int(hidden_dim))
    optimizer = optim.Adam(model.parameters(), lr=float(lr))

    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        pred_log_sigma2 = model(x_train)
        loss = gaussian_nll_loss(pred_log_sigma2, y_train)
        loss.backward()
        optimizer.step()

    train_metrics = _evaluate(model, x_train, y_train, target_scale=target_scale)
    val_metrics = _evaluate(model, x_val, y_val, target_scale=target_scale)
    variance_ref = max(1e-6, float(target_scale))

    checkpoint = {
        "kind": "track_variance_head",
        "version": "1.0",
        "feature_names": list(FEATURE_NAMES),
        "feature_dim": int(x_train.shape[1]),
        "hidden_dim": int(hidden_dim),
        "target": "pseudo_temporal_residual",
        "loss": "gaussian_nll_scalar",
        "variance_ref": float(variance_ref),
        "target_scale": float(target_scale),
        "state_dict": model.state_dict(),
    }
    report = {
        "kind": "track_variance_head_report",
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "feature_names": list(FEATURE_NAMES),
        "target": "pseudo_temporal_residual",
        "loss": "gaussian_nll_scalar",
        "target_stats": target_stats,
        "target_scale": float(target_scale),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "paths": {
            "pose_tracks": str(pose_tracks_path),
        },
    }
    return checkpoint, report


def save_variance_head(path: Path, checkpoint: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_variance_head(path: Optional[Path]) -> Tuple[Optional[VarianceHeadMLP], Dict[str, Any]]:
    if path is None or not path.exists():
        return None, {}
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("state_dict"), dict):
        return None, {}
    feature_dim = int(ckpt.get("feature_dim", len(FEATURE_NAMES)))
    hidden_dim = int(ckpt.get("hidden_dim", 16))
    model = VarianceHeadMLP(in_dim=feature_dim, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, ckpt


def sigma2_to_uq(sigma2: float, variance_ref: float) -> float:
    sigma2 = max(1e-6, float(sigma2))
    variance_ref = max(1e-6, float(variance_ref))
    return _clamp01(sigma2 / (sigma2 + variance_ref))


def predict_variance_index(pose_tracks_path: Path, model_path: Path) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], Dict[str, Any]]:
    model, ckpt = load_variance_head(model_path)
    if model is None:
        return {}, {}

    samples = build_variance_samples(pose_tracks_path)
    if not samples:
        return {}, ckpt

    x, _ = _tensorize(samples)
    with torch.no_grad():
        pred_log_sigma2 = torch.clamp(model(x), min=-12.0, max=4.0)
        pred_var = torch.exp(pred_log_sigma2)

    variance_ref = max(1e-6, float(ckpt.get("variance_ref", ckpt.get("target_scale", 1e-4))))
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for idx, sample in enumerate(samples):
        log_sigma2 = float(pred_log_sigma2[idx].item())
        sigma2 = float(pred_var[idx].item() * variance_ref)
        out[(int(sample["frame"]), int(sample["track_id"]))] = {
            "log_sigma2": log_sigma2,
            "sigma2": sigma2,
            "uq_variance": sigma2_to_uq(sigma2, variance_ref),
        }
    return out, ckpt
