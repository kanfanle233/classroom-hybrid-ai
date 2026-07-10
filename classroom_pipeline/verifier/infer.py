import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from classroom_pipeline.contracts.schemas import (
    SCHEMA_VERSION,
    validate_jsonl_file,
    validate_verified_event_record,
    write_jsonl,
)
from classroom_pipeline.verifier.model import VerifierMLP, VerifierRuntimeConfig, build_feature_vector

# Student judge constants
STUDENT_FEATURE_VERSION = "v1.0"
STUDENT_FUSION_MODE = "llm_distilled_student_v4"
LEGACY_STUDENT_FUSION_MODE = "llm_distilled_student"
DEFAULT_LLM_STUDENT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "llm_judge_pipeline"
    / "models"
    / "student_judge_v4_best.joblib"
)
_STUDENT_BEHAVIOR_CODES = ["tt", "dx", "dk", "zt", "xt", "js", "zl", "jz"]


def resolve_llm_student_model_path(value: str = "auto") -> Optional[Path]:
    """Resolve CLI/model config for the V4 distilled student.

    auto/default/v4 -> project V4 model path, off/none/false -> disabled.
    Missing auto model is handled by the existing loader fallback.
    """
    text = str(value or "auto").strip()
    lowered = text.lower()
    if lowered in {"auto", "default", "v4"}:
        return DEFAULT_LLM_STUDENT_MODEL.resolve()
    if lowered in {"off", "none", "disable", "disabled", "false", "0"}:
        return None
    return Path(text).resolve()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _load_uq_index(path: Optional[Path]) -> Dict[int, float]:
    if path is None or (not path.exists()):
        return {}
    rows = _load_jsonl(path)
    by_track: Dict[int, List[float]] = {}
    for row in rows:
        persons = row.get("persons")
        if isinstance(persons, list):
            for person in persons:
                if not isinstance(person, dict):
                    continue
                tid = person.get("track_id")
                uq = person.get("uq_track", person.get("uq_score"))
                if isinstance(tid, int) and isinstance(uq, (int, float)):
                    by_track.setdefault(tid, []).append(float(uq))
            continue
        tid = row.get("track_id")
        uq = row.get("uq_score", row.get("uq_track"))
        if isinstance(tid, int) and isinstance(uq, (int, float)):
            by_track.setdefault(tid, []).append(float(uq))
    return {tid: (sum(vals) / len(vals) if vals else 0.5) for tid, vals in by_track.items()}


VALID_KINDS = {"trainable_verifier", "student_judge_v1"}


def _load_model(model_path: Optional[Path]) -> Tuple[Optional[VerifierMLP], VerifierRuntimeConfig, Dict[str, Any]]:
    """Load a verifier or student judge checkpoint.

    Returns (model, runtime_cfg, teacher_provenance).
    teacher_provenance is empty dict for trainable_verifier kind.
    """
    runtime_cfg = VerifierRuntimeConfig()
    teacher_provenance: Dict[str, Any] = {}
    if model_path is None or (not model_path.exists()):
        return None, runtime_cfg, teacher_provenance
    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return None, runtime_cfg, teacher_provenance

    kind = ckpt.get("kind", "trainable_verifier")
    if kind not in VALID_KINDS:
        print(f"[WARN] unknown checkpoint kind '{kind}', treating as trainable_verifier")
        kind = "trainable_verifier"

    in_dim = int(ckpt.get("in_dim", 4))
    hidden_dim = int(ckpt.get("hidden_dim", 16))
    state_dict = ckpt.get("state_dict")
    if isinstance(ckpt.get("runtime_config"), dict):
        runtime_cfg = VerifierRuntimeConfig.from_dict(ckpt["runtime_config"])
    if not isinstance(state_dict, dict):
        return None, runtime_cfg, teacher_provenance

    model = VerifierMLP(in_dim=in_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if kind == "student_judge_v1":
        teacher_provenance = ckpt.get("teacher_provenance", {})
        model._teacher_provenance = teacher_provenance

    return model, runtime_cfg, teacher_provenance


# ── Student judge feature builder ─────────────────────────────────────

def _build_student_features(
    *,
    event_type: str,
    query_text: str,
    cand: Dict[str, Any],
    query_source: str = "unknown",
    audio_confidence: float = 0.0,
) -> np.ndarray:
    """Build the 16-dim feature vector expected by the student judge model.

    Feature order must match exactly with feature_names used during training:
      [overlap, action_confidence, uq_score, text_score, audio_confidence,
       stability_score,
       behavior_code_tt, dx, dk, zt, xt, js, zl, jz,
       event_type_known, query_source_asr]
    """
    overlap = _safe_float(cand.get("overlap", 0.0), 0.0)
    action_conf = _safe_float(cand.get("action_confidence", cand.get("confidence", 0.0)), 0.0)
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
    text_score = feat[2]
    stability = 1.0 - uq
    behavior_code = str(cand.get("behavior_code", "")).strip().lower()

    row = [
        overlap,
        action_conf,
        uq,
        text_score,
        _safe_float(audio_confidence, 0.0),
        min(1.0, max(0.0, stability)),
    ]
    # behavior_code one-hot
    for code in _STUDENT_BEHAVIOR_CODES:
        row.append(1.0 if behavior_code == code else 0.0)
    # event_type_known
    et = str(event_type or "").strip().lower()
    row.append(0.0 if et in ("", "unknown") else 1.0)
    # query_source_asr
    qs = str(query_source or "").strip().lower()
    row.append(1.0 if qs == "asr" else 0.0)

    return np.array([row], dtype=np.float64)


def _load_student_judge(model_path: Optional[Path]) -> Any:
    """Load a sklearn student judge model from a joblib file.

    Returns None if the model cannot be loaded (file missing, bad format, etc.).
    """
    if model_path is None or not model_path.exists():
        return None
    try:
        import joblib as _jl
        ckpt = _jl.load(model_path)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            print(f"[WARN] student model {model_path} has no 'model' key", file=sys.stderr)
            return None
        model = ckpt["model"]
        # Verify it has predict_proba
        if not hasattr(model, "predict_proba"):
            print(f"[WARN] student model {model_path} lacks predict_proba", file=sys.stderr)
            return None
        # Attach scaler and metadata for forward pass
        model._student_scaler = ckpt.get("scaler")
        model._student_feature_names = ckpt.get("feature_names", [])
        model._student_classes = ckpt.get("classes", [])
        model._student_model_name = ckpt.get("model_name", "unknown")
        model._student_path = str(model_path)
        model._teacher_source = ckpt.get("teacher_source", "")
        model._teacher_dataset = ckpt.get("teacher_dataset", "")
        model._evaluation_kind = ckpt.get("evaluation_kind", "")
        return model
    except Exception as e:
        print(f"[WARN] failed to load student judge {model_path}: {e}", file=sys.stderr)
        return None


# ── Core prediction ────────────────────────────────────────────────────

def _predict_one(
    *,
    model: Optional[VerifierMLP],
    runtime_cfg: VerifierRuntimeConfig,
    event_type: str,
    query_text: str,
    cand: Dict[str, Any],
    uq_default: float,
    query_source: str = "unknown",
    audio_confidence: float = 0.0,
    student_model: Any = None,
    student_feature_version: str = STUDENT_FEATURE_VERSION,
) -> Dict[str, Any]:
    overlap = _safe_float(cand.get("overlap", 0.0), 0.0)
    action_conf = _safe_float(cand.get("action_confidence", cand.get("confidence", 0.0)), 0.0)
    uq = _safe_float(cand.get("uq_score", cand.get("uq_track", uq_default)), uq_default)
    action_label = str(cand.get("semantic_id", cand.get("action", ""))).strip().lower()
    raw_action = str(cand.get("action", "")).strip().lower()
    behavior_code = str(cand.get("behavior_code", "")).strip().lower()
    behavior_label_zh = str(cand.get("behavior_label_zh", "")).strip()
    behavior_label_en = str(cand.get("behavior_label_en", "")).strip()
    semantic_label_zh = str(cand.get("semantic_label_zh", "")).strip()
    semantic_label_en = str(cand.get("semantic_label_en", "")).strip()
    taxonomy_version = str(cand.get("taxonomy_version", "")).strip()

    feat = build_feature_vector(
        event_type=event_type,
        query_text=query_text,
        action_label=action_label,
        overlap=overlap,
        action_confidence=action_conf,
        uq_score=uq,
    )
    text_score = feat[2]
    visual_score = _clamp01(0.65 * feat[0] + 0.35 * feat[1])

    # === Confidence-Weighted Dynamic Late Fusion ===
    # Reference: Kiziltepe et al. (IEEE Access 2024); CMU-MMAC Survey (arXiv 2025)
    # Key insight: modality weights adapt to real-time confidence, not fixed 0.55/0.45
    is_visual_fallback = "fallback" in str(query_source).lower()
    has_real_audio = (
        audio_confidence > 0.0
        and not is_visual_fallback
        and str(query_source).strip().lower() not in ("unknown", "placeholder", "fallback", "")
    )

    if has_real_audio:
        # Dynamic modality weights: tracking stability vs audio quality
        w_visual = _clamp01(1.0 - uq)           # lower uncertainty = higher visual weight
        w_audio  = _clamp01(audio_confidence)    # ASR quality as audio weight
        p_match = _clamp01(
            (w_visual * visual_score + w_audio * text_score) / (w_visual + w_audio + 1e-9)
        )
        fusion_mode = "audio_visual_dynamic"
    elif is_visual_fallback:
        # Visual fallback: no real audio, don't fake text_score contribution
        p_match = visual_score
        text_score = float('nan')  # mark as not applicable
        fusion_mode = "visual_only_fallback"
    else:
        # No audio available at all
        p_match = visual_score
        fusion_mode = "visual_only"

    # Override: if MLP model is provided, use it (trained on gold labels)
    if model is not None:
        x = torch.tensor([feat], dtype=torch.float32)
        with torch.no_grad():
            logits = model(x).squeeze(0)
        p_match = float(torch.sigmoid(logits / max(1e-6, runtime_cfg.temperature)).item())
        fusion_mode = "mlp_trained"

    # Override: if student judge model is provided, use it
    # Student takes priority over both heuristic and MLP
    if student_model is not None:
        fallback_fusion_mode = fusion_mode
        try:
            X_student = _build_student_features(
                event_type=event_type,
                query_text=query_text,
                cand=cand,
                query_source=query_source,
                audio_confidence=audio_confidence,
            )
            scaler = getattr(student_model, "_student_scaler", None)
            if scaler is not None:
                X_student = scaler.transform(X_student)

            probs = student_model.predict_proba(X_student)[0]  # [p_match, p_mismatch, p_uncertain]
            # Classes are ["match", "mismatch", "uncertain"] in that order
            if len(probs) >= 3:
                p_match = float(probs[0])  # match probability
            else:
                p_match = float(probs[0])

            # Clamp
            p_match = _clamp01(p_match)
            fusion_mode = STUDENT_FUSION_MODE
        except Exception as e:
            print(f"[WARN] student judge inference failed: {e}, falling back", file=sys.stderr)
            # Keep the previous p_match and its exact fallback fusion mode.
            fusion_mode = fallback_fusion_mode

    p_mismatch = _clamp01(1.0 - p_match)
    reliability = _clamp01(p_match * (1.0 - runtime_cfg.uq_gate * _clamp01(uq)))

    if reliability >= runtime_cfg.match_threshold:
        label = "match"
    elif reliability >= runtime_cfg.uncertain_threshold:
        label = "uncertain"
    else:
        label = "mismatch"

    # Student model metadata
    student_model_path = ""
    student_feature_ver = ""
    teacher_source = ""
    teacher_dataset = ""
    if student_model is not None and fusion_mode in (STUDENT_FUSION_MODE, LEGACY_STUDENT_FUSION_MODE):
        student_model_path = getattr(student_model, "_student_path", "")
        student_model_name = getattr(student_model, "_student_model_name", "")
        student_feature_ver = student_feature_version
        teacher_source = getattr(student_model, "_teacher_source", "") or ""
        teacher_dataset = getattr(student_model, "_teacher_dataset", "") or ""

    return {
        "p_match": p_match,
        "p_mismatch": p_mismatch,
        "match_label": label,
        "reliability_score": reliability,
        "uncertainty": _clamp01(1.0 - reliability),
        "visual_score": visual_score,
        "text_score": text_score,
        "uq_score": _clamp01(uq),
        "fusion_mode": fusion_mode,
        "student_model_path": student_model_path,
        "student_feature_version": student_feature_ver,
        "teacher_source": teacher_source,
        "teacher_dataset": teacher_dataset,
        "w_visual": _clamp01(1.0 - uq) if has_real_audio else 1.0,
        "w_audio": _clamp01(audio_confidence) if has_real_audio else 0.0,
        "action": action_label,
        "raw_action": raw_action,
        "behavior_code": behavior_code,
        "behavior_label_zh": behavior_label_zh,
        "behavior_label_en": behavior_label_en,
        "semantic_id": action_label,
        "semantic_label_zh": semantic_label_zh,
        "semantic_label_en": semantic_label_en,
        "taxonomy_version": taxonomy_version,
        "runtime_config": runtime_cfg.to_dict(),
    }


def infer_verified_rows(
    *,
    event_queries_path: Path,
    aligned_path: Path,
    pose_uq_path: Optional[Path],
    model_path: Optional[Path],
    keep_all_candidates: bool = False,
    llm_student_model_path: Optional[Path] = None,
    student_feature_version: str = STUDENT_FEATURE_VERSION,
) -> List[Dict[str, Any]]:
    queries = _load_jsonl(event_queries_path)
    q_index = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}
    aligned_obj = _load_json(aligned_path)
    aligned = aligned_obj if isinstance(aligned_obj, list) else []
    uq_index = _load_uq_index(pose_uq_path)
    model, runtime_cfg, teacher_provenance = _load_model(model_path)
    threshold_source = "model_runtime_config" if model is not None else "heuristic_default"

    # Build model_version from provenance (for student judge) or default
    if teacher_provenance and teacher_provenance.get("teacher_model"):
        model_version = f"student_judge:{teacher_provenance['teacher_model']}"
    elif model is not None:
        model_version = "verifier_mlp_trained"
    else:
        model_version = "heuristic_v1"

    # Load student judge model (sklearn joblib)
    student_model = _load_student_judge(llm_student_model_path)
    if student_model is not None:
        model_version = f"student_judge:{getattr(student_model, '_student_model_name', 'sklearn')}"

    rows: List[Dict[str, Any]] = []
    for block in aligned:
        if not isinstance(block, dict):
            continue
        query_id = str(block.get("query_id", block.get("event_id", "")))
        query = q_index.get(query_id, {})
        event_type = str(query.get("event_type", block.get("event_type", "unknown")))
        query_text = str(query.get("query_text", block.get("query_text", "")))
        query_source = str(query.get("source", block.get("source", "unknown")))
        audio_conf = _safe_float(query.get("confidence", 0.0), 0.0)
        window = block.get("window", {})
        if not isinstance(window, dict):
            window = {
                "start": _safe_float(block.get("window_start", 0.0), 0.0),
                "end": _safe_float(block.get("window_end", 0.0), 0.0),
            }
        w_start = _safe_float(window.get("start", 0.0), 0.0)
        w_end = _safe_float(window.get("end", max(w_start, 0.0)), max(w_start, 0.0))
        if w_end < w_start:
            w_start, w_end = w_end, w_start

        candidates = block.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        scored: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            tid = int(cand.get("track_id", -1))
            uq_default = float(uq_index.get(tid, 0.5))
            score = _predict_one(
                model=model,
                runtime_cfg=runtime_cfg,
                event_type=event_type,
                query_text=query_text,
                cand=cand,
                uq_default=uq_default,
                query_source=query_source,
                audio_confidence=audio_conf,
                student_model=student_model,
                student_feature_version=student_feature_version,
            )
            scored.append((cand, score))
        scored.sort(key=lambda x: x[1]["reliability_score"], reverse=True)

        if not scored:
            row = {
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "track_id": -1,
                "event_type": event_type,
                "query_text": query_text,
                "window": {"start": w_start, "end": w_end},
                "match_label": "mismatch",
                "reliability_score": 0.0,
                "uncertainty": 1.0,
                "p_match": 0.0,
                "p_mismatch": 1.0,
                "action": "",
                "raw_action": "",
                "behavior_code": "",
                "behavior_label_zh": "",
                "behavior_label_en": "",
                "semantic_id": "",
                "semantic_label_zh": "",
                "semantic_label_en": "",
                "taxonomy_version": "",
                "threshold_source": threshold_source,
                "model_version": model_version,
                "runtime_config": runtime_cfg.to_dict(),
                "evidence": {"visual_score": 0.0, "text_score": 0.0, "uq_score": 1.0, "fusion_mode": "no_candidates", "w_visual": 1.0, "w_audio": 0.0},
            }
            # Tag with student model info if loaded
            if student_model is not None:
                sm_path = getattr(student_model, "_student_path", "")
                if sm_path:
                    row["evidence"]["student_model_path"] = sm_path
                    row["evidence"]["student_feature_version"] = student_feature_version
                ts_src = getattr(student_model, "_teacher_source", "") or ""
                ts_ds = getattr(student_model, "_teacher_dataset", "") or ""
                if ts_src:
                    row["evidence"]["teacher_source"] = ts_src
                if ts_ds:
                    row["evidence"]["teacher_dataset"] = ts_ds
            rows.append(row)
            continue

        selected = scored if keep_all_candidates else [scored[0]]
        for cand, score in selected:
            # Build evidence block with student model metadata
            evidence = {
                "visual_score": float(score.get("visual_score", 0.0)),
                "text_score": float(score.get("text_score", 0.0)) if not (isinstance(score.get("text_score"), float) and score["text_score"] != score["text_score"]) else None,
                "uq_score": float(score.get("uq_score", 0.0)),
                "fusion_mode": str(score.get("fusion_mode", "unknown")),
                "w_visual": float(score.get("w_visual", 1.0)),
                "w_audio": float(score.get("w_audio", 0.0)),
            }
            sm_path = str(score.get("student_model_path", ""))
            sm_fv = str(score.get("student_feature_version", ""))
            ts_src = str(score.get("teacher_source", ""))
            ts_ds = str(score.get("teacher_dataset", ""))
            if sm_path:
                evidence["student_model_path"] = sm_path
                evidence["student_feature_version"] = sm_fv
            if ts_src:
                evidence["teacher_source"] = ts_src
            if ts_ds:
                evidence["teacher_dataset"] = ts_ds
            row = {
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "track_id": int(cand.get("track_id", -1)),
                "event_type": event_type,
                "query_text": query_text,
                "window": {"start": w_start, "end": w_end},
                "match_label": str(score["match_label"]),
                "reliability_score": float(score["reliability_score"]),
                "uncertainty": float(score["uncertainty"]),
                "p_match": float(score["p_match"]),
                "p_mismatch": float(score["p_mismatch"]),
                "action": str(score.get("action", "")),
                "raw_action": str(score.get("raw_action", "")),
                "behavior_code": str(score.get("behavior_code", "")),
                "behavior_label_zh": str(score.get("behavior_label_zh", "")),
                "behavior_label_en": str(score.get("behavior_label_en", "")),
                "semantic_id": str(score.get("semantic_id", "")),
                "semantic_label_zh": str(score.get("semantic_label_zh", "")),
                "semantic_label_en": str(score.get("semantic_label_en", "")),
                "taxonomy_version": str(score.get("taxonomy_version", "")),
                "threshold_source": threshold_source,
                "model_version": model_version,
                "runtime_config": score["runtime_config"],
                "evidence": evidence,
            }
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier inference -> verified_events.jsonl")
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--aligned", required=True, type=str)
    parser.add_argument("--pose_uq", default="", type=str)
    parser.add_argument("--model", default="", type=str, help="verifier.pt (MLP)")
    parser.add_argument("--llm_student_model", default="auto", type=str,
                        help="student judge .joblib model path; auto uses V4 default, off disables")
    parser.add_argument("--llm_student_features", default=STUDENT_FEATURE_VERSION, type=str,
                        help=f"student feature version (default: {STUDENT_FEATURE_VERSION})")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--keep_all_candidates", type=int, default=0)
    parser.add_argument("--validate", type=int, default=1)
    args = parser.parse_args()

    event_queries = Path(args.event_queries).resolve()
    aligned = Path(args.aligned).resolve()
    pose_uq = Path(args.pose_uq).resolve() if args.pose_uq else None
    model = Path(args.model).resolve() if args.model else None
    llm_student_model = resolve_llm_student_model_path(args.llm_student_model)
    out = Path(args.out).resolve()

    rows = infer_verified_rows(
        event_queries_path=event_queries,
        aligned_path=aligned,
        pose_uq_path=pose_uq,
        model_path=model,
        keep_all_candidates=bool(int(args.keep_all_candidates)),
        llm_student_model_path=llm_student_model,
        student_feature_version=str(args.llm_student_features),
    )
    write_jsonl(out, rows)

    if int(args.validate) == 1:
        ok, _, errors = validate_jsonl_file(out, validate_verified_event_record)
        if not ok:
            first_error = errors[0] if errors else "unknown schema error"
            raise ValueError(f"invalid verified events schema: {first_error}")

    print(f"[DONE] verified events: {out} ({len(rows)})")


if __name__ == "__main__":
    main()
