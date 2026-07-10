import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from paths import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

from classroom_pipeline.contracts.schemas import SCHEMA_VERSION
from classroom_pipeline.model_resolver import resolve_model_reference, try_resolve_model_reference
from classroom_pipeline.verifier.dataset import build_training_samples, convert_to_contract_samples, save_training_samples


DEFAULT_WHISPER_MODEL = "Systran/faster-whisper-small"
DEFAULT_BEHAVIOR_DET_MODEL = "runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt"
PIPELINE_STAGE_RECORDS: List[Dict[str, Any]] = []
PIPELINE_MANIFEST_WRITER: Callable[[str, str], None] | None = None


STEP_ORDER = [
    1,
    15,
    2,
    4,
    35,
    45,
    456,
    5,
    55,
    46,
    47,
    471,
    4711,
    472,
    473,
    474,
    58,
    59,
    6,
    65,
    656,
    66,
    67,
    70,
    71,
    72,
    73,
    80,
    90,
    91,
]
STEP_RANK = {step_id: idx for idx, step_id in enumerate(STEP_ORDER)}


def _step_rank(step_id: int) -> int:
    step_id = int(step_id)
    return STEP_RANK.get(step_id, step_id * 1000)


def run_step(py_exe: str, script: Path, args: List[str]) -> None:
    cmd = [py_exe, str(script)] + args
    print("\n" + "=" * 80)
    print(f"[RUN] {script.name} {' '.join(args)}")
    print("=" * 80)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Step failed: {script.name} (exit={r.returncode})")


def file_is_fresh(output_path: Path, inputs: List[Path], min_bytes: int = 10) -> bool:
    if not output_path.exists() or not output_path.is_file():
        return False
    try:
        if output_path.stat().st_size < min_bytes:
            return False
        out_m = output_path.stat().st_mtime
        for ip in inputs:
            if ip.exists() and ip.stat().st_mtime > out_m:
                return False
        return True
    except Exception:
        return False


def assert_outputs_ready(step_id: int, step_name: str, outputs: List[Path], min_bytes: int = 10) -> None:
    problems: List[str] = []
    for output in outputs:
        if not output.exists() or not output.is_file():
            problems.append(f"missing output: {output}")
            continue
        try:
            size = output.stat().st_size
        except Exception:
            problems.append(f"unreadable output: {output}")
            continue
        if size < min_bytes:
            problems.append(f"empty output: {output} ({size} bytes)")
    if problems:
        joined = "; ".join(problems)
        raise RuntimeError(f"step{step_id:03d} {step_name} produced invalid artifacts: {joined}")


def maybe_run(step_id, step_name, outputs, inputs, force, from_step, run_fn, min_bytes=10):
    to_step = int(getattr(maybe_run, "to_step", 999999))
    step_rank = _step_rank(int(step_id))
    started = time.perf_counter()
    record = {
        "step_id": int(step_id),
        "step_name": str(step_name),
        "status": "pending",
        "input_files": [str(p) for p in inputs],
        "output_files": [str(p) for p in outputs],
        "input_check": [{"path": str(p), "exists": Path(p).exists()} for p in inputs],
        "output_check": [],
        "duration_sec": None,
        "failure_reason": "",
    }

    def _finish(status: str, failure_reason: str = ""):
        record["status"] = status
        record["duration_sec"] = round(time.perf_counter() - started, 6)
        record["failure_reason"] = failure_reason
        checks = []
        for output in outputs:
            p = Path(output)
            checks.append(
                {
                    "path": str(p),
                    "exists": p.exists(),
                    "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0,
                }
            )
        record["output_check"] = checks
        PIPELINE_STAGE_RECORDS.append(record)
        _refresh_pipeline_manifest("partial", failure_reason)

    if step_rank < _step_rank(int(from_step)):
        print(f"[SKIP] step{step_id:03d} {step_name}")
        _finish("skipped")
        return
    if step_rank > _step_rank(to_step):
        print(f"[SKIP] step{step_id:03d} {step_name}")
        _finish("skipped")
        return
    try:
        if force:
            print(f"[FORCE] step{step_id:03d} {step_name}")
            run_fn()
            assert_outputs_ready(step_id, step_name, outputs, min_bytes=min_bytes)
            _finish("ok")
            return

        all_fresh = all(file_is_fresh(o, inputs, min_bytes=min_bytes) for o in outputs)
        if all_fresh:
            print(f"[CACHE] step{step_id:03d} {step_name}")
            assert_outputs_ready(step_id, step_name, outputs, min_bytes=min_bytes)
            _finish("cached")
        else:
            print(f"[DO] step{step_id:03d} {step_name}")
            run_fn()
            assert_outputs_ready(step_id, step_name, outputs, min_bytes=min_bytes)
            _finish("ok")
    except Exception as exc:
        _finish("error", str(exc))
        raise


def resolve_model_or_fail(model_arg: str) -> str:
    return resolve_model_reference(model_arg, project_root=PROJECT_ROOT, required=True)


def resolve_llm_student_model_arg(model_arg: str):
    text = str(model_arg or "auto").strip()
    lowered = text.lower()
    if lowered in {"auto", "default", "v4"}:
        return (PROJECT_ROOT / "output" / "llm_judge_pipeline" / "models" / "student_judge_v4_best.joblib").resolve()
    if lowered in {"off", "none", "disable", "disabled", "false", "0"}:
        return None
    candidate = Path(text)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def resolve_video_path(video_arg: str) -> Path:
    candidate = Path(video_arg)
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_out_dir(args, video_path: Path, base_dir: Path) -> Path:
    if args.out_dir:
        return resolve_video_path(args.out_dir)
    if args.output_root:
        root = resolve_video_path(args.output_root)
        view_dir = root / args.view if args.view else root
        case_id = args.case_id or video_path.stem
        if args.video_id:
            vid = args.video_id
        else:
            view_code = args.view_code or args.view or "case"
            vid = f"{view_code}__{case_id}"
        return view_dir / vid
    return (base_dir / "output" / (args.name or video_path.stem)).resolve()


def _auto_pose_model_path(user_arg: str) -> str:
    if user_arg and user_arg.lower() != "auto":
        return resolve_model_or_fail(user_arg)
    candidates = [
        PROJECT_ROOT / "models" / "yolo11x-pose.pt",
        PROJECT_ROOT / "models" / "yolo11l-pose.pt",
        PROJECT_ROOT / "models" / "yolo11m-pose.pt",
        PROJECT_ROOT / "models" / "yolo11s-pose.pt",
        PROJECT_ROOT / "models" / "yolo11n-pose.pt",
        PROJECT_ROOT / "yolo11x-pose.pt",
        PROJECT_ROOT / "yolo11l-pose.pt",
        PROJECT_ROOT / "yolo11m-pose.pt",
        PROJECT_ROOT / "models" / "classroom_yolo_enhanced.pt",
        PROJECT_ROOT / "yolo11s-pose.pt",
        PROJECT_ROOT / "yolo11n-pose.pt",
    ]
    for cand in candidates:
        cand = cand.resolve()
        if cand.exists():
            print(f"[INFO] using pose model: {cand}")
            return str(cand)
    return resolve_model_or_fail("yolo11x-pose.pt")


def _is_placeholder_transcript(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = str(obj.get("text", ""))
                if "[ASR_EMPTY:" in text:
                    return True
                return False
    except Exception:
        return True
    return True


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, str):
        return value.replace("\\", "/")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_pipeline_manifest(
    *,
    out_dir: Path,
    case_id: str,
    video_id: str,
    artifacts: Dict[str, Path],
    config_snapshot: Dict[str, Any],
    status: str = "complete",
    failure_reason: str = "",
) -> None:
    payload = {
        "case_id": case_id,
        "video_id": video_id,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "failure_reason": failure_reason,
        "artifacts": {k: str(v) for k, v in artifacts.items() if v.exists()},
        "config_snapshot": config_snapshot,
        "stages": PIPELINE_STAGE_RECORDS,
    }
    payload = _json_safe(payload)
    _write_json(out_dir / "pipeline_manifest.json", payload)


def _refresh_pipeline_manifest(status: str = "partial", failure_reason: str = "") -> None:
    if PIPELINE_MANIFEST_WRITER is None:
        return
    try:
        PIPELINE_MANIFEST_WRITER(status, failure_reason)
    except Exception as exc:
        print(f"[WARN] could not refresh pipeline_manifest.json: {exc}")


def main() -> None:
    PIPELINE_STAGE_RECORDS.clear()
    base_dir = PROJECT_ROOT
    # Main pipeline stage scripts live inside the package; legacy scripts/pipeline
    # paths are compatibility wrappers only.
    scripts_dir = PROJECT_ROOT / "classroom_pipeline"

    parser = argparse.ArgumentParser(
        description="Formal pipeline (fixed schema): pose -> uq -> actions -> asr -> event queries -> align -> verifier"
    )
    parser.add_argument("--video", type=str, default="data/videos/demo3.mp4")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--view", type=str, default="")
    parser.add_argument("--view_code", type=str, default="")
    parser.add_argument("--case_id", type=str, default="")
    parser.add_argument("--video_id", type=str, default="")
    parser.add_argument("--py", type=str, default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from_step", type=int, default=1)
    parser.add_argument("--to_step", type=int, default=999999)
    parser.add_argument("--skip_vis", action="store_true")
    parser.add_argument("--export_compat", type=int, default=1)
    parser.add_argument("--track_backend", choices=["pose", "behavior", "hybrid"], default="hybrid")

    parser.add_argument("--pose_model", type=str, default="auto")
    parser.add_argument("--pose_conf", type=float, default=0.20)
    parser.add_argument("--pose_imgsz", type=int, default=960)
    parser.add_argument("--pose_device", type=str, default="")
    parser.add_argument("--pose_half", type=int, default=0)
    parser.add_argument("--pose_track_person_conf", type=float, default=-1.0)
    parser.add_argument("--pose_infer_mode", choices=["full", "sliced", "full_sliced", "roi_sr_sliced"], default="full_sliced")
    parser.add_argument("--pose_slice_grid", type=str, default="2x2", help="NxM, auto/adaptive, rear_adaptive, or rear_dense")
    parser.add_argument("--pose_slice_overlap", type=float, default=0.25)
    parser.add_argument("--pose_slice_roi", type=str, default="auto_rear")
    parser.add_argument(
        "--sr_backend",
        choices=["off", "opencv", "realesrgan", "basicvsrpp", "realbasicvsr", "nvidia_vsr", "maxine_vfx"],
        default="off",
    )
    parser.add_argument("--sr_scale", type=float, default=2.0)
    parser.add_argument("--sr_preprocess", choices=["off", "denoise", "deblock", "deblur", "artifact_deblur", "clahe"], default="off")
    parser.add_argument("--sr_cache_external_command", type=str, default="")
    parser.add_argument("--sr_allow_unavailable", type=int, default=0)
    parser.add_argument("--pose_track_max_lost_frames", type=int, default=500)
    parser.add_argument("--pose_track_iou_thres", type=float, default=0.05)
    parser.add_argument("--pose_track_max_center_dist_ratio", type=float, default=0.15)
    parser.add_argument("--pose_track_max_dx_ratio", type=float, default=0.05)
    parser.add_argument("--pose_track_height_penalty", type=float, default=0.60)
    parser.add_argument("--pose_track_seat_prior_mode", choices=["off", "x_anchor"], default="x_anchor")
    parser.add_argument("--track_match_mode", choices=["hungarian", "greedy"], default="hungarian")
    parser.add_argument("--track_motion_model", choices=["none", "kalman"], default="kalman")
    parser.add_argument("--det_model", type=str, default="yolo11x.pt")
    parser.add_argument("--action_model", type=str, default="")
    parser.add_argument("--action_mode", choices=["auto", "slowfast", "rules"], default="auto")
    parser.add_argument("--asr_backend", choices=["auto", "api", "whisper", "openai"], default="whisper")
    parser.add_argument("--asr_model", type=str, default="paraformer-realtime-v1")
    parser.add_argument("--asr_lang", type=str, default="zh")
    parser.add_argument(
        "--whisper_model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help="faster-whisper / CTranslate2 model id or local path",
    )
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--whisper_compute_type", type=str, default="float16")
    parser.add_argument("--whisper_beam_size", type=int, default=10)
    parser.add_argument("--whisper_vad_filter", type=int, default=0)
    parser.add_argument("--whisper_condition_on_previous_text", type=int, default=0)
    parser.add_argument("--whisper_audio_filter", type=str, default="")
    parser.add_argument("--whisper_min_avg_logprob", type=str, default="-0.6")
    parser.add_argument("--whisper_max_no_speech_prob", type=str, default="0.65")
    parser.add_argument("--whisper_max_compression_ratio", type=str, default="")
    parser.add_argument("--whisper_min_text_chars", type=int, default=1)
    parser.add_argument("--whisper_retry_on_fail", type=int, default=1)
    parser.add_argument("--whisper_retry_compute_type", type=str, default="int8_float16")
    parser.add_argument("--whisper_fallback_device", type=str, default="cpu")
    parser.add_argument("--whisper_fallback_compute_type", type=str, default="int8")
    parser.add_argument("--openai_asr_model", type=str, default="gpt-4o-mini-transcribe")
    parser.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai_timeout_sec", type=float, default=180.0)
    parser.add_argument("--openai_max_file_mb", type=float, default=24.5)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--train_verifier", type=int, default=0, help="1=train verifier model before step07")
    parser.add_argument("--verifier_model", type=str, default="", help="path to verifier checkpoint (.pt)")
    parser.add_argument(
        "--llm_student_model",
        type=str,
        default="auto",
        help="student judge .joblib path; auto uses V4 default, off disables",
    )
    parser.add_argument("--eval_target_field", type=str, default="auto")
    parser.add_argument("--calibration_target_field", type=str, default="auto")
    parser.add_argument("--calibration_prob_field", type=str, default="p_match")
    parser.add_argument("--calibration_num_bins", type=int, default=10)
    parser.add_argument("--disable_temperature_scaling", type=int, default=0)

    # Legacy compatibility args (accepted but no longer on default chain).
    parser.add_argument("--fuse_action_obj", type=int, default=0)
    parser.add_argument("--fuse_window", type=float, default=0.8)
    parser.add_argument("--fuse_alpha", type=float, default=0.75)
    parser.add_argument("--fuse_beta", type=float, default=0.25)
    parser.add_argument("--enable_object_evidence", type=int, default=1)
    parser.add_argument("--object_conf", type=float, default=0.10)
    parser.add_argument("--object_iou", type=float, default=0.50)
    parser.add_argument("--object_imgsz", type=int, default=1280)
    parser.add_argument("--object_stride", type=int, default=1)
    parser.add_argument("--object_classes", type=str, default="67,73,63")
    parser.add_argument("--enable_behavior_det", type=int, default=1)
    parser.add_argument("--behavior_det_model", type=str, default=DEFAULT_BEHAVIOR_DET_MODEL)
    parser.add_argument("--behavior_det_conf", type=float, default=0.25)
    parser.add_argument("--behavior_det_iou", type=float, default=0.50)
    parser.add_argument("--behavior_det_imgsz", type=int, default=832)
    parser.add_argument("--behavior_det_stride", type=int, default=1)
    parser.add_argument("--behavior_det_device", type=str, default="")
    parser.add_argument("--behavior_infer_mode", choices=["full", "sliced", "full_sliced"], default="full_sliced")
    parser.add_argument("--behavior_action_mode", choices=["off", "append", "behavior_only"], default="append")
    parser.add_argument("--behavior_tracker", choices=["bytetrack", "botsort", "simple"], default="bytetrack")
    parser.add_argument("--behavior_same_person_iou", type=float, default=0.55)
    parser.add_argument("--behavior_track_iou_thres", type=float, default=0.30)
    parser.add_argument("--behavior_track_max_gap", type=int, default=3)
    parser.add_argument("--behavior_track_buffer", type=int, default=30)
    parser.add_argument("--behavior_track_low_thresh", type=float, default=0.10)
    parser.add_argument("--behavior_new_track_thresh", type=float, default=0.25)
    parser.add_argument("--behavior_track_min_frames", type=int, default=3)
    parser.add_argument("--behavior_link_pose_iou", type=float, default=0.12)
    parser.add_argument("--behavior_hybrid_match_mode", choices=["strict", "relaxed", "no_prune"], default="no_prune")
    parser.add_argument("--behavior_unlinked_track_offset", type=int, default=200000)
    parser.add_argument("--behavior_extra_track_offset", type=int, default=100000)
    parser.add_argument("--enable_pose_behavior_video", type=int, default=1)
    parser.add_argument("--pose_behavior_video_out", type=str, default="pose_behavior_fusion_yolo11x.mp4")
    parser.add_argument("--pose_behavior_video_label_source", choices=["frame", "segment"], default="segment")
    parser.add_argument("--pose_behavior_video_show_behavior_bbox", type=int, default=1)
    parser.add_argument("--pose_behavior_video_min_conf", type=float, default=0.25)
    parser.add_argument("--pose_behavior_video_keypoint_conf", type=float, default=0.35)
    parser.add_argument("--pose_behavior_video_show_unmatched_label", type=int, default=0)
    parser.add_argument("--pose_behavior_video_compact_label", type=int, default=1)
    parser.add_argument("--pose_behavior_video_show_unlinked_behavior_bbox", type=int, default=1)
    parser.add_argument("--pose_behavior_video_show_unlinked_behavior_label", type=int, default=0)
    parser.add_argument("--pose_behavior_video_show_unlinked_behavior_legend", type=int, default=1)
    parser.add_argument("--pose_behavior_video_unlinked_behavior_min_conf", type=float, default=0.25)
    parser.add_argument("--fusion_contract_v2", type=int, default=1)
    parser.add_argument(
        "--semantic_taxonomy",
        type=str,
        default="classroom_pipeline/profiles/action_semantics_8class.yaml",
    )
    parser.add_argument("--fusion_contract_strict", type=int, default=1)
    parser.add_argument("--fusion_min_asr_queries", type=int, default=2)
    parser.add_argument("--fusion_visual_topk", type=int, default=12)
    parser.add_argument("--fusion_visual_min_conf", type=float, default=0.35)
    parser.add_argument("--fusion_object_window", type=float, default=0.8)
    parser.add_argument("--fusion_object_beta", type=float, default=0.20)
    parser.add_argument("--enable_peer_aware", type=int, default=0)
    parser.add_argument("--peer_radius", type=float, default=0.15)
    parser.add_argument("--interaction_model", choices=["igformer", "legacy"], default="igformer")
    parser.add_argument("--enable_mllm", type=int, default=0)
    parser.add_argument("--interpolate_occluded", type=int, default=0)
    parser.add_argument("--occlusion_conf_thres", type=float, default=0.2)
    parser.add_argument("--track_min_frames", type=int, default=75)
    parser.add_argument("--track_min_frames_ratio", type=float, default=0.0)
    parser.add_argument("--track_min_frames_min", type=int, default=10)
    parser.add_argument("--viz_gap_tol", type=float, default=1.5)
    parser.add_argument("--viz_min_dur", type=float, default=0.1)
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    name = args.name if args.name else (args.case_id or video_path.stem)
    out_dir = _resolve_out_dir(args, video_path, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maybe_run.to_step = int(args.to_step)

    pose_demo_video = out_dir / "pose_demo_yolo11x.mp4"
    pose_behavior_video = out_dir / str(args.pose_behavior_video_out)
    pose_behavior_preview = pose_behavior_video.with_name(pose_behavior_video.stem + "_preview.jpg")
    pose_behavior_report = out_dir / "pose_behavior_video.report.json"
    rear_row_contact_sheet = out_dir / "rear_row_compare_contact_sheet.jpg"
    rear_row_contact_report = out_dir / "rear_row_compare_contact_sheet.report.json"
    rear_row_pose_diagnostics = out_dir / "rear_row_pose_diagnostics.json"
    rear_row_behavior_diagnostics = out_dir / "rear_row_behavior_diagnostics.json"
    rear_row_sr_cache_dir = out_dir / "rear_roi_sr" / str(args.sr_backend)
    rear_row_sr_report = rear_row_sr_cache_dir / "sr_cache.report.json"
    pose_keypoints = out_dir / "pose_keypoints_v2.jsonl"
    pose_tracks = out_dir / "pose_tracks_smooth.jsonl"
    pose_tracks_report = out_dir / "pose_tracks.report.json"
    pose_uq = out_dir / "pose_tracks_smooth_uq.jsonl"
    objects_jsonl = out_dir / "objects.jsonl"
    objects_semantic_jsonl = out_dir / "objects.semantic.jsonl"
    behavior_det_jsonl = out_dir / "behavior_det.jsonl"
    behavior_det_semantic_jsonl = out_dir / "behavior_det.semantic.jsonl"
    student_tracks_jsonl = out_dir / "student_tracks.jsonl"
    student_tracks_report = out_dir / "student_tracks.report.json"
    behavior_unmatched_jsonl = out_dir / "behavior_unmatched.jsonl"
    pose_keypoints_behavior_compat = out_dir / "pose_keypoints_behavior_compat.jsonl"
    pose_tracks_behavior_compat = out_dir / "pose_tracks_behavior_compat.jsonl"
    pose_uq_behavior_compat = out_dir / "pose_tracks_behavior_compat_uq.jsonl"
    behavior_actions_jsonl = out_dir / "actions.behavior.jsonl"
    behavior_actions_semantic_jsonl = out_dir / "actions.behavior.semantic.jsonl"
    actions_behavior_aug_jsonl = out_dir / "actions.behavior_aug.jsonl"
    actions_raw_jsonl = out_dir / "actions.raw.jsonl"
    actions_jsonl = out_dir / "actions.jsonl"
    actions_fusion_v2_jsonl = out_dir / "actions.fusion_v2.jsonl"
    transcript_jsonl = out_dir / "transcript.jsonl"
    asr_quality_report = out_dir / "asr_quality_report.json"
    event_queries_jsonl = out_dir / "event_queries.jsonl"
    event_queries_visual_fallback_jsonl = out_dir / "event_queries.visual_fallback.jsonl"
    event_queries_fusion_v2_jsonl = out_dir / "event_queries.fusion_v2.jsonl"
    aligned_json = out_dir / "align_multimodal.json"
    verified_events_jsonl = out_dir / "verified_events.jsonl"
    verified_events_coherence_v3_jsonl = out_dir / "verified_events.coherence_v3.jsonl"
    fusion_contract_report = out_dir / "fusion_contract_report.json"
    verifier_model = Path(args.verifier_model).resolve() if args.verifier_model else (out_dir / "verifier.pt")
    llm_student_model = resolve_llm_student_model_arg(args.llm_student_model)
    verifier_report_raw = out_dir / "verifier_report.raw.json"
    verifier_samples_raw = out_dir / "verifier_samples.raw.jsonl"
    verifier_samples_train = out_dir / "verifier_samples_train.jsonl"
    verifier_eval_report = out_dir / "verifier_eval_report.json"
    verifier_calibration_report = out_dir / "verifier_calibration_report.json"
    verifier_reliability_diagram = out_dir / "verifier_reliability_diagram.svg"
    per_person_json = out_dir / "per_person_sequences.json"
    timeline_png = out_dir / "timeline_chart.png"
    timeline_json = out_dir / "timeline_chart.json"
    timeline_students_csv = out_dir / "timeline_students.csv"
    student_id_map_json = out_dir / "student_id_map.json"
    pipeline_contract_report = out_dir / "pipeline_contract_v2_report.json"
    manifest_artifacts = {
        "pose_demo_video": pose_demo_video,
        "pose_keypoints": pose_keypoints,
        "pose_tracks": pose_tracks,
        "pose_tracks_smooth_uq": pose_uq,
        "objects": objects_jsonl,
        "objects_semantic": objects_semantic_jsonl,
        "behavior_det": behavior_det_jsonl,
        "behavior_det_semantic": behavior_det_semantic_jsonl,
        "student_tracks": student_tracks_jsonl,
        "student_tracks_report": student_tracks_report,
        "behavior_unmatched": behavior_unmatched_jsonl,
        "pose_tracks_behavior_compat": pose_tracks_behavior_compat,
        "pose_tracks_behavior_compat_uq": pose_uq_behavior_compat,
        "actions_behavior": behavior_actions_jsonl,
        "actions_behavior_semantic": behavior_actions_semantic_jsonl,
        "actions_behavior_aug": actions_behavior_aug_jsonl,
        "actions_fusion_v2": actions_fusion_v2_jsonl,
        "actions_raw": actions_raw_jsonl,
        "actions": actions_jsonl,
        "transcript": transcript_jsonl,
        "asr_quality_report": asr_quality_report,
        "event_queries": event_queries_jsonl,
        "event_queries_fusion_v2": event_queries_fusion_v2_jsonl,
        "event_queries_visual_fallback": event_queries_visual_fallback_jsonl,
        "event_queries_used_for_align": event_queries_jsonl,
        "align_multimodal": aligned_json,
        "verifier_samples_train": verifier_samples_train,
        "verified_events": verified_events_jsonl,
        "verified_events_coherence_v3": verified_events_coherence_v3_jsonl,
        "fusion_contract_report": fusion_contract_report,
        "pipeline_contract_v2_report": pipeline_contract_report,
        "verifier_model": verifier_model,
        "llm_student_model": llm_student_model if llm_student_model is not None else Path("off"),
        "verifier_eval_report": verifier_eval_report,
        "verifier_calibration_report": verifier_calibration_report,
        "verifier_reliability_diagram": verifier_reliability_diagram,
        "per_person_sequences": per_person_json,
        "timeline_chart_png": timeline_png,
        "timeline_chart_json": timeline_json,
        "timeline_students_csv": timeline_students_csv,
        "student_id_map": student_id_map_json,
        "rear_row_pose_diagnostics": rear_row_pose_diagnostics,
        "rear_row_behavior_diagnostics": rear_row_behavior_diagnostics,
        "pose_behavior_video": pose_behavior_video,
        "pose_behavior_preview": pose_behavior_preview,
        "pose_behavior_report": pose_behavior_report,
        "rear_row_contact_sheet": rear_row_contact_sheet,
        "rear_row_contact_report": rear_row_contact_report,
    }
    runtime_config_snapshot: Dict[str, Any] = {
        "video": str(video_path),
        "out_dir": str(out_dir),
        "pose_model_arg": str(args.pose_model),
        "det_model_arg": str(args.det_model),
        "behavior_det_model_arg": str(args.behavior_det_model),
        "model_resolution_warnings": [],
    }

    def _write_current_manifest(status: str = "partial", failure_reason: str = "") -> None:
        case_id = args.case_id or video_path.stem
        if args.video_id:
            video_id = args.video_id
        else:
            view_code = args.view_code or args.view or "case"
            video_id = f"{view_code}__{case_id}"
        _write_pipeline_manifest(
            out_dir=out_dir,
            case_id=case_id,
            video_id=video_id,
            artifacts=manifest_artifacts,
            config_snapshot=dict(runtime_config_snapshot),
            status=status,
            failure_reason=failure_reason,
        )

    global PIPELINE_MANIFEST_WRITER
    PIPELINE_MANIFEST_WRITER = _write_current_manifest
    _refresh_pipeline_manifest("partial", "")

    py = args.py
    yolo_cfg_dir = (base_dir / ".ultralytics").resolve()
    yolo_cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_cfg_dir)
    print(f"[INFO] YOLO_CONFIG_DIR={yolo_cfg_dir}")

    track_backend = str(args.track_backend).strip().lower()
    use_pose_backend = track_backend in {"pose", "hybrid"}
    use_behavior_tracker = track_backend in {"behavior", "hybrid"}
    pose_model_abs = _auto_pose_model_path(args.pose_model) if use_pose_backend else ""
    pose_conf = max(0.0, min(1.0, float(args.pose_conf)))
    pose_track_person_conf = pose_conf if float(args.pose_track_person_conf) < 0.0 else max(0.0, min(1.0, float(args.pose_track_person_conf)))
    det_model_abs = resolve_model_or_fail(args.det_model)
    object_evidence_enabled = int(args.enable_object_evidence) == 1 or int(args.fuse_action_obj) == 1
    behavior_det_enabled = int(args.enable_behavior_det) == 1
    behavior_det_model_abs = ""
    if use_behavior_tracker and not behavior_det_enabled:
        if track_backend == "hybrid":
            print("[WARN] track_backend=hybrid requires behavior detection; falling back to track_backend=pose")
            track_backend = "pose"
            use_behavior_tracker = False
        else:
            raise ValueError("track_backend=behavior requires --enable_behavior_det 1")
    behavior_action_mode = str(args.behavior_action_mode).strip().lower()
    if behavior_det_enabled:
        resolved_behavior_model = try_resolve_model_reference(args.behavior_det_model, project_root=PROJECT_ROOT)
        if resolved_behavior_model is None:
            if str(args.behavior_det_model).replace("\\", "/") == DEFAULT_BEHAVIOR_DET_MODEL:
                warning = (
                    f"default behavior detector missing: {args.behavior_det_model}; "
                    "behavior detection disabled for this run"
                )
                print(f"[WARN] {warning}")
                runtime_config_snapshot["model_resolution_warnings"].append(warning)
                behavior_det_enabled = False
                if track_backend == "hybrid":
                    print("[WARN] track_backend=hybrid falling back to pose because behavior detector is unavailable")
                    track_backend = "pose"
                    use_behavior_tracker = False
                    use_pose_backend = True
                elif track_backend == "behavior":
                    raise FileNotFoundError(
                        "track_backend=behavior requires a valid behavior detector; "
                        f"missing default model: {args.behavior_det_model}"
                    )
            else:
                behavior_det_model_abs = resolve_model_or_fail(args.behavior_det_model)
        else:
            behavior_det_model_abs = resolved_behavior_model
    runtime_config_snapshot.update(
        {
            "pose_model": str(pose_model_abs),
            "det_model": str(det_model_abs),
            "behavior_det_enabled": bool(behavior_det_enabled),
            "behavior_det_model": str(behavior_det_model_abs),
            "track_backend": str(track_backend),
        }
    )
    fusion_contract_enabled = int(args.fusion_contract_v2) == 1
    fusion_scripts_dir = base_dir / "classroom_pipeline" / "fusion"
    semantic_taxonomy = resolve_video_path(args.semantic_taxonomy)

    if use_pose_backend:
        maybe_run(
            1,
            "pose video demo",
            [pose_demo_video],
            [video_path],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "pipeline/01_pose_video_demo.py",
                [
                    "--video",
                    str(video_path),
                    "--out",
                    str(pose_demo_video),
                    "--model",
                    pose_model_abs,
                    "--conf",
                    str(float(pose_conf)),
                    "--imgsz",
                    str(int(args.pose_imgsz)),
                    "--device",
                    str(args.pose_device),
                    "--half",
                    str(int(args.pose_half)),
                ],
            ),
            min_bytes=1024,
        )

        if str(args.sr_backend).strip().lower() not in {"off", "none"}:
            maybe_run(
                15,
                "rear row ROI SR cache",
                [rear_row_sr_report],
                [video_path],
                args.force,
                args.from_step,
                lambda: run_step(
                    py,
                    scripts_dir / "pipeline/02c_build_rear_roi_sr_cache.py",
                    [
                        "--video",
                        str(video_path),
                        "--out_dir",
                        str(rear_row_sr_cache_dir),
                        "--backend",
                        str(args.sr_backend),
                        "--scale",
                        str(float(args.sr_scale)),
                        "--roi",
                        str(args.pose_slice_roi),
                        "--preprocess",
                        str(args.sr_preprocess),
                        "--external_command",
                        str(args.sr_cache_external_command),
                        "--allow_unavailable",
                        str(int(args.sr_allow_unavailable)),
                    ],
                ),
                min_bytes=10,
            )
        else:
            print("[SKIP] step015 rear row ROI SR cache")

        maybe_run(
            2,
            "pose keypoints",
            [pose_keypoints, rear_row_pose_diagnostics] if str(args.pose_infer_mode) != "full" else [pose_keypoints],
            [video_path] + ([rear_row_sr_report] if str(args.sr_backend).strip().lower() not in {"off", "none"} else []),
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "pipeline/02_export_keypoints_jsonl.py",
                [
                    "--video",
                    str(video_path),
                    "--out",
                    str(pose_keypoints),
                    "--model",
                    pose_model_abs,
                    "--conf",
                    str(float(pose_conf)),
                    "--imgsz",
                    str(int(args.pose_imgsz)),
                    "--device",
                    str(args.pose_device),
                    "--half",
                    str(int(args.pose_half)),
                    "--infer_mode",
                    str(args.pose_infer_mode),
                    "--slice_grid",
                    str(args.pose_slice_grid),
                    "--slice_overlap",
                    str(float(args.pose_slice_overlap)),
                    "--slice_roi",
                    str(args.pose_slice_roi),
                    "--sr_backend",
                    str(args.sr_backend),
                    "--sr_scale",
                    str(float(args.sr_scale)),
                    "--sr_cache_dir",
                    str(rear_row_sr_cache_dir),
                    "--diagnostics_out",
                    str(rear_row_pose_diagnostics),
                    "--interpolate_occluded",
                    str(int(args.interpolate_occluded)),
                    "--occlusion_conf_thres",
                    str(float(args.occlusion_conf_thres)),
                ],
            ),
        )

        maybe_run(
            4,
            "track and smooth",
            [pose_tracks, pose_tracks_report],
            [pose_keypoints],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "pipeline/03_track_and_smooth.py",
                [
                    "--in",
                    str(pose_keypoints),
                    "--video",
                    str(video_path),
                    "--out",
                    str(pose_tracks),
                    "--person_conf_thres",
                    str(float(pose_track_person_conf)),
                    "--track_min_frames",
                    str(int(args.track_min_frames)),
                    "--track_min_frames_ratio",
                    str(float(args.track_min_frames_ratio)),
                    "--track_min_frames_min",
                    str(int(args.track_min_frames_min)),
                    "--track_max_lost_frames",
                    str(int(args.pose_track_max_lost_frames)),
                    "--track_iou_thres",
                    str(float(args.pose_track_iou_thres)),
                    "--track_max_center_dist_ratio",
                    str(float(args.pose_track_max_center_dist_ratio)),
                    "--track_max_dx_ratio",
                    str(float(args.pose_track_max_dx_ratio)),
                    "--track_height_penalty",
                    str(float(args.pose_track_height_penalty)),
                    "--seat_prior_mode",
                    str(args.pose_track_seat_prior_mode),
                    "--track_match_mode",
                    str(args.track_match_mode),
                    "--track_motion_model",
                    str(args.track_motion_model),
                    "--report_out",
                    str(pose_tracks_report),
                ],
            ),
        )

        maybe_run(
            35,
            "estimate uncertainty",
            [pose_uq],
            [pose_tracks],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "pipeline/03c_estimate_track_uncertainty.py",
                ["--in", str(pose_tracks), "--out", str(pose_uq), "--validate", "1"],
            ),
        )
    else:
        print("[SKIP] pose backend disabled; behavior tracker will emit pose-compatible artifacts")

    if object_evidence_enabled:
        maybe_run(
            45,
            "object detection evidence",
            [objects_jsonl],
            [video_path],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "pipeline/02b_export_objects_jsonl.py",
                [
                    "--video",
                    str(video_path),
                    "--out",
                    str(objects_jsonl),
                    "--model",
                    str(det_model_abs),
                    "--conf",
                    str(float(args.object_conf)),
                    "--iou",
                    str(float(args.object_iou)),
                    "--imgsz",
                    str(int(args.object_imgsz)),
                    "--stride",
                    str(int(args.object_stride)),
                    "--classes",
                    str(args.object_classes),
                ],
            ),
        )
        if fusion_contract_enabled:
            maybe_run(
                456,
                "semanticize object evidence",
                [objects_semantic_jsonl],
                [objects_jsonl],
                args.force,
                args.from_step,
                lambda: run_step(
                    py,
                    fusion_scripts_dir / "semanticize_objects.py",
                    [
                        "--in",
                        str(objects_jsonl),
                        "--out",
                        str(objects_semantic_jsonl),
                    ],
                ),
            )

    if use_pose_backend:
        def _run_actions():
            action_out = actions_raw_jsonl if object_evidence_enabled else actions_jsonl
            cmd = [
                "--video",
                str(video_path),
                "--pose",
                str(pose_tracks),
                "--out",
                str(action_out),
                "--model_mode",
                str(args.action_mode),
            ]
            if args.action_model:
                cmd += ["--model_weight", resolve_model_or_fail(args.action_model)]
            run_step(py, scripts_dir / "pipeline/05_slowfast_actions.py", cmd)

        maybe_run(
            5,
            "action recognition",
            [actions_raw_jsonl if object_evidence_enabled else actions_jsonl],
            [video_path, pose_tracks],
            args.force,
            args.from_step,
            _run_actions,
        )

        if object_evidence_enabled:
            maybe_run(
                55,
                "fuse actions with object evidence",
                [actions_jsonl],
                [actions_raw_jsonl, objects_jsonl],
                args.force,
                args.from_step,
                lambda: run_step(
                    py,
                    scripts_dir / "pipeline/05b_fuse_actions_with_objects.py",
                    [
                        "--actions",
                        str(actions_raw_jsonl),
                        "--objects",
                        str(objects_jsonl),
                        "--out",
                        str(actions_jsonl),
                        "--window",
                        str(float(args.fuse_window)),
                        "--alpha",
                        str(float(args.fuse_alpha)),
                        "--beta",
                        str(float(args.fuse_beta)),
                    ],
                ),
            )
    else:
        print("[SKIP] pose-based action recognition disabled for track_backend=behavior")

    if behavior_det_enabled:
        behavior_pose_keypoints_out = pose_keypoints if track_backend == "behavior" else pose_keypoints_behavior_compat
        behavior_pose_tracks_out = pose_tracks if track_backend == "behavior" else pose_tracks_behavior_compat
        behavior_pose_uq_out = pose_uq if track_backend == "behavior" else pose_uq_behavior_compat
        maybe_run(
            46,
            "behavior detection evidence",
            [behavior_det_jsonl, rear_row_behavior_diagnostics] if str(args.behavior_infer_mode) != "full" else [behavior_det_jsonl],
            [video_path] + ([rear_row_sr_report] if str(args.sr_backend).strip().lower() not in {"off", "none"} else []),
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "pipeline/02d_export_behavior_det_jsonl.py",
                [
                    "--video",
                    str(video_path),
                    "--out",
                    str(behavior_det_jsonl),
                    "--model",
                    str(behavior_det_model_abs),
                    "--conf",
                    str(float(args.behavior_det_conf)),
                    "--iou",
                    str(float(args.behavior_det_iou)),
                    "--imgsz",
                    str(int(args.behavior_det_imgsz)),
                    "--stride",
                    str(int(args.behavior_det_stride)),
                    "--device",
                    str(args.behavior_det_device),
                    "--infer_mode",
                    str(args.behavior_infer_mode),
                    "--slice_grid",
                    str(args.pose_slice_grid),
                    "--slice_overlap",
                    str(float(args.pose_slice_overlap)),
                    "--slice_roi",
                    str(args.pose_slice_roi),
                    "--sr_backend",
                    str(args.sr_backend),
                    "--sr_scale",
                    str(float(args.sr_scale)),
                    "--sr_cache_dir",
                    str(rear_row_sr_cache_dir),
                    "--diagnostics_out",
                    str(rear_row_behavior_diagnostics),
                ],
            ),
        )
        maybe_run(
            47,
            "convert behavior det to actions",
            [behavior_actions_jsonl],
            [behavior_det_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "pipeline/05c_behavior_det_to_actions.py",
                [
                    "--in",
                    str(behavior_det_jsonl),
                    "--out",
                    str(behavior_actions_jsonl),
                    "--fps",
                    str(float(args.fps)),
                    "--iou_thres",
                    str(float(args.behavior_track_iou_thres)),
                    "--max_gap_frames",
                    str(int(args.behavior_track_max_gap)),
                ],
            ),
        )
        if fusion_contract_enabled:
            maybe_run(
                471,
                "semanticize behavior detections",
                [behavior_det_semantic_jsonl],
                [behavior_det_jsonl, semantic_taxonomy],
                args.force,
                args.from_step,
                lambda: run_step(
                    py,
                    fusion_scripts_dir / "semanticize_behavior_det.py",
                    [
                        "--in",
                        str(behavior_det_jsonl),
                        "--out",
                        str(behavior_det_semantic_jsonl),
                        "--taxonomy",
                        str(semantic_taxonomy),
                        "--strict",
                        str(int(args.fusion_contract_strict)),
                    ],
                ),
            )
            if use_behavior_tracker:
                maybe_run(
                    4711,
                    "track behavior students",
                    [student_tracks_jsonl] + ([pose_keypoints, pose_tracks, pose_uq] if track_backend == "behavior" else []),
                    [behavior_det_semantic_jsonl] + ([pose_tracks] if use_pose_backend else []),
                    args.force,
                    args.from_step,
                    lambda: run_step(
                        py,
                        scripts_dir / "pipeline/03e_track_behavior_students.py",
                        [
                            "--in",
                            str(behavior_det_semantic_jsonl),
                            "--out",
                            str(student_tracks_jsonl),
                            "--track_backend",
                            str("hybrid" if track_backend == "hybrid" else "behavior"),
                            "--pose_tracks",
                            str(pose_tracks if use_pose_backend else ""),
                            "--behavior_unmatched_out",
                            str(behavior_unmatched_jsonl if track_backend == "hybrid" else ""),
                            "--pose_keypoints_compat_out",
                            str(behavior_pose_keypoints_out),
                            "--pose_tracks_compat_out",
                            str(behavior_pose_tracks_out),
                            "--pose_uq_out",
                            str(behavior_pose_uq_out),
                            "--report",
                            str(student_tracks_report),
                            "--fps",
                            str(float(args.fps)),
                            "--tracker",
                            str(args.behavior_tracker),
                            "--same_person_iou",
                            str(float(args.behavior_same_person_iou)),
                            "--iou_thres",
                            str(float(args.behavior_track_iou_thres)),
                            "--track_low_thresh",
                            str(float(args.behavior_track_low_thresh)),
                            "--new_track_thresh",
                            str(float(args.behavior_new_track_thresh)),
                            "--track_buffer",
                            str(int(args.behavior_track_buffer)),
                            "--min_track_frames",
                            str(int(args.behavior_track_min_frames)),
                            "--link_pose",
                            str(1 if track_backend == "hybrid" else 0),
                            "--link_iou",
                            str(float(args.behavior_link_pose_iou)),
                            "--hybrid_match_mode",
                            str(args.behavior_hybrid_match_mode),
                            "--unlinked_track_offset",
                            str(int(args.behavior_unlinked_track_offset)),
                            "--validate_uq",
                            "1",
                        ],
                    ),
                )
            maybe_run(
                472,
                "convert semantic behavior det to actions v2",
                [behavior_actions_semantic_jsonl],
                [student_tracks_jsonl if use_behavior_tracker else behavior_det_semantic_jsonl],
                args.force,
                args.from_step,
                lambda: run_step(
                    py,
                    fusion_scripts_dir / "behavior_det_to_actions_v2.py",
                    [
                        "--in",
                        str(student_tracks_jsonl if use_behavior_tracker else behavior_det_semantic_jsonl),
                        "--out",
                        str(behavior_actions_semantic_jsonl),
                        "--fps",
                        str(float(args.fps)),
                        "--iou_thres",
                        str(float(args.behavior_track_iou_thres)),
                        "--max_gap_frames",
                        str(int(args.behavior_track_max_gap)),
                    ],
                ),
            )
            if (
                int(args.enable_pose_behavior_video) == 1
                and use_pose_backend
                and use_behavior_tracker
                and track_backend == "hybrid"
            ):
                maybe_run(
                    473,
                    "pose behavior fusion video",
                    [pose_behavior_video, pose_behavior_preview, pose_behavior_report],
                    [video_path, pose_tracks, student_tracks_jsonl, behavior_actions_semantic_jsonl, behavior_unmatched_jsonl],
                    args.force,
                    args.from_step,
                    lambda: run_step(
                        py,
                        scripts_dir / "pipeline/06_overlay_pose_behavior_video.py",
                        [
                            "--video",
                            str(video_path),
                            "--pose_tracks",
                            str(pose_tracks),
                            "--student_tracks",
                            str(student_tracks_jsonl),
                            "--actions",
                            str(behavior_actions_semantic_jsonl),
                            "--unmatched_behaviors",
                            str(behavior_unmatched_jsonl),
                            "--out",
                            str(pose_behavior_video),
                            "--preview_out",
                            str(pose_behavior_preview),
                            "--report",
                            str(pose_behavior_report),
                            "--label_source",
                            str(args.pose_behavior_video_label_source),
                            "--show_behavior_bbox",
                            str(int(args.pose_behavior_video_show_behavior_bbox)),
                            "--min_conf",
                            str(float(args.pose_behavior_video_min_conf)),
                            "--keypoint_conf",
                            str(float(args.pose_behavior_video_keypoint_conf)),
                            "--show_unmatched_label",
                            str(int(args.pose_behavior_video_show_unmatched_label)),
                            "--compact_label",
                            str(int(args.pose_behavior_video_compact_label)),
                            "--show_unlinked_behavior_bbox",
                            str(int(args.pose_behavior_video_show_unlinked_behavior_bbox)),
                            "--show_unlinked_behavior_label",
                            str(int(args.pose_behavior_video_show_unlinked_behavior_label)),
                            "--show_unlinked_behavior_legend",
                            str(int(args.pose_behavior_video_show_unlinked_behavior_legend)),
                            "--unlinked_behavior_min_conf",
                            str(float(args.pose_behavior_video_unlinked_behavior_min_conf)),
                        ],
                    ),
                    min_bytes=1024,
                )
                maybe_run(
                    474,
                    "rear row compare contact sheet",
                    [rear_row_contact_sheet, rear_row_contact_report],
                    [pose_demo_video, pose_behavior_video, pose_keypoints],
                    args.force,
                    args.from_step,
                    lambda: run_step(
                        py,
                        scripts_dir / "pipeline/06d_build_rear_row_contact_sheet.py",
                        [
                            "--pose_demo_video",
                            str(pose_demo_video),
                            "--fusion_video",
                            str(pose_behavior_video),
                            "--pose_jsonl",
                            str(pose_keypoints),
                            "--out",
                            str(rear_row_contact_sheet),
                            "--report",
                            str(rear_row_contact_report),
                            "--frames",
                            "0,25,50,100",
                        ],
                    ),
                    min_bytes=10,
                )

    actions_for_downstream = behavior_actions_semantic_jsonl if track_backend == "behavior" else actions_jsonl
    if behavior_det_enabled and behavior_action_mode == "behavior_only":
        actions_for_downstream = behavior_actions_semantic_jsonl if fusion_contract_enabled else behavior_actions_jsonl
    elif behavior_det_enabled and behavior_action_mode == "append" and use_pose_backend:
        maybe_run(
            58,
            "merge rule actions with behavior actions",
            [actions_behavior_aug_jsonl],
            [actions_jsonl, behavior_actions_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "pipeline/05d_merge_action_sources.py",
                [
                    "--primary",
                    str(actions_jsonl),
                    "--extra",
                    str(behavior_actions_jsonl),
                    "--out",
                    str(actions_behavior_aug_jsonl),
                    "--extra_track_offset",
                    str(int(args.behavior_extra_track_offset)),
                ],
            ),
        )
        actions_for_downstream = actions_behavior_aug_jsonl

    if fusion_contract_enabled:
        fusion_primary_actions = behavior_actions_semantic_jsonl if track_backend == "behavior" else actions_jsonl
        fusion_behavior_actions = "" if track_backend == "behavior" else str(behavior_actions_semantic_jsonl if behavior_det_enabled else "")
        fusion_behavior_track_mode = "linked" if track_backend == "hybrid" and use_behavior_tracker else "offset"
        fusion_behavior_track_offset = 0 if fusion_behavior_track_mode == "linked" else int(args.behavior_extra_track_offset)
        maybe_run(
            59,
            "merge fusion contract v2 actions",
            [actions_fusion_v2_jsonl],
            [
                fusion_primary_actions,
                behavior_actions_semantic_jsonl if behavior_det_enabled else actions_jsonl,
                objects_semantic_jsonl if object_evidence_enabled else actions_jsonl,
                semantic_taxonomy,
            ],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                fusion_scripts_dir / "merge_fusion_actions_v2.py",
                [
                    "--actions",
                    str(fusion_primary_actions),
                    "--behavior_actions",
                    str(fusion_behavior_actions),
                    "--objects",
                    str(objects_semantic_jsonl if object_evidence_enabled else ""),
                    "--out",
                    str(actions_fusion_v2_jsonl),
                    "--taxonomy",
                    str(semantic_taxonomy),
                    "--behavior_track_offset",
                    str(int(fusion_behavior_track_offset)),
                    "--behavior_track_mode",
                    str(fusion_behavior_track_mode),
                    "--object_window",
                    str(float(args.fusion_object_window)),
                    "--object_beta",
                    str(float(args.fusion_object_beta)),
                    "--strict",
                    str(int(args.fusion_contract_strict)),
                ],
            ),
        )
        actions_for_downstream = actions_fusion_v2_jsonl

    def _run_asr():
        def _run_api() -> None:
            run_step(
                py,
                scripts_dir / "pipeline/06_api_asr_realtime.py",
                ["--video", str(video_path), "--out_dir", str(out_dir), "--asr_model", str(args.asr_model)],
            )

        def _unlink_transcript_if_exists() -> None:
            try:
                if transcript_jsonl.exists() and transcript_jsonl.is_file():
                    transcript_jsonl.unlink()
            except Exception:
                pass

        def _read_asr_error_reason() -> str:
            try:
                if not asr_quality_report.exists():
                    return ""
                with asr_quality_report.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                return str(obj.get("error_reason", "") or "")
            except Exception:
                return ""

        def _run_whisper_once(device: str, compute_type: str) -> None:
            _unlink_transcript_if_exists()
            run_step(
                py,
                scripts_dir / "pipeline/06_asr_whisper_to_jsonl.py",
                [
                    "--video",
                    str(video_path),
                    "--out_dir",
                    str(out_dir),
                    "--lang",
                    str(args.asr_lang),
                    "--model",
                    str(args.whisper_model),
                    "--device",
                    str(device),
                    "--compute_type",
                    str(compute_type),
                    "--beam_size",
                    str(int(args.whisper_beam_size)),
                    "--vad_filter",
                    str(int(args.whisper_vad_filter)),
                    "--condition_on_previous_text",
                    str(int(args.whisper_condition_on_previous_text)),
                    "--audio_filter",
                    str(args.whisper_audio_filter),
                    "--min_avg_logprob",
                    str(args.whisper_min_avg_logprob),
                    "--max_no_speech_prob",
                    str(args.whisper_max_no_speech_prob),
                    "--max_compression_ratio",
                    str(args.whisper_max_compression_ratio),
                    "--min_text_chars",
                    str(int(args.whisper_min_text_chars)),
                ],
            )
            reason = _read_asr_error_reason()
            if _is_placeholder_transcript(transcript_jsonl) and reason in {
                "whisper_inference_failed",
                "whisper_decode_failed",
            }:
                raise RuntimeError(f"whisper {device}/{compute_type} produced placeholder: {reason}")

        def _run_whisper() -> None:
            primary_device = str(args.whisper_device).strip().lower()
            primary_compute = str(args.whisper_compute_type).strip()
            try:
                _run_whisper_once(primary_device, primary_compute)
                return
            except Exception as e1:
                if int(args.whisper_retry_on_fail) != 1 or primary_device != "cuda":
                    raise
                print(f"[WARN] whisper primary failed on cuda ({e1}); trying recovery chain...")

            retry_compute = str(args.whisper_retry_compute_type).strip()
            if retry_compute and retry_compute != primary_compute:
                try:
                    _run_whisper_once(primary_device, retry_compute)
                    print(f"[INFO] whisper recovered by retry compute_type={retry_compute}")
                    return
                except Exception as e2:
                    print(f"[WARN] whisper retry compute_type={retry_compute} failed ({e2})")

            fallback_device = str(args.whisper_fallback_device).strip().lower()
            if fallback_device:
                fallback_compute = str(args.whisper_fallback_compute_type).strip() or primary_compute
                _run_whisper_once(fallback_device, fallback_compute)
                print(f"[INFO] whisper recovered by fallback device={fallback_device} compute_type={fallback_compute}")
                return

            raise RuntimeError("whisper recovery chain exhausted")

        def _run_openai() -> None:
            run_step(
                py,
                scripts_dir / "pipeline/06c_asr_openai_to_jsonl.py",
                [
                    "--video",
                    str(video_path),
                    "--out_dir",
                    str(out_dir),
                    "--model",
                    str(args.openai_asr_model),
                    "--lang",
                    str(args.asr_lang),
                    "--api_key_env",
                    str(args.openai_api_key_env),
                    "--timeout_sec",
                    str(float(args.openai_timeout_sec)),
                    "--max_file_mb",
                    str(float(args.openai_max_file_mb)),
                ],
            )

        if args.asr_backend == "api":
            _run_api()
            return
        if args.asr_backend == "whisper":
            _run_whisper()
            return
        if args.asr_backend == "openai":
            _run_openai()
            return

        # auto: try api first, fallback whisper when still placeholder.
        _run_api()
        if _is_placeholder_transcript(transcript_jsonl):
            print("[INFO] ASR placeholder detected, fallback to whisper...")
            _run_whisper()

    maybe_run(
        6,
        "asr transcript",
        [transcript_jsonl, asr_quality_report],
        [video_path],
        args.force,
        args.from_step,
        _run_asr,
    )

    maybe_run(
        65,
        "extract event queries",
        [event_queries_jsonl],
        [transcript_jsonl],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "pipeline/06b_event_query_extraction.py",
            ["--transcript", str(transcript_jsonl), "--out", str(event_queries_jsonl), "--validate", "1"],
        ),
    )

    event_queries_for_downstream = event_queries_jsonl
    if fusion_contract_enabled:
        maybe_run(
            656,
            "build fusion contract v2 event queries",
            [event_queries_fusion_v2_jsonl],
            [event_queries_jsonl, actions_for_downstream],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                fusion_scripts_dir / "build_event_queries_fusion_v2.py",
                [
                    "--event_queries",
                    str(event_queries_jsonl),
                    "--actions",
                    str(actions_for_downstream),
                    "--out",
                    str(event_queries_fusion_v2_jsonl),
                    "--visual_out",
                    str(event_queries_visual_fallback_jsonl),
                    "--min_asr_queries",
                    str(int(args.fusion_min_asr_queries)),
                    "--visual_topk",
                    str(int(args.fusion_visual_topk)),
                    "--visual_min_conf",
                    str(float(args.fusion_visual_min_conf)),
                    "--validate",
                    "1",
                ],
            ),
        )
        event_queries_for_downstream = event_queries_fusion_v2_jsonl

    maybe_run(
        66,
        "adaptive multimodal align",
        [aligned_json],
        [event_queries_for_downstream, actions_for_downstream, pose_uq],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "pipeline/xx_align_multimodal.py",
            [
                "--event_queries",
                str(event_queries_for_downstream),
                "--actions",
                str(actions_for_downstream),
                "--pose_uq",
                str(pose_uq),
                "--out",
                str(aligned_json),
                "--require_semantic",
                str(1 if fusion_contract_enabled else 0),
            ],
        ),
    )

    contract_samples = convert_to_contract_samples(
        build_training_samples(
            event_queries_path=event_queries_for_downstream,
            aligned_path=aligned_json,
            actions_path=actions_for_downstream,
        )
    )
    save_training_samples(verifier_samples_train, contract_samples)

    if int(args.train_verifier) == 1:
        maybe_run(
            67,
            "train verifier",
            [verifier_model, verifier_report_raw],
            [event_queries_for_downstream, aligned_json, actions_for_downstream],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                PROJECT_ROOT / "classroom_pipeline" / "verifier" / "train.py",
                [
                    "--event_queries",
                    str(event_queries_for_downstream),
                    "--aligned",
                    str(aligned_json),
                    "--actions",
                    str(actions_for_downstream),
                    "--out_model",
                    str(verifier_model),
                    "--out_report",
                    str(verifier_report_raw),
                    "--out_samples",
                    str(verifier_samples_raw),
                ],
            ),
        )

        if verifier_samples_raw.exists():
            save_training_samples(verifier_samples_train, convert_to_contract_samples(_iter_jsonl(verifier_samples_raw)))

    verifier_inputs = [event_queries_for_downstream, aligned_json, pose_uq, actions_for_downstream, verifier_model]
    if llm_student_model is not None:
        verifier_inputs.append(llm_student_model)

    maybe_run(
        70,
        "dual verification",
        [verified_events_jsonl],
        verifier_inputs,
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "pipeline/07_dual_verification.py",
            [
                "--actions",
                str(actions_for_downstream),
                "--event_queries",
                str(event_queries_for_downstream),
                "--pose_uq",
                str(pose_uq),
                "--aligned",
                str(aligned_json),
                "--out",
                str(verified_events_jsonl),
                "--verifier_model",
                str(verifier_model if verifier_model.exists() else ""),
                "--llm_student_model",
                str(llm_student_model if llm_student_model is not None else "off"),
                "--per_person_out",
                str(per_person_json),
                "--validate",
                "1",
                "--require_semantic",
                str(1 if fusion_contract_enabled else 0),
            ],
        ),
    )

    maybe_run(
        71,
        "verifier evaluation report",
        [verifier_eval_report],
        [verified_events_jsonl],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            PROJECT_ROOT / "classroom_pipeline" / "verifier" / "eval.py",
            [
                "--verified",
                str(verified_events_jsonl),
                "--out",
                str(verifier_eval_report),
                "--split",
                "val",
                "--target_field",
                str(args.eval_target_field),
            ],
        ),
    )

    maybe_run(
        72,
        "verifier calibration report",
        [verifier_calibration_report],
        [verified_events_jsonl],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            PROJECT_ROOT / "classroom_pipeline" / "verifier" / "calibration.py",
            [
                "--verified",
                str(verified_events_jsonl),
                "--out",
                str(verifier_calibration_report),
                "--split",
                "val",
                "--target_field",
                str(args.calibration_target_field),
                "--prob_field",
                str(args.calibration_prob_field),
                "--num_bins",
                str(int(args.calibration_num_bins)),
                "--disable_temperature_scaling",
                str(int(args.disable_temperature_scaling)),
                "--diagram_out",
                str(verifier_reliability_diagram),
            ],
        ),
    )

    coherence_objects = objects_semantic_jsonl if objects_semantic_jsonl.exists() else objects_jsonl
    coherence_inputs = [
        verified_events_jsonl,
        aligned_json,
        event_queries_for_downstream,
        asr_quality_report,
    ]
    if coherence_objects.exists():
        coherence_inputs.append(coherence_objects)
    maybe_run(
        73,
        "multimodal coherence diagnosis",
        [verified_events_coherence_v3_jsonl],
        coherence_inputs,
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "pipeline/08_coherence_diagnosis.py",
            [
                "--verified",
                str(verified_events_jsonl),
                "--aligned",
                str(aligned_json),
                "--event_queries",
                str(event_queries_for_downstream),
                "--asr_quality",
                str(asr_quality_report),
                "--objects",
                str(coherence_objects) if coherence_objects.exists() else "",
                "--case_id",
                str(args.case_id or name),
                "--fps",
                str(float(args.fps)),
                "--out",
                str(verified_events_coherence_v3_jsonl),
                "--validate",
                "1",
            ],
        ),
    )

    if not args.skip_vis:
        maybe_run(
            80,
            "timeline visualization",
            [timeline_png, timeline_json, timeline_students_csv, student_id_map_json],
            [verified_events_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "pipeline/10_visualize_timeline.py",
                [
                    "--src",
                    str(per_person_json),
                    "--out",
                    str(timeline_png),
                    "--gap_tol",
                    str(args.viz_gap_tol),
                    "--min_dur",
                    str(args.viz_min_dur),
                    "--fps",
                    str(args.fps),
                ],
            ),
        )

    if fusion_contract_enabled:
        maybe_run(
            90,
            "check fusion contract v2",
            [fusion_contract_report],
            [actions_for_downstream, event_queries_for_downstream, aligned_json, verified_events_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                fusion_scripts_dir / "check_fusion_contract.py",
                [
                    "--output_dir",
                    str(out_dir),
                    "--report",
                    str(fusion_contract_report),
                    "--strict",
                    str(int(args.fusion_contract_strict)),
                ],
            ),
        )
        maybe_run(
            91,
            "check full pipeline contract v2",
            [pipeline_contract_report],
            [
                pose_keypoints,
                pose_tracks,
                behavior_det_semantic_jsonl,
                actions_for_downstream,
                event_queries_for_downstream,
                aligned_json,
                verified_events_jsonl,
                per_person_json,
            ] + ([] if args.skip_vis else [timeline_png, timeline_students_csv, student_id_map_json]),
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                fusion_scripts_dir / "check_pipeline_contract_v2.py",
                [
                    "--output_dir",
                    str(out_dir),
                    "--report",
                    str(pipeline_contract_report),
                    "--strict",
                    str(int(args.fusion_contract_strict)),
                    "--require_timeline",
                    "0" if args.skip_vis else "1",
                    "--track_backend",
                    str(track_backend),
                ],
            ),
        )

    if int(args.export_compat) == 1:
        case_id = args.case_id or name
        if args.video_id:
            video_id = args.video_id
        else:
            view_code = args.view_code or args.view or "case"
            video_id = f"{view_code}__{case_id}"
        _write_pipeline_manifest(
            out_dir=out_dir,
            case_id=case_id,
            video_id=video_id,
            artifacts={
                "pose_demo_video": pose_demo_video,
                "pose_behavior_video": pose_behavior_video,
                "pose_behavior_video_preview": pose_behavior_preview,
                "pose_behavior_video_report": pose_behavior_report,
                "rear_row_compare_contact_sheet": rear_row_contact_sheet,
                "rear_row_compare_contact_sheet_report": rear_row_contact_report,
                "rear_row_pose_diagnostics": rear_row_pose_diagnostics,
                "rear_row_behavior_diagnostics": rear_row_behavior_diagnostics,
                "rear_row_sr_cache_report": rear_row_sr_report,
                "pose_keypoints": pose_keypoints,
                "pose_tracks_smooth": pose_tracks,
                "pose_tracks_report": pose_tracks_report,
                "pose_tracks_smooth_uq": pose_uq,
                "objects": objects_jsonl,
                "objects_semantic": objects_semantic_jsonl,
                "behavior_det": behavior_det_jsonl,
                "behavior_det_semantic": behavior_det_semantic_jsonl,
                "student_tracks": student_tracks_jsonl,
                "student_tracks_report": student_tracks_report,
                "behavior_unmatched": behavior_unmatched_jsonl,
                "pose_keypoints_behavior_compat": pose_keypoints_behavior_compat,
                "pose_tracks_behavior_compat": pose_tracks_behavior_compat,
                "pose_tracks_behavior_compat_uq": pose_uq_behavior_compat,
                "actions_behavior": behavior_actions_jsonl,
                "actions_behavior_semantic": behavior_actions_semantic_jsonl,
                "actions_behavior_aug": actions_behavior_aug_jsonl,
                "actions_fusion_v2": actions_fusion_v2_jsonl,
                "actions_raw": actions_raw_jsonl,
                "actions": actions_jsonl,
                "actions_used_for_align": actions_for_downstream,
                "transcript": transcript_jsonl,
                "asr_quality_report": asr_quality_report,
                "event_queries": event_queries_jsonl,
                "event_queries_visual_fallback": event_queries_visual_fallback_jsonl,
                "event_queries_fusion_v2": event_queries_fusion_v2_jsonl,
                "event_queries_used_for_align": event_queries_for_downstream,
                "align_multimodal": aligned_json,
                "verifier_samples_train": verifier_samples_train,
                "verified_events": verified_events_jsonl,
                "verified_events_coherence_v3": verified_events_coherence_v3_jsonl,
                "fusion_contract_report": fusion_contract_report,
                "pipeline_contract_v2_report": pipeline_contract_report,
                "verifier_model": verifier_model,
                "llm_student_model": llm_student_model if llm_student_model is not None else Path("off"),
                "verifier_eval_report": verifier_eval_report,
                "verifier_calibration_report": verifier_calibration_report,
                "verifier_reliability_diagram": verifier_reliability_diagram,
                "per_person_sequences": per_person_json,
                "timeline_chart_png": timeline_png,
                "timeline_chart_json": timeline_json,
                "timeline_students_csv": timeline_students_csv,
                "student_id_map": student_id_map_json,
            },
            config_snapshot={
                "from_step": int(args.from_step),
                "to_step": int(args.to_step),
                "track_backend": str(track_backend),
                "pose_model": str(pose_model_abs),
                "pose_conf": float(pose_conf),
                "pose_imgsz": int(args.pose_imgsz),
                "pose_device": str(args.pose_device),
                "pose_half": int(args.pose_half),
                "pose_track_person_conf": float(pose_track_person_conf),
                "pose_track_max_lost_frames": int(args.pose_track_max_lost_frames),
                "pose_track_iou_thres": float(args.pose_track_iou_thres),
                "pose_track_max_center_dist_ratio": float(args.pose_track_max_center_dist_ratio),
                "pose_track_max_dx_ratio": float(args.pose_track_max_dx_ratio),
                "pose_track_height_penalty": float(args.pose_track_height_penalty),
                "pose_track_seat_prior_mode": str(args.pose_track_seat_prior_mode),
                "track_match_mode": str(args.track_match_mode),
                "track_motion_model": str(args.track_motion_model),
                "pose_infer_mode": str(args.pose_infer_mode),
                "pose_slice_grid": str(args.pose_slice_grid),
                "pose_slice_overlap": float(args.pose_slice_overlap),
                "pose_slice_roi": str(args.pose_slice_roi),
                "sr_backend": str(args.sr_backend),
                "sr_scale": float(args.sr_scale),
                "sr_preprocess": str(args.sr_preprocess),
                "sr_cache_dir": str(rear_row_sr_cache_dir),
                "sr_cache_external_command": str(args.sr_cache_external_command),
                "sr_allow_unavailable": int(args.sr_allow_unavailable),
                "train_verifier": int(args.train_verifier),
                "llm_student_model": str(llm_student_model if llm_student_model is not None else "off"),
                "action_mode": str(args.action_mode),
                "asr_backend": str(args.asr_backend),
                "asr_model": str(args.asr_model),
                "asr_lang": str(args.asr_lang),
                "whisper_model": str(args.whisper_model),
                "whisper_device": str(args.whisper_device),
                "whisper_compute_type": str(args.whisper_compute_type),
                "whisper_beam_size": int(args.whisper_beam_size),
                "whisper_vad_filter": int(args.whisper_vad_filter),
                "whisper_condition_on_previous_text": int(args.whisper_condition_on_previous_text),
                "whisper_audio_filter": str(args.whisper_audio_filter),
                "whisper_min_avg_logprob": str(args.whisper_min_avg_logprob),
                "whisper_max_no_speech_prob": str(args.whisper_max_no_speech_prob),
                "whisper_max_compression_ratio": str(args.whisper_max_compression_ratio),
                "whisper_min_text_chars": int(args.whisper_min_text_chars),
                "whisper_retry_on_fail": int(args.whisper_retry_on_fail),
                "whisper_retry_compute_type": str(args.whisper_retry_compute_type),
                "whisper_fallback_device": str(args.whisper_fallback_device),
                "whisper_fallback_compute_type": str(args.whisper_fallback_compute_type),
                "openai_asr_model": str(args.openai_asr_model),
                "openai_api_key_env": str(args.openai_api_key_env),
                "openai_timeout_sec": float(args.openai_timeout_sec),
                "openai_max_file_mb": float(args.openai_max_file_mb),
                "det_model": str(det_model_abs),
                "object_evidence_enabled": bool(object_evidence_enabled),
                "object_conf": float(args.object_conf),
                "object_iou": float(args.object_iou),
                "object_imgsz": int(args.object_imgsz),
                "object_stride": int(args.object_stride),
                "object_classes": str(args.object_classes),
                "behavior_det_enabled": bool(behavior_det_enabled),
                "behavior_det_model": str(behavior_det_model_abs),
                "behavior_det_conf": float(args.behavior_det_conf),
                "behavior_det_iou": float(args.behavior_det_iou),
                "behavior_det_imgsz": int(args.behavior_det_imgsz),
                "behavior_det_stride": int(args.behavior_det_stride),
                "behavior_det_device": str(args.behavior_det_device),
                "behavior_infer_mode": str(args.behavior_infer_mode),
                "behavior_action_mode": str(behavior_action_mode),
                "behavior_tracker": str(args.behavior_tracker),
                "behavior_same_person_iou": float(args.behavior_same_person_iou),
                "behavior_track_iou_thres": float(args.behavior_track_iou_thres),
                "behavior_track_max_gap": int(args.behavior_track_max_gap),
                "behavior_track_buffer": int(args.behavior_track_buffer),
                "behavior_track_low_thresh": float(args.behavior_track_low_thresh),
                "behavior_new_track_thresh": float(args.behavior_new_track_thresh),
                "behavior_track_min_frames": int(args.behavior_track_min_frames),
                "behavior_link_pose_iou": float(args.behavior_link_pose_iou),
                "behavior_hybrid_match_mode": str(args.behavior_hybrid_match_mode),
                "behavior_unlinked_track_offset": int(args.behavior_unlinked_track_offset),
                "behavior_extra_track_offset": int(args.behavior_extra_track_offset),
                "enable_pose_behavior_video": int(args.enable_pose_behavior_video),
                "pose_behavior_video_out": str(args.pose_behavior_video_out),
                "pose_behavior_video_label_source": str(args.pose_behavior_video_label_source),
                "pose_behavior_video_show_behavior_bbox": int(args.pose_behavior_video_show_behavior_bbox),
                "pose_behavior_video_min_conf": float(args.pose_behavior_video_min_conf),
                "pose_behavior_video_keypoint_conf": float(args.pose_behavior_video_keypoint_conf),
                "pose_behavior_video_show_unmatched_label": int(args.pose_behavior_video_show_unmatched_label),
                "pose_behavior_video_compact_label": int(args.pose_behavior_video_compact_label),
                "pose_behavior_video_show_unlinked_behavior_bbox": int(args.pose_behavior_video_show_unlinked_behavior_bbox),
                "pose_behavior_video_show_unlinked_behavior_label": int(args.pose_behavior_video_show_unlinked_behavior_label),
                "pose_behavior_video_show_unlinked_behavior_legend": int(args.pose_behavior_video_show_unlinked_behavior_legend),
                "pose_behavior_video_unlinked_behavior_min_conf": float(args.pose_behavior_video_unlinked_behavior_min_conf),
                "fusion_contract_v2": bool(fusion_contract_enabled),
                "semantic_taxonomy": str(semantic_taxonomy),
                "fusion_contract_strict": int(args.fusion_contract_strict),
                "fusion_min_asr_queries": int(args.fusion_min_asr_queries),
                "fusion_visual_topk": int(args.fusion_visual_topk),
                "fusion_visual_min_conf": float(args.fusion_visual_min_conf),
                "fusion_object_window": float(args.fusion_object_window),
                "fusion_object_beta": float(args.fusion_object_beta),
                "actions_used_for_align": str(actions_for_downstream),
                "event_queries_used_for_align": str(event_queries_for_downstream),
                "fuse_window": float(args.fuse_window),
                "fuse_alpha": float(args.fuse_alpha),
                "fuse_beta": float(args.fuse_beta),
                "interaction_model": str(args.interaction_model),
                "enable_mllm": int(args.enable_mllm),
                "track_min_frames": int(args.track_min_frames),
                "track_min_frames_ratio": float(args.track_min_frames_ratio),
                "track_min_frames_min": int(args.track_min_frames_min),
                "eval_target_field": str(args.eval_target_field),
                "calibration_target_field": str(args.calibration_target_field),
                "calibration_prob_field": str(args.calibration_prob_field),
                "calibration_num_bins": int(args.calibration_num_bins),
                "disable_temperature_scaling": int(args.disable_temperature_scaling),
            },
        )

        # Keep legacy aliases for downstream tools that still consume old names.
        if pose_tracks.exists():
            shutil.copy2(pose_tracks, out_dir / f"{case_id}.jsonl")
        if timeline_json.exists():
            shutil.copy2(timeline_json, out_dir / "timeline_viz.json")

        if fusion_contract_enabled and _step_rank(int(args.from_step)) <= _step_rank(91) <= _step_rank(int(args.to_step)):
            # The manifest is written after most artifacts exist, so refresh the final contract once
            # more to let the report include pipeline_manifest.json in the same run.
            run_step(
                py,
                fusion_scripts_dir / "check_pipeline_contract_v2.py",
                [
                    "--output_dir",
                    str(out_dir),
                    "--report",
                    str(pipeline_contract_report),
                    "--strict",
                    str(int(args.fusion_contract_strict)),
                    "--require_timeline",
                    "0" if args.skip_vis else "1",
                    "--track_backend",
                    str(track_backend),
                ],
            )

    print("\n[DONE] formal pipeline completed")
    print(f"Output dir             : {out_dir}")
    print(f"Track backend          : {track_backend}")
    if use_pose_backend:
        print(f"Pose demo video        : {pose_demo_video}")
        if str(args.sr_backend).strip().lower() not in {"off", "none"}:
            print(f"Rear ROI SR cache      : {rear_row_sr_report}")
        if int(args.enable_pose_behavior_video) == 1 and pose_behavior_video.exists():
            print(f"Pose behavior video    : {pose_behavior_video}")
    print(f"Pose tracks UQ         : {pose_uq}")
    if object_evidence_enabled:
        print(f"Object evidence        : {objects_jsonl}")
    if behavior_det_enabled:
        print(f"Behavior detections    : {behavior_det_jsonl}")
        if use_behavior_tracker:
            print(f"Student tracks         : {student_tracks_jsonl}")
        print(f"Behavior actions       : {behavior_actions_jsonl}")
        print(f"Behavior action mode   : {behavior_action_mode}")
        print(f"Actions for align      : {actions_for_downstream}")
    if fusion_contract_enabled:
        print(f"Fusion v2 actions      : {actions_fusion_v2_jsonl}")
        print(f"Fusion v2 event queries: {event_queries_fusion_v2_jsonl}")
        print(f"Fusion v2 report       : {fusion_contract_report}")
        print(f"Pipeline contract v2   : {pipeline_contract_report}")
    print(f"Event queries          : {event_queries_jsonl}")
    print(f"Verified events        : {verified_events_jsonl}")
    print(f"Verifier eval report   : {verifier_eval_report}")
    print(f"Verifier calibration   : {verifier_calibration_report}")
    print(f"Reliability diagram    : {verifier_reliability_diagram}")
    print(f"Verifier model         : {verifier_model if verifier_model.exists() else 'not_used'}")
    print(f"LLM student model      : {llm_student_model if llm_student_model is not None else 'off'}")
    print(f"Timeline chart         : {timeline_png}")
    if not args.skip_vis:
        print(f"Timeline students CSV  : {timeline_students_csv}")
        print(f"Student ID map         : {student_id_map_json}")


if __name__ == "__main__":
    main()
