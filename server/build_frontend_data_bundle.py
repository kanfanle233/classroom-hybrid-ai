import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "2026-05-01+frontend_bundle_v2"


def _git_head_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _resolve(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
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
                rows.append(row)
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if value in [None, ""]:
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except Exception:
        return default


def _artifact_path(base_dir: Path, case_dir: Path, artifacts: Dict[str, Any], key: str, fallback_name: str) -> Path:
    raw = str(artifacts.get(key, "") or "").strip()
    if raw:
        candidate = _resolve(base_dir, raw)
        if candidate.exists() and candidate.is_file():
            return candidate
    return case_dir / fallback_name


def _lightweight_segments(jsonl_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in jsonl_rows:
        seg = {
            "track_id": row.get("track_id"),
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "behavior_code": row.get("behavior_code"),
            "semantic_label_zh": row.get("semantic_label_zh", row.get("action")),
            "semantic_label_en": row.get("semantic_label_en"),
            "confidence": row.get("conf"),
        }
        out.append({k: v for k, v in seg.items() if v is not None})
    return out


def _derive_failure_reason(ve: Dict[str, Any]) -> str:
    """Derive a human-readable failure reason from a verified event."""
    label = str(ve.get("match_label") or ve.get("label") or "")
    evidence = ve.get("evidence", {}) if isinstance(ve.get("evidence"), dict) else {}
    p_match = _safe_float(ve.get("p_match"), 0.0)
    p_mismatch = _safe_float(ve.get("p_mismatch"), 0.0)
    reliability = _safe_float(ve.get("reliability_score") or ve.get("reliability"), 0.0)
    v_score = _safe_float(evidence.get("visual_score") or evidence.get("c_visual"), 0.0)
    t_score = _safe_float(evidence.get("text_score") or evidence.get("c_text"), 0.0)
    uq = _safe_float(evidence.get("uq_score") or evidence.get("uq_track") or ve.get("uncertainty"), 0.0)

    reasons: List[str] = []
    if label == "mismatch":
        if p_mismatch > 0.6:
            reasons.append("strong mismatch signal")
        if v_score < 0.5:
            reasons.append("low visual score")
        if t_score < 0.3:
            reasons.append("low text score")
        if not reasons:
            reasons.append("visual-semantic conflict")
    elif label == "uncertain":
        if reliability < 0.5:
            reasons.append("low reliability")
        if uq > 0.3:
            reasons.append("high track uncertainty")
        if abs(p_match - p_mismatch) < 0.2:
            reasons.append("ambiguous match/mismatch")
        if not reasons:
            reasons.append("borderline decision")
    return "; ".join(reasons) if reasons else label


def _sample_tracks(
    student_tracks_rows: List[Dict[str, Any]],
    *,
    max_per_track: int = 40,
    sample_every_n_frames: int = 12,
    max_frames_per_track: int = 20,
) -> List[Dict[str, Any]]:
    # student_tracks.jsonl has rows: {frame, t, persons: [{track_id, ...}, ...]}
    by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in student_tracks_rows:
        frame = row.get("frame")
        t_val = row.get("t")
        for person in row.get("persons", []) or []:
            tid = person.get("track_id")
            if tid is None:
                continue
            tid_int = _safe_int(tid)
            if tid_int < 0:
                continue
            by_track[tid_int].append({
                "frame": frame,
                "t": t_val,
                **{k: v for k, v in person.items() if k != "behavior_candidates_topk"},
            })

    sampled: List[Dict[str, Any]] = []
    for tid in sorted(by_track):
        frames = by_track[tid]
        if len(frames) <= max_frames_per_track:
            picked = frames
        else:
            picked = frames[::sample_every_n_frames][:max_per_track]
        for row in picked:
            light = {
                "track_id": _safe_int(row.get("track_id", -1)),
                "frame": row.get("frame"),
                "t": row.get("t"),
                "bbox": row.get("bbox"),
                "behavior_code": row.get("behavior_code"),
                "semantic_label_zh": row.get("semantic_label_zh"),
                "behavior_match_status": row.get("behavior_match_status"),
                "linked_pose_track_id": row.get("linked_pose_track_id"),
            }
            sampled.append({k: v for k, v in light.items() if v is not None})
    return sampled


def _segments_by_student(segments: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        tid = seg.get("track_id")
        if tid is None:
            continue
        tid_int = _safe_int(tid)
        if tid_int < 0:
            continue
        by_track[tid_int].append(seg)
    return dict(by_track)


def _build_student_list(timeline_csv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[int, Dict[str, Any]] = {}
    for row in timeline_csv:
        tid = _safe_int(row.get("track_id", -1))
        if tid < 0:
            continue
        sid = row.get("student_id", f"S{tid:02d}")
        if tid not in seen:
            seen[tid] = {"student_id": sid, "track_id": tid}
    return sorted(seen.values(), key=lambda x: x["track_id"])


def _compact_metrics(metrics_path: Optional[Path], contract_path: Optional[Path]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if metrics_path and metrics_path.exists():
        raw = _read_json(metrics_path)
        metrics["gt_status"] = raw.get("gt_status", "missing")
        groups: Dict[str, Any] = {}
        if "person_recall" in raw:
            groups["detection"] = {
                "person_recall": raw.get("person_recall"),
                "person_precision": raw.get("person_precision"),
                "person_f1": raw.get("person_f1"),
                "AP50": raw.get("AP50"),
                "mAP50_95": raw.get("mAP50_95"),
            }
        if "PCK_0_10" in raw:
            groups["pose"] = {"PCK_0_10": raw.get("PCK_0_10"), "OKS_AP": raw.get("OKS_AP")}
        if "IDF1" in raw:
            groups["tracking"] = {
                "IDF1": raw.get("IDF1"),
                "IDSW": raw.get("IDSW"),
                "HOTA": raw.get("HOTA"),
                "MOTA": raw.get("MOTA"),
            }
        if "behavior_macro_f1" in raw:
            groups["behavior"] = {
                "behavior_macro_f1": raw.get("behavior_macro_f1"),
                "temporal_mAP_0_5": raw.get("temporal_mAP_0_5"),
            }
        if "stage_runtime_sec" in raw:
            groups["runtime"] = {
                "stage_runtime_sec": raw.get("stage_runtime_sec"),
                "sr_cache_bytes": raw.get("sr_cache_bytes"),
            }
        metrics["groups"] = {k: {kk: vv for kk, vv in v.items() if vv is not None}
                           for k, v in groups.items()}
    elif contract_path and contract_path.exists():
        contract = _read_json(contract_path)
        counts = contract.get("counts", {})
        metrics["gt_status"] = "missing"
        metrics["contract_status"] = contract.get("status", "unknown")
        metrics["groups"] = {
            "pipeline": {
                "tracked_students": counts.get("tracked_students"),
                "student_track_students": counts.get("student_track_students"),
                "actions_fusion_v2": counts.get("actions_fusion_v2"),
                "unlinked_rows_ge_200000": counts.get("actions_fusion_v2_unlinked_rows_ge_200000", 0),
            }
        }
    return metrics


def _collect_ablation_summary(ablation_json: Optional[Path]) -> List[Dict[str, Any]]:
    if not ablation_json or not ablation_json.exists():
        return []
    data = _read_json(ablation_json)
    rows = data.get("rows") or data.get("variants") or []
    if isinstance(rows, dict):
        rows = list(rows.values())
    return rows if isinstance(rows, list) else []


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Build a lightweight frontend data bundle from a pipeline case directory."
    )
    parser.add_argument("--case_dir", required=True, help="Pipeline output case directory")
    parser.add_argument("--out_dir", required=True, help="Output bundle directory")
    parser.add_argument("--case_id", default="", help="Case identifier (default: derive from dir name)")
    parser.add_argument("--ablation_summary", default="", help="Path to sr_ablation_paper_summary.json")
    parser.add_argument("--metrics_file", default="", help="Path to rear_row_metrics.json (if in different location)")
    parser.add_argument("--max_track_samples", type=int, default=20, help="Max track frames to include")
    args = parser.parse_args()

    case_dir = _resolve(base_dir, args.case_dir)
    out_dir = _resolve(base_dir, args.out_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"case_dir not found: {case_dir}")

    case_id = args.case_id or case_dir.name
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(case_dir / "pipeline_manifest.json")
    contract = _read_json(case_dir / "pipeline_contract_v2_report.json")
    artifacts = manifest.get("artifacts", {})

    # --- Read source files ---
    timeline_csv = _read_csv(case_dir / "timeline_students.csv")
    student_id_map = _read_json(case_dir / "student_id_map.json")
    source_paths = {
        "timeline_students": case_dir / "timeline_students.csv",
        "student_tracks": _artifact_path(base_dir, case_dir, artifacts, "student_tracks", "student_tracks.jsonl"),
        "actions_behavior_semantic": _artifact_path(
            base_dir,
            case_dir,
            artifacts,
            "actions_behavior_semantic",
            "actions.behavior.semantic.jsonl",
        ),
        "actions_fusion_v2": _artifact_path(base_dir, case_dir, artifacts, "actions_fusion_v2", "actions.fusion_v2.jsonl"),
        "verified_events": _artifact_path(base_dir, case_dir, artifacts, "verified_events", "verified_events.jsonl"),
        "event_queries": case_dir / "event_queries.fusion_v2.jsonl",
        "align_multimodal": case_dir / "align_multimodal.json",
        "asr_quality": case_dir / "asr_quality_report.json",
    }
    student_tracks_raw = _read_jsonl(source_paths["student_tracks"])
    behavior_semantic = _read_jsonl(source_paths["actions_behavior_semantic"])
    fusion_actions = _read_jsonl(source_paths["actions_fusion_v2"])
    verified_events = _read_jsonl(source_paths["verified_events"])
    event_queries_raw = _read_jsonl(source_paths["event_queries"])
    align_multimodal_raw = _read_json(source_paths["align_multimodal"])
    asr_quality_raw = _read_json(source_paths["asr_quality"])

    # --- Metrics ---
    metrics_path = _resolve(base_dir, args.metrics_file) if args.metrics_file else (
        case_dir / "rear_row_metrics.json")
    metrics_summary = _compact_metrics(
        metrics_path if metrics_path.exists() else None,
        case_dir / "pipeline_contract_v2_report.json",
    )

    ablation_rows: List[Dict[str, Any]] = []
    if args.ablation_summary:
        ablation_rows = _collect_ablation_summary(_resolve(base_dir, args.ablation_summary))

    # --- Sample tracks early (needed by student list fallback) ---
    tracks_sampled = _sample_tracks(
        student_tracks_raw,
        max_per_track=args.max_track_samples * 2,
        max_frames_per_track=args.max_track_samples,
    )

    # --- Transform ---
    students = _build_student_list(timeline_csv)
    if not students:
        tids = sorted({int(row["track_id"]) for row in tracks_sampled if _safe_int(row.get("track_id", -1)) >= 0})
        students = [{"student_id": f"S{tid:02d}", "track_id": tid} for tid in tids]

    timeline_json = {
        "case_id": case_id,
        "students": students,
        "segments": [
            {
                "student_id": r.get("student_id", ""),
                "track_id": _safe_int(r.get("track_id", -1)),
                "start_time": _safe_float(r.get("start_time", 0)),
                "end_time": _safe_float(r.get("end_time", 0)),
                "behavior_code": r.get("behavior_code", ""),
                "semantic_label_zh": r.get("semantic_label_zh", ""),
                "semantic_label_en": r.get("semantic_label_en", ""),
                "confidence": _safe_float(r.get("confidence", 0)),
                "source": r.get("source", ""),
            }
            for r in timeline_csv
            if _safe_int(r.get("track_id", -1)) >= 0
        ],
    }

    behavior_segments = _lightweight_segments(behavior_semantic)
    fusion_segments = _lightweight_segments(fusion_actions)

    # --- Copy assets ---
    asset_copies: Dict[str, str] = {}
    for key, src_field in [
        ("pose_behavior_video", "pose_behavior_video"),
        ("preview", "pose_behavior_video_preview"),
        ("contact_sheet", "rear_row_compare_contact_sheet"),
    ]:
        src = artifacts.get(src_field, "")
        if src:
            src_path = _resolve(base_dir, str(src))
            if src_path.exists() and src_path.is_file():
                dst = assets_dir / src_path.name
                if not dst.exists():
                    try:
                        shutil.copy2(src_path, dst)
                    except OSError:
                        pass
                asset_copies[key] = str(dst.relative_to(out_dir).as_posix())

    # --- Write bundle ---
    _write_json(out_dir / "timeline_students.json", timeline_json)
    _write_json(out_dir / "behavior_segments.json", {
        "case_id": case_id,
        "segments": behavior_segments,
        "by_track_id": _segments_by_student(behavior_segments),
    })
    _write_json(out_dir / "fusion_segments.json", {
        "case_id": case_id,
        "segments": fusion_segments,
        "by_track_id": _segments_by_student(fusion_segments),
    })
    _write_json(out_dir / "tracks_sampled.json", {
        "case_id": case_id,
        "sample_strategy": {"max_frames_per_track": args.max_track_samples},
        "frames": tracks_sampled,
    })
    _write_json(out_dir / "verified_events.json", {
        "case_id": case_id,
        "events": verified_events,
    })
    _write_json(out_dir / "student_id_map.json", student_id_map)
    _write_json(out_dir / "metrics_summary.json", {"case_id": case_id, **metrics_summary})
    # Populate failure cases from verified_events (mismatch + uncertain)
    failure_items: List[Dict[str, Any]] = []
    student_id_by_track: Dict[int, str] = {s["track_id"]: s["student_id"] for s in students}
    for ve in verified_events:
        label = str(ve.get("match_label") or ve.get("label") or "")
        if label in ("mismatch", "uncertain"):
            tid = _safe_int(ve.get("track_id"), -1)
            evidence = ve.get("evidence", {}) if isinstance(ve.get("evidence"), dict) else {}
            failure_items.append({
                "event_id": str(ve.get("event_id") or ve.get("query_id") or ""),
                "query_text": str(ve.get("query_text", "")),
                "track_id": tid,
                "student_id": student_id_by_track.get(tid, f"S{tid:02d}"),
                "label": label,
                "p_match": _safe_float(ve.get("p_match"), 0.0),
                "p_mismatch": _safe_float(ve.get("p_mismatch"), 0.0),
                "reliability_score": _safe_float(ve.get("reliability_score") or ve.get("reliability"), 0.0),
                "uncertainty": _safe_float(ve.get("uncertainty"), 0.0),
                "visual_score": _safe_float(evidence.get("visual_score") or evidence.get("c_visual"), 0.0),
                "text_score": _safe_float(evidence.get("text_score") or evidence.get("c_text"), 0.0),
                "uq_score": _safe_float(evidence.get("uq_score") or evidence.get("uq_track"), 0.0),
                "failure_reason": _derive_failure_reason(ve),
            })
    _write_json(out_dir / "failure_cases.json", {"case_id": case_id, "items": failure_items})

    # Write new v2 files
    _write_json(out_dir / "event_queries.json", {
        "case_id": case_id,
        "queries": event_queries_raw,
    })
    _write_json(out_dir / "align_multimodal.json", {
        "case_id": case_id,
        "alignments": align_multimodal_raw if isinstance(align_multimodal_raw, list) else [],
    })
    _write_json(out_dir / "asr_quality.json", {
        "case_id": case_id,
        "report": asr_quality_raw,
    })
    _write_json(out_dir / "contract_summary.json", {
        "case_id": case_id,
        "contract_status": contract.get("status", "unknown"),
        "counts": contract.get("counts", {}),
        "checks": contract.get("checks", contract.get("results", [])),
    })

    if ablation_rows:
        _write_json(out_dir / "ablation_summary.json", {
            "case_id": case_id,
            "variants": ablation_rows,
        })

    # --- Manifest ---
    bundle_manifest = {
        "case_id": case_id,
        "schema_version": SCHEMA_VERSION,
        "source_case_dir": str(case_dir),
        "contract_status": contract.get("status", "unknown"),
        "counts": contract.get("counts", {}),
        "files": {
            "timeline_students": "timeline_students.json",
            "behavior_segments": "behavior_segments.json",
            "fusion_segments": "fusion_segments.json",
            "tracks_sampled": "tracks_sampled.json",
            "verified_events": "verified_events.json",
            "student_id_map": "student_id_map.json",
            "metrics_summary": "metrics_summary.json",
            "failure_cases": "failure_cases.json",
            "event_queries": "event_queries.json",
            "align_multimodal": "align_multimodal.json",
            "asr_quality": "asr_quality.json",
            "contract_summary": "contract_summary.json",
        },
        "assets": asset_copies,
        "students": students,
        "tracked_students": len(students),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_commit": _git_head_commit(),
        "compat_with": ["v1"],
        "source_files": {k: str(v) for k, v in source_paths.items()},
    }
    if ablation_rows:
        bundle_manifest["files"]["ablation_summary"] = "ablation_summary.json"
    _write_json(out_dir / "frontend_data_manifest.json", bundle_manifest)

    print(f"[DONE] Bundle written to {out_dir}")
    print(f"  students: {len(students)}")
    print(f"  timeline segments: {len(timeline_json['segments'])}")
    print(f"  behavior segments: {len(behavior_segments)}")
    print(f"  fusion segments: {len(fusion_segments)}")
    print(f"  tracks sampled: {len(tracks_sampled)}")
    print(f"  verified events: {len(verified_events)}")
    print(f"  event queries: {len(event_queries_raw)}")
    print(f"  align multimodal events: {len(align_multimodal_raw) if isinstance(align_multimodal_raw, list) else 0}")
    print(f"  asr quality: {'yes' if asr_quality_raw else 'no'}")
    print(f"  failure cases: {len(failure_items)}")
    print(f"  metrics gt_status: {metrics_summary.get('gt_status', 'unknown')}")
    print(f"  assets: {list(asset_copies.keys())}")


if __name__ == "__main__":
    main()
