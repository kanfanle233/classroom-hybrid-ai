"""Compare batch 1/2/4 on the exact S1 screenshots; never claim action quality."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classroom_hybrid_ai.config import load_config
from classroom_hybrid_ai.hashing import sha256_file
from classroom_hybrid_ai.localization import run_yolo_screenshots, build_difficulty_queue
from classroom_hybrid_ai.performance_benchmark import _code_state, _canonical
from classroom_hybrid_ai.performance_study import prepare_seed_control
from classroom_hybrid_ai.runtime import GpuLease
from classroom_hybrid_ai.storage import atomic_write_json, iter_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import yaml
    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_directory = output / "seed_control"
    os.environ.update(prepare_seed_control(seed_directory, 0))
    os.environ["PYTHONPATH"] = os.pathsep.join((os.environ["PYTHONPATH"], str(config.project_root / "src")))
    scenes = list(iter_jsonl(args.run_dir / "artifacts/screenshot_schedule.v1.jsonl"))
    if not scenes:
        raise ValueError("S1 screenshots required")
    model = config.model_path("yolo.tracker_model")
    manifest = yaml.safe_load(config.model_manifest_path.read_text(encoding="utf-8"))
    key = config.get("localization.model_manifest_key", "yolo26_pose")
    expected_sha = manifest["models"][key]["sha256"]
    source_sha = _code_state(config.project_root / "src", config.project_root)
    config_sha = sha256_file(config.source_path)
    inputs = [{"frame_index":row["frame_index"], "pts_seconds":row["pts_seconds"],
               "image_sha256":sha256_file(Path(row["path"]))} for row in scenes]
    report = {"schema_version":"classroom_hybrid_ai.localization_batches.v1", "status":"running",
        "scope":"S2 YOLO only; full fixed S1 screenshot set; cold worker per trial",
        "quality_status":"not_evaluated", "seed":0, "source_sha256":source_sha,
        "config_sha256":config_sha, "model_sha256":expected_sha, "inputs":inputs, "trials":[]}
    reference = None
    for repeat in range(3):
        batches = (1,2,4)
        for batch in batches[repeat:] + batches[:repeat]:
            name = f"r{repeat+1}-batch{batch}"
            with GpuLease(config.repository_root / ".runtime/classroom_hybrid_ai/gpu.lock"):
                started = time.perf_counter()
                rows, stage = run_yolo_screenshots(scenes, model_path=model, model_sha256=expected_sha,
                    cache_dir=output / name, code_fingerprint=source_sha, imgsz=int(config.get("yolo.image_size",960)),
                    conf=float(config.get("yolo.confidence",.2)), batch_size=batch,
                    device=config.get("runtime.device",0), config_dir=output / "ultralytics", reuse_responses=False)
                elapsed = time.perf_counter()-started
            semantics = [{"frame_index":r["frame_index"], "detections":r["detections"]} for r in rows]
            reference = semantics if reference is None else reference
            queue, difficulty = build_difficulty_queue(rows,
                audit_interval_seconds=float(config.get("localization.audit_interval_seconds",30)),
                low_confidence=float(config.get("localization.low_confidence",.28)))
            pid = stage["lifecycle"].get("pid")
            seed_audit = json.loads((seed_directory / "audit" / f"{pid}.json").read_text(encoding="utf-8"))
            if seed_audit.get("status") != "ok" or seed_audit.get("seed") != 0:
                raise RuntimeError("worker_seed_not_verified")
            trial = {"repeat":repeat+1, "batch_size":batch, "wall_seconds":elapsed,
                "stage":stage, "detections_sha256":_canonical(semantics),
                "exact_reference_detections":semantics == reference, "difficult_screenshots":len(queue),
                "difficulty":difficulty, "seed_audit":seed_audit}
            atomic_write_json(output / name / "detections.json", rows)
            report["trials"].append(trial)
            atomic_write_json(output / "report.json", report)
            print(json.dumps({k:trial[k] for k in ("repeat","batch_size","wall_seconds","exact_reference_detections","difficult_screenshots")}),flush=True)
    report["summary"] = {str(batch):{"median_wall_seconds":statistics.median(
        t["wall_seconds"] for t in report["trials"] if t["batch_size"]==batch),
        "all_exact_reference_detections":all(t["exact_reference_detections"] for t in report["trials"] if t["batch_size"]==batch)}
        for batch in (1,2,4)}
    if source_sha != _code_state(config.project_root / "src", config.project_root) or config_sha != sha256_file(config.source_path):
        raise RuntimeError("source_or_config_changed_during_measurement")
    report["status"] = "ok"
    report["default_promoted"] = False
    atomic_write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
