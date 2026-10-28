"""Reproduce CPU evidence worker comparisons on a fixed, local input video.

Uses synthetic boxes solely to exercise production crop/context/composite I/O.
No detector, behavior model or behavior-accuracy metric is run. Each trial uses
a fresh Python process; worker order rotates across rounds to reduce bias.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import psutil
import yaml

from classroom_hybrid_ai.evidence import build_event_evidence, build_temporal_composite
from classroom_hybrid_ai.hashing import canonical_sha256, python_source_digest, sha256_file
from classroom_hybrid_ai.performance import PerformanceRecorder, recording
from classroom_hybrid_ai.storage import atomic_write_json
from classroom_hybrid_ai.video import probe_video


def _fixed_inputs(info):
    """Six synthetic seats and eight uniformly spaced events per seat."""
    students = [f"synthetic-seat-{index:02d}" for index in range(6)]
    boxes = {}
    for index, student in enumerate(students):
        row, column = divmod(index, 3)
        boxes[student] = [round(info.width * (column / 3 + 0.025)), round(info.height * (row / 2 + 0.025)), round(info.width * ((column + 1) / 3 - 0.025)), round(info.height * ((row + 1) / 2 - 0.025))]
    start = min(0.75, info.duration_seconds / 4)
    end = max(start, (info.frame_count - 1) / info.fps - start)
    triggers = [start + index * (end - start) / 7 for index in range(8)]
    candidates = [{"event_id": f"cpu-io-t{time_index:02d}-s{student_index:02d}", "track_id": student, "trigger_pts_seconds": trigger} for time_index, trigger in enumerate(triggers) for student_index, student in enumerate(students)]
    observations = [{"track_id": student, "student_id": student, "frame_index": index, "pts_seconds": index / info.fps, "bbox_xyxy": boxes[student]} for index in range(info.frame_count) for student in students]
    return candidates, observations


def _trial(video: Path, output: Path, workers: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    info = probe_video(video)
    candidates, observations = _fixed_inputs(info)
    config = yaml.safe_load((PROJECT_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))["evidence"]
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    samples = [baseline_rss]
    stop = threading.Event()

    def observe():
        while not stop.wait(0.005):
            samples.append(process.memory_info().rss)

    sampler = threading.Thread(target=observe, daemon=True)
    sampler.start()
    recorder = PerformanceRecorder(f"cpu-preprocess-workers-{workers}")
    started = time.perf_counter()
    try:
        with recording(recorder):
            receipts = build_event_evidence(
                video_path=video, candidates=candidates, observations=observations,
                output_dir=output / "evidence",
                before_seconds=float(config["before_seconds"]), after_seconds=float(config["after_seconds"]),
                frames_per_event=int(config["frames_per_event"]), context_ratio=float(config["crop_context_ratio"]),
                maximum_crop_side=int(config["maximum_crop_side"]), maximum_context_side=int(config.get("maximum_context_side", 768)),
                jpeg_quality=int(config["jpeg_quality"]), decode_strategy="sequential",
                preprocess_workers=workers, preprocess_queue_size=2,
            )
            for receipt in receipts:
                composite = build_temporal_composite(receipt, output_path=output / "evidence" / receipt["event_id"] / "temporal_composite.jpg", jpeg_quality=int(config["jpeg_quality"]))
                if composite is None:
                    raise RuntimeError("benchmark_composite_missing")
                receipt["qwen_composite"] = composite
    finally:
        wall = time.perf_counter() - started
        samples.append(process.memory_info().rss)
        stop.set()
        sampler.join(timeout=1)
    image_hashes = {path.relative_to(output).as_posix(): sha256_file(path) for path in sorted((output / "evidence").rglob("*.jpg"))}
    # Normalize only output paths, preserving every evidence frame/PTS/box/hash.
    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items() if key != "path"}
        if isinstance(value, list):
            return [portable(item) for item in value]
        return value
    result = {
        "workers": workers, "queue_size": 2, "wall_seconds": wall,
        "rss_baseline_bytes": baseline_rss, "rss_peak_bytes": max(samples), "rss_growth_peak_bytes": max(samples) - baseline_rss,
        "rss_sample_interval_seconds": 0.005, "rss_samples": len(samples),
        "candidate_count": len(candidates), "jpeg_count": len(image_hashes),
        "image_hashes": image_hashes, "image_manifest_sha256": canonical_sha256(image_hashes),
        "normalized_receipts_sha256": canonical_sha256(portable(receipts)),
        "workload_sha256": canonical_sha256({"candidates": candidates, "observations": observations, "evidence_config": config}),
        "evidence_config": config, "performance": recorder.payload(),
    }
    atomic_write_json(output / "trial.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trial-workers", type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    video = args.video.resolve()
    output = args.output.resolve()
    if args.trial_workers is not None:
        trial = _trial(video, output, args.trial_workers)
        print(json.dumps({key: trial[key] for key in ("workers", "wall_seconds", "rss_peak_bytes", "jpeg_count")}, sort_keys=True))
        return 0
    if (output / "report.json").exists():
        parser.error("output already contains a report; use a new directory to preserve the prior experiment")
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    worker_options = [1, 2, 4]
    reference = None
    for repetition in range(args.repeats):
        order = worker_options[repetition % 3:] + worker_options[:repetition % 3]
        for workers in order:
            directory = output / f"round-{repetition + 1:02d}-workers-{workers}"
            started = time.perf_counter()
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(video), str(directory), "--trial-workers", str(workers)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if completed.returncode != 0:
                raise RuntimeError(f"cpu_trial_failed:{completed.stderr[-4000:]}")
            trial = json.loads((directory / "trial.json").read_text(encoding="utf-8"))
            trial["subprocess_wall_seconds"] = time.perf_counter() - started
            trial["round"] = repetition + 1
            trial["trial_uri"] = directory.relative_to(output).as_posix() + "/trial.json"
            hashes = (trial["image_manifest_sha256"], trial["normalized_receipts_sha256"], trial["workload_sha256"])
            if reference is None:
                reference = hashes
            trial["output_equal_to_reference"] = hashes == reference
            trials.append(trial)
            print(completed.stdout.strip(), flush=True)
    aggregates = []
    for workers in worker_options:
        values = [trial for trial in trials if trial["workers"] == workers]
        aggregates.append({"workers": workers, "runs": len(values), "median_wall_seconds": statistics.median(trial["wall_seconds"] for trial in values), "median_rss_peak_bytes": statistics.median(trial["rss_peak_bytes"] for trial in values), "maximum_rss_peak_bytes": max(trial["rss_peak_bytes"] for trial in values), "median_rss_growth_peak_bytes": statistics.median(trial["rss_growth_peak_bytes"] for trial in values)})
    baseline = aggregates[0]
    rss_limit = max(baseline["median_rss_peak_bytes"] * 1.10, baseline["median_rss_peak_bytes"] + 32 * 1024 ** 2)
    for row in aggregates:
        row["memory_eligible"] = row["median_rss_peak_bytes"] <= rss_limit and row["median_rss_growth_peak_bytes"] <= baseline["median_rss_growth_peak_bytes"] + 32 * 1024 ** 2
        row["wall_reduction_vs_1_worker_percent"] = 100 * (1 - row["median_wall_seconds"] / baseline["median_wall_seconds"])
    outputs_equal = all(trial["output_equal_to_reference"] for trial in trials)
    winner = min((row for row in aggregates if row["memory_eligible"]), key=lambda row: row["median_wall_seconds"]) if outputs_equal else None
    report = {
        "schema_version": "classroom_hybrid_ai.evidence_cpu_benchmark.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "not_evaluated", "reason": "synthetic_boxes_and_candidates_measure_io_only_no_behavior_model_or_gold",
        "source": {**probe_video(video).to_dict(), "sha256": sha256_file(video)},
        "script_sha256": sha256_file(Path(__file__)), "code_sha256": python_source_digest(),
        "environment": {"python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(), "logical_cpus": os.cpu_count(), "psutil": psutil.__version__},
        "design": {"workers": worker_options, "queue_size": 2, "repeats": args.repeats, "order": "rotating_1_2_4_then_2_4_1_then_4_1_2", "fresh_process_per_trial": True, "decode_strategy": "sequential", "wall_scope": "production_evidence_and_composites_excludes_imports_and_independent_hash_audit", "rss_scope": "sampled_process_working_set_5ms", "synthetic_students": 6, "synthetic_events_per_student": 8},
        "acceptance": {"all_jpeg_bytes_and_normalized_receipts_equal": outputs_equal, "memory_limit_bytes": rss_limit, "memory_rule": "median_RSS <= max(baseline*1.10,baseline+32MiB) and median_growth <= baseline_growth+32MiB", "selected_workers": None if winner is None else winner["workers"], "selection_rule": "lowest_median_wall_among_memory_eligible_identical_outputs", "default_configuration_changed": False},
        "aggregates": aggregates, "trials": [{key: value for key, value in trial.items() if key not in {"image_hashes", "evidence_config"}} for trial in trials],
        "reproduction_command": [sys.executable, str(Path(__file__).resolve()), str(video), "<new-output-directory>", "--repeats", str(args.repeats)],
    }
    atomic_write_json(output / "report.json", report)
    print(json.dumps({"report": str(output / "report.json"), "selected_workers": report["acceptance"]["selected_workers"], "aggregates": aggregates}, ensure_ascii=False), flush=True)
    return 0 if outputs_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
