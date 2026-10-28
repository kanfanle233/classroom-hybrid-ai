"""Reproduce frozen HOTA equivalence and an alternating CPU microbenchmark."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classroom_hybrid_ai.paper_evaluation as score
from classroom_hybrid_ai.storage import atomic_write_json


def run(baseline_source: Path, *, repeats: int = 3) -> dict:
    source = baseline_source.read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "_hota_metrics")
    namespace = dict(score.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(baseline_source), "exec"), namespace)
    baseline = namespace["_hota_metrics"]
    # The baseline function runs with the unchanged matching/IoU helpers.
    # Verify their source ASTs too so later helper edits cannot silently turn
    # this experiment into a comparison against a modified reference.
    baseline_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    current_source = Path(score.__file__).read_text(encoding="utf-8")
    current_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(current_source).body if isinstance(node, ast.FunctionDef)}
    for helper in ("_hungarian_max", "_iou", "_frame_groups", "_safe_ratio", "_as_nonempty_text"):
        if baseline_functions[helper] != current_functions[helper]:
            raise ValueError(f"baseline_helper_changed:{helper}")
    for seed in range(100):
        rng = random.Random(seed)
        gold, pred = [], []
        for frame in range(rng.randrange(1, 12)):
            for person in range(rng.randrange(0, 7)):
                x, y = rng.randrange(0, 60), rng.randrange(0, 60)
                gold.append({"frame_index": frame, "pts_seconds": frame / 10, "gold_person_id": f"g{person}", "bbox_px": [x, y, x + 20, y + 20]})
                if rng.random() < 0.8:
                    dx, dy = rng.randrange(-15, 16), rng.randrange(-15, 16)
                    pred.append({"frame_index": frame, "pts_seconds": frame / 10, "pred_person_id": f"p{person if rng.random() < 0.8 else rng.randrange(7)}", "bbox_px": [x + dx, y + dy, x + 20 + dx, y + 20 + dy]})
        g, p = SimpleNamespace(boxes=gold), SimpleNamespace(detections=pred)
        if baseline(g, p) != score._hota_metrics(g, p):
            raise AssertionError(f"random_equivalence_failed:seed={seed}")
    rows = [{"frame_index": frame, "pts_seconds": frame / 10, "gold_person_id": f"g{person}", "bbox_px": [person * 12, 0, person * 12 + 20, 20]} for frame in range(60) for person in range(20)]
    predictions = [{**row, "pred_person_id": row["gold_person_id"].replace("g", "p")} for row in rows]
    g, p = SimpleNamespace(boxes=rows), SimpleNamespace(detections=predictions)
    durations: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for _ in range(repeats):
        for name, function in (("baseline", baseline), ("candidate", score._hota_metrics)):
            start = time.perf_counter()
            result = function(g, p)
            durations[name].append(time.perf_counter() - start)
            if result["HOTA"] != 1.0:
                raise AssertionError("microbenchmark_perfect_tracking_expected")
    medians = {name: statistics.median(values) for name, values in durations.items()}
    return {
        "schema_version": "classroom_hybrid_ai.hota_microbenchmark.v1",
        "status": "passed", "measurement_scope": "CPU HOTA computation only",
        "limitations": ["Measured while other implementation/test tasks may be running on this CPU.", "This is a local microbenchmark, not an end-to-end pipeline speedup."],
        "baseline_source": str(baseline_source.resolve()),
        "baseline_source_sha256": hashlib.sha256(baseline_source.read_bytes()).hexdigest(),
        "candidate_source": str(Path(score.__file__).resolve()),
        "candidate_source_sha256": hashlib.sha256(Path(score.__file__).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "interpreter": sys.executable, "platform": platform.platform(),
        "equivalence": {"seed_range_inclusive": [0, 99], "cases": 100, "comparison": "exact equality of complete HOTA result dictionaries", "unchanged_reference_helpers_verified": True},
        "benchmark": {"frames": 60, "people_per_frame": 20, "repeats": repeats, "order": ["baseline", "candidate"], "durations_seconds": durations, "median_seconds": medians, "elapsed_reduction_fraction": 1.0 - medians["candidate"] / medians["baseline"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = run(args.baseline_source, repeats=args.repeats)
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
