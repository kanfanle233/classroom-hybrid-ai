"""Command-line entry points for planning, checking and running the pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

from .config import load_config
from .evaluation import compare_candidate, evaluate_rows
from .research_backend import ResearchBackend, status_payload
from .preflight import run_preflight
from .registry import audit_registry
from .storage import read_jsonl
from .storage import atomic_write_json
from .benchmark import freeze_manifest, score_predictions


_SOURCE_DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
_WORKSPACE_DEFAULTS = (
    Path.cwd() / "classroom_hybrid_ai" / "configs" / "default.yaml",
    Path.cwd() / "configs" / "default.yaml",
)
DEFAULT_CONFIG = next(
    (candidate.resolve() for candidate in (_SOURCE_DEFAULT, *_WORKSPACE_DEFAULTS) if candidate.is_file()),
    Path(resources.files("classroom_hybrid_ai").joinpath("configs/default.yaml")),
)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classroom-hybrid",
        description="Local LocateAnything-first + stateless Qwen3.5 video analysis",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="check interpreter, CUDA and local models")
    preflight.add_argument("--verify-hashes", action="store_true")

    registry_check = commands.add_parser(
        "registry-check", help="verify the append-only shared action candidate registry"
    )
    registry_check.add_argument(
        "--root",
        type=Path,
        help="registry directory; defaults to paths.action_registry_root from the config",
    )

    plan = commands.add_parser("plan", help="estimate Qwen work before running models")
    plan.add_argument("video", type=Path)
    plan.add_argument("--average-tracks", type=int, default=30)
    plan.add_argument("--profile", default=None)
    plan.add_argument("--mode", choices=("production_adaptive", "exhaustive_validation"), default=None)

    run = commands.add_parser("run", help="run the sequential offline pipeline")
    run.add_argument("video", type=Path)
    run.add_argument("--max-frames", type=int)
    run.add_argument("--skip-qwen", action="store_true", help="development-degraded only; never research-valid")
    run.add_argument("--profile", default=None, choices=("core8_v1", "scb_student_11_candidate_v1"))
    run.add_argument("--mode", choices=("production_adaptive", "exhaustive_validation"), default=None)
    run.add_argument("--watch", action="store_true", help="write v3 stage state for a read-only watcher")
    run.add_argument(
        "--development-degraded",
        "--allow-incomplete-preflight",
        dest="development_degraded",
        action="store_true",
        help="explicit development mode; allows incomplete preflight/skip-qwen and marks research validity pending or invalid",
    )

    status = commands.add_parser("status", help="read a v3 run state without starting models")
    status.add_argument("run_dir", type=Path)

    resume = commands.add_parser("resume", help="resume a v3 run from its last valid stage")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--allow-incomplete-preflight", action="store_true")

    watch = commands.add_parser("watch", help="serve a read-only v3 state/SSE page")
    watch.add_argument("run_dir", type=Path)
    watch.add_argument("--host", default="127.0.0.1")
    watch.add_argument("--port", type=int, default=8765)

    report = commands.add_parser("report", help="print the v3 report and output index")
    report.add_argument("run_dir", type=Path)

    smoke = commands.add_parser("smoke-qwen", help="load Qwen3.5-9B 4-bit and review image/video evidence")
    smoke.add_argument(
        "image",
        type=Path,
        nargs="+",
        help="one or more images/videos; video inputs contribute their first frame and inputs rotate by request",
    )
    smoke.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="stateless requests; defaults to 1/20/100 for none/development/delivery gate",
    )
    smoke.add_argument("--gate", choices=("none", "development", "delivery"), default="none")
    smoke.add_argument("--output", type=Path, help="optional JSON receipt path")
    smoke.add_argument(
        "--production-prompt",
        action="store_true",
        help="use the exact S5 classification prompt; required for a realistic 8 GiB memory gate",
    )

    evaluate = commands.add_parser("evaluate", help="compare two Qwen candidates on shared rows")
    evaluate.add_argument("baseline", type=Path)
    evaluate.add_argument("candidate", type=Path)

    benchmark_freeze = commands.add_parser("benchmark-freeze", help="freeze the local 8,884-image blind manifest")
    benchmark_freeze.add_argument("source", type=Path)
    benchmark_freeze.add_argument("output", type=Path)
    benchmark_score = commands.add_parser("benchmark-score", help="score post-hoc eight-label blind predictions")
    benchmark_score.add_argument("predictions", type=Path)
    benchmark_score.add_argument("--output", type=Path)

    performance = commands.add_parser("benchmark-performance", help="measure production work in isolated baseline/candidate processes")
    performance.add_argument("video", type=Path)
    performance.add_argument("--component", choices=("pipeline", "locate", "qwen", "io"), default="pipeline")
    performance.add_argument("--repeats", type=int, default=3)
    performance.add_argument("--output", type=Path, required=True)
    performance.add_argument("--baseline-source", type=Path, help="snapshot src directory containing classroom_hybrid_ai")
    performance.add_argument("--baseline-config", type=Path)
    performance.add_argument("--run-dir", type=Path, help="matching production S1/S4 evidence; copied before measurement")
    performance.add_argument("--max-frames", type=int, help="explicitly mark a bounded workload")
    performance.add_argument("--max-requests", type=int, help="bound the same production Qwen candidates in both variants")
    performance.add_argument("--profile-model-timings", action="store_true", help="enable synchronised CUDA/first-token probes; affects throughput")

    stability = commands.add_parser("benchmark-qwen-stability", help="replay production S4 evidence in one isolated worker")
    stability.add_argument("run_dir", type=Path)
    stability.add_argument("--requests", type=int, default=100)
    stability.add_argument("--output", type=Path, required=True)

    study = commands.add_parser("benchmark-study", help="run locked whole-pipeline experimental variants")
    study.add_argument("manifest", type=Path)
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--dry-run", action="store_true")
    diagnose = commands.add_parser("diagnose-unknown", help="audit unique candidate abstention and missing evidence")
    diagnose.add_argument("run_dir", type=Path)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--reviews", type=Path)
    annotate = commands.add_parser("prepare-annotation", help="prepare a manual video annotation package, never automatic gold")
    annotate.add_argument("video", type=Path)
    annotate.add_argument("--output", type=Path, required=True)
    annotate.add_argument("--run-dir", type=Path)
    annotate.add_argument("--classroom-id")
    annotate.add_argument("--lesson-id")
    annotate.add_argument("--session-id")

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.command == "preflight":
        result = run_preflight(config, verify_hashes=args.verify_hashes)
        _print_json(result)
        return 0 if result["status"] == "ok" else 2
    if args.command == "registry-check":
        root = args.root.resolve() if args.root is not None else config.action_registry_root
        result = audit_registry(root)
        _print_json(result)
        return 0 if result["status"] in {"ok", "empty"} and not result.get("errors") else 2
    if args.command == "plan":
        _print_json(ResearchBackend(config).plan(args.video, profile_id=args.profile, mode=args.mode, average_tracks=args.average_tracks))
        return 0
    if args.command == "run":
        if args.max_frames is not None and args.max_frames < 1:
            raise ValueError("max_frames_must_be_positive")
        if args.skip_qwen and not args.development_degraded:
            raise ValueError("skip_qwen_requires_development_degraded")
        result = ResearchBackend(config).run(
            args.video, profile_id=args.profile, mode=args.mode,
            max_frames=args.max_frames, skip_qwen=args.skip_qwen,
            watch=args.watch,
            allow_incomplete_preflight=args.development_degraded,
        )
        _print_json(result.to_dict())
        # Optional-stage degradation (for example an absent WSL
        # LocateAnything worker) is reported in the JSON as ``warning`` but
        # does not make the main run a process failure.  A Qwen-stage warning
        # still exits non-zero so a model/schema failure is not hidden.
        optional_warnings_only = bool(result.warnings) and all(
            str(item).startswith("locate_anything_") for item in result.warnings
        )
        return 0 if result.status == "ok" or optional_warnings_only else 1
    if args.command == "status":
        _print_json(status_payload(args.run_dir))
        return 0
    if args.command == "resume":
        result = ResearchBackend(config).resume(args.run_dir, allow_incomplete_preflight=args.allow_incomplete_preflight)
        _print_json(result.to_dict())
        return 0 if result.status in {"ok", "warning"} else 1
    if args.command == "watch":
        from .watch_server import serve_watch
        server = serve_watch(args.run_dir, host=args.host, port=args.port)
        print(json.dumps({"status": "listening", "host": args.host, "port": args.port, "run_dir": str(args.run_dir.resolve())}, ensure_ascii=False))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
        finally:
            server.server_close()
        return 0
    if args.command == "report":
        root = args.run_dir.resolve()
        report_path = root / "report.html"
        index_path = root / "frontend" / "index.v1.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"report_not_found:{report_path}")
        _print_json({"run_dir": str(root), "report": "report.html", "frontend_index": "frontend/index.v1.json" if index_path.is_file() else None, "status": status_payload(root) if (root / "run_state.v1.json").is_file() else None})
        return 0
    if args.command == "smoke-qwen":
        from .smoke import run_qwen_smoke

        result = run_qwen_smoke(
            config,
            args.image,
            repeats=args.repeats,
            gate=args.gate,
            command=sys.argv,
            checkpoint_path=args.output,
            production_prompt=bool(getattr(args, "production_prompt", False)),
        )
        if args.output is not None:
            atomic_write_json(args.output.resolve(), result)
        _print_json(result)
        return 0 if result["status"] in {"ok", "incomplete"} else 1
    if args.command == "evaluate":
        baseline = evaluate_rows(read_jsonl(args.baseline))
        candidate = evaluate_rows(read_jsonl(args.candidate))
        gate = config.get("qwen_candidates", {})
        _print_json(compare_candidate(
            baseline,
            candidate,
            minimum_macro_f1_gain_pp=float(gate.get("minimum_event_macro_f1_gain_pp", 3.0)),
            maximum_any_class_drop_pp=float(gate.get("maximum_any_class_f1_drop_pp", 2.0)),
        ))
        return 0
    if args.command == "benchmark-study":
        from .performance_study import run_study
        result = run_study(args.manifest, output=args.output, dry_run=args.dry_run, python=config.python_executable)
        _print_json(result)
        return 0 if result["status"] in {"ok", "planned"} else 130 if result["status"] in {"paused", "paused_by_user"} else 1
    if args.command == "diagnose-unknown":
        from .quality_diagnostics import diagnose_unknown
        _print_json(diagnose_unknown(args.run_dir, output_path=args.output, review_path=args.reviews))
        return 0
    if args.command == "prepare-annotation":
        from .quality_diagnostics import prepare_annotation_package
        _print_json(prepare_annotation_package(args.video, output_dir=args.output, run_dir=args.run_dir,
            classroom_id=args.classroom_id, lesson_id=args.lesson_id, session_id=args.session_id))
        return 0
    if args.command == "benchmark-performance":
        from .performance_benchmark import run_benchmark
        result = run_benchmark(
            config, args.video, component=args.component, repeats=args.repeats,
            output=args.output, baseline_source=args.baseline_source,
            baseline_config=args.baseline_config, run_dir=args.run_dir,
            max_frames=args.max_frames, max_requests=args.max_requests,
            profile_model_timings=args.profile_model_timings,
        )
        _print_json({
            "schema_version": result["schema_version"], "status": result["status"],
            "output": str(args.output.resolve()), "variants": result.get("variants", {}),
            "comparison": result.get("comparison", {}),
        })
        return 0 if result["status"] == "ok" else 1
    if args.command == "benchmark-qwen-stability":
        from .performance_stability import run_production_stability
        result = run_production_stability(config, args.run_dir, output=args.output, requests=args.requests)
        _print_json({key: result.get(key) for key in ("schema_version", "status", "completed_calls", "wall_seconds", "failure")})
        return {"passed": 0, "incomplete": 2, "paused_by_user": 130}.get(result["status"], 1)
    if args.command == "benchmark-freeze":
        result = freeze_manifest(args.source.resolve(), args.output.resolve())
        _print_json({
            "schema_version": result["schema_version"],
            "sample_count": result["sample_count"],
            "labels": result["labels"],
            "sha256": result["sha256"],
            "output": str(args.output.resolve()),
        })
        return 0
    if args.command == "benchmark-score":
        result = score_predictions(read_jsonl(args.predictions))
        if args.output is not None:
            atomic_write_json(args.output.resolve(), result)
        _print_json(result)
        return 0
    raise AssertionError(f"unhandled_command:{args.command}")


def _error_payload(exc: Exception, *, command: str | None) -> dict[str, Any]:
    """Return the stable machine-readable shape for runtime command errors."""

    message = str(exc).strip() or type(exc).__name__
    return {
        "schema_version": "classroom_hybrid_ai.cli_error.v1",
        "status": "error",
        "command": command,
        "error": {
            "type": type(exc).__name__,
            "code": message.split(":", 1)[0],
            "message": message,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one public command and keep runtime failures JSON-only.

    ``argparse`` is intentionally outside the ``try`` block: malformed command
    syntax should retain argparse's conventional usage/error behaviour.  Once
    syntax is valid, configuration, filesystem, model and watcher failures are
    represented by one JSON object and a non-zero return code.
    """

    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except Exception as exc:
        _print_json(_error_payload(exc, command=getattr(args, "command", None)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
