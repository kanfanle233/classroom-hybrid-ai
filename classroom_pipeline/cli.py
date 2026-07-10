from __future__ import annotations

import argparse
import sys

from .cleanup import main as cleanup_main
from .orchestrator import run_legacy_pipeline
from .preflight import run_doctor
from .validate import main as validate_main


COMMANDS = {"run", "doctor", "validate", "clean"}


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["--help"]
    if argv[0] in COMMANDS or argv[0] in {"-h", "--help"}:
        return argv
    return ["run", *argv]


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    normalized = _normalize_argv(raw)

    if normalized and normalized[0] == "run":
        pipeline_args = normalized[1:]
        if any(arg in {"-h", "--help"} for arg in pipeline_args):
            raise SystemExit(run_legacy_pipeline(pipeline_args))
        code = run_doctor(video=_option_value(pipeline_args, "--video"), strict_video=True)
        if code != 0:
            raise SystemExit(code)
        raise SystemExit(run_legacy_pipeline(pipeline_args))

    parser = argparse.ArgumentParser(description="YOLOv11 classroom behavior pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the full classroom behavior pipeline.")
    run_parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)

    doctor_parser = sub.add_parser("doctor", help="Check pipeline files and environment.")
    doctor_parser.add_argument("--video", default="")
    doctor_parser.add_argument("--strict-video", action="store_true")

    validate_parser = sub.add_parser("validate", help="Validate a case output directory.")
    validate_parser.add_argument("--case-dir", required=True)

    clean_parser = sub.add_parser("clean", help="Create a safe cleanup manifest.")
    clean_parser.add_argument("--dry-run", action="store_true", default=True)
    clean_parser.add_argument("--apply", action="store_true")
    clean_parser.add_argument("--manifest", default="output/cleanup_manifest.json")

    args, unknown = parser.parse_known_args(normalized)

    if args.command == "doctor":
        raise SystemExit(run_doctor(video=args.video, strict_video=args.strict_video))
    if args.command == "validate":
        raise SystemExit(validate_main(["--case-dir", args.case_dir]))
    if args.command == "clean":
        clean_args = ["--manifest", args.manifest]
        if args.apply:
            clean_args.append("--apply")
        elif args.dry_run:
            clean_args.append("--dry-run")
        clean_args.extend(unknown)
        raise SystemExit(cleanup_main(clean_args))


def _option_value(args: list[str], name: str) -> str:
    prefix = f"{name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    try:
        idx = args.index(name)
    except ValueError:
        return ""
    if idx + 1 >= len(args):
        return ""
    return args[idx + 1]
