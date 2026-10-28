"""Read-only command line wrapper for :mod:`classroom_hybrid_ai.paper_evaluation`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:  # direct ``python scripts/evaluate_research_run.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classroom_hybrid_ai.paper_evaluation import evaluate_research_run


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline paper evaluation; never starts a model or edits S7 artifacts.")
    parser.add_argument("run_dir", type=Path, help="completed ResearchBackend run directory")
    parser.add_argument("--gold-manifest", type=Path, help="gold_manifest.v1.json; omit for not_evaluated output")
    parser.add_argument("--profile", default=None, help="profile id or action_profiles.yaml path")
    parser.add_argument("--output", type=Path, help="write a new metrics JSON file instead of stdout")
    parser.add_argument("--split-manifest", action="append", type=Path, default=[], help="manifest for split audit; repeat for all manifests")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = evaluate_research_run(
            args.run_dir, args.gold_manifest, args.profile,
            split_manifests=args.split_manifest,
            require_split_audit=args.gold_manifest is not None,
        )
        encoded = _json_dump(payload)
        if args.output:
            target = args.output.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            # This is a user-selected evaluation output, never a mutation of
            # metrics.v1.json or any completed S7 artifact.
            target.write_text(encoded + "\n", encoding="utf-8")
        else:
            sys.stdout.write(encoded + "\n")
        return 2 if payload.get("evaluation_status") in {"invalid_gold", "invalid_split"} else 0
    except Exception as exc:
        error = {"schema_version": "classroom_hybrid_ai.metrics.v1", "evaluation_status": "error", "reason": f"{type(exc).__name__}:{exc}"}
        encoded = _json_dump(error)
        if args.output:
            target = args.output.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(encoded + "\n", encoding="utf-8")
        else:
            sys.stdout.write(encoded + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
