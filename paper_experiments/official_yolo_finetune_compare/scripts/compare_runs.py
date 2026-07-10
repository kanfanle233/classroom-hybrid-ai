from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
}


def _read_rows(results_csv: Path) -> list[dict[str, str]]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _summarize_run(label: str, results_csv: Path) -> dict[str, object]:
    if not results_csv.exists():
        return {"label": label, "results_csv": str(results_csv), "exists": False}

    rows = _read_rows(results_csv)
    if not rows:
        return {"label": label, "results_csv": str(results_csv), "exists": True, "rows": 0}

    final_row = rows[-1]
    best_row = max(rows, key=lambda row: _float(row, KEYS["mAP50_95"]))

    return {
        "label": label,
        "results_csv": str(results_csv),
        "exists": True,
        "rows": len(rows),
        "final_epoch": int(float(final_row["epoch"])),
        "final": {
            "precision": round(_float(final_row, KEYS["precision"]), 5),
            "recall": round(_float(final_row, KEYS["recall"]), 5),
            "mAP50": round(_float(final_row, KEYS["mAP50"]), 5),
            "mAP50_95": round(_float(final_row, KEYS["mAP50_95"]), 5),
        },
        "best_mAP50_95": {
            "epoch": int(float(best_row["epoch"])),
            "precision": round(_float(best_row, KEYS["precision"]), 5),
            "recall": round(_float(best_row, KEYS["recall"]), 5),
            "mAP50": round(_float(best_row, KEYS["mAP50"]), 5),
            "mAP50_95": round(_float(best_row, KEYS["mAP50_95"]), 5),
        },
    }


def _build_markdown(runs: list[dict[str, object]], reference_label: str) -> str:
    reference = next((run for run in runs if run.get("label") == reference_label and run.get("exists")), None)
    lines = [
        "# Run Comparison",
        "",
        f"- reference_run: `{reference_label}`",
        "",
        "| label | exists | final_epoch | final_mAP50 | final_mAP50_95 | best_mAP50_95_epoch | delta_vs_ref_mAP50_95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    ref_value = None
    if reference:
        ref_value = float(reference["final"]["mAP50_95"])

    for run in runs:
        if not run.get("exists"):
            lines.append(f"| {run['label']} | False | - | - | - | - | - |")
            continue
        final = run["final"]
        best = run["best_mAP50_95"]
        delta = "-"
        if ref_value is not None:
            delta_val = float(final["mAP50_95"]) - ref_value
            delta = f"{delta_val:+.5f}"
        lines.append(
            f"| {run['label']} | True | {run['final_epoch']} | {final['mAP50']:.5f} | {final['mAP50_95']:.5f} | {best['epoch']} | {delta} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `case_yolo_train` and `wisdom8_current` should not be treated as a fully fair apples-to-apples result without confirming the exact dataset yaml and class-order mapping.",
            "- `wisdom8_current` is the locked official baseline for `data/processed/classroom_yolo/dataset.yaml`.",
            "- `wisdom8_ft70` should only be kept if it improves the target metric; on the current numbers it does not beat `wisdom8_current`.",
            "- A future custom-YOLO run should be added here for a fair architecture comparison against the official baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detection training runs from results.csv files.")
    parser.add_argument("--profile", required=True, help="Path to YAML profile")
    parser.add_argument("--reference", default="wisdom8_current", help="Reference run label")
    parser.add_argument("--out_json", default="", help="Optional JSON output path")
    parser.add_argument("--out_md", default="", help="Optional markdown output path")
    args = parser.parse_args()

    profile_path = Path(args.profile).resolve()
    if not profile_path.exists():
        raise FileNotFoundError(f"profile not found: {profile_path}")

    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    repo_root = Path(profile["repo_root"]).resolve()

    runs = []
    for label, rel_path in profile.get("comparison_runs", {}).items():
        results_csv = (repo_root / str(rel_path)).resolve()
        runs.append(_summarize_run(str(label), results_csv))

    payload = {
        "reference": args.reference,
        "runs": runs,
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(json_text)

    if args.out_json:
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json_text + "\n", encoding="utf-8")

    if args.out_md:
        out_md = Path(args.out_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_build_markdown(runs, args.reference) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
