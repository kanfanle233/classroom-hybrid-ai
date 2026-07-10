import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.contracts.schemas import ARTIFACT_VERSION, validate_verifier_calibration_report, write_json
from classroom_pipeline.verifier.metrics import (
    binary_target_from_label,
    brier_score,
    ece_and_bins,
    fit_temperature_brier,
    parse_prob,
    parse_reference_label,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _write_reliability_diagram(path: Path, before_bins: List[Dict[str, Any]], after_bins: List[Dict[str, Any]]) -> None:
    width = 640
    height = 420
    margin_left = 70
    margin_right = 24
    margin_top = 36
    margin_bottom = 56
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def px_x(value: float) -> float:
        return margin_left + _clamp01(value) * plot_w

    def px_y(value: float) -> float:
        return margin_top + (1.0 - _clamp01(value)) * plot_h

    def render_points(rows: List[Dict[str, Any]], color: str) -> List[str]:
        parts: List[str] = []
        for row in rows:
            conf = _clamp01(row.get("conf", 0.0))
            acc = _clamp01(row.get("acc", 0.0))
            count = max(0, int(row.get("count", 0)))
            radius = 3.0 + min(8.0, count / 25.0)
            parts.append(
                f"<circle cx='{px_x(conf):.2f}' cy='{px_y(acc):.2f}' r='{radius:.2f}' "
                f"fill='{color}' fill-opacity='0.75' stroke='white' stroke-width='1' />"
            )
        return parts

    grid_lines: List[str] = []
    for tick in range(6):
        value = tick / 5.0
        x = px_x(value)
        y = px_y(value)
        grid_lines.append(
            f"<line x1='{x:.2f}' y1='{margin_top}' x2='{x:.2f}' y2='{height - margin_bottom}' "
            "stroke='#d9dde3' stroke-width='1' />"
        )
        grid_lines.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' "
            "stroke='#d9dde3' stroke-width='1' />"
        )

    ticks: List[str] = []
    for tick in range(6):
        value = tick / 5.0
        x = px_x(value)
        y = px_y(value)
        label = f"{value:.1f}"
        ticks.append(
            f"<text x='{x:.2f}' y='{height - margin_bottom + 22}' font-size='12' text-anchor='middle' "
            "fill='#344054'>" + label + "</text>"
        )
        ticks.append(
            f"<text x='{margin_left - 14}' y='{y + 4:.2f}' font-size='12' text-anchor='end' "
            "fill='#344054'>" + label + "</text>"
        )

    diagonal = (
        f"<line x1='{px_x(0.0):.2f}' y1='{px_y(0.0):.2f}' "
        f"x2='{px_x(1.0):.2f}' y2='{px_y(1.0):.2f}' "
        "stroke='#98a2b3' stroke-width='2' stroke-dasharray='8 6' />"
    )
    before_points = "".join(render_points(before_bins, "#f97316"))
    after_points = "".join(render_points(after_bins, "#2563eb"))

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
  <rect width='{width}' height='{height}' fill='#f8fafc' />
  <text x='{margin_left}' y='22' font-size='18' font-weight='700' fill='#0f172a'>Verifier Reliability Diagram</text>
  <text x='{margin_left}' y='{height - 16}' font-size='14' fill='#475467'>Predicted confidence</text>
  <text x='18' y='{margin_top + plot_h / 2:.2f}' font-size='14' fill='#475467'
        transform='rotate(-90 18 {margin_top + plot_h / 2:.2f})'>Empirical accuracy</text>
  {''.join(grid_lines)}
  <rect x='{margin_left}' y='{margin_top}' width='{plot_w}' height='{plot_h}' fill='none' stroke='#101828' stroke-width='2' />
  {diagonal}
  {before_points}
  {after_points}
  {''.join(ticks)}
  <rect x='{width - 180}' y='18' width='14' height='14' rx='3' fill='#f97316' />
  <text x='{width - 160}' y='30' font-size='12' fill='#344054'>Before scaling</text>
  <rect x='{width - 180}' y='42' width='14' height='14' rx='3' fill='#2563eb' />
  <text x='{width - 160}' y='54' font-size='12' fill='#344054'>After scaling</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate verified_events probabilities and export verifier_calibration_report.json"
    )
    parser.add_argument("--verified", required=True, type=str, help="verified_events.jsonl")
    parser.add_argument("--out", required=True, type=str, help="verifier_calibration_report.json")
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--target_field", default="auto", type=str, help="ground-truth label field or auto")
    parser.add_argument("--prob_field", default="p_match", type=str)
    parser.add_argument("--num_bins", default=10, type=int)
    parser.add_argument("--disable_temperature_scaling", type=int, default=0)
    parser.add_argument("--diagram_out", default="", type=str, help="optional reliability diagram svg output")
    args = parser.parse_args()

    verified_path = Path(args.verified).resolve()
    out_path = Path(args.out).resolve()
    diagram_out = Path(args.diagram_out).resolve() if args.diagram_out else None
    rows = _load_jsonl(verified_path)

    probs: List[float] = []
    targets: List[int] = []
    ref_source_counts: Dict[str, int] = {}
    for row in rows:
        label, source = parse_reference_label(row, target_field=str(args.target_field))
        target = binary_target_from_label(label)
        prob = parse_prob(row, field=str(args.prob_field))
        probs.append(prob)
        targets.append(target)
        ref_source_counts[source] = ref_source_counts.get(source, 0) + 1

    before_ece, before_bins = ece_and_bins(probs, targets, num_bins=int(args.num_bins))
    before_brier = brier_score(probs, targets)

    use_scaling = int(args.disable_temperature_scaling) != 1
    if use_scaling:
        temperature, scaled_probs = fit_temperature_brier(probs, targets)
    else:
        temperature, scaled_probs = 1.0, list(probs)

    after_ece, after_bins = ece_and_bins(scaled_probs, targets, num_bins=int(args.num_bins))
    after_brier = brier_score(scaled_probs, targets)

    report: Dict[str, Any] = {
        "split": str(args.split),
        "ece": float(after_ece),
        "brier": float(after_brier),
        "temperature": float(temperature),
        "temperature_scaling_enabled": bool(use_scaling),
        "bin_stats": after_bins,
        "before_after": {
            "before": {
                "ece": float(before_ece),
                "brier": float(before_brier),
                "temperature": 1.0,
                "bin_stats": before_bins,
            },
            "after": {
                "ece": float(after_ece),
                "brier": float(after_brier),
                "temperature": float(temperature),
                "bin_stats": after_bins,
            },
        },
        "config": {
            "prob_field": str(args.prob_field),
            "target_field": str(args.target_field),
            "num_bins": int(args.num_bins),
            "source_file": str(verified_path),
            "reference_source": ref_source_counts,
        },
        "artifact_version": ARTIFACT_VERSION,
    }

    ok, msg = validate_verifier_calibration_report(report)
    if not ok:
        raise ValueError(f"invalid calibration report payload: {msg}")
    write_json(out_path, report)
    if diagram_out is not None:
        _write_reliability_diagram(diagram_out, before_bins, after_bins)
    print(f"[DONE] calibration report: {out_path}")
    if diagram_out is not None:
        print(f"[DONE] reliability diagram: {diagram_out}")
    print(
        f"[INFO] total={len(rows)} ece_before={before_ece:.4f} ece_after={after_ece:.4f} "
        f"brier_before={before_brier:.4f} brier_after={after_brier:.4f} T={temperature:.3f}"
    )


if __name__ == "__main__":
    main()
