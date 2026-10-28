"""Run one LocateAnything-only request and render its spatial detections.

This utility deliberately does not import or invoke YOLO, OCR, or Qwen.
It is intended for validating the isolated LocateAnything HTTP worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT / "src"))

from classroom_hybrid_ai.adapters import LocateAnythingAdapter  # noqa: E402


COLORS = (
    "#00E5FF",
    "#FFB300",
    "#7C4DFF",
    "#00C853",
    "#FF1744",
    "#2979FF",
    "#F50057",
    "#64DD17",
)


def _post_empty(url: str, timeout: float) -> dict[str, Any]:
    req = urllib_request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as opened:
        value = json.loads(opened.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("unload response must be an object")
    return value


def _image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render(
    image_path: Path,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    width, height = image.size
    line_width = max(2, round(min(width, height) / 180))
    label_colors: dict[str, str] = {}
    for index, detection in enumerate(detections):
        label = str(detection.get("label") or "object")
        color = label_colors.setdefault(label, COLORS[len(label_colors) % len(COLORS)])
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        x1 = max(0.0, min(float(width - 1), x1))
        x2 = max(0.0, min(float(width - 1), x2))
        y1 = max(0.0, min(float(height - 1), y1))
        y2 = max(0.0, min(float(height - 1), y2))
        if x2 <= x1 and y2 <= y1:
            radius = max(3, line_width * 2)
            draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), outline=color, width=line_width)
        else:
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        caption = f"{index + 1} {label}"
        text_box = draw.textbbox((x1, y1), caption, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0.0, y1 - text_height - 6)
        draw.rectangle((x1, text_y, x1 + text_width + 8, text_y + text_height + 6), fill=color)
        draw.text((x1 + 4, text_y + 2), caption, fill="black", font=font, stroke_width=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=94, subsampling=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="LocateAnything-only HTTP demo")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["student", "teacher", "desk"])
    parser.add_argument("--service-url", default="http://127.0.0.1:8011/locate")
    parser.add_argument("--health-url", default="http://127.0.0.1:8011/health")
    parser.add_argument("--unload-url", default="http://127.0.0.1:8011/unload")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.is_file():
        raise SystemExit(f"image does not exist: {image_path}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = LocateAnythingAdapter(
        backend="http",
        base_url=args.service_url,
        health_url=args.health_url,
        timeout=args.timeout,
    )
    health_before = adapter.health()
    unload: dict[str, Any] | None = None
    try:
        with Image.open(image_path) as opened:
            image_size = opened.size
        result = adapter.detect(image_path, args.labels, image_size=image_size)
    finally:
        adapter.close()
        unload = _post_empty(args.unload_url, min(args.timeout, 60.0))

    annotated_path = output_dir / "locate_annotated.jpg"
    _render(image_path, list(result["detections"]), annotated_path)
    receipt = {
        "schema_version": "classroom_hybrid_ai.locate_only_demo.v1",
        "status": "ok",
        "exclusive_model": "nvidia/LocateAnything-3B",
        "other_models_invoked": [],
        "input_image": str(image_path),
        "input_sha256": _image_sha256(image_path),
        "image_size": list(image_size),
        "labels": list(args.labels),
        "health_before": health_before,
        "detections": result["detections"],
        "raw_answer": result["raw_answer"],
        "unload": unload,
        "annotated_image": str(annotated_path),
    }
    receipt_path = output_dir / "locate_result.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "detections": len(receipt["detections"]),
        "annotated_image": receipt["annotated_image"],
        "receipt": str(receipt_path),
        "unload": unload,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
