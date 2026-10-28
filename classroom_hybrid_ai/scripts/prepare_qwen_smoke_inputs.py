"""Prepare the four auditable 480x416 evidence roles for the Qwen v3 gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from classroom_hybrid_ai.hashing import canonical_sha256, sha256_file
from classroom_hybrid_ai.storage import atomic_write_json


WIDTH = 480
HEIGHT = 416


def _read_resize(source: Path) -> np.ndarray:
    image = cv2.imread(str(source.resolve()))
    if image is None:
        raise RuntimeError(f"smoke_source_decode_failed:{source.resolve()}")
    return cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"smoke_input_write_failed:{path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--multi", type=Path, required=True)
    parser.add_argument("--occluded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    roles = {
        "single": (args.single.resolve(), _read_resize(args.single)),
        "multi": (args.multi.resolve(), _read_resize(args.multi)),
        "occluded": (args.occluded.resolve(), _read_resize(args.occluded)),
        "empty": (None, np.full((HEIGHT, WIDTH, 3), 24, dtype=np.uint8)),
    }
    rows: list[dict[str, object]] = []
    for role, (source, image) in roles.items():
        destination = output_dir / f"{role}.jpg"
        _write(destination, image)
        rows.append({
            "role": role,
            "path": str(destination),
            "source": None if source is None else str(source),
            "shape": [WIDTH, HEIGHT],
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        })
    manifest = {
        "schema_version": "classroom_hybrid_ai.qwen_smoke_inputs.v1",
        "roles": rows,
    }
    manifest["manifest_sha256"] = canonical_sha256({"roles": rows})
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
