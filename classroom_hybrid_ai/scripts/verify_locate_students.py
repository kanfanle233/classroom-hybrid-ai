"""Call the local LocateAnything worker and preserve raw/student-box evidence."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from urllib import request as urllib_request

import cv2

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from classroom_hybrid_ai.hashing import sha256_file
from classroom_hybrid_ai.storage import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--prompt", default="student")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011/locate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    image_path = args.image.resolve()
    raw_image = image_path.read_bytes()
    payload = {
        "prompt": args.prompt,
        "image": "data:image/jpeg;base64," + base64.b64encode(raw_image).decode("ascii"),
    }
    http_request = urllib_request.Request(
        args.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(http_request, timeout=args.timeout) as opened:
        response = json.loads(opened.read().decode("utf-8"))
    if not isinstance(response, dict):
        raise TypeError("locate_response_must_be_object")

    detections = response.get("detections", [])
    if not isinstance(detections, list):
        raise TypeError("locate_detections_must_be_list")
    canvas = cv2.imread(str(image_path))
    if canvas is None:
        raise RuntimeError(f"image_decode_failed:{image_path}")
    for index, detection in enumerate(detections, start=1):
        if not isinstance(detection, dict):
            continue
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        label = str(detection.get("label") or args.prompt)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # This is explicitly a per-image detection index, not a cross-frame
        # identity.  The untouched raw model answer is stored beside it.
        cv2.putText(
            canvas, f"{index}:{label}", (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )
        detection["detection_index"] = index

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = output_dir / "student_boxes.jpg"
    if not cv2.imwrite(str(annotated), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"annotated_image_write_failed:{annotated}")
    receipt = {
        "schema_version": "classroom_hybrid_ai.locate_student_verification.v1",
        "input": str(image_path),
        "input_sha256": sha256_file(image_path),
        "prompt": args.prompt,
        "raw_answer": response.get("answer"),
        "detections": detections,
        "detection_count": len(detections),
        "annotated_image": str(annotated),
        "annotated_sha256": sha256_file(annotated),
        "index_scope": "single_image_detection_order_not_persistent_track_id",
    }
    atomic_write_json(output_dir / "response.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
