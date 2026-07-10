from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .model_resolver import resolve_model_reference, try_resolve_model_reference
from .stage_registry import PROJECT_ROOT, STAGES, missing_required_paths


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_doctor(*, video: str = "", strict_video: bool = False) -> int:
    """Check the file-level pipeline contract before starting a run."""
    errors: list[str] = []
    warnings: list[str] = []

    for path in missing_required_paths():
        errors.append(f"missing required pipeline file: {_rel(path)}")

    if video:
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.exists():
            msg = f"input video not found: {_rel(video_path)}"
            if strict_video:
                errors.append(msg)
            else:
                warnings.append(msg)

    for module_name in ("cv2", "ultralytics"):
        if importlib.util.find_spec(module_name) is None:
            warnings.append(f"python module not importable: {module_name}")

    for model_name in ("yolo11x.pt", "yolo11n-pose.pt"):
        try:
            resolve_model_reference(model_name, project_root=PROJECT_ROOT, required=True)
        except FileNotFoundError as exc:
            errors.append(str(exc))

    behavior_model = "runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt"
    if try_resolve_model_reference(behavior_model, project_root=PROJECT_ROOT) is None:
        warnings.append(
            "default behavior detector missing; runs will auto-disable behavior detection unless --behavior_det_model points to a valid model"
        )

    report: dict[str, Any] = {
        "status": "ok" if not errors else "failed",
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "registered_stages": len(STAGES),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2
