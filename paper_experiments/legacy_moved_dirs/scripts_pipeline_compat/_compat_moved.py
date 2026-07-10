from __future__ import annotations

import runpy
import sys
from pathlib import Path


MAINTENANCE_NAMES = {
    "_audit_step1_data.py",
    "_audit_step2_balanced_sampling.py",
    "_audit_step3_teacher_labels.py",
    "_audit_step6_machine_checks.py",
}


def main(wrapper_file: str) -> None:
    root = Path(wrapper_file).resolve().parents[2]
    name = Path(wrapper_file).name
    if name in MAINTENANCE_NAMES:
        target = root / "scripts" / "maintenance" / "pipeline_audit" / name
    else:
        target = root / "scripts" / "experiments" / "pipeline_legacy" / name
    if not target.exists():
        raise FileNotFoundError(f"Moved script target not found: {target}")
    sys.path.insert(0, str(root))
    runpy.run_path(str(target), run_name="__main__")
