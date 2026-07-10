from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MAIN = PROJECT_ROOT / "classroom_pipeline" / "orchestrator_legacy.py"


def run_legacy_pipeline(args: Sequence[str]) -> int:
    """Run the existing production orchestrator behind the new public entry."""
    if not LEGACY_MAIN.exists():
        print(f"[ERROR] legacy pipeline entry not found: {LEGACY_MAIN}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(LEGACY_MAIN), *list(args)]
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)
