"""Run the source checkout without installing it into the Conda environment."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classroom_hybrid_ai.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
