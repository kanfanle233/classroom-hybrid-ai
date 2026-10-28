"""Launch the packaged LocateAnything worker with the project's source tree."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from classroom_hybrid_ai.resources.locate_anything_service import (  # noqa: E402,F401
    LocateWorker, _decode_data_url, main, make_handler,
)

if __name__ == "__main__":
    raise SystemExit(main())
