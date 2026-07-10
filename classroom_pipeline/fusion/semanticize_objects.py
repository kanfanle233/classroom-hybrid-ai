import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classroom_pipeline.fusion.core import command_semanticize_objects


if __name__ == "__main__":
    raise SystemExit(command_semanticize_objects())
