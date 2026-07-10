import runpy
import sys
from pathlib import Path


def main(wrapper_file: str) -> None:
    root = Path(wrapper_file).resolve().parents[2]
    target = root / "classroom_pipeline" / "pipeline" / Path(wrapper_file).name
    if not target.exists():
        raise FileNotFoundError(f"Pipeline stage moved target not found: {target}")
    sys.path.insert(0, str(root))
    runpy.run_path(str(target), run_name="__main__")
