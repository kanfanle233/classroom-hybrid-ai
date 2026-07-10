import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    target = root / "run_pipeline.py"
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        legacy = root / "classroom_pipeline" / "orchestrator_legacy.py"
        completed = subprocess.run([sys.executable, str(legacy), *sys.argv[1:]], cwd=str(root), check=False)
        raise SystemExit(completed.returncode)
    print("[DEPRECATED] scripts/main/09_run_pipeline.py is a compatibility wrapper. Use: python run_pipeline.py run ...")
    completed = subprocess.run([sys.executable, str(target), "run", *sys.argv[1:]], cwd=str(root), check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
