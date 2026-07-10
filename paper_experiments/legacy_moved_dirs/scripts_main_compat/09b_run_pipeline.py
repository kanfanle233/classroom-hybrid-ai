import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility alias for run_pipeline.py.")
    _, extra = parser.parse_known_args()

    root = Path(__file__).resolve().parents[2]
    target = root / "run_pipeline.py"
    if any(arg in {"-h", "--help"} for arg in extra):
        legacy = root / "classroom_pipeline" / "orchestrator_legacy.py"
        completed = subprocess.run([sys.executable, str(legacy)] + extra, check=False, cwd=str(root))
        raise SystemExit(completed.returncode)
    print("[DEPRECATED] scripts/main/09b_run_pipeline.py is a compatibility wrapper. Use: python run_pipeline.py run ...")
    cmd = [sys.executable, str(target), "run"] + extra
    completed = subprocess.run(cmd, check=False, cwd=str(root))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
