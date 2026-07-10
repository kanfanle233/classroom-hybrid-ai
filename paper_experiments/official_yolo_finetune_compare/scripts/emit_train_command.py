from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _ps_quote(value: object) -> str:
    text = str(value).replace('"', '`"')
    return f'"{text}"'


def _cmd_line(py: str, script: str, args: list[tuple[str, object]]) -> str:
    parts = [f"& {_ps_quote(py)} {_ps_quote(script)}"]
    for key, value in args:
        parts.append(f"--{key} {_ps_quote(value)}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit official YOLO baseline training commands for PowerShell.")
    parser.add_argument("--profile", required=True, help="Path to YAML profile")
    parser.add_argument("--smoke_epochs", type=int, default=-1, help="Optional override")
    parser.add_argument("--full_epochs", type=int, default=-1, help="Optional override")
    parser.add_argument("--out_ps1", default="", help="Optional path for emitted commands")
    args = parser.parse_args()

    profile_path = Path(args.profile).resolve()
    if not profile_path.exists():
        raise FileNotFoundError(f"profile not found: {profile_path}")

    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    repo_root = Path(profile["repo_root"]).resolve()
    smoke_epochs = args.smoke_epochs if args.smoke_epochs > 0 else int(profile["smoke_epochs"])
    full_epochs = args.full_epochs if args.full_epochs > 0 else int(profile["full_epochs"])
    baseline_name = str(profile["baseline_name"])

    py = str(profile["python"])
    train_script = str((repo_root / str(profile["train_script"])).resolve())

    common_args = [
        ("data", str(profile["data"])),
        ("model", str(profile["model"])),
        ("imgsz", profile["imgsz"]),
        ("batch", profile["batch"]),
        ("device", profile["device"]),
        ("workers", profile["workers"]),
        ("project", str(profile["project"])),
        ("patience", profile["patience"]),
    ]

    smoke_cmd = _cmd_line(
        py,
        train_script,
        common_args + [("epochs", smoke_epochs), ("name", f"{baseline_name}_smoke{smoke_epochs}")],
    )
    full_cmd = _cmd_line(
        py,
        train_script,
        common_args + [("epochs", full_epochs), ("name", baseline_name)],
    )

    lines = [
        f'Set-Location {_ps_quote(repo_root)}',
        f'$py = {_ps_quote(py)}',
        f'$env:YOLO_CONFIG_DIR = {_ps_quote(profile["yolo_config_dir"])}',
        "",
        "# smoke run",
        smoke_cmd,
        "",
        "# full run",
        full_cmd,
        "",
        "# note",
        "# If the 80-epoch run is still improving near the end, rerun this script with --full_epochs 120 or --full_epochs 150.",
    ]

    content = "\n".join(lines) + "\n"
    print(content)

    if args.out_ps1:
        out_path = Path(args.out_ps1).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
