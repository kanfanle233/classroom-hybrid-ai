from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_TOP_LEVEL = {
    ".agents",
    ".claude",
    ".git",
    ".ultralytics",
    ".vscode",
    "data",
    "docs",
    "models",
    "paper_experiments",
}
PROTECTED_NAMES = {
    "pipeline_manifest.json",
    "verified_events.jsonl",
    "actions.fusion_v2.jsonl",
    "event_queries.fusion_v2.jsonl",
    "gold_events.official.visual_223.jsonl",
}
CLEAN_SUFFIXES = {".pyc", ".tmp", ".log"}
CLEAN_DIR_NAMES = {"__pycache__", ".pytest_cache", "codex_smoke"}
ROOT_CLEAN_DIR_NAMES = {"tmp"}


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _is_empty_dir(path: Path) -> bool:
    try:
        return path.is_dir() and not any(path.iterdir())
    except OSError:
        return False


def _is_md_only_leaf_dir(path: Path) -> bool:
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    if not children:
        return False
    if any(child.is_dir() for child in children):
        return False
    files = [child for child in children if child.is_file()]
    return bool(files) and all(child.suffix.lower() == ".md" for child in files)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_protected(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT)
    parts = rel.parts
    if parts and parts[0] in PROTECTED_TOP_LEVEL:
        return True
    return path.name in PROTECTED_NAMES


def iter_clean_candidates(root: Path = PROJECT_ROOT) -> Iterable[dict]:
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        kept_dirs: list[str] = []
        for dirname in dirnames:
            child = current / dirname
            if _is_protected(child):
                continue
            if dirname in CLEAN_DIR_NAMES or (current == root and dirname in ROOT_CLEAN_DIR_NAMES):
                yield {
                    "path": str(child),
                    "kind": "directory",
                    "reason": "cache/temp directory",
                    "size_bytes": 0,
                    "sha256": "",
                    "protected": False,
                }
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in filenames:
            path = current / filename
            if _is_protected(path) or path.suffix.lower() not in CLEAN_SUFFIXES:
                continue
            yield {
                "path": str(path),
                "kind": "file",
                "reason": "regenerable cache/log/temp file",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "protected": False,
            }

    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        current = Path(dirpath)
        if current == root or _is_protected(current):
            continue
        if _is_empty_dir(current):
            yield {
                "path": str(current),
                "kind": "directory",
                "reason": "empty directory",
                "size_bytes": 0,
                "sha256": "",
                "protected": False,
            }
        elif _is_md_only_leaf_dir(current):
            yield {
                "path": str(current),
                "kind": "directory",
                "reason": "md-only leaf directory",
                "size_bytes": _dir_size(current),
                "sha256": "",
                "protected": False,
            }


def write_manifest(manifest_path: Path, rows: list[dict], *, status: str = "dry_run", errors: list[str] | None = None) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "candidate_count": len(rows),
        "total_size_bytes": sum(int(r.get("size_bytes", 0)) for r in rows),
        "errors": errors or [],
        "candidates": rows,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_cleanup(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    for row in sorted(rows, key=lambda r: len(Path(str(r["path"])).parts), reverse=True):
        path = Path(str(row["path"]))
        if not path.exists():
            continue
        if not path.is_relative_to(PROJECT_ROOT):
            errors.append(f"refused outside workspace: {path}")
            continue
        if _is_protected(path):
            errors.append(f"refused protected path: {path}")
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe cleanup for regenerable pipeline artifacts.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true", help="Delete candidates listed by the current rules.")
    parser.add_argument("--manifest", default="output/cleanup_manifest.json")
    args = parser.parse_args(argv)

    rows = list(iter_clean_candidates(PROJECT_ROOT))
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    if args.apply:
        errors = apply_cleanup(rows)
        write_manifest(manifest, rows, status="applied" if not errors else "applied_with_errors", errors=errors)
        print(f"[APPLY] cleanup candidates={len(rows)} manifest={manifest} errors={len(errors)}")
        return 0 if not errors else 2

    write_manifest(manifest, rows, status="dry_run")
    print(f"[DRY-RUN] cleanup candidates={len(rows)} manifest={manifest}")
    return 0
