"""Auditable run-directory storage with atomic metadata writes."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from .hashing import canonical_sha256, sha256_file


SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(value: str, *, fallback: str = "item") -> str:
    normalized = SAFE_ID.sub("-", value.strip()).strip("-.")
    return normalized or fallback


def atomic_write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def atomic_write_jsonl(path: Path, rows: Iterable[Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if hasattr(row, "to_dict"):
                row = row.to_dict()
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": count,
    }


def append_jsonl(path: Path, row: Any) -> dict[str, Any]:
    """Append one JSON object without rewriting the existing JSONL file.

    Stage progress is a single-writer log. Rebuilding that file for every
    update makes a long run quadratic in both I/O and wall time, so this
    primitive writes exactly one newline-terminated record in append mode.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    if not isinstance(row, dict):
        raise TypeError("jsonl_row_must_be_object")
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        offset = handle.tell()
        handle.write(encoded)
        handle.flush()
    return {
        "path": path.as_posix(),
        "offset": offset,
        "bytes": len(encoded.encode("utf-8")),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_row_not_object:{path}:{line_number}")
            yield value


class JsonlStreamWriter:
    """Write large JSONL artifacts atomically without retaining rows in RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.temporary = self.path.with_name(self.path.name + ".tmp")
        self._handle: TextIO | None = None
        self.rows = 0
        self.receipt: dict[str, Any] | None = None

    def __enter__(self) -> "JsonlStreamWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, row: Any) -> None:
        if self._handle is None:
            raise RuntimeError("jsonl_writer_not_open")
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        if not isinstance(row, dict):
            raise TypeError("jsonl_row_must_be_object")
        self._handle.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self.rows += 1

    def close(self, *, commit: bool = True) -> dict[str, Any] | None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        if not commit:
            if self.temporary.exists():
                self.temporary.unlink()
            return None
        os.replace(self.temporary, self.path)
        self.receipt = {
            "path": self.path.as_posix(),
            "sha256": sha256_file(self.path),
            "bytes": self.path.stat().st_size,
            "rows": self.rows,
        }
        return self.receipt

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close(commit=exc_type is None)


class RunStore:
    def __init__(
        self,
        output_root: Path,
        video_id: str,
        config_digest: str,
        *,
        unique_run: bool = False,
        run_id: str | None = None,
    ) -> None:
        self.video_id = safe_id(video_id, fallback="video")
        experiment_digest = canonical_sha256({"video_id": self.video_id, "config": config_digest})
        if run_id is not None:
            self.run_id = safe_id(run_id, fallback=experiment_digest[:12])
        elif unique_run:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            self.run_id = f"{timestamp}-{experiment_digest[:12]}"
        else:
            # Compatibility path for the legacy HybridPipeline.  Formal v3.1
            # runs always pass ``unique_run=True``.
            self.run_id = experiment_digest[:24]
        self.run_dir = output_root.resolve() / self.video_id / self.run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.evidence_dir = self.run_dir / "evidence"
        self.responses_dir = self.run_dir / "responses"
        for path in (self.artifacts_dir, self.evidence_dir, self.responses_dir):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def open(cls, run_dir: str | Path) -> "RunStore":
        """Open an existing run without deriving or creating a different ID."""

        root = Path(run_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        self = cls.__new__(cls)
        self.run_dir = root
        self.run_id = root.name
        self.video_id = root.parent.name
        self.artifacts_dir = root / "artifacts"
        self.evidence_dir = root / "evidence"
        self.responses_dir = root / "responses"
        for path in (self.artifacts_dir, self.evidence_dir, self.responses_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def write_json(self, relative: str, value: Any) -> dict[str, Any]:
        return atomic_write_json(self.run_dir / relative, value)

    def write_jsonl(self, relative: str, rows: Iterable[Any]) -> dict[str, Any]:
        return atomic_write_jsonl(self.run_dir / relative, rows)

    def stream_jsonl(self, relative: str) -> JsonlStreamWriter:
        return JsonlStreamWriter(self.run_dir / relative)
