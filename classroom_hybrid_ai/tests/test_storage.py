from __future__ import annotations

import json
from pathlib import Path

from classroom_hybrid_ai.storage import (
    JsonlStreamWriter,
    RunStore,
    atomic_write_jsonl,
    iter_jsonl,
    safe_id,
)


def test_safe_id_removes_path_characters() -> None:
    assert safe_id("../lesson 01/clip") == "lesson-01-clip"


def test_jsonl_write_is_complete(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    receipt = atomic_write_jsonl(path, [{"b": 2, "a": 1}, {"x": "中文"}])
    assert receipt["rows"] == 2
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        {"a": 1, "b": 2},
        {"x": "中文"},
    ]


def test_run_store_is_content_addressed(tmp_path: Path) -> None:
    left = RunStore(tmp_path, "lesson", "a" * 64)
    right = RunStore(tmp_path, "lesson", "a" * 64)
    assert left.run_id == right.run_id
    assert left.evidence_dir.is_dir()


def test_stream_jsonl_writes_atomically(tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    with JsonlStreamWriter(path) as writer:
        writer.write({"frame": 1})
        writer.write({"frame": 2})
    assert writer.receipt is not None
    assert writer.receipt["rows"] == 2
    assert list(iter_jsonl(path)) == [{"frame": 1}, {"frame": 2}]
    assert not path.with_name(path.name + ".tmp").exists()
