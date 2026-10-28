"""Small read-only localhost watcher for a v3 run directory.

The watcher intentionally uses only the Python standard library.  It serves
durable JSON state and append-only stage events; it does not start models,
modify a run, or expose a non-loopback listener.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


LOOPBACK_HOST = "127.0.0.1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"run_artifact_not_found:{path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"run_artifact_must_be_object:{path.name}")
    return value


def _read_events_incremental(
    path: Path,
    *,
    since: int = 0,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int, int]:
    """Read only appended bytes and return rows, next byte offset and seq."""

    if not path.is_file():
        return [], 0, max(0, int(since))
    file_size = path.stat().st_size
    start = max(0, int(offset))
    if start > file_size:
        start = 0
    rows: list[dict[str, Any]] = []
    last_seq = max(0, int(since))
    with path.open("rb") as handle:
        handle.seek(start)
        if start:
            # A caller may persist a byte cursor taken before a line was
            # flushed. Align only when the cursor is inside a line. At an
            # exact newline boundary, seeking to the cursor must preserve the
            # first record appended afterwards.
            handle.seek(start - 1)
            previous_byte = handle.read(1)
            handle.seek(start)
            if previous_byte != b"\n":
                handle.readline()
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                # Do not consume a partial writer line; retrying with the same
                # offset will see the complete record later.
                handle.seek(line_start)
                break
            if not line.strip():
                continue
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(value, dict):
                continue
            try:
                seq = int(value.get("seq", last_seq + 1))
            except (TypeError, ValueError):
                seq = last_seq + 1
            last_seq = max(last_seq, seq)
            if seq > int(since):
                value.setdefault("_byte_offset", line_start)
                rows.append(value)
        next_offset = handle.tell()
    return rows, next_offset, last_seq


def _read_events(path: Path, *, since: int = 0, offset: int = 0) -> list[dict[str, Any]]:
    """Compatibility wrapper returning rows while honoring a byte cursor."""

    rows, _next_offset, _last_seq = _read_events_incremental(path, since=since, offset=offset)
    for row in rows:
        row.pop("_byte_offset", None)
    return rows


def _status_payload(run_dir: Path, cursor: _EventCursor | None = None) -> dict[str, Any]:
    state = _read_json(run_dir / "run_state.v1.json")
    active_cursor = cursor or _EventCursor(run_dir / "stage_events.v1.jsonl")
    last_event, _offset = active_cursor.status_last()
    return {
        "schema_version": "classroom_hybrid_ai.watch_status.v1",
        "run_dir": str(run_dir),
        "state": state,
        "last_event": last_event,
        "event_count": active_cursor.count,
        "event_seq": active_cursor.last_seq,
        "event_byte_offset": active_cursor.offset,
    }


def _relative_artifact(root: Path, uri: str) -> Path:
    """Resolve a URL artifact URI without allowing path traversal."""

    candidate = (root / unquote(uri)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("watch_artifact_must_be_relative") from exc
    return candidate


class _EventCursor:
    """Per-server append cursor used by status and SSE endpoints."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.offset = 0
        self.last_seq = 0
        self.count = 0
        self.last_event: dict[str, Any] | None = None
        self.seq_offsets: dict[int, int] = {}

    def read(self, *, since: int = 0, offset: int | None = None) -> tuple[list[dict[str, Any]], int]:
        with self.lock:
            requested_since = max(0, int(since))
            if offset is None:
                if requested_since == 0:
                    # ``since=0`` is the initial full-history request only.
                    # Once this cursor has consumed the file, keep the byte
                    # offset so repeated SSE/status polls read appended bytes
                    # instead of rescanning an already-consumed 100k-row log.
                    start = 0 if (self.offset == 0 and self.last_seq == 0 and self.count == 0) else self.offset
                elif requested_since >= self.last_seq:
                    start = self.offset
                else:
                    start = self.seq_offsets.get(requested_since + 1, 0)
            else:
                start = max(0, int(offset))
            rows, next_offset, last_seq = _read_events_incremental(
                self.path,
                since=requested_since,
                offset=start,
            )
            # ``_read_events_incremental`` only emits rows after ``since``;
            # cursor metadata records every line read so a later seq request
            # can jump directly to its byte offset.
            for row in rows:
                try:
                    seq = int(row.get("seq", 0))
                    byte_offset = int(row.pop("_byte_offset", next_offset))
                except (TypeError, ValueError):
                    continue
                self.seq_offsets.setdefault(seq, byte_offset)
            self.offset = max(self.offset, next_offset)
            self.last_seq = max(self.last_seq, last_seq)
            if rows:
                self.count += len(rows)
                self.last_event = rows[-1]
            return rows, next_offset

    def status_last(self) -> tuple[dict[str, Any] | None, int]:
        if self.offset == 0 and self.last_seq == 0:
            self.read(since=0, offset=0)
        else:
            self.read(since=self.last_seq, offset=self.offset)
        return self.last_event, self.offset


class _WatchHandler(BaseHTTPRequestHandler):
    server: "WatchHTTPServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        # The CLI already reports the listener.  Avoid writing request data to
        # stdout, which must remain machine-readable for command callers.
        return

    def _send_bytes(self, body: bytes, *, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, value: Any, *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, content_type="application/json; charset=utf-8", status=status)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({
            "schema_version": "classroom_hybrid_ai.watch_error.v1",
            "status": "error",
            "error": {"code": message.split(":", 1)[0], "message": message},
        }, status=status)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib HTTP handler API
        self._handle_request()

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler API
        self._handle_request()

    def _handle_request(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path in {"/", "/index.html"}:
                html = (
                    "<!doctype html><meta charset='utf-8'>"
                    "<title>Classroom Hybrid v3 watch</title>"
                    "<h1>Classroom Hybrid v3</h1>"
                    "<pre id='state'>loading…</pre>"
                    "<script>fetch('/api/status').then(r=>r.json()).then(x=>"
                    "document.querySelector('#state').textContent=JSON.stringify(x,null,2))</script>"
                ).encode("utf-8")
                self._send_bytes(html, content_type="text/html; charset=utf-8")
                return
            if path in {"/api/status", "/status.json"}:
                self._send_json(_status_payload(self.server.run_dir, self.server.event_cursor))
                return
            if path == "/events" or path == "/api/events":
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=False)
                try:
                    since = max(0, int(query.get("since", ["0"])[0]))
                    offset_value = query.get("offset", [None])[0]
                    offset = None if offset_value in (None, "") else max(0, int(offset_value))
                except (TypeError, ValueError) as exc:
                    raise ValueError("watch_event_cursor_invalid") from exc
                rows, next_offset = self.server.event_cursor.read(since=since, offset=offset)
                body = b"".join(
                    (f"id: {row.get('seq', index)}\nevent: stage\ndata: "
                     + json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode("utf-8")
                    for index, row in enumerate(rows, since + 1)
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Events-Next-Offset", str(next_offset))
                self.send_header("X-Events-Last-Seq", str(self.server.event_cursor.last_seq))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
                return
            if path.startswith("/artifact/"):
                artifact = _relative_artifact(self.server.run_dir, path[len("/artifact/"):])
                if not artifact.is_file():
                    raise FileNotFoundError(f"run_artifact_not_found:{artifact.name}")
                content_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
                self._send_bytes(artifact.read_bytes(), content_type=content_type)
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "watch_route_not_found")
        except (ValueError, FileNotFoundError, OSError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST if isinstance(exc, ValueError) else HTTPStatus.NOT_FOUND, str(exc))


class WatchHTTPServer(ThreadingHTTPServer):
    """Threaded read-only server bound to one run directory."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], run_dir: Path) -> None:
        self.run_dir = run_dir
        self.event_cursor = _EventCursor(run_dir / "stage_events.v1.jsonl")
        super().__init__(server_address, _WatchHandler)


# Keep a descriptive alias for callers that treat the watcher as a server
# object.  ``serve_watch`` remains the public constructor used by the CLI.
WatchServer = WatchHTTPServer


def serve_watch(run_dir: str | Path, *, host: str = LOOPBACK_HOST, port: int = 8765) -> WatchHTTPServer:
    """Create (but do not start) a watcher bound strictly to 127.0.0.1."""

    if host != LOOPBACK_HOST:
        raise ValueError("watch_host_must_be_127.0.0.1")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("watch_port_invalid")
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"watch_run_dir_not_found:{root}")
    return WatchHTTPServer((LOOPBACK_HOST, port), root)


__all__ = ["LOOPBACK_HOST", "WatchHTTPServer", "WatchServer", "serve_watch"]
