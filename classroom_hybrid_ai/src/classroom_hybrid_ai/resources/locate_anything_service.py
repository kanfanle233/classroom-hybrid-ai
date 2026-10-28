"""Isolated local HTTP worker for NVIDIA LocateAnything.

Run this in its isolated LocateAnything environment, never in the Qwen process:

    python scripts/locate_anything_service.py --model-dir /models/locate-anything-3b

The released model card names Linux as the supported OS.  This project also
ships a pinned Transformers 4.57.1 runtime that has been verified with the
configured Windows py310 interpreter; the HTTP boundary remains mandatory so
Qwen's Transformers runtime is never imported into this process.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classroom_hybrid_ai.adapters import LocateAnythingAdapter  # noqa: E402


class LocateWorker:
    def __init__(
        self,
        model_dir: Path,
        modules_cache: Path | None = None,
        transformers_runtime: Path | None = None,
    ) -> None:
        self.model_dir = model_dir.resolve()
        self.modules_cache = (
            modules_cache.resolve()
            if modules_cache is not None
            else (self.model_dir / ".hf_modules_nf4_457").resolve()
        )
        self.modules_cache.mkdir(parents=True, exist_ok=True)
        self.transformers_runtime = (
            transformers_runtime.resolve()
            if transformers_runtime is not None
            else (self.model_dir / ".runtime" / "transformers_4_57").resolve()
        )
        if not (self.transformers_runtime / "transformers" / "__init__.py").is_file():
            raise FileNotFoundError(
                f"pinned Transformers runtime is missing: {self.transformers_runtime}"
            )
        runtime_text = str(self.transformers_runtime)
        if runtime_text not in sys.path:
            sys.path.insert(0, runtime_text)
        # Transformers otherwise falls back to
        # C:\Users\<user>\.cache\huggingface\modules.  Keep the remote-code
        # index beside this pinned model snapshot so the worker is portable and
        # never depends on a protected user-profile cache.
        os.environ["HF_MODULES_CACHE"] = str(self.modules_cache)
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.adapter: LocateAnythingAdapter | None = None
        self.lock = threading.Lock()
        self.peak_memory: dict[str, int] = {}

    def locate(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload.get("prompt")
        encoded = payload.get("image")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt_required")
        started = time.perf_counter()
        image = _decode_data_url(encoded)
        decode_seconds = time.perf_counter() - started
        try:
            with self.lock:
                if self.adapter is None:
                    self.adapter = LocateAnythingAdapter(backend="local", model_path=self.model_dir)
                from classroom_hybrid_ai.qwen_worker import _cuda_memory_metrics, _reset_cuda_peaks
                self.adapter.profile_timings = bool(payload.get("profile_timings", False))
                self.adapter.release_cuda_cache = bool(payload.get("release_cuda_cache", False))
                _reset_cuda_peaks()
                try:
                    result = self.adapter.locate(
                        image, prompt, image_size=image.size,
                        max_new_tokens=payload.get("max_new_tokens", 256),
                        maximum_image_side=payload.get("maximum_image_side"),
                        generation_mode=payload.get("generation_mode", "hybrid"),
                    )
                finally:
                    metrics = _cuda_memory_metrics()
                    for key in ("max_allocated_bytes", "max_reserved_bytes"):
                        if isinstance(metrics.get(key), int):
                            self.peak_memory[key] = max(self.peak_memory.get(key, 0), metrics[key])
                            metrics[key] = self.peak_memory[key]
                    self.adapter.last_performance["memory"] = metrics
                    self.adapter.last_performance["timings_seconds"]["http_image_decode"] = decode_seconds
                return {"answer": result["raw_answer"], "detections": result["detections"], "loaded": True,
                        "metadata": result["metadata"]}
        finally:
            image.close()

    def unload(self) -> dict[str, Any]:
        with self.lock:
            if self.adapter is not None:
                self.adapter.close()
                self.adapter = None
        return {"status": "unloaded", "loaded": False}

    def health(self) -> dict[str, Any]:
        distributions = sorted(self.transformers_runtime.glob("transformers-*.dist-info"))
        runtime_version = (
            distributions[0].name.removeprefix("transformers-").removesuffix(".dist-info")
            if distributions else "unknown"
        )
        return {
            "status": "ok",
            "loaded": self.adapter is not None,
            "model_dir": str(self.model_dir),
            "modules_cache": str(self.modules_cache),
            "transformers_runtime": str(self.transformers_runtime),
            "transformers_version": runtime_version,
            "protocol_version": "locate.parameters.v1",
            "generation_parameters": ["max_new_tokens", "maximum_image_side", "generation_mode"],
        }


def _decode_data_url(value: Any) -> Image.Image:
    if not isinstance(value, str) or not value.startswith("data:image/") or "," not in value:
        raise ValueError("image_must_be_data_url")
    _, encoded = value.split(",", 1)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("image_data_url_invalid") from exc
    image = Image.open(BytesIO(raw)).convert("RGB")
    image.load()
    return image


def make_handler(worker: LocateWorker):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._write(HTTPStatus.OK, worker.health())

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._payload()
                if self.path == "/locate":
                    self._write(HTTPStatus.OK, worker.locate(payload))
                elif self.path == "/unload":
                    self._write(HTTPStatus.OK, worker.unload())
                elif self.path == "/shutdown":
                    result = worker.unload()
                    self._write(HTTPStatus.OK, {**result, "status": "stopping"})
                    # HTTPServer.shutdown() must run outside the active request
                    # handler thread. This gives an owned worker a clean exit
                    # code after its GPU state has been released.
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except Exception as exc:
                adapter = getattr(worker, "adapter", None)
                self._write(HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}:{exc}",
                            "performance": getattr(adapter, "last_performance", None)})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 32 * 1024 * 1024:
                raise ValueError("request_too_large")
            value = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(value, dict):
                raise ValueError("payload_must_be_object")
            return value

        def _write(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Local LocateAnything HTTP worker")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--modules-cache",
        type=Path,
        default=None,
        help="Transformers remote-code cache; defaults to MODEL_DIR/.hf_modules_nf4_457",
    )
    parser.add_argument(
        "--transformers-runtime",
        type=Path,
        default=None,
        help="Pinned import target; defaults to MODEL_DIR/.runtime/transformers_4_57",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model_dir}")
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(LocateWorker(args.model_dir, args.modules_cache, args.transformers_runtime)),
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
