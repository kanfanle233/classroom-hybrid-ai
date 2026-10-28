"""LocateAnything adapter for open-vocabulary object/context localisation.

LocateAnything returns spatial boxes, not persistent video IDs.  This
subproject deliberately exposes only the object/context path; OCR, text
transcription and GUI-element recognition are outside its contract.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit, urlunsplit

from .qwen import (
    _coerce_http_response, _empty_cuda_cache, _get, _image_data_url, _image_file_bytes, _is_oom,
    _performance_record, _profile_sync, _to_local_pil,
)


_LOCATION_PATTERN = re.compile(
    r"(?:<ref>\s*(?P<label>.*?)\s*</ref>\s*)?<box>\s*(?P<coords>.*?)\s*</box>",
    flags=re.IGNORECASE | re.DOTALL,
)
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_UNSET = object()


class LocateAnythingResponseError(ValueError):
    """Raised when a spatial response is malformed."""


class LocateAnythingHealthError(ConnectionError):
    """Raised when the optional HTTP worker cannot report a healthy state."""


def _scale_location(coords: Sequence[float], image_size: tuple[int, int] | None) -> list[float]:
    if image_size is None:
        return [float(value) for value in coords]
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be (positive width, positive height)")
    if len(coords) == 2:
        return [coords[0] * width / 1000.0, coords[1] * height / 1000.0]
    return [
        coords[0] * width / 1000.0,
        coords[1] * height / 1000.0,
        coords[2] * width / 1000.0,
        coords[3] * height / 1000.0,
    ]


def _usable_device(value: Any) -> Any:
    if value is None:
        return None
    text = str(value)
    if text in {"meta", "disk"}:
        return None
    if isinstance(value, int) or text.isdigit():
        return f"cuda:{value}"
    return value


def _component_device(model: Any, component: str) -> Any:
    """Find the Accelerate device that consumes text or image inputs."""

    if component == "text":
        get_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_embeddings):
            try:
                embeddings = get_embeddings()
                device = _usable_device(getattr(getattr(embeddings, "weight", None), "device", None))
                if device is not None:
                    return device
            except (AttributeError, RuntimeError):
                pass

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        needles = (
            ("embed_tokens", "word_embeddings", "wte")
            if component == "text"
            else ("vision", "visual", "vision_tower", "image")
        )
        for module_name, mapped_device in device_map.items():
            if any(needle in str(module_name).lower() for needle in needles):
                device = _usable_device(mapped_device)
                if device is not None:
                    return device
        for mapped_device in device_map.values():
            device = _usable_device(mapped_device)
            if device is not None and str(device).startswith("cuda"):
                return device
    return _usable_device(getattr(model, "device", None))


def _move_value(value: Any, device: Any) -> Any:
    move = getattr(value, "to", None)
    if callable(move) and device is not None:
        return move(device)
    return value


def parse_locate_anything_response(
    text: str,
    *,
    image_size: tuple[int, int] | None = None,
    default_label: str = "object",
) -> list[dict[str, Any]]:
    """Parse official output, validate its range and normalize box corners."""

    if not isinstance(text, str):
        raise TypeError("LocateAnything response must be text")
    locations: list[dict[str, Any]] = []
    # LocateAnything commonly emits one <ref> followed by many <box> tags,
    # for example ``<ref>student</ref><box>...</box><box>...</box>``.  The
    # reference remains active until another <ref> appears.
    active_label = default_label
    for match in _LOCATION_PATTERN.finditer(text):
        numbers = [float(value) for value in _NUMBER_PATTERN.findall(match.group("coords"))]
        if len(numbers) not in {2, 4}:
            raise LocateAnythingResponseError(
                f"a <box> must contain two point coordinates or four box coordinates: {numbers!r}"
            )
        if any(value < 0 or value > 1000 for value in numbers):
            raise LocateAnythingResponseError(
                f"LocateAnything coordinates must be within [0, 1000]: {numbers!r}"
            )
        raw_numbers = list(numbers)
        coordinates_reordered = False
        if len(numbers) == 4:
            # The released model can occasionally emit the two valid corner
            # points in reverse order.  Sorting each axis preserves the exact
            # rectangle and prevents one recoverable detection from aborting
            # an hours-long video.  Keep the emitted values for auditability.
            numbers = [
                min(raw_numbers[0], raw_numbers[2]),
                min(raw_numbers[1], raw_numbers[3]),
                max(raw_numbers[0], raw_numbers[2]),
                max(raw_numbers[1], raw_numbers[3]),
            ]
            coordinates_reordered = numbers != raw_numbers
        emitted_label = match.group("label")
        if emitted_label is not None and emitted_label.strip():
            active_label = emitted_label.strip()
        label = active_label
        scaled = _scale_location(numbers, image_size)
        kind = "point" if len(numbers) == 2 else "box"
        if kind == "point":
            bbox = [scaled[0], scaled[1], scaled[0], scaled[1]]
        else:
            bbox = scaled
        locations.append(
            {
                "label": label,
                "bbox": bbox,
                "score": None,
                "class_id": None,
                "track_id": None,
                "source": "locate_anything",
                "attributes": {
                    "location_type": kind,
                    "coordinates_1000": numbers,
                    "coordinates_reordered": coordinates_reordered,
                    **(
                        {"raw_coordinates_1000": raw_numbers}
                        if coordinates_reordered
                        else {}
                    ),
                    "coordinate_space": "pixels" if image_size is not None else "normalized_0_1000",
                },
            }
        )
    return locations


class LocateAnythingAdapter:
    """HTTP client or lazy local NF4 LocateAnything inference."""

    def __init__(
        self,
        *,
        backend: str = "http",
        model_path: os.PathLike[str] | str | None = None,
        base_url: str = "http://127.0.0.1:8010/locate",
        health_url: str | None = None,
        timeout: float = 120.0,
        http_client: Any = None,
        local_loader: Callable[[Path], tuple[Any, Any]] | None = None,
        device_map: str = "auto",
        max_new_tokens: int = 256,
        maximum_image_side: int | None = None,
        generation_mode: str = "hybrid",
        release_cuda_cache: bool = False,
        profile_timings: bool = False,
    ) -> None:
        backend = backend.strip().lower()
        if backend not in {"http", "local"}:
            raise ValueError("backend must be 'http' or 'local'")
        self.backend = backend
        self.model_path: Path | None = None
        if backend == "local":
            if model_path is None:
                raise ValueError("model_path is required for the local backend")
            self.model_path = Path(model_path).expanduser().resolve()
            if not self.model_path.is_dir():
                raise FileNotFoundError(
                    f"local LocateAnything model must be an existing directory: {self.model_path}"
                )
        self.base_url = base_url.rstrip("/")
        self.health_url = (health_url or self._derive_health_url(self.base_url)).rstrip("/")
        self.timeout = float(timeout)
        self.http_client = http_client
        self.local_loader = local_loader
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.maximum_image_side = maximum_image_side
        self.generation_mode = generation_mode
        self.release_cuda_cache = bool(release_cuda_cache)
        self.profile_timings = bool(profile_timings)
        self._parameters(max_new_tokens, maximum_image_side, generation_mode)
        self.last_performance = _performance_record(backend)
        self._effective_parameters: dict[str, Any] | None = None
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None

    @staticmethod
    def _parameters(tokens: int, side: int | None, mode: str) -> dict[str, Any]:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if side is not None and (isinstance(side, bool) or not isinstance(side, int) or side < 1):
            raise ValueError("maximum_image_side must be a positive integer or null")
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("generation_mode must be non-empty text")
        return {"max_new_tokens": tokens, "maximum_image_side": side, "generation_mode": mode}

    @staticmethod
    def _derive_health_url(base_url: str) -> str:
        """Derive the documented ``/health`` endpoint from ``/locate``."""

        parsed = urlsplit(base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/locate"):
            path = path[: -len("/locate")] or "/"
        if not path.endswith("/"):
            path += "/"
        return urlunsplit((parsed.scheme, parsed.netloc, f"{path}health", "", ""))

    @property
    def loaded(self) -> bool:
        return self._model is not None

    parse_response = staticmethod(parse_locate_anything_response)

    def _load_local(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor, self._tokenizer
        assert self.model_path is not None
        if self.local_loader is not None:
            loaded = self.local_loader(self.model_path)
            if len(loaded) == 2:
                self._model, self._processor = loaded
                self._tokenizer = getattr(self._processor, "tokenizer", None)
            elif len(loaded) == 3:
                self._model, self._processor, self._tokenizer = loaded
            else:
                raise ValueError("local_loader must return (model, processor[, tokenizer])")
            return self._model, self._processor, self._tokenizer

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        dtype = torch.float16
        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
        except (AttributeError, RuntimeError):
            pass
        quantization = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        # LocateAnything supplies custom modelling/processing code.  Both flags
        # are deliberate: execute the local snapshot's code and never fetch at
        # runtime.
        common = {"trust_remote_code": True, "local_files_only": True}
        # Transformers 5.x detects that this Qwen2.5-derived snapshot carries
        # the historical Mistral tokenizer regex.  Without the explicit fix,
        # real inference degenerates into repeated unrelated tokens and never
        # emits box blocks even though health/interface tests pass.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(self.model_path), fix_mistral_regex=True, **common
        )
        processor = transformers.AutoProcessor.from_pretrained(
            str(self.model_path), fix_mistral_regex=True, **common
        )
        model_class = (
            getattr(transformers, "AutoModel", None)
            or getattr(transformers, "AutoModelForImageTextToText", None)
            or getattr(transformers, "AutoModelForVision2Seq", None)
            or getattr(transformers, "AutoModelForCausalLM")
        )
        model = model_class.from_pretrained(
            str(self.model_path),
            quantization_config=quantization,
            device_map=self.device_map,
            torch_dtype=dtype,
            **common,
        )
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
        self._model, self._processor, self._tokenizer = model, processor, tokenizer
        return model, processor, tokenizer

    def _http_answer(self, prompt: str, image: Any, parameters: Mapping[str, Any]) -> str:
        started = time.perf_counter()
        payload = {"prompt": prompt, "image": _image_data_url(image), **parameters,
                   "profile_timings": self.profile_timings, "release_cuda_cache": self.release_cuda_cache}
        self.last_performance["timings_seconds"]["image_preprocess"] = time.perf_counter() - started
        self.last_performance["counts"]["request_bytes"] = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        started = time.perf_counter()
        self.last_performance["counts"]["http_requests"] = 1
        client = self.http_client
        if client is not None:
            post = getattr(client, "post", None)
            if callable(post):
                response = post(self.base_url, json=payload, timeout=self.timeout)
                response = _coerce_http_response(response)
            elif callable(client):
                response = client(payload)
            else:
                raise TypeError("http_client must be callable or requests-like")
        else:
            request = urllib_request.Request(
                self.base_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib_request.urlopen(request, timeout=self.timeout) as opened:
                    response = _coerce_http_response(opened)
            except urllib_error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                    payload = json.loads(body)
                    detail = payload.get("error", body) if isinstance(payload, Mapping) else body
                    if isinstance(payload, Mapping) and isinstance(payload.get("performance"), Mapping):
                        self.last_performance["worker_performance"] = dict(payload["performance"])
                        self.last_performance["counts"]["inference_calls"] = _get(_get(payload["performance"], "counts", {}), "inference_calls")
                except Exception:
                    detail = str(exc)
                self.last_performance["timings_seconds"]["http_request"] = time.perf_counter() - started
                raise LocateAnythingResponseError(
                    f"LocateAnything worker returned HTTP {exc.code}: {detail}"
                ) from exc
        self.last_performance["timings_seconds"]["http_request"] = time.perf_counter() - started
        metadata = _get(response, "metadata", {})
        effective = _get(metadata, "effective_parameters")
        self._effective_parameters = dict(effective) if isinstance(effective, Mapping) else None
        remote_performance = _get(metadata, "performance")
        if isinstance(remote_performance, Mapping):
            self.last_performance["worker_performance"] = dict(remote_performance)
            self.last_performance["counts"]["inference_calls"] = _get(_get(remote_performance, "counts", {}), "inference_calls")
        else:
            self.last_performance["counts"]["inference_calls"] = None
            self.last_performance["missing_reasons"]["inference_calls"] = "HTTP worker omitted performance telemetry"
        if effective is None:
            self.last_performance["missing_reasons"]["effective_parameters"] = "legacy HTTP worker did not acknowledge generation parameters"
        answer = _get(response, "answer", _get(response, "content", _get(response, "text")))
        if not isinstance(answer, str):
            raise LocateAnythingResponseError("LocateAnything HTTP response has no textual answer")
        return answer

    def _http_health(self) -> dict[str, Any]:
        """Read the worker health endpoint without sending an image or loading a model."""

        client = self.http_client
        try:
            if client is not None:
                get = getattr(client, "get", None)
                if not callable(get):
                    raise TypeError("http_client must expose get() for LocateAnything health checks")
                response = _coerce_http_response(get(self.health_url, timeout=self.timeout))
            else:
                health_request = urllib_request.Request(self.health_url, method="GET")
                with urllib_request.urlopen(health_request, timeout=self.timeout) as opened:
                    response = _coerce_http_response(opened)
        except Exception as exc:
            raise LocateAnythingHealthError(
                f"LocateAnything service health check failed at {self.health_url}: "
                f"{type(exc).__name__}:{exc}"
            ) from exc
        if not isinstance(response, Mapping):
            raise LocateAnythingHealthError("LocateAnything health response must be an object")
        status = str(response.get("status", "")).lower()
        if status not in {"ok", "ready", "healthy"}:
            raise LocateAnythingHealthError(
                f"LocateAnything service is not ready: status={response.get('status')!r}"
            )
        return dict(response)

    def health(self) -> dict[str, Any]:
        """Return service state for HTTP, or local model state for the local backend."""

        if self.backend == "http":
            return self._http_health()
        return {"status": "ok", "loaded": self.loaded, "backend": "local"}

    def _local_answer(self, prompt: str, image: Any, parameters: Mapping[str, Any]) -> str:
        started = time.perf_counter()
        try:
            model, processor, tokenizer = self._load_local()
        finally:
            self.last_performance["timings_seconds"]["model_load"] = time.perf_counter() - started
        started = time.perf_counter()
        converted = None
        inputs: Any = None
        generated: Any = None
        input_mapping: Any = None
        generate_kwargs: Any = None
        pixel_values: Any = None
        pixel_to: Any = None
        try:
            converted = _to_local_pil(image)
            original_size = converted.size
            side = parameters["maximum_image_side"]
            if side is not None and max(original_size) > side:
                scale = side / max(original_size)
                resized = converted.resize(tuple(max(1, round(value * scale)) for value in original_size))
                converted.close()
                converted = resized
            self._effective_parameters = {
                **parameters, "input_image_size": list(original_size),
                "model_image_size": list(converted.size), "temperature": 0.7,
                "top_p": 0.9, "do_sample": True, "repetition_penalty": 1.1, "use_cache": True,
            }
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": converted},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            apply_template = getattr(processor, "py_apply_chat_template", None)
            process_vision = getattr(processor, "process_vision_info", None)
            if not callable(apply_template) or not callable(process_vision):
                raise TypeError(
                    "LocateAnything processor must expose py_apply_chat_template() and "
                    "process_vision_info()"
                )
            text = apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos = process_vision(messages)
            inputs = processor(
                text=[text],
                images=images,
                videos=videos,
                return_tensors="pt",
            )
            input_mapping = dict(inputs) if isinstance(inputs, Mapping) else inputs
            if not isinstance(input_mapping, Mapping):
                raise TypeError("LocateAnything processor must return a tensor mapping")
            self.last_performance["timings_seconds"]["image_preprocess"] = time.perf_counter() - started
            ids = input_mapping.get("input_ids")
            self.last_performance["counts"]["input_tokens"] = len(ids[0]) if ids is not None else None
            ids = None
            _profile_sync(self.profile_timings)
            started = time.perf_counter()
            text_device = _component_device(model, "text")
            vision_device = _component_device(model, "vision") or text_device
            input_mapping = dict(input_mapping)
            for key in ("input_ids", "attention_mask"):
                if key in input_mapping:
                    input_mapping[key] = _move_value(input_mapping[key], text_device)
            for key in ("pixel_values", "image_grid_hws"):
                if key in input_mapping:
                    input_mapping[key] = _move_value(input_mapping[key], vision_device)
            generate_kwargs: dict[str, Any] = {
                "pixel_values": input_mapping["pixel_values"],
                "input_ids": input_mapping["input_ids"],
                "attention_mask": input_mapping["attention_mask"],
                "tokenizer": tokenizer,
                "generation_mode": parameters["generation_mode"],
                "max_new_tokens": parameters["max_new_tokens"],
                "use_cache": True,
                # Match the released LocateAnything inference defaults.  Its
                # custom hybrid/PBD decoder consumes these kwargs directly;
                # omitting them silently switches to a different greedy path.
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "verbose": False,
            }
            pixel_values = generate_kwargs["pixel_values"]
            pixel_to = getattr(pixel_values, "to", None)
            if callable(pixel_to):
                try:
                    torch = importlib.import_module("torch")
                    dtype = torch.bfloat16
                    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                        dtype = torch.float16
                    generate_kwargs["pixel_values"] = pixel_to(dtype)
                except (AttributeError, RuntimeError):
                    pass
            if input_mapping.get("image_grid_hws") is not None:
                generate_kwargs["image_grid_hws"] = input_mapping["image_grid_hws"]
            _profile_sync(self.profile_timings)
            self.last_performance["timings_seconds"]["input_transfer"] = time.perf_counter() - started
            try:
                torch = importlib.import_module("torch")
                inference_mode = getattr(torch, "inference_mode", None)
            except ImportError:
                inference_mode = None
            started = time.perf_counter()
            self.last_performance["counts"]["inference_calls"] += 1
            try:
                if callable(inference_mode):
                    with inference_mode():
                        generated = model.generate(**generate_kwargs)
                else:
                    generated = model.generate(**generate_kwargs)
                _profile_sync(self.profile_timings)
            finally:
                self.last_performance["timings_seconds"]["generation"] = time.perf_counter() - started
            answer = generated[0] if isinstance(generated, tuple) else generated
            if isinstance(answer, str):
                return answer
            if isinstance(answer, Sequence) and answer and isinstance(answer[0], str):
                return answer[0]
            raise LocateAnythingResponseError(
                "LocateAnything local generate() did not return its documented answer text"
            )
        finally:
            started = time.perf_counter()
            generated = None
            inputs = None
            input_mapping = None
            generate_kwargs = None
            pixel_values = None
            pixel_to = None
            close = getattr(converted, "close", None)
            if callable(close):
                close()
            self.last_performance["timings_seconds"]["request_cleanup"] = time.perf_counter() - started

    def locate(
        self,
        image: Any,
        prompt: str,
        *,
        image_size: tuple[int, int] | None = None,
        default_label: str = "object",
        max_new_tokens: int | None = None,
        maximum_image_side: Any = _UNSET,
        generation_mode: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        parameters = self._parameters(
            self.max_new_tokens if max_new_tokens is None else max_new_tokens,
            self.maximum_image_side if maximum_image_side is _UNSET else maximum_image_side,
            self.generation_mode if generation_mode is None else generation_mode,
        )
        self.last_performance = _performance_record(self.backend)
        self.last_performance["counts"]["image_file_read_bytes"] = _image_file_bytes([image])
        self._effective_parameters = None
        started = time.perf_counter()
        try:
            answer = (
                self._http_answer(prompt, image, parameters)
                if self.backend == "http"
                else self._local_answer(prompt, image, parameters)
            )
            parse_started = time.perf_counter()
            try:
                # Normalized coordinates always scale directly to the caller's
                # original dimensions, including downscaled model images.
                detections = parse_locate_anything_response(answer, image_size=image_size, default_label=default_label)
            finally:
                self.last_performance["timings_seconds"]["parse"] = time.perf_counter() - parse_started
            self.last_performance["counts"]["output_characters"] = len(answer)
            self.last_performance["counts"]["output_tokens"] = None
            self.last_performance["missing_reasons"]["output_tokens"] = "custom hybrid decoder returns answer text without token IDs"
            self.last_performance["missing_reasons"]["first_token"] = "custom hybrid decoder exposes no first-token callback"
        except Exception as exc:
            self.last_performance["error"] = f"{type(exc).__name__}:{exc}"[:500]
            if self.backend == "local" and _is_oom(exc):
                _empty_cuda_cache()
            raise
        finally:
            cleanup_started = time.perf_counter()
            if self.backend == "local" and self.release_cuda_cache:
                _empty_cuda_cache()
            self.last_performance["timings_seconds"]["cache_cleanup"] = time.perf_counter() - cleanup_started
            self.last_performance["timings_seconds"]["total"] = time.perf_counter() - started
        return {
            "detections": detections,
            "raw_answer": answer,
            "metadata": {
                "mode": "spatial_localization",
                "provides_track_ids": False,
                "requested_parameters": parameters,
                "effective_parameters": self._effective_parameters,
                "model_path": str(self.model_path) if self.model_path else None,
                "performance": self.last_performance,
            },
        }

    def detect(
        self,
        image: Any,
        labels: Sequence[str] | str,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        label_list = [labels] if isinstance(labels, str) else list(labels)
        if not label_list or any(not isinstance(label, str) or not label.strip() for label in label_list):
            raise ValueError("labels must contain at least one non-empty label")
        prompt = (
            "Locate all the instances that matches the following description: "
            + ", ".join(label.strip() for label in label_list)
            + "."
        )
        return self.locate(image, prompt, image_size=image_size, default_label=label_list[0])

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None
        _empty_cuda_cache()


__all__ = [
    "LocateAnythingAdapter",
    "LocateAnythingHealthError",
    "LocateAnythingResponseError",
    "parse_locate_anything_response",
]
