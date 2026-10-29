"""Stateless Qwen3.5 visual-review adapter.

Each call creates exactly one user message from the supplied evidence.  No chat
history, generated tokens, images, or KV cache are retained by this adapter
between calls.  Model weights remain loaded for the optional local backend.
"""

from __future__ import annotations

import base64
import gc
import importlib
import io
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence
from urllib import request as urllib_request


VALID_DECISIONS = frozenset({"confirmed", "rejected", "uncertain"})


def qwen35_9b_8gb_device_map(gpu_index: int = 0) -> dict[str, int | str]:
    """Offload the two huge BF16 token tables, keeping 4-bit blocks intact."""

    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise ValueError("gpu_index must be a non-negative integer")
    return {
        "model.visual": gpu_index,
        "model.language_model.embed_tokens": "cpu",
        "model.language_model.layers": gpu_index,
        "model.language_model.norm": gpu_index,
        "model.language_model.rotary_emb": gpu_index,
        "lm_head": "cpu",
    }


def qwen35_9b_cpu_embedding_gpu_head_device_map(gpu_index: int = 0) -> dict[str, int | str]:
    """Experimental placement; the CPU embedding also needs its hook replaced."""
    mapping = qwen35_9b_8gb_device_map(gpu_index)
    mapping["lm_head"] = gpu_index
    return mapping


def _performance_record(backend: str) -> dict[str, Any]:
    return {
        "schema_version": "model_performance.v1", "backend": backend,
        "timings_seconds": {}, "counts": {"inference_calls": 0},
        "missing_reasons": {}, "memory": {},
        "timing_scope": "host_wall_clock; CUDA boundaries synchronized only in profile_timings mode",
    }


def _profile_sync(enabled: bool) -> None:
    if enabled:
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, AttributeError, RuntimeError):
            pass


def _image_file_bytes(images: Sequence[Any]) -> int:
    """Logical source bytes read for local evidence files; excludes PIL buffers."""
    total = 0
    for image in images:
        if isinstance(image, (str, os.PathLike)) and not str(image).startswith("data:image/"):
            try:
                total += Path(image).expanduser().stat().st_size
            except OSError:
                pass  # Image loading supplies the actionable error.
    return total


def _install_cpu_embedding_lookup(model: Any, target_device: Any, counters: dict[str, int]) -> None:
    """Remove Accelerate table offloading and move only IDs and lookup results.

    Installed only for the explicit experimental strategy.  Restoring the
    CPU table before removing the hook handles Accelerate's meta placeholder.
    """
    torch = importlib.import_module("torch")
    embedding = model.get_input_embeddings()
    if getattr(embedding, "_classroom_cpu_lookup", False):
        return
    output = model.get_output_embeddings()
    if output is not None and output.weight is embedding.weight:
        raise ValueError("cpu_embedding_gpu_head requires untied embedding and output weights")
    hook = getattr(embedding, "_hf_hook", None)
    cpu_weight = None
    if str(embedding.weight.device) == "meta":
        def find_weight(candidate: Any) -> Any:
            weights = getattr(candidate, "weights_map", None)
            if weights is not None:
                try:
                    return weights["weight"]
                except KeyError:
                    pass
            for child in getattr(candidate, "hooks", ()):
                value = find_weight(child)
                if value is not None:
                    return value
            return None
        cpu_weight = find_weight(hook)
        if cpu_weight is None:
            raise RuntimeError("CPU embedding weight unavailable in Accelerate offload hook")
    if hook is not None:
        importlib.import_module("accelerate.hooks").remove_hook_from_module(embedding)
    if cpu_weight is not None:
        embedding.weight = torch.nn.Parameter(cpu_weight.detach().to("cpu"), requires_grad=False)
    else:
        embedding.to("cpu")

    def before(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
        return tuple(value.to("cpu") if torch.is_tensor(value) else value for value in args), {
            key: value.to("cpu") if torch.is_tensor(value) else value for key, value in kwargs.items()
        }

    def after(_module: Any, _args: Any, result: Any) -> Any:
        counters["embedding_lookup_calls"] = counters.get("embedding_lookup_calls", 0) + 1
        if str(result.device) != str(torch.device(target_device)):
            counters["embedding_activation_transfer_bytes"] = counters.get("embedding_activation_transfer_bytes", 0) + result.numel() * result.element_size()
        return result.to(target_device)

    embedding.register_forward_pre_hook(before, with_kwargs=True)
    embedding.register_forward_hook(after)
    embedding._classroom_cpu_lookup = True


class _FirstTokenTimer:
    """Generation streamer observer; never retains token tensors or text."""
    def __init__(self, started: float, synchronize: bool) -> None:
        self.started = started
        self.synchronize = synchronize
        self.prompt_seen = False
        self.first_token_seconds: float | None = None

    def put(self, _value: Any) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first_token_seconds is None:
            _profile_sync(self.synchronize)
            self.first_token_seconds = time.perf_counter() - self.started

    def end(self) -> None:
        pass


class QwenResponseError(ValueError):
    """Raised when a model response violates the strict JSON contract."""


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _is_oom(error: BaseException) -> bool:
    name = type(error).__name__.lower()
    message = str(error).lower()
    return "outofmemory" in name or "out of memory" in message or "cuda oom" in message


def _empty_cuda_cache() -> None:
    gc.collect()
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            # Synchronize before sampling physical usage.  Without this, the
            # allocator can report a stable reserved pool while deferred
            # kernels and IPC blocks remain visible to nvidia-smi.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except (ImportError, AttributeError, RuntimeError):
        pass


def _strict_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise QwenResponseError("Qwen response content must be text")

    def reject_constant(value: str) -> Any:
        raise QwenResponseError(f"non-standard JSON numeric constant is forbidden: {value}")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QwenResponseError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        # Deliberately do not strip Markdown fences or search for an embedded
        # object: accepting a partial response would hide schema failures.
        # Keep a short escaped excerpt in audit receipts.  We still reject the
        # response completely; this only makes malformed local generation
        # diagnosable without retaining whole prompts or images.
        excerpt = text.replace("\r", " ").replace("\n", " ").strip()[:400]
        raise QwenResponseError(
            f"Qwen response is not one strict JSON document: {excerpt!r}"
        ) from exc
    if not isinstance(value, dict):
        raise QwenResponseError("Qwen response JSON must be an object")
    return value


def _mime_for_path(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed if guessed and guessed.startswith("image/") else "image/jpeg"


def _image_data_url(image: Any) -> str:
    """Encode a local image without introducing a module-level PIL dependency."""

    if isinstance(image, str) and image.startswith("data:image/"):
        return image
    if isinstance(image, (str, os.PathLike)):
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"evidence image does not exist: {path}")
        payload = path.read_bytes()
        mime = _mime_for_path(path)
    elif isinstance(image, (bytes, bytearray, memoryview)):
        payload = bytes(image)
        mime = "image/jpeg"
    elif hasattr(image, "save"):
        buffer = io.BytesIO()
        converted = image.convert("RGB") if hasattr(image, "convert") else image
        try:
            converted.save(buffer, format="JPEG")
            payload = buffer.getvalue()
            mime = "image/jpeg"
        finally:
            if converted is not image:
                close = getattr(converted, "close", None)
                if callable(close):
                    close()
            buffer.close()
    else:
        # ndarray support is optional and stays lazy.
        try:
            pil_image = importlib.import_module("PIL.Image")
            converted = pil_image.fromarray(image)
        except (ImportError, AttributeError, TypeError) as exc:
            raise TypeError("image must be a local path, bytes, PIL image, or ndarray") from exc
        try:
            return _image_data_url(converted)
        finally:
            converted.close()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _to_local_pil(image: Any) -> Any:
    pil_image = importlib.import_module("PIL.Image")
    if hasattr(image, "copy") and hasattr(image, "save"):
        return image.copy().convert("RGB")
    if isinstance(image, str) and image.startswith("data:image/"):
        try:
            encoded = image.split(",", 1)[1]
            return pil_image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except (IndexError, ValueError) as exc:
            raise ValueError("invalid image data URL") from exc
    if isinstance(image, (str, os.PathLike)):
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"evidence image does not exist: {path}")
        with pil_image.open(path) as opened:
            return opened.convert("RGB").copy()
    if isinstance(image, (bytes, bytearray, memoryview)):
        with pil_image.open(io.BytesIO(bytes(image))) as opened:
            return opened.convert("RGB").copy()
    return pil_image.fromarray(image).convert("RGB")


def _extract_content(response: Any) -> str:
    choices = _get(response, "choices")
    if choices:
        message = _get(choices[0], "message")
        content = _get(message, "content")
    else:
        content = _get(response, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts = []
        for item in content:
            text = _get(item, "text")
            if text is not None:
                parts.append(str(text))
        if parts:
            return "".join(parts)
    raise QwenResponseError("OpenAI-compatible response has no textual message content")


def _coerce_http_response(response: Any) -> Any:
    raise_for_status = getattr(response, "raise_for_status", None)
    if callable(raise_for_status):
        raise_for_status()
    json_method = getattr(response, "json", None)
    if callable(json_method):
        return json_method()
    if hasattr(response, "read"):
        return json.loads(response.read().decode("utf-8"))
    return response


class Qwen35Adapter:
    """Qwen3.5 HTTP or local bitsandbytes-NF4 adapter.

    Parameters
    ----------
    model:
        Required model identifier for HTTP, or an existing local model directory
        for ``backend="local"``.  There is intentionally no implicit download or
        hard-coded fallback model.
    """

    def __init__(
        self,
        model: os.PathLike[str] | str | None = None,
        *,
        model_path: os.PathLike[str] | str | None = None,
        backend: str = "http",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_new_tokens: int = 128,
        max_oom_retries: int = 2,
        http_client: Any = None,
        local_loader: Callable[[Path], tuple[Any, Any]] | None = None,
        trust_remote_code: bool = False,
        release_cuda_cache: bool = False,
        device_map: str | Mapping[Any, Any] = "auto",
        max_memory: Mapping[Any, str | int] | None = None,
        offload_folder: os.PathLike[str] | str | None = None,
        offload_state_dict: bool = True,
        enable_thinking: bool = False,
        device_strategy: str = "legacy_offload",
        profile_timings: bool = False,
        memory_fraction: float | None = None,
    ) -> None:
        backend = backend.strip().lower()
        if backend not in {"http", "local"}:
            raise ValueError("backend must be 'http' or 'local'")
        if model is not None and model_path is not None:
            raise ValueError("supply model or model_path, not both")
        configured_model = model_path if model_path is not None else model
        if configured_model is None or not str(configured_model).strip():
            raise ValueError("model must be supplied by configuration")
        if isinstance(max_new_tokens, bool) or not 1 <= int(max_new_tokens) <= 128:
            raise ValueError("max_new_tokens must be between 1 and 128")
        if max_oom_retries < 0:
            raise ValueError("max_oom_retries must be non-negative")

        self.backend = backend
        self.model = str(configured_model)
        self.model_path: Path | None = None
        if backend == "local":
            self.model_path = Path(configured_model).expanduser().resolve()
            if not self.model_path.is_dir():
                raise FileNotFoundError(
                    f"local Qwen model must be an existing directory: {self.model_path}"
                )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_new_tokens = int(max_new_tokens)
        self.max_oom_retries = int(max_oom_retries)
        self.http_client = http_client
        self.local_loader = local_loader
        self.trust_remote_code = bool(trust_remote_code)
        self.release_cuda_cache = bool(release_cuda_cache)
        self.device_map = device_map
        self.max_memory = dict(max_memory) if max_memory is not None else None
        self.offload_folder = (
            Path(offload_folder).expanduser().resolve() if offload_folder is not None else None
        )
        self.offload_state_dict = bool(offload_state_dict)
        self.enable_thinking = bool(enable_thinking)
        if device_strategy not in {"legacy_offload", "cpu_embedding_gpu_head"}:
            raise ValueError("unsupported Qwen device_strategy")
        if device_strategy == "cpu_embedding_gpu_head" and backend != "local":
            raise ValueError("cpu_embedding_gpu_head requires the local backend")
        self.device_strategy = device_strategy
        self.profile_timings = bool(profile_timings)
        if memory_fraction is not None:
            if (
                isinstance(memory_fraction, bool)
                or not isinstance(memory_fraction, (int, float))
                or not 0.0 < float(memory_fraction) <= 1.0
            ):
                raise ValueError("memory_fraction must be a number in (0, 1]")
            memory_fraction = float(memory_fraction)
        self.memory_fraction = memory_fraction
        self._memory_fraction_applied = False
        self.last_performance = _performance_record(backend)
        self._embedding_counters: dict[str, int] = {}
        self._kernel_compatibility: dict[str, Any] = {}
        self._local_model: Any = None
        self._processor: Any = None
        self._load_failure: Exception | None = None

    @property
    def loaded(self) -> bool:
        return self._local_model is not None

    def _chat_url(self) -> str:
        suffix = "/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions"
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + suffix

    def _load_local(self) -> tuple[Any, Any]:
        if self._local_model is not None and self._processor is not None:
            self.last_performance["timings_seconds"]["model_load"] = 0.0
            return self._local_model, self._processor
        if self._load_failure is not None:
            raise RuntimeError("qwen_local_load_previously_failed") from self._load_failure
        assert self.model_path is not None
        load_started = time.perf_counter()
        try:
            torch = importlib.import_module("torch")
            self._apply_memory_fraction(torch)
            if self.local_loader is not None:
                self._local_model, self._processor = self.local_loader(self.model_path)
                self._configure_device_strategy(self._local_model)
                return self._local_model, self._processor

            # Fail before importing sentencepiece/transformers: a directory
            # without a model config would otherwise be probed by the tokenizer
            # and processor loaders, which takes tens of seconds to reject.
            if not (self.model_path / "config.json").is_file():
                raise FileNotFoundError(
                    f"qwen_local_model_config_missing:{self.model_path}"
                )

            transformers = importlib.import_module("transformers")
            compute_dtype = torch.float16
            try:
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    compute_dtype = torch.bfloat16
            except (AttributeError, RuntimeError):
                pass
            quantization = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_enable_fp32_cpu_offload=True,
        )
            common = {
            "local_files_only": True,
            "trust_remote_code": self.trust_remote_code,
        }
            processor = transformers.AutoProcessor.from_pretrained(str(self.model_path), **common)
            model_class = (
            getattr(transformers, "AutoModelForImageTextToText", None)
            or getattr(transformers, "AutoModelForVision2Seq", None)
            or getattr(transformers, "AutoModelForCausalLM")
        )
            load_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "dtype": compute_dtype,
            "low_cpu_mem_usage": True,
            "offload_state_dict": self.offload_state_dict,
            **common,
        }
        # Pre-quantized Hub snapshots carry their own quantization_config.
        # Transformers ignores most of a second BitsAndBytesConfig in that
        # case, including CPU-offload permission. Patch the in-memory config;
        # the pinned local config.json remains untouched.
            config_file = self.model_path / "config.json"
            embedded_quantization = False
            if config_file.is_file():
                try:
                    raw_config = json.loads(config_file.read_text(encoding="utf-8"))
                    embedded_quantization = isinstance(raw_config.get("quantization_config"), Mapping)
                except (OSError, json.JSONDecodeError):
                    embedded_quantization = False
            if embedded_quantization:
                model_config = transformers.AutoConfig.from_pretrained(str(self.model_path), **common)
                quantization_dict = dict(getattr(model_config, "quantization_config"))
                quantization_dict["llm_int8_enable_fp32_cpu_offload"] = True
                quantization_dict["bnb_4bit_compute_dtype"] = (
                    "bfloat16" if compute_dtype == torch.bfloat16 else "float16"
                )
                model_config.quantization_config = quantization_dict
                load_kwargs["config"] = model_config
            else:
                load_kwargs["quantization_config"] = quantization
            if self.max_memory is not None:
                load_kwargs["max_memory"] = self.max_memory
            if self.offload_folder is not None:
                self.offload_folder.mkdir(parents=True, exist_ok=True)
                load_kwargs["offload_folder"] = str(self.offload_folder)
            model = model_class.from_pretrained(
                str(self.model_path),
                **load_kwargs,
            )
            eval_method = getattr(model, "eval", None)
            if callable(eval_method):
                eval_method()
            self._configure_device_strategy(model)
            self._local_model, self._processor = model, processor
            # Inspect loaded implementation flags without importing optional
            # native kernels or probing/installing anything in the environment.
            module = importlib.import_module(type(model).__module__)
            self._kernel_compatibility = {
                "model_class": type(model).__name__,
                "model_module": type(model).__module__,
                "is_fast_path_available": getattr(module, "is_fast_path_available", None),
                "causal_conv1d_fn_available": callable(getattr(module, "causal_conv1d_fn", None)),
                "chunk_gated_delta_rule_available": callable(getattr(module, "chunk_gated_delta_rule", None)),
            }
            return model, processor
        except Exception as exc:
            self._local_model = None
            self._processor = None
            self._load_failure = exc
            _empty_cuda_cache()
            raise
        finally:
            self.last_performance["timings_seconds"]["model_load"] = time.perf_counter() - load_started

    def _apply_memory_fraction(self, torch: Any) -> None:
        """Cap this process's caching allocator before any model allocation.

        Windows WDDM silently pages allocations above dedicated VRAM into
        shared system memory, which turns an 8 GiB card into an unpredictably
        slow device.  A hard cap makes the allocator reuse its cache or fail
        loudly instead of spilling.
        """

        if self.memory_fraction is None or self._memory_fraction_applied:
            return
        setter = getattr(torch.cuda, "set_per_process_memory_fraction", None)
        if setter is None or not torch.cuda.is_available():
            raise RuntimeError("qwen_memory_fraction_requires_cuda")
        try:
            setter(float(self.memory_fraction))
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError(
                f"qwen_memory_fraction_unavailable:{type(exc).__name__}:{exc}"
            ) from exc
        self._memory_fraction_applied = True

    def _configure_device_strategy(self, model: Any) -> None:
        if self.device_strategy == "cpu_embedding_gpu_head":
            head = model.get_output_embeddings()
            target = getattr(getattr(head, "weight", None), "device", None)
            if target is None or str(target) in {"meta", "cpu"}:
                raise ValueError("cpu_embedding_gpu_head requires an output head resident on CUDA")
            _install_cpu_embedding_lookup(model, target, self._embedding_counters)

    def warmup(self) -> None:
        """Load once before a batch so an OOM cannot retry per candidate."""
        if self.backend == "local":
            self.last_performance = _performance_record(self.backend)
            self.last_performance["operation"] = "warmup"
            try:
                self._load_local()
            finally:
                self.last_performance["kernel_compatibility"] = dict(self._kernel_compatibility)

    def _call_http_client(self, payload: dict[str, Any]) -> Any:
        client = self.http_client
        if client is not None:
            completions = _get(_get(client, "chat"), "completions")
            create = _get(completions, "create")
            if callable(create):
                sdk_payload = dict(payload)
                template_kwargs = sdk_payload.pop("chat_template_kwargs", None)
                if template_kwargs is not None:
                    sdk_payload["extra_body"] = {
                        "chat_template_kwargs": template_kwargs,
                    }
                return create(**sdk_payload)
            post = getattr(client, "post", None)
            if callable(post):
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                return _coerce_http_response(
                    post(self._chat_url(), json=payload, headers=headers, timeout=self.timeout)
                )
            if callable(client):
                return client(payload)
            raise TypeError("http_client must be callable, requests-like, or OpenAI-compatible")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib_request.Request(
            self._chat_url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=self.timeout) as response:
            return _coerce_http_response(response)

    def _invoke_http(self, prompt: str, images: Sequence[Any], token_limit: int) -> str:
        started = time.perf_counter()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        try:
            for image in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}}
                )
            # One request, one current-evidence message, no history/system turns.
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": token_limit,
                "temperature": 0,
                "do_sample": False,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            }
            self.last_performance["timings_seconds"]["image_preprocess"] = time.perf_counter() - started
            self.last_performance["counts"]["request_bytes"] = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            started = time.perf_counter()
            self.last_performance["counts"]["inference_calls"] += 1
            try:
                response = self._call_http_client(payload)
            finally:
                self.last_performance["timings_seconds"]["http_request"] = time.perf_counter() - started
            usage = _get(response, "usage", {})
            for field, key in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
                value = _get(usage, field)
                self.last_performance["counts"][key] = value
                if value is None:
                    self.last_performance["missing_reasons"][key] = "HTTP endpoint omitted token usage"
            self.last_performance["missing_reasons"]["first_token"] = "non-streaming HTTP endpoint"
            return _extract_content(response)
        finally:
            content.clear()

    def _move_inputs(self, inputs: Any, model: Any) -> Any:
        device = getattr(model, "device", None)
        if self.device_strategy == "cpu_embedding_gpu_head":
            device = model.get_output_embeddings().weight.device
        move = getattr(inputs, "to", None)
        if callable(move) and device is not None:
            return move(device)
        if isinstance(inputs, MutableMapping) and device is not None:
            for key, value in list(inputs.items()):
                value_move = getattr(value, "to", None)
                if callable(value_move):
                    inputs[key] = value_move(device)
        return inputs

    def _invoke_local(self, prompt: str, images: Sequence[Any], token_limit: int) -> str:
        model, processor = self._load_local()
        preprocess_started = time.perf_counter()
        opened_images: list[Any] = []
        content: list[dict[str, Any]] = []
        inputs: Any = None
        generated: Any = None
        input_mapping: Any = None
        first: Any = None
        output_tokens: Any = None
        try:
            for image in images:
                converted = _to_local_pil(image)
                opened_images.append(converted)
                content.append({"type": "image", "image": converted})
            content.append({"type": "text", "text": prompt})
            # Seed the current assistant turn with the first JSON character.
            # Qwen3.5 otherwise sometimes wraps an otherwise valid object in
            # ```json fences even under greedy decoding.  This is not prior
            # conversation history: every invocation still contains only the
            # current user evidence plus this fixed one-character output
            # prefix, and the prefix is included in the strict parser input.
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "{"},
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=self.enable_thinking,
            )
            self.last_performance["timings_seconds"]["image_preprocess"] = time.perf_counter() - preprocess_started
            _profile_sync(self.profile_timings)
            transfer_started = time.perf_counter()
            inputs = self._move_inputs(inputs, model)
            _profile_sync(self.profile_timings)
            self.last_performance["timings_seconds"]["input_transfer"] = time.perf_counter() - transfer_started
            input_mapping = dict(inputs) if isinstance(inputs, Mapping) else inputs
            if not isinstance(input_mapping, Mapping):
                raise TypeError("processor.apply_chat_template must return a mapping")
            torch = importlib.import_module("torch")
            inference_mode = getattr(torch, "inference_mode", None)
            input_ids = input_mapping.get("input_ids")
            prefix_length = len(input_ids[0]) if input_ids is not None else 0
            self.last_performance["counts"]["input_tokens"] = prefix_length
            _profile_sync(self.profile_timings)
            generation_started = time.perf_counter()
            timer = _FirstTokenTimer(generation_started, self.profile_timings)
            generation_options: dict[str, Any] = {"max_new_tokens": token_limit, "do_sample": False}
            if self.profile_timings:
                generation_options["streamer"] = timer
            self.last_performance["counts"]["inference_calls"] += 1
            try:
                if callable(inference_mode):
                    with inference_mode():
                        generated = model.generate(
                            **input_mapping,
                            **generation_options,
                        )
                else:  # pragma: no cover - compatibility with minimal test doubles
                    generated = model.generate(
                        **input_mapping,
                        **generation_options,
                    )
                _profile_sync(self.profile_timings)
            finally:
                elapsed = time.perf_counter() - generation_started
                self.last_performance["timings_seconds"].update({
                    "generation": elapsed, "first_token": timer.first_token_seconds,
                    "decode": None if timer.first_token_seconds is None else max(0.0, elapsed - timer.first_token_seconds),
                })
            if timer.first_token_seconds is None:
                self.last_performance["missing_reasons"]["first_token"] = "profile_timings disabled or generate did not call streamer"
            first = generated[0]
            output_tokens = first[prefix_length:]
            self.last_performance["counts"]["output_tokens"] = len(output_tokens)
            # Reconstruct the complete assistant turn exactly as generated:
            # the opening brace belongs to the prompt prefill, all remaining
            # characters belong to the model output.
            decode_started = time.perf_counter()
            answer = "{" + processor.decode(output_tokens, skip_special_tokens=True)
            self.last_performance["timings_seconds"]["token_decode"] = time.perf_counter() - decode_started
            return answer
        finally:
            cleanup_started = time.perf_counter()
            generated = None
            output_tokens = None
            first = None
            input_mapping = None
            inputs = None
            content.clear()
            for image in opened_images:
                close = getattr(image, "close", None)
                if callable(close):
                    close()
            opened_images.clear()
            self.last_performance["timings_seconds"]["request_cleanup"] = time.perf_counter() - cleanup_started

    @staticmethod
    def _reduced_images(images: Sequence[Any]) -> list[Any]:
        count = len(images)
        if count <= 1:
            return list(images)
        target = max(1, (count + 1) // 2)
        if target == 1:
            return [images[count // 2]]
        indices = [round(index * (count - 1) / (target - 1)) for index in range(target)]
        return [images[index] for index in indices]

    def complete(
        self,
        prompt: str,
        *,
        images: Sequence[Any] = (),
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return one strict JSON object from one stateless request."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        requested = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        token_limit = min(128, requested)
        if token_limit < 1:
            raise ValueError("max_new_tokens must be positive")
        current_images = list(images)
        attempts = 0
        started = time.perf_counter()
        self.last_performance = _performance_record(self.backend)
        self.last_performance.update({"device_strategy": self.device_strategy, "profile_timings": self.profile_timings})
        self.last_performance["effective_parameters"] = {"max_new_tokens": token_limit, "do_sample": False, "enable_thinking": self.enable_thinking}
        self.last_performance["counts"]["images"] = len(current_images)
        self.last_performance["counts"]["image_file_read_bytes"] = _image_file_bytes(current_images)
        self._embedding_counters.clear()
        try:
            while True:
                try:
                    if self.backend == "http":
                        text = self._invoke_http(prompt, current_images, token_limit)
                    else:
                        text = self._invoke_local(prompt, current_images, token_limit)
                    parse_started = time.perf_counter()
                    try:
                        result = _strict_json_object(text)
                        self.last_performance.pop("error", None)
                        return result
                    finally:
                        self.last_performance["timings_seconds"]["parse"] = time.perf_counter() - parse_started
                except Exception as exc:
                    self.last_performance["error"] = f"{type(exc).__name__}:{exc}"[:500]
                    if (
                        not _is_oom(exc)
                        or attempts >= self.max_oom_retries
                        or len(current_images) <= 1
                    ):
                        raise
                    attempts += 1
                    reduced = self._reduced_images(current_images)
                    current_images.clear()
                    current_images = reduced
                    _empty_cuda_cache()
        finally:
            cleanup_started = time.perf_counter()
            current_images.clear()
            if self.backend == "local" and self.release_cuda_cache:
                _empty_cuda_cache()
            self.last_performance["timings_seconds"]["cache_cleanup"] = time.perf_counter() - cleanup_started
            self.last_performance["timings_seconds"]["total"] = time.perf_counter() - started
            self.last_performance["counts"].update(self._embedding_counters)
            self.last_performance["counts"]["oom_retries"] = attempts
            self.last_performance["kernel_compatibility"] = dict(self._kernel_compatibility)

    def review(
        self,
        candidate_event_id: str,
        evidence: Any,
        *,
        event_type: str | None = None,
        evidence_frame_indices: Sequence[int] = (),
        prompt: str | None = None,
        compact_response: bool = False,
    ) -> dict[str, Any]:
        """Review one candidate and validate the pipeline-facing decision schema.

        Classification can use a compact transport response because a full
        64-character vocabulary hash plus a long candidate id cannot fit
        reliably inside the 128-token generation cap.  The caller supplies
        those request-bound fields after parsing; the final pipeline artifact
        still contains the complete strict schema.
        """

        if not isinstance(candidate_event_id, str) or not candidate_event_id.strip():
            raise ValueError("candidate_event_id must be non-empty")
        images: list[Any]
        evidence_metadata: Any = None
        if isinstance(evidence, Mapping):
            raw_images = evidence.get("images", evidence.get("frames", ()))
            if (
                isinstance(raw_images, (str, bytes, os.PathLike))
                or hasattr(raw_images, "save")
                or hasattr(raw_images, "shape")
            ):
                images = [raw_images]
            else:
                images = list(raw_images or ())
            evidence_metadata = {
                key: value for key, value in evidence.items() if key not in {"images", "frames"}
            }
        elif (
            isinstance(evidence, (str, bytes, os.PathLike))
            or hasattr(evidence, "save")
            or hasattr(evidence, "shape")
        ):
            images = [evidence]
        else:
            images = list(evidence or ())

        frame_indices = [int(index) for index in evidence_frame_indices]
        instruction = prompt or (
            "Review only the supplied current-event visual evidence. Return exactly one JSON "
            "object and no Markdown. Required keys: candidate_event_id, decision, "
            "reviewed_event_type, behavior, rationale, evidence_frame_indices, "
            "self_reported_confidence. decision must be confirmed, rejected, or uncertain."
        )
        context = {
            "candidate_event_id": candidate_event_id,
            "candidate_event_type": event_type,
            "evidence_frame_indices": frame_indices,
            "evidence_metadata": evidence_metadata,
        }
        full_prompt = instruction + "\nCurrent evidence metadata:\n" + json.dumps(
            context, ensure_ascii=False, separators=(",", ":"), default=str
        )
        try:
            # ``complete`` copies the sequence for its retry loop.  Clear the
            # caller-side list as soon as the request ends, including schema
            # and transport failures, so one review can never retain image
            # references for a later request.
            result = self.complete(full_prompt, images=images)
        finally:
            images.clear()

        if compact_response:
            # Local generation occasionally emits a harmless leading space in
            # a JSON member name (for example ``" rationale"``).  The whole
            # response must still be one JSON object, and semantic keys remain
            # closed-world; normalize surrounding whitespace once so that a
            # formatting artifact cannot turn an otherwise valid response
            # into a false smoke-gate failure.  Reject collisions after
            # normalization, so this never hides duplicate or spoofed fields.
            normalized_result: dict[str, Any] = {}
            for raw_key, value in result.items():
                key = str(raw_key).strip()
                if not key:
                    raise QwenResponseError("Qwen compact response has an empty key")
                if key in normalized_result:
                    raise QwenResponseError(
                        f"Qwen compact response has duplicate keys after whitespace normalization: {key}"
                    )
                normalized_result[key] = value
            result = normalized_result
            compact_required = {
                "decision", "behavior", "rationale", "evidence_frame_indices", "observability",
            }
            compact_allowed = compact_required | {
                "candidate_event_id", "reviewed_event_type", "self_reported_confidence", "vocabulary_sha256",
            }
            unknown = set(result).difference(compact_allowed)
            if unknown:
                raise QwenResponseError(f"Qwen compact response has unexpected keys: {sorted(unknown)}")
            missing_compact = compact_required.difference(result)
            if missing_compact:
                raise QwenResponseError(
                    f"Qwen compact response is missing required keys: {sorted(missing_compact)}"
                )
            result.setdefault("candidate_event_id", candidate_event_id)
            result.setdefault("reviewed_event_type", event_type or "observable_action")
            result.setdefault("self_reported_confidence", None)
        if result.get("candidate_event_id") != candidate_event_id:
            raise QwenResponseError("Qwen candidate_event_id does not match the current request")
        decision = result.get("decision")
        if decision not in VALID_DECISIONS:
            raise QwenResponseError(f"decision must be one of {sorted(VALID_DECISIONS)}")
        required_keys = {
            "candidate_event_id",
            "decision",
            "reviewed_event_type",
            "behavior",
            "rationale",
            "evidence_frame_indices",
            "self_reported_confidence",
        }
        missing = required_keys.difference(result)
        if missing:
            raise QwenResponseError(f"Qwen response is missing required keys: {sorted(missing)}")
        if "evidence_frame_indices" in result:
            indices = result["evidence_frame_indices"]
            if (
                not isinstance(indices, list)
                or any(isinstance(value, bool) or not isinstance(value, int) for value in indices)
            ):
                raise QwenResponseError("evidence_frame_indices must be a JSON integer array")
        if "rationale" in result and not isinstance(result["rationale"], str):
            raise QwenResponseError("rationale must be text")
        for field_name in ("reviewed_event_type", "behavior"):
            value = result.get(field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise QwenResponseError(f"{field_name} must be non-empty text or null")
        result.setdefault("reviewer", "qwen3.5")
        result["schema_valid"] = True
        return result

    def close(self) -> None:
        started = time.perf_counter()
        self._local_model = None
        self._processor = None
        self._load_failure = None
        _empty_cuda_cache()
        self.last_performance["timings_seconds"]["teardown"] = time.perf_counter() - started


__all__ = [
    "Qwen35Adapter",
    "QwenResponseError",
    "VALID_DECISIONS",
    "qwen35_9b_8gb_device_map",
    "qwen35_9b_cpu_embedding_gpu_head_device_map",
]
