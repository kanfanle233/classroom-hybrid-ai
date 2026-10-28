"""Production parameter and telemetry contracts without loading GPU models."""
from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from classroom_hybrid_ai.adapters import LocateAnythingAdapter, Qwen35Adapter
from classroom_hybrid_ai.adapters.qwen import (
    _install_cpu_embedding_lookup, qwen35_9b_8gb_device_map,
    qwen35_9b_cpu_embedding_gpu_head_device_map,
)
from classroom_hybrid_ai.qwen_worker import _cuda_memory_metrics
from classroom_hybrid_ai.resources.locate_anything_service import LocateWorker


def _local_locate(tmp_path, *, fail=False, **kwargs):
    seen = {}

    class Processor:
        def py_apply_chat_template(self, messages, **_kwargs):
            seen["image_size"] = messages[0]["content"][0]["image"].size
            return "prompt"

        def process_vision_info(self, messages):
            return [messages[0]["content"][0]["image"]], None

        def __call__(self, **_kwargs):
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]], "pixel_values": []}

    class Model:
        device = None

        def generate(self, **generation):
            seen["generation"] = generation
            if fail:
                raise RuntimeError("CUDA out of memory")
            return "<ref>student</ref><box>100,200,900,800</box>"

    adapter = LocateAnythingAdapter(backend="local", model_path=tmp_path,
                                   local_loader=lambda _: (Model(), Processor()), **kwargs)
    return adapter, seen


def test_locate_default_generation_and_cleanup_match_compatible_path(tmp_path, monkeypatch):
    cleanups = []
    monkeypatch.setattr("classroom_hybrid_ai.adapters.locate_anything._empty_cuda_cache", lambda: cleanups.append(1))
    adapter, seen = _local_locate(tmp_path)
    with Image.new("RGB", (640, 360)) as image:
        result = adapter.locate(image, "student", image_size=image.size)
    assert seen["image_size"] == (640, 360)
    assert seen["generation"]["max_new_tokens"] == 256
    assert seen["generation"]["generation_mode"] == "hybrid"
    assert seen["generation"]["do_sample"] is True
    assert cleanups == []
    assert result["metadata"]["effective_parameters"]["maximum_image_side"] is None
    assert adapter.last_performance["counts"]["inference_calls"] == 1
    assert adapter.last_performance["counts"]["input_tokens"] == 3
    adapter.close()
    assert cleanups == [1]


def test_locate_resize_retains_original_coordinate_space_and_explicit_override(tmp_path):
    adapter, seen = _local_locate(tmp_path, maximum_image_side=320, max_new_tokens=72)
    with Image.new("RGB", (640, 360)) as image:
        result = adapter.locate(image, "student", image_size=image.size)
        assert seen["image_size"] == (320, 180)
        assert seen["generation"]["max_new_tokens"] == 72
        assert result["detections"][0]["bbox"] == [64., 72., 576., 288.]
        adapter.locate(image, "student", image_size=image.size, maximum_image_side=None)
        assert seen["image_size"] == (640, 360)


def test_locate_oom_keeps_diagnostics_and_releases_cache(tmp_path, monkeypatch):
    cleanups = []
    monkeypatch.setattr("classroom_hybrid_ai.adapters.locate_anything._empty_cuda_cache", lambda: cleanups.append(1))
    adapter, _seen = _local_locate(tmp_path, fail=True)
    with Image.new("RGB", (10, 10)) as image, pytest.raises(RuntimeError, match="out of memory"):
        adapter.locate(image, "student")
    assert cleanups == [1]
    assert adapter.last_performance["counts"]["inference_calls"] == 1
    assert adapter.last_performance["timings_seconds"]["generation"] >= 0
    assert "out of memory" in adapter.last_performance["error"]


@pytest.mark.parametrize("kwargs", [{"max_new_tokens": 0}, {"max_new_tokens": True},
                                   {"maximum_image_side": -1}, {"generation_mode": ""}])
def test_locate_parameters_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        LocateAnythingAdapter(**kwargs)


def test_locate_http_worker_propagates_parameters_and_performance(tmp_path, monkeypatch):
    from classroom_hybrid_ai import qwen_worker
    monkeypatch.setattr(qwen_worker, "_reset_cuda_peaks", lambda: None)
    monkeypatch.setattr(qwen_worker, "_cuda_memory_metrics", lambda: {})
    worker = LocateWorker.__new__(LocateWorker)
    worker.lock = threading.Lock()
    worker.peak_memory = {}
    worker.adapter, seen = _local_locate(tmp_path)
    adapter = LocateAnythingAdapter(http_client=worker.locate, max_new_tokens=88, maximum_image_side=20,
                                   generation_mode="hybrid")
    with Image.new("RGB", (40, 30)) as image:
        result = adapter.locate(image, "student", image_size=image.size)
    assert seen["image_size"] == (20, 15)
    assert seen["generation"]["max_new_tokens"] == 88
    assert result["detections"][0]["bbox"] == [4., 6., 36., 24.]
    assert result["metadata"]["effective_parameters"]["max_new_tokens"] == 88
    assert adapter.last_performance["counts"]["inference_calls"] == 1
    assert "worker_performance" in adapter.last_performance


def test_legacy_http_worker_does_not_claim_effective_parameters():
    adapter = LocateAnythingAdapter(http_client=lambda _: {"answer": "<box>0,0,10,10</box>"})
    result = adapter.locate(b"image", "student")
    assert result["metadata"]["effective_parameters"] is None
    assert "effective_parameters" in adapter.last_performance["missing_reasons"]


def test_qwen_peak_measurements_are_distinct_from_after_request(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, memory_allocated=lambda: 5,
                           memory_reserved=lambda: 10, max_memory_allocated=lambda: 25,
                           max_memory_reserved=lambda: 40, memory_stats=lambda: {})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    metrics = _cuda_memory_metrics()
    assert metrics["allocated_bytes"] == 5
    assert metrics["max_allocated_bytes"] == 25
    assert metrics["reserved_bytes"] == 10
    assert metrics["request_peak_reserved_bytes"] == 40
    assert "WDDM" in metrics["scope"]


def test_cpu_embedding_removes_accelerate_table_reload_and_preserves_values():
    import torch
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.head = torch.nn.Linear(4, 16, bias=False)

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.head

    model = Tiny()
    ids = torch.tensor([[1, 2, 3]])
    expected = model.embedding(ids).detach().clone()
    add_hook_to_module(model.embedding, AlignDevicesHook(execution_device="cpu", offload=True))
    assert model.embedding.weight.device.type == "meta"
    counters = {}
    _install_cpu_embedding_lookup(model, "cpu", counters)
    assert not hasattr(model.embedding, "_hf_hook")
    assert model.embedding.weight.device.type == "cpu"
    assert torch.equal(model.embedding(ids), expected)
    assert torch.equal(model.embedding(input=ids), expected)
    assert counters["embedding_lookup_calls"] == 2
    assert counters.get("embedding_activation_transfer_bytes", 0) == 0
    assert qwen35_9b_8gb_device_map()["lm_head"] == "cpu"
    assert qwen35_9b_cpu_embedding_gpu_head_device_map()["lm_head"] == 0


def test_qwen_telemetry_is_separate_from_json_and_records_token_usage():
    adapter = Qwen35Adapter("stub", http_client=lambda _: {
        "choices": [{"message": {"content": '{"ok":true}'}}],
        "usage": {"prompt_tokens": 41, "completion_tokens": 5},
    })
    assert adapter.complete("JSON only") == {"ok": True}
    assert adapter.last_performance["counts"]["input_tokens"] == 41
    assert adapter.last_performance["counts"]["output_tokens"] == 5
    assert adapter.last_performance["timings_seconds"]["parse"] >= 0
    assert adapter.last_performance["counts"]["inference_calls"] == 1


def test_qwen_local_profiling_first_token_and_output_counts(tmp_path):
    class Processor:
        def apply_chat_template(self, *_args, **_kwargs):
            return {"input_ids": [[1, 2]]}

        def decode(self, _tokens, **_kwargs):
            return '"ok":true}'

    class Model:
        device = None

        def generate(self, **kwargs):
            streamer = kwargs["streamer"]
            streamer.put([[1, 2]])
            streamer.put([3])
            streamer.end()
            return [[1, 2, 3]]

    adapter = Qwen35Adapter(tmp_path, backend="local", local_loader=lambda _: (Model(), Processor()),
                            profile_timings=True)
    assert adapter.complete("JSON only") == {"ok": True}
    assert adapter.last_performance["counts"]["input_tokens"] == 2
    assert adapter.last_performance["counts"]["output_tokens"] == 1
    assert adapter.last_performance["timings_seconds"]["first_token"] >= 0
    assert adapter.last_performance["timings_seconds"]["decode"] >= 0
