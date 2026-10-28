"""Optional model/service adapters for the classroom hybrid pipeline."""

from .locate_anything import (
    LocateAnythingAdapter,
    LocateAnythingHealthError,
    LocateAnythingResponseError,
    parse_locate_anything_response,
)
from .qwen import Qwen35Adapter, QwenResponseError, VALID_DECISIONS, qwen35_9b_8gb_device_map
from ..qwen_worker import QwenProcessClient, QwenWorkerError
from .yolo import COCO_CLASS_NAMES, YoloAnomalyAdapter, YoloTrackerAdapter

__all__ = [
    "COCO_CLASS_NAMES",
    "LocateAnythingAdapter",
    "LocateAnythingHealthError",
    "LocateAnythingResponseError",
    "Qwen35Adapter",
    "QwenProcessClient",
    "QwenResponseError",
    "QwenWorkerError",
    "VALID_DECISIONS",
    "qwen35_9b_8gb_device_map",
    "YoloAnomalyAdapter",
    "YoloTrackerAdapter",
    "parse_locate_anything_response",
]
