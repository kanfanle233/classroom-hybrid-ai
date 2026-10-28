"""Lazy, local-only Ultralytics adapters.

The module intentionally does not import :mod:`ultralytics` at import time.  A
worker can therefore import the pipeline on a machine where YOLO is served by a
different process, and tests can use a tiny injected model factory.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


# A class map made only from these labels is not evidence of a trained classroom
# anomaly detector.  In particular, the stock COCO ``person`` detector must not
# silently become an "anomaly" detector.
COCO_CLASS_NAMES = frozenset(
    {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    }
)


def _local_file(path: os.PathLike[str] | str, field_name: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{field_name} must be an existing local file: {candidate}")
    return candidate


def _plain(value: Any) -> Any:
    """Turn torch/numpy-like values into ordinary Python values."""

    if value is None:
        return None
    for method in ("detach", "cpu"):
        fn = getattr(value, method, None)
        if callable(fn):
            value = fn()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _rows(value: Any) -> list[Any]:
    value = _plain(value)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return [value]
    value = list(value)
    if value and not isinstance(value[0], (list, tuple)):
        return [value]
    return value


def _vector(value: Any) -> list[Any]:
    value = _plain(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _name_map(names: Any) -> dict[int, str]:
    if isinstance(names, Mapping):
        result: dict[int, str] = {}
        for key, value in names.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return result
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _optional_float(values: Sequence[Any], index: int) -> Optional[float]:
    if index >= len(values) or values[index] is None:
        return None
    return float(values[index])


def _optional_int(values: Sequence[Any], index: int) -> Optional[int]:
    if index >= len(values) or values[index] is None:
        return None
    return int(values[index])


class YoloTrackerAdapter:
    """Run local Ultralytics tracking and return dependency-free dictionaries."""

    def __init__(
        self,
        model_path: os.PathLike[str] | str,
        *,
        config_dir: os.PathLike[str] | str | None = None,
        tracker: str | os.PathLike[str] | None = None,
        device: Any = None,
        model_factory: Callable[[str], Any] | None = None,
        default_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_path = _local_file(model_path, "model_path")
        configured_dir = config_dir if config_dir is not None else os.environ.get("YOLO_CONFIG_DIR")
        self.config_dir = (
            Path(configured_dir).expanduser().resolve()
            if configured_dir is not None
            else (self.model_path.parent / ".ultralytics").resolve()
        )
        self.tracker = str(tracker) if tracker is not None else None
        self.device = device
        self._model_factory = model_factory
        self._default_kwargs = dict(default_kwargs or {})
        self._model: Any = None
        self._model_checked = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            # Ultralytics reads this while importing/initialising settings.
            os.environ["YOLO_CONFIG_DIR"] = str(self.config_dir)
            os.environ["YOLO_AUTOINSTALL"] = "false"
            os.environ["YOLO_OFFLINE"] = "true"
            factory = self._model_factory
            if factory is None:
                factory = importlib.import_module("ultralytics").YOLO
            self._model = factory(str(self.model_path))
        if not self._model_checked:
            self._after_model_loaded(self._model)
            self._model_checked = True
        return self._model

    def _after_model_loaded(self, model: Any) -> None:
        """Subclass hook for checks that require checkpoint metadata."""

    def _inference_kwargs(self, persist: bool, overrides: Mapping[str, Any]) -> dict[str, Any]:
        kwargs = dict(self._default_kwargs)
        kwargs.update(overrides)
        kwargs["persist"] = bool(persist)
        kwargs.setdefault("verbose", False)
        if self.tracker is not None:
            kwargs.setdefault("tracker", self.tracker)
        if self.device is not None:
            kwargs.setdefault("device", self.device)
        return kwargs

    def track(self, frame: Any, persist: bool = True, **kwargs: Any) -> dict[str, Any]:
        """Track one frame.

        Missing ``boxes.id`` is represented as ``track_id=None``.  That occurs
        before a tracker confirms a track and is not treated as an exception.
        """

        model = self._ensure_model()
        results = model.track(frame, **self._inference_kwargs(persist, kwargs))
        return self._convert_results(results, persist=bool(persist))

    def predict_batch(self, images: Sequence[Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Detect screenshots independently, preserving one result per image.

        This deliberately calls ``predict`` rather than ``track``. Sparse
        screenshot associations belong to the CPU stage, not a video tracker.
        """
        if not images:
            return []
        model = self._ensure_model()
        call_kwargs = dict(self._default_kwargs)
        call_kwargs.update(kwargs)
        for tracking_only in ("persist", "tracker", "stream"):
            call_kwargs.pop(tracking_only, None)
        call_kwargs.setdefault("verbose", False)
        if self.device is not None:
            call_kwargs.setdefault("device", self.device)
        results = list(model.predict(list(images), stream=False, **call_kwargs))
        if len(results) != len(images):
            raise RuntimeError(f"yolo_predict_result_count_mismatch:{len(results)}:{len(images)}")
        outputs = []
        for result in results:
            output = self._convert_results([result], persist=False)
            for detection in output["detections"]:
                detection["raw_track_id"] = detection.pop("track_id", None)
            output["metadata"].update({"operation": "predict", "cross_frame_ids": False})
            outputs.append(output)
        return outputs

    def __call__(self, frame: Any, persist: bool = True, **kwargs: Any) -> dict[str, Any]:
        return self.track(frame, persist=persist, **kwargs)

    def _convert_results(
        self,
        results: Any,
        *,
        persist: bool,
        names_override: Mapping[int, str] | None = None,
    ) -> dict[str, Any]:
        if results is None:
            result_items: list[Any] = []
        elif isinstance(results, (list, tuple)):
            result_items = list(results)
        elif isinstance(results, Iterable) and not isinstance(results, (str, bytes, Mapping)):
            result_items = list(results)
        else:
            result_items = [results]

        detections: list[dict[str, Any]] = []
        tasks: list[str] = []
        for result in result_items:
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            xyxy = _rows(getattr(boxes, "xyxy", None)) if boxes is not None else []
            scores = _vector(getattr(boxes, "conf", None)) if boxes is not None else []
            classes = _vector(getattr(boxes, "cls", None)) if boxes is not None else []
            track_ids = _vector(getattr(boxes, "id", None)) if boxes is not None else []
            pose_xy = _rows(getattr(keypoints, "xy", None)) if keypoints is not None else []
            pose_scores = _rows(getattr(keypoints, "conf", None)) if keypoints is not None else []
            names = dict(names_override or _name_map(getattr(result, "names", None)))
            if not names and self._model is not None:
                names = _name_map(getattr(self._model, "names", None))

            count = max(len(xyxy), len(pose_xy))
            for index in range(count):
                class_id = _optional_int(classes, index)
                track_id = _optional_int(track_ids, index)
                points = pose_xy[index] if index < len(pose_xy) else None

                if index < len(xyxy):
                    raw_bbox = list(xyxy[index])
                    if len(raw_bbox) < 4:
                        continue
                    bbox = [float(value) for value in raw_bbox[:4]]
                else:
                    valid_points = [
                        point
                        for point in (points or [])
                        if isinstance(point, (list, tuple)) and len(point) >= 2
                    ]
                    if not valid_points:
                        continue
                    xs = [float(point[0]) for point in valid_points]
                    ys = [float(point[1]) for point in valid_points]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]

                attributes: dict[str, Any] = {}
                if points is not None:
                    pose: dict[str, Any] = {
                        "keypoints": [
                            [float(coordinate) for coordinate in point[:2]]
                            for point in points
                            if isinstance(point, (list, tuple)) and len(point) >= 2
                        ]
                    }
                    if index < len(pose_scores):
                        pose["scores"] = [float(value) for value in pose_scores[index]]
                    attributes["pose"] = pose

                detections.append(
                    {
                        "label": names.get(class_id, str(class_id) if class_id is not None else "person"),
                        "bbox": bbox,
                        "score": _optional_float(scores, index),
                        "class_id": class_id,
                        "track_id": str(track_id) if track_id is not None else None,
                        "source": "yolo",
                        "attributes": attributes,
                    }
                )
            task = getattr(result, "task", None)
            if task is not None:
                tasks.append(str(task))

        return {
            "detections": detections,
            "metadata": {
                "persist": persist,
                "result_count": len(result_items),
                "tasks": tasks,
                "model_path": str(self.model_path),
            },
        }

    def close(self) -> None:
        self._model = None
        self._model_checked = False
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass


class YoloAnomalyAdapter(YoloTrackerAdapter):
    """YOLO adapter that refuses unverified generic weights.

    ``class_map`` must be the exact custom map embedded in the trained
    checkpoint.  At least one label must be outside the stock COCO vocabulary.
    This makes it impossible to accidentally label detections from ``yolo26n.pt``
    as classroom anomalies.
    """

    def __init__(
        self,
        model_path: os.PathLike[str] | str,
        *,
        class_map: Mapping[int, str],
        config_dir: os.PathLike[str] | str | None = None,
        tracker: str | os.PathLike[str] | None = None,
        device: Any = None,
        model_factory: Callable[[str], Any] | None = None,
        default_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        normalized = _name_map(class_map)
        if not normalized or any(not label.strip() for label in normalized.values()):
            raise ValueError("class_map must contain explicit non-empty custom anomaly labels")
        if all(label.strip().lower() in COCO_CLASS_NAMES for label in normalized.values()):
            raise ValueError(
                "class_map contains only stock COCO labels; provide a trained anomaly checkpoint"
            )
        self.class_map = {key: label.strip() for key, label in normalized.items()}
        super().__init__(
            model_path,
            config_dir=config_dir,
            tracker=tracker,
            device=device,
            model_factory=model_factory,
            default_kwargs=default_kwargs,
        )

    def _after_model_loaded(self, model: Any) -> None:
        actual = _name_map(getattr(model, "names", None))
        if not actual:
            raise ValueError("YOLO anomaly checkpoint does not expose a verifiable class map")
        if actual != self.class_map:
            raise ValueError(
                f"YOLO anomaly class map mismatch: configured={self.class_map!r}, "
                f"checkpoint={actual!r}"
            )

    def track(self, frame: Any, persist: bool = True, **kwargs: Any) -> dict[str, Any]:
        output = super().track(frame, persist=persist, **kwargs)
        output["metadata"]["role"] = "known_anomaly_candidate_generator"
        return output

    def predict(self, frame: Any, **kwargs: Any) -> dict[str, Any]:
        model = self._ensure_model()
        call_kwargs = dict(self._default_kwargs)
        call_kwargs.update(kwargs)
        call_kwargs.setdefault("verbose", False)
        if self.device is not None:
            call_kwargs.setdefault("device", self.device)
        results = model.predict(frame, **call_kwargs)
        output = self._convert_results(
            results,
            persist=False,
            names_override=self.class_map,
        )
        output["metadata"]["role"] = "known_anomaly_candidate_generator"
        return output

    infer = predict


__all__ = ["COCO_CLASS_NAMES", "YoloAnomalyAdapter", "YoloTrackerAdapter"]
