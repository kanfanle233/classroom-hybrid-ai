"""Configuration loading with explicit local path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPECTED_SCHEMA = "classroom_hybrid_ai.config.v1"


class ConfigError(ValueError):
    """Raised when the pipeline configuration is incomplete or inconsistent."""


def _deep_get(payload: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Raw policy plus deterministic roots used to resolve local artifacts."""

    source_path: Path
    project_root: Path
    repository_root: Path
    payload: dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        return _deep_get(self.payload, dotted, default)

    def section(self, name: str) -> dict[str, Any]:
        value = self.payload.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"section_must_be_mapping:{name}")
        return value

    def resolve(self, value: str | Path | None, *, base: str = "repository") -> Path | None:
        if value in (None, ""):
            return None
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        anchor = self.repository_root if base == "repository" else self.project_root
        return (anchor / candidate).resolve()

    @property
    def python_executable(self) -> Path:
        value = self.get("runtime.python_executable")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("runtime.python_executable_missing")
        return Path(value).resolve()

    @property
    def output_root(self) -> Path:
        value = self.get("paths.output_root")
        path = self.resolve(value)
        if path is None:
            raise ConfigError("paths.output_root_missing")
        return path

    @property
    def dataset_root(self) -> Path:
        value = self.get("paths.dataset_root")
        path = self.resolve(value)
        if path is None:
            raise ConfigError("paths.dataset_root_missing")
        return path

    @property
    def model_manifest_path(self) -> Path:
        value = self.get("paths.model_manifest")
        path = self.resolve(value, base="project")
        if path is None:
            raise ConfigError("paths.model_manifest_missing")
        return path

    @property
    def behavior_taxonomy_path(self) -> Path:
        """Closed action taxonomy used by runtime classification and evaluation."""
        value = self.get("paths.behavior_taxonomy")
        path = self.resolve(value, base="project")
        if path is None:
            raise ConfigError("paths.behavior_taxonomy_missing")
        return path

    @property
    def action_registry_root(self) -> Path:
        value = self.get("paths.action_registry_root", ".runtime/classroom_hybrid_ai/action_registry")
        path = self.resolve(value)
        if path is None:
            raise ConfigError("paths.action_registry_root_missing")
        return path

    @property
    def action_profiles_path(self) -> Path:
        value = self.get("profiles.action_profiles", "configs/action_profiles.yaml")
        path = self.resolve(value, base="project")
        if path is None:
            raise ConfigError("profiles.action_profiles_missing")
        return path

    def model_path(self, key: str) -> Path | None:
        value = self.get(key)
        return self.resolve(value)

    def validate_static(self) -> list[str]:
        errors: list[str] = []
        if self.payload.get("schema_version") != EXPECTED_SCHEMA:
            errors.append(f"unsupported_schema:{self.payload.get('schema_version')}")
        if self.get("runtime.max_gpu_workers") != 1:
            errors.append("runtime.max_gpu_workers_must_be_one_on_8gb_gpu")
        if str(self.get("qwen.prompt_version", "legacy_v1")) not in {"legacy_v1", "compact_v1"}:
            errors.append("qwen_prompt_version_invalid")
        for key in ("video.preview_decode_strategy", "video.evidence_decode_strategy"):
            if str(self.get(key, "sequential")) not in {"sequential", "auto"}:
                errors.append(f"{key}_invalid")
        if str(self.get("video.encoder_backend", "opencv")) not in {"opencv", "nvenc"}:
            errors.append("video_encoder_backend_invalid")
        try:
            if int(self.get("performance.preprocess_workers", 2)) not in {1, 2, 4}:
                errors.append("preprocess_workers_must_be_1_2_or_4")
            if int(self.get("performance.preprocess_queue_size", 2)) < 1:
                errors.append("preprocess_queue_size_must_be_positive")
            if int(self.get("video.screenshot_cache_bytes", 2 * 1024**3)) < 0:
                errors.append("screenshot_cache_bytes_must_be_nonnegative")
            if int(self.get("locate_anything.max_new_tokens", 256)) < 1:
                errors.append("locate_max_new_tokens_must_be_positive")
            side = self.get("locate_anything.maximum_image_side")
            if side is not None and int(side) < 1:
                errors.append("locate_maximum_image_side_must_be_positive_or_null")
        except (TypeError, ValueError):
            errors.append("performance_configuration_invalid")
        primary_detector = str(self.get("pipeline.primary_detector", "yolo"))
        if primary_detector not in {"locate_anything", "yolo", "hybrid"}:
            errors.append("pipeline_primary_detector_invalid")
        try:
            from .research_policy import localization_route, budget_policy
            route = localization_route(self)
            budget_policy(self)
            if primary_detector == "hybrid" and route != "yolo_locate_hybrid":
                errors.append("hybrid_primary_requires_explicit_hybrid_route")
            if route != "locate_only":
                batch = self.get("localization.batch_size", 1)
                if type(batch) is not int or batch not in (1, 2, 4):
                    errors.append("localization_batch_size_must_be_1_2_or_4")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        if primary_detector == "locate_anything" and self.get("locate_anything.enabled") is not True:
            errors.append("locate_primary_detector_must_be_enabled")
        if (
            primary_detector == "locate_anything"
            and self.get("locate_anything.backend") == "http"
            and self.get("locate_anything.auto_start_service") is not True
            and self.get("locate_anything.required") is True
        ):
            errors.append("locate_primary_http_requires_auto_start_service")
        try:
            if float(self.get("locate_anything.service_start_timeout_seconds", 0)) <= 0:
                errors.append("locate_service_start_timeout_must_be_positive")
        except (TypeError, ValueError):
            errors.append("locate_service_start_timeout_invalid")
        try:
            if int(self.get("locate_first.screenshot_frame_stride", 0)) < 1:
                errors.append("locate_screenshot_frame_stride_must_be_positive")
        except (TypeError, ValueError):
            errors.append("locate_screenshot_frame_stride_invalid")
        if self.get("qwen.batch_size") != 1:
            errors.append("qwen.batch_size_must_be_one_for_9b_on_8gb_gpu")
        if self.get("qwen.stateless_requests") is not True:
            errors.append("qwen.stateless_requests_must_be_true")
        if int(self.get("qwen.max_input_images", 0)) < 1:
            errors.append("qwen.max_input_images_invalid")
        if int(self.get("qwen.max_input_images", 0)) != 1:
            errors.append("qwen.max_input_images_must_be_one_until_8gb_gate_passes")
        if int(self.get("qwen.max_new_tokens", 0)) > 128:
            errors.append("qwen.max_new_tokens_exceeds_8gb_policy")
        k_of_m = self.get("triggering.k_of_m")
        if not (
            isinstance(k_of_m, list)
            and len(k_of_m) == 2
            and all(isinstance(v, int) for v in k_of_m)
            and 1 <= k_of_m[0] <= k_of_m[1]
        ):
            errors.append("triggering.k_of_m_invalid")
        if self.get("triggering.require_full_window") is not True:
            errors.append("triggering.require_full_window_must_be_true")
        if self.get("fusion.qwen_self_reported_confidence_is_probability") is not False:
            errors.append("qwen_self_reported_confidence_cannot_be_probability")
        if self.get("runtime.execution_layout") != "sequential_offline_passes":
            errors.append("8gb_gpu_requires_sequential_offline_passes")
        profile = str(self.get("research.profile", "core8_v1"))
        if not profile:
            errors.append("research_profile_missing")
        if str(self.get("research.mode", "production_adaptive")) not in {"production_adaptive", "exhaustive_validation"}:
            errors.append("research_mode_invalid")
        try:
            if int(self.get("locate_first.maximum_screenshots", 1500)) < 1:
                errors.append("maximum_screenshots_must_be_positive")
            if int(self.get("locate_first.maximum_screenshots", 1500)) > 1500:
                errors.append("maximum_screenshots_must_be_at_most_1500")
            if int(self.get("locate_first.locate_cap", 1500)) < 1:
                errors.append("locate_cap_must_be_positive")
            if int(self.get("locate_first.locate_cap", 1500)) > 1500:
                errors.append("locate_cap_must_be_at_most_1500")
            if int(self.get("locate_first.reserved_change_events", 300)) < 0:
                errors.append("reserved_change_events_must_be_non_negative")
            if int(self.get("locate_first.reserved_change_events", 300)) > 300:
                errors.append("reserved_change_events_must_be_at_most_300")
            if int(self.get("locate_first.change_anchor_sample_stride", 10)) < 1:
                errors.append("change_anchor_sample_stride_must_be_positive")
            if int(self.get("locate_first.exhaustive_max_screenshots", 300)) < 1:
                errors.append("exhaustive_max_screenshots_must_be_positive")
            if int(self.get("locate_first.exhaustive_max_screenshots", 300)) > 300:
                errors.append("exhaustive_max_screenshots_must_be_at_most_300")
            if not 0.0 <= float(self.get("locate_first.change_anchor_threshold", 0.08)) <= 1.0:
                errors.append("change_anchor_threshold_out_of_range")
            if float(self.get("locate_first.change_anchor_min_gap_seconds", 0.5)) < 0:
                errors.append("change_anchor_min_gap_seconds_must_be_non_negative")
        except (TypeError, ValueError):
            errors.append("locate_cap_invalid")
        action_mode = str(self.get("qwen.action_mode", "open_vocabulary"))
        if action_mode not in {"fixed_eight", "open_vocabulary"}:
            errors.append("qwen_action_mode_invalid")
        if action_mode == "fixed_eight":
            if self.get("open_vocabulary.discovery_pass") is not False:
                errors.append("fixed_eight_must_disable_discovery_pass")
        elif self.get("open_vocabulary.enabled") is not True:
            errors.append("open_vocabulary_must_be_enabled")
        if int(self.get("open_vocabulary.maximum_session_actions", 0)) < 1:
            errors.append("open_vocabulary.maximum_session_actions_invalid")
        open_vocab_gate = self.get("open_vocabulary.open_vocab_gate", {})
        if isinstance(open_vocab_gate, Mapping) and bool(open_vocab_gate.get("enabled", False)):
            provider = str(open_vocab_gate.get("provider", "none")).strip().lower()
            if provider not in {"xclip", "mobileviclip"}:
                errors.append("open_vocab_gate_provider_invalid")
            if open_vocab_gate.get("benchmark_verified") is not True:
                errors.append("open_vocab_gate_requires_same_evidence_benchmark")
        for key in (
            "tracking.max_stitch_gap_seconds",
            "tracking.max_normalized_center_distance",
            "tracking.max_appearance_distance",
            "tracking.minimum_match_margin",
        ):
            try:
                if float(self.get(key)) < 0:
                    errors.append(f"{key}_must_be_non_negative")
            except (TypeError, ValueError):
                errors.append(f"{key}_invalid")
        try:
            if float(self.get("tracking.max_stitch_gap_seconds")) > 2.0:
                errors.append("tracking.max_stitch_gap_seconds_must_be_at_most_2_seconds")
            if float(self.get("tracking.max_normalized_center_distance")) > 0.6:
                errors.append("tracking.max_normalized_center_distance_must_be_at_most_0_6")
            if float(self.get("tracking.max_appearance_distance")) > 0.35:
                errors.append("tracking.max_appearance_distance_must_be_at_most_0_35")
            if float(self.get("tracking.minimum_match_margin")) < 0.15:
                errors.append("tracking.minimum_match_margin_must_be_at_least_0_15")
        except (TypeError, ValueError):
            # The individual checks above report the malformed value.
            pass
        if str(self.get("qwen.backend", "local")) == "local" and self.get("qwen.process_isolation") is not True:
            errors.append("local_qwen_process_isolation_must_be_true")
        for key in ("yolo.keypoint_motion_window_seconds", "yolo.keypoint_motion_threshold"):
            try:
                if float(self.get(key)) <= 0:
                    errors.append(f"{key}_must_be_positive")
            except (TypeError, ValueError):
                errors.append(f"{key}_invalid")
        try:
            if float(self.get("yolo.keypoint_motion_min_gap_seconds")) < 0:
                errors.append("yolo.keypoint_motion_min_gap_seconds_must_be_non_negative")
        except (TypeError, ValueError):
            errors.append("yolo.keypoint_motion_min_gap_seconds_invalid")
        try:
            if float(self.get("triggering.forced_audit_seconds")) <= 0:
                errors.append("triggering.forced_audit_seconds_must_be_positive")
        except (TypeError, ValueError):
            errors.append("triggering.forced_audit_seconds_invalid")
        return errors


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigError(f"config_not_found:{source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ConfigError("config_root_must_be_mapping")
    project_root = source.parent.parent.resolve()
    repository_value = _deep_get(payload, "paths.repository_root", "..")
    if repository_value == "@cwd":
        # The wheel ships a generic config while models/data intentionally stay
        # outside the wheel.  Users invoke the installed command from the
        # local workspace that owns those artifacts.
        repository_root = Path.cwd().resolve()
    else:
        repository_path = Path(str(repository_value))
        repository_root = (
            repository_path.resolve()
            if repository_path.is_absolute()
            else (project_root / repository_path).resolve()
        )
    config = ProjectConfig(
        source_path=source,
        project_root=project_root,
        repository_root=repository_root,
        payload=payload,
    )
    errors = config.validate_static()
    if errors:
        raise ConfigError(";".join(errors))
    return config
