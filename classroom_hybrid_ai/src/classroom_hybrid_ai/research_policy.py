"""The effective research route and candidate budget, shared by plan and run."""
from __future__ import annotations

import math
from typing import Any

ROUTES = ("locate_only", "yolo_only", "yolo_locate_hybrid")
DEFAULT_BUDGET = {"selection_mode": "fair", "minimum": 48, "maximum": 240,
                  "per_student": 2.0, "per_minute": 2.0}


def localization_route(config: Any) -> str:
    route = config.get("pipeline.localization_route", None)
    if route is None:
        route = "yolo_only" if config.get("pipeline.primary_detector", "locate_anything") == "yolo" else "locate_only"
    if route not in ROUTES:
        raise ValueError(f"unsupported_localization_route:{route}")
    return str(route)


def budget_policy(config: Any) -> dict[str, Any]:
    result = {key: config.get(f"research_budget.{key}", default) for key, default in DEFAULT_BUDGET.items()}
    if result["selection_mode"] not in {"fair", "all_candidates"}:
        raise ValueError("research_budget_selection_mode_invalid")
    for key in ("minimum", "maximum"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"research_budget_{key}_invalid")
    if result["minimum"] > result["maximum"]:
        raise ValueError("research_budget_minimum_exceeds_maximum")
    for key in ("per_student", "per_minute"):
        value = result[key]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError(f"research_budget_{key}_invalid")
    return result


def qwen_selection_budget(config: Any, duration_seconds: float, effective_students: int,
                          candidate_count: int) -> int:
    policy = budget_policy(config)
    if type(candidate_count) is not int or candidate_count < 0:
        raise ValueError("candidate_count_invalid")
    if policy["selection_mode"] == "all_candidates":
        return candidate_count
    cap = min(policy["maximum"], max(policy["minimum"],
        math.ceil(policy["per_student"] * max(0, effective_students)),
        math.ceil(policy["per_minute"] * max(0.0, duration_seconds) / 60.0)))
    return cap


def effective_configuration_report(config: Any) -> dict[str, Any]:
    ignored = {
        "locate_first.screenshot_frame_stride": "adaptive_scheduler_selects_times",
        "locate_first.maximum_screenshots": "use_locate_first.locate_cap",
        "locate_anything.max_scene_images": "S2_consumes_screenshot_schedule_or_difficulty_queue",
        "open_vocabulary.total_qwen_requests_per_minute": "use_research_budget.per_minute",
        "open_vocabulary.total_qwen_requests_minimum": "use_research_budget.minimum",
        "open_vocabulary.total_qwen_requests_maximum": "use_research_budget.maximum",
        "locate_first.qwen_every_screenshot": "use_research_budget.selection_mode=all_candidates",
    }
    return {"schema_version": "classroom_hybrid_ai.effective_research_policy.v1",
            "localization_route": localization_route(config), "budget": budget_policy(config),
            "ignored_legacy_fields": [{"field": key, "requested": config.get(key), "reason": reason}
                                      for key, reason in ignored.items() if config.get(key) is not None],
            "sampling_policy": "AdaptiveScreenshotScheduler; existing_PTS_and_cap_semantics",
            "all_candidates_experiment": budget_policy(config)["selection_mode"] == "all_candidates"}
