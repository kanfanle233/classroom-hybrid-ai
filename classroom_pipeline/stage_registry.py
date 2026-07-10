from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: str
    required_paths: tuple[str, ...] = ()
    optional_paths: tuple[str, ...] = ()

    def required_abs(self) -> Iterable[Path]:
        for raw in self.required_paths:
            yield PROJECT_ROOT / raw


STAGES: tuple[StageSpec, ...] = (
    StageSpec("pose", "script", ("classroom_pipeline/pipeline/01_pose_video_demo.py",)),
    StageSpec("keypoints", "script", ("classroom_pipeline/pipeline/02_export_keypoints_jsonl.py",)),
    StageSpec("sr_cache", "script", ("classroom_pipeline/pipeline/02c_build_rear_roi_sr_cache.py",)),
    StageSpec("pose_tracks", "script", ("classroom_pipeline/pipeline/03_track_and_smooth.py",)),
    StageSpec("pose_uncertainty", "script", ("classroom_pipeline/pipeline/03c_estimate_track_uncertainty.py",)),
    StageSpec("objects", "script", ("classroom_pipeline/pipeline/02b_export_objects_jsonl.py",)),
    StageSpec("behavior_detection", "script", ("classroom_pipeline/pipeline/02d_export_behavior_det_jsonl.py",)),
    StageSpec("behavior_tracks", "script", ("classroom_pipeline/pipeline/03e_track_behavior_students.py",)),
    StageSpec("slowfast_actions", "script", ("classroom_pipeline/pipeline/05_slowfast_actions.py",)),
    StageSpec("object_action_fusion", "script", ("classroom_pipeline/pipeline/05b_fuse_actions_with_objects.py",)),
    StageSpec("behavior_to_actions", "script", ("classroom_pipeline/pipeline/05c_behavior_det_to_actions.py",)),
    StageSpec("merge_action_sources", "script", ("classroom_pipeline/pipeline/05d_merge_action_sources.py",)),
    StageSpec("pose_behavior_overlay", "script", ("classroom_pipeline/pipeline/06_overlay_pose_behavior_video.py",)),
    StageSpec("rear_contact_sheet", "script", ("classroom_pipeline/pipeline/06d_build_rear_row_contact_sheet.py",)),
    StageSpec("asr_api_realtime", "script", ("classroom_pipeline/pipeline/06_api_asr_realtime.py",)),
    StageSpec("asr_whisper", "script", ("classroom_pipeline/pipeline/06_asr_whisper_to_jsonl.py",)),
    StageSpec("asr_openai", "script", ("classroom_pipeline/pipeline/06c_asr_openai_to_jsonl.py",)),
    StageSpec("event_queries", "script", ("classroom_pipeline/pipeline/06b_event_query_extraction.py",)),
    StageSpec("multimodal_alignment", "script", ("classroom_pipeline/pipeline/xx_align_multimodal.py",)),
    StageSpec("dual_verification", "script", ("classroom_pipeline/pipeline/07_dual_verification.py",)),
    StageSpec("coherence_diagnosis", "script", ("classroom_pipeline/pipeline/08_coherence_diagnosis.py",)),
    StageSpec("timeline", "script", ("classroom_pipeline/pipeline/10_visualize_timeline.py",)),
    StageSpec("fusion_semantic_objects", "script", ("classroom_pipeline/fusion/semanticize_objects.py",)),
    StageSpec("fusion_semantic_behavior", "script", ("classroom_pipeline/fusion/semanticize_behavior_det.py",)),
    StageSpec("fusion_behavior_actions", "script", ("classroom_pipeline/fusion/behavior_det_to_actions_v2.py",)),
    StageSpec("fusion_merge", "script", ("classroom_pipeline/fusion/merge_fusion_actions_v2.py",)),
    StageSpec("fusion_queries", "script", ("classroom_pipeline/fusion/build_event_queries_fusion_v2.py",)),
    StageSpec("fusion_contract", "script", ("classroom_pipeline/fusion/check_fusion_contract.py",)),
    StageSpec("pipeline_contract", "script", ("classroom_pipeline/fusion/check_pipeline_contract_v2.py",)),
    StageSpec("verifier_train", "script", ("classroom_pipeline/verifier/train.py",)),
    StageSpec("verifier_eval", "script", ("classroom_pipeline/verifier/eval.py",)),
    StageSpec("verifier_calibration", "script", ("classroom_pipeline/verifier/calibration.py",)),
    StageSpec("semantic_profile", "config", ("classroom_pipeline/profiles/action_semantics_8class.yaml",)),
)


def missing_required_paths() -> list[Path]:
    missing: list[Path] = []
    for stage in STAGES:
        for path in stage.required_abs():
            if not path.exists():
                missing.append(path)
    return missing
