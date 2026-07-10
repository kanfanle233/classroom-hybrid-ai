# 主流程 Python 脚本清单

本项目现在采用论文/GitHub 仓库常见结构：根目录只保留推荐入口和少量公共目录，真实主流程代码集中在 `classroom_pipeline/` 包内。

## 推荐入口

```bash
python run_pipeline.py run ...
python run_pipeline.py doctor
python run_pipeline.py validate --case-dir output/demo1
python run_pipeline.py clean --dry-run
```

## 主流程阶段

这些文件会被流水线直接调用，统一位于 `classroom_pipeline/pipeline/`：

```text
01_pose_video_demo.py
02_export_keypoints_jsonl.py
02b_export_objects_jsonl.py
02c_build_rear_roi_sr_cache.py
02d_export_behavior_det_jsonl.py
03_track_and_smooth.py
03c_estimate_track_uncertainty.py
03e_track_behavior_students.py
05_slowfast_actions.py
05b_fuse_actions_with_objects.py
05c_behavior_det_to_actions.py
05d_merge_action_sources.py
06_api_asr_realtime.py
06_asr_whisper_to_jsonl.py
06_overlay_pose_behavior_video.py
06b_event_query_extraction.py
06c_asr_openai_to_jsonl.py
06d_build_rear_row_contact_sheet.py
07_dual_verification.py
08_coherence_diagnosis.py
10_visualize_timeline.py
xx_align_multimodal.py
```

## 融合与契约

这些文件位于 `classroom_pipeline/fusion/`：

```text
semanticize_objects.py
semanticize_behavior_det.py
behavior_det_to_actions_v2.py
merge_fusion_actions_v2.py
build_event_queries_fusion_v2.py
check_fusion_contract.py
check_pipeline_contract_v2.py
```

## 验证器与契约包

- 契约：`classroom_pipeline/contracts/`
- 验证器：`classroom_pipeline/verifier/`
- LLM/学生判别器实验模块：`classroom_pipeline/judge/`

`python run_pipeline.py doctor` 会在启动前检查阶段注册表中的 32 个必需脚本和 1 个配置（共 33 个阶段项），缺失时直接失败。`08_coherence_diagnosis.py` 在验证器之后生成 `verified_events.coherence_v3.jsonl` 旁路产物，服务端优先读取该文件。

## 已归档内容

旧 `scripts/pipeline/`、`scripts/main/`、`web_viz/` 等不再作为主流程目录使用；兼容目录记录在 `paper_experiments/legacy_moved_dirs/scripts_pipeline_compat/`、`scripts_main_compat/` 和 `web_viz_empty/`。仍需留档但不进入主线的脚本位于 `paper_experiments/legacy_pipeline_scripts/`。
