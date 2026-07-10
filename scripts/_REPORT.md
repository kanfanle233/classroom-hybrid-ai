# `scripts/` 维护说明

> 更新日期：2026-07-10

`scripts/` 只保留兼容入口、实验、论文构建、维护和通用工具；生产主流程已经迁移到 `classroom_pipeline/`。推荐入口始终是根目录的 `run_pipeline.py`。

## 当前目录

```text
scripts/
├─ 09_run_pipeline.py                    # 旧命令兼容包装器
├─ experiments/
│  ├─ 16_run_rear_row_sr_ablation.py
│  ├─ 17_build_sr_ablation_paper_summary.py
│  ├─ 18_build_rear_row_gt_template.py
│  ├─ 19_eval_rear_row_metrics.py
│  ├─ 25_split_test_set.py
│  └─ 26_diag_tracking_quality.py
├─ maintenance/
│  ├─ _audit_step1_data.py
│  ├─ _audit_step2_balanced_sampling.py
│  ├─ _audit_step3_teacher_labels.py
│  └─ _audit_step6_machine_checks.py
├─ paper/
│  ├─ 30_build_paper_figures_tables.py
│  ├─ 31_build_paper_figures_curated.py
│  ├─ 32_build_d3og_selected_figures.py
│  └─ 33_build_repo_report_alignment.py
└─ utils/
   ├─ 02b_check_jsonl_schema.py
   ├─ 12_export_features.py
   ├─ 13_semantic_projection.py
   ├─ build_verifier_training_samples.py
   ├─ generate_test_fixtures.py
   ├─ object_evidence_mapping.py
   └─ sliced_inference_utils.py
```

## 与主流程的对应关系

- `scripts/09_run_pipeline.py`：只负责把旧命令转发到 `run_pipeline.py run`。
- 姿态、目标、跟踪、ASR、对齐、验证与一致性诊断：位于 `classroom_pipeline/pipeline/`。
- 语义化、动作融合和契约检查：位于 `classroom_pipeline/fusion/`。
- 验证器：位于 `classroom_pipeline/verifier/`。
- 前端数据打包：位于 `server/build_frontend_data_bundle.py`。
- 旧实验脚本：位于 `paper_experiments/legacy_pipeline_scripts/`，不得被新主流程调用。

完整阶段清单见 `docs/guides/main_flow_scripts.md`，机器可读注册表见 `classroom_pipeline/stage_registry.py`。

## 维护约束

1. 新增或移动主流程脚本时，必须同步修改编排器、`stage_registry.py`、主流程清单和回归测试。
2. `scripts/` 中的脚本必须能从仓库根目录直接执行；包内导入统一使用 `classroom_pipeline.*`。
3. 论文脚本不得引用不存在的旧路径，数据缺失时应给出明确错误或跳过说明。
4. 模型文件统一放在 `models/`，可再生成产物统一放在 `output/`。
