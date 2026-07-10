#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
REPORT_DATE = "20260423"
OUT_DIR = ROOT / "docs" / "assets" / "tables" / f"repo_alignment_{REPORT_DATE}"
REPORT_PATH = ROOT / "docs" / f"repo_report_alignment_{REPORT_DATE}.md"
DEEP_REPORT_PATH = Path(r"D:\Users\Lenovo\Downloads\deep-research-report (1).md")

FULL_AUDIT_DIR = ROOT / "docs" / "assets" / "tables" / "full_audit_20260422"
SCAN_SUMMARY_PATH = FULL_AUDIT_DIR / "scan_summary.json"
SCRIPT_INDEX_PATH = FULL_AUDIT_DIR / "script_index.csv"
MAINLINE_MERGED_SUMMARY_PATH = ROOT / "paper_experiments" / "real_cases" / "mainline_branch_merged_summary.json"
MAINLINE_MATRIX_CSV_PATH = ROOT / "paper_experiments" / "real_cases" / "mainline_branch_experiment_matrix.csv"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def safe_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def mean_from_rows(rows: Sequence[Dict[str, str]], key: str) -> float:
    values = [safe_float(row.get(key)) for row in rows]
    filtered = [v for v in values if v is not None]
    if not filtered:
        return 0.0
    return float(statistics.mean(filtered))


def sum_from_rows(rows: Sequence[Dict[str, str]], key: str) -> float:
    values = [safe_float(row.get(key)) for row in rows]
    filtered = [v for v in values if v is not None]
    return float(sum(filtered))


def count_rows(rows: Sequence[Dict[str, str]], key: str, *, truthy: bool = True) -> int:
    if truthy:
        return sum(1 for row in rows if safe_bool(row.get(key)))
    return sum(1 for row in rows if not safe_bool(row.get(key)))


def path_exists(raw: str) -> bool:
    p = Path(raw)
    if p.is_absolute():
        return p.exists()
    return (ROOT / p).exists()


def resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return ROOT / p


SCRIPT_CATEGORY_SPECS: List[Dict[str, Any]] = [
    {
        "script_path": "run_pipeline.py",
        "category": "主线流程",
        "subsystem": "pipeline_orchestration",
        "role": "总编排入口，串起 pose / track / action / ASR / align / verify / timeline",
        "primary_outputs": "pipeline_manifest.json; verified_events.jsonl; timeline_chart.png",
        "inspection_note": "直接读取脚本头部，确认包含 step orchestration、cache/freshness 判定、模型路径解析。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "classroom_pipeline/pipeline/02_export_keypoints_jsonl.py",
        "category": "主线流程",
        "subsystem": "vision_pose",
        "role": "YOLO pose 关键点导出",
        "primary_outputs": "pose_tracks.jsonl / keypoint rows",
        "inspection_note": "与研究报告和主线文档一致，作为视觉前端起点。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "classroom_pipeline/pipeline/03_track_and_smooth.py",
        "category": "主线流程",
        "subsystem": "tracking",
        "role": "跟踪与轨迹平滑",
        "primary_outputs": "pose_tracks_smooth.jsonl",
        "inspection_note": "研究报告明确将其视为自定义课堂 tracker 主体。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "classroom_pipeline/pipeline/03c_estimate_track_uncertainty.py",
        "category": "主线流程",
        "subsystem": "uq",
        "role": "轨迹级不确定性估计",
        "primary_outputs": "pose_tracks_smooth_uq.jsonl",
        "inspection_note": "研究报告已核对 uq_track / uq_motion 等结构化字段。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "classroom_pipeline/pipeline/05_slowfast_actions.py",
        "category": "主线流程",
        "subsystem": "action_recognition",
        "role": "动作片段识别 / rules 回退",
        "primary_outputs": "actions.jsonl",
        "inspection_note": "研究报告明确当前 mainline 使用 rules 模式，非端到端 YOLO 行为分类。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "classroom_pipeline/pipeline/06_asr_whisper_to_jsonl.py",
        "category": "主线流程",
        "subsystem": "text_asr",
        "role": "Whisper ASR + transcript 非空兜底",
        "primary_outputs": "transcript.jsonl",
        "inspection_note": "研究报告明确其承担文本流主路径。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "classroom_pipeline/pipeline/06b_event_query_extraction.py",
        "category": "主线流程",
        "subsystem": "text_event_query",
        "role": "事件查询抽取",
        "primary_outputs": "event_queries.jsonl",
        "inspection_note": "研究报告明确这是文本语义流的现实落点。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "classroom_pipeline/pipeline/xx_align_multimodal.py",
        "category": "主线流程",
        "subsystem": "multimodal_alignment",
        "role": "固定窗 / UQ 自适应窗跨模态对齐",
        "primary_outputs": "align_multimodal.json",
        "inspection_note": "直接读取脚本头部，确认包含 uq_index、motion_stability、adaptive_uq 窗口逻辑。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "classroom_pipeline/pipeline/07_dual_verification.py",
        "category": "主线流程",
        "subsystem": "verification",
        "role": "query + align + verifier -> verified_events.jsonl",
        "primary_outputs": "verified_events.jsonl; verifier_eval_report.json",
        "inspection_note": "直接读取脚本头部，确认 fallback alignment、infer_verified_rows、contract validation。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "classroom_pipeline/pipeline/08_coherence_diagnosis.py",
        "category": "主线流程",
        "subsystem": "multimodal_coherence",
        "role": "验证后多模态一致性与冲突类型诊断",
        "primary_outputs": "verified_events.coherence_v3.jsonl",
        "inspection_note": "主编排器 step73 调用，输出旁路 v3 事件契约。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "scripts/experiments/16_run_rear_row_sr_ablation.py",
        "category": "实验分支",
        "subsystem": "rear_row_sr_ablation",
        "role": "后排 ROI 超分辨率消融运行器",
        "primary_outputs": "rear-row SR ablation case outputs",
        "inspection_note": "当前保留的后排小目标增强实验入口。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "scripts/experiments/17_build_sr_ablation_paper_summary.py",
        "category": "实验分支",
        "subsystem": "rear_row_sr_summary",
        "role": "汇总后排超分消融的论文表格与摘要",
        "primary_outputs": "sr_ablation_paper_summary.json; summary tables",
        "inspection_note": "与 16 号消融运行器成对使用。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "scripts/experiments/18_build_rear_row_gt_template.py",
        "category": "实验分支",
        "subsystem": "rear_row_ground_truth",
        "role": "生成后排评测 GT 模板",
        "primary_outputs": "rear-row GT template",
        "inspection_note": "为 19 号后排指标评估提供人工标注模板。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "scripts/experiments/19_eval_rear_row_metrics.py",
        "category": "实验分支",
        "subsystem": "rear_row_metrics",
        "role": "计算后排检测与跟踪指标",
        "primary_outputs": "rear_row_metrics.json; metric tables",
        "inspection_note": "读取 18 号 GT 与流水线产物形成可复现评估。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "scripts/experiments/25_split_test_set.py",
        "category": "数据划分",
        "subsystem": "held_out_split",
        "role": "构造可复现测试集划分",
        "primary_outputs": "test split manifests",
        "inspection_note": "数据划分工具，不参与在线主流程。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "scripts/paper/30_build_paper_figures_tables.py",
        "category": "图表生成",
        "subsystem": "paper_auto_figures",
        "role": "从 run logs / exp outputs 生成 paper_auto 图表表格",
        "primary_outputs": "docs/assets/charts/paper_auto/*; docs/assets/tables/paper_auto/*",
        "inspection_note": "直接读取脚本头部，确认 selection_matrix 校验、batch compare、heatmap、reliability bins。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "scripts/paper/31_build_paper_figures_curated.py",
        "category": "图表生成",
        "subsystem": "paper_curated_figures",
        "role": "从全仓遍历后筛选 paper_curated 图表",
        "primary_outputs": "docs/assets/charts/paper_curated/*; docs/assets/tables/paper_curated/*",
        "inspection_note": "直接读取脚本头部，确认 batch CI、delta heatmap、confusion aggregation、quality gate。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "server/build_frontend_data_bundle.py",
        "category": "展示系统",
        "subsystem": "pages_demo",
        "role": "从单个 case 构建前端数据包",
        "primary_outputs": "frontend_data_manifest.json; data/assets bundle",
        "inspection_note": "读取 pipeline manifest 与标准产物，生成轻量展示数据。",
        "include_in_primary_table": "yes",
    },
    {
        "script_path": "server/app.py",
        "category": "展示系统",
        "subsystem": "backend_server",
        "role": "FastAPI 可视化后端服务原型",
        "primary_outputs": "HTTP API + static/docs/output mounts",
        "inspection_note": "直接读取脚本头部，确认 FastAPI、/output /data /docs /assets 静态挂载。",
        "include_in_primary_table": "yes",
    },
]


FIGURE_SPECS: List[Dict[str, Any]] = [
    {
        "artifact_id": "fig_pipeline_flow",
        "path": "docs/assets/charts/pipeline_flow.svg",
        "artifact_type": "figure",
        "recommended_section": "方法总览",
        "quality_tier": "main_text",
        "paper_ready": "yes",
        "evidence_basis": "方法/系统结构图，非性能图",
        "notes": "最稳妥的主文方法图。",
    },
    {
        "artifact_id": "fig_batch_compare_ci",
        "path": "docs/assets/charts/paper_curated/fig01_batch_compare_with_ci.png",
        "artifact_type": "figure",
        "recommended_section": "系统比较 / 运行稳定性",
        "quality_tier": "main_text",
        "paper_ready": "yes",
        "evidence_basis": "full_traversal_figure_selection_20260422 推荐保留在主文",
        "notes": "适合展示 run-level CI 对比。",
    },
    {
        "artifact_id": "fig_case_delta_top20",
        "path": "docs/assets/charts/paper_curated/fig03_case_delta_heatmap_top20.png",
        "artifact_type": "figure",
        "recommended_section": "案例差异分析",
        "quality_tier": "main_text",
        "paper_ready": "yes",
        "evidence_basis": "top-20 paired case delta",
        "notes": "适合 mainline vs behavior_aug 的变化热力图。",
    },
    {
        "artifact_id": "fig_epoch_three_lines",
        "path": "docs/assets/charts/paper_curated/fig04_epoch_three_lines_curated.png",
        "artifact_type": "figure",
        "recommended_section": "训练过程 / 附录",
        "quality_tier": "main_text",
        "paper_ready": "yes",
        "evidence_basis": "full_traversal_figure_selection_20260422 推荐主文保留",
        "notes": "若正文篇幅有限，可转附录。",
    },
    {
        "artifact_id": "fig_confusion_grid",
        "path": "docs/assets/charts/paper_curated/fig06_confusion_aggregate_grid.png",
        "artifact_type": "figure",
        "recommended_section": "错误结构分析",
        "quality_tier": "main_text",
        "paper_ready": "yes",
        "evidence_basis": "聚合 confusion matrices",
        "notes": "适合主文错误结构展示。",
    },
    {
        "artifact_id": "fig_reliability_before",
        "path": "docs/assets/charts/reliability_before.svg",
        "artifact_type": "figure",
        "recommended_section": "可靠性 / 附录",
        "quality_tier": "appendix_only",
        "paper_ready": "no",
        "evidence_basis": "exp_b sample_count=4",
        "notes": "工程验证可用，不能作最终论文主结论。",
    },
    {
        "artifact_id": "fig_reliability_after",
        "path": "docs/assets/charts/reliability_after.svg",
        "artifact_type": "figure",
        "recommended_section": "可靠性 / 附录",
        "quality_tier": "appendix_only",
        "paper_ready": "no",
        "evidence_basis": "exp_b sample_count=4",
        "notes": "与 before 成对使用。",
    },
    {
        "artifact_id": "fig_case_visual_fix",
        "path": "docs/assets/cases/case_visual_fix.svg",
        "artifact_type": "figure",
        "recommended_section": "案例解释",
        "quality_tier": "appendix_only",
        "paper_ready": "yes",
        "evidence_basis": "案例示意图",
        "notes": "可与文本纠偏案例成组使用。",
    },
    {
        "artifact_id": "fig_case_text_fix",
        "path": "docs/assets/cases/case_text_fix.svg",
        "artifact_type": "figure",
        "recommended_section": "案例解释",
        "quality_tier": "appendix_only",
        "paper_ready": "yes",
        "evidence_basis": "案例示意图",
        "notes": "适合说明文本流纠偏。",
    },
    {
        "artifact_id": "fig_case_conflict_uncertain",
        "path": "docs/assets/cases/case_conflict_uncertain.svg",
        "artifact_type": "figure",
        "recommended_section": "冲突 / uncertain 案例",
        "quality_tier": "appendix_only",
        "paper_ready": "yes",
        "evidence_basis": "案例示意图",
        "notes": "适合展示系统为何输出 uncertain。",
    },
    {
        "artifact_id": "fig_gold_contact_sheet",
        "path": "paper_experiments/gold/annotation_workbench/evidence/1_front__001_e_000000_00/contact_sheet.jpg",
        "artifact_type": "figure",
        "recommended_section": "gold 标注示例 / 附录",
        "quality_tier": "appendix_only",
        "paper_ready": "yes",
        "evidence_basis": "真实证据包 contact sheet",
        "notes": "适合说明 gold 工作台与证据包流程。",
    },
    {
        "artifact_id": "demo_index_html",
        "path": "server/templates/index.html",
        "artifact_type": "html_page",
        "recommended_section": "系统展示",
        "quality_tier": "engineering_only",
        "paper_ready": "no",
        "evidence_basis": "FastAPI 主页面模板",
        "notes": "不是论文图，但能支持本地服务录屏与系统展示。",
    },
    {
        "artifact_id": "demo_dashboard_html",
        "path": "docs/paper_v2_dashboard.html",
        "artifact_type": "html_page",
        "recommended_section": "系统展示",
        "quality_tier": "engineering_only",
        "paper_ready": "no",
        "evidence_basis": "paper dashboard 原型",
        "notes": "需要人工截图后再入论文。",
    },
]


RELATED_WORK_ROWS: List[Dict[str, Any]] = [
    {
        "paper_id": "sensors-2023-improved-yolov8",
        "title": "Student Behavior Detection in the Classroom Based on Improved YOLOv8",
        "year": 2023,
        "venue": "Sensors",
        "primary_source_url": "https://www.mdpi.com/1424-8220/23/20/8385",
        "task": "课堂学生行为检测",
        "modality": "visual-only",
        "dataset": "customized classroom student behavior dataset",
        "core_metrics": "mAP@0.5",
        "reliability_metrics": "",
        "innovation": "改进 YOLOv8 主干/特征融合与检测头，提升复杂课堂检测性能。",
        "borrowable_metrics": "mAP@0.5",
        "direct_compare_level": "partial",
        "comparison_role": "visual baseline",
        "notes": "适合作为视觉-only 检测基线；不提供校准指标。",
    },
    {
        "paper_id": "systems-2023-deformable-detr-swin",
        "title": "Students' Classroom Behavior Detection System Incorporating Deformable DETR with Swin Transformer and Light-Weight FPN",
        "year": 2023,
        "venue": "Systems",
        "primary_source_url": "https://www.mdpi.com/2079-8954/11/7/372",
        "task": "课堂行为检测系统",
        "modality": "visual-only",
        "dataset": "ClaBehavior",
        "core_metrics": "AP / mAP / FLOPs",
        "reliability_metrics": "",
        "innovation": "Deformable DETR + Swin Transformer + light-weight FPN 的系统化检测框架。",
        "borrowable_metrics": "AP; mAP; FLOPs",
        "direct_compare_level": "partial",
        "comparison_role": "visual system baseline",
        "notes": "更接近视觉检测系统，不覆盖跨模态验证。",
    },
    {
        "paper_id": "applsci-2023-time-series-images",
        "title": "Research on Students' Action Behavior Recognition Method Based on Classroom Time-Series Images",
        "year": 2023,
        "venue": "Applied Sciences",
        "primary_source_url": "https://www.mdpi.com/2076-3417/13/18/10426",
        "task": "课堂时序动作识别",
        "modality": "visual temporal",
        "dataset": "classroom time-series image setting",
        "core_metrics": "Accuracy / mAP",
        "reliability_metrics": "",
        "innovation": "围绕时间序列图像的动作识别改进与时序建模。",
        "borrowable_metrics": "Accuracy; mAP",
        "direct_compare_level": "partial",
        "comparison_role": "temporal visual baseline",
        "notes": "可借鉴时序动作表述，但不具备文本验证链路。",
    },
    {
        "paper_id": "frontiers-2023-multiview-multiscale",
        "title": "Multi-view and Multi-scale Behavior Recognition Algorithm Based on Attention Mechanism",
        "year": 2023,
        "venue": "Frontiers in Neurorobotics",
        "primary_source_url": "https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2023.1163245/full",
        "task": "多视角课堂行为识别",
        "modality": "visual multi-view",
        "dataset": "EuClass / IsoGD",
        "core_metrics": "Recognition accuracy",
        "reliability_metrics": "",
        "innovation": "多视角、多尺度、注意力机制行为识别。",
        "borrowable_metrics": "Recognition accuracy",
        "direct_compare_level": "concept_only",
        "comparison_role": "future extension reference",
        "notes": "仓库当前不是多视角联合推理，宜作扩展方向参考。",
    },
    {
        "paper_id": "applsci-2024-classroom-deep-learning",
        "title": "Student Behavior Recognition in Classroom Based on Deep Learning",
        "year": 2024,
        "venue": "Applied Sciences",
        "primary_source_url": "https://www.mdpi.com/2076-3417/14/17/7981",
        "task": "课堂行为识别",
        "modality": "visual-only",
        "dataset": "SBD",
        "core_metrics": "Precision / Recall / F1 / mAP",
        "reliability_metrics": "",
        "innovation": "自建 SBD 数据集，ResNet18 + LSTM + attention。",
        "borrowable_metrics": "Precision; Recall; F1; mAP",
        "direct_compare_level": "partial",
        "comparison_role": "visual recognition baseline",
        "notes": "与仓库的 behavior_det 分支最接近。",
    },
    {
        "paper_id": "srep-2025-improved-yolov8s-realtime",
        "title": "Real-time Classroom Student Behavior Detection Based on Improved YOLOv8s",
        "year": 2025,
        "venue": "Scientific Reports",
        "primary_source_url": "https://www.nature.com/articles/s41598-025-99243-x",
        "task": "实时课堂行为检测",
        "modality": "visual-only",
        "dataset": "SCB-Dataset3-S",
        "core_metrics": "Precision / Recall / AP / FPS",
        "reliability_metrics": "",
        "innovation": "面向实时检测的改进 YOLOv8s 结构。",
        "borrowable_metrics": "Precision; Recall; AP; FPS",
        "direct_compare_level": "partial",
        "comparison_role": "realtime visual baseline",
        "notes": "适合支撑实时性指标口径，但不覆盖跨模态可靠性。",
    },
    {
        "paper_id": "srep-2025-wad-yolov8",
        "title": "A WAD-YOLOv8-based Method for Classroom Student Behavior Detection",
        "year": 2025,
        "venue": "Scientific Reports",
        "primary_source_url": "https://www.nature.com/articles/s41598-025-87661-w",
        "task": "课堂学生行为检测",
        "modality": "visual-only",
        "dataset": "classroom student behavior dataset",
        "core_metrics": "mAP@0.5 / mAP@0.5:0.95 / FPS",
        "reliability_metrics": "",
        "innovation": "面向密集遮挡课堂场景的 WAD-YOLOv8 改进。",
        "borrowable_metrics": "mAP@0.5; mAP@0.5:0.95; FPS",
        "direct_compare_level": "partial",
        "comparison_role": "visual detection baseline",
        "notes": "适合对齐视觉检测类三线表。",
    },
    {
        "paper_id": "cvpr-2023-mccl",
        "title": "Multiclass Confidence and Localization Calibration for Object Detection",
        "year": 2023,
        "venue": "CVPR",
        "primary_source_url": "https://openaccess.thecvf.com/content/CVPR2023/html/Pathiraja_Multiclass_Confidence_and_Localization_Calibration_for_Object_Detection_CVPR_2023_paper.html",
        "task": "目标检测置信度与定位联合校准",
        "modality": "visual calibration",
        "dataset": "object detection benchmarks",
        "core_metrics": "Detection calibration error / localization calibration / mAP",
        "reliability_metrics": "ECE-style calibration metrics",
        "innovation": "把多类别置信度与定位误差联合纳入检测校准。",
        "borrowable_metrics": "ECE-style calibration; reliability diagram; localization-aware calibration",
        "direct_compare_level": "concept_only",
        "comparison_role": "calibration metric source",
        "notes": "适合作为仓库 ECE/Brier 之外的校准指标参考来源。",
    },
    {
        "paper_id": "ijcv-2025-uncertainty-calibration-detectors",
        "title": "A Systematic Evaluation of Uncertainty Calibration in Pre-trained Object Detectors",
        "year": 2025,
        "venue": "IJCV",
        "primary_source_url": "https://link.springer.com/article/10.1007/s11263-024-02219-z",
        "task": "预训练检测器不确定性校准评估",
        "modality": "visual calibration",
        "dataset": "object detection benchmarks",
        "core_metrics": "mAP / NLL / entropy",
        "reliability_metrics": "Brier / TCE / MCE / dMCE",
        "innovation": "系统比较检测器不确定性与多种校准误差指标。",
        "borrowable_metrics": "NLL; Brier; TCE; MCE; dMCE; entropy",
        "direct_compare_level": "concept_only",
        "comparison_role": "reliability metric source",
        "notes": "最适合补充仓库当前仅有 ECE/Brier 的可靠性维度。",
    },
    {
        "paper_id": "coling-2025-mllm-calibration",
        "title": "Unveiling Uncertainty: A Deep Dive into Calibration and Performance of Multimodal Large Language Models",
        "year": 2025,
        "venue": "COLING",
        "primary_source_url": "https://aclanthology.org/2025.coling-main.208/",
        "task": "多模态大模型校准与自我不确定性评估",
        "modality": "multimodal calibration",
        "dataset": "IDK benchmark and multimodal evaluation tasks",
        "core_metrics": "task accuracy / abstention behavior",
        "reliability_metrics": "temperature scaling / prompt optimization / calibration analysis",
        "innovation": "把自我不确定性表达与多模态模型校准放到统一评估框架。",
        "borrowable_metrics": "abstention / IDK rate; temperature scaling; prompt-based calibration",
        "direct_compare_level": "concept_only",
        "comparison_role": "uncertain / abstention metric reference",
        "notes": "最适合给仓库的 uncertain 机制补充更前沿的评估口径。",
    },
]


LOCAL_REPORT_ALIGNMENT_ROWS: List[Dict[str, Any]] = [
    {
        "claim": "核心主线（pose -> track/UQ -> action -> ASR -> query -> align -> dual verify -> timeline）",
        "research_report_position": "已实现",
        "local_workspace_status": "已实现且有本地结果产物",
        "evidence": "run_pipeline.py; classroom_pipeline/pipeline/07_dual_verification.py; classroom_pipeline/pipeline/10_visualize_timeline.py; paper_experiments/real_cases/random6_20260420_mainline_v3_summary.json",
        "recommendation": "可写入方法链路与系统流程，但不要写成新 backbone。",
    },
    {
        "claim": "03d_train_track_variance_head / 05c / 05d / 06c / 15_run_paper_experiment / docs/js/data_source.js 在远程分支未确认",
        "research_report_position": "本地已有但远程抓取未确认",
        "local_workspace_status": "本地已存在",
        "evidence": "classroom_pipeline/verifier/variance_head.py; classroom_pipeline/pipeline/05c_behavior_det_to_actions.py; classroom_pipeline/pipeline/05d_merge_action_sources.py; classroom_pipeline/pipeline/06c_asr_openai_to_jsonl.py; classroom_pipeline/orchestrator_legacy.py; docs/js/data_source.js",
        "recommendation": "说明本地工作区已明显超过当时的远程可见状态。",
    },
    {
        "claim": "behavior_det 增强分支",
        "research_report_position": "仅弱确认/待实验",
        "local_workspace_status": "本地稳定运行 30/30（behavior_aug_v1）",
        "evidence": "paper_experiments/real_cases/random6_20260420_behavior_aug_v1_summary.json; classroom_pipeline/pipeline/05c_behavior_det_to_actions.py",
        "recommendation": "可以写为本地增强分支，但结果暂不当主结论。",
    },
    {
        "claim": "OpenAI / API ASR 多后端",
        "research_report_position": "远程分支未完整确认",
        "local_workspace_status": "本地具备 DashScope + OpenAI + Whisper 脚本和消融日志",
        "evidence": "classroom_pipeline/pipeline/06_api_asr_realtime.py; classroom_pipeline/pipeline/06c_asr_openai_to_jsonl.py; docs/asr_*.md; paper_experiments/run_logs/asr_ablation_*.csv",
        "recommendation": "可以写为后端消融与工程扩展，不宜写成最终性能创新。",
    },
    {
        "claim": "OCR / 敏感文本 / 黑板文字理解 / 幻灯片文字事件抽取",
        "research_report_position": "不能写成已实现",
        "local_workspace_status": "仍未形成主线证据",
        "evidence": "本地主线与实验分支未发现对应稳定产物目录或统一实验指标文件",
        "recommendation": "保持为 future work。",
    },
    {
        "claim": "完整、稳定、可部署的前后端系统",
        "research_report_position": "不能写成已完成",
        "local_workspace_status": "存在 FastAPI 原型与 Pages demo，但不等于生产系统",
        "evidence": "server/app.py; server/build_frontend_data_bundle.py; docs/paper_v2_dashboard.html",
        "recommendation": "可写为展示原型/录屏支撑系统。",
    },
    {
        "claim": "大规模真实 held-out gold 主结果",
        "research_report_position": "缺失或不足",
        "local_workspace_status": "已有 gold base/wave1，wave2/wave3 仍 draft，不能当大规模最终结果",
        "evidence": "paper_experiments/gold/README.md; paper_experiments/gold/gold_validation_report.json; mainline_branch_merged_summary.json",
        "recommendation": "保守写为 gold 体系已建立但尚未完全冻结。",
    },
]


MAINLINE_RUN_CASE_PATHS: Dict[str, str] = {
    "random6_20260420_baseline": "paper_experiments/run_logs/random6_20260420_case_metrics.csv",
    "random6_20260420_mainline_v3": "paper_experiments/real_cases/random6_20260420_mainline_v3_case_metrics.csv",
    "random6_20260420_behavior_aug_v1": "paper_experiments/real_cases/random6_20260420_behavior_aug_v1_case_metrics.csv",
    "real_exp_a_uq_alignment_20260421_yolo11x": "paper_experiments/real_cases/real_exp_a_uq_alignment_20260421_yolo11x_case_metrics.csv",
    "real_exp_b_reliability_calibration_20260421_yolo11x": "paper_experiments/real_cases/real_exp_b_reliability_calibration_20260421_yolo11x_case_metrics.csv",
}


MAINLINE_ROLE_MAP = {
    "random6_20260420_baseline": "baseline",
    "random6_20260420_mainline_v3": "stable_mainline",
    "random6_20260420_behavior_aug_v1": "behavior_augmented",
    "real_exp_a_uq_alignment_20260421_yolo11x": "real_exp_a",
    "real_exp_b_reliability_calibration_20260421_yolo11x": "real_exp_b",
}


def build_script_evidence_index(script_index_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    purpose_map = {row["script_path"]: row for row in script_index_rows if row.get("script_path")}
    out: List[Dict[str, Any]] = []
    for spec in SCRIPT_CATEGORY_SPECS:
        script_path = spec["script_path"]
        index_row = purpose_map.get(script_path, {})
        out.append(
            {
                "script_path": script_path,
                "exists": "yes" if (ROOT / script_path).exists() else "no",
                "category": spec["category"],
                "subsystem": spec["subsystem"],
                "role": spec["role"],
                "purpose_from_audit": index_row.get("purpose", ""),
                "paper_related_from_audit": index_row.get("paper_related", ""),
                "audit_reason": index_row.get("reason", ""),
                "primary_outputs": spec["primary_outputs"],
                "inspection_note": spec["inspection_note"],
                "include_in_primary_table": spec["include_in_primary_table"],
            }
        )
    return out


def pick_run(summary_rows: Sequence[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
    for row in summary_rows:
        if str(row.get("run_id", "")).strip() == run_id:
            return row
    raise KeyError(f"run_id not found: {run_id}")


def summarize_case_metrics(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    total_cases = len(rows)
    ok_cases = sum(1 for row in rows if str(row.get("status", "")).strip().lower() == "ok")
    attention_cases = sum(1 for row in rows if str(row.get("attention", "")).strip())
    return {
        "case_rows": total_cases,
        "ok_cases_from_case_metrics": ok_cases,
        "attention_cases_from_case_metrics": attention_cases,
        "mean_elapsed_sec": round(mean_from_rows(rows, "elapsed_sec"), 4),
        "mean_pose_track_count": round(mean_from_rows(rows, "pose_track_count"), 4),
        "mean_actions_count": round(mean_from_rows(rows, "actions_count"), 4),
        "mean_objects_det_count": round(mean_from_rows(rows, "objects_det_count"), 4),
        "transcript_placeholder_cases": count_rows(rows, "transcript_placeholder"),
        "transcript_placeholder_ratio": round(
            count_rows(rows, "transcript_placeholder") / max(1, total_cases), 4
        ),
        "mean_event_queries_count": round(mean_from_rows(rows, "event_queries_count"), 4),
        "mean_align_events": round(mean_from_rows(rows, "align_events"), 4),
        "mean_align_avg_candidates": round(mean_from_rows(rows, "align_avg_candidates"), 4),
        "mean_verified_count": round(mean_from_rows(rows, "verified_count"), 4),
        "mean_verified_p_match_mean": round(mean_from_rows(rows, "verified_p_match_mean"), 4),
        "sum_verified_count": round(sum_from_rows(rows, "verified_count"), 4),
    }


def build_mainline_table(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_id in [
        "random6_20260420_baseline",
        "random6_20260420_mainline_v3",
        "random6_20260420_behavior_aug_v1",
    ]:
        summary = pick_run(summary_rows, run_id)
        case_path = resolve_path(MAINLINE_RUN_CASE_PATHS[run_id])
        case_rows = read_csv_rows(case_path)
        metrics = summarize_case_metrics(case_rows)
        rows.append(
            {
                "run_id": run_id,
                "role": MAINLINE_ROLE_MAP[run_id],
                "evidence_tier": "appendix_only",
                "paper_use_recommendation": "system_metrics_only",
                "source_summary_path": summary.get("summary_path", ""),
                "source_case_metrics_path": str(MAINLINE_RUN_CASE_PATHS[run_id]).replace("\\", "/"),
                "total_cases": summary.get("total", ""),
                "ok_cases": summary.get("ok", ""),
                "failed_cases": summary.get("failed", ""),
                "elapsed_sec_total": summary.get("elapsed_sec", ""),
                "tracks_sum": summary.get("tracks_sum", ""),
                "actions_sum": summary.get("actions_sum", ""),
                "event_queries_sum": summary.get("event_queries_sum", ""),
                "verified_sum": summary.get("verified_sum", ""),
                "attention_cases": summary.get("attention_cases", ""),
                "asr_placeholder_cases": summary.get("asr_placeholder_cases", ""),
                "action_mode": summary.get("action_mode", ""),
                "asr_backend": summary.get("asr_backend", ""),
                "pose_model": summary.get("pose_model", ""),
                "det_model": summary.get("det_model", ""),
                "enable_object_evidence": summary.get("enable_object_evidence", ""),
                "enable_behavior_det": summary.get("enable_behavior_det", ""),
                "behavior_action_mode": summary.get("behavior_action_mode", ""),
                **metrics,
            }
        )
    return rows


def build_real_exp_table(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_id in [
        "real_exp_a_uq_alignment_20260421_yolo11x",
        "real_exp_b_reliability_calibration_20260421_yolo11x",
    ]:
        summary = pick_run(summary_rows, run_id)
        case_path = resolve_path(MAINLINE_RUN_CASE_PATHS[run_id])
        case_rows = read_csv_rows(case_path)
        metrics = summarize_case_metrics(case_rows)
        rows.append(
            {
                "run_id": run_id,
                "role": MAINLINE_ROLE_MAP[run_id],
                "evidence_tier": "appendix_only",
                "paper_use_recommendation": "candidate_after_gold_freeze",
                "source_summary_path": summary.get("summary_path", ""),
                "source_case_metrics_path": str(MAINLINE_RUN_CASE_PATHS[run_id]).replace("\\", "/"),
                "total_cases": summary.get("total", ""),
                "ok_cases": summary.get("ok", ""),
                "failed_cases": summary.get("failed", ""),
                "elapsed_sec_total": summary.get("elapsed_sec", ""),
                "tracks_sum": summary.get("tracks_sum", ""),
                "actions_sum": summary.get("actions_sum", ""),
                "event_queries_sum": summary.get("event_queries_sum", ""),
                "verified_sum": summary.get("verified_sum", ""),
                "attention_cases": summary.get("attention_cases", ""),
                "asr_placeholder_cases": summary.get("asr_placeholder_cases", ""),
                "action_mode": summary.get("action_mode", ""),
                "asr_backend": summary.get("asr_backend", ""),
                "pose_model": summary.get("pose_model", ""),
                "det_model": summary.get("det_model", ""),
                "enable_object_evidence": summary.get("enable_object_evidence", ""),
                "enable_behavior_det": summary.get("enable_behavior_det", ""),
                "behavior_action_mode": summary.get("behavior_action_mode", ""),
                **metrics,
            }
        )
    return rows


def engineering_tier_for(data_mode: str, notes: Iterable[str]) -> str:
    data_mode_l = str(data_mode).lower()
    joined_notes = " ".join(str(x).lower() for x in notes)
    if any(token in data_mode_l for token in ("sample", "weak", "smoke")):
        return "engineering_only"
    if any(token in joined_notes for token in ("weak_self_labels", "sample", "smoke", "fallback")):
        return "engineering_only"
    return "appendix_only"


def build_branch_summary_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    baseline = read_json(resolve_path("output/paper_experiments/baseline/metrics.json"))
    baseline_metrics = baseline.get("metrics", {})
    heuristic = baseline_metrics.get("baselines", {}).get("heuristic_dual_verification", {})
    rows.append(
        {
            "experiment_id": "baseline",
            "objective": baseline.get("objective", ""),
            "data_mode": baseline.get("data_mode", ""),
            "evidence_tier": engineering_tier_for(baseline.get("data_mode", ""), baseline.get("notes", [])),
            "paper_use_recommendation": "engineering_baseline_only",
            "source_path": "output/paper_experiments/baseline/metrics.json",
            "best_setting_or_variant": "heuristic_dual_verification",
            "precision": "",
            "recall": "",
            "f1": "",
            "accuracy": "",
            "auroc": "",
            "ece": "",
            "brier": "",
            "flip_count": "",
            "beneficial_flip_rate": "",
            "harmful_flip_rate": "",
            "semantic_hit_rate_top1": heuristic.get("semantic_top1_hit_rate", ""),
            "semantic_hit_rate_topk": "",
            "confusion_summary": "",
            "notes": "pipeline_check.can_run_to_verified_events="
            + str(baseline_metrics.get("pipeline_check", {}).get("can_run_to_verified_events", "")),
        }
    )

    exp_a = read_json(resolve_path("output/paper_experiments/exp_a_uq_align/metrics.json"))
    exp_a_fixed = exp_a.get("baseline_comparison", {}).get("fixed", {})
    exp_a_adaptive = exp_a.get("baseline_comparison", {}).get("adaptive_uq", {})
    rows.append(
        {
            "experiment_id": "exp_a_uq_align",
            "objective": exp_a.get("objective", ""),
            "data_mode": exp_a.get("data_mode", ""),
            "evidence_tier": engineering_tier_for(exp_a.get("data_mode", ""), exp_a.get("notes", [])),
            "paper_use_recommendation": "candidate_after_gold_freeze",
            "source_path": "output/paper_experiments/exp_a_uq_align/metrics.json",
            "best_setting_or_variant": "adaptive_uq_vs_fixed",
            "precision": exp_a_adaptive.get("alignment_precision", ""),
            "recall": exp_a_adaptive.get("alignment_recall_at_1", ""),
            "f1": "",
            "accuracy": "",
            "auroc": "",
            "ece": "",
            "brier": "",
            "flip_count": "",
            "beneficial_flip_rate": "",
            "harmful_flip_rate": "",
            "semantic_hit_rate_top1": "",
            "semantic_hit_rate_topk": "",
            "confusion_summary": "",
            "notes": (
                f"fixed_recall@1={exp_a_fixed.get('alignment_recall_at_1', '')}; "
                f"adaptive_recall@1={exp_a_adaptive.get('alignment_recall_at_1', '')}; "
                f"delta_overlap={exp_a.get('baseline_comparison', {}).get('delta_adaptive_minus_fixed', {}).get('mean_temporal_overlap', '')}"
            ),
        }
    )

    exp_b = read_json(resolve_path("output/paper_experiments/exp_b_reliability_calibration/metrics.json"))
    exp_b_metrics = exp_b.get("metrics", {})
    exp_b_cal = exp_b_metrics.get("calibrated_uq_gate", {})
    rows.append(
        {
            "experiment_id": "exp_b_reliability_calibration",
            "objective": exp_b.get("objective", ""),
            "data_mode": exp_b.get("data_mode", ""),
            "evidence_tier": engineering_tier_for(exp_b.get("data_mode", ""), exp_b.get("notes", [])),
            "paper_use_recommendation": "candidate_after_gold_freeze",
            "source_path": "output/paper_experiments/exp_b_reliability_calibration/metrics.json",
            "best_setting_or_variant": "calibrated_uq_gate",
            "precision": exp_b_cal.get("Precision", ""),
            "recall": exp_b_cal.get("Recall", ""),
            "f1": exp_b_cal.get("F1", ""),
            "accuracy": "",
            "auroc": "",
            "ece": exp_b_cal.get("ECE", ""),
            "brier": exp_b_cal.get("Brier", ""),
            "flip_count": "",
            "beneficial_flip_rate": "",
            "harmful_flip_rate": "",
            "semantic_hit_rate_top1": "",
            "semantic_hit_rate_topk": "",
            "confusion_summary": "",
            "notes": (
                f"delta_ece_calibrated_vs_uq_gate={exp_b_metrics.get('delta_ece_calibrated_vs_uq_gate', '')}; "
                f"delta_brier_calibrated_vs_uq_gate={exp_b_metrics.get('delta_brier_calibrated_vs_uq_gate', '')}"
            ),
        }
    )

    exp_c_rows = read_csv_rows(resolve_path("output/paper_experiments/exp_c_negative_sampling/metrics_compare.csv"))
    best_exp_c = max(exp_c_rows, key=lambda row: safe_float(row.get("F1")) or 0.0)
    rows.append(
        {
            "experiment_id": "exp_c_negative_sampling",
            "objective": "Compare positive_only / +temporal_shift / +semantic_mismatch / +both.",
            "data_mode": "sample",
            "evidence_tier": "engineering_only",
            "paper_use_recommendation": "appendix_ablation_only",
            "source_path": "output/paper_experiments/exp_c_negative_sampling/metrics_compare.csv",
            "best_setting_or_variant": best_exp_c.get("setting", ""),
            "precision": best_exp_c.get("Precision", ""),
            "recall": best_exp_c.get("Recall", ""),
            "f1": best_exp_c.get("F1", ""),
            "accuracy": "",
            "auroc": best_exp_c.get("AUROC", ""),
            "ece": best_exp_c.get("ECE", ""),
            "brier": best_exp_c.get("Brier", ""),
            "flip_count": "",
            "beneficial_flip_rate": "",
            "harmful_flip_rate": "",
            "semantic_hit_rate_top1": "",
            "semantic_hit_rate_topk": "",
            "confusion_summary": "",
            "notes": "best_by_f1 derived from metrics_compare.csv",
        }
    )

    exp_d = read_json(resolve_path("output/paper_experiments/exp_d_semantic_embedding/metrics.json"))
    exp_d_metrics = exp_d.get("metrics", {})
    best_mode = max(
        ("rule", "embedding", "hybrid"),
        key=lambda mode: safe_float(exp_d_metrics.get(mode, {}).get("F1")) or 0.0,
    )
    best_mode_metrics = exp_d_metrics.get(best_mode, {})
    rows.append(
        {
            "experiment_id": "exp_d_semantic_embedding",
            "objective": exp_d.get("objective", ""),
            "data_mode": exp_d.get("data_mode", ""),
            "evidence_tier": engineering_tier_for(exp_d.get("data_mode", ""), exp_d.get("notes", [])),
            "paper_use_recommendation": "future_work_or_appendix",
            "source_path": "output/paper_experiments/exp_d_semantic_embedding/metrics.json",
            "best_setting_or_variant": best_mode,
            "precision": "",
            "recall": "",
            "f1": best_mode_metrics.get("F1", ""),
            "accuracy": "",
            "auroc": "",
            "ece": best_mode_metrics.get("ECE", ""),
            "brier": best_mode_metrics.get("Brier", ""),
            "flip_count": "",
            "beneficial_flip_rate": "",
            "harmful_flip_rate": "",
            "semantic_hit_rate_top1": best_mode_metrics.get("semantic_mismatch_detection_rate", ""),
            "semantic_hit_rate_topk": "",
            "confusion_summary": "",
            "notes": "embedding/hybrid currently backed by hashing_fallback in this smoke run.",
        }
    )

    exp_e = read_json(resolve_path("output/paper_experiments/exp_e_object_evidence/metrics.json"))
    exp_e_metrics = exp_e.get("metrics", {})
    rows.append(
        {
            "experiment_id": "exp_e_object_evidence",
            "objective": exp_e.get("objective", ""),
            "data_mode": exp_e.get("data_mode", ""),
            "evidence_tier": engineering_tier_for(exp_e.get("data_mode", ""), exp_e.get("notes", [])),
            "paper_use_recommendation": "appendix_only",
            "source_path": "output/paper_experiments/exp_e_object_evidence/metrics.json",
            "best_setting_or_variant": "action_plus_object_evidence",
            "precision": exp_e_metrics.get("action_plus_object_evidence", {}).get("ambiguity_subset_precision", ""),
            "recall": exp_e_metrics.get("action_plus_object_evidence", {}).get("ambiguity_subset_recall", ""),
            "f1": exp_e_metrics.get("action_plus_object_evidence", {}).get("ambiguity_subset_f1", ""),
            "accuracy": "",
            "auroc": "",
            "ece": "",
            "brier": "",
            "flip_count": exp_e_metrics.get("flip_count", ""),
            "beneficial_flip_rate": exp_e_metrics.get("beneficial_flip_rate", ""),
            "harmful_flip_rate": exp_e_metrics.get("harmful_flip_rate", ""),
            "semantic_hit_rate_top1": "",
            "semantic_hit_rate_topk": "",
            "confusion_summary": "",
            "notes": f"ambiguity_subset_size={exp_e_metrics.get('ambiguity_subset_size', '')}",
        }
    )

    exp_object = read_json(resolve_path("output/paper_experiments/exp_object_action_fusion/metrics.json"))
    exp_object_metrics = exp_object.get("metrics", {})
    rows.append(
        {
            "experiment_id": "exp_object_action_fusion",
            "objective": exp_object.get("objective", ""),
            "data_mode": exp_object.get("data_mode", ""),
            "evidence_tier": engineering_tier_for(exp_object.get("data_mode", ""), exp_object.get("notes", [])),
            "paper_use_recommendation": "engineering_only",
            "source_path": "output/paper_experiments/exp_object_action_fusion/metrics.json",
            "best_setting_or_variant": "object_fusion",
            "precision": "",
            "recall": "",
            "f1": "",
            "accuracy": exp_object_metrics.get("pseudo_gt_accuracy_after", ""),
            "auroc": "",
            "ece": "",
            "brier": "",
            "flip_count": exp_object_metrics.get("label_changed_count", ""),
            "beneficial_flip_rate": "",
            "harmful_flip_rate": "",
            "semantic_hit_rate_top1": "",
            "semantic_hit_rate_topk": "",
            "confusion_summary": "",
            "notes": (
                f"pseudo_gt_accuracy_before={exp_object_metrics.get('pseudo_gt_accuracy_before', '')}; "
                f"mean_confidence_delta={exp_object_metrics.get('mean_confidence_delta', '')}"
            ),
        }
    )

    exp_uq = read_json(resolve_path("output/paper_experiments/exp_uq_adaptive_alignment/metrics.json"))
    exp_uq_metrics = exp_uq.get("metrics", {})
    rows.append(
        {
            "experiment_id": "exp_uq_adaptive_alignment",
            "objective": exp_uq.get("objective", ""),
            "data_mode": exp_uq.get("data_mode", ""),
            "evidence_tier": engineering_tier_for(exp_uq.get("data_mode", ""), exp_uq.get("notes", [])),
            "paper_use_recommendation": "engineering_only",
            "source_path": "output/paper_experiments/exp_uq_adaptive_alignment/metrics.json",
            "best_setting_or_variant": "adaptive_alignment",
            "precision": "",
            "recall": exp_uq_metrics.get("semantic_hit_rate_top1", ""),
            "f1": "",
            "accuracy": "",
            "auroc": "",
            "ece": "",
            "brier": "",
            "flip_count": "",
            "beneficial_flip_rate": "",
            "harmful_flip_rate": "",
            "semantic_hit_rate_top1": exp_uq_metrics.get("semantic_hit_rate_top1", ""),
            "semantic_hit_rate_topk": exp_uq_metrics.get("semantic_hit_rate_topk", ""),
            "confusion_summary": "",
            "notes": f"avg_window_size_sec={exp_uq_metrics.get('avg_window_size_sec', '')}",
        }
    )

    exp_dual = read_json(resolve_path("output/paper_experiments/exp_dual_verifier_reliability/metrics.json"))
    exp_dual_eval = read_json(resolve_path("output/paper_experiments/exp_dual_verifier_reliability/verifier_eval_report.json"))
    exp_dual_metrics = exp_dual.get("metrics", {})
    confusion = exp_dual_eval.get("confusion_matrix", {}).get("matrix", [])
    confusion_summary = ""
    if isinstance(confusion, list) and len(confusion) == 3:
        confusion_summary = f"match={confusion[0]}; uncertain={confusion[1]}; mismatch={confusion[2]}"
    rows.append(
        {
            "experiment_id": "exp_dual_verifier_reliability",
            "objective": exp_dual.get("objective", ""),
            "data_mode": exp_dual.get("data_mode", ""),
            "evidence_tier": engineering_tier_for(exp_dual.get("data_mode", ""), exp_dual.get("notes", [])),
            "paper_use_recommendation": "engineering_only",
            "source_path": "output/paper_experiments/exp_dual_verifier_reliability/metrics.json",
            "best_setting_or_variant": "dual_verifier_smoke",
            "precision": exp_dual_metrics.get("precision", ""),
            "recall": exp_dual_metrics.get("recall", ""),
            "f1": exp_dual_metrics.get("f1", ""),
            "accuracy": exp_dual_metrics.get("accuracy", ""),
            "auroc": "",
            "ece": exp_dual_metrics.get("ece", ""),
            "brier": exp_dual_metrics.get("brier", ""),
            "flip_count": "",
            "beneficial_flip_rate": "",
            "harmful_flip_rate": "",
            "semantic_hit_rate_top1": "",
            "semantic_hit_rate_topk": "",
            "confusion_summary": confusion_summary,
            "notes": "reference_quality=weak_self_labels",
        }
    )

    rows.append(
        {
            "experiment_id": "exp_f_pages_demo",
            "objective": "Build static Pages demo for appendix/system showcase.",
            "data_mode": "demo_system",
            "evidence_tier": "engineering_only",
            "paper_use_recommendation": "system_demo_only",
            "source_path": "output/paper_experiments/exp_f_pages_demo/handoff_summary.json",
            "best_setting_or_variant": "pages_demo",
            "precision": "",
            "recall": "",
            "f1": "",
            "accuracy": "",
            "auroc": "",
            "ece": "",
            "brier": "",
            "flip_count": "",
            "beneficial_flip_rate": "",
            "harmful_flip_rate": "",
            "semantic_hit_rate_top1": "",
            "semantic_hit_rate_topk": "",
            "confusion_summary": "",
            "notes": "No formal metrics.json; treat strictly as demo system branch.",
        }
    )

    return rows


def build_figure_manifest() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in FIGURE_SPECS:
        rel_path = str(spec["path"]).replace("\\", "/")
        exists = path_exists(rel_path)
        rows.append(
            {
                "artifact_id": spec["artifact_id"],
                "artifact_type": spec["artifact_type"],
                "path": rel_path,
                "exists": "yes" if exists else "no",
                "recommended_section": spec["recommended_section"],
                "quality_tier": spec["quality_tier"],
                "paper_ready": spec["paper_ready"],
                "evidence_basis": spec["evidence_basis"],
                "notes": spec["notes"],
            }
        )
    return rows


def build_file_scope_summary(
    scan_summary: Dict[str, Any],
    script_evidence_rows: Sequence[Dict[str, Any]],
    deep_report_text: str,
) -> Dict[str, Any]:
    categories: Dict[str, int] = {}
    for row in script_evidence_rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    remote_missing_now_local = [
        item["claim"]
        for item in LOCAL_REPORT_ALIGNMENT_ROWS
        if "本地已存在" in item["local_workspace_status"] or "本地具备" in item["local_workspace_status"]
    ]
    moved_paths = {
        "scripts/03d_train_track_variance_head.py": "classroom_pipeline/verifier/variance_head.py",
        "scripts/05c_behavior_det_to_actions.py": "classroom_pipeline/pipeline/05c_behavior_det_to_actions.py",
        "scripts/05d_merge_action_sources.py": "classroom_pipeline/pipeline/05d_merge_action_sources.py",
        "scripts/06c_asr_openai_to_jsonl.py": "classroom_pipeline/pipeline/06c_asr_openai_to_jsonl.py",
        "scripts/15_run_paper_experiment.py": "classroom_pipeline/orchestrator_legacy.py",
        "docs/js/data_source.js": "docs/js/data_source.js",
    }
    missing_file_mentions = [
        current
        for reported, current in moved_paths.items()
        if reported in deep_report_text and (ROOT / current).exists()
    ]
    return {
        "generated_at": now_iso(),
        "root": str(ROOT),
        "deep_report_path": str(DEEP_REPORT_PATH),
        "docs_alias_assumption": "user-mentioned does folder is treated as docs/",
        "full_traversal": {
            "total_files": scan_summary.get("total_files", 0),
            "category_counts": scan_summary.get("category_counts", {}),
            "top_dir_counts": scan_summary.get("top_dir_counts", {}),
            "inventory_csv": scan_summary.get("inventory_csv", ""),
        },
        "selected_script_categories": categories,
        "selected_script_count": len(script_evidence_rows),
        "spot_checked_subsystems": sorted({row["subsystem"] for row in script_evidence_rows}),
        "remote_missing_report_items_now_present_locally": remote_missing_now_local,
        "deep_report_missing_file_mentions_now_present_locally": missing_file_mentions,
    }


def validate_outputs(
    *,
    mainline_table: Sequence[Dict[str, Any]],
    real_exp_table: Sequence[Dict[str, Any]],
    branch_table: Sequence[Dict[str, Any]],
    figure_manifest: Sequence[Dict[str, Any]],
    script_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    matrix_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    matrix_map = {row.get("run_id", ""): row for row in matrix_rows}
    failures = 0

    for table_name, rows in [("tbl_mainline_vs_behavior", mainline_table), ("tbl_real_pilot_exp_ab", real_exp_table)]:
        for row in rows:
            run_id = row["run_id"]
            summary = pick_run(summary_rows, run_id)
            matrix = matrix_map.get(run_id, {})
            ok = str(row["total_cases"]) == str(summary.get("total", ""))
            ok = ok and str(row["ok_cases"]) == str(summary.get("ok", ""))
            ok = ok and str(row["failed_cases"]) == str(summary.get("failed", ""))
            if matrix:
                ok = ok and str(row["total_cases"]) == str(matrix.get("total", ""))
                ok = ok and str(row["ok_cases"]) == str(matrix.get("ok", ""))
                ok = ok and str(row["failed_cases"]) == str(matrix.get("failed", ""))
            checks.append(
                {
                    "name": f"{table_name}:{run_id}:summary_consistency",
                    "passed": ok,
                    "details": {
                        "summary_path": summary.get("summary_path", ""),
                        "case_metrics_path": row.get("source_case_metrics_path", ""),
                    },
                }
            )
            if not ok:
                failures += 1

    for row in figure_manifest:
        passed = row["exists"] == "yes"
        checks.append(
            {
                "name": f"figure_exists:{row['artifact_id']}",
                "passed": passed,
                "details": {"path": row["path"], "quality_tier": row["quality_tier"]},
            }
        )
        if not passed:
            failures += 1

    for row in script_rows:
        passed = row["exists"] == "yes"
        checks.append(
            {
                "name": f"script_exists:{row['script_path']}",
                "passed": passed,
                "details": {"category": row["category"], "subsystem": row["subsystem"]},
            }
        )
        if not passed:
            failures += 1

    for row in branch_table:
        joined = f"{row.get('data_mode', '')} {row.get('notes', '')}".lower()
        should_be_engineering = any(token in joined for token in ("sample", "smoke", "weak", "fallback"))
        passed = not should_be_engineering or row.get("evidence_tier") == "engineering_only"
        checks.append(
            {
                "name": f"evidence_tier:{row['experiment_id']}",
                "passed": passed,
                "details": {
                    "data_mode": row.get("data_mode", ""),
                    "evidence_tier": row.get("evidence_tier", ""),
                },
            }
        )
        if not passed:
            failures += 1

    for row in RELATED_WORK_ROWS:
        url = str(row["primary_source_url"])
        passed = url.startswith("https://")
        checks.append(
            {
                "name": f"related_work_url:{row['paper_id']}",
                "passed": passed,
                "details": {"url": url},
            }
        )
        if not passed:
            failures += 1

    return {
        "generated_at": now_iso(),
        "failure_count": failures,
        "passed": failures == 0,
        "checks": checks,
    }


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    headers = [title for _, title in columns]
    sep = ["---"] * len(columns)
    body_lines = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"] + body_lines
    )


def build_report(
    *,
    scan_summary: Dict[str, Any],
    file_scope_summary: Dict[str, Any],
    mainline_rows: Sequence[Dict[str, Any]],
    real_exp_rows: Sequence[Dict[str, Any]],
    branch_rows: Sequence[Dict[str, Any]],
    figure_rows: Sequence[Dict[str, Any]],
    validation: Dict[str, Any],
) -> str:
    report_inputs = [
        f"`{DEEP_REPORT_PATH}`",
        "`docs/`",
        "`paper_experiments/real_cases/*`",
        "`output/paper_experiments/*`",
        "`docs/assets/tables/full_audit_20260422/*`",
    ]

    mainline_md = markdown_table(
        mainline_rows,
        [
            ("run_id", "run_id"),
            ("role", "角色"),
            ("total_cases", "cases"),
            ("ok_cases", "ok"),
            ("failed_cases", "failed"),
            ("mean_elapsed_sec", "mean_elapsed_sec"),
            ("mean_actions_count", "mean_actions_count"),
            ("mean_event_queries_count", "mean_event_queries_count"),
            ("mean_align_avg_candidates", "mean_align_avg_candidates"),
            ("mean_verified_p_match_mean", "mean_verified_p_match_mean"),
            ("evidence_tier", "证据等级"),
        ],
    )

    real_exp_md = markdown_table(
        real_exp_rows,
        [
            ("run_id", "run_id"),
            ("total_cases", "cases"),
            ("ok_cases", "ok"),
            ("mean_elapsed_sec", "mean_elapsed_sec"),
            ("mean_objects_det_count", "mean_objects_det_count"),
            ("mean_align_avg_candidates", "mean_align_avg_candidates"),
            ("mean_verified_p_match_mean", "mean_verified_p_match_mean"),
            ("evidence_tier", "证据等级"),
        ],
    )

    branch_preview = markdown_table(
        branch_rows,
        [
            ("experiment_id", "实验"),
            ("data_mode", "data_mode"),
            ("evidence_tier", "证据等级"),
            ("best_setting_or_variant", "主比较"),
            ("f1", "F1"),
            ("accuracy", "Accuracy"),
            ("ece", "ECE"),
            ("brier", "Brier"),
            ("paper_use_recommendation", "推荐用途"),
        ],
    )

    figure_preview = markdown_table(
        figure_rows,
        [
            ("artifact_id", "artifact"),
            ("artifact_type", "类型"),
            ("quality_tier", "等级"),
            ("paper_ready", "可直接入文"),
            ("recommended_section", "建议章节"),
            ("path", "路径"),
        ],
    )

    related_preview = markdown_table(
        RELATED_WORK_ROWS[:6],
        [
            ("title", "论文"),
            ("year", "年份"),
            ("venue", "Venue"),
            ("task", "任务"),
            ("core_metrics", "核心指标"),
            ("reliability_metrics", "可靠性指标"),
            ("comparison_role", "作用"),
        ],
    )

    local_alignment_preview = markdown_table(
        LOCAL_REPORT_ALIGNMENT_ROWS,
        [
            ("claim", "研究报告中的判断"),
            ("research_report_position", "研究报告位置"),
            ("local_workspace_status", "本地仓库状态"),
            ("recommendation", "整理结论"),
        ],
    )

    top_dir_counts = file_scope_summary["full_traversal"]["top_dir_counts"]
    top_dir_keys = ["data", "output", "paper_experiments", "docs", "scripts", "verifier", "server", "tools"]
    top_dir_lines = [
        f"- `{key}`: {top_dir_counts.get(key, 0)}"
        for key in top_dir_keys
        if key in top_dir_counts
    ]

    main_text_figs = [row for row in figure_rows if row["quality_tier"] == "main_text"]
    appendix_figs = [row for row in figure_rows if row["quality_tier"] == "appendix_only"]

    diagnostic_excluded = [
        "`random6_20260420_mainline_v2`: 30/30 failed，诊断失败样本，只保留为问题记录。",
        "`random6_20260420_trackfix`: 7-case 诊断运行，不进入论文主结果。",
        "ASR smoke / retry / hybrid recover 系列：保留为后端消融与恢复日志，不进主结果表。",
        "所有 `data_mode=sample`、`sample_or_weak_labels`、`weak_self_labels` 或 notes 含 `fallback` 的分支结果：统一降级为 `engineering_only`。",
    ]

    three_line_suggestions = [
        "- 表 A（系统/主线对照）: `tbl_mainline_vs_behavior.csv`。推荐列：`run_id`、`mean_elapsed_sec`、`mean_pose_track_count`、`mean_actions_count`、`mean_event_queries_count`、`mean_align_avg_candidates`、`mean_verified_p_match_mean`。用途：系统流程差异，不写成最终性能结论。",
        "- 表 B（真实样本实验）: `tbl_real_pilot_exp_ab.csv`。推荐列：`run_id`、`mean_elapsed_sec`、`mean_objects_det_count`、`mean_align_avg_candidates`、`mean_verified_p_match_mean`。用途：real pilot 现状对照，等待 gold freeze 后再升级为主结论。",
        "- 表 C（分支消融/校准）: `tbl_branch_ablation_summary.csv`。推荐列：`experiment_id`、`best_setting_or_variant`、`F1`、`Accuracy`、`AUROC`、`ECE`、`Brier`、`flip_count`、`semantic_hit_rate_top1/topk`。用途：附录或候选主结果。",
        "- 表 D（相关工作指标矩阵）: `tbl_related_work_metrics.csv`。推荐列：`title`、`task`、`modality`、`dataset`、`core_metrics`、`reliability_metrics`、`borrowable_metrics`、`comparison_role`。",
    ]

    borrowable_metrics = [
        "- 视觉检测/识别类：`mAP@0.5`、`mAP@0.5:0.95`、`Precision`、`Recall`、`F1`、`Accuracy`、`FPS`、`FLOPs`。",
        "- 可靠性/校准类：`ECE`、`Brier`、`NLL`、`TCE`、`MCE`、`dMCE`、`entropy`、`reliability diagram`。",
        "- uncertain / 自我不确定性类：`abstention rate`、`IDK rate`、temperature scaling 前后差值。",
        "- 仓库当前最能对齐的主文指标仍是：`Precision/Recall/F1` 与 `ECE/Brier` 的并列报告。",
    ]

    cannot_claim = [
        "- OCR 主链、敏感文本识别、黑板文字理解、幻灯片文字事件抽取：本次整理未发现主线级稳定证据。",
        "- “提出新的 YOLOv11 backbone/检测结构”：不成立；当前仓库更接近工程整合与验证框架。",
        "- “在真实课堂数据上显著优于所有现有方法”：现有本地结果不足以支撑。",
        "- “完整稳定可部署的生产系统”：不成立；当前证据更适合写成展示原型与 Pages demo。",
        "- 所有 sample/smoke/weak-label/fallback 主导的定量结果：不能写成论文主结论。",
    ]

    output_files = [
        "- `docs/repo_report_alignment_20260423.md`",
        "- `docs/assets/tables/repo_alignment_20260423/script_evidence_index.csv`",
        "- `docs/assets/tables/repo_alignment_20260423/file_scope_summary.json`",
        "- `docs/assets/tables/repo_alignment_20260423/tbl_mainline_vs_behavior.csv`",
        "- `docs/assets/tables/repo_alignment_20260423/tbl_real_pilot_exp_ab.csv`",
        "- `docs/assets/tables/repo_alignment_20260423/tbl_branch_ablation_summary.csv`",
        "- `docs/assets/tables/repo_alignment_20260423/tbl_related_work_metrics.csv`",
        "- `docs/assets/tables/repo_alignment_20260423/tbl_figure_manifest.csv`",
        "- `docs/assets/tables/repo_alignment_20260423/build_validation.json`",
    ]

    text = f"""# YOLOv11 仓库与研究报告对照整理报告（{REPORT_DATE}）

## 1. 输入基线与覆盖范围

- 本次整理以 {", ".join(report_inputs)} 为事实基线。
- 用户提到的 “does 文件夹” 本次按 `docs/` 处理。
- “遍历所有文件” 通过既有全仓审计索引完成覆盖，再对论文相关脚本、结果、文档做重点深读，而不是逐个手工摘要 88,209 个文件。

### 1.1 全仓覆盖摘要

- 审计时间：`{file_scope_summary['generated_at']}`
- 根目录：`{file_scope_summary['root']}`
- 文件总数：**{file_scope_summary['full_traversal']['total_files']}**
- 代码文件数：**{scan_summary.get('category_counts', {}).get('code', 0)}**
- 文档文件数：**{scan_summary.get('category_counts', {}).get('document', 0)}**
- 结构化数据文件数：**{scan_summary.get('category_counts', {}).get('structured_data', 0)}**

重点目录分布：
{chr(10).join(top_dir_lines)}

## 2. 研究报告内容 vs 本地仓库现状

下表回答“研究报告里的判断，在当前本地工作区里是否仍然成立”：

{local_alignment_preview}

结论：

- 研究报告关于**主线能力已实现**的判断，在当前本地工作区仍然成立。
- 研究报告中因为“远程分支不可见”而列为**未确认/待实现**的一批文件，在本地工作区已经存在，说明本地仓库状态比当时远程可见状态更完整。
- 研究报告中关于 **OCR/敏感文本/完整生产系统/大规模 gold 主结果** 的保守边界，在当前本地仓库里仍然应当保持。

## 3. 脚本证据与文件范围

- 重点脚本索引见：`docs/assets/tables/repo_alignment_{REPORT_DATE}/script_evidence_index.csv`
- 范围摘要见：`docs/assets/tables/repo_alignment_{REPORT_DATE}/file_scope_summary.json`

六类脚本/文件分组如下：

- 主线流程：pose、tracking/UQ、action、ASR、query、align、verify、timeline。
- 实验分支：`exp_a/b/c/d/e` 与 paper experiment runner。
- 对照/基线：baseline 包装与固定对照口径。
- 图表生成：`paper_auto` 与 `paper_curated` 图表脚本。
- gold/标注：gold 生成、校验、peer review 准备。
- 展示系统：FastAPI 原型、Pages demo 打包与静态页面。

本次实际抽查过的代表子系统包括：`{", ".join(file_scope_summary['spot_checked_subsystems'])}`。

## 4. 主线、对照组与真实样本实验

固定口径如下：

- `random6_20260420_baseline`：基线
- `random6_20260420_mainline_v3`：稳定主线
- `random6_20260420_behavior_aug_v1`：增强分支
- `real_exp_a_uq_alignment_20260421_yolo11x`：真实样本实验 A
- `real_exp_b_reliability_calibration_20260421_yolo11x`：真实样本实验 B

### 4.1 主线 vs 对照

{mainline_md}

### 4.2 真实样本实验 A/B

{real_exp_md}

### 4.3 明确排除出主结果表的诊断项

{chr(10).join(diagnostic_excluded)}

## 5. 分支实验矩阵与指标边界

完整分支表见：`docs/assets/tables/repo_alignment_{REPORT_DATE}/tbl_branch_ablation_summary.csv`

{branch_preview}

整理规则：

- 任何 `data_mode=sample`、`sample_or_weak_labels`、`smoke`、`weak_self_labels`，或 notes 明确含 `fallback` 的分支，一律降级为 `engineering_only`。
- `exp_a_uq_align` 与 `exp_b_reliability_calibration` 是**候选主结果协议**，但当前仍不能直接作为最终论文主结论。
- `exp_c/d/e`、`exp_object_action_fusion`、`exp_uq_adaptive_alignment` 更适合作为附录/工程验证。
- `exp_f_pages_demo` 只证明展示系统链路存在，不提供正式论文量化指标。

## 6. 论文可用图片与三线表建议

完整图片清单见：`docs/assets/tables/repo_alignment_{REPORT_DATE}/tbl_figure_manifest.csv`

{figure_preview}

主文优先保留：

{chr(10).join(f"- `{row['artifact_id']}` -> `{row['path']}`" for row in main_text_figs)}

附录优先保留：

{chr(10).join(f"- `{row['artifact_id']}` -> `{row['path']}`" for row in appendix_figs)}

三线表填写建议：

{chr(10).join(three_line_suggestions)}

## 7. 相关工作与可借用创新指标

完整相关工作矩阵见：`docs/assets/tables/repo_alignment_{REPORT_DATE}/tbl_related_work_metrics.csv`

已按主来源核验的代表论文包括 10 篇，其中课堂视觉论文与可靠性/校准论文都已覆盖：

{related_preview}

可直接借用或对齐的指标口径：

{chr(10).join(borrowable_metrics)}

说明：

- 课堂视觉论文更适合提供 `mAP/Precision/Recall/F1/FPS` 一类传统对比口径。
- 检测与多模态校准论文更适合提供 `ECE/Brier/NLL/TCE/MCE/dMCE` 与 `abstention/IDK` 一类可靠性指标。
- 结合仓库现状，**最推荐写进主文的“创新性指标”** 仍然是：`F1 + ECE + Brier + uncertain 机制说明`。

## 8. 当前不能直接声称的结论与风险边界

{chr(10).join(cannot_claim)}

额外风险：

- `random6` 与 `real_pilot` 的稳定运行结果可以证明工程链路打通，但不自动等于“最终性能优于 SOTA”。
- gold base / wave1 已完成，wave2 / wave3 仍是 draft；这意味着真实 held-out 结论仍需继续冻结标签。
- `verified_p_match_mean` 可以用于系统状态与案例比较，但不应替代正式人工 gold 指标。

## 9. 输出文件

{chr(10).join(output_files)}

## 10. 本次校验结果

- 校验结果文件：`docs/assets/tables/repo_alignment_{REPORT_DATE}/build_validation.json`
- 是否通过：**{validation['passed']}**
- 失败数量：**{validation['failure_count']}**

本次校验覆盖：

- 主线 run 数量/成功率/失败数 与 `mainline_branch_merged_summary.json`、`mainline_branch_experiment_matrix.csv`、对应 case metrics CSV 的一致性。
- 所有入选图表资产的本地存在性。
- 代表脚本的本地存在性。
- sample/smoke/weak/fallback 结果的降级规则。
- 相关工作主来源 URL 的合法性。
"""
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the repository-to-paper alignment report.")
    parser.add_argument(
        "--deep-report",
        default=str(DEEP_REPORT_PATH),
        help="Optional external deep-research report used only for reconciliation notes.",
    )
    parser.add_argument(
        "--strict-deep-report",
        action="store_true",
        help="Fail when --deep-report does not exist instead of continuing without it.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    deep_report_path = Path(args.deep_report).expanduser()
    if deep_report_path.exists():
        deep_report_text = read_text(deep_report_path)
    elif args.strict_deep_report:
        raise FileNotFoundError(f"deep report not found: {deep_report_path}")
    else:
        print(f"[WARN] deep report not found; reconciliation notes will be empty: {deep_report_path}")
        deep_report_text = ""
    scan_summary = read_json(SCAN_SUMMARY_PATH)
    script_index_rows = read_csv_rows(SCRIPT_INDEX_PATH)
    merged_summary = read_json(MAINLINE_MERGED_SUMMARY_PATH)
    summary_rows = merged_summary.get("mainline_runs", [])
    if not isinstance(summary_rows, list):
        raise ValueError("mainline_runs is not a list")
    matrix_rows = read_csv_rows(MAINLINE_MATRIX_CSV_PATH)

    script_evidence_rows = build_script_evidence_index(script_index_rows)
    mainline_rows = build_mainline_table(summary_rows)
    real_exp_rows = build_real_exp_table(summary_rows)
    branch_rows = build_branch_summary_table()
    figure_rows = build_figure_manifest()
    file_scope_summary = build_file_scope_summary(scan_summary, script_evidence_rows, deep_report_text)
    validation = validate_outputs(
        mainline_table=mainline_rows,
        real_exp_table=real_exp_rows,
        branch_table=branch_rows,
        figure_manifest=figure_rows,
        script_rows=script_evidence_rows,
        summary_rows=summary_rows,
        matrix_rows=matrix_rows,
    )

    write_csv(
        OUT_DIR / "script_evidence_index.csv",
        script_evidence_rows,
        [
            "script_path",
            "exists",
            "category",
            "subsystem",
            "role",
            "purpose_from_audit",
            "paper_related_from_audit",
            "audit_reason",
            "primary_outputs",
            "inspection_note",
            "include_in_primary_table",
        ],
    )
    write_json(OUT_DIR / "file_scope_summary.json", file_scope_summary)
    write_csv(
        OUT_DIR / "tbl_mainline_vs_behavior.csv",
        mainline_rows,
        list(mainline_rows[0].keys()),
    )
    write_csv(
        OUT_DIR / "tbl_real_pilot_exp_ab.csv",
        real_exp_rows,
        list(real_exp_rows[0].keys()),
    )
    write_csv(
        OUT_DIR / "tbl_branch_ablation_summary.csv",
        branch_rows,
        list(branch_rows[0].keys()),
    )
    write_csv(
        OUT_DIR / "tbl_related_work_metrics.csv",
        RELATED_WORK_ROWS,
        list(RELATED_WORK_ROWS[0].keys()),
    )
    write_csv(
        OUT_DIR / "tbl_figure_manifest.csv",
        figure_rows,
        list(figure_rows[0].keys()),
    )
    write_json(OUT_DIR / "build_validation.json", validation)

    report_text = build_report(
        scan_summary=scan_summary,
        file_scope_summary=file_scope_summary,
        mainline_rows=mainline_rows,
        real_exp_rows=real_exp_rows,
        branch_rows=branch_rows,
        figure_rows=figure_rows,
        validation=validation,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8-sig")

    print(f"[DONE] report: {REPORT_PATH}")
    print(f"[DONE] table dir: {OUT_DIR}")
    print(f"[DONE] validation: {OUT_DIR / 'build_validation.json'}")
    if not validation["passed"]:
        raise SystemExit("validation failed")


if __name__ == "__main__":
    main()
