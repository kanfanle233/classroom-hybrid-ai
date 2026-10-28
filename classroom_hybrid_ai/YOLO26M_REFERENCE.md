# 外部 YOLO26m＋TrackTrack 工程参考审计

2026-09-13。外部工程为 `E:/PythonProject/pythonProject/yolo实验`。本次只读取该工程，输出保存在主工程。

**右边缘教师接管学生 ID 的定点修复已经存在。当前源码对应修复后的运行；旧基线的“运行源码与当前代码一致”结论现已过时。** 文件、SHA 和逐条记录核验置信度高。身份判断引用现有人工报告，本次没有重新逐帧标注身份。

完整证据路径、SHA、原始验收回执和核验结果见 [机器可读审计](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/external_yolo26m_reference.v1.json)。

## 1. 本次独立重算确认

两次运行处理同一源视频 `1008.mp4` 的原视频 **[10,40) 秒**，对应原帧 300–1199。源视频完整时长为 503.4 秒，不能把此片段的 900 帧计为整片验收。

| 项目 | 原基线 `20260909T133049Z_94112f34` | 边缘恢复 `20260909T152542Z_9f5389e4` |
|---|---:|---:|
| 帧记录数 | 900 | 900 |
| 框记录数 | 24,602 | 24,602 |
| 非空原始轨迹 ID 数 | 43 | 43 |
| 存在的裁剪图引用 | 1,668 | 1,668 |
| ID 13 的记录数 | 252 | 877 |
| ID 27 的记录数 | 878 | 253 |

本次遍历全部框记录，确认原帧号连续、PTS 严格递增、框对应的帧和 PTS 一致、坐标有限且面积为正，全部裁剪图引用存在。逐 ID 比较所有记录字段，只有 ID 13 和 27 发生变化，其余 41 个 ID 的记录完全一致，包括前排 ID 1–5。

源视频和模型均重新读取文件内容计算 SHA，结果与两次 metadata 一致：

- 源视频：`be9570bb2e3a22e8c1af90fbba9f6bdaeb1308c7ab5e822d331d09fdf15ff794`。
- 本地 YOLO26m 权重：`401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`。

原 `inspection/validation.json` 保存两次标注视频重新解码 900 帧的结果，并明确标注为结构验收。本次没有重新解码这些外部视频，也没有运行模型或测试。2 FPS 是裁剪图保存频率；原运行的检测和跟踪处理每一帧。43 个轨迹含教师、重复框和同一人的多段编号，不能解释为学生人数。

## 2. 源码对应关系

`code_snapshot.json` 保存文件 SHA 清单，不包含归档源码内容。本次将其中的 SHA 与当前 `src/yolo_research/**/*.py` 内容比较：

| 比较 | 结果 |
|---|---|
| 基线清单的 14 个文件 | 12 个一致，`pipeline.py`、`tracking.py` 已改变 |
| 基线之后新增文件 | `edge_tracking.py` |
| 边缘恢复清单的 15 个文件 | 当前全部 15 个文件一致 |

该结论只覆盖上述 Python 源码，不能扩展为全部配置、文档、依赖环境都一致。原基线产物继续保留；本次没有根据 SHA 清单恢复旧工程。

已有修复详见 [右边缘 ID 恢复报告](/E:/PythonProject/pythonProject/yolo实验/docs/edge_id_recovery_result.md)。报告描述了固定机位下的边缘静坐轨迹：遮挡导致框缩窄、预测向画面外漂移，使用受历史长度、边缘位置、静止程度和缩窄程度约束的历史框参考。修复后的原始输出恢复 ID 13，原教师 ID 27 在 18.667 秒结束。报告只声称指定事件通过，其他左侧跨人框、重复框和后排碎片仍存在。

## 3. 可复用的非金标回归案例

以下身份结论来自外部已有人工报告。本次确认其图像、同名原帧 JSON 或其他证据文件存在，并记录了文件 SHA；未将这些结论升级为全体学生逐帧金标。

| 案例 | 原运行 ID | 适合检查的行为 |
|---|---|---|
| 右边缘教师遮挡并接管学生 ID | 13 / 27 | 学生确认不能沿教师轨迹转移；恢复不能新增重复框 |
| 左边缘前后排框跨人与重叠 | 11 / 47 / 51 | 大框含多人时保留歧义，不能直接合并三条轨迹 |
| 右后排同座位学生断轨返回 | 21 → 55 | 断轨恢复、重新确认；17 是另一座位，不能误接 |
| 其他后排多段编号 | 33 → 43 → 64；34 → 45 | 邻座串号与较长间断的区分 |
| 前排同人双框 | 14 / 16 | 重复候选和覆盖统计 |
| 仅有手臂或人体局部 | 6 / 49 | 有人体框不能自动确认完整学生身份 |
| 学生仍可见、原轨迹提前结束 | 17 / 18 / 39 | 输出缺失与检测失败的区分 |

证据入口：[连续性报告](/E:/PythonProject/pythonProject/yolo实验/docs/tracking_continuity_audit.md)、[全部 ID 补查](/E:/PythonProject/pythonProject/yolo实验/docs/tracking_all_ids_review.md)、[学生主体与重叠检查](/E:/PythonProject/pythonProject/yolo实验/docs/student_detection_quality_review.md)。

右边缘还保存了原始检测与关联输入：[回放资产目录](/E:/PythonProject/pythonProject/yolo实验/outputs/reports/edge_association/20260909T151618Z_34654653)。其中 `raw.jsonl`、`returned.jsonl`、`association.jsonl` 适合构造固定输入的算法回归；本次仅核验文件存在及 SHA，没有重新执行回放。

已有 [指定事件自动验收](/E:/PythonProject/pythonProject/yolo实验/outputs/predictions/student_tracking_edge_recovery/20260909T152542Z_9f5389e4/1008_b4cefff160/inspection/edge_acceptance/verification.json) 与 [人工复核记录](/E:/PythonProject/pythonProject/yolo实验/outputs/predictions/student_tracking_edge_recovery/20260909T152542Z_9f5389e4/1008_b4cefff160/inspection/edge_acceptance/manual_review.json) 分别保存逐 ID 差异、9 帧和 14 帧空缺，以及 60 张学生裁剪图的复核范围。人工报告说明部分遮挡时刻单图身份判断受限，并将整段身份一致的置信度列为中。

后续接入时保留原始 `track_id`、来源 SHA、原帧号和 PTS；跨运行数字 ID 不作为身份对齐依据。不得将整条裁剪目录直接当作同一学生的行为训练样本。此参考集的质量状态保持 `not_evaluated`；它不提供八类 F1、全局 IDF1/HOTA、跨课堂泛化或当前 Locate–Qwen 完整流程提速的验收证据。
