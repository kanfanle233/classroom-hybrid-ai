# 论文评测协议 v1

本协议冻结 `classroom_hybrid_ai` 的离线评分输入、匹配规则和指标含义。评分器只能读取人工金标；
模型预测、Locate 结果、Qwen 结果和时间轴不能反向充当金标。

## 1. 金标包

一个金标包由 `gold_manifest.v1.json`、`boxes.v1.jsonl` 和 `actions.v1.jsonl` 组成；新包可增加
`annotated_frames.v1.jsonl`，表示已人工复核的全部评测帧，包括确认无学生的帧。

`gold_manifest.v1.json` 必须包含：

- `schema_version=classroom_hybrid_ai.gold_manifest.v1`；
- `dataset_id`、`split_id`、`split`；
- `source_video_sha256`、`source_lesson_id`、`classroom_id`、`capture_session_id`；
- `taxonomy_profile_id`、`taxonomy_contract_sha256`；
- 两个 JSONL 的相对 URI 和 SHA256；
- `annotation_protocol`、双人复核状态和分歧处理记录；
- `license_status` 与 `permitted_uses`。

`boxes.v1.jsonl` 每行必须包含 `frame_index`、`pts_seconds`、`gold_person_id`、`bbox_px`，可选
`confidence=1.0`。同一帧同一 `gold_person_id` 只能出现一次。

`actions.v1.jsonl` 每行必须包含 `gold_person_id`、`start_pts_seconds`、`end_pts_seconds`、
`action_id`。区间必须属于所选 profile、起止有序、同一学生不重叠。无法观察的时间不伪造动作标签，
直接不写入动作金标覆盖范围。

金标包与运行输入的源视频 SHA、profile ID/SHA 不一致时，评分状态为 `invalid_gold`，不产生论文指标。
SCB 或其他许可不明确的数据必须被拒绝；`license_status=granted` 且 `permitted_uses` 明确包含
`evaluation` 才能进入评分。

## 2. 数据拆分与泄漏检查

`source_lesson_id` 是最小拆分单位。同一课、同一源视频及其相邻帧只能属于一个 split。
同一 `capture_session_id` 不得同时进入训练与测试；论文主结果还要求同一 `classroom_id` 不跨训练/测试。
拆分清单出现重复 SHA、lesson/session/classroom 交叉时，评分立即失败。

## 3. 检测指标

评测类别固定为 `student/person`。每帧用 IoU 做一对一最大权重匹配：

- P/R 与漏检率使用 IoU ≥ 0.50；
- `precision=TP/(TP+FP)`；
- `recall=TP/(TP+FN)`；
- `miss_rate=FN/(TP+FN)`；
- AP50 按预测置信度全局降序、每帧金标最多匹配一次，使用 COCO 101 点插值；单类别时
  `mAP50=AP50`。

预测没有真实置信度时省略 AP50/mAP50，并把检测评测标为 `partial`，不得用常数置信度伪造 AP。
旧金标包的检测与跟踪继续只在 `boxes.v1.jsonl` 明确出现的帧上评分。新包可在 manifest 增加
`annotated_frames_uri` 和 `annotated_frames_sha256`；该 JSONL 每行必须有 `frame_index`、
`pts_seconds`、布尔值 `confirmed_empty`、`review_status="complete"` 和至少两个不同的
`reviewers` 人工复核 ID。文件内容必须通过 SHA 核验，帧号不能重复；所有框帧必须在列表内，
PTS 必须与框记录一致。有框帧的 `confirmed_empty=false`，无框帧必须为 `true`，未完成复核的
模板不得参与评分。确认空帧上的预测计入 FP；列表外的帧仍不计分。全部评测帧确认无学生时允许
空 boxes/actions JSONL。评分回执记录所用规则、确认空帧数、评测帧数及实际评分框数。

## 4. 跟踪指标

基础帧匹配使用 IoU ≥ 0.50。`IDF1` 先对每个 gold/pred track 对累计所有评测帧中
IoU ≥ 0.50 的潜在匹配次数，再在整个视频上做一次全局二分匹配，最大化匹配检测数；不得先用
逐帧 Hungarian 丢弃其它潜在身份配对。`IDTP` 是被选 track 对的潜在匹配次数之和，
`IDFP=预测检测总数-IDTP`，`IDFN=金标检测总数-IDTP`。

`ID switch` 按每条 gold track 的时间顺序统计：当前帧匹配到的预测 ID 与上一次有效匹配不同，
计一次切换。`fragment_count` 是每条 gold track 在其有金标的评测帧序列中，被成功跟踪的连续片段数
减一，再对全视频求和；源视频帧号不连续本身不算碎片。

HOTA 在 α=0.05,0.10,…,0.95 上计算，并与官方 TrackEval 的 `hota.py` 对齐。先对每帧的定位
similarity 计算成对归一化潜在匹配，跨帧累计得到 global alignment score；逐帧 Hungarian 的
优化分数为 `global_alignment_score(track_pair) × localization_similarity`，再按各 α 过滤定位
similarity。每个 α 的 `DetA=TP/(TP+FP+FN)`；每个 gold/pred track 对的关联准确率为
`TPA/(TPA+FPA+FNA)`，`AssA` 按该 track 对贡献的匹配检测数加权；
`HOTA_α=sqrt(DetA×AssA)`，最终 HOTA 是 19 个阈值的算术平均。评分回执必须保存各 α 的
DetA、AssA 和 HOTA，便于与官方 TrackEval 结果核对。

## 5. 核心八类动作指标

动作评分只在人工标注覆盖区间内进行。先用跟踪的全局 ID 映射将预测学生绑定到
`gold_person_id`，再把金标与预测区间的所有边界切成原子时间段。每个原子段按持续秒数累加到
`8×9` 混淆矩阵：八个 gold 类为行，八个预测类加 `UNKNOWN` 为列。

每类 P/R/F1 与 macro-F1 使用时长权重计算；macro-F1 对八个 gold 类等权平均。
`UNKNOWN_ratio` 是金标覆盖时长中预测为 UNKNOWN/未观测的比例。完整矩阵必须同时输出稳定的
行列标签、秒数矩阵和可选样本计数，不能把 UNKNOWN 当第九个 gold 类。

实验十一类必须使用独立目录、金标和报告，不得与 core8 聚合。三个新增类每类至少 200 个双人复核
样本，整体覆盖至少 10 段视频和 3 个课堂后，才能输出推广性统计；未达标时状态为
`exploratory_insufficient_support`。

## 6. 时序指标

对每名学生分别合并相邻同类区间。UNKNOWN/灰色区间不作为预测动作 segment；它会使重叠 gold
segment 未匹配并形成 FN。对同类 gold/pred segment 做一对一时序 IoU 匹配，分别在
0.10、0.25、0.50 阈值统计 micro segmental precision/recall/F1。

`flip_count` 只统计连续可观测区间中不同非 UNKNOWN 动作之间的直接切换；UNKNOWN 会断开序列。
`flip_rate_per_observed_student_minute=flip_count/可观测学生分钟`。
`gray_time_ratio=UNKNOWN学生时长/完整学生视频时长`。

## 7. 系统指标

系统指标只读取当前运行的 stage events、receipt 和诊断回执：

- S0–S7 各阶段 wall time 与总 wall time；
- Locate 实际请求数与 Qwen `actual_calls`；
- 已真实记录的峰值 VRAM、峰值 RAM；
- `realtime_factor=总 wall time/视频时长` 和 `processing_fps=帧数/总 wall time`。

缺失的系统观测写入 `unavailable_metrics` 及原因，禁止填 `null`、0 或估算值。

## 8. 无金标输出

没有金标包时，`metrics.v1.json` 必须写：

```json
{
  "schema_version": "classroom_hybrid_ai.metrics.v1",
  "evaluation_status": "not_evaluated",
  "reason": "gold_annotations_not_provided",
  "observed_run_summary": {},
  "system": {}
}
```

此状态下不得出现 precision、recall、mAP50、IDF1、HOTA、macro_F1、confusion_matrix 或
segmental_F1 字段。运行摘要可以报告学生数、UNKNOWN/灰色覆盖和调用量，但字段名必须明确是
预测产物统计，不能称为准确率。

## 9. 可复现性

评测回执记录评分代码 SHA、金标 manifest/SHA、预测 manifest/SHA、profile SHA、匹配阈值、
运行命令、解释器和依赖版本。所有排序使用稳定的 `(frame_index, ID, original_ordinal)` 次序；相同
输入必须产生字节稳定的 JSON。测试 fixture 必须用可手算的小样本验证每个指标，并用第二个独立
实现或公开 TrackEval 结果核对 IDF1/HOTA 后，才能在论文中引用。

## 10. 第二轮 UNKNOWN 诊断与独立标注准备

`diagnose-unknown` 按 `candidate_event_id` 去重，保留同一候选各次响应；重复运行不能增加独立
质量样本量。原图/证据文件路径、SHA、尺寸、框和 PTS 单独审计。诊断原因包含目标过小或被遮挡、
关键动作部位缺失、身份不确定、模型主动弃权、解析失败和预算未覆盖。有效 schema 的 UNKNOWN
可归为模型弃权；模型 rationale 对遮挡或缺少接触证据的描述单列为模型主张，不能自动变成人工
确认的裁剪缺陷。传输错误与解析错误分别记录，没有成功响应也不能自动推断为预算耗尽。

可提供人工复核 JSONL：`candidate_event_id`、`reason_categories`、`reviewer_id`、
`evidence_note`。这些记录仅解释证据问题，不能用作八类动作金标。视觉复核逐项检查手部、桌面
接触、交互对象和目标身份。旧资产没有保存精确提示或原始生成文本时，报告明确记录该缺测原因。

`prepare-annotation` 读取源视频，记录每帧实际 PTS，输出首尾及每秒目标帧模板；可补入同源运行
的截图帧。它不复制模型框、模型学生 ID 或模型动作到人工标注文件。课堂、课次和会话信息缺失时
保持未决，双人复核状态为 pending，创建的 preparation manifest 不可当作 gold manifest。
人工完成标注、来源隔离审查及分歧处理后，另行生成冻结金标 manifest 并重新计算 SHA。

2026-09-13 对当前 `classroom_v2_20260808_005` split manifest 的只读核验：8884 行、478 个
非空 source_group，未发现 source_group 跨 split。source_group 文件名不能证明课堂、课次或
拍摄会话身份，相关来源隔离状态仍为 `not_evaluated`。原冻结清单继续保持只读；独立开发集和
测试集须根据可核验的来源证据建立，不使用冻结测试图调整提示、阈值或裁剪。

当前 100 次稳定性请求来自 30 个候选轮转。94 次 UNKNOWN 涉及 28 个不同候选；28 个候选均有
schema 有效的模型弃权。该结果不能解释为准确率。没有独立视频金标时，报告继续保留
`evaluation_status=not_evaluated`，不允许触发默认配置的质量晋升门。
