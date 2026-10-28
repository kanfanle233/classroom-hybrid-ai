# 论文研究后端实施计划 v3.1

本文件是 `classroom_hybrid_ai` 唯一有效实施方案。

2026-09-09 工程提速实施补充见 `PERFORMANCE_ENGINEERING.md`：保持下述模型、动作与预算合同，
增加生产性能基准、S2 请求恢复、阶段 v2 依赖指纹以及视频/评测的等价优化。
紧凑提示、Qwen CPU 查表/GPU 输出层和 NVENC 均有显式实验入口；未通过对应验收不进入默认模型决策路径。

指定解释器：`E:\miniconda\envs\py310\python.exe`。

## 冻结架构

正式入口固定为 `ResearchBackend`：视频读取 → 自适应截图 → LocateAnything 全学生框
→ CPU 跨帧 ID → 学生证据与公平候选 → Qwen3.5-9B BNB4 无历史分类 → 灰色补全
→ 最终视频、时间轴、指标、报告。`HybridPipeline` 只提供内部阶段函数。

- 默认合同：`core8_v1 + production_adaptive + research_strict`。
- `core8_v1` 为论文主结果；`scb_student_11_candidate_v1` 只能进入独立扩展实验目录。
- 扩展颜色冻结：`bs=#9C755F`、`zs=#17BECF`、`hd=#2F4B7C`。
- Locate 单帧零学生可以记录空结果；整段 S2 零学生在严格模式下失败。
- 自定义 Locate 关联器是默认跟踪器；真实 ByteTrack 仅是可选对照实验。
- Locate 卸载 ACK 和进程回执验证前不得启动 Qwen。
- Qwen 每视频一个子进程；每请求一个学生、一个 user message、一张时序拼图、零历史、零跨请求 KV。
- 正式模式关闭 `qwen_every_screenshot`，只消费预算后的公平候选。
- 插值最长 2 秒；超时后隐藏人体框并把时间轴置灰。
- watch 仅绑定 `127.0.0.1`，只读，无数据库、账号、云服务和第三方 Web 框架。
- `diagnostics/` 保存 PID、显存、错误等内部审计；`frontend/` 只含相对 URI 和公开数据。
- 运行 ID 使用 UTC 时间戳加实验摘要前 12 位，防止重复实验覆盖。

## 公共 CLI

```text
plan <video> [--profile ...] [--mode ...]
run <video> --profile core8_v1 --mode production_adaptive [--watch]
status <run_dir>
resume <run_dir>
watch <run_dir> --host 127.0.0.1 --port 8765
report <run_dir>
```

`run --resume` 已弃用。严格运行失败返回结构化错误和非零退出码；开发降级运行必须显式请求，且
`research_validity` 只能是 `pending/invalid`，不能进入论文主指标。

## S0–S7 磁盘边界

权重固定为 `5/10/20/15/10/25/7/8`。receipt 仅在产物原子写入、哈希验证后完成。

- S0：`run_manifest.v1.json`、preflight、首中尾输入预览。
- S1：真实截图调度、已提取截图、联系表。
- S2：Locate 原始响应、学生框、卸载 ACK、worker 退出回执。
- S3：跨帧 ID、学生列表、关联质量和歧义记录。
- S4：预算前候选、公平队列、裁框、480×416 时序拼图。
- S5：逐请求 Qwen 响应、动作观察、worker 退出回执；已成功请求恢复时不得重复。
- S6：无重叠动作区间、传播来源、完整灰色缺口、`timeline.v2.json`。
- S7：阶段预览、三种语义不同的视频、前端索引、真实指标、报告和完整 manifest。

所有阶段只能从磁盘产物恢复。输入、配置、代码、profile、Locate 模型、Qwen 模型或上游
artifact SHA 变化时，当前阶段及下游失效。失败运行不得显示 100%。

## 调度、事件与输出

`production_adaptive` 生成 PTS 基础锚点和 CPU 场景/全局变化锚点；Locate 总量 ≤1500，
变化额度 ≤300。Qwen 预算为 `min(240,max(48,2×有效学生数,ceil(2×视频分钟数)))`，
后半段、边缘座位、新 ID 有审计额度。未执行候选必须记录原因。

`stage_events.v1.jsonl` 是单写者 append-only 日志，事件有单调 `seq`，写入频率最多每秒或每增加
1%。`run_state.v1.json` 是原子快照。watch 按 seq 增量 SSE 推送；10 万事件不得全文件重写。

`tracking_preview.mp4` 与 `action_preview.mp4` 只含首/中/尾各 10 秒，语义分别为 ID 跟踪与动作；
`annotated_final.mp4` 是全片。传播透明度 0.55，未观测灰色。Overlay 每 60 秒流式分块，前端只用
相对 URI。音轨可用 ffmpeg 时复用，否则写警告并保留视觉结果。

## 评测与最终门禁

真实金标才允许报告 Locate P/R/mAP50/漏检率、IDF1/HOTA/ID switch/碎片、核心八类混淆矩阵与
macro-F1、segmental F1@.1/.25/.5、翻转率和灰色比例。系统指标记录阶段耗时、调用量、峰值
VRAM/RAM。数据按源视频拆分，SCB 未许可数据或权重不得进入训练、发布和主实验。

代码冻结后运行 `preflight --verify-hashes`、短视频 E2E、S2/S5 中断恢复、完整课程、wheel 和
干净环境验收。Qwen smoke 是按需执行的硬件诊断，保存的回执不参与 preflight。最后执行独立
只读审查；存在 P0/P1、混合 profile、虚假阶段完成或无金标指标时，不得标记研究完成。
