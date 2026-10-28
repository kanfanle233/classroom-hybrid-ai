# 第二轮完整实验

`engineering_study.locked.json` 锁定真实 55598 完整视频、四个独立源码变体、配置以及 Locate/Qwen 目录内权重、tokenizer 和自定义代码。每个变体三次，每轮旋转顺序；每次启动独立解释器，执行完整 S0–S7 并核验三种输出视频。没有截断工作量或跳过模型、交付视频的开关。

```powershell
$env:PYTHONPATH = 'classroom_hybrid_ai\src'
& E:\miniconda\envs\py310\python.exe -B -m classroom_hybrid_ai.cli benchmark-study classroom_hybrid_ai\experiments\round2\engineering_study.locked.json --output classroom_hybrid_ai\diagnostics\round2\engineering --dry-run
```

移除 `--dry-run` 执行实验。预览核验文件、读取视频元数据，不加载模型、不创建输出目录。实际学生数、困难截图和证据尚未生成时，调用数显示上下界；缺少同一变体完整视频校准数据时，耗时区间为 `null` 并记录原因。估算和实测分开保存。

在输出目录创建 `STOP`，当前完整视频结束后暂停；移除文件并执行相同命令可恢复。已完成收据和所有输出内容重新核验，不重复成功试验。只有 job 没有可靠收据的中断试验标为 `needs_attention`；若收据已原子写入但进度尚未提交，恢复时可直接采用该收据。

## 工程消融的实际范围

源码位于仓库 `.runtime/round2_engineering_variants_20260913/`，构建收据记录源 SHA、文件及符号，四个变体已编译并通过独立进程导入检查。

| 变体 | 实际修改 |
|---|---|
| `original` | 20260909T151631Z 原始快照 |
| `request_overhead` | Locate 正常请求取消全量 GC/CUDA 清理；异常、退出清理保留 |
| `video_cache` | S1 共享扫描、S4 图片复用、两个任务线程及两个任务槽 |
| `combined` | 上述两项一起启用 |

S0 定位预览、S7 按需预览、阶段指纹、HOTA 和第二轮定位算法变化没有混入上述因素。构建成功不代表输出等价或性能验收通过。`build_engineering_variants` 可在新目录复现，视频优化的供体使用冻结第一轮源码 `.runtime/performance_candidates/20260913T062453Z/classroom_hybrid_ai`。

## 定位算法比较

算法实验使用独立的 `algorithm_quality_cost` 类型，`pipeline.localization_route` 分别为 `locate_only`、`yolo_only`、`yolo_locate_hybrid`。源码、输入视频、调度、预算、提示和设备策略固定。模型文件按每份完整配置中的 `yolo.tracker_model` 锁定，不把权重名写死为 n-pose。

纯 YOLO 路线只确定人体；未经 Locate 确认的目标不进入学生目录。它可能合法完成 S0–S7 且 Qwen 请求为零，结果只可解释为人体定位诊断，覆盖率和质量门仍需通过。算法比较不要求原始学生 ID 字符串相同，减少实际调用也不能直接视作相同工作量提速。

源码和配置冻结后，调用 `write_study_manifest(path, repository_root=..., videos=[{id,path}], variants=[{id,source_path,config}], comparison_type='algorithm_quality_cost', baseline_variant='locate_only')` 生成可执行清单。该函数计算真实全内容 SHA，不接受占位值或未生效覆盖参数。

## 随机种子、计时和质量

每个试验的局部 `sitecustomize.py` 为 Python、NumPy、Torch 设置 seed=0，保存主进程、Qwen、YOLO 和自有 Locate HTTP worker 的 PID 收据。任何实际推理 worker 缺少成功种子收据都会失败。CUDA 确定性算法未强制启用，生成参数保持不变。

冷启动计时覆盖解释器、种子初始化、模型加载、完整阶段、独立媒体审计和退出。输入内容复核位于试验外；生产阶段与媒体审计耗时分别保存。请求 p50/p95 来自统一外层适配器观察器，YOLO 批次和截图数分别解释。

没有独立金标时质量固定为 `not_evaluated`，任何变体均不晋升默认。后续可调用 `assess_study_quality(output, quality_report)`，复核原试验后写入独立的 `study.quality.json`，不运行模型、不改变原始清单或报告。

`performance_study_quality.v1` 质量报告绑定视频、各次语义输出及变体源码/配置/模型 SHA；要求来源隔离审核通过、独立金标、全部 core8 正支持量、至少两课指标和按课 bootstrap 的 95% 区间。门限为时长加权 macro-F1 下降最多 1 个百分点、单类 F1 最多 2 个百分点、学生召回/IDF1/HOTA/时序 F1/UNKNOWN 时间比例最多 1 个百分点；完整中位耗时还需下降至少 20%。
