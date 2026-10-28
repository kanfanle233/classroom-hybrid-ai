# 第二轮实施与实测记录

2026-09-13。代码已实现并保存独立快照。默认仍为 `locate_only`，未晋升提示、设备、模型或批量策略。**尚未完成含 Qwen 的四组完整流程验收，不能宣称端到端提速达到 20%。**

## 已实现

- 三种定位路线；YOLO 逐截图原子缓存；困难截图合并、Locate 完整替换；YOLO 进程退出后才加载 Locate。增加对本地 YOLO26m 的清单键与真实 SHA 校验。
- 未确认人体单独保存在通用定位审计中；学生目录、候选和动作统计只接收确认学生。S2/S3 预览显示真实来源。有效空结果可完整交付；失败结果保留错误和恢复依据。
- S4 预算集中配置，显式全候选实验；计划报告未生效旧字段，纯 YOLO 路线明确预计零学生动作请求。
- S5 每个 pass 建立一次学生/PTS 索引和动作定义模板，多提示组共享截图 SHA。回归检查保留元数据顺序、时间窗口和提示字节；本轮没有使用真实 Qwen tokenizer 另测输入 token 等价性。
- `benchmark-study` 固定源代码、配置、模型和视频内容 SHA，轮转三次、独立 worker 种子记录、完整媒体审核、原子试验收据与恢复。工程和算法比较分开。
- UNKNOWN 去重诊断、人工视频标注包、显式已标注帧和无人帧验证、双人复核要求、严格质量晋升门。
- 修正混合配置被旧校验器拒绝，以及 Qwen 超时被 OSError 分支提前捕获的问题；超时后不再把迟到响应当作关闭确认。

交付回归共 **505 项通过，0 失败、0 错误、0 跳过**，耗时 236.820 秒，见 [测试回执](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/delivery_tests.xml)。该结果覆盖代码回归，不表示以下尚缺的真实模型质量与完整性能验收已经完成。

## 实际运行结果

两模型使用同一视频的相同 20 张生产截图，输入 960、conf=0.20。每种 batch 独立启动 worker，三轮旋转，固定 seed=0。外部工程的 FP16、conf=0.10、TrackTrack 和 2 FPS 裁剪参数未混入本次权重对照。

| 结果 | YOLO26n-pose | YOLO26m |
|---|---:|---:|
| 20 张截图的人体框总记录（batch 1） | 327 | 479 |
| 单图框数最小/中位/最大 | 13 / 16 / 20 | 20 / 23 / 28 |
| batch 1 冷启动中位耗时 | 8.756 秒 | 8.637 秒 |
| batch 2 冷启动中位耗时 | 8.660 秒 | 8.674 秒 |
| batch 4 冷启动中位耗时 | 8.567 秒 | 8.556 秒 |
| batch 1 PyTorch allocated 峰值 | 99,447,296 字节 | 290,654,208 字节 |
| batch 4 PyTorch allocated 峰值 | 171,218,432 字节 | 543,026,176 字节 |
| 困难截图 | 20 / 20 | 20 / 20 |

峰值属于相应 YOLO worker 的 PyTorch 分配器，物理 WDDM/驱动及其他进程不包含在该数值内。冷启动指独立进程和模型重新加载，文件系统缓存没有强制清空。

两模型全部截图都包含低于 0.28 的人体检测，另有新轨迹、歧义与断轨触发。因此本片段混合路线没有节省 Locate 请求。batch 2/4 出现检测输出差异，未达到精确等价条件；YOLO26m 的框记录数分别从 batch 1 的 479 变为 478 和 477。耗时收益也未接近 20%，保留 batch 1。框记录增加不能换算成召回率或学生人数。

详细记录：[n-pose 批量比较](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/yolo26n_batches/report.json)、[YOLO26m 批量比较](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/yolo26m_batches/report.json)。

两种纯人体定位配置各完成三次完整 55598 视频的 S0–S7，没有帧或请求截断参数。整片为 530 帧、17.667 秒；每次三种媒体均通过解码、PTS 和语义校验，主进程与 YOLO worker 种子收据通过。

六次运行的三种媒体（共 18 个输出）均记录 `audio_status=unavailable`：两个预览的原因是 `preview_noncontiguous_clips_audio_not_muxed`，完整标注视频的原因是 `ffmpeg_not_found`。上述媒体校验覆盖视频图像、帧数和 PTS；音轨缺失被明确保留，音轨保留及音画同步尚未通过真实验收。

| 整片定位诊断 | n-pose | YOLO26m |
|---|---:|---:|
| 三次冷启动总耗时中位数（含交付审计） | 40.715 秒 | 40.941 秒 |
| YOLO 请求 p95 | 0.449 秒 | 0.439 秒 |
| 每次 YOLO 截图请求 | 20 | 20 |
| 实际 Locate / Qwen 请求 | 0 / 0 | 0 / 0 |
| 已确认学生、动作判断 | 0 | 0 |

该对照只检查人体定位流程。YOLO26m 中位耗时增加约 0.55%；质量状态为 `not_evaluated`，默认晋升为 false。零动作请求由未确认身份规则产生，不代表完整动作分析提速。详见 [六次整片诊断](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/person_detector_study/study.json)。

另完成一次真实混合 S2 检查：20 张 YOLO 截图→正常关闭→20 次 Locate 请求→正常释放，共 153.150 秒。这是模型交接验证，未执行本次 S5/S7，也不是三次正式性能对照。详见 [混合 S2 记录](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/hybrid_s2_integration.json)。

CPU 索引小测采用合成的 1,500 条定位记录、45,000 个框、240 个候选，包含索引构建成本。三轮结果完全一致，中位数由 0.124102 秒降为 0.088638 秒，减少 28.6%。此结果不外推到端到端耗时；见 [报告](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/index_microbenchmark.json) 和 [复现脚本](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/index_microbenchmark.py)。

## 质量与外部工程

历史 100 次请求去重为 30 个候选；94 次 UNKNOWN 对应 28 个有效弃权候选。5 个候选的响应提及部位/接触缺失，视觉原因仍待人工确认。没有依据这些描述自动修改裁剪。报告核验了 30 个候选的原图引用（实际共 1 张原截图）、30 张不同拼图及对应目标框，并记录旧产物缺少精确原始生成文本的限制。

- [UNKNOWN 原因报告](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/unknown_diagnostics.v2.json)
- [530 帧标注包](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/annotation_55598/annotation_package.v1.json)：19 个待复核目标、完整 PTS 清单、来源身份待补齐，尚未形成金标。
- [外部 YOLO26m 工程核验](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/YOLO26M_REFERENCE.md)：确认后续右边缘 ID 13/27 修复已存在，其他 41 个 ID 记录保持一致。旧验收源码与当前源码已有差异，修复后的源码账本与当前一致。

本轮没有独立金标质量分数。时长加权八类 F1、每类退化、学生召回、IDF1/HOTA、时序 F1 和 UNKNOWN 时长比例均不作新质量收益结论。

## 可复现入口与待完成验收

完整工程四组对应实际构建的独立源码：[工程清单](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/experiments/round2/engineering_study.locked.json)。新定位五组采用修正公开配置校验后的源码：[算法清单 v2](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/experiments/round2/algorithm_study.v2.locked.json)。旧 `algorithm_study.locked.json` 保留作开发审计，五组执行请使用 v2。

```powershell
& E:\miniconda\envs\py310\python.exe -B classroom_hybrid_ai/scripts/classroom_hybrid.py benchmark-study classroom_hybrid_ai/experiments/round2/algorithm_study.v2.locked.json --output classroom_hybrid_ai/diagnostics/round2/algorithm_full --dry-run
```

移除 `--dry-run` 会启动该清单完整实验。工程清单同样适用。`STOP` 只在当前完整视频结束后暂停；恢复先复核收据。实际运行估算、收据和质量结果分别保存。

以下项目仍未完成，不能用上述定位诊断替代：四组含 Qwen 完整工程实验各三次；Locate 与混合路线完整动作流程各三次；来源明确的独立金标与质量门；最终通过质量门组合的新一轮 100 请求稳定性；长视频稀疏读取与 NVENC 实拍验收。原有 100 次稳定性记录保留，本轮没有选出新的可晋升组合。

按现有组件记录、假设每次视频实际执行 48–60 次 Qwen 请求，工程 12 次试验仅模型部分估算约 8–11 小时；这不是总耗时上限。若每次达到 240 请求，报告对应的模型部分估算约 40–41 小时，其他阶段耗时还需另计。完整试验尚未执行，默认性能目标状态保持未验收。参见 [组件估算及假设](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/engineering_estimate_from_components.json)。

论文组织与结果边界见 [方法及实验提纲](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/ROUND2_METHODS_AND_EXPERIMENTS.md)。代码差异仅覆盖本轮：[35 文件补丁](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/implementation.patch)；独立交付快照见 [快照账本](/E:/PythonProject/pythonProject/YOLOv11/classroom_hybrid_ai/diagnostics/round2/delivery_snapshot.json)。工作区原有 308 个跟踪删除和 2 个修改保持原状态。

置信度：代码、文件一致性及本轮实际运行记录高；八类动作质量变化与完整端到端提速未知。
