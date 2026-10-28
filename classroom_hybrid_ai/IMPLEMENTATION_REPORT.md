# 课堂视频流程工程提速实施记录

本轮落实工程代码、恢复与评测修复、性能测量入口。研究主流程仍为 Locate → CPU 关联 → Qwen → 时序融合，使用 core8 八类及 UNKNOWN。没有论文正文，也没有引入连续动作模型或新训练集。

## 实施范围

| 计划 | 已实现内容 | 默认状态 |
|---|---|---|
| P0 性能基准 | 独立进程、冻结源码基线、交替重复、冷启动总时长、阶段/请求计时、实际调用与 token、allocator 峰值、WDDM 采样、输入/配置/代码/模型指纹 | 提供 benchmark-performance CLI；缺少指标明确标注 |
| P1 请求成本 | Locate 256 token/原图兼容参数贯通；取消常规全量清理；截图请求原子缓存、校验和恢复；Qwen 单 worker 常驻 | 工程实现启用；紧凑提示保持实验选项 |
| P2 视频与图片 | S0 预检短路；可验证的首中尾定位；S1 合并扫描及 2GiB 缓存回退；S4 合并窗口、上下文复用、有界线程队列；S7 三视频共用解码 | 2 个预处理线程/2 个任务槽；证据稀疏读取和 NVENC 可选 |
| P3 权重搬运 | CPU embedding 查表、仅搬运结果、GPU 输出层；替身模块测试和真实生产请求测量 | 独立配置，尚未晋升默认 |
| P4 恢复与评测 | 阶段预览按需读取；提交内共享 SHA；阶段依赖指纹；HOTA 匹配复用；八类/UNKNOWN/缺测一致；无金标系统指标 | 工程实现启用 |

独立基线保存在仓库 `.runtime/performance_baselines/20260909T151631Z/`，原有工作区修改和删除状态未恢复。旧收据可读；缺少 v2 阶段依赖指纹时重新计算相应阶段。

## 验证说明

回归覆盖请求中断恢复、缓存损坏/失效、阶段依赖失效、CFR/VFR 与异常 PTS 回退、截图和拼图字节一致、固定八类及 UNKNOWN、HOTA 等价、跨阶段峰值聚合，以及基准与稳定性报告不能误判成功的失败分支。VFR 测试使用实际生成的非等间隔 PTS 视频，比较帧号、返回 PTS 和像素与旧顺序读取一致；损坏、重复、缺失或非单调 PTS 的真实文件矩阵仍未全面覆盖。

暂停前全量回归为 **307 passed**（183.76 秒），仅有既有 `taxonomy.EIGHT_CODES` 弃用提示；记录保留在 `diagnostics/performance/tests_final.xml`。对应历史 wheel 验证为 `diagnostics/performance/build_validation.json`。9 月 13 日继续执行后补强了 worker 关闭确认、暂停终态、S5 清理错误传递及完整流程视频语义审计，最终版本的回归和构建记录单独保存。

9 月 13 日最终全量回归为 **359 passed**（186.38 秒），仍仅有上述弃用提示；记录为 `diagnostics/performance/tests_final_20260913.xml`。最终 wheel 位于 `diagnostics/wheels/20260913-final/`，构建与源码逐字节核验记录为 `diagnostics/performance/build_validation_20260913_final.json`。此前一次定向回归为 75 passed / 1 failed：测试进行时添加可选依赖组改变了共享依赖指纹，恢复用例按规则重跑阶段；固定文件后上述全量回归通过。失败记录保留在 `tests_final_gates_20260913.xml`。

新增完整流程审计检查动作观测、关联映射、时序结果、显示数据，以及三种视频的实际像素、帧数、PTS 和音轨。已有 S0–S7 测试产物通过该审计，133 个历史文件的前后 SHA 一致；记录为 `diagnostics/performance/pipeline_artifact_audit_20260913.json`。该产物使用真实视频编码和模型替身，不计作完整实模端到端验收。

稳定性入口要求至少 100 次实际推理、同一 worker、生产拼图和八类提示、每次内存与峰值有效、零 OOM、正常退出及无持续增长；无效或缺失观测不能通过。稳定性与准确率分别记录。

## 100 次生产请求稳定性结果

9 月 13 日 CPU embedding / GPU 输出层候选完成 **100/100 次，通过稳定性门**。30 个真实 S4 候选轮转，沿用 `legacy_v1` 八类提示、128 token 上限、batch=1、无历史。总耗时 1,438.980 秒（约 23.98 分钟）；请求中位数 13.838 秒、p95 15.173 秒，首次请求含加载耗时 48.912 秒。同一候选重复判断一致，实际推理数为 100；零 OOM、零格式失败，worker 收到有效关闭确认，退出码 0，没有清理错误。

请求后 allocated 的预热后窗口中位数为 6,224.209 MiB，reserved 为 8,382 MiB，后段窗口增长均为 0；worker RSS 窗口中位数下降约 539.73 MiB。请求瞬时 allocated 峰值为 7,727,007,232 字节，reserved 峰值为 8,789,164,032 字节；WDDM 全设备采样峰值为 7,897 MiB。allocator 保留量与 WDDM 设备采样属于不同统计范围，不能相加或当作同一个物理显存指标。

本轮输出包含 94 次 `UNKNOWN`、3 次 `a_dx`、3 次 `a_dk`，没有独立金标，质量状态继续为 `not_evaluated`。这些结果不足以证明八类识别质量达标，候选继续保持实验配置。

原始结果为 `diagnostics/performance/qwen_production_100_20260913_v2.json`，独立复核为 `stability_verification_20260913.json`。完整 100 次期间，106 个源码、配置、脚本、测试及文档文件均与 `.runtime/performance_candidates/20260913T055228Z/` 快照一致。之后的变更补强外层 S5 和基准验收，不改变本轮设备策略、提示或模型生成实现。

两份中断记录继续保留：9 月 10 日用户暂停的 `qwen_production_100.json` 标为 `paused_by_user`；9 月 13 日发现关闭确认错误传递缺陷后主动停止的 `qwen_production_100_20260913.json` 标为 `interrupted_for_gate_fix`。两者均不计入通过样本。

## 已完成的局部测量

| 测量 | 对照中位耗时 | 候选中位耗时 | 耗时下降 | 一致性与范围 |
|---|---:|---:|---:|---|
| S0+S1，含冷进程启动与审计 | 8.760 秒 | 8.442 秒 | 3.63% | 同一 530 帧视频，各三次；截图调度及图片 SHA 一致 |
| 证据生成，1→2 个线程 | 4.208 秒 | 2.730 秒 | 35.12% | 已优化代码内的线程对照；48 个模拟候选、240 张 JPEG/轮；图片及收据一致 |
| HOTA，每帧匹配复用 | 0.8628 秒 | 0.08210 秒 | 90.48% | 60 帧×20 人，各三次；另以 100 个随机种子验证完整评分等价 |
| Qwen 设备候选，单请求 | 49.395 秒 | 13.197 秒 | 73.28% | 同一生产候选、legacy_v1 完整八类提示、128 token 上限、各三次；六次输出一致 |
| 同一 Qwen 短测，含冷进程启动与审计 | 84.667 秒 | 51.162 秒 | 39.57% | 每轮新进程、每轮一个实际请求；非完整视频流程 |
| Locate 冷启动观察值，未通过等价门 | 50.081 秒 | 46.953 秒 | 6.25% | 同一生产截图，各三次；框坐标不同，不能计作合格的等价提速结果 |
| Locate 固定种子对照，含冷启动 | 45.864 秒 | 46.901 秒 | **-2.26%** | 六次输出完全相同；未测得提速 |

I/O 冷启动比较未达到 20% 目标。证据线程比较的基线是当前实现的单线程配置，不能解释为相对全部旧版 S4 的收益。4 个线程中位 2.681 秒，比 2 个线程仅快约 1.79%；保留默认 2 个线程和 2 个任务槽。

对应原始报告：`diagnostics/performance/io_paired_v2.json`、`diagnostics/performance/hota_microbenchmark.v1.json`、仓库根目录 `diagnostics/performance/evidence_cpu_20260909/report.json`。第一次 I/O 基准因测试专用预检提示被旧执行器当作阶段警告而停止；该失败保留在 `io_paired.json`，修正测试入口的预检状态后重新完整运行，没有将失败轮次计入成功样本。

Qwen 配对报告为 `diagnostics/performance/qwen_device_paired.json`。请求 p95 从 50.232 秒降到 13.452 秒。两侧都关闭额外同步计时，使用同一外部请求计时器；旧代码没有的内部 token/首 token/加载子项继续标为缺测。候选的 CUDA 请求峰值 allocated 约 7.19GiB、reserved 约 7.53GiB，显存余量有限。单独开启详细计时的 `qwen_device_short.json` 记录 2,341 个输入 token、65 个输出 token、19,701,760 字节 embedding 查表结果传输；这项计数是该 hook 的张量负载字节数，不是完整 PCIe 流量测量。

环境清单见 `diagnostics/performance/environment_inventory.json`，覆盖 Python、Torch、Transformers、Accelerate、bitsandbytes、PyAV、OpenCV 与 NVIDIA 驱动。源码变更摘要见 `diagnostics/performance/implementation_changes.json`。

Locate 原始报告为 `diagnostics/performance/locate_paired.json`。三次原策略输出出现两种摘要，三次候选输出出现三种摘要；输入图片 SHA 相同，原策略自身也有框坐标差异。该组随机生成未受控，无法据此判断差异由工程改动造成，也不能绕过严格输出等价要求。

追加的 `diagnostics/performance/locate_seeded_paired.json` 使用进程启动时的固定种子 0，保留原生产生成参数；六个 Locate worker 均有成功应用种子的 PID 记录，六次输出完全一致。请求中位耗时 32.322→32.649 秒，冷启动总耗时 45.864→46.901 秒。本组支持输出等价，**没有提供 Locate 提速证据**。重现入口为 `diagnostics/performance/run_seeded_locate.py --output <new.json> --repeats 3 --seed 0`，种子控制文件、控制器摘要及逐 PID 记录保存在结果旁，生产默认采样行为未改动。

## 结果边界

局部性能与有界请求基准不能代替完整课堂流程。端到端中位耗时下降至少 20% 仍需完整配对实验验证。四组论文对照（原始、请求优化、视频与缓存优化、合并优化）尚不能据局部结果填入完整端到端表格。

无独立金标时质量状态保持 `not_evaluated`。紧凑提示和设备策略须通过独立质量评测，证明总体 F1 下降不超过绝对 1 个百分点后才能作为默认。已有数据来源跨旧划分问题未在本轮修改，冻结图片保持只读。

当前环境未找到可用 FFmpeg，真实 NVENC 画质、时长及音轨验证尚未完成，默认保留 OpenCV。Qwen3.5 线性注意力快速算子缺失已记录，不更换系统或覆盖依赖。

低调用预算与每次判断约 2 秒的覆盖限制仍存在，不能据本轮工程优化宣称连续八类行为识别。

运行命令和配置说明见 [PERFORMANCE_ENGINEERING.md](PERFORMANCE_ENGINEERING.md)。实测记录保存于本项目 `diagnostics/performance/` 及仓库 `diagnostics/performance/evidence_cpu_20260909/`。
