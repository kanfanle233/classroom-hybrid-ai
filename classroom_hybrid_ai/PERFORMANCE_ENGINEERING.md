# 工程性能优化与验收

本轮保留 v3.1 的 Locate → CPU 关联 → Qwen → 时序融合顺序。连续动作模型、训练及数据重划分未进入本轮实现。

## 默认实现

- Locate 采用原来实际生效的 256 token 上限和原图尺寸；参数现在贯穿客户端、HTTP worker 与模型。常规请求释放临时引用，完整显存清理保留在异常恢复及退出。
- S2 每个截图/提示组原子保存响应；请求键包含截图内容、参数及阶段指纹。损坏响应、变更参数、模型或相关代码不能复用旧缓存。
- S0 预检失败立即结束；首中尾图通过经过 CFR/PTS 验证的稀疏读取取得，VFR、异常 PTS 和不可验证输入回退顺序读取。
- S1 变化分析与截图共享扫描，临时图片缓存最多 2GiB；超限或无法复用目标帧时回退旧两遍读取。
- S4 合并候选窗口，同一上下文只读/缩放一次；默认两个预处理线程、两个任务槽。稀疏证据解码保留为 `video.evidence_decode_strategy: auto` 选项，默认顺序解码。
- S7 保持一遍源视频读取生成三种视频，阶段预览只加载所需数据。默认 OpenCV 编码；显式 `video.encoder_backend: nvenc` 才探测 FFmpeg/NVENC，失败回退并记录真实额外解码次数。
- 阶段 v2 指纹按代码与配置依赖失效：Qwen 变更从 S5 重跑，显示/编码从 S7 重跑，共享代码影响全部依赖阶段。README 变更不触发重跑。全项目摘要继续保留在审计信息中。
- 单次提交共享 SHA 结果；恢复和独立审计重新验证文件内容。提交哈希耗时写入阶段收据的 `stats.commit_hash`。
- 评测统一八类与 `UNKNOWN`，缺测不填零。HOTA 每帧匹配一次并应用 19 个阈值。无金标仍可报告系统性能，质量状态为 `not_evaluated`。

## 性能接口

从仓库根目录运行，指定解释器仍为 `E:\miniconda\envs\py310\python.exe`：

```powershell
$env:PYTHONPATH = 'classroom_hybrid_ai\src'
& 'E:\miniconda\envs\py310\python.exe' -m classroom_hybrid_ai.cli `
  --config classroom_hybrid_ai\configs\default.yaml `
  benchmark-performance <video.mp4> --component io --repeats 3 --output <new_report.json>
```

组件为 `pipeline|locate|qwen|io`。`pipeline` 执行完整生产流程；`io` 测量预览与截图。模型组件可用 `--run-dir` 指定源视频 SHA 相符的 S1/S4 产物；每轮复制输入并重定位路径，不修改原结果、不继承响应缓存。未指定时自动准备所需阶段，准备时间单列。

用 `--baseline-source <snapshot/classroom_hybrid_ai/src>` 比较冻结源码；可同时指定 `--baseline-config`。基线启动对应快照中的服务实现。每轮先运行基线再运行候选，各自使用新进程，报告完整冷进程生命周期及阶段执行时间，避免忽略模型加载成本。修改试验中的源代码会导致本次试验失败。

`--max-frames` 和 Qwen 专用 `--max-requests` 用于明确限定短测工作量；报告会标记为有界实验，不能代替整课端到端测量。`--profile-model-timings` 开启额外 CUDA 同步与首 token 计时，可能影响吞吐；吞吐对照应使用相同设置。

报告包含输入、配置、源码、模型和 profile 摘要，Python/依赖版本，逐请求和阶段耗时，真实 allocator 峰值，实际调用数及缺测原因。物理显存为 WDDM 全设备采样，包含桌面程序；采样峰值与瞬时 allocator 峰值分别保存。嵌套计时或多线程累计时间不能相加当作端到端耗时。

完整 `pipeline` 另行审计动作观测、关联映射、时间轴、显示数据和三种视频。视频按实际解码的 RGB 像素、帧数及 PTS 比较，音轨按实际解码样本和 PTS 比较；审计耗时单列在 `postrun_audit_seconds`。缺件、不可读媒体或缺少有效审计记录不能通过完整流程性能门。双方共有的历史音轨缺失仍明确保留，候选丢失基线音轨会使等价比较失败。此处严格等价仅适用于相同编码策略。

完整媒体审计和进程内存观测需要 PyAV、psutil；wheel 提供 `performance` 可选依赖组。本轮使用现有环境中的版本，没有覆盖依赖环境。缺少这些工具时按缺测处理。

## 实验选项

- `configs/performance_compact_prompt.yaml`：每个请求只保留一份动作定义，完整 profile 仍留在审计记录中。
- `configs/performance_cpu_embedding_gpu_head.yaml`：embedding 在 CPU 查表，只传输结果，输出层常驻 GPU。默认配置继续使用原设备策略。
- NVENC 使用显式 `video.ffmpeg_path` 或 PATH 中的 FFmpeg；启动及编码结束后验证帧数、首尾画质和时间，音轨可验证时额外检查同步。缺少工具时保留可用视觉结果并报告原因。

连续 100 次请求必须复用同一个 worker 和生产 S4 拼图，使用生产八类提示及 128 token 上限：

```powershell
& 'E:\miniconda\envs\py310\python.exe' -m classroom_hybrid_ai.cli `
  --config classroom_hybrid_ai\configs\default.yaml `
  benchmark-qwen-stability <run_dir> --requests 100 --output <new_stability.json>
```

该命令轮转真实候选，不保留历史；逐请求更新 `<output>.progress.json`。小于 100 次仅记为 `incomplete`。worker、格式或持续内存增长失败会显式结束。稳定性通过不代表动作准确率通过。

本轮 CPU embedding / GPU 输出层候选的稳定性复现使用：

```powershell
& 'E:\miniconda\envs\py310\python.exe' -m classroom_hybrid_ai.cli `
  --config classroom_hybrid_ai\configs\performance_cpu_embedding_gpu_head.yaml `
  benchmark-qwen-stability <run_dir> --requests 100 --output <new_candidate_stability.json>
```

需要暂停时，创建 `<output>.stop` 文件。执行器完成当前请求后关闭 worker，保存 `paused_by_user` 终态；重新验收需使用新输出文件完整运行 100 次。Ctrl+C 也会保存中断终态。CLI 仅 `passed` 返回 0；`incomplete`、暂停、失败分别返回 2、130、1，避免自动脚本将缺测当作通过。worker 关闭时还必须收到有效退出确认；清理错误或缺少确认会传递至验收结果。

## 验收解释

纯工程改动比较截图、候选、证据图和时序输出的一致性。Locate 当前沿用原来的随机生成，独立重复可能不完全相同；报告会据实显示等价门未通过，不把结果差异隐藏为性能提升。

20% 是完整端到端耗时下降目标。HOTA 或预处理的局部收益单列。提示或数值行为变化需要独立金标证明总体 F1 降幅不超过绝对 1 个百分点，达到后才能晋升默认。现有冻结图片不用于新训练、提示示例或阈值选择。

复现局部 CPU 测量可使用 `scripts/benchmark_hota.py` 和 `scripts/benchmark_evidence_cpu.py`。后者使用明确标记的模拟候选，只比较 I/O、证据生成和输出字节，不提供动作准确率结论。
