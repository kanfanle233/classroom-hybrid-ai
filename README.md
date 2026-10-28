# Classroom Hybrid AI

本仓库只保留 `classroom_hybrid_ai/` 这一套可执行工程。它以本地课堂视频为输入，按
S0–S7 顺序执行 LocateAnything 学生定位、CPU 跨帧关联、Qwen3.5 动作判断、时序融合和
交付物生成。`models/` 与 `data/` 分别保存本地模型和输入数据，不属于运行输出。

## 流程

| 阶段 | 工作 | 主要结果 |
| --- | --- | --- |
| S0 | 输入、配置和模型预检 | 运行清单、模型哈希、首中尾预览 |
| S1 | 自适应截图 | PTS 基础样本与变化锚点 |
| S2 | LocateAnything 定位 | 每帧学生框、Locate 卸载回执 |
| S3 | CPU 关联 | 稳定的本地 `student_id` |
| S4 | 证据预算 | 公平候选队列、480×416 时序拼图 |
| S5 | Qwen3.5 判断 | 严格 JSON 动作观察、worker 退出回执 |
| S6 | 时序融合 | 无重叠区间和显式灰色缺口 |
| S7 | 交付 | 时间轴、报告、前端 JSON 和三类 MP4 |

默认模式一次只占用一个 GPU 模型：LocateAnything 完成并确认卸载后，才启动隔离的
Qwen worker。Qwen 每次请求只接收一个学生的一张时序拼图，不保留对话历史。

## 使用

在仓库根目录执行：

```powershell
$env:PYTHONPATH = 'classroom_hybrid_ai\src'
$py = 'E:\miniconda\envs\py310\python.exe'

& $py -m classroom_hybrid_ai.cli --config classroom_hybrid_ai\configs\default.yaml preflight --verify-hashes
& $py -m classroom_hybrid_ai.cli --config classroom_hybrid_ai\configs\default.yaml plan <video.mp4>
& $py -m classroom_hybrid_ai.cli --config classroom_hybrid_ai\configs\default.yaml run <video.mp4> --profile core8_v1 --mode production_adaptive
```

开发环境可以用少量帧验证完整磁盘合同：

```powershell
& $py -m classroom_hybrid_ai.cli --config classroom_hybrid_ai\configs\default.yaml run <video.mp4> --max-frames 20 --skip-qwen --development-degraded
```

`smoke-qwen` 是显式调用的硬件诊断，不是 `preflight` 的前置回执。需要保存或审计诊断
结果时，由调用方提供输出路径和回执路径。模型来源、版本与 SHA-256 统一记录在
`classroom_hybrid_ai/configs/models.local.yaml`。

运行结果写入 `output/classroom_hybrid_ai/`。这些结果可重新生成，不应被当作源码或模型
资产。详细架构、恢复规则、输出合同和测试命令见
`classroom_hybrid_ai/README.md` 与 `classroom_hybrid_ai/PLAN_RESEARCH_BACKEND_V3_1.md`。
