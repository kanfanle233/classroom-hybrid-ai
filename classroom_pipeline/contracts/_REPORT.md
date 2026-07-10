# contracts/ 目录 — 完整维护报告

> **位置**：`F:\PythonProject\pythonProject\YOLOv11\contracts\`
> **报告生成日期**：2026-05-03

## 一、大致背景

`contracts/` 是项目的**数据契约层**。在整个流水线中，各步骤之间通过JSONL文件传递数据。为了确保下游步骤能正确解析上游输出，这里定义了标准化的数据Schema和样例文件。

## 二、位置与目录结构

```
contracts/
├── __init__.py                  # Python包初始化
├── schemas.py                   # 数据Schema定义（核心文件）
└── examples/                    # 样例文件
    ├── event_queries.sample.jsonl           # 事件查询样例
    ├── verified_events.sample.jsonl         # 验证后事件样例
    ├── verifier_samples_train.sample.jsonl  # 验证器训练样例
    └── pose_tracks_smooth_uq.sample.jsonl   # 带UQ的姿态轨迹样例
```

## 三、是干什么的

| 文件 | 功能 |
|------|------|
| `__init__.py` | 使 `contracts/` 成为可导入的Python包 |
| `schemas.py` | 定义流水线各阶段的JSONL数据格式（字段名、类型、必填/可选），是整个流水线的"接口契约" |
| `examples/*.jsonl` | 提供每种数据格式的实际样例，方便开发者理解数据形状和用于单元测试 |

### 数据契约覆盖的阶段

- **pose_tracks_smooth_uq** — 轨迹平滑后的姿态数据，含不确定性估计(UQ)字段
- **event_queries** — 从ASR文本中抽取的事件查询
- **verified_events** — 经过双重验证器校验后的最终事件
- **verifier_samples_train** — 用于训练可学习验证器的标注样本

## 四、有什么用

1. **流水线契约**：确保上游输出和下游输入格式一致，类似API的接口定义
2. **新人上手**：通过样例文件快速理解数据在各阶段的数据形态
3. **单元测试**：schema可用于自动化校验流水线输出是否符合预期
4. **文档化**：contracts本质上是"可执行的文档"

## 五、维护注意事项

- 修改任一流水线步骤的输出字段时，**必须同步更新** `schemas.py` 和对应样例文件
- JSONL样例文件应保持与当前代码版本一致，避免样例过时误导新人
- 该目录属于"少而精"的类型，不应在其中放实验性或临时文件
