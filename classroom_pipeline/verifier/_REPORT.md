# verifier/ 目录 — 完整维护报告

> **位置**：`classroom_pipeline/verifier/`
> **更新日期**：2026-07-10

## 一、大致背景

`verifier/` 是实现**双重验证框架**的核心模块。在流水线中，视觉动作识别和ASR语音转写可能产生矛盾的行为判定（如：视觉识别为"听讲"但语音检测到讨论声），verifier作为"二次校验"环节，融合视觉和语义两个模态进行最终判定。

这是论文的核心创新之一，因此本目录既是论文实验的支撑模块，也是产品流水线的关键组件。

## 二、位置与目录结构

```
classroom_pipeline/verifier/
├── __init__.py              # Python包初始化
├── model.py                 # 验证器模型定义（核心）
├── dataset.py               # 验证器训练数据集
├── train.py                 # 训练脚本
├── eval.py                  # 评估脚本
├── infer.py                 # 推理脚本（在流水线中调用）
├── contracts.py             # 验证器输入输出数据契约
├── metrics.py               # 评估指标计算
├── variance_head.py         # 方差预测头（不确定性估计）
├── calibration.py           # 可靠性校准
├── text_similarity.py       # 文本语义相似度计算
```

## 三、是干什么的

| 文件 | 功能 | 在流水线中的位置 |
|------|------|-----------------|
| `model.py` | 定义可学习的验证器模型架构，输入为视觉特征 + 语义特征 → 输出为验证后的行为标签及置信度 | 核心 |
| `dataset.py` | 从Gold标注数据构建训练集 | 离线训练 |
| `train.py` | 训练验证器模型 | 离线训练 |
| `eval.py` | 评估验证器性能（精度、召回等） | 离线评估 |
| `infer.py` | 在流水线中对新视频进行在线推理 | `07_dual_verification.py` 调用 |
| `contracts.py` | 定义输入输出数据格式 | 全流程 |
| `metrics.py` | 计算F1、准确率、可靠性等指标 | 离线评估 |
| `variance_head.py` | 输出预测方差，用于不确定性估计(UQ) | 不确定性量化 |
| `calibration.py` | 可靠性校准：使预测置信度与实际准确率对齐 | Exp B实验 |
| `text_similarity.py` | 计算ASR文本与行为标签的语义相似度 | 语义验证 |

## 四、有什么用

1. **双重验证**：视觉特征 + 语义特征的交叉校验，提高行为识别准确率
2. **不确定性量化(UQ)**：通过 `variance_head.py` 输出预测方差，识别不可靠的预测
3. **可靠性校准**：校准器(calibration)使模型输出的置信度与实际准确率对齐
4. **可训练可迭代**：完整的 train/eval/infer 链路，支持模型持续迭代优化
5. **论文核心贡献**：双重验证框架 + UQ对齐 + 可靠性校准是论文的主打创新点

## 五、维护注意事项

- 训练数据来自 `paper_experiments/gold/` 目录下的Gold标注
- 推理时被 `classroom_pipeline/pipeline/07_dual_verification.py` 调用
- 通用 JSON/JSONL schema 位于 `classroom_pipeline/contracts/schemas.py`；本目录的 `contracts.py` 只提供验证器侧适配入口
- 模型架构修改需要同步更新 `train.py` 和 `infer.py` 的模型加载逻辑
- 校准(curve)功能是独立的开关，需要评估是否每次推理都要进行校准
