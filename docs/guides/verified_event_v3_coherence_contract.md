# Verified Event v3 Coherence Contract

本文档定义下一步实现的 v3 事件证据契约。目标是在兼容现有 `verified_events.jsonl` 的基础上，增加多模态一致性诊断、冲突类型和可视分析所需字段。

## 1. 兼容原则

现有字段继续保留：

```text
event_id
query_id
track_id
event_type
query_text
query_time
window_start
window_end
window
p_match
p_mismatch
reliability_score
uncertainty
label
match_label
threshold_source
model_version
thresholds
action
behavior_code
semantic_id
evidence
```

v3 新字段只追加，不破坏旧 reader：

- 旧前端继续读取 `p_match`、`reliability_score`、`uncertainty`、`evidence.visual_score`。
- 新前端优先读取 `coherence` 块。
- API 先在 v2 evidence response 中嵌入 `coherence`，再视需要提供 `/v3` 别名。

## 2. v3 输出结构

建议每条 `verified_events.jsonl` 追加：

```json
{
  "schema_version": "2026-04-01+coherence_v3",
  "coherence": {
    "c_visual": 0.82,
    "c_text": 0.64,
    "c_object": null,
    "align_score": 0.91,
    "coherence_score": 0.76,
    "coherence_valid": true,
    "conflict_type": "none",
    "quality_flags": ["missing_object"],
    "channel_weights": {
      "visual": 0.87,
      "text": 0.62,
      "object": 0.0,
      "temporal": 0.91
    },
    "channel_valid": {
      "visual": true,
      "text": true,
      "object": false,
      "temporal": true
    }
  },
  "evidence_refs": {
    "video": {
      "case_id": "front__005",
      "start_sec": 2.884,
      "end_sec": 5.116
    },
    "query": {
      "file": "event_queries.fusion_v2.jsonl",
      "event_id": "e_000000_00"
    },
    "alignment": {
      "file": "align_multimodal.json",
      "selected_rank": 1,
      "candidate_count": 8
    },
    "verified_event": {
      "file": "verified_events.jsonl"
    },
    "quality": {
      "uq_file": "pose_tracks_smooth_uq.jsonl",
      "asr_file": "asr_quality_report.json"
    }
  }
}
```

## 3. 字段定义

### 3.1 Channel Scores

`c_visual`

- 类型：`number | null`
- 范围：`[0, 1]`
- 含义：视觉行为候选支持当前事件的程度。
- 初始计算：

```text
c_visual = clamp01(0.65 * action_confidence + 0.35 * semantic_match(event_type, action))
```

`c_text`

- 类型：`number | null`
- 范围：`[0, 1]`
- 含义：ASR 文本与事件/行为语义的一致性。
- ASR 缺失、placeholder、visual fallback 时为 `null`。
- 初始计算：

```text
c_text = clamp01(asr_confidence * semantic_match(query_text or event_type, action))
```

`c_object`

- 类型：`number | null`
- 范围：`[0, 1]`
- 含义：对象证据对行为语义的支持度。
- 对象文件缺失或窗口内无可用对象时为 `null`。
- 初始规则来自 `classroom_pipeline/contracts/object_evidence_mapping.json`。

`align_score`

- 类型：`number`
- 范围：`[0, 1]`
- 含义：事件窗口与行为候选的时间重叠/对齐质量。
- 初始取 selected candidate 的 `overlap`。

### 3.2 Quality Weights

`channel_weights.visual`

```text
visual_weight = clamp01((1 - uq_score) * action_confidence)
```

`channel_weights.text`

```text
text_weight = asr_confidence
```

当 `source=visual_fallback`、ASR placeholder 或 `asr_quality_report.status != ok` 时，`text_weight=0`。

`channel_weights.object`

```text
object_weight = object_confidence
```

无对象证据时为 0。

`channel_weights.temporal`

```text
temporal_weight = align_score
```

### 3.3 Coherence Score

只使用 `channel_valid=true` 且 weight 大于 0 的通道。

```text
A = 1 - sum(w_i * w_j * abs(s_i - s_j)) / (sum(w_i * w_j) + eps)
coherence_score = A * align_score
```

规则：

- 可信通道少于 2 个：`coherence_valid=false`，`coherence_score=null`。
- 缺失通道不是负证据。
- `coherence_score` 不参与替代 `p_match`，只用于解释、筛选和观察卡片门控。

### 3.4 Conflict Type

固定枚举：

```text
none
low_evidence
temporal_mismatch
semantic_mismatch
visual_only
text_only
```

固定优先级：

```text
low_evidence -> temporal_mismatch -> semantic_mismatch -> visual_only/text_only -> none
```

初始判定：

- `low_evidence`：无可信语义通道，或 `reliability_score < 0.4`，或严重 quality flag 存在；单一高支持通道优先归入 `visual_only` 或 `text_only`。
- `temporal_mismatch`：`align_score < 0.4` 且至少一个语义通道支持度高。
- `semantic_mismatch`：两个以上可信通道存在，高低支持差异明显，例如 `max(s_i) >= 0.6` 且 `min(s_i) < 0.4`。
- `visual_only`：只有视觉可信，且视觉支持高。
- `text_only`：只有文本可信，且文本支持高。
- `none`：`coherence_valid=true` 且 `coherence_score >= 0.6`，没有严重 quality flag。

## 4. Quality Flags

固定枚举：

```text
high_uq
low_visual_confidence
occlusion_risk
track_fragmentation
low_asr_quality
missing_text
missing_object
object_conflict
low_alignment
no_candidates
```

初始来源：

- `high_uq`：`uq_score >= 0.35`。
- `low_visual_confidence`：`action_confidence < 0.4`。
- `occlusion_risk`：gold 标注中 `occlusion_level in {medium, high}`，或 UQ 中 visible ratio 过低。
- `track_fragmentation`：同一窗口内 track 缺失、短片段或候选断裂。
- `low_asr_quality`：`asr_quality_report.status != ok`、`segments_good=0`、或 ASR confidence < 0.4。
- `missing_text`：无真实 ASR 文本或 `visual_fallback`。
- `missing_object`：无对象证据输入或窗口内无对象候选。
- `object_conflict`：对象支持语义与视觉行为相冲突。
- `low_alignment`：`align_score < 0.4`。
- `no_candidates`：alignment candidate 为空。

## 5. 实现路线

### 5.1 Pipeline

新增轻量阶段：

```text
classroom_pipeline/pipeline/08_coherence_diagnosis.py
```

输入：

- `verified_events.jsonl`
- `align_multimodal.json`
- `event_queries.fusion_v2.jsonl`
- `pose_tracks_smooth_uq.jsonl`
- `asr_quality_report.json`
- 可选对象证据文件

输出：

- 覆盖或旁路生成 `verified_events.coherence_v3.jsonl`。
- 默认先旁路输出，等验证稳定后再接入主 `verified_events.jsonl`。

### 5.2 Contracts

在 `classroom_pipeline/contracts/schemas.py` 增加：

```text
validate_coherence_block
validate_evidence_refs
validate_verified_event_v3_record
```

## 7. Current Integration Status

- Pipeline stage: `classroom_pipeline/pipeline/08_coherence_diagnosis.py` can generate sidecar `verified_events.coherence_v3.jsonl`.
- Registry: `verified_events_coherence_v3` is registered as an optional artifact and resolves `verified_events.coherence_v3.jsonl`.
- Evidence API: the evidence response now prefers `verified_events_coherence_v3`, falls back to legacy `verified_events`, and can compute runtime coherence when a sidecar is absent.
- Response fields: `coherence`, `evidence_refs`, and `observation_card` are embedded in the normal evidence payload.
- Blinded mode: `coherence`, `evidence_refs`, and `observation_card` are hidden to avoid leaking review evidence.
- Demo UI: `server/templates/front_vsumvis.html` renders an Evidence Coherence card with a four-channel glyph, quality flags, conflict type, and observation-card gate state.

The first integrated smoke case is:

```text
output/智慧课堂学生行为数据集/real_pilot/front/front__005/verified_events.coherence_v3.jsonl
```

要求：

- v3 record 必须通过现有 `validate_verified_event_record`。
- 追加字段单独校验。
- `coherence_score=null` 时必须 `coherence_valid=false`。

### 5.3 Server

扩展现有 evidence service：

- `/api/case/{case_id}/evidence/{event_id}/v2`
- `/api/v2/cases/{case_id}/events/{event_id}/evidence`

新增返回块：

```json
{
  "coherence": {},
  "evidence_refs": {},
  "observation_card": {
    "eligible": false,
    "blocked_by": ["missing_text", "missing_object"]
  }
}
```

### 5.4 Frontend

在现有 `server/templates/` 页面上加：

- conflict filter。
- evidence glyph。
- quality flag chips。
- Top-K candidate panel。
- video jump + ASR text + UQ/source files。

不新增独立数据源；所有视图消费同一 evidence API。

### 5.5 Gold Review

扩展现有 gold draft 字段：

```text
manual_conflict_type
manual_quality_flags
manual_coherence_notes
```

保留现有：

```text
manual_noise_type
manual_occlusion_level
manual_asr_quality
```

这样可把旧 gold 协议平滑升级到冲突类型标注。

## 6. 验收清单

第一阶段工程验收：

- v3 旁路输出能在至少 5 个 real_pilot case 上生成。
- 所有 v3 rows 通过旧 verified schema 和新 coherence schema。
- `coherence_valid=false` 的事件不会产生观察卡片。
- 至少覆盖以下案例：visual fallback、真实 ASR、无候选、高 UQ、低 ASR。

论文验收：

- 主结果只使用 peer-reviewed/adjudicated gold。
- 视觉 gold 和跨模态 gold 分表报告。
- 每个截图中的事件都能回溯到 `evidence_refs`。
- ClassEcho 代码不进入本仓库，除非完成许可证审查。
