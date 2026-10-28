# classroom-hybrid-ai

Local, offline-first classroom video analysis for the Windows workspace. The
runtime uses the required interpreter:

```powershell
E:\miniconda\envs\py310\python.exe
```

The default GPU order is LocateAnything, confirmed unload, then an isolated
Qwen3.5-9B BNB4 worker. A CPU association/crop stage runs between them. Only
one GPU model is resident at a time; the parent process never owns Qwen's
weights. YOLO/ByteTrack remains an explicit fallback mode and is not loaded by
the default Locate-first configuration.

The v3.1 research backend plan is the only valid implementation plan:
[PLAN_RESEARCH_BACKEND_V3_1.md](PLAN_RESEARCH_BACKEND_V3_1.md).

The implemented engineering performance changes, benchmark commands and
experimental promotion rules are documented in
[PERFORMANCE_ENGINEERING.md](PERFORMANCE_ENGINEERING.md). The default path
retains the same models, eight actions, candidate budget and temporal bounds.

## v3.1 research backend

The default command is `core8_v1 + production_adaptive + research_strict`.
LocateAnything supplies every student box, a CPU association stage keeps the
cross-frame `student_id`, and Qwen3.5-9B BNB4 reviews one student's temporal
composite per stateless request. The eight fixed detailed names are written to
the final video and to the vertically stacked class timeline. Unknown or
unreliable periods remain gray.

Each run writes eight atomic stage snapshots and a read-only frontend contract:
`run_state.v1.json`, `stage_events.v1.jsonl`, `stage_receipts/`,
`timeline.v2.json`, `metrics.v1.json`, `videos/annotated_final.mp4`,
`frontend/index.v1.json`, and 60-second `frontend/overlay_chunks/`.
Use `status`, `resume`, `watch`, and `report` from the CLI to inspect a run
without loading a model.

## What the pipeline does

1. Production mode combines PTS-based baseline samples with CPU scene/global
   change anchors. Locate is capped at 1,500 screenshots, including at most
   300 change anchors, and always protects start/middle/end coverage. The
   every-10-frame policy is reserved for bounded exhaustive validation.
2. LocateAnything receives every saved screenshot first and returns all
   student boxes. The raw frame-local model index (`L1`, `L2`, ...) and box are
   saved before any Qwen request.
3. A dedicated CPU association stage matches boxes across nearby screenshots
   by geometry and assigns local IDs such as `student-001`. Ambiguous matches
   receive a new ID so an uncertain association cannot silently swap two
   students. Annotated screenshots visibly show both the persistent ID and the
   corresponding Locate index.
4. A crop router builds all candidate evidence, then applies the hard budget
   `min(240,max(48,2×students,ceil(2×video_minutes)))`. The fair queue protects
   different students, the latter half, edge seats, new IDs and change/anomaly
   evidence. Each selected job stores the exact 480×416 temporal composite;
   every unexecuted candidate stores its reason.
5. Qwen3.5-9B selects one of the fixed eight detailed actions. Every call has
   one current user message, one current composite, zero history messages and
   no cross-request KV-cache reuse. The weights stay loaded for the current
   video, then the entire child process exits.
6. The output timeline keeps one primary action per student at a time. Missing,
   conflicting, skipped or budget-exhausted intervals are explicit
   `unobserved` and use the fixed color `#B8B8B8`.

OCR, ASR, GUI-element recognition and real-student identification are outside
this subproject.

## Eight fixed action labels

The same closed set is used by online Qwen classification and the frozen
evaluation command.

| ID | 中文动作 | 色块颜色 | 可见动作定义 |
| --- | --- | --- | --- |
| `dx` | 低头写字 | `#4E79A7` | 低头并持续在纸张或作业上书写。 |
| `dk` | 低头看书 | `#59A14F` | 低头阅读书本、讲义或纸张。 |
| `tt` | 抬头听课 | `#F28E2B` | 面向授课区域，未持续书写、阅读或与同伴交谈。 |
| `zt` | 转头 | `#B07AA1` | 头部或上半身持续朝后方或后侧。 |
| `js` | 举手 | `#E15759` | 一只或两只手保持在肩部以上。 |
| `zl` | 站立 | `#76B7B2` | 从坐姿变为站姿，或持续站立。 |
| `xt` | 小组讨论 | `#EDC948` | 明显面向同伴、交谈或传递物品。 |
| `jz` | 教师指导 | `#FF9DA7` | 与教师发生讲解、指向或辅导互动。 |

These descriptions are observable actions. A lowered head alone is not labeled
as sleeping, and a track ID is never used to infer identity.

## LocateAnything worker and missing-service warning

LocateAnything performs per-image student/context localization. One `<ref>student</ref>`
may be followed by many boxes; the parser carries that label across every box.
Display numbers are detection indices inside the current image. Persistent
video IDs come from the dedicated Locate-box association stage.
The model does not transcribe text. Its pinned Transformers 4.57.1 runtime is
isolated in a separate HTTP worker. The bundled Windows command is:

```powershell
E:\miniconda\envs\py310\python.exe scripts\locate_anything_service.py --model-dir ..\models\external\locate-anything-3b
```

The worker exposes `GET /health`, `POST /locate`, `POST /unload` and
`POST /shutdown`. The final endpoint is used only for a process started by the
current pipeline, producing a clean worker exit after model unload.
Its Transformers remote-code index is pinned to
`MODEL_DIR/.hf_modules_nf4_457`; it does not use
`%USERPROFILE%\.cache\huggingface\modules`.

The default `auto_start_service: true` policy probes `/health`, starts the
bundled worker with `E:\miniconda\envs\py310\python.exe` when needed, verifies
that `/health` reports the configured model directory, runs Locate, requires a
successful `/unload`, and stops the process before Qwen starts. A compatible
worker that was already running is reused and left running with its model
unloaded.

With auto-start explicitly disabled and `required: false`, an absent worker is
recorded as:

```json
{
  "status": "warning",
  "reason": "service_unavailable",
  "requests": 0,
  "remote_unload": null
}
```

The default Locate-first mode has `required: true`; failure to obtain student
boxes stops the run before Qwen.

## Qwen process isolation and optional smoke diagnostics

For a local run, the parent starts one `classroom-qwen-worker` process for the
whole video. The worker loads Qwen once, serves one review at a time, and exits
after discovery and classification finish. The pipe carries only the current
candidate, one composite image path and the current prompt; no conversation
history is sent. A worker load failure or crash becomes an auditable gray
review and never falls back to another model.

The discovery response uses the full visible-facts schema. Classification uses
a compact transport response for the model-generated decision fields; the
parent attaches the request-bound student event ID, event type, exact
vocabulary SHA-256 and evidence indices before writing the final strict
response. This keeps the full audit contract while fitting the 128-token cap.
For local generation the current assistant turn is prefixed with the single
character `{`, and Qwen generates the remainder. The parser still rejects
Markdown fences and embedded/partial objects; the prefix prevents the model
from starting a fenced code block without weakening validation.

Normal requests use `torch.inference_mode()` and release image/output
references. `empty_cache()` is reserved for an OOM retry, stage teardown or
worker exit. `smoke-qwen` is an optional, explicitly invoked hardware
diagnostic. It resizes inputs to the production 480x416 single-image profile,
rotates multiple inputs round-robin, and uses the same compact classification
transport as the pipeline. `--gate none`, `development`, and `delivery` run
different diagnostic workloads; a saved receipt describes only that run and
is never required by `preflight`. While a smoke run is active, an adjacent
`*.progress.json` file is atomically refreshed after every request.

## Commands

From `classroom_hybrid_ai`:

```powershell
$env:PYTHONPATH = 'src'
E:\miniconda\envs\py310\python.exe -m pytest -q
E:\miniconda\envs\py310\python.exe -m compileall -q src tests
E:\miniconda\envs\py310\python.exe -m classroom_hybrid_ai.cli --config configs\default.yaml preflight --verify-hashes
E:\miniconda\envs\py310\python.exe -m classroom_hybrid_ai.cli --config configs\default.yaml smoke-qwen <single.jpg> <multi.jpg> <occluded.jpg> <empty.jpg> --gate development --output output\classroom_hybrid_ai\_verification\qwen_smoke_dev_20.json
E:\miniconda\envs\py310\python.exe -m classroom_hybrid_ai.cli --config configs\default.yaml run <video.mp4> --profile core8_v1 --mode production_adaptive
E:\miniconda\envs\py310\python.exe -m classroom_hybrid_ai.cli --config configs\default.yaml run <video.mp4> --skip-qwen --development-degraded
```

Check the append-only shared action memory independently. An empty registry is
valid before the first course run; a broken version chain or changed colour
returns a non-zero exit code:

```powershell
E:\miniconda\envs\py310\python.exe -m classroom_hybrid_ai.cli --config configs\default.yaml registry-check
```

`--skip-locate` is rejected while LocateAnything is the primary detector. Use
`--allow-incomplete-preflight` only for development smoke runs.

### Frozen eight-label scoring

The live pipeline and evaluator use the same fixed eight-label vocabulary.
Write one JSONL prediction row per annotated student box:

```json
{"sample_id":"sample-...","box_index":0,"predicted_label":"UNKNOWN","schema_valid":true}
```

Then run the offline joiner from the repository root. It reads the gold label
from the hashed annotation, rejects duplicate or out-of-range boxes, and turns
every omitted box into `UNKNOWN` so coverage is visible:

```powershell
E:\miniconda\envs\py310\python.exe classroom_hybrid_ai\scripts\score_frozen_benchmark.py `
  --manifest classroom_hybrid_ai\configs\blind_benchmark_manifest.v1.json `
  --predictions <predictions.jsonl> `
  --split test `
  --verify-files `
  --output output\classroom_hybrid_ai\_verification\blind_report.json
```

The default split is `test`; use `--split all` only as an explicit post-hoc
analysis. The command never runs Qwen and never changes prompts, thresholds,
the action registry or the online session vocabulary.

To inspect a saved smoke receipt, pass its path explicitly to the read-only
audit command:

```powershell
E:\miniconda\envs\py310\python.exe classroom_hybrid_ai\scripts\audit_delivery_receipt.py `
  --config classroom_hybrid_ai\configs\default.yaml `
  --receipt <receipt.json>
```

The command has no implicit receipt under `output/`. It reports
`historical_receipt_stale` when source or configuration digests differ;
`--strict` returns exit code 2 unless the supplied receipt and optional blind
quality report satisfy the audit contract. It never rewrites a receipt or
runs Qwen.

To create post-hoc Qwen predictions, use the resumable evaluator with a small
limit first. It uses annotation boxes only for cropping and keeps the action
label out of the prompt:

```powershell
E:\miniconda\envs\py310\python.exe classroom_hybrid_ai\scripts\run_frozen_qwen.py `
  --manifest classroom_hybrid_ai\configs\blind_benchmark_manifest.v1.json `
  --split test `
  --limit 20 `
  --output output\classroom_hybrid_ai\_verification\frozen_qwen_predictions.jsonl
```

The output contains Qwen's raw phrase and defaults `predicted_label` to
`UNKNOWN`. A separately reviewed phrase map may be supplied with
`--phrase-map`; only then should the JSONL be passed to the frozen scorer.
Use `--resume` for later batches. This evaluator can be slow because it keeps
the same isolated Qwen worker and one request per annotated student box.

### Paper metrics from independent gold

The paper evaluator never starts a model and never treats prediction artifacts
as labels. A gold package must contain a licensed, double-reviewed manifest,
hashed box/action JSONL files, the exact source-video SHA and the exact profile
SHA. Without that package the formal result is `not_evaluated`.

```powershell
E:\miniconda\envs\py310\python.exe classroom_hybrid_ai\scripts\evaluate_research_run.py `
  <run_dir> `
  --gold-manifest <gold_manifest.v1.json> `
  --profile core8_v1 `
  --output <new_metrics_file.json>
```

Detection/tracking score only explicitly annotated gold frames. Core8 reports
the 8×9 duration matrix (including predicted `UNKNOWN`), per-class P/R/F1,
macro-F1, IDF1/HOTA/switches/fragments and temporal F1. The eleven-class
extension stays exploratory until independently verified support reaches 200
reviewed samples per added class, 10 videos and 3 classrooms.

## Local model and delivery status

- Qwen model: `Qwen3.5-9B`, local BNB4, batch size 1, thinking disabled,
  isolated per video in a child process.
- LocateAnything: isolated Transformers 4.57.1 service at `127.0.0.1:8011`;
  real local verification returned 37 student boxes on the multi-student
  classroom fixture.
- BGE-small-zh: retained for the disabled open-vocabulary mode; it is not
  loaded by the default fixed-eight path.
- X-CLIP/MobileViCLIP: reserved optional open-vocabulary gate, disabled by
  default and not a release dependency. An external same-evidence benchmark
  must set `open_vocabulary.open_vocab_gate.benchmark_verified: true` before a
  provider can be injected; enabling an unverified or missing provider fails
  explicitly instead of changing candidate ranking silently.
- Large visual MoE models are not part of the 8 GB VRAM/16 GB RAM profile.

`configs/models.local.yaml` is the only model manifest used by the source tree
and packaged configuration. It records the local Qwen weights and their
digest, without asserting that an old smoke run proves the current checkout.
Run `smoke-qwen` when hardware diagnostics are needed and evaluate action
quality separately on held-out classroom data.

The blind 8,884-image manifest is frozen for post-hoc evaluation only. No
training, prompt construction or threshold tuning uses that test set.

## Output contract

Each formal run is written below
`output/classroom_hybrid_ai/<video>/<utc_timestamp>-<experiment_digest>/`:

```text
run_manifest.v1.json
run_state.v1.json
stage_events.v1.jsonl
stage_receipts/S0.json ... S7.json
previews/S0/ ... previews/S7/
manifest.json
students.jsonl
action_observations.jsonl
action_intervals.jsonl
timeline.v2.json
metrics.v1.json
report.html
responses/classification/<stage_fingerprint>/*.json
evidence/<candidate_event_id>/temporal_composite.jpg
artifacts/locate_raw_results.jsonl
artifacts/locate_student_boxes.jsonl
artifacts/track_observations.jsonl
artifacts/qwen_crop_queue.jsonl
artifacts/candidates.selection_audit.v1.jsonl
frontend/index.v1.json
frontend/artifact_index.v1.json
frontend/overlay_chunks/*.json
videos/tracking_preview.mp4
videos/action_preview.mp4
videos/annotated_final.mp4
```

The timeline is ordered by spatial student position and then time, so a client
can render students as rows and action intervals as colored blocks.
