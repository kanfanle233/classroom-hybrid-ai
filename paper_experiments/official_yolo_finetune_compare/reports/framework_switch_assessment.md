# Framework Switch Assessment

## Short Answer

Switching away from YOLO will not automatically improve the current detector.

- If the task is still 8-class object or behavior detection from single frames, keep YOLO as the detector.
- If the task becomes temporal action understanding from pose or track sequences, use a temporal model after YOLO or pose extraction.

## What the Current Model Really Is

The current `wisdom8_yolo11s_detect_v1` run is an official Ultralytics YOLO finetune:

- pretrained detector: `yolo11s.pt`
- training data: `data/processed/classroom_yolo/dataset.yaml`
- training mode: standard `YOLO(...).train(...)`

That means your current run is already the correct baseline for frame-level detection.

## Would RNN or LSTM Help?

Sometimes, but not as a detector replacement.

- RNN/LSTM can model time, which is useful for short action sequences.
- They need sequence labels, not only frame-level detection boxes.
- They are weaker than modern transformer or graph-based temporal models at long-range dependencies.

Use RNN/LSTM only if:

- you have track-level or clip-level labels,
- you want a light temporal baseline,
- and you need lower compute than a transformer.

## Would Transformer Help?

Potentially yes, but only in the temporal branch.

- Video transformers can model long-range dependencies across frames.
- They usually need more data and more compute than YOLO finetuning.
- They are better candidates for the "visual-semantic dual verification" part than for replacing the detector.

Good fit:

- use YOLO or YOLO-pose to produce detections, poses, tracks,
- feed short clips or pose sequences into a transformer temporal head,
- align the temporal output with transcript events.

## Better Candidate Than Plain RNN/LSTM

For pose-driven classroom behavior analysis, a skeleton GCN is often a better match than plain RNN/LSTM.

- ST-GCN treats joints as a graph and models spatial plus temporal structure directly.
- This matches classroom actions such as raising hands, standing up, turning, and head-down behavior.

## Recommendation for This Repo

Priority order:

1. Keep official YOLO11s as the detection baseline.
2. If you modify YOLO internals, compare against that baseline.
3. For a real performance jump, add a temporal head after pose or track extraction.
4. First temporal candidate: ST-GCN or another skeleton GCN.
5. Second temporal candidate: transformer temporal head if you have enough labeled clips and GPU budget.

## Source Pointers

- Ultralytics training docs: https://docs.ultralytics.com/modes/train/
- Ultralytics model YAML customization docs: https://docs.ultralytics.com/guides/model-yaml-config/
- LRCN paper: https://arxiv.org/abs/1411.4389
- ST-GCN paper: https://arxiv.org/abs/1801.07455
- TimeSformer paper: https://arxiv.org/abs/2102.05095
