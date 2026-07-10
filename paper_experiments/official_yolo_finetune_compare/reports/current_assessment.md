# Current Assessment

## Dataset

- detection dataset yaml: `data/processed/classroom_yolo/dataset.yaml`
- total images: `8883`
- train images: `7416`
- val images: `1467`
- total boxes: `267861`
- readiness verdict: `detection_ready = true`

The dataset is large enough for official YOLO detection finetuning.

The main risk is class imbalance:

- largest class: `tt = 117528`
- smallest class: `jz = 680`
- imbalance ratio: about `172.835`

That means the next gains will likely come from long-tail handling rather than just increasing epochs.

## Existing Runs

For the processed classroom dataset baseline:

- `wisdom8_yolo11s_detect_v1`: final `mAP50 = 0.93183`, final `mAP50-95 = 0.79836`
- `wisdom8_yolo11s_detect_v1_ft70`: final `mAP50 = 0.92612`, final `mAP50-95 = 0.79521`
- `official_yolo11s_detect_baseline_smoke10`: final `mAP50 = 0.91846`, final `mAP50-95 = 0.76591` (10-epoch smoke run)

Decision:

- keep `wisdom8_yolo11s_detect_v1` as the official baseline on the processed dataset
- do not replace it with `wisdom8_yolo11s_detect_v1_ft70`
- smoke10 run is healthy and can be used as a launch gate before longer training

`case_yolo_train` reports a higher validation number (`mAP50-95 = 0.81140`), but that run is tied to another dataset yaml and class mapping path. Treat it as a historical reference, not as a clean same-split winner.

## Model Family Decision

Your current `wisdom8_yolo11s_detect_v1` is already an official Ultralytics YOLO finetune.

If you now train another official YOLO11s run with the same dataset, the goal is reproducibility and controlled ablation, not a fundamentally different model family.

## Better Next Step

Priority:

1. lock an official YOLO baseline with the processed dataset
2. compare against a true custom-YOLO run only after the custom modules are wired and trainable
3. add a temporal model only for sequence understanding, not as a detector replacement
