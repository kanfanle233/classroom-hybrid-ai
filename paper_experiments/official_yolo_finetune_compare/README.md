# Official YOLO Finetune Compare

This folder is for one specific question:

1. Is the current `data/processed/classroom_yolo` dataset enough for an official Ultralytics YOLO finetune?
2. How do we rerun an official YOLO baseline in a controlled way?
3. How do we compare that baseline with the runs that already exist in this repo?
4. Would switching to RNN/LSTM/Transformer improve the current project?

Important:

- `runs/detect/wisdom8_yolo11s_detect_v1` is already an official Ultralytics YOLO finetune run.
- If you train another official YOLO11s run with the same data and similar hyperparameters, that is mainly a reproducibility baseline, not a new model family.
- RNN/LSTM/Transformer are not direct replacements for the detector. They are temporal models and belong after detection/pose features, not instead of the YOLO detector.

## Structure

- `profiles/official_yolo11s_detect.yaml`: locked baseline config.
- `scripts/check_dataset_readiness.py`: checks whether the current detection dataset is strong enough for finetuning.
- `scripts/emit_train_command.py`: emits PowerShell commands for smoke and full training.
- `scripts/compare_runs.py`: compares existing detection training runs.
- `reports/`: generated readiness and comparison reports plus framework notes.

## Quick Start

```powershell
Set-Location "F:\PythonProject\pythonProject\YOLOv11"
$py = "F:\miniconda\envs\pytorch_env\python.exe"
```

Check dataset readiness:

```powershell
& $py ".\paper_experiments\official_yolo_finetune_compare\scripts\check_dataset_readiness.py" `
  --data ".\data\processed\classroom_yolo\dataset.yaml" `
  --out_json ".\paper_experiments\official_yolo_finetune_compare\reports\dataset_readiness.json" `
  --out_md ".\paper_experiments\official_yolo_finetune_compare\reports\dataset_readiness.md"
```

Emit official YOLO training commands:

```powershell
& $py ".\paper_experiments\official_yolo_finetune_compare\scripts\emit_train_command.py" `
  --profile ".\paper_experiments\official_yolo_finetune_compare\profiles\official_yolo11s_detect.yaml" `
  --out_ps1 ".\paper_experiments\official_yolo_finetune_compare\reports\official_train_commands.ps1"
```

Compare current runs:

```powershell
& $py ".\paper_experiments\official_yolo_finetune_compare\scripts\compare_runs.py" `
  --profile ".\paper_experiments\official_yolo_finetune_compare\profiles\official_yolo11s_detect.yaml" `
  --out_json ".\paper_experiments\official_yolo_finetune_compare\reports\run_comparison.json" `
  --out_md ".\paper_experiments\official_yolo_finetune_compare\reports\run_comparison.md"
```

## Practical Reading

- If the goal is better detection on the current 8 classes, keep YOLO as the detector and tune dataset, architecture, and losses around it.
- If the goal is better temporal action understanding, add a temporal head after pose/action evidence. For this project, skeleton GCN or video Transformer is more appropriate than replacing YOLO with RNN/LSTM.
