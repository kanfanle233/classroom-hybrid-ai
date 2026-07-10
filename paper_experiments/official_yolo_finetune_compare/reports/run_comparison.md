# Run Comparison

- reference_run: `wisdom8_current`

| label | exists | final_epoch | final_mAP50 | final_mAP50_95 | best_mAP50_95_epoch | delta_vs_ref_mAP50_95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| old_case_baseline | True | 80 | 0.93345 | 0.81140 | 79 | +0.01304 |
| wisdom8_current | True | 80 | 0.93183 | 0.79836 | 79 | +0.00000 |
| wisdom8_ft70 | True | 70 | 0.92612 | 0.79521 | 47 | -0.00315 |
| official_smoke10 | True | 10 | 0.91846 | 0.76591 | 10 | -0.03245 |

## Notes

- `case_yolo_train` and `wisdom8_current` should not be treated as a fully fair apples-to-apples result without confirming the exact dataset yaml and class-order mapping.
- `wisdom8_current` is the locked official baseline for `data/processed/classroom_yolo/dataset.yaml`.
- `wisdom8_ft70` should only be kept if it improves the target metric; on the current numbers it does not beat `wisdom8_current`.
- A future custom-YOLO run should be added here for a fair architecture comparison against the official baseline.

