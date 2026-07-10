# Dataset Readiness

- dataset_yaml: `F:\PythonProject\pythonProject\YOLOv11\data\processed\classroom_yolo\dataset.yaml`
- dataset_root: `F:\PythonProject\pythonProject\YOLOv11\data\processed\classroom_yolo`
- train_images: `7416`
- val_images: `1467`
- total_images: `8883`
- total_boxes: `267861`
- detection_ready: `True`
- strong_detection_ready: `False`
- min_class_box_count: `680`
- class_imbalance_ratio: `172.835`

## Class Box Counts

| class_id | class_name | box_count |
| --- | --- | ---: |
| 0 | tt | 117528 |
| 1 | dx | 72461 |
| 2 | dk | 58906 |
| 3 | zt | 5339 |
| 4 | xt | 4663 |
| 5 | js | 4183 |
| 6 | zl | 4101 |
| 7 | jz | 680 |

## Notes

- Official YOLO detection finetune is feasible with the current dataset.
- Ablations are still possible, but stability may depend more on the split and augmentation.
- This dataset is a detection dataset. It is not, by itself, a fully labeled temporal sequence dataset for RNN/LSTM/Transformer action modeling.

