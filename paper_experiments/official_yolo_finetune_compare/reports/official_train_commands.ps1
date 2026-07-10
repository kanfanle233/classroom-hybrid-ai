Set-Location "F:\PythonProject\pythonProject\YOLOv11"
$py = "F:/miniconda/envs/pytorch_env/python.exe"
$env:YOLO_CONFIG_DIR = "F:/PythonProject/pythonProject/YOLOv11/.ultralytics"

# smoke run
& "F:/miniconda/envs/pytorch_env/python.exe" "F:\PythonProject\pythonProject\YOLOv11\scripts\intelligence_class\training\03_train_case_yolo.py" --data "data/processed/classroom_yolo/dataset.yaml" --model "yolo11s.pt" --imgsz "832" --batch "8" --device "0" --workers "0" --project "runs/detect" --patience "30" --epochs "10" --name "official_yolo11s_detect_baseline_smoke10"

# full run
& "F:/miniconda/envs/pytorch_env/python.exe" "F:\PythonProject\pythonProject\YOLOv11\scripts\intelligence_class\training\03_train_case_yolo.py" --data "data/processed/classroom_yolo/dataset.yaml" --model "yolo11s.pt" --imgsz "832" --batch "8" --device "0" --workers "0" --project "runs/detect" --patience "30" --epochs "80" --name "official_yolo11s_detect_baseline"

# note
# If the 80-epoch run is still improving near the end, rerun this script with --full_epochs 120 or --full_epochs 150.
