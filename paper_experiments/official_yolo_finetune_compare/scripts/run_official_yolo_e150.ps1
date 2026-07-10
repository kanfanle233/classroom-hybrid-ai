param(
    [string]$RepoRoot = "F:\PythonProject\pythonProject\YOLOv11",
    [string]$PythonExe = "F:\miniconda\envs\pytorch_env\python.exe",
    [string]$RunName = "official_yolo11s_detect_e150_v1",
    [int]$Epochs = 150,
    [int]$Imgsz = 832,
    [int]$Batch = 8,
    [int]$Workers = 0,
    [string]$Device = "0",
    [string]$DataYaml = "data/processed/classroom_yolo/dataset.yaml",
    [string]$BaseModel = "yolo11s.pt",
    [string]$Project = "runs/detect",
    [int]$Patience = 30
)

$ErrorActionPreference = "Continue"
$global:PSNativeCommandUseErrorActionPreference = $false
Set-Location -Path $RepoRoot
$env:YOLO_CONFIG_DIR = Join-Path $RepoRoot ".ultralytics"

$logDir = Join-Path $RepoRoot "official_yolo_finetune_compare\reports\runtime_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir ($RunName + "_" + $stamp + ".log")
$pidPath = Join-Path $logDir ($RunName + ".pid")
$metaPath = Join-Path $logDir ($RunName + "_meta.txt")

$header = @(
    "RUN_NAME=$RunName"
    "PID=$PID"
    "STARTED_AT=$(Get-Date -Format s)"
    "REPO_ROOT=$RepoRoot"
    "PYTHON=$PythonExe"
    "YOLO_CONFIG_DIR=$($env:YOLO_CONFIG_DIR)"
    "LOG_PATH=$logPath"
    "EPOCHS=$Epochs"
    "WORKERS=$Workers"
)
$header | Add-Content -Path $logPath -Encoding UTF8
$header | Set-Content -Path $metaPath -Encoding UTF8
Set-Content -Path $pidPath -Value "$PID" -Encoding UTF8

$trainArgs = @(
    ".\scripts\intelligence_class\training\03_train_case_yolo.py",
    "--data", $DataYaml,
    "--model", $BaseModel,
    "--epochs", "$Epochs",
    "--imgsz", "$Imgsz",
    "--batch", "$Batch",
    "--device", "$Device",
    "--workers", "$Workers",
    "--project", $Project,
    "--name", $RunName,
    "--patience", "$Patience"
)
try {
    & $PythonExe @trainArgs *>> $logPath
    $exitCode = $LASTEXITCODE
    "FINISHED_AT=$(Get-Date -Format s)" | Add-Content -Path $logPath -Encoding UTF8
    "EXIT_CODE=$exitCode" | Add-Content -Path $logPath -Encoding UTF8
    exit $exitCode
}
catch {
    "FAILED_AT=$(Get-Date -Format s)" | Add-Content -Path $logPath -Encoding UTF8
    "ERROR=$($_.Exception.Message)" | Add-Content -Path $logPath -Encoding UTF8
    "DETAIL=$($_ | Out-String)" | Add-Content -Path $logPath -Encoding UTF8
    exit 1
}
