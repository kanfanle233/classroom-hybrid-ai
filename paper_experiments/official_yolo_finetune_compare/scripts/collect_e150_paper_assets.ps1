param(
    [string]$RepoRoot = "F:\PythonProject\pythonProject\YOLOv11",
    [string]$TrainRun = "runs\detect\official_yolo11s_detect_e150_v1",
    [string]$PipelineOut = "output\codex_reports\run_full_e150_001\full_integration_001",
    [string]$PaperDir = "codex_reports\smart_classroom_yolo_feasibility\paper_assets\run_full_e150_001"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $RepoRoot

$paperPath = Join-Path $RepoRoot $PaperDir
New-Item -ItemType Directory -Force -Path $paperPath | Out-Null

$items = @(
    @{figure_id="e150_results_curve"; source="$TrainRun\results.png"; target="e150_results.png"; section="train"; caption="Training and validation loss plus detection metrics across epochs."},
    @{figure_id="e150_confusion_matrix"; source="$TrainRun\confusion_matrix.png"; target="e150_confusion_matrix.png"; section="train"; caption="Confusion matrix of the e150 YOLO detector on the validation split."},
    @{figure_id="e150_confusion_matrix_normalized"; source="$TrainRun\confusion_matrix_normalized.png"; target="e150_confusion_matrix_normalized.png"; section="train"; caption="Normalized confusion matrix of the e150 YOLO detector."},
    @{figure_id="e150_pr_curve"; source="$TrainRun\BoxPR_curve.png"; target="e150_BoxPR_curve.png"; section="train"; caption="Precision-recall curve for the e150 detector."},
    @{figure_id="e150_f1_curve"; source="$TrainRun\BoxF1_curve.png"; target="e150_BoxF1_curve.png"; section="train"; caption="F1-confidence curve for threshold selection."},
    @{figure_id="e150_p_curve"; source="$TrainRun\BoxP_curve.png"; target="e150_BoxP_curve.png"; section="train"; caption="Precision-confidence curve for threshold selection."},
    @{figure_id="e150_r_curve"; source="$TrainRun\BoxR_curve.png"; target="e150_BoxR_curve.png"; section="train"; caption="Recall-confidence curve for threshold selection."},
    @{figure_id="e150_labels"; source="$TrainRun\labels.jpg"; target="e150_labels.jpg"; section="dataset"; caption="Label distribution and annotation geometry overview."},
    @{figure_id="e150_train_batch0"; source="$TrainRun\train_batch0.jpg"; target="e150_train_batch0.jpg"; section="dataset"; caption="Training batch visualization."},
    @{figure_id="e150_val0_pred"; source="$TrainRun\val_batch0_pred.jpg"; target="e150_val_batch0_pred.jpg"; section="qualitative"; caption="Validation prediction examples, batch 0."},
    @{figure_id="e150_val1_pred"; source="$TrainRun\val_batch1_pred.jpg"; target="e150_val_batch1_pred.jpg"; section="qualitative"; caption="Validation prediction examples, batch 1."},
    @{figure_id="e150_val2_pred"; source="$TrainRun\val_batch2_pred.jpg"; target="e150_val_batch2_pred.jpg"; section="qualitative"; caption="Validation prediction examples, batch 2."},
    @{figure_id="e150_timeline_chart"; source="$PipelineOut\timeline_chart.png"; target="e150_timeline_chart.png"; section="pipeline"; caption="Timeline visualization from the e150 full integration run."},
    @{figure_id="e150_reliability_diagram"; source="$PipelineOut\verifier_reliability_diagram.svg"; target="e150_verifier_reliability_diagram.svg"; section="verifier"; caption="Verifier reliability diagram from the full integration run."}
)

$rows = @()
foreach ($item in $items) {
    $src = Join-Path $RepoRoot $item.source
    $dst = Join-Path $paperPath $item.target
    $status = "missing"
    $bytes = 0
    if (Test-Path $src) {
        Copy-Item -LiteralPath $src -Destination $dst -Force
        $copied = Get-Item -LiteralPath $dst
        $bytes = $copied.Length
        $status = if ($bytes -gt 0) { "ok" } else { "empty" }
    }
    $rows += [pscustomobject]@{
        figure_id = $item.figure_id
        source_path = $src
        target_path = $dst
        section = $item.section
        caption_draft = $item.caption
        status = $status
        bytes = $bytes
    }
}

$manifest = Join-Path $paperPath "paper_image_manifest.csv"
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $manifest

$summary = [pscustomobject]@{
    paper_dir = $paperPath
    manifest = $manifest
    total = $rows.Count
    ok = ($rows | Where-Object { $_.status -eq "ok" }).Count
    missing = ($rows | Where-Object { $_.status -eq "missing" }).Count
    empty = ($rows | Where-Object { $_.status -eq "empty" }).Count
}

$summary | ConvertTo-Json -Depth 3

