param(
    [int]$PrecomputePid = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONPATH = "src"

$Config = "configs\experiments\exp4_r32_full_facexformer.yaml"
$OutputDir = "outputs\event_stream_original_interval_r32_fullvideo_tinyllama_siglip_facexformer_mtcnn"
$Checkpoint = Join-Path $OutputDir "final"
$LogDir = "logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

function Write-Step {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] $Message" | Tee-Object -FilePath (Join-Path $LogDir "exp4_facexformer_pipeline.log") -Append
}

function Run-Checked {
    param(
        [string]$Name,
        [string[]]$Command
    )
    Write-Step "START $Name"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Command[0] $Command[1..($Command.Count - 1)] 2>&1 |
        Tee-Object -FilePath (Join-Path $LogDir "exp4_facexformer_pipeline.log") -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
    Write-Step "DONE $Name"
}

if ($PrecomputePid -gt 0) {
    Write-Step "Waiting for FaceXFormer precompute PID $PrecomputePid"
    Wait-Process -Id $PrecomputePid
    Write-Step "FaceXFormer precompute finished"
}

Run-Checked "build full FXF manifests" @(
    "python",
    "scripts\build_feature_manifest_splits.py",
    "--manifest-dir",
    "data\manifests\full_features",
    "--feature-dir",
    "data\processed\features\siglip_large_384_2fps_1plus3x3",
    "--face-feature-dir",
    "data\processed\features\facexformer_2fps_mtcnn_face_token_raw256",
    "--output-dir",
    "data\manifests\full_features_facexformer_mtcnn",
    "--splits",
    "train",
    "val",
    "test"
)

Run-Checked "train exp4 FXF" @(
    "conda",
    "run",
    "-n",
    "video-mm",
    "python",
    "scripts\train.py",
    "--config",
    $Config
)

Run-Checked "teacher forcing val" @(
    "conda",
    "run",
    "-n",
    "video-mm",
    "python",
    "scripts\evaluate_teacher_forcing_fullvideo.py",
    "--config",
    $Config,
    "--checkpoint",
    $Checkpoint,
    "--split",
    "val",
    "--limit",
    "0",
    "--output",
    (Join-Path $OutputDir "val_teacher_forcing_fullvideo.jsonl")
)

Run-Checked "teacher forcing train" @(
    "conda",
    "run",
    "-n",
    "video-mm",
    "python",
    "scripts\evaluate_teacher_forcing_fullvideo.py",
    "--config",
    $Config,
    "--checkpoint",
    $Checkpoint,
    "--split",
    "train",
    "--limit",
    "0",
    "--output",
    (Join-Path $OutputDir "train_teacher_forcing_fullvideo.jsonl")
)

Run-Checked "video streaming val threshold 0.85" @(
    "conda",
    "run",
    "-n",
    "video-mm",
    "python",
    "scripts\evaluate_video_stream.py",
    "--config",
    $Config,
    "--checkpoint",
    $Checkpoint,
    "--split",
    "val",
    "--limit",
    "0",
    "--threshold",
    "0.85",
    "--max-new-tokens",
    "8",
    "--output",
    (Join-Path $OutputDir "val_video_stream_predictions_threshold_085.jsonl")
)

Write-Step "PIPELINE COMPLETE"
