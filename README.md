# Streaming Emotion LLM

Streaming Emotion LLM is a research codebase for video-level streaming emotion
prediction with a lightweight multimodal language model. The current system
adapts an online VideoLLM/LIVE-style training objective to full-video emotion
streams: each video frame is represented by visual tokens, and the language
model learns when to emit an emotion label versus an interval token.

## Key Features

- Full-video streaming dataset construction for temporally aligned emotion
  annotations.
- TinyLlama + SigLIP multimodal baseline with LoRA fine-tuning.
- Original-style interval-token training using the comma token id.
- Teacher-forcing evaluation for label memorization and timing quality.
- Full-video autoregressive streaming evaluation for online behavior.
- Gradio demo for side-by-side streaming and teacher-forcing inspection.
- Offline-first W&B logging support.

## Environment

Python 3.10 or later is recommended. Training and evaluation require a
CUDA-capable PyTorch installation. Install PyTorch according to the target
machine's CUDA version, then install this package in editable mode:

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[dev]"
pip install -e ".[tracking]"
pip install -e ".[demo]"
```

Core runtime dependencies are declared in `pyproject.toml` and include:

- `torch`
- `transformers`
- `peft`
- `accelerate`
- `numpy`
- `pandas`
- `pyyaml`
- `opencv-python`
- `pillow`
- `torchvision`
- `tqdm`

Optional groups add `pytest`/`ruff` for development, `wandb` for experiment
tracking, `gradio` for the demo app, and `librosa`/`soundfile` for future audio
experiments.

## Data Setup

Data is not included in this repository. Prepare the following local structure:

```text
data/
  raw/videos/
  annotations/
  manifests/
  processed/features/
```

The active configs expect SigLIP features at:

```text
data/processed/features/siglip_large_384_2fps_1plus3x3
```

Each frame is represented by 10 visual tokens: one CLS token plus a 3x3 pooled
feature grid. Features are extracted at 2 FPS with SigLIP Large Patch16 384.

Precompute features:

```powershell
python scripts/precompute_video_features.py --config configs/precompute/siglip_large_384_2fps.yaml
```

Build manifests as needed:

```powershell
python scripts/build_manifest.py
python scripts/build_feature_manifest_splits.py
python scripts/build_feature_subset_manifest.py
```

## Experiment Configs

The cleaned experiment configs are:

| Config | Purpose |
|---|---|
| `configs/experiments/exp1_r8_32videos.yaml` | 32-video overfit run with LoRA rank 8 |
| `configs/experiments/exp2_r32_32videos.yaml` | 32-video overfit run with LoRA rank 32 |
| `configs/experiments/exp3_r32_128videos.yaml` | 128-video overfit run with LoRA rank 32 |
| `configs/experiments/exp4_r32_full.yaml` | full training split run with LoRA rank 32 |
| `configs/experiments/smoke_r32_fullvideo.yaml` | small smoke test for pipeline validation |

The most useful demo checkpoint from the current experiments is Exp2, because
it shows the strongest small-subset overfit behavior.

## Training

Run training from the repository root:

```powershell
$env:PYTHONPATH='src'
python scripts/train.py --config configs/experiments/exp2_r32_32videos.yaml
```

Training uses offline W&B logging by default. Sync completed runs manually when
network access is stable:

```powershell
wandb sync wandb/offline-run-*
```

## Evaluation

Teacher-forcing evaluation measures how well the model predicts the next label
when conditioned on the ground-truth history:

```powershell
python scripts/evaluate_teacher_forcing_fullvideo.py --config configs/experiments/exp2_r32_32videos.yaml --checkpoint outputs/event_stream_original_interval_r32_fullvideo_subset_overfit_tinyllama_siglip_rtx4060_8gb/final --split train
```

Full-video streaming evaluation runs autoregressively over the video and emits
predictions when the interval-token probability falls below the threshold:

```powershell
python scripts/evaluate_video_stream.py --config configs/experiments/exp2_r32_32videos.yaml --checkpoint outputs/event_stream_original_interval_r32_fullvideo_subset_overfit_tinyllama_siglip_rtx4060_8gb/final --split train --threshold 0.80
```

The main evaluation metrics are:

- teacher-forcing emotion exact match
- teacher-forcing interval accuracy
- time difference in seconds
- language-model perplexity
- fluency
- streaming exact match within a 2-second tolerance

## Demo

Use the Gradio demo in `scripts/app_gradio.py` to inspect teacher-forcing and
full-video streaming predictions on train/validation samples.

## Tests

Run the regression tests before handing off changes:

```powershell
$env:PYTHONPATH='src'
pytest -q
```

The current tests cover config loading and full-video stream data construction,
including the one-sample-per-video behavior that is critical for this project.
