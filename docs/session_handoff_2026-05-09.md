# Session Handoff: 2026-05-09

This document summarizes the project state and decisions made while creating and organizing the `streaming-emotion-llm` repository.

## Project Purpose

This repository is an independent research project derived from earlier work reproducing an online VideoLLM-style streaming dialogue repository. It is no longer only a reproduction branch.

Current research direction:

- streaming facial emotion understanding
- online video-based emotion token prediction
- TinyLlama + LoRA adaptation
- SigLIP visual feature extraction
- projector-based multimodal alignment
- future audio-video fusion
- future emotion reasoning generation

The current baseline is intentionally narrowed to:

```text
video-only full-video feature sequence -> TinyLlama generates open-vocabulary emotion token
```

No closed-set classification head is used. Reasoning generation is disabled for now because the reasoning annotations often depend on audio cues such as tone, pitch, speech content, and vocal pace.

## Repository Location

```text
D:\Documents\projects\multimodal-video-project\streaming-emotion-llm
```

The original paper reproduction repository is:

```text
D:\Documents\projects\multimodal-video-project\videollm-online
```

The new repository has been initialized as an independent Git repository.

## Environment Policy

The repository documentation must not include private local environment names, interpreter paths, or machine-specific paths.

Committed `.vscode/settings.json` is allowed, but only with project-generic settings:

```json
{
  "python.terminal.activateEnvironment": true,
  "python.analysis.extraPaths": [
    "${workspaceFolder}/src"
  ],
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "${workspaceFolder}/src"
  }
}
```

README and docs should describe requirements generically:

- Python 3.10+
- GPU-compatible PyTorch for training/inference
- editable install with `pip install -e .`

## Data State

The user placed two archives under `data/`:

```text
data/all_videos.tar
data/responses.zip
```

They were extracted to:

```text
data/raw/videos/
data/annotations/responses/
```

Counts after extraction:

```text
videos: 1551 mp4 files
annotation files: 1539 txt files
```

Annotation files are JSON-like arrays. Valid event format:

```json
{
  "timestamp": 0.031,
  "emotion": "solemn",
  "detailed_reasoning": "Long explanation text.",
  "summary_reasoning": "Short cue summary."
}
```

Annotation quality:

```text
valid JSON files: 1508
invalid JSON files: 31
events in valid files: 11594
videos missing annotations: 12
all annotations have matching videos
```

Generated manifests:

```text
data/manifests/all_valid.jsonl
data/manifests/train.jsonl
data/manifests/val.jsonl
data/manifests/test.jsonl
data/manifests/invalid_annotations.txt
data/manifests/videos_missing_annotations.txt
```

Initial split:

```text
train: 1206
val: 150
test: 152
```

## Current Training Target

The training target is only the `emotion` field.

Use:

```json
{
  "timestamp": 0.031,
  "emotion": "solemn"
}
```

Ignore for now:

```json
{
  "detailed_reasoning": "...",
  "summary_reasoning": "..."
}
```

Reasoning will be revisited after audio encoders and audio-video fusion are added.

## Feature Precomputation

SigLIP features are precomputed before training. This follows the original paper-style approach:

- sample full videos at fixed FPS
- resize and pad frames
- run frozen SigLIP
- save one `.pt` feature file per video
- train downstream on precomputed features

Current feature script:

```text
scripts/precompute_video_features.py
```

Default output:

```text
data/processed/features/siglip_large_384_2fps_1plus3x3/
```

Default feature setting:

```text
vision model: google/siglip-large-patch16-384
resolution: 384
fps: 2
token mode: live1+
tokens per frame: 10
token shape: 1 CLS + 3x3 pooled spatial tokens
hidden size: 1024
saved dtype: bfloat16
```

Example feature tensor:

```text
vid_0001_clip1.pt
shape: (102, 10, 1024)
dtype: torch.bfloat16
```

Precomputation status:

```text
completed feature files: 440
target valid videos: 1508
```

The full precompute job ran for about 2 hours and stopped around 440 videos. The log showed many video decoding warnings/errors from corrupted or difficult H.264 streams, then the process exited. Existing `.pt` files are usable.

Feature subset manifest was generated from existing `.pt` files:

```text
data/manifests/feature_subset/all.jsonl
data/manifests/feature_subset/train.jsonl
data/manifests/feature_subset/val.jsonl
data/manifests/feature_subset/test.jsonl
```

Subset counts:

```text
all: 400
train: 320
val: 40
test: 40
```

Subset statistics:

```text
samples: 400
events: 3089
unique emotions: 381
feature frames min/max/avg: 41 / 120 / 104.24
visual tokens min/max/avg: 410 / 1200 / 1042.38
```

## Current Training Mode

The user decided not to use fixed 16-frame windows.

Current mode:

```text
full-video feature sequence per sample
```

Config now uses:

```yaml
streaming_window:
  mode: full_video
  max_num_frames: 120
  stride: null
  fps: 2.0
```

This means each sample loads the whole precomputed video feature sequence, capped at 120 sampled feature frames for the first subset experiment.

Important tradeoff:

- fewer samples can reduce total training time
- full-video features increase per-sample sequence length and GPU memory
- on an 8GB RTX 4060, start with batch size 1 and LoRA/projector training only

If OOM occurs, reduce:

```text
max_num_frames: 120 -> 80 -> 64
```

This is still 鈥渇ull-video style鈥?truncation/subsampling, not a fixed 16-frame local event window.

## Original Paper Code Ported

Core streaming framework components were ported from the original repository into this project as native modules.

Mapping:

| Current project file | Original repository file | Role |
| --- | --- | --- |
| `src/streaming_emotion_llm/models/configuration_live.py` | `models/configuration_live.py` | Streaming config fields |
| `src/streaming_emotion_llm/models/tokenization_live.py` | `models/tokenization_live.py` | `<v>` token, streaming chat template, learn ranges |
| `src/streaming_emotion_llm/models/vision_live.py` | `models/vision_live.py` | SigLIP/CLIP frame encoder |
| `src/streaming_emotion_llm/models/projector.py` | `models/live_llama/modeling_live_llama.py` | Visual-to-LLM MLP connector |
| `src/streaming_emotion_llm/models/modeling_live.py` | `models/modeling_live.py` | `visual_embed`, `joint_embed`, stream evaluation, greedy generation |
| `src/streaming_emotion_llm/models/live_llama/configuration_live_llama.py` | `models/live_llama/configuration_live_llama.py` | Llama config plus streaming fields |
| `src/streaming_emotion_llm/models/live_llama/modeling_live_llama.py` | `models/live_llama/modeling_live_llama.py` | TinyLlama/Llama wrapper with frame embeddings |
| `src/streaming_emotion_llm/models/llm.py` | `models/modeling_live.py` | LoRA/PEFT setup ideas |
| `src/streaming_emotion_llm/training/arguments.py` | `models/arguments_live.py` | live1/live1+ frame-token presets |
| `src/streaming_emotion_llm/training/generation_trainer.py` | `engine/trainer_with_gen2eval.py` | Generation-based evaluation hook |
| `scripts/precompute_video_features.py` | `scripts/sanity/prepare_ego4d_subset_features.py`, `data/preprocess/encode.py`, `data/utils.py` | Full-video feature extraction |
| `scripts/train.py` | `train.py` | Training entrypoint, currently still a skeleton |
| `scripts/evaluate.py` | `evaluate.py` | Evaluation entrypoint, currently still a skeleton |
| `src/streaming_emotion_llm/inference/streaming_runner.py` | `demo/inference.py` | Future streaming inference runner |

Original code intentionally not copied wholesale:

- original dataset builders
- original Ego4D/COIN shell scripts
- demo UI assets
- original dialogue/reasoning metrics

## Current Important Files

Project overview:

```text
README.md
docs/current_emotion_token_pipeline.md
docs/ported_framework.md
docs/original_module_reuse.md
docs/environment.md
```

Configs:

```text
configs/experiments/baseline_tinyllama_siglip_lora.yaml
configs/models/tinyllama_siglip_lora.yaml
configs/data/streaming_emotion.yaml
configs/precompute/siglip_large_384_2fps.yaml
configs/inference/streaming_video.yaml
configs/eval/emotion_streaming.yaml
```

Data scripts:

```text
scripts/inspect_annotations.py
scripts/build_manifest.py
scripts/split_manifest.py
scripts/precompute_video_features.py
scripts/build_feature_subset_manifest.py
```

Model framework:

```text
src/streaming_emotion_llm/models/configuration_live.py
src/streaming_emotion_llm/models/tokenization_live.py
src/streaming_emotion_llm/models/vision_live.py
src/streaming_emotion_llm/models/projector.py
src/streaming_emotion_llm/models/modeling_live.py
src/streaming_emotion_llm/models/live_llama/modeling_live_llama.py
src/streaming_emotion_llm/models/llm.py
```

Still incomplete:

```text
src/streaming_emotion_llm/evaluation/metrics.py
src/streaming_emotion_llm/inference/streaming_runner.py
```

## Commands Already Used

Inspect annotations:

```bash
python scripts/inspect_annotations.py
```

Build valid manifest:

```bash
python scripts/build_manifest.py
```

Split valid manifest:

```bash
python scripts/split_manifest.py
```

Precompute features:

```bash
conda run -n video-mm python scripts/precompute_video_features.py --batch-size 2
```

Build subset from available features:

```bash
python scripts/build_feature_subset_manifest.py --limit 400
```

Run tests:

```bash
python -m pytest
```

Current tests pass:

```text
1 passed
```

## Next Step

Run a tiny overfit experiment on a few event-level samples, then implement generation metrics.

Target output from `StreamingEmotionDataset` + collator:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,   # only assistant emotion token(s) are learned
    "frames": ...,   # full-video precomputed SigLIP features, shape T x 10 x 1024
}
```

Dataset logic:

1. Read `data/manifests/feature_subset/train.jsonl`.
2. For each sample, load `feature_path`.
3. Use full feature sequence, capped/subsampled to `max_num_frames`.
4. Build a streaming prompt with `<v>` placeholders.
5. Use the first event, all events, or a chosen event policy to decide the target emotion.
6. For the first baseline, simplest target policy:

```text
Use the dominant or first annotated emotion for the clip.
```

Alternative better policy:

```text
Train one sample per event but still provide the full-video feature sequence.
```

Recommended first implementation:

- one training item per event
- same `feature_path` reused for all events in the clip
- target is that event's `emotion`
- full-video features are supplied
- no reasoning text
- no audio

## Cautions

- The open tabs may still include `src/streaming_emotion_llm/reproduction/__init__.py`; that file was removed earlier. Close the stale IDE tab.
- `data/raw/`, `data/processed/`, archives, checkpoints, and outputs are ignored by Git.
- Do not commit private environment names or local paths.
- Full-video visual token length is about 400-1200 tokens per sample in the current subset, so 8GB VRAM may require careful batch size and sequence length control.

