# Streaming Emotion LLM

Research codebase for streaming facial emotion understanding, online multimodal interaction, and temporal event-triggered response generation.

This project starts from lessons learned while reproducing an online VideoLLM-style streaming dialogue framework, but it is organized as an independent research repository. The main direction is no longer pure paper reproduction; it is a new framework for streaming facial emotion understanding, future audio-video fusion, and emotion-aware reasoning generation.

## Project Scope

- Streaming facial emotion understanding from video.
- Online multimodal interaction with temporal response triggers.
- Lightweight LLM adaptation with LoRA.
- Projector-based alignment between visual/audio encoders and LLM token space.
- Future audio-video fusion with synchronized temporal modeling.
- Future emotion prediction and natural-language reasoning generation.

## Current Baseline

- TinyLlama backbone with LoRA fine-tuning.
- SigLIP visual encoder.
- Multimodal projector for visual-to-language alignment.
- Streaming inference pipeline.
- Ego4D-style streaming supervision.

## Planned Extensions

- Audio encoder integration using Whisper, BEATs, or compatible speech/audio models.
- Synchronized audio-video feature fusion.
- Facial region-specific encoding for expression-aware representation learning.
- Emotion state prediction, transition modeling, and reasoning generation.
- Event-triggered online response generation under latency constraints.

## Repository Layout

```text
streaming-emotion-llm/
  configs/                  Experiment, model, data, training, eval, and inference configs
  data/                     Dataset manifests, annotation schemas, and preprocessing specs
  docs/                     Research notes, design docs, protocols, and paper assets
  examples/                 Minimal runnable examples and notebooks
  scripts/                  CLI entrypoints for setup, preprocessing, training, eval, inference
  src/streaming_emotion_llm/ Main research package
  tests/                    Unit and integration tests
```

## Installation

Use Python 3.10+ with a GPU-compatible PyTorch installation for training and inference.

```bash
pip install -e .
```

Optional development and audio dependencies:

```bash
pip install -e ".[dev]"
pip install -e ".[audio]"
```

Install model-specific dependencies as needed for TinyLlama, SigLIP, LoRA fine-tuning, audio encoders, and distributed training.

## Data Organization

Keep raw datasets outside git. Use manifests and annotation files to point to local data roots.

```text
data/
  manifests/
  annotations/
  schemas/
  README.md
```

Recommended dataset categories:

- `video_streaming`: frame/video-level streaming supervision.
- `facial_emotion`: facial affect labels and temporal emotion annotations.
- `audio_visual`: synchronized audio-video streams.
- `instruction_tuning`: multimodal dialogue and reasoning examples.

## Training

Example:

```bash
python scripts/train.py --config configs/experiments/baseline_tinyllama_siglip_lora.yaml
```

Training code should keep the following concerns separate:

- encoder loading and freezing policy
- projector architecture
- LoRA target modules
- streaming sampler/windowing policy
- loss definitions
- checkpointing and experiment logging

## Inference

Example:

```bash
python scripts/infer_stream.py --config configs/inference/streaming_video.yaml --input path/to/video.mp4
```

Inference should support:

- online frame ingestion
- temporal state caching
- event-triggered response generation
- emotion prediction heads
- optional audio stream ingestion

## Evaluation

Example:

```bash
python scripts/evaluate.py --config configs/eval/emotion_streaming.yaml
```

Suggested metrics:

- emotion classification accuracy/F1
- emotion transition detection
- temporal localization quality
- streaming latency
- response trigger precision/recall
- generated reasoning quality

## Main Modules

- `models`: TinyLlama/LLM loading, SigLIP visual encoding, future audio encoders, projectors, fusion modules, and prediction heads.
- `data`: manifests, dataset schemas, streaming samplers, temporal windows, and annotation utilities.
- `streaming`: online state cache, temporal event triggers, and streaming-window policies.
- `training`: LoRA fine-tuning loops, losses, optimizers, checkpointing, and logging.
- `evaluation`: emotion metrics, transition detection, trigger quality, latency, and generation quality.
- `inference`: online video and future audio-video inference runners.
- `prompts`: prompt templates for streaming dialogue, emotion prediction, and reasoning.

## Ported Framework

Core streaming VideoLLM-style components have been ported into this project as native modules, including streaming tokenization, visual encoding, projector/connector logic, a TinyLlama/Llama-compatible streaming wrapper, and generation-based evaluation hooks.

See `docs/source_code_mapping.md`, `docs/file_inventory_mapping.md`, `docs/original_vs_ported_code_review.md`, and `docs/ported_framework.md` for the current mapping and remaining adaptation work.

## Current Handoff

For the latest project state, training result, evaluation result, and next implementation step, see `docs/session_handoff_2026-05-12.md`.

## Reference Project

This repository is informed by an earlier reproduction of the original streaming VideoLLM / VideoLLM-online style project. That previous repository should be treated as background reference only. New research code in this repository should remain self-contained, with implementation decisions documented in `docs/`.

## Citation

Add citations for:

- the original streaming VideoLLM / VideoLLM-online paper and repository used as reference
- TinyLlama
- SigLIP
- LoRA / PEFT
- Whisper and BEATs when audio experiments are added
- all facial emotion, video, audio, and instruction-tuning datasets used in experiments

## License

Choose a license compatible with the original repository and all model/data dependencies before release.
