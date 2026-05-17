# File Inventory and Original Repository Mapping

This document records what each current project file does and whether it has a direct counterpart in the original `videollm-online` repository.

## Core Model Code

| Current file | Purpose | Original repository counterpart |
| --- | --- | --- |
| `src/streaming_emotion_llm/models/configuration_live.py` | Streaming config fields: `<v>`, frame tokens, interval token, stream loss weight. | `models/configuration_live.py` |
| `src/streaming_emotion_llm/models/tokenization_live.py` | Streaming chat template, visual placeholder token, and learn-range construction. | `models/tokenization_live.py` |
| `src/streaming_emotion_llm/models/vision_live.py` | SigLIP/CLIP visual encoding and CLS plus pooled spatial frame-token extraction. | `models/vision_live.py` |
| `src/streaming_emotion_llm/models/modeling_live.py` | `LiveMixin`, `visual_embed`, `joint_embed`, stream evaluation, greedy generation, LoRA build flow. | `models/modeling_live.py` |
| `src/streaming_emotion_llm/models/live_llama/configuration_live_llama.py` | Llama/TinyLlama config with live streaming fields. | `models/live_llama/configuration_live_llama.py` |
| `src/streaming_emotion_llm/models/live_llama/modeling_live_llama.py` | Llama/TinyLlama CausalLM wrapper that accepts visual frame features through `frames`. | `models/live_llama/modeling_live_llama.py` |
| `src/streaming_emotion_llm/models/live_llama/__init__.py` | Exports `LiveLlamaForCausalLM` and `build_live_llama`. | `models/live_llama/__init__.py` |
| `src/streaming_emotion_llm/models/projector.py` | Standalone MLP connector from visual hidden size to LLM hidden size. | Connector logic originally lived inside `models/live_llama/modeling_live_llama.py`. |
| `src/streaming_emotion_llm/models/llm.py` | General LLM/LoRA helper utilities kept as a lightweight fallback. | Inspired by `models/modeling_live.py`, not a direct copy. |
| `src/streaming_emotion_llm/models/__init__.py` | Lightweight package entrypoint that avoids importing heavy model dependencies automatically. | Same package role as `models/__init__.py`, but project-specific. |

## Data, Training, Evaluation, and Inference Code

| Current file | Purpose | Original repository counterpart |
| --- | --- | --- |
| `src/streaming_emotion_llm/data/stream.py` | Event-level dataset. Each timestamped emotion event becomes one sample using the full-video feature tensor. | Adapted from `data/stream.py`. |
| `src/streaming_emotion_llm/data/data_collator.py` | Batches text, frames, and labels. Uses offset mapping so only assistant emotion tokens receive LM loss. | `data/data_collator.py` |
| `src/streaming_emotion_llm/data/manifest.py` | JSONL manifest reader utilities for this project's manifest format. | Project-specific. |
| `src/streaming_emotion_llm/data/__init__.py` | Data package marker. | Package role only. |
| `src/streaming_emotion_llm/training/arguments.py` | live1/live1+ argument presets adapted to emotion manifests. | `models/arguments_live.py` |
| `src/streaming_emotion_llm/training/generation_trainer.py` | Generation-based evaluation hook. | `engine/trainer_with_gen2eval.py` |
| `src/streaming_emotion_llm/training/trainer.py` | Builds `LiveLlamaForCausalLM`, event dataset, collator, and Hugging Face `Trainer`. | Replaces the original top-level `train.py` responsibility. |
| `src/streaming_emotion_llm/training/__init__.py` | Training package marker. | Package role only. |
| `src/streaming_emotion_llm/evaluation/metrics.py` | Placeholder for emotion-token exact/normalized match metrics. | Project-specific because original metrics target dialogue/reasoning tasks. |
| `src/streaming_emotion_llm/evaluation/__init__.py` | Evaluation package marker. | Package role only. |
| `src/streaming_emotion_llm/inference/streaming_runner.py` | Placeholder for future streaming inference state machine. | Future adaptation target from `demo/inference.py`. |
| `src/streaming_emotion_llm/inference/__init__.py` | Inference package marker. | Package role only. |
| `src/streaming_emotion_llm/streaming/state.py` | Future online stream state/cache utilities. | Inspired by `demo/inference.py`. |
| `src/streaming_emotion_llm/streaming/triggers.py` | Future temporal event trigger policies. | Related to original stream interval trigger behavior, but not active in current baseline. |
| `src/streaming_emotion_llm/streaming/__init__.py` | Streaming package marker. | Package role only. |
| `src/streaming_emotion_llm/prompts/templates.py` | Emotion-token prompt templates. | Project-specific. |
| `src/streaming_emotion_llm/prompts/__init__.py` | Prompt package marker. | Package role only. |
| `src/streaming_emotion_llm/config.py` | YAML config loader. | Project-specific. |
| `src/streaming_emotion_llm/__init__.py` | Main package marker. | Package role only. |

## Scripts

| Current file | Purpose | Original repository counterpart |
| --- | --- | --- |
| `scripts/precompute_video_features.py` | Samples videos, runs SigLIP, and saves one full-video `.pt` feature tensor per clip. | Adapted from `scripts/sanity/prepare_ego4d_subset_features.py`, `data/preprocess/encode.py`, and `data/utils.py`. |
| `scripts/train.py` | Loads YAML experiment config and calls the project training entrypoint. | Original top-level `train.py`. |
| `scripts/evaluate.py` | Evaluation entrypoint skeleton. | Original evaluation scripts, but current metric target is project-specific. |
| `scripts/infer_stream.py` | Inference entrypoint skeleton. | `demo/cli.py` and `demo/inference.py`. |
| `scripts/inspect_annotations.py` | Inspects current emotion annotation JSON-like files and reports quality/counts. | Project-specific. |
| `scripts/build_manifest.py` | Builds valid video/annotation manifests for this dataset. | Project-specific. |
| `scripts/split_manifest.py` | Splits manifests into train/val/test. | Project-specific. |
| `scripts/build_feature_subset_manifest.py` | Builds a small manifest subset from clips whose feature files already exist. | Project-specific. |
| `scripts/preprocess.py` | General preprocessing entrypoint placeholder. | Related in role to `data/preprocess/*`. |

## Configs

| Current file | Purpose | Original repository counterpart |
| --- | --- | --- |
| `configs/experiments/baseline_tinyllama_siglip_lora.yaml` | Main current experiment: TinyLlama, SigLIP full-video features, projector, LoRA, emotion-token target. | YAML equivalent of original shell-script experiment settings such as `scripts/ego4d/live1+.sh`. |
| `configs/models/tinyllama_siglip_lora.yaml` | Model/backbone/projector/LoRA settings. | Original training arguments and shell parameters. |
| `configs/data/streaming_emotion.yaml` | Dataset manifests, feature directory, full-video mode, max frames. | Project-specific. |
| `configs/precompute/siglip_large_384_2fps.yaml` | SigLIP feature precompute settings. | Original preprocess settings. |
| `configs/training/lora_finetune.yaml` | Optimizer, scheduler, precision, logging, checkpoint settings. | Original training script arguments. |
| `configs/eval/emotion_streaming.yaml` | Emotion-token evaluation settings. | Project-specific. |
| `configs/inference/streaming_video.yaml` | Baseline inference settings. Triggering is disabled for the first baseline. | Future adaptation from `demo/inference.py`. |
| `configs/README.md` | Config organization notes. | Project-specific. |

## Documentation

| Current file | Purpose | Original repository counterpart |
| --- | --- | --- |
| `README.md` | Project overview and current baseline. | Replaces the original README for this independent project. |
| `docs/source_code_mapping.md` | Compact mapping of core ported modules to original files. | Project-specific. |
| `docs/file_inventory_mapping.md` | Full file inventory and original-repository correspondence. | Project-specific. |
| `docs/ported_framework.md` | Describes the ported streaming framework pieces. | Project-specific. |
| `docs/current_emotion_token_pipeline.md` | Current full-video open-vocabulary emotion-token pipeline. | Project-specific. |
| `docs/session_handoff_2026-05-09.md` | Handoff/progress record. | Project-specific. |
| `docs/original_module_reuse.md` | Which original modules to reuse, adapt, or avoid. | Project-specific. |
| `docs/architecture.md` | Current and future architecture notes. | Project-specific. |
| `docs/research_plan.md` | Research direction and experiment roadmap. | Project-specific. |
| `docs/experiment_protocol.md` | Experiment metadata/reporting protocol. | Project-specific. |
| `docs/environment.md` | Environment policy and install notes. | Project-specific. |
| `docs/references.md` | Reference/citation notes. | Project-specific. |

## Data and Generated Artifacts

| Current path | Purpose | Original repository counterpart |
| --- | --- | --- |
| `data/annotations/responses/*.txt` | Timestamped emotion event annotations. | No direct counterpart; original uses Ego4D/livechat formats. |
| `data/manifests/*.jsonl` | Valid sample manifests and train/val/test splits. | Project-specific. |
| `data/manifests/feature_subset/*.jsonl` | Small splits built only from clips with existing feature files. | Project-specific. |
| `data/processed/features/.../*.pt` | Precomputed SigLIP full-video features. | Same idea as original pre-extracted visual features, with project-specific paths and manifests. |
| `data/schemas/streaming_emotion.schema.json` | Schema for current emotion annotations. | Project-specific. |
| `data/README.md` | Data organization notes. | Same role as original `data/README.md`, different content. |
| `outputs/*.log` | Local precompute logs. | Project-specific generated artifacts. |

## Tests and Project Files

| Current file | Purpose | Original repository counterpart |
| --- | --- | --- |
| `tests/conftest.py` | Adds `src` to import path during tests. | Project-specific. |
| `tests/test_config.py` | Tests config loading and baseline config fields. | Project-specific. |
| `pyproject.toml` | Python package metadata and dependencies. | Project-specific packaging. |
| `.gitignore` | Ignores raw/processed data, outputs, cache, checkpoints, and local clutter. | Same role as original ignore files. |
| `.vscode/settings.json` | Generic local IDE settings. | Project-specific. |

