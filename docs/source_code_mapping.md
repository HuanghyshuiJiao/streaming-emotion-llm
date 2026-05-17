# Source Code Mapping

This document keeps the current project aligned with the original `videollm-online` source tree while using the new emotion-token baseline.

For a more detailed explanation of what changed from the original code and why, see:

```text
docs/original_vs_ported_code_review.md
```

## Current Baseline

```text
full-video SigLIP features -> projector -> TinyLlama/LoRA -> open-vocabulary emotion token
```

The current baseline does not use a closed-set classification head, reasoning generation, audio fusion, or fixed 16-frame local windows.

## One-to-One Module Mapping

| Current project file | Original source file | Status | Current adaptation |
| --- | --- | --- | --- |
| `src/streaming_emotion_llm/models/configuration_live.py` | `models/configuration_live.py` | aligned name | Keeps `LiveConfigMixin` streaming fields. |
| `src/streaming_emotion_llm/models/tokenization_live.py` | `models/tokenization_live.py` | aligned name | Keeps `<v>`, chat template, and learn-range logic. |
| `src/streaming_emotion_llm/models/vision_live.py` | `models/vision_live.py` | aligned name | Keeps SigLIP/CLIP frame-token extraction for precompute/inference. |
| `src/streaming_emotion_llm/models/modeling_live.py` | `models/modeling_live.py` | aligned name | Keeps `LiveMixin`, `joint_embed`, stream eval, and LoRA build flow. |
| `src/streaming_emotion_llm/models/live_llama/configuration_live_llama.py` | `models/live_llama/configuration_live_llama.py` | aligned name | Uses `LiveLlamaConfig` with streaming fields. |
| `src/streaming_emotion_llm/models/live_llama/modeling_live_llama.py` | `models/live_llama/modeling_live_llama.py` | aligned name | Uses `LiveLlamaForCausalLM` with the visual connector. |
| `src/streaming_emotion_llm/data/stream.py` | `data/stream.py` | adapted | Replaces Ego4D dialogue stream loading with event-level emotion samples. |
| `src/streaming_emotion_llm/data/data_collator.py` | `data/data_collator.py` | adapted | Keeps offset-based label construction for assistant emotion tokens. |
| `src/streaming_emotion_llm/training/generation_trainer.py` | `engine/trainer_with_gen2eval.py` | ported | Generation-eval hook retained for later evaluation. |
| `src/streaming_emotion_llm/training/arguments.py` | `models/arguments_live.py` | adapted | Keeps live1/live1+ presets, but points to local emotion manifests. |
| `scripts/precompute_video_features.py` | `scripts/sanity/prepare_ego4d_subset_features.py`, `data/preprocess/encode.py`, `data/utils.py` | adapted | Precomputes one full-video SigLIP feature tensor per clip. |
| `scripts/train.py` | `train.py` | first pass | Wires `LiveLlamaForCausalLM`, event dataset, collator, and `Trainer`. |
| `scripts/evaluate.py` | `evaluate.py` | skeleton | Next step after tiny overfit. |

## Intentionally Removed From Current Baseline

These were only future placeholders or old-plan leftovers:

```text
src/streaming_emotion_llm/models/audio.py
src/streaming_emotion_llm/models/fusion.py
src/streaming_emotion_llm/models/heads.py
src/streaming_emotion_llm/training/losses.py
configs/models/future_audio_video_fusion.yaml
```

They should be reintroduced later only when audio-video fusion, auxiliary heads, or reasoning experiments become active implementation targets.
