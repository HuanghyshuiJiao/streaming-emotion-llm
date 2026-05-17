# Ported Streaming Framework

This document tracks the original streaming VideoLLM framework pieces that have been ported into this independent project.

## Ported Modules

### Streaming Config

New path:

```text
src/streaming_emotion_llm/models/configuration_live.py
```

Purpose:

- stores streaming-video config fields
- keeps visual placeholder settings
- tracks frame token and interval token settings
- stores stream loss weight

### Streaming Tokenization

New path:

```text
src/streaming_emotion_llm/models/tokenization_live.py
```

Purpose:

- adds the visual placeholder token
- creates streaming chat templates
- computes stream learn ranges
- keeps the old frame-interleaved prompt mechanism available for adaptation

### Vision Encoder

New path:

```text
src/streaming_emotion_llm/models/vision_live.py
```

Purpose:

- loads SigLIP or CLIP vision backbones
- normalizes frames
- extracts CLS and pooled spatial tokens
- provides the basis for future face-region-specific encoding

### Full-Video Feature Precomputation

New path:

```text
scripts/precompute_video_features.py
```

Original references:

```text
scripts/sanity/prepare_ego4d_subset_features.py
data/preprocess/encode.py
data/utils.py
```

Purpose:

- sample each video at fixed FPS
- resize and pad frames to the SigLIP resolution
- encode the full video once
- save one `.pt` feature file per video
- write metadata for downstream timestamp-to-frame indexing

### Projector / Connector

New path:

```text
src/streaming_emotion_llm/models/projector.py
```

Purpose:

- provides the MLP connector from visual hidden size to LLM hidden size
- can be reused by visual-only, audio-video, and facial-region branches

### Streaming Model Mixin

New path:

```text
src/streaming_emotion_llm/models/modeling_live.py
```

Purpose:

- combines visual features and text embeddings
- replaces `<v>` placeholder tokens with projected frame embeddings
- keeps streaming evaluation and greedy online generation utilities

### Streaming Llama Wrapper

New paths:

```text
src/streaming_emotion_llm/models/live_llama/configuration_live_llama.py
src/streaming_emotion_llm/models/live_llama/modeling_live_llama.py
```

Purpose:

- wraps Llama/TinyLlama-style causal language models
- adds a visual connector
- supports frame embeddings through `frames=...`
- keeps stream-token weighted loss

### Training Arguments

New path:

```text
src/streaming_emotion_llm/training/arguments.py
```

Purpose:

- preserves the original live1/live1+ frame-token presets
- changes the system prompt toward facial emotion understanding
- adds manifest paths for this repository's data format

### Stream Dataset and Data Collator

New paths:

```text
src/streaming_emotion_llm/data/stream.py
src/streaming_emotion_llm/data/data_collator.py
```

Purpose:

- follows the original `data/stream.py` and `data/data_collator.py` split
- expands each timestamped emotion event into one training item
- loads full-video precomputed SigLIP features
- builds original-style text, frame tensors, and learn ranges
- applies loss only to the assistant emotion token text

### Generation Evaluation Trainer

New path:

```text
src/streaming_emotion_llm/training/generation_trainer.py
```

Purpose:

- supports evaluation through model generation
- useful for future emotion reasoning generation metrics

## Important Notes

- The ported code is not yet fully adapted to the current annotation format.
- Current annotations are timestamped emotion events with reasoning text, not original Ego4D dialogue turns.
- Current baseline uses only the open-vocabulary `emotion` field as generated token text.
- Reasoning fields are intentionally disabled until audio support is added.
- Dataset and collator code now support event-level full-video feature training.
- PEFT is required for LoRA model construction.
- Heavy model modules are not imported at package import time, so lightweight utilities can run before all training dependencies are installed.
