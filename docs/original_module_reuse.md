# Original Module Reuse Plan

This project is independent from the original streaming VideoLLM reproduction, but several implementation ideas and small modules can be reused after adaptation.

## Reuse Policy

Do not copy the original repository wholesale. Prefer extracting small, well-scoped utilities and adapting them into this repository's package structure.

Before copying code:

- check license compatibility
- preserve required attribution
- remove paper-specific assumptions
- rename modules to match this project's emotion-streaming terminology
- add tests for the adapted behavior

## High-Value Modules to Adapt

### Stream Placeholder and Tokenization Logic

Original source:

```text
models/tokenization_live.py
```

Recommended target:

```text
src/streaming_emotion_llm/models/tokenization_live.py
src/streaming_emotion_llm/prompts/
```

Why useful:

- handles visual placeholder tokens such as `<v>`
- builds streaming chat templates
- computes learn ranges for interleaved stream and assistant tokens

Required changes:

- rename from `live` to `streaming_emotion`
- adapt prompts from activity-assistant dialogue to emotion understanding
- support timestamped emotion events
- keep visual/audio placeholders extensible

### Vision Encoder Utilities

Original source:

```text
models/vision_live.py
```

Recommended target:

```text
src/streaming_emotion_llm/models/vision_live.py
```

Why useful:

- SigLIP/CLIP frame normalization
- CLS and pooled spatial token extraction
- compact implementation for visual feature extraction

Required changes:

- support current SigLIP config names
- add face-crop or region-specific branches later
- make preprocessing explicit in config
- avoid hard-coding only one verified backbone

### Projector / Connector Pattern

Original source:

```text
models/live_llama/modeling_live_llama.py
```

Recommended target:

```text
src/streaming_emotion_llm/models/projector.py
src/streaming_emotion_llm/models/multimodal_model.py
```

Why useful:

- simple MLP connector from visual hidden size to LLM hidden size
- clean conceptual bridge between encoder features and language tokens

Required changes:

- decouple from `LlamaForCausalLM` inheritance
- support TinyLlama explicitly
- support future audio/video fusion outputs
- expose projector as a standalone module

### LoRA Model-Build Flow

Original source:

```text
models/modeling_live.py
models/arguments_live.py
```

Recommended target:

```text
src/streaming_emotion_llm/models/llm.py
src/streaming_emotion_llm/training/
configs/models/
```

Why useful:

- proven PEFT/LoRA setup
- separates trainable connector from frozen backbone
- already compatible with TinyLlama-style adaptation

Required changes:

- move dataclass arguments into YAML configs
- reduce hard-coded original experiment defaults
- choose LoRA target modules based on actual TinyLlama module names

### Streaming Inference State Machine

Original source:

```text
demo/inference.py
```

Recommended target:

```text
src/streaming_emotion_llm/inference/streaming_runner.py
src/streaming_emotion_llm/streaming/state.py
src/streaming_emotion_llm/streaming/triggers.py
```

Why useful:

- maintains past key values for online generation
- queues video frames and user queries
- uses model confidence on interval tokens as an event trigger

Required changes:

- remove UI/demo coupling
- replace activity dialogue query logic with emotion-event triggering
- support timestamped annotation format
- prepare for audio queue synchronization

### Generation-to-Evaluation Trainer Hook

Original source:

```text
engine/trainer_with_gen2eval.py
```

Recommended target:

```text
src/streaming_emotion_llm/training/trainer.py
```

Why useful:

- allows evaluation by generation instead of only teacher-forced loss
- useful for reasoning-generation metrics

Required changes:

- adapt input keys to this project's dataset format
- integrate emotion classification and reasoning metrics
- avoid depending on original evaluator names

## Modules to Avoid Copying Directly

### Original Dataset Builders

Avoid direct copying unless needed for comparison. They are tied to Ego4D-style supervision and the original task format. This project should use its own manifest-based dataset reader.

### Original Shell Scripts

The original `scripts/ego4d`, `scripts/coin`, and sanity scripts are useful as historical reference, but the command structure should be rewritten around this repository's configs.

### Full LiveLlama Model Class

The original `LiveLlamaForCausalLM` is useful as a reference, but copying it directly risks locking this project into the old model architecture. Extract the connector, `joint_embed`, and streaming-generation ideas instead.

### Demo UI Assets

Do not copy demo images, Gradio/UI code, or rendering assets unless building a new demo. They are not part of the research core.

## Suggested Adaptation Order

1. Adapt vision encoder utilities.
2. Adapt projector/connector module.
3. Adapt tokenizer streaming placeholders.
4. Build a project-native multimodal TinyLlama wrapper.
5. Adapt LoRA model-building flow.
6. Adapt streaming inference state machine.
7. Adapt generation-based evaluation.

This order keeps each step testable and prevents the project from becoming a renamed copy of the original repository.
