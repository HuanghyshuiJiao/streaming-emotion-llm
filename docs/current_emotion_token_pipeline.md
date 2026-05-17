# Current Pipeline: Open-Vocabulary Emotion Token Prediction

The current research stage is intentionally narrow:

- video-only input
- TinyLlama + LoRA language backbone
- precomputed SigLIP-large 384 visual features
- event-specific prefix windows up to the annotation timestamp
- projector-based visual-to-language alignment
- open-vocabulary emotion prediction as generated text tokens
- no closed-set classification head
- no reasoning generation yet
- no audio input yet

## Why Token Prediction Instead of Classification

The annotations contain open-vocabulary emotion labels such as `solemn`, `triumphant`, `contemplative`, `sarcastic`, `pained`, and many others. Treating this as a fixed 7-class emotion classification task would discard useful label nuance.

The current target should therefore be generated as language tokens:

```text
Input: recent video stream
Output: solemn
```

This keeps the task aligned with TinyLlama and leaves room for future generation tasks.

## Why Reasoning Is Disabled for Now

The annotation reasoning often mentions audio-dependent cues such as tone, pitch, voice, pace, and spoken words. Since the current model is video-only, training it to generate full reasoning would encourage it to hallucinate audio evidence.

For the first baseline, use only:

```json
{
  "timestamp": 0.031,
  "emotion": "solemn"
}
```

Ignore for training:

```json
{
  "detailed_reasoning": "...",
  "summary_reasoning": "..."
}
```

Reasoning can be re-enabled after audio encoders and audio-video fusion are added.

## Training Example Shape

A training sample is now built as an original-style stream window ending at one target event:

```text
system: You observe a streaming video of a person's face and predict emotion.
stream: <v><v>..., frames up to event 1
assistant: calm
stream: <v><v>..., frames from event 1 to event 2
assistant: sarcastic
stream: <v><v>..., frames from event 2 to target event
assistant: solemn
```

The loss is applied to assistant emotion token(s), not to reasoning text. No user turn is used for the current baseline; the model responds from the system prompt plus the video stream.

## First Dataset Policy

For each target event:

- load the corresponding video clip
- load precomputed full-video SigLIP features
- select a causal recent feature window up to the target event timestamp
- convert events inside the window into `stream -> assistant` turns
- insert visual placeholder tokens into the prompt
- train TinyLlama to generate the open-vocabulary `emotion` string

Recommended first version:

- `event_stream_window`
- 2 FPS
- maximum 64 sampled feature frames for the RTX 4060 baseline
- full-video feature precomputation with one `.pt` file per video
- `max_new_tokens: 8`
- no audio
- no reasoning
- no closed-set emotion head

## Feature Precomputation

Full-video SigLIP features are generated with:

```bash
python scripts/precompute_video_features.py --batch-size 2
```

Default output:

```text
data/processed/features/siglip_large_384_2fps_1plus3x3/
```

Each video becomes one feature file:

```text
vid_0001_clip1.pt
```

Feature tensor shape:

```text
num_sampled_frames x 10 x 1024
```

The `10` visual tokens come from `1 CLS token + 3x3 pooled spatial tokens`, matching the original `live1+` design.

The script skips existing `.pt` files by default, so it can be stopped and resumed safely. Use `--overwrite` only when regenerating features intentionally.

Progress check on Windows PowerShell:

```powershell
(Get-ChildItem data\processed\features\siglip_large_384_2fps_1plus3x3 -Filter *.pt).Count
```

For a smaller first training run, build manifests only from videos whose full-video features already exist:

```bash
python scripts/build_feature_subset_manifest.py --limit 400
```

Default subset manifests:

```text
data/manifests/feature_subset/train.jsonl
data/manifests/feature_subset/val.jsonl
data/manifests/feature_subset/test.jsonl
```

## Files Used in This Project

```text
configs/experiments/baseline_tinyllama_siglip_lora.yaml
configs/models/tinyllama_siglip_lora.yaml
src/streaming_emotion_llm/models/tokenization_live.py
src/streaming_emotion_llm/models/vision_live.py
src/streaming_emotion_llm/models/projector.py
src/streaming_emotion_llm/models/modeling_live.py
src/streaming_emotion_llm/models/live_llama/modeling_live_llama.py
src/streaming_emotion_llm/data/stream.py
src/streaming_emotion_llm/data/data_collator.py
src/streaming_emotion_llm/training/trainer.py
src/streaming_emotion_llm/training/generation_trainer.py
src/streaming_emotion_llm/evaluation/metrics.py
src/streaming_emotion_llm/inference/streaming_runner.py
scripts/precompute_video_features.py
```

## One-to-One Mapping to Original Paper Code

| Current project file | Original repository file | Role |
| --- | --- | --- |
| `src/streaming_emotion_llm/models/configuration_live.py` | `models/configuration_live.py` | Streaming config fields: visual placeholder, frame token settings, stream loss weight |
| `src/streaming_emotion_llm/models/tokenization_live.py` | `models/tokenization_live.py` | Streaming chat template, `<v>` placeholder token, learn ranges |
| `src/streaming_emotion_llm/models/vision_live.py` | `models/vision_live.py` | SigLIP/CLIP visual encoder and frame token extraction |
| `scripts/precompute_video_features.py` | `scripts/sanity/prepare_ego4d_subset_features.py`, `data/preprocess/encode.py`, `data/utils.py` | Full-video feature extraction with fixed FPS and one `.pt` file per video |
| `src/streaming_emotion_llm/models/projector.py` | `models/live_llama/modeling_live_llama.py` | MLP connector/projector from visual hidden size to LLM hidden size |
| `src/streaming_emotion_llm/models/modeling_live.py` | `models/modeling_live.py` | `visual_embed`, `joint_embed`, streaming evaluation, greedy generation |
| `src/streaming_emotion_llm/models/live_llama/configuration_live_llama.py` | `models/live_llama/configuration_live_llama.py` | Llama config plus streaming fields |
| `src/streaming_emotion_llm/models/live_llama/modeling_live_llama.py` | `models/live_llama/modeling_live_llama.py` | TinyLlama/Llama CausalLM wrapper with frame embeddings |
| `src/streaming_emotion_llm/data/stream.py` | `data/stream.py` | Event-level full-video feature samples |
| `src/streaming_emotion_llm/data/data_collator.py` | `data/data_collator.py` | Offset-based assistant-token labels and frame batching |
| `src/streaming_emotion_llm/models/llm.py` | `models/modeling_live.py` | LoRA/PEFT setup and checkpoint loading logic |
| `src/streaming_emotion_llm/training/arguments.py` | `models/arguments_live.py` | live1/live1+ frame-token presets and training defaults |
| `src/streaming_emotion_llm/training/generation_trainer.py` | `engine/trainer_with_gen2eval.py` | Generation-based evaluation hook |
| `scripts/train.py` | `train.py` | Training entrypoint, but now config-driven and project-native |
| `scripts/evaluate.py` | `evaluate.py` | Evaluation entrypoint, but adapted for emotion-token prediction |
| `src/streaming_emotion_llm/inference/streaming_runner.py` | `demo/inference.py` | Future streaming inference state machine |

## Original Files Not Used Directly

| Original file or folder | Reason |
| --- | --- |
| `data/` package from original repo | Original dataset format is Ego4D/dialogue-specific; current data uses timestamped emotion events |
| `scripts/ego4d/*`, `scripts/coin/*` | Original experiment scripts are paper-task-specific |
| `demo/app.py`, UI assets | Demo/UI is not needed for the first research baseline |
| Original reasoning/dialogue metrics | Current target is only emotion token prediction |

## Latest Training Result

The RTX 4060 8GB baseline in `configs/experiments/rtx4060_8gb_tinyllama_siglip_lora.yaml` completed 3 epochs and saved:

```text
outputs/emotion_token_tinyllama_siglip_lora_rtx4060_8gb/final
```

Generation evaluation on the validation split:

```text
val events: 327
exact match: 23 / 327 = 7.03%
```

The model collapsed to a small set of common labels because the current event-level dataset gives the same full-video feature sequence to multiple different event labels from the same clip.

## Next Implementation Step

Change `StreamingEmotionDataset` from full-video-same-input event supervision to event-specific temporal context. Prefer `prefix_until_event` first:

```text
event timestamp -> frames before timestamp, capped to max_num_frames
```

The manifest rows should still become:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,      # only assistant emotion tokens are learned
    "frames": ...,      # full-video precomputed SigLIP features
}
```

After that, re-train the 4060-safe baseline and re-run generation evaluation.

## Current Experimental Status

The dataset has moved past the original full-video-same-input supervision. The current default is `event_stream_window`, which is closer to the original online VideoLLM stream format:

```text
system -> stream -> assistant emotion -> stream -> assistant emotion
```

Current main baseline:

```text
configs/experiments/rtx4060_8gb_tinyllama_siglip_lora.yaml
outputs/emotion_token_event_stream_tinyllama_siglip_lora_rtx4060_8gb/final
```

Validation on the feature subset:

| Run | Window | Exact | Unique pred | Top-5 pred share | Pre-response mean | Generation mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Event-stream baseline | 64 frames | 6.42% | 54 | 38.53% | 0.1472s | 0.2565s |
| Event-stream full-video | full prefix | 5.20% | 63 | 43.43% | 0.3377s | 0.3973s |

Conclusion:

- The full-video run is feasible on RTX 4060 8GB, but it is slower and does not improve validation exact match.
- Keep the 64-frame event-stream setup as the default TinyLlama baseline.
- The next useful work is to precompute more/all video features and test a larger LLM backbone.
