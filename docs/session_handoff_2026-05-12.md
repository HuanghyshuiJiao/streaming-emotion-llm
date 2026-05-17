# Session Handoff: 2026-05-12

This handoff summarizes the current project state after aligning the repository with the new emotion-token baseline, fixing the event-level streaming supervision to use causal prefix video context, running RTX 4060 8GB training jobs, and evaluating generated emotion tokens plus pre-response latency.

## Current Baseline

The active baseline is:

```text
precomputed SigLIP features -> event timestamp prefix window -> projector -> TinyLlama/LoRA -> open-vocabulary emotion token
```

Current baseline constraints:

- video-only
- no audio
- no reasoning generation
- no closed-set classification head
- no event trigger training yet
- 2 FPS precomputed SigLIP features
- causal `prefix_until_event` frame selection for event-level supervision
- `live1+` visual tokens: `1 CLS + 3x3 pooled spatial tokens = 10 tokens/frame`

## Environment

The project VS Code interpreter is set to:

```text
D:\Users\15373\anaconda3\envs\video-mm\python.exe
```

Confirmed in `video-mm`:

```text
torch 2.7.1+cu128
transformers 4.55.4
peft 0.19.1
accelerate 1.13.0
```

Model/tokenizer loading defaults to local cache:

```yaml
local_files_only: true
```

The training code also sets offline Hugging Face environment variables when `local_files_only` is true.

## Code Alignment With Original Repository

The project was renamed/reorganized so core files match the original `videollm-online` layout where useful:

| Current file | Original file | Status |
| --- | --- | --- |
| `src/streaming_emotion_llm/models/configuration_live.py` | `models/configuration_live.py` | aligned |
| `src/streaming_emotion_llm/models/tokenization_live.py` | `models/tokenization_live.py` | aligned |
| `src/streaming_emotion_llm/models/vision_live.py` | `models/vision_live.py` | aligned |
| `src/streaming_emotion_llm/models/modeling_live.py` | `models/modeling_live.py` | aligned |
| `src/streaming_emotion_llm/models/live_llama/configuration_live_llama.py` | `models/live_llama/configuration_live_llama.py` | aligned |
| `src/streaming_emotion_llm/models/live_llama/modeling_live_llama.py` | `models/live_llama/modeling_live_llama.py` | aligned |
| `src/streaming_emotion_llm/data/stream.py` | `data/stream.py` | adapted to event-level emotion samples |
| `src/streaming_emotion_llm/data/data_collator.py` | `data/data_collator.py` | adapted |
| `src/streaming_emotion_llm/training/generation_trainer.py` | `engine/trainer_with_gen2eval.py` | ported |
| `src/streaming_emotion_llm/training/arguments.py` | `models/arguments_live.py` | adapted |

Full inventory:

```text
docs/source_code_mapping.md
docs/file_inventory_mapping.md
```

## Data State

Full manifest after conservative annotation repair:

```text
data/manifests/all_valid.jsonl: 1534 video records / 11800 events
data/manifests/train.jsonl: 1227 video records / 9506 events
data/manifests/val.jsonl: 153 video records / 1118 events
data/manifests/test.jsonl: 154 video records / 1176 events
```

Annotation repair status:

```text
fixed mechanically repairable JSON files: 26
remaining invalid annotation files: 5
videos still missing annotation files: 12
```

Feature subset currently used:

```text
data/manifests/feature_subset/train.jsonl: 320 video records
data/manifests/feature_subset/val.jsonl: 40 video records
data/manifests/feature_subset/test.jsonl: 40 video records
```

The event-level dataset expands each video record into multiple timestamped emotion events.

Available precomputed feature files:

```text
data/processed/features/siglip_large_384_2fps_1plus3x3/: 440 .pt files
```

## Implemented Since Last Handoff

Core implementation:

- `StreamingEmotionDataset` in `src/streaming_emotion_llm/data/stream.py`
- original-style collator in `src/streaming_emotion_llm/data/data_collator.py`
- training loop in `src/streaming_emotion_llm/training/trainer.py`
- automatic checkpoint resume using latest checkpoint in output dir
- local-only model/tokenizer loading
- generation evaluation script:

```text
scripts/evaluate_generation.py
```

New configs:

```text
configs/experiments/smoke_tinyllama_siglip_lora.yaml
configs/experiments/rtx4060_8gb_tinyllama_siglip_lora.yaml
```

Removed current-baseline distractions:

```text
src/streaming_emotion_llm/models/audio.py
src/streaming_emotion_llm/models/fusion.py
src/streaming_emotion_llm/models/heads.py
src/streaming_emotion_llm/training/losses.py
configs/models/future_audio_video_fusion.yaml
```

These should be reintroduced only when audio-video fusion, auxiliary heads, or reasoning generation become active targets.

## Training Run

RTX 4060 8GB config:

```text
configs/experiments/rtx4060_8gb_tinyllama_siglip_lora.yaml
```

Key parameters:

```yaml
max_num_frames: 64
batch_size: 1
gradient_accumulation_steps: 16
lora_r: 8
lora_alpha: 16
gradient_checkpointing: true
precision: bf16
local_files_only: true
```

Original full-video training output:

```text
outputs/emotion_token_tinyllama_siglip_lora_rtx4060_8gb/
```

Saved artifacts:

```text
checkpoint-200
checkpoint-300
checkpoint-400
final
```

The first run hit the tool timeout at `checkpoint-300`; the trainer was updated to auto-resume, then training continued to completion.

Training completed:

```text
total steps: 462
epochs: 3.0
final adapter: outputs/emotion_token_tinyllama_siglip_lora_rtx4060_8gb/final
```

Observed loss trend:

```text
step 1:   195.7792
step 50:   15.4552
step 150:  14.3921
step 280:  12.9820
step 380:  11.8865
step 460:  12.4084
```

The training chain is functional and does not OOM on RTX 4060 8GB.

## Generation Evaluation

Evaluation command:

```powershell
$env:PYTHONPATH='src'
& 'D:\Users\15373\anaconda3\envs\video-mm\python.exe' scripts\evaluate_generation.py `
  --config configs\experiments\rtx4060_8gb_tinyllama_siglip_lora.yaml `
  --checkpoint outputs\emotion_token_tinyllama_siglip_lora_rtx4060_8gb\final `
  --split val `
  --limit 0
```

Original full-video validation result:

```text
val events: 327
exact match: 23 / 327 = 7.03%
```

Prediction distribution:

```text
amused: 99
calm: 64
sad: 42
confident: 40
passionate: 22
reflective: 13
somber: 12
serious: 12
frustrated: 9
resigned: 6
bitter: 4
informative: 4
```

Gold distribution:

```text
unique gold labels: 126
unique predicted labels: 12
```

Interpretation:

- The model learned to generate plausible common emotion words.
- It collapses heavily toward a small set of labels.
- Exact match is low because open-vocabulary labels are highly diverse.
- More importantly, the current dataset policy gives the same full-video feature sequence to many different event targets from the same clip.

## Main Issue Found

Current event-level training has a supervision ambiguity:

```text
same full-video feature sequence -> multiple different emotion labels
```

For one clip, different event targets may be:

```text
calm
grateful
proud
inspirational
reflective
...
```

Since the prompt does not identify the target timestamp and the visual input is identical for all events in the same clip, the model has no way to know which event emotion is requested.

This likely causes the observed label collapse.

## Recommended Next Step

Change the dataset from full-video-same-input event supervision to event-specific temporal context.

Recommended first policy:

```text
event timestamp -> use frames before the timestamp, capped to a fixed prefix window
```

Example config direction:

```yaml
event_context:
  mode: prefix_until_event
  max_num_frames: 64
  fps: 2.0
```

Alternative:

```yaml
event_context:
  mode: centered_window
  pre_seconds: 24
  post_seconds: 4
  max_num_frames: 64
  fps: 2.0
```

For online consistency, prefer `prefix_until_event` first.

After dataset change:

1. Rebuild event-level samples so each event receives distinct temporal features.
2. Train a new 4060-safe baseline.
3. Re-run `scripts/evaluate_generation.py`.
4. Then implement a normal streaming inference runner and measure pre-response latency.

Important distinction:

```text
training: event-specific feature/window by event timestamp
inference: regular KV cache for online efficiency
```

KV cache is useful for streaming inference, but it does not solve the current training ambiguity by itself.
The attempted sliding-window branch was reverted because the dense training mask did not provide real training-memory savings, and sliding KV cache only affects inference.

## Update After This Handoff

The dataset was updated to implement the recommended `prefix_until_event` policy:

```text
event timestamp -> feature frames up to that timestamp -> assistant emotion token
```

The user turn was removed from training samples. The prompt now contains only:

```text
system prompt
stream prefix
assistant emotion
```

Config output directories were changed to new `prefix` names so new runs do not resume older full-video checkpoints.

## Prefix Training Run

New RTX 4060 8GB prefix config:

```text
configs/experiments/rtx4060_8gb_tinyllama_siglip_lora.yaml
```

Training output:

```text
outputs/emotion_token_prefix_tinyllama_siglip_lora_rtx4060_8gb/
```

Saved artifacts:

```text
checkpoint-300
checkpoint-400
checkpoint-462
final
```

Training completed:

```text
total steps: 462
epochs: 3.0
runtime: 4375.28 s
final train loss: 15.1254
final adapter: outputs/emotion_token_prefix_tinyllama_siglip_lora_rtx4060_8gb/final
```

Observed loss trend:

```text
step 1:   188.4895
step 50:   15.0137
step 150:  14.2355
step 300:  12.8109
step 400:  12.1528
step 460:  11.9840
```

## Prefix Generation Evaluation

Evaluation command:

```powershell
$env:PYTHONPATH='D:\Documents\projects\multimodal-video-project\streaming-emotion-llm\src'
& 'D:\Users\15373\anaconda3\envs\video-mm\python.exe' scripts\evaluate_generation.py `
  --config configs\experiments\rtx4060_8gb_tinyllama_siglip_lora.yaml `
  --checkpoint outputs\emotion_token_prefix_tinyllama_siglip_lora_rtx4060_8gb\final `
  --split val `
  --limit 0 `
  --measure-latency `
  --output outputs\emotion_token_prefix_tinyllama_siglip_lora_rtx4060_8gb\val_generation_predictions_with_latency.jsonl
```

Validation result:

```text
val events: 327
exact match: 20 / 327 = 6.12%
```

Latency result on the local RTX 4060 run:

```text
pre_response_time_s mean=0.1665 min=0.0460 max=0.4819
generation_time_s    mean=0.2399 min=0.0821 max=0.5428
```

Prediction distribution top labels:

```text
amused: 58
calm: 46
confident: 30
passionate: 26
sad: 26
resigned: 23
frustrated: 17
encouraging: 17
determined: 13
serious: 13
```

Interpretation:

- The corrected prefix pipeline is now semantically online: each event sees only frames up to that event timestamp.
- Exact-match accuracy is still low, mainly because open-vocabulary labels are very diverse and sparse.
- Prediction collapse remains, but it is no longer caused by identical full-video inputs for multiple events.
- The latency script now records per-sample `pre_response_time_s` and `generation_time_s`.

## Commands To Reuse

Run smoke training:

```powershell
$env:PYTHONPATH='src'
conda run -n video-mm python scripts/train.py --config configs/experiments/smoke_tinyllama_siglip_lora.yaml
```

Run 4060 training:

```powershell
$env:PYTHONPATH='src'
conda run -n video-mm python scripts/train.py --config configs/experiments/rtx4060_8gb_tinyllama_siglip_lora.yaml
```

Run generation eval:

```powershell
$env:PYTHONPATH='src'
& 'D:\Users\15373\anaconda3\envs\video-mm\python.exe' scripts\evaluate_generation.py --config configs\experiments\rtx4060_8gb_tinyllama_siglip_lora.yaml --checkpoint outputs\emotion_token_prefix_tinyllama_siglip_lora_rtx4060_8gb\final --split val --limit 0 --measure-latency
```

Run tests:

```powershell
python -m pytest
```

## Current Open Items

- Add normalized/semantic metrics beyond exact match.
- Inspect whether label collapse is driven by sparse open-vocabulary labels, prompt format, or insufficient projector/LoRA capacity.
- Consider label normalization or grouped emotion taxonomy for evaluation while keeping generation open-vocabulary.
- Implement a regular streaming inference runner and keep latency measurement in the evaluation loop.

## Update 2026-05-14: Event-Stream and Full-Video Results

The current best TinyLlama/SigLIP baseline is the original-style event stream format:

```text
system -> stream frames -> assistant emotion -> stream frames -> assistant emotion ...
```

Main config:

```text
configs/experiments/rtx4060_8gb_tinyllama_siglip_lora.yaml
```

Training/evaluation artifacts:

```text
outputs/emotion_token_event_stream_tinyllama_siglip_lora_rtx4060_8gb/final
outputs/emotion_token_event_stream_tinyllama_siglip_lora_rtx4060_8gb/val_generation_predictions_with_latency.jsonl
```

Validation result on the 327-event feature subset val split:

```text
exact match: 21 / 327 = 6.42%
unique predictions: 54
top-5 prediction share: 38.53%
pre_response_time_s mean: 0.1472
generation_time_s mean: 0.2565
```

A full-video event-stream subset run was also completed with `max_num_frames: 0`.

Config and artifacts:

```text
configs/experiments/fullvideo_tinyllama_siglip_lora_rtx4060_8gb.yaml
outputs/fullvideo_event_stream_tinyllama_siglip_lora_rtx4060_8gb/final
outputs/fullvideo_event_stream_tinyllama_siglip_lora_rtx4060_8gb/val_generation_predictions_with_latency.jsonl
```

Training completed on RTX 4060 8GB without OOM:

```text
steps: 462
epochs: 3.0
runtime: 6949.47 s
final train loss: 6.9231
```

Validation result:

```text
exact match: 17 / 327 = 5.20%
unique predictions: 63
top-5 prediction share: 43.43%
pre_response_time_s mean: 0.3377
generation_time_s mean: 0.3973
```

Interpretation:

- Full-video training is feasible on 4060 8GB for the current TinyLlama subset.
- It is slower and has worse exact match than the 64-frame event-stream baseline.
- The 64-frame event window is the better current default.
- The next high-value direction is more/full feature extraction and a larger LLM backbone, not longer TinyLlama context.
