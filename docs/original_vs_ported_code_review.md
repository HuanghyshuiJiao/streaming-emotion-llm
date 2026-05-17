# Original vs Ported Code Review

This document compares the current `streaming-emotion-llm` port against the original `videollm-online` codebase.

Principle going forward:

```text
Keep original code unchanged where task semantics allow it.
Only adapt dataset format, prompts, config loading, local-cache behavior, and output/evaluation code required by the new emotion-token task.
```

## Summary

The core modeling path is mostly structurally aligned with the original repository:

```text
configuration_live.py
tokenization_live.py
vision_live.py
modeling_live.py
live_llama/configuration_live_llama.py
live_llama/modeling_live_llama.py
data/data_collator.py
```

The biggest required adaptation is `data/stream.py`, because the original dataset uses Ego4D/livechat conversations while this project uses timestamped emotion events.

The biggest mistake in the first training attempt was not in `LiveLlama` or the collator. It was the dataset semantics: the first port used the same full-video feature sequence for multiple event labels. That violated the original streaming causal assumption. The dataset first moved to:

```text
event timestamp -> feature prefix up to that timestamp -> assistant emotion
```

It has now been moved closer to the original training structure:

```text
event timestamp -> recent stream window -> stream/assistant history -> target assistant emotion
```

## File-by-File Comparison

### `models/configuration_live.py`

Original:

```text
videollm-online/models/configuration_live.py
```

Current:

```text
src/streaming_emotion_llm/models/configuration_live.py
```

Original purpose:

- Define `LiveConfigMixin`.
- Store vision backbone name, frame resolution, frame token settings, placeholder ids, interval token id, stream loss weight, and vision hidden size.

Current changes:

- Formatting changed to multiline style.
- Type hints changed from loose `str = None` style to `str | None`.
- Added compatibility alias:

```python
StreamingConfigMixin = LiveConfigMixin
```

Why changed:

- Formatting/type hints were stylistic.
- Alias was added because earlier project code temporarily used `StreamingConfigMixin`.

Should keep?

- Keep `LiveConfigMixin` exactly as the public class.
- The alias is not necessary for the original-style code path. It can be kept harmlessly for backwards compatibility, but new code should use `LiveConfigMixin`.
- No task-specific behavior should live here.

Risk:

- Low. This file is semantically equivalent to the original.

### `models/tokenization_live.py`

Original:

```text
videollm-online/models/tokenization_live.py
```

Current:

```text
src/streaming_emotion_llm/models/tokenization_live.py
```

Original purpose:

- Build the streaming chat template.
- Add `<v>` as an additional special token.
- Compute visual placeholder length.
- Compute learn ranges for stream tokens and assistant tokens.
- Attach `tokenizer.get_learn_ranges`.

Current changes:

- Imports changed from relative package imports to absolute project imports.
- Formatting changed.
- `chat_template(self, stream_placeholder_jinja2)` became `chat_template(stream_placeholder_jinja2)`.
- Added `local_files_only=True` to `AutoTokenizer.from_pretrained`.
- Added compatibility alias:

```python
build_streaming_tokenizer_and_update_config = build_live_tokenizer_and_update_config
```

What stayed the same:

- Template roles remain the same: `system`, `stream`, `user`, `assistant`.
- `add_stream_generation_prompt` still produces:

```text
]
Assistant:
```

- Learn-range logic is the same: assistant learn ranges are calculated by character offsets, then collator maps them to token labels.

Why changed:

- Absolute imports match this package layout.
- `local_files_only` prevents unwanted Hugging Face network checks on this machine.
- Alias exists only for compatibility with earlier renamed code.

Should keep?

- Keep the original template and learn-range logic.
- Keep `local_files_only` because the user has local model cache and network is restricted.
- Avoid further template changes unless a test verifies prompt and label offsets.

Risk:

- Medium. This file is offset-sensitive. Even small template changes can break label masking.
- Any prompt format change must be covered by tests showing decoded prompt and decoded supervised label tokens.

### `models/vision_live.py`

Original:

```text
videollm-online/models/vision_live.py
```

Current:

```text
src/streaming_emotion_llm/models/vision_live.py
```

Original purpose:

- Load SigLIP or CLIP vision model.
- Normalize frames.
- Extract one CLS token and/or pooled spatial tokens.
- For `live1+`, output `1 + 3x3 = 10` tokens per frame.

Current changes:

- Function names changed:

```text
_siglip_vision_encode -> encode_siglip_frames
_clip_vision_encode -> encode_clip_frames
```

- `build_live_vision` now accepts a dictionary-like config instead of only `LiveConfigMixin`.
- Vision backbone check is broader:

```python
if "siglip" in name_or_path.lower()
elif "clip" in name_or_path.lower()
```

instead of exact original model-name checks.

- Autocast changed from unconditional `torch.cuda.amp.autocast()` to `enabled=frames.is_cuda`.
- CLIP pooled side calculation was corrected to use `sqrt(num_tokens - 1)` for non-CLS spatial tokens.
- Added compatibility alias:

```python
build_vision_encoder = build_live_vision
```

Why changed:

- Dict config was used by local precompute script and inference helper.
- Broader backbone matching makes the utility less brittle.
- Conditional autocast avoids CPU autocast issues.
- CLIP side calculation was a defensive correctness fix.

Should keep?

- For SigLIP path used by this project, behavior is equivalent to the original.
- Dict config support is useful for scripts.
- The broader model matching is acceptable but should be documented.
- If strict original behavior is desired, exact model-name checks can be restored.

Risk:

- Low for current SigLIP precomputed features.
- Medium if future CLIP path is used, because original CLIP code may itself have assumptions; test before relying on it.

### `models/projector.py`

Original:

```text
No standalone file. Connector lived inside models/live_llama/modeling_live_llama.py.
```

Current:

```text
src/streaming_emotion_llm/models/projector.py
```

Original connector:

```python
torch.nn.Sequential(
    torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
    GELUActivation(config.hidden_size),
    torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
)
```

Current behavior:

- Provides `MLPProjector`.
- `build_llm_connector(vision_hidden_size, llm_hidden_size)` creates the same two-layer connector shape.

Why changed:

- Extracted as standalone module so future audio/video/fusion branches can reuse a connector pattern.

Should keep?

- This is a structural change, not a behavioral change.
- If the goal is maximum original fidelity, connector can be put back inline in `LiveLlamaForCausalLM`.
- Keeping it standalone is acceptable if we treat it as the same connector.

Risk:

- Low. The architecture is equivalent for the current model.

### `models/modeling_live.py`

Original:

```text
videollm-online/models/modeling_live.py
```

Current:

```text
src/streaming_emotion_llm/models/modeling_live.py
```

Original purpose:

- Define `LiveMixin`.
- Replace `<v>` token embeddings with projected visual embeddings.
- Provide `stream_evaluate`.
- Provide `fast_greedy_generate`.
- Build model + tokenizer + LoRA/PEFT via `build_live`.

Current changes:

- Imports changed to absolute project imports.
- `set_vision_inside` builds vision from a dict instead of passing model config directly.
- Removed explicit `torch.cuda.amp.autocast()` wrapper around inside-vision inference.
- Added `GenerationConfig.from_model_config(config)` and `_original_object_hash`.
- Added `local_files_only=True` path for model/config/tokenizer loading.
- Default `attn_implementation` changed from original `flash_attention_2` to `sdpa`.
- Added aliases:

```python
StreamingMixin = LiveMixin
build_streaming_model = build_live
```

What stayed the same:

- `visual_embed` still projects visual features and flattens them.
- `joint_embed` still replaces `<v>` positions.
- `stream_evaluate` logic is materially the same.
- `fast_greedy_generate` logic is materially the same.
- LoRA construction logic is materially the same.

Why changed:

- Absolute imports: package layout.
- Dict vision config: compatibility with current `vision_live.py`.
- `local_files_only`: required to avoid network retries.
- `GenerationConfig`: required because Transformers 4.55 probes remote `custom_generate/generate.py` unless a generation config is supplied.
- `sdpa`: safer on this environment than requiring flash-attn.

Should keep?

- Keep original `LiveMixin`, `joint_embed`, `stream_evaluate`, and LoRA flow.
- Keep `local_files_only` and local `GenerationConfig` because they are environment fixes.
- Keep `sdpa` as default for this machine.
- Do not add task logic here.

Risk:

- Medium. Model-loading changes are environment-specific but necessary here.
- The core forward/embedding semantics remain original-compatible.

### `models/live_llama/configuration_live_llama.py`

Original:

```text
videollm-online/models/live_llama/configuration_live_llama.py
```

Current:

```text
src/streaming_emotion_llm/models/live_llama/configuration_live_llama.py
```

Original purpose:

- Combine `LlamaConfig` and `LiveConfigMixin`.

Current changes:

- Class name retained as `LiveLlamaConfig`.
- Added compatibility alias:

```python
StreamingLlamaConfig = LiveLlamaConfig
```

Should keep?

- Keep `LiveLlamaConfig`.
- Alias is harmless but should not be used by new code.

Risk:

- Low.

### `models/live_llama/modeling_live_llama.py`

Original:

```text
videollm-online/models/live_llama/modeling_live_llama.py
```

Current:

```text
src/streaming_emotion_llm/models/live_llama/modeling_live_llama.py
```

Original purpose:

- Subclass `LlamaForCausalLM`.
- Add visual connector.
- Override `forward` to use `joint_embed`.
- Compute weighted LM loss.
- Provide `generate_after_embed`.
- Provide `build_live_llama`.

Current changes:

- Connector construction moved to `projector.py`.
- Imports changed to absolute package imports.
- Type hints updated.
- Added compatibility aliases:

```python
StreamingLlamaForCausalLM = LiveLlamaForCausalLM
build_streaming_llama = build_live_llama
```

What stayed the same:

- Forward flow is the same:

```text
input_ids + frames -> joint_embed -> Llama forward
```

- Loss logic is the same:

```python
v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
weight = v_mask * self.config.stream_loss_weight + ~v_mask
cross_entropy(logits.flatten(0, 1), labels.flatten())
```

- `generate_after_embed` is the same idea.

Why changed:

- Standalone connector was extracted for reuse.
- Aliases were added for compatibility with earlier project naming.

Should keep?

- The modeling behavior should remain as close to original as possible.
- Connector extraction is acceptable but not necessary.
- If strict fidelity is preferred, inline the connector exactly as original.

Risk:

- Low. Core behavior is equivalent.

### `data/data_collator.py`

Original:

```text
videollm-online/data/data_collator.py
```

Current:

```text
src/streaming_emotion_llm/data/data_collator.py
```

Original purpose:

- Tokenize batch text.
- Use offset mappings and learn ranges to create labels.
- Set non-learned positions to `-100`.
- Concatenate frame features.
- Return `sample_idxs` and optional `evaluation_kwargs`.

Current changes:

- Formatting and variable names changed.
- Evaluation kwargs behavior was restored to the original pattern: return only the first `evaluation_kwargs` when present, because generation evaluation supports batch size 1.

What stayed the same:

- Label construction algorithm is the same.
- Offset mapping logic is the same.
- Visual placeholder target correction is the same.
- Frame concatenation is the same.

Should keep?

- Training behavior is equivalent.
- Keep the original single-dict evaluation behavior for compatibility with `TrainerWithGenerationEval`.

Risk:

- Low for training.
- Low after restoring original evaluation kwargs behavior.

## Second Audit Fixes

The second audit found and fixed two places where current code had drifted from the original streaming semantics:

1. Generation prompt after a closed stream.

Problem:

```text
system + stream with add_stream_generation_prompt produced an extra closing bracket.
```

Fix:

```text
Use add_generation_prompt=True when the conversation already contains a complete stream turn.
```

2. `data_collator` evaluation kwargs.

Problem:

```text
Current collator returned evaluation_kwargs as a list.
```

Fix:

```text
Restored original batch-size-1 behavior: return batch_eval_kwargs[0].
```

Tests were added for:

```text
prefix_until_event frame ranges
no user turn
single stream close bracket before Assistant:
```

### `data/stream.py`

Original:

```text
videollm-online/data/stream.py
```

Current:

```text
src/streaming_emotion_llm/data/stream.py
```

Original purpose:

- Generic `StreamMixIn` used by original datasets.
- Accepts an already-built `conversation` and `load_ranges`.
- Loads only the requested feature ranges.
- Clips to `max_num_frames` by conversation stream turns.
- Optionally applies augmentation.
- Prepends system prompt.
- Returns:

```python
text, frames, learn_ranges
```

Current purpose:

- Reads this project's JSONL manifests.
- Expands timestamped emotion events into samples.
- Selects feature frames for each event.
- Supports two causal construction modes:

```text
prefix_until_event: one event per prompt, system + stream prefix + assistant
event_stream_window: original-style stream/assistant history inside the recent event window
```

- Builds an original-style event-stream conversation in the active mode:

```python
[
    {"role": "system", "content": system_prompt},
    {"role": "stream", "num_frames": ...},
    {"role": "assistant", "content": emotion_1, "learn": True},
    {"role": "stream", "num_frames": ...},
    {"role": "assistant", "content": emotion_2, "learn": True},
]
```

Current changes:

- Original `StreamMixIn.__getitem__(conversation, load_ranges, ...)` was not preserved as the main API.
- Added `StreamingEmotionDataset`.
- Added JSONL manifest reading.
- Added event expansion.
- Added `context_mode`.
- Added `prefix_until_event`:

```text
stop = floor(timestamp * fps) + 1
start = max(stop - max_num_frames, 0)
frames = frames[start:stop]
```

- Added `event_stream_window`:

```text
target event -> recent max_num_frames window
events inside the window -> stream segment + assistant emotion turns
eval mode -> omit target assistant and append Assistant: generation prompt
```

- Removed user prompt for the current baseline.
- Does not use original augmentation.

Why changed:

- Original dataset format is not compatible with this project's annotation files.
- Current project has timestamped emotion events, not original Ego4D dialogue/livechat turns.
- Removing user input matches the current user instruction: only system prompt plus stream should trigger response.
- `prefix_until_event` restores the streaming causal assumption for a single event.
- `event_stream_window` is closer to the original paper code because it trains on stream/assistant alternation within one sample.

Should keep?

- Keep the original temporal principle.
- This file must be adapted; it cannot remain identical to original because the data format is different.
- However, it should better preserve the original `StreamMixIn` API if possible:

```python
StreamMixIn.__getitem__(conversation=..., load_ranges=...)
```

Then `StreamingEmotionDataset` should build `conversation` and `load_ranges` and delegate to `StreamMixIn`, closer to the original design.

Risk:

- High. This file defines the task semantics.
- It needs tests for:
  - timestamp to frame range
  - no future frames
  - no user turn
  - label mask covers only assistant emotion
  - same video different events have different frame ranges when timestamps differ

### `models/arguments_live.py` vs `training/arguments.py`

Original:

```text
videollm-online/models/arguments_live.py
```

Current:

```text
src/streaming_emotion_llm/training/arguments.py
```

Original purpose:

- Dataclass `TrainingArguments` presets for `live1` and `live1+`.
- Store system prompt, datasets, LoRA settings, vision settings, frame token settings, and original Ego4D paths.

Current changes:

- Class names changed:

```text
LiveTrainingArguments -> StreamingTrainingArguments
LiveOneTrainingArguments -> StreamingOneArguments
LiveOnePlusTrainingArguments -> StreamingOnePlusArguments
```

- System prompt changed to emotion understanding.
- Ego4D-specific fields replaced by local manifest fields.
- Defaults mostly retained for live1/live1+.

Why changed:

- Current project uses YAML configs and local manifests, not original CLI dataclasses.

Should keep?

- This file is currently secondary because training uses YAML config.
- To stay closer to original, class names could be restored to `LiveTrainingArguments`, `LiveOneTrainingArguments`, and `LiveOnePlusTrainingArguments`.
- Since it is not on the active training path, this is low priority.

Risk:

- Low.

### `engine/trainer_with_gen2eval.py` vs `training/generation_trainer.py`

Original:

```text
videollm-online/engine/trainer_with_gen2eval.py
```

Current:

```text
src/streaming_emotion_llm/training/generation_trainer.py
```

Original purpose:

- Override `Trainer.prediction_step`.
- During eval, call generation/evaluator function instead of only returning logits/loss.

Current changes:

- Imports and formatting changed.
- Same conceptual behavior retained.

Should keep?

- Keep as ported.
- It is not currently the main evaluation path; `scripts/evaluate_generation.py` is used instead.

Risk:

- Medium if later using it directly, because current collator returns `evaluation_kwargs` as a list instead of original single dict.

## Project-Specific Files With No Direct Original Equivalent

These files are not direct ports:

```text
src/streaming_emotion_llm/config.py
src/streaming_emotion_llm/data/manifest.py
src/streaming_emotion_llm/prompts/templates.py
src/streaming_emotion_llm/training/trainer.py
scripts/build_manifest.py
scripts/split_manifest.py
scripts/build_feature_subset_manifest.py
scripts/inspect_annotations.py
scripts/evaluate_generation.py
configs/**/*.yaml
docs/**/*.md
tests/**/*.py
```

Reason:

- The original repository is script/dataclass/Ego4D-oriented.
- This project uses YAML configs, local timestamped emotion annotations, and event-level generation evaluation.

## Changes That Are Required

These changes should stay:

1. Local-only model/tokenizer loading.

Reason:

```text
User has local model cache; network checks fail or are undesired.
```

2. Project-specific `StreamingEmotionDataset`.

Reason:

```text
Original dataset format is not compatible with timestamped emotion events.
```

3. Event-specific causal frame selection.

Reason:

```text
Restores original streaming causality: the model can only see frames up to the response/event position.
```

4. Original-style `event_stream_window` sample construction.

Reason:

```text
Keeps the causal timestamp constraint while restoring stream -> assistant -> stream -> assistant training structure.
```

5. Emotion system prompt and no user turn.

Reason:

```text
Current task is autonomous stream emotion prediction, not user-question answering.
```

6. YAML config and RTX 4060 config.

Reason:

```text
Practical local training setup.
```

## Changes That Are Optional or Should Be Reconsidered

1. Compatibility aliases such as `StreamingMixin`, `StreamingLlamaForCausalLM`.

Recommendation:

```text
Keep only if old project code imports them. New code should use original-style Live* names.
```

2. Standalone `projector.py`.

Recommendation:

```text
Acceptable, but not required. For maximum original fidelity, inline the connector back into LiveLlamaForCausalLM.
```

3. Broad vision backbone matching in `vision_live.py`.

Recommendation:

```text
Acceptable for flexibility, but exact original model-name checks are safer if strict reproduction is desired.
```

4. Renamed training argument classes.

Recommendation:

```text
Not harmful, but restoring original Live* class names would reduce mental translation.
```

5. `data_collator` returning list of evaluation kwargs.

Recommendation:

```text
Fine for current scripts. If using original TrainerWithGenerationEval, restore original single-dict behavior for eval batch size 1.
```

## What Must Be Tested Before Further Training

Before another large run, add and run tests/dry-run checks for:

```text
timestamp -> frame_start/frame_stop
frame_stop <= floor(timestamp * fps) + 1
same video different events produce different frame ranges
prompt contains no User:
prompt contains system + stream + assistant
labels supervise only assistant emotion tokens
number of <v> placeholders == frames.shape[0] * frame_num_tokens
```

## Recommended Cleanup Plan

1. Keep original modeling files close to upstream.
2. Move all task adaptation into:

```text
data/stream.py
prompts/templates.py
configs/
scripts/evaluate_generation.py
training/trainer.py
```

3. Add tests before any new training.
4. Do not change tokenizer template, collator offset logic, `joint_embed`, or `LiveLlama.forward` unless there is a failing test proving the need.
