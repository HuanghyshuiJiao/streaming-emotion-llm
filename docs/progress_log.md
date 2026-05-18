# Progress Log

## 2026-05-17

### Current Baseline

- Adapted the original VideoLLM-style streaming framework to open-vocabulary facial emotion prediction.
- Current main experiment uses video-only streaming features:
  - LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - Vision features: frozen precomputed SigLIP-large-patch16-384 features
  - Training: LoRA fine-tuning with event-stream supervision
  - Data: feature subset with 320 train videos, 40 validation videos, and 40 test videos
- Main full-video event-stream validation results:
  - Emotion exact match: 5.20%
  - Emotion token accuracy: 49.77%
  - Total token accuracy: 97.99%
  - Time difference: 5.20 s
  - Fluency: 0.1714
  - LM PPL: 2.4327

### Interpretation

- The model learns interval/continuation tokens reliably.
- Fine-grained open-vocabulary emotion label prediction remains weak.
- Teacher-forced token-level evaluation can produce mixed strings such as `sadjected`, where the first predicted token is wrong but later positions recover because the ground-truth prefix is still used as input.

### Next Plan

- Increase LoRA rank to test whether the current setup can overfit the training set.
- Prefer full-video/full-data training once all features are available.
- Add autoregressive evaluation in addition to teacher-forced streaming evaluation.
- Integrate Weights & Biases for experiment tracking.
- Build a Gradio demo for qualitative streaming emotion visualization.

### LoRA Rank 32 Follow-up

- Added higher-rank LoRA configs for `r=32` and `r=64`.
- Ran the full-video event-stream `r=32, alpha=64` experiment on the feature subset.
- Training completed at `checkpoint-462`.
- Final logged training loss: 3.1399.
- Full validation streaming evaluation:
  - Emotion exact match: 6.73%
  - Emotion token accuracy: 50.38%
  - Time difference: 5.20 s
  - Fluency: 0.1714
  - LM PPL: 2.9187
  - Interval accuracy: 100%
- Full train streaming evaluation:
  - Emotion exact match: 53.11%
  - Emotion token accuracy: 75.93%
  - LM PPL: 1.2260
  - Interval accuracy: 100%
- Interpretation: increasing LoRA rank substantially improves train-set fitting while validation only improves slightly, suggesting the current setup can memorize the feature subset but still generalizes weakly on fine-grained open-vocabulary emotion labels.

### Autoregressive Evaluation, W&B, and Gradio

- Added `scripts/evaluate_autoregressive.py` for true greedy autoregressive emotion generation.
- This differs from `scripts/evaluate_generation.py`: the original script is teacher-forced streaming evaluation, while autoregressive evaluation prompts the model with `Assistant:` and does not feed the gold emotion tokens.
- `r=32` autoregressive train evaluation:
  - Emotion exact match: 52.98%
  - Emotion token accuracy: 51.01%
- `r=32` autoregressive validation evaluation:
  - Emotion exact match: 6.42%
  - Emotion token accuracy: 6.41%
- Interpretation:
  - The train-set overfitting signal remains visible under autoregressive generation.
  - Validation exact match remains low, close to the teacher-forced exact match.
  - Autoregressive token accuracy is not directly comparable to teacher-forced token accuracy, because teacher-forced token accuracy measures next-token prediction under the gold prefix, while autoregressive token accuracy compares the generated emotion string against the gold emotion string.
- Added W&B tracking support through `training.report_to: wandb` and the `tracking` config block.
- Added W&B experiment config:
  - `configs/experiments/fullvideo_lora_r32_wandb_tinyllama_siglip_rtx4060_8gb.yaml`
- Added a Gradio qualitative demo:
  - `scripts/app_gradio.py`
  - It loads a checkpoint, selects a manifest sample/event, displays the source video when available, and shows gold emotion, generated emotion, exact match, token accuracy, and the prompt.

Useful commands:

```powershell
$env:PYTHONPATH='src'
conda run -n video-mm python scripts\evaluate_autoregressive.py --config configs\experiments\fullvideo_lora_r32_tinyllama_siglip_rtx4060_8gb.yaml --checkpoint outputs\fullvideo_event_stream_tinyllama_siglip_lora_r32_rtx4060_8gb\checkpoint-462 --split val --limit 0
conda run -n video-mm python scripts\app_gradio.py --config configs\experiments\fullvideo_lora_r32_tinyllama_siglip_rtx4060_8gb.yaml --checkpoint outputs\fullvideo_event_stream_tinyllama_siglip_lora_r32_rtx4060_8gb\checkpoint-462
```
