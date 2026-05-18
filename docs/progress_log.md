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
