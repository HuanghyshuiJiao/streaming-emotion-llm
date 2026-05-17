# Research Plan

## Core Question

How can a lightweight online multimodal model understand facial emotion dynamics in a streaming setting and generate timely, context-aware responses?

## Axes of Development

1. Streaming visual understanding with SigLIP-style frame features.
2. Facial region-aware representation learning.
3. Lightweight LLM adaptation through LoRA.
4. Online temporal state modeling and event-triggered generation.
5. Audio-video fusion for speech, prosody, and facial expression alignment.
6. Emotion prediction plus natural-language reasoning generation.

## Baseline Experiments

- TinyLlama + SigLIP + projector + LoRA.
- Full-video precomputed SigLIP feature sequence.
- One training item per timestamped emotion event.
- Open-vocabulary emotion label generated as text.
- No classification head, no reasoning generation, and no audio in the first baseline.
- Event-triggered generation under streaming latency constraints after the token baseline works.

## Publication Hygiene

- Keep configs immutable per experiment.
- Log dataset versions and manifest checksums.
- Store prompt templates with experiment configs.
- Separate reproduced baselines from new modules.
- Report latency, memory, and model size alongside accuracy metrics.
