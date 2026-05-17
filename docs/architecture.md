# Architecture

## High-Level Pipeline

```text
video frames -> frozen SigLIP -> saved full-video features -> projector -> TinyLlama
timestamped event -> emotion token target
```

## Current Baseline

The current baseline uses precomputed full-video SigLIP features, a projector for visual-to-language alignment, and TinyLlama with LoRA for lightweight adaptation. Each timestamped annotation event becomes one training item. The model receives the full-video feature sequence, capped by `max_num_frames`, and learns only the assistant emotion token text.

No closed-set classification head, reasoning generation, audio stream, or fixed 16-frame local window is used in this baseline.

## Future Multimodal Direction

Audio support should be added as a parallel stream with explicit timestamp alignment. The fusion module should operate on synchronized video and audio features before passing compact multimodal tokens into the LLM.

## Facial Emotion Direction

Facial region-specific encoding can be added as a specialized visual branch. Possible regions include full face, eyes, mouth, and action-unit-inspired patches. These features can be fused with global visual features for emotion prediction and reasoning.
