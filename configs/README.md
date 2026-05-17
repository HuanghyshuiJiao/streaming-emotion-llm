# Configs

Use YAML configs to make experiments reproducible and publication-friendly.

- `models/`: backbone, SigLIP feature, projector, and LoRA settings.
- `data/`: dataset roots, manifests, sampling windows, and annotation formats.
- `training/`: optimizer, scheduler, precision, checkpointing, and logging.
- `eval/`: metric suites and benchmark protocols.
- `inference/`: full-video feature limits, decoding, and later trigger settings.
- `experiments/`: full runnable experiment configs composed from the above.

Suggested convention:

```text
{task}_{backbone}_{encoders}_{training_strategy}.yaml
```

Example:

```text
emotion_tinyllama_siglip_lora.yaml
emotion_token_tinyllama_siglip_lora.yaml
```
