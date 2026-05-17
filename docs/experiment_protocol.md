# Experiment Protocol

## Required Metadata

Every experiment should record:

- git commit hash
- config file path
- dataset manifest paths
- dataset version or checksum
- model checkpoint paths
- random seed
- hardware summary
- training duration

## Recommended Baselines

- visual-only open-vocabulary emotion token prediction
- visual-only event-level emotion token prediction with full-video features
- streaming response triggering, after the token baseline works
- audio-only emotion prediction, when audio is available
- audio-video fusion with synchronized temporal modeling

## Reporting

Report both model quality and online behavior:

- exact/normalized match for generated emotion tokens
- transition detection precision/recall
- response trigger precision/recall
- average and percentile latency
- memory usage
- generated reasoning quality
