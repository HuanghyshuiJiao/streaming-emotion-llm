# Data

Do not commit raw video, audio, or large processed features.

Recommended layout:

```text
data/
  annotations/       Small annotation examples and label maps
  manifests/         JSONL files pointing to local media paths
  schemas/           Dataset schema definitions
  raw/               Ignored local raw data
  processed/         Ignored local processed features
```

Current local layout after unpacking:

```text
data/
  raw/videos/                 Extracted `.mp4` clips from `all_videos.tar`
  annotations/responses/      Per-clip emotion/reasoning `.txt` files from `responses.zip`
  annotations/emotion_label_map.json
  manifests/all_valid.jsonl   Generated manifest for clips with valid JSON annotations
```

Manifest entries should be explicit about timing:

```json
{
  "sample_id": "example_0001",
  "video_path": "/path/to/video.mp4",
  "audio_path": "/path/to/audio.wav",
  "events": [
    {
      "start_sec": 1.2,
      "end_sec": 3.4,
      "emotion": "surprise",
      "reasoning": "The subject raises their eyebrows and opens their mouth."
    }
  ]
}
```

The current response annotation files are JSON-like arrays with this event format:

```json
{
  "timestamp": 0.031,
  "emotion": "solemn",
  "detailed_reasoning": "Long natural-language explanation.",
  "summary_reasoning": "Short cue summary."
}
```

Some files contain invalid JSON because of unescaped quotes or encoding artifacts. Use:

```bash
python scripts/inspect_annotations.py
python scripts/build_manifest.py
python scripts/split_manifest.py
```

to inspect annotation quality and generate a manifest from valid files.
