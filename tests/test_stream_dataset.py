import json

import torch

from streaming_emotion_llm.data.stream import StreamingEmotionDataset


class DummyTokenizer:
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in conversation:
            if message["role"] == "system":
                parts.append("<s>" + message["content"] + "\n")
            elif message["role"] == "stream":
                parts.append("\n[" + "<v>" * message["num_frames"] + "]")
            elif message["role"] == "user":
                parts.append("\nUser: " + message["content"])
            elif message["role"] == "assistant":
                parts.append("\nAssistant: " + message["content"] + "</s>")
        if add_generation_prompt:
            parts.append("\nAssistant:")
        return "".join(parts)

    def get_learn_ranges(self, conversation):
        return [range(0, 1)]


def write_manifest(path, feature_path):
    record = {
        "sample_id": "vid_test",
        "feature_path": str(feature_path),
        "events": [
            {"timestamp": 4.5, "emotion": "calm"},
            {"timestamp": 30.0, "emotion": "amused"},
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_prefix_until_event_uses_only_past_window(tmp_path):
    feature_path = tmp_path / "features.pt"
    features = torch.arange(100 * 10 * 2).reshape(100, 10, 2)
    torch.save(features, feature_path)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_path, feature_path)

    dataset = StreamingEmotionDataset(
        manifest_path,
        DummyTokenizer(),
        max_num_frames=8,
        fps=2.0,
        context_mode="prefix_until_event",
    )

    _, first_frames, _, _, first_meta = dataset[0]
    _, second_frames, _, _, second_meta = dataset[1]

    assert first_meta["timestamp"] == 4.5
    assert torch.equal(first_frames, features[2:10])
    assert second_meta["timestamp"] == 30.0
    assert torch.equal(second_frames, features[53:61])


def test_event_stream_window_builds_multi_turn_stream_conversation(tmp_path):
    feature_path = tmp_path / "features.pt"
    features = torch.arange(100 * 10 * 2).reshape(100, 10, 2)
    torch.save(features, feature_path)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_path, feature_path)

    dataset = StreamingEmotionDataset(
        manifest_path,
        DummyTokenizer(),
        max_num_frames=64,
        fps=2.0,
        context_mode="event_stream_window",
    )

    text, frames, _, _, meta = dataset[1]

    assert meta["timestamp"] == 30.0
    assert meta["frame_start"] == 0
    assert meta["frame_stop"] == 61
    assert meta["included_events"] == 2
    assert torch.equal(frames, features[:61])
    assert text.count("Assistant:") == 2
    assert "\nAssistant: calm" in text
    assert "\nAssistant: amused" in text


def test_stream_prompt_has_no_user_and_generation_prompt_has_single_close_bracket(tmp_path):
    feature_path = tmp_path / "features.pt"
    features = torch.zeros(12, 10, 2)
    torch.save(features, feature_path)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_path, feature_path)

    train_dataset = StreamingEmotionDataset(
        manifest_path,
        DummyTokenizer(),
        max_num_frames=8,
        fps=2.0,
        context_mode="event_stream_window",
    )
    train_text, _, _, _, _ = train_dataset[0]
    assert "User:" not in train_text
    assert "\nAssistant: calm" in train_text

    eval_dataset = StreamingEmotionDataset(
        manifest_path,
        DummyTokenizer(),
        max_num_frames=8,
        fps=2.0,
        context_mode="event_stream_window",
        add_generation_prompt=True,
    )
    eval_text, _, _, _, _ = eval_dataset[0]
    assert "User:" not in eval_text
    assert "]]\nAssistant:" not in eval_text
    assert eval_text.endswith("]\nAssistant:")


def test_full_video_stream_builds_one_sample_per_video_and_drops_trailing_frames(tmp_path):
    feature_path = tmp_path / "features.pt"
    features = torch.arange(100 * 10 * 2).reshape(100, 10, 2)
    torch.save(features, feature_path)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(manifest_path, feature_path)

    dataset = StreamingEmotionDataset(
        manifest_path,
        DummyTokenizer(),
        max_num_frames=0,
        fps=2.0,
        context_mode="full_video_stream",
        timestamp_alignment="ceil",
        trailing_stream="drop",
    )

    text, frames, _, _, meta = dataset[0]

    assert len(dataset) == 1
    assert meta["event_index"] is None
    assert meta["included_events"] == 2
    assert meta["frame_stop"] == 61
    assert torch.equal(frames, features[:61])
    assert text.count("Assistant:") == 2
    assert "\nAssistant: calm" in text
    assert "\nAssistant: amused" in text
