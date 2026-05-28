from pathlib import Path

from streaming_emotion_llm.config import load_config


def test_load_main_experiment_config():
    config = load_config(Path("configs/experiments/exp2_r32_32videos.yaml"))

    assert config.values["experiment"]["name"] == "exp2_r32_32videos"
    assert config.values["data"]["task"]["type"] == "open_vocab_emotion_token_prediction"
    assert config.values["data"]["streaming_window"]["mode"] == "full_video_stream"
