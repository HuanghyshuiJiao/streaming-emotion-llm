from pathlib import Path

from streaming_emotion_llm.config import load_config


def test_load_baseline_config():
    config = load_config(Path("configs/experiments/baseline_tinyllama_siglip_lora.yaml"))

    assert config.values["experiment"]["name"] == "emotion_token_tinyllama_siglip_lora"
    assert config.values["data"]["task"]["type"] == "open_vocab_emotion_token_prediction"
    assert config.values["data"]["streaming_window"]["mode"] == "event_stream_window"
