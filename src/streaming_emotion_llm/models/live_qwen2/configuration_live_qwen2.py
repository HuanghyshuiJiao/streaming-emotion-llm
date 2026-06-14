"""Qwen2 config with streaming multimodal fields."""

from transformers import Qwen2Config

from streaming_emotion_llm.models.configuration_live import LiveConfigMixin


class LiveQwen2Config(Qwen2Config, LiveConfigMixin):
    pass


StreamingQwen2Config = LiveQwen2Config
