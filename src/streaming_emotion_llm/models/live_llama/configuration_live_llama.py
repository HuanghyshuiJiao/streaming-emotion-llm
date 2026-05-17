"""Llama config with streaming multimodal fields."""

from transformers import LlamaConfig

from streaming_emotion_llm.models.configuration_live import LiveConfigMixin


class LiveLlamaConfig(LlamaConfig, LiveConfigMixin):
    pass


StreamingLlamaConfig = LiveLlamaConfig
