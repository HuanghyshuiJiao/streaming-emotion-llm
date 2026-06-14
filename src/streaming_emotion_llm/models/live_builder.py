"""Architecture selector for streaming causal language model wrappers."""

from streaming_emotion_llm.models.live_llama import build_live_llama
from streaming_emotion_llm.models.live_qwen2 import build_live_qwen2


def build_live_model(*, model_family: str = "llama", **kwargs):
    family = model_family.lower().replace("-", "_")
    if family in {"llama", "llama2", "llama3"}:
        return build_live_llama(**kwargs)
    if family in {"qwen2", "qwen2_5", "qwen"}:
        return build_live_qwen2(**kwargs)
    raise ValueError(f"Unsupported live model family: {model_family}")
