"""Configuration mixin for streaming multimodal models.

Ported and adapted from the original online VideoLLM framework.
"""

from transformers import PretrainedConfig


class LiveConfigMixin(PretrainedConfig):
    def __init__(
        self,
        *,
        vision_pretrained: str | None = None,
        frame_resolution: int | None = None,
        frame_token_cls: bool | None = None,
        frame_token_pooled: list[int] | None = None,
        frame_num_tokens: int | None = None,
        v_placeholder: str = "<v>",
        frame_token_interval: str | None = None,
        v_placeholder_id: int | None = None,
        frame_token_interval_id: int | None = None,
        stream_loss_weight: float = 1.0,
        vision_hidden_size: int = 1024,
        face_hidden_size: int = 1024,
        face_num_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_pretrained = vision_pretrained
        self.frame_resolution = frame_resolution
        self.frame_token_cls = frame_token_cls
        self.frame_token_pooled = frame_token_pooled
        self.frame_num_tokens = frame_num_tokens
        self.vision_hidden_size = vision_hidden_size
        self.face_hidden_size = face_hidden_size
        self.face_num_tokens = face_num_tokens
        self.stream_loss_weight = stream_loss_weight
        self.v_placeholder = v_placeholder
        self.frame_token_interval = frame_token_interval
        self.v_placeholder_id = v_placeholder_id
        self.frame_token_interval_id = frame_token_interval_id


StreamingConfigMixin = LiveConfigMixin
