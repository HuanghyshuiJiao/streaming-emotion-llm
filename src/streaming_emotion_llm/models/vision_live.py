"""Visual encoder utilities for SigLIP, CLIP, and future facial encoders."""

import math
from functools import partial

import torch
from torch import Tensor, nn
from torchvision.transforms.functional import normalize
from transformers import AutoModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


def encode_siglip_frames(
    vision_model: nn.Module,
    frames: Tensor,
    frame_token_cls: bool,
    frame_token_pooled: tuple[int, int] | list[int] | None,
    mean: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
    rescale_factor: float = 0.00392156862745098,
) -> Tensor:
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast(enabled=frames.is_cuda):
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        spatial_tokens = None
        cls_token = None

        if frame_token_pooled:
            side = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state.reshape(
                    last_hidden_state.shape[0],
                    side,
                    side,
                    last_hidden_state.shape[-1],
                ).permute(0, 3, 1, 2),
                frame_token_pooled,
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens

        if frame_token_cls:
            cls_token = vision_outputs.pooler_output[:, None]
            if not frame_token_pooled:
                return cls_token

    return torch.cat([cls_token, spatial_tokens], dim=1)


def encode_clip_frames(
    vision_model: nn.Module,
    frames: Tensor,
    frame_token_cls: bool,
    frame_token_pooled: tuple[int, int] | list[int] | None,
    mean=OPENAI_CLIP_MEAN,
    std=OPENAI_CLIP_STD,
    rescale_factor: float = 0.00392156862745098,
) -> Tensor:
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast(enabled=frames.is_cuda):
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        spatial_tokens = None
        cls_token = None

        if frame_token_pooled:
            side = int(math.sqrt(last_hidden_state.shape[1] - 1))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state[:, 1:].reshape(
                    last_hidden_state.shape[0],
                    side,
                    side,
                    last_hidden_state.shape[-1],
                ).permute(0, 3, 1, 2),
                frame_token_pooled,
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens

        if frame_token_cls:
            cls_token = last_hidden_state[:, :1]
            if not frame_token_pooled:
                return cls_token

    return torch.cat([cls_token, spatial_tokens], dim=1)


def build_live_vision(config: dict) -> tuple[nn.Module, callable]:
    name_or_path = config.get("name_or_path") or config.get("vision_pretrained")
    if not name_or_path:
        raise ValueError("Vision config must define `name_or_path` or `vision_pretrained`.")

    model = AutoModel.from_pretrained(
        name_or_path,
        local_files_only=bool(config.get("local_files_only", True)),
    ).vision_model
    frame_token_cls = config.get("frame_token_cls", True)
    frame_token_pooled = config.get("frame_token_pooled")

    if "siglip" in name_or_path.lower():
        encoder = partial(
            encode_siglip_frames,
            frame_token_cls=frame_token_cls,
            frame_token_pooled=frame_token_pooled,
        )
    elif "clip" in name_or_path.lower():
        encoder = partial(
            encode_clip_frames,
            frame_token_cls=frame_token_cls,
            frame_token_pooled=frame_token_pooled,
        )
    else:
        raise ValueError(f"Unverified vision encoder: {name_or_path}")

    return model, encoder


build_vision_encoder = build_live_vision
