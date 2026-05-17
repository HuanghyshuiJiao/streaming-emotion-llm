"""Projector modules for aligning encoder features with LLM hidden space."""

import torch
from torch import nn
from transformers.activations import GELUActivation


class MLPProjector(nn.Sequential):
    """Two-layer connector used to map encoder features into the LLM hidden space."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__(
            nn.Linear(input_size, hidden_size, bias=True),
            GELUActivation(),
            nn.Linear(hidden_size, output_size, bias=True),
        )


def build_projector(config: dict) -> nn.Module:
    projector_type = config.get("type", "mlp")
    if projector_type != "mlp":
        raise ValueError(f"Unsupported projector type: {projector_type}")

    return MLPProjector(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        output_size=int(config["output_size"]),
    )


def build_llm_connector(vision_hidden_size: int, llm_hidden_size: int) -> nn.Module:
    """Compatibility connector matching the original online VideoLLM MLP pattern."""

    return MLPProjector(
        input_size=vision_hidden_size,
        hidden_size=llm_hidden_size,
        output_size=llm_hidden_size,
    )
