from types import SimpleNamespace

import torch
import pytest
from torch import nn

pytest.importorskip("peft")

from streaming_emotion_llm.models.modeling_live import LiveMixin


class DummyLiveModel(nn.Module):
    pass


def test_visual_embed_routes_face_features_through_face_connector():
    model = DummyLiveModel()
    model.config = SimpleNamespace(frame_num_tokens=3, face_num_tokens=1)
    model.connector = nn.Linear(4, 5)
    model.face_connector = nn.Linear(2, 5)
    model.dtype = torch.float32

    frames = {
        "vision": torch.randn(2, 2, 4),
        "face": torch.randn(2, 1, 2),
    }

    embeds = LiveMixin.visual_embed(model, frames)

    assert embeds.shape == (2 * 3, 5)


def test_visual_embed_ignores_face_features_when_face_connector_is_disabled():
    model = DummyLiveModel()
    model.config = SimpleNamespace(frame_num_tokens=2, face_num_tokens=0)
    model.connector = nn.Linear(4, 5)
    model.dtype = torch.float32

    frames = {
        "vision": torch.randn(2, 2, 4),
        "face": torch.randn(2, 1, 2),
    }

    embeds = LiveMixin.visual_embed(model, frames)

    assert embeds.shape == (2 * 2, 5)
