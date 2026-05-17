"""Training argument presets ported from the original streaming framework."""

from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class StreamingTrainingArguments(TrainingArguments):
    live_version: str = "live1+"
    system_prompt: str = (
        "A multimodal assistant observes a streaming video of a person's face and "
        "tracks emotional state changes over time."
    )
    train_datasets: list[str] | None = None
    eval_datasets: list[str] | None = None
    stream_loss_weight: float = 1.0
    llm_pretrained: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    vision_pretrained: str = "google/siglip-large-patch16-384"
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ["connector"])
    frame_fps: int = 2
    frame_token_cls: bool | None = None
    frame_token_pooled: list[int] | None = None
    frame_resolution: int = 384
    frame_token_interval: str | None = None
    frame_token_interval_threshold: float = 0.0
    augmentation: bool = False
    attn_implementation: str = "sdpa"
    output_dir: str = "outputs/debug"
    train_manifest: str = "data/manifests/train.jsonl"
    val_manifest: str = "data/manifests/val.jsonl"
    test_manifest: str = "data/manifests/test.jsonl"


@dataclass
class StreamingOneArguments(StreamingTrainingArguments):
    live_version: str = "live1"
    frame_token_cls: bool = True
    frame_num_tokens: int = 1
    frame_token_interval: str = ""
    embed_mark: str = "2fps_384_1"
    max_num_frames: int = 7200


@dataclass
class StreamingOnePlusArguments(StreamingTrainingArguments):
    live_version: str = "live1+"
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3, 3])
    frame_num_tokens: int = 10
    embed_mark: str = "2fps_384_1+3x3"
    frame_token_interval: str = ","
    max_num_frames: int = 1200


def get_args_class(live_version: str):
    if live_version == "live1":
        return StreamingOneArguments
    if live_version == "live1+":
        return StreamingOnePlusArguments
    raise NotImplementedError(f"Unknown streaming version: {live_version}")
