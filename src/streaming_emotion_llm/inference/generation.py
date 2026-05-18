"""Autoregressive generation helpers for streaming emotion checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from streaming_emotion_llm.data.stream import StreamingEmotionDataset
from streaming_emotion_llm.models.live_llama import build_live_llama
from streaming_emotion_llm.prompts.templates import EMOTION_TOKEN_PROMPT


def get_base_model(model):
    return getattr(getattr(model, "base_model", model), "model", model)


def normalize_emotion(text: str) -> str:
    text = text.strip().lower()
    text = text.splitlines()[0] if text else text
    for stop in [".", ",", ";", ":", "</s>"]:
        text = text.split(stop)[0]
    return text.strip().strip('"').strip("'")


def load_streaming_model(config: dict, checkpoint: str | Path):
    model_config = config["model"]
    llm_config = model_config["llm"]
    vision_config = model_config["vision_encoder"]
    projector_config = model_config.get("projector", {})

    model, tokenizer = build_live_llama(
        is_training=False,
        llm_pretrained=llm_config["name_or_path"],
        resume_from_checkpoint=str(checkpoint),
        attn_implementation=llm_config.get("attn_implementation", "sdpa"),
        torch_dtype=torch.bfloat16,
        local_files_only=bool(llm_config.get("local_files_only", True)),
        vision_pretrained=vision_config.get("name_or_path"),
        frame_resolution=int(vision_config.get("frame_size", 384)),
        frame_token_cls=bool(vision_config.get("frame_token_cls", True)),
        frame_token_pooled=vision_config.get("frame_token_pooled", [3, 3]),
        frame_num_tokens=int(vision_config.get("frame_num_tokens", 10)),
        frame_token_interval=",",
        stream_loss_weight=1.0,
        vision_hidden_size=int(projector_config.get("input_size", 1024)),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, tokenizer, device


def build_generation_dataset(
    config: dict,
    tokenizer,
    *,
    split: str,
) -> StreamingEmotionDataset:
    data_config = config["data"]
    streaming_window = data_config["streaming_window"]
    manifest_key = f"{split}_manifest"
    return StreamingEmotionDataset(
        data_config[manifest_key],
        tokenizer,
        is_training=False,
        system_prompt=EMOTION_TOKEN_PROMPT,
        max_num_frames=int(streaming_window.get("max_num_frames", 64)),
        fps=float(streaming_window.get("fps", 2.0)),
        context_mode=streaming_window.get("mode", "prefix_until_event"),
        add_generation_prompt=True,
    )


@torch.no_grad()
def greedy_generate_ids(
    *,
    model,
    tokenizer,
    text: str,
    frames: torch.Tensor,
    device: str,
    max_new_tokens: int,
) -> torch.Tensor:
    tokenized = tokenizer(
        [text],
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    frames = frames.to(device=device, dtype=dtype)
    base_model = get_base_model(model)

    outputs = base_model(
        input_ids=tokenized.input_ids,
        frames=frames,
        return_dict=True,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1:].argmax(dim=-1)
    generated = []
    for _ in range(max_new_tokens):
        generated.append(next_token)
        if int(next_token.item()) == tokenizer.eos_token_id:
            break
        inputs_embeds = base_model.get_input_embeddings()(next_token)
        outputs = base_model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
    return torch.cat(generated, dim=1) if generated else torch.empty((1, 0), device=device)


def decode_generated_emotion(tokenizer, generated_ids: torch.Tensor) -> str:
    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )


def token_overlap_metrics(tokenizer, prediction: str, gold: str) -> tuple[int, int]:
    pred_ids = tokenizer(prediction, add_special_tokens=False).input_ids
    gold_ids = tokenizer(gold, add_special_tokens=False).input_ids
    total = len(gold_ids)
    correct = sum(
        int(pred_id == gold_id)
        for pred_id, gold_id in zip(pred_ids[:total], gold_ids)
    )
    return correct, total
