"""Autoregressive generation helpers for streaming emotion checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from streaming_emotion_llm.data.stream import StreamingEmotionDataset
from streaming_emotion_llm.models.live_builder import build_live_model
from streaming_emotion_llm.models.modeling_live import fast_greedy_generate
from streaming_emotion_llm.prompts.templates import EMOTION_TOKEN_PROMPT


def get_base_model(model):
    return getattr(getattr(model, "base_model", model), "model", model)


def move_frames(
    frames: torch.Tensor | dict[str, torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(frames, dict):
        return {key: value.to(device=device, dtype=dtype) for key, value in frames.items()}
    return frames.to(device=device, dtype=dtype)


def slice_frames(
    frames: torch.Tensor | dict[str, torch.Tensor],
    stop: int | None = None,
    index: int | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if index is not None:
        if isinstance(frames, dict):
            return {key: value[index] for key, value in frames.items()}
        return frames[index]
    if stop is None:
        return frames
    if isinstance(frames, dict):
        return {key: value[:stop] for key, value in frames.items()}
    return frames[:stop]


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
    face_config = model_config.get("face_encoder", {})
    projector_config = model_config.get("projector", {})
    face_enabled = bool(face_config.get("enabled", False))
    frame_num_tokens = int(vision_config.get("frame_num_tokens", 10))
    if face_enabled:
        frame_num_tokens += int(face_config.get("frame_num_tokens", 1))

    model, tokenizer = build_live_model(
        is_training=False,
        model_family=llm_config.get("family", llm_config.get("model_family", "llama")),
        llm_pretrained=llm_config["name_or_path"],
        resume_from_checkpoint=str(checkpoint),
        attn_implementation=llm_config.get("attn_implementation", "sdpa"),
        torch_dtype=torch.bfloat16,
        local_files_only=bool(llm_config.get("local_files_only", True)),
        vision_pretrained=vision_config.get("name_or_path"),
        frame_resolution=int(vision_config.get("frame_size", 384)),
        frame_token_cls=bool(vision_config.get("frame_token_cls", True)),
        frame_token_pooled=vision_config.get("frame_token_pooled", [3, 3]),
        frame_num_tokens=frame_num_tokens,
        frame_token_interval=",",
        stream_loss_weight=float(config.get("training", {}).get("stream_loss_weight", 1.0)),
        label_loss_weight=float(config.get("training", {}).get("label_loss_weight", 1.0)),
        vision_hidden_size=int(projector_config.get("input_size", 1024)),
        face_hidden_size=int(face_config.get("feature_dim", 1024)),
        face_num_tokens=int(face_config.get("frame_num_tokens", 1)) if face_enabled else 0,
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
        timestamp_alignment=streaming_window.get("timestamp_alignment", "ceil"),
        first_stream_learn=streaming_window.get("first_stream_learn", "skip_first"),
        trailing_stream=streaming_window.get("trailing_stream", "drop"),
    )


@torch.no_grad()
def greedy_generate_ids(
    *,
    model,
    tokenizer,
    text: str,
    frames: torch.Tensor | dict[str, torch.Tensor],
    device: str,
    max_new_tokens: int,
) -> torch.Tensor:
    tokenized = tokenizer(
        [text],
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    frames = move_frames(frames, device=device, dtype=dtype)
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


@torch.no_grad()
def stream_autoregressive_features(
    *,
    model,
    tokenizer,
    frames: torch.Tensor | dict[str, torch.Tensor],
    device: str,
    system_prompt: str = EMOTION_TOKEN_PROMPT,
    fps: float = 2.0,
    max_new_tokens: int = 8,
    frame_token_interval_threshold: float = 0.85,
    max_frames: int | None = None,
) -> list[dict]:
    """Run original-style online streaming over a full feature sequence.

    The model consumes frames in order using KV cache. After each frame it
    predicts either the frame interval token, meaning "continue streaming", or
    another token, meaning "close the stream and generate an assistant emotion".
    """

    base_model = get_base_model(model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    frames = move_frames(frames, device=device, dtype=dtype)
    if max_frames is not None and max_frames > 0:
        frames = slice_frames(frames, stop=max_frames)

    hidden_size = base_model.config.hidden_size
    frame_num_tokens = int(base_model.config.frame_num_tokens)
    frame_token_interval_id = base_model.config.frame_token_interval_id
    eos_token_id = base_model.config.eos_token_id
    v_placeholder_id = base_model.config.v_placeholder_id

    start_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        add_stream_prompt=True,
        return_tensors="pt",
    ).to(device)
    added_stream_prompt_ids = tokenizer.apply_chat_template(
        [{}],
        add_stream_prompt=True,
        return_tensors="pt",
    ).to(device)
    added_stream_generation_ids = tokenizer.apply_chat_template(
        [{}],
        add_stream_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    frame_placeholder_ids = torch.full(
        (1, frame_num_tokens),
        v_placeholder_id,
        dtype=torch.long,
        device=device,
    )
    inplace_output_ids = torch.zeros(
        1,
        max_new_tokens,
        dtype=torch.long,
        device=device,
    )

    past_key_values = None
    last_ids = torch.empty((1, 0), dtype=torch.long, device=device)
    predictions = []

    num_frames = next(iter(frames.values())).shape[0] if isinstance(frames, dict) else frames.shape[0]
    for frame_index in range(num_frames):
        frame = slice_frames(frames, index=frame_index)
        if past_key_values is None:
            prefix_ids = start_ids
        elif last_ids.numel() == 1 and int(last_ids.item()) == eos_token_id:
            prefix_ids = torch.cat([last_ids, added_stream_prompt_ids], dim=1)
        else:
            prefix_ids = last_ids

        frame_embeds = base_model.joint_embed(
            input_ids=frame_placeholder_ids,
            frames={key: value.unsqueeze(0) for key, value in frame.items()}
            if isinstance(frame, dict)
            else frame.unsqueeze(0),
        ).view(1, -1, hidden_size)
        inputs_embeds = torch.cat(
            [
                base_model.get_input_embeddings()(prefix_ids).view(1, -1, hidden_size),
                frame_embeds,
            ],
            dim=1,
        )
        outputs = base_model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

        next_score = outputs.logits[:, -1:].softmax(dim=-1)
        if frame_token_interval_threshold > 0:
            interval_score = float(next_score[:, :, frame_token_interval_id].item())
            if interval_score < frame_token_interval_threshold:
                next_score[:, :, frame_token_interval_id].zero_()
        last_ids = next_score.argmax(dim=-1)
        if int(last_ids.item()) == frame_token_interval_id:
            continue

        generation_prompt_embeds = base_model.get_input_embeddings()(
            added_stream_generation_ids
        )
        output_ids, past_key_values = fast_greedy_generate(
            model=base_model,
            inputs_embeds=generation_prompt_embeds,
            past_key_values=past_key_values,
            eos_token_id=eos_token_id,
            inplace_output_ids=inplace_output_ids.zero_(),
        )
        raw_prediction = decode_generated_emotion(tokenizer, output_ids)
        predictions.append(
            {
                "frame_index": frame_index,
                "timestamp": frame_index / fps,
                "raw_prediction": raw_prediction,
                "prediction": normalize_emotion(raw_prediction),
            }
        )
        last_ids = output_ids[:, -1:]

    return predictions
