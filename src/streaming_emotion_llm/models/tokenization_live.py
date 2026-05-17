"""Streaming tokenizer helpers.

Ported and adapted from the original online VideoLLM framework.
"""

from functools import partial

import torch
from transformers import AutoTokenizer

from streaming_emotion_llm.models.configuration_live import LiveConfigMixin


def get_stream_placeholder_len(num_frames: int, model_config: LiveConfigMixin) -> int:
    return (
        num_frames * model_config.frame_num_tokens * len(model_config.v_placeholder)
        + len(model_config.frame_token_interval) * (num_frames - 1)
    )


def get_stream_placeholder_jinja2(model_config: LiveConfigMixin) -> str:
    return (
        f"'{model_config.frame_token_interval}'.join("
        f"[{model_config.frame_num_tokens} * '{model_config.v_placeholder}'] "
        "* message['num_frames'])"
    )


def get_stream_learn_ranges(num_frames: int, model_config: LiveConfigMixin) -> torch.Tensor:
    len_frame_placeholder_with_interval = (
        model_config.frame_num_tokens * len(model_config.v_placeholder)
        + len(model_config.frame_token_interval)
    )
    intermediate_interval_idxs = torch.arange(
        len_frame_placeholder_with_interval,
        len_frame_placeholder_with_interval * num_frames + 1,
        len_frame_placeholder_with_interval,
    ) - len(model_config.frame_token_interval)
    len_learn = (
        len(model_config.frame_token_interval)
        if model_config.frame_token_interval
        else len(model_config.v_placeholder)
    )
    return torch.stack([intermediate_interval_idxs, intermediate_interval_idxs + len_learn], dim=1)


def chat_template(stream_placeholder_jinja2: str) -> str:
    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ bos_token + messages[0]['content'] + '\n' }}"
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{% if add_stream_query_prompt %}"
        "{{ ']\nUser: ' + message['content'] }}"
        "{% else %}"
        "{{ '\nUser: ' + message['content'] }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '\nAssistant: '  + message['content'] + eos_token }}"
        "{% elif message['role'] == 'stream' and message['num_frames'] > 0: %}"
        "{{ '\n[' + STREAM_PLACEHOLDER + ']' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '\nAssistant:' }}"
        "{% elif add_stream_prompt %}"
        "{{ '\n[' }}"
        "{% elif add_stream_generation_prompt %}"
        "{{ ']\nAssistant:' }}"
        "{% endif %}"
    )
    return template.replace("STREAM_PLACEHOLDER", stream_placeholder_jinja2)


def chat_template_transition(tokenizer):
    return {
        (None, "system"): tokenizer.bos_token,
        ("system", "user"): "\n\nUser: ",
        ("system", "stream"): "\n\n[",
        ("user", "assistant"): "\nAssistant: ",
        ("user", "stream"): "\n[",
        ("user", "user"): "\nUser: ",
        ("assistant", "user"): f"{tokenizer.eos_token}\nUser: ",
        ("assistant", "stream"): f"{tokenizer.eos_token}\n[",
        ("stream", "user"): "]\nUser: ",
        ("stream", "assistant"): "]\nAssistant: ",
        "assistant": "Assistant: ",
        "eos_token": tokenizer.eos_token,
    }


def chat_template_offsets(tokenizer):
    return {key: len(value) for key, value in chat_template_transition(tokenizer).items()}


def get_learn_ranges(
    conversation: list[dict],
    *,
    template_offsets: dict,
    model_config: LiveConfigMixin,
):
    offset = 0
    learn_ranges = []
    last_role = None
    for message in conversation:
        role = message["role"]
        offset += template_offsets[(last_role, role)]
        last_role = role
        if role == "stream":
            if message.get("learn", False):
                ranges = get_stream_learn_ranges(message["num_frames"], model_config) + offset
                ranges[-1, 1] += 1
                if not isinstance(message["learn"], bool):
                    ranges = ranges[: message["learn"]]
                learn_ranges.extend([range(r[0], r[1]) for r in ranges])
            offset += get_stream_placeholder_len(message["num_frames"], model_config)
        else:
            if role == "assistant" and message.get("learn", False):
                learn_ranges.append(
                    range(
                        offset - template_offsets["assistant"],
                        offset + len(message["content"]) + template_offsets["eos_token"],
                    )
                )
            offset += len(message["content"])
    return learn_ranges


def build_live_tokenizer_and_update_config(
    llm_pretrained: str,
    model_config: LiveConfigMixin,
    local_files_only: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        llm_pretrained,
        use_fast=True,
        padding_side="left",
        local_files_only=local_files_only,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [model_config.v_placeholder]})
    v_placeholder_id = len(tokenizer) - 1
    if model_config.frame_token_interval:
        frame_token_interval_id = tokenizer.convert_tokens_to_ids(model_config.frame_token_interval)
    else:
        frame_token_interval_id = None

    tokenizer.pad_token = tokenizer.eos_token
    model_config.update(
        {
            "v_placeholder_id": v_placeholder_id,
            "frame_token_interval_id": frame_token_interval_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
    tokenizer.chat_template = chat_template(get_stream_placeholder_jinja2(model_config))
    tokenizer.get_learn_ranges = partial(
        get_learn_ranges,
        template_offsets=chat_template_offsets(tokenizer),
        model_config=model_config,
    )
    return tokenizer


build_streaming_tokenizer_and_update_config = build_live_tokenizer_and_update_config
