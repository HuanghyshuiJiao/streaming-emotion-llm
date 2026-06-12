"""Streaming Llama wrapper.

Ported and adapted from the original online VideoLLM LiveLlama wrapper.
"""

import torch
from torch import nn
from transformers import LlamaForCausalLM

from streaming_emotion_llm.models.live_llama.configuration_live_llama import LiveLlamaConfig
from streaming_emotion_llm.models.modeling_live import LiveMixin, build_live
from streaming_emotion_llm.models.projector import build_llm_connector


class LiveLlamaForCausalLM(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)
        self.connector = build_llm_connector(
            vision_hidden_size=config.vision_hidden_size,
            llm_hidden_size=config.hidden_size,
        )
        if config.face_num_tokens > 0:
            self.face_connector = build_llm_connector(
                vision_hidden_size=config.face_hidden_size,
                llm_hidden_size=config.hidden_size,
            )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        frames: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)
        outputs = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        loss = None
        if labels is not None:
            logits = outputs[0]
            flat_input_ids = input_ids.flatten(0, 1)
            flat_labels = labels.flatten()
            valid_mask = flat_labels >= 0
            stream_mask = flat_input_ids == self.config.v_placeholder_id
            weight = torch.ones_like(flat_labels, dtype=logits.dtype)
            weight = torch.where(
                stream_mask,
                weight * float(self.config.stream_loss_weight),
                weight,
            )
            weight = torch.where(
                valid_mask & ~stream_mask,
                weight * float(getattr(self.config, "label_loss_weight", 1.0)),
                weight,
            )
            token_loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                flat_labels,
                reduction="none",
            )
            weighted_loss = token_loss * weight
            loss = weighted_loss.sum() / weight[valid_mask].sum().clamp_min(1.0)

        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs

        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, frames), **kwargs)


def build_live_llama(**kwargs):
    return build_live(
        config_class=LiveLlamaConfig,
        model_class=LiveLlamaForCausalLM,
        **kwargs,
    )


StreamingLlamaForCausalLM = LiveLlamaForCausalLM
build_streaming_llama = build_live_llama
