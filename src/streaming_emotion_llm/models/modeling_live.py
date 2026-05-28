"""Streaming multimodal model mixin.

Ported and adapted from the original online VideoLLM framework.
"""

import copy
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, Cache, GenerationConfig
from transformers.utils import logging

from streaming_emotion_llm.models.tokenization_live import build_live_tokenizer_and_update_config
from streaming_emotion_llm.models.vision_live import build_live_vision

logger = logging.get_logger(__name__)


class LiveMixin(AutoModelForCausalLM):
    def set_vision_inside(self):
        logger.warning_once(
            "Setting the vision encoder inside the model is intended for inference. "
            "For efficient training, prefer pre-extracted visual features."
        )
        vision_config = {
            "name_or_path": self.config.vision_pretrained,
            "frame_token_cls": self.config.frame_token_cls,
            "frame_token_pooled": self.config.frame_token_pooled,
        }
        self.vision_encoder, self.vision_encode = build_live_vision(vision_config)

    def unset_vision_inside(self):
        del self.vision_encoder
        del self.vision_encode

    def visual_embed(self, frames: torch.Tensor):
        if hasattr(self, "vision_encode"):
            frames = self.vision_encode(self.vision_encoder, frames)
            frames = frames.to(self.dtype)
        else:
            frames = frames.to(dtype=self.dtype)
        frames = self.connector(frames)
        return frames.view(-1, frames.shape[-1])

    def joint_embed(
        self,
        input_ids: torch.Tensor | None = None,
        frames: torch.Tensor | None = None,
    ):
        if frames is None:
            return self.get_input_embeddings()(input_ids)
        if input_ids is None:
            return self.visual_embed(frames)

        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size - 1))
        v_mask = input_ids == self.config.v_placeholder_id
        if v_mask.any():
            inputs_embeds[v_mask] = self.visual_embed(frames).to(dtype=inputs_embeds.dtype)
        return inputs_embeds

    @torch.no_grad()
    def stream_evaluate(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        frames: torch.Tensor,
        ignore_token_id: int = -100,
        frame_token_interval_threshold: float = 0.0,
        **kwargs,
    ):
        assert input_ids.size(0) == labels.size(0) == 1
        input_id, label = input_ids[0], labels[0]
        device = input_id.device
        zero = torch.tensor(0, dtype=torch.int, device=device)
        one = torch.tensor(1, dtype=torch.int, device=device)

        turn_stops = ((input_id == self.config.eos_token_id).nonzero() + 1)[:, 0].tolist()
        turn_starts = [0] + turn_stops[:-1]
        num_turns = len(turn_starts)

        outputs = self.forward(input_ids=input_ids, frames=frames, return_dict=True, use_cache=True)
        logit, past_key_values = outputs.logits[0], outputs.past_key_values

        v_placeholder_id = self.config.v_placeholder_id
        use_interval = self.config.frame_token_interval_id is not None
        frame_token_interval_id = (
            self.config.frame_token_interval_id if use_interval else self.config.eos_token_id
        )
        frame_num_tokens = self.config.frame_token_cls
        if self.config.frame_token_pooled:
            frame_num_tokens += self.config.frame_token_pooled[0] * self.config.frame_token_pooled[1]

        past_num_frames = 0
        lm_ppls, frame_diffs, fluencies, lm_correctness = [], [], [], []
        for turn_index, (turn_start, turn_stop) in enumerate(zip(turn_starts, turn_stops)):
            turn_label = label[turn_start:turn_stop]
            turn_learn_mask = turn_label != ignore_token_id
            if not turn_learn_mask.any():
                continue

            turn_logit = logit[turn_start:turn_stop]
            turn_input_id = input_id[turn_start:turn_stop]
            turn_v_mask = turn_input_id == v_placeholder_id
            turn_num_frames = turn_v_mask.sum() // frame_num_tokens
            turn_stream_mask = turn_v_mask & turn_learn_mask
            turn_lm_mask = turn_learn_mask & ~turn_stream_mask

            if turn_lm_mask.any():
                turn_lm_masked_logit = turn_logit[turn_lm_mask]
                turn_lm_masked_label = turn_label[turn_lm_mask]
                lm_ppl = torch.nn.functional.cross_entropy(
                    turn_lm_masked_logit,
                    turn_lm_masked_label,
                ).exp()
                lm_ppls.append(lm_ppl)
                wrong_mask = turn_lm_masked_logit.argmax(dim=-1) != turn_lm_masked_label
                if wrong_mask.any():
                    num_lm_correct_tokens = wrong_mask.nonzero()[0, 0]
                else:
                    num_lm_correct_tokens = (~wrong_mask).sum()
                lm_correctness.append(num_lm_correct_tokens / turn_lm_masked_label.numel())

            if turn_stream_mask.any():
                turn_score = turn_logit.softmax(dim=-1)
                turn_stream_masked_score = turn_score[turn_stream_mask]
                if frame_token_interval_threshold > 0:
                    lower_threshold_mask = (
                        turn_stream_masked_score[:, frame_token_interval_id]
                        < frame_token_interval_threshold
                    )
                    turn_stream_masked_score[lower_threshold_mask] = 0
                pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id
                if pred_mask.any():
                    frame_diff = turn_stream_mask.sum() - pred_mask.nonzero()[0, 0] - 1
                else:
                    turn_last_stream_idx = turn_stream_mask.nonzero()[-1, 0]
                    past_before_assistant = self.trim_past_key_values(
                        past_key_values,
                        0,
                        turn_start + turn_last_stream_idx + 1,
                    )
                    if turn_index == num_turns - 1:
                        frame_diff = zero
                    else:
                        next_turn_num_frames = (
                            input_id[turn_starts[turn_index + 1] : turn_stops[turn_index + 1]]
                            == v_placeholder_id
                        ).sum() // frame_num_tokens
                        to_append_num_frames = min(next_turn_num_frames, turn_num_frames - 1)
                        if to_append_num_frames == 0:
                            frame_diff = zero
                        else:
                            to_append_frames = frames[
                                past_num_frames
                                + turn_num_frames : past_num_frames
                                + turn_num_frames
                                + to_append_num_frames
                            ]
                            frame_placeholder = [v_placeholder_id] * frame_num_tokens
                            if use_interval:
                                frame_placeholder = [frame_token_interval_id] + frame_placeholder
                            to_append_input_id = torch.tensor(
                                frame_placeholder * to_append_num_frames,
                                dtype=torch.long,
                                device=device,
                            )
                            to_append_logit = self.forward(
                                input_ids=to_append_input_id[None],
                                past_key_values=past_before_assistant,
                                frames=to_append_frames,
                                return_dict=True,
                                use_cache=True,
                            ).logits[0]
                            idxs = torch.arange(
                                len(frame_placeholder) - 1,
                                len(to_append_input_id),
                                len(frame_placeholder),
                                device=device,
                            )
                            to_append_score = to_append_logit[idxs].softmax(dim=-1)
                            if frame_token_interval_threshold > 0:
                                lower_threshold_mask = (
                                    to_append_score[:, frame_token_interval_id]
                                    < frame_token_interval_threshold
                                )
                                to_append_score[lower_threshold_mask] = 0
                            to_append_pred_mask = (
                                to_append_score.argmax(dim=-1) != frame_token_interval_id
                            )
                            if to_append_pred_mask.any():
                                frame_diff = -(to_append_pred_mask.nonzero()[0, 0] + 1)
                            else:
                                frame_diff = -to_append_num_frames
                frame_diffs.append(frame_diff.abs())

            if turn_lm_mask.any() and turn_stream_mask.any():
                num_learn_v_tokens = turn_stream_mask.sum()
                num_learn_valid_tokens = turn_lm_masked_label.numel() + num_learn_v_tokens
                if frame_diff == 0:
                    fluency = (num_learn_v_tokens + num_lm_correct_tokens) / num_learn_valid_tokens
                elif frame_diff > 0:
                    fluency = (num_learn_v_tokens - frame_diff) / num_learn_valid_tokens
                else:
                    fluency = (num_learn_v_tokens - 1) / num_learn_valid_tokens
                fluencies.append(fluency)

            past_num_frames += turn_num_frames

        lm_ppl = torch.stack(lm_ppls).mean() if lm_ppls else one
        frame_diff = torch.stack(frame_diffs).float().mean() if frame_diffs else zero
        fluency = torch.stack(fluencies).float().mean() if fluencies else one
        lm_correctness = torch.stack(lm_correctness).float().mean() if lm_correctness else one
        return torch.stack([lm_ppl, frame_diff, fluency, lm_correctness])

    def trim_past_key_values(self, past_key_values, start, stop):
        if isinstance(past_key_values, Cache):
            if start != 0:
                raise NotImplementedError("Cache trimming currently supports start=0 only.")
            trimmed = copy.deepcopy(past_key_values)
            trimmed.crop(stop)
            return trimmed
        return [
            [past_keys[:, :, start:stop], past_values[:, :, start:stop]]
            for past_keys, past_values in past_key_values
        ]


def fast_greedy_generate(
    *,
    model: LiveMixin,
    inputs_embeds: torch.Tensor,
    past_key_values: Cache,
    eos_token_id: int,
    inplace_output_ids: torch.Tensor,
):
    for i in range(inplace_output_ids.size(1)):
        outputs = model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
        inplace_output_ids[:, i] = new_token_id
        if new_token_id == eos_token_id:
            break
        inputs_embeds = model.get_input_embeddings()(new_token_id)
    return inplace_output_ids[:, : i + 1], past_key_values


def build_live(
    *,
    is_training: bool,
    config_class: type,
    model_class: type,
    llm_pretrained: str | None = None,
    finetune_modules: list[str] | None = None,
    lora_modules=None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    set_vision_inside: bool = False,
    resume_from_checkpoint: str = "",
    attn_implementation: str = "sdpa",
    torch_dtype="auto",
    local_files_only: bool = True,
    **kwargs,
):
    config = config_class.from_pretrained(
        llm_pretrained,
        local_files_only=local_files_only,
        **kwargs,
    )
    generation_config = GenerationConfig.from_model_config(config)
    generation_config._original_object_hash = hash(generation_config)
    model = model_class.from_pretrained(
        llm_pretrained,
        config=config,
        generation_config=generation_config,
        device_map="cpu",
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    tokenizer = build_live_tokenizer_and_update_config(
        llm_pretrained,
        model.config,
        local_files_only=local_files_only,
    )
    if is_training:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            modules_to_save=finetune_modules,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        if resume_from_checkpoint:
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=False)
        else:
            logger.warning(
                "No checkpoint was provided. Returning a newly initialized streaming model wrapper."
            )
        if set_vision_inside:
            model.set_vision_inside()
        model.requires_grad_(False)
    return model, tokenizer


StreamingMixin = LiveMixin
build_streaming_model = build_live
