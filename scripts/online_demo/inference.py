"""Adapted original VideoLLM-online LiveInfer for emotion streaming."""

from __future__ import annotations

import collections
from pathlib import Path

import torch
import transformers

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.inference.generation import get_base_model, normalize_emotion
from streaming_emotion_llm.models.live_llama import build_live_llama
from streaming_emotion_llm.models.modeling_live import fast_greedy_generate
from streaming_emotion_llm.prompts.templates import EMOTION_TOKEN_PROMPT


logger = transformers.logging.get_logger("original-online")


class LiveInfer:
    def __init__(
        self,
        *,
        config_path: str,
        checkpoint: str,
        frame_token_interval_threshold: float = 0.725,
        max_new_tokens: int = 8,
    ) -> None:
        config = load_config(config_path).values
        model_config = config["model"]
        llm_config = model_config["llm"]
        vision_config = model_config["vision_encoder"]
        projector_config = model_config.get("projector", {})

        self.model, self.tokenizer = build_live_llama(
            is_training=False,
            llm_pretrained=llm_config["name_or_path"],
            resume_from_checkpoint=checkpoint,
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
            set_vision_inside=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        self.base_model = get_base_model(self.model)
        if not hasattr(self.base_model, "vision_encode"):
            self.base_model.set_vision_inside()
            self.base_model.vision_encoder.to(self.device)

        self.hidden_size = self.base_model.config.hidden_size
        self.frame_fps = float(config["data"].get("streaming_window", {}).get("fps", 2.0))
        self.frame_interval = 1 / self.frame_fps
        self.frame_resolution = self.base_model.config.frame_resolution
        self.frame_num_tokens = self.base_model.config.frame_num_tokens
        self.frame_v_placeholder = self.base_model.config.v_placeholder * self.frame_num_tokens
        self.frame_token_interval_id = self.base_model.config.frame_token_interval_id
        self.frame_placeholder_ids = torch.tensor(
            self.base_model.config.v_placeholder_id,
            device=self.device,
        ).repeat(self.frame_num_tokens).reshape(1, -1)

        self.system_prompt = EMOTION_TOKEN_PROMPT
        self.inplace_output_ids = torch.zeros(
            1,
            max_new_tokens,
            device=self.device,
            dtype=torch.long,
        )
        self.frame_token_interval_threshold = frame_token_interval_threshold
        self.eos_token_id = self.base_model.config.eos_token_id
        self._start_ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_prompt}],
            add_stream_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template(
            [{}],
            add_stream_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        self._added_stream_generation_ids = self.tokenizer.apply_chat_template(
            [{}],
            add_stream_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        self.reset()

    def _call_for_response(self, video_time, query):
        if query is not None:
            self.last_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                add_stream_query_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            self.last_ids = self._added_stream_generation_ids
        inputs_embeds = self.base_model.get_input_embeddings()(self.last_ids)
        output_ids, self.past_key_values = fast_greedy_generate(
            model=self.base_model,
            inputs_embeds=inputs_embeds,
            past_key_values=self.past_key_values,
            eos_token_id=self.eos_token_id,
            inplace_output_ids=self.inplace_output_ids.zero_(),
        )
        self.last_ids = output_ids[:, -1:]
        prediction = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        if query:
            query = f"(Video Time = {video_time}s) User: {query}"
        response = f"(Video Time = {video_time}s) Assistant:{prediction}"
        return query, response

    def _call_for_streaming(self):
        while self.frame_embeds_queue:
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            video_time, frame_embeds = self.frame_embeds_queue.popleft()
            if not self.past_key_values:
                self.last_ids = self._start_ids
            elif self.last_ids.numel() == 1 and int(self.last_ids.item()) == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
            inputs_embeds = torch.cat(
                [
                    self.base_model.get_input_embeddings()(self.last_ids).view(
                        1,
                        -1,
                        self.hidden_size,
                    ),
                    frame_embeds.view(1, -1, self.hidden_size),
                ],
                dim=1,
            )
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=self.past_key_values,
            )
            self.past_key_values = outputs.past_key_values
            if self.query_queue and video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            next_score = outputs.logits[:, -1:].softmax(dim=-1)
            if next_score[:, :, self.frame_token_interval_id] < self.frame_token_interval_threshold:
                next_score[:, :, self.frame_token_interval_id].zero_()
            self.last_ids = next_score.argmax(dim=-1)
            if int(self.last_ids.item()) != int(self.frame_token_interval_id):
                return video_time, None
        return None, None

    def reset(self):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device=self.device, dtype=torch.long)
        self.past_key_values = None

    def input_query_stream(self, query, history=None, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if not self.past_key_values:
            return (
                f'(NOTE: No video stream here. Please load a video. Then the assistant '
                f'will answer "{query} (at {self.video_time}s)" in the video stream)'
            )
        return (
            f'(NOTE: Received "{query}" (at {self.video_time}s). '
            "Please wait until previous frames have been processed)"
        )

    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            frames_embeds = self.base_model.visual_embed(self.video_tensor[ranger]).split(
                self.frame_num_tokens
            )
            self.frame_embeds_queue.extend(
                [(r / self.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, frames_embeds)]
            )
        self.last_frame_idx = frame_idx
        self.video_time = video_time

    def load_video(self, video_path):
        try:
            self.video_tensor = torch.load(video_path, map_location="cpu", weights_only=True)
        except TypeError:
            self.video_tensor = torch.load(video_path, map_location="cpu")
        self.video_tensor = self.video_tensor.to(self.device, non_blocking=True)
        self.num_video_frames = self.video_tensor.size(0)
        self.video_duration = self.video_tensor.size(0) / self.frame_fps
        logger.warning(f"{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS")

    def __call__(self):
        while not self.frame_embeds_queue:
            continue
        video_time, query = self._call_for_streaming()
        response = None
        if video_time is not None:
            query, response = self._call_for_response(video_time, query)
        return query, response

    @staticmethod
    def normalize_response(response: str | None) -> str | None:
        if response is None:
            return None
        return normalize_emotion(response.split("Assistant:", 1)[-1])


__all__ = ["LiveInfer"]
