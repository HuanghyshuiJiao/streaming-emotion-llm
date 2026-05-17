"""Dataset utilities aligned with the original ``data/stream.py`` pattern."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from streaming_emotion_llm.data.manifest import iter_jsonl
from streaming_emotion_llm.prompts.templates import EMOTION_TOKEN_PROMPT


def _load_feature_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class StreamMixIn(Dataset):
    """Build original-style stream samples from pre-extracted frame features."""

    def __init__(
        self,
        *,
        is_training: bool,
        system_prompt: str,
        max_num_frames: int,
        fps: float,
        context_mode: str,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ):
        super().__init__()
        if system_prompt is None:
            raise ValueError("Please provide a system prompt.")
        self.is_training = is_training
        self.system_prompt = system_prompt
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.context_mode = context_mode
        self.tokenizer = tokenizer

    def load_frames(
        self,
        feature_path: str | Path,
        *,
        timestamp: float | None = None,
        start_frame: int | None = None,
        stop_frame: int | None = None,
    ) -> torch.Tensor:
        feature_path = Path(feature_path)
        frames = _load_feature_tensor(feature_path)

        if start_frame is not None or stop_frame is not None:
            start = 0 if start_frame is None else max(int(start_frame), 0)
            stop = frames.shape[0] if stop_frame is None else min(int(stop_frame), frames.shape[0])
            return frames[start:stop]

        if self.context_mode == "prefix_until_event":
            if timestamp is None:
                raise ValueError("prefix_until_event requires an event timestamp.")
            stop = min(max(int(float(timestamp) * self.fps) + 1, 1), frames.shape[0])
            start = max(stop - self.max_num_frames, 0) if self.max_num_frames else 0
            return frames[start:stop]

        if self.context_mode != "full_video":
            raise ValueError(f"Unsupported context mode: {self.context_mode}")

        if self.max_num_frames and frames.shape[0] > self.max_num_frames:
            idx = torch.linspace(0, frames.shape[0] - 1, self.max_num_frames).long()
            frames = frames.index_select(0, idx)
        return frames

    def build_text_and_ranges(
        self,
        *,
        frames: torch.Tensor,
        emotion: str,
        add_generation_prompt: bool = False,
    ):
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "stream", "num_frames": int(frames.shape[0]), "learn": False},
        ]
        if not add_generation_prompt:
            conversation.append({"role": "assistant", "content": emotion, "learn": True})

        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        learn_ranges = [] if add_generation_prompt else self.tokenizer.get_learn_ranges(conversation)
        return text, learn_ranges

    def build_text_and_ranges_from_conversation(
        self,
        conversation: list[dict],
        *,
        add_generation_prompt: bool = False,
    ):
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        learn_ranges = [] if add_generation_prompt else self.tokenizer.get_learn_ranges(conversation)
        return text, learn_ranges


class StreamingEmotionDataset(StreamMixIn):
    """Event-level emotion-token dataset using precomputed SigLIP features."""

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        *,
        is_training: bool = True,
        system_prompt: str = EMOTION_TOKEN_PROMPT,
        max_num_frames: int = 120,
        fps: float = 2.0,
        context_mode: str = "prefix_until_event",
        add_generation_prompt: bool = False,
    ):
        super().__init__(
            is_training=is_training,
            system_prompt=system_prompt,
            max_num_frames=max_num_frames,
            fps=fps,
            context_mode=context_mode,
            tokenizer=tokenizer,
        )
        self.manifest_path = Path(manifest_path)
        self.add_generation_prompt = add_generation_prompt
        self.samples = self._expand_events(self.manifest_path)

    @staticmethod
    def _expand_events(manifest_path: Path) -> list[dict]:
        samples = []
        for record in iter_jsonl(manifest_path):
            events = [
                event
                for event in record.get("events", [])
                if str(event.get("emotion", "")).strip()
            ]
            for event_index, event in enumerate(events):
                emotion = str(event.get("emotion", "")).strip()
                samples.append(
                    {
                        "sample_id": record["sample_id"],
                        "event_index": event_index,
                        "timestamp": event.get("timestamp"),
                        "emotion": emotion,
                        "events": events,
                        "feature_path": record["feature_path"],
                    }
                )
        return samples

    def _frame_stop_for_timestamp(self, timestamp: float, num_frames: int) -> int:
        return min(max(int(float(timestamp) * self.fps) + 1, 1), num_frames)

    def _build_event_stream_window(self, sample: dict):
        all_frames = _load_feature_tensor(Path(sample["feature_path"]))
        target_stop = self._frame_stop_for_timestamp(sample["timestamp"], all_frames.shape[0])
        start = max(target_stop - self.max_num_frames, 0) if self.max_num_frames else 0
        frames = all_frames[start:target_stop]

        conversation = []
        cursor = start
        included_events = 0
        target_event = sample["events"][sample["event_index"]]
        events = sample["events"][: sample["event_index"] + 1]
        for event in events:
            emotion = str(event.get("emotion", "")).strip()
            if not emotion:
                continue
            event_stop = self._frame_stop_for_timestamp(event["timestamp"], all_frames.shape[0])
            if event_stop <= start:
                continue
            event_stop = min(event_stop, target_stop)
            num_stream_frames = event_stop - cursor
            if num_stream_frames <= 0:
                if event is target_event:
                    if conversation and conversation[-1]["role"] == "assistant":
                        conversation.pop()
                    if not self.add_generation_prompt:
                        conversation.append({"role": "assistant", "content": emotion, "learn": True})
                    included_events += 1
                continue
            if num_stream_frames > 0:
                conversation.append(
                    {
                        "role": "stream",
                        "num_frames": num_stream_frames,
                        "learn": included_events > 0,
                    }
                )
                cursor = event_stop
            if event is not target_event or not self.add_generation_prompt:
                conversation.append({"role": "assistant", "content": emotion, "learn": True})
            included_events += 1

        text, learn_ranges = self.build_text_and_ranges_from_conversation(
            conversation,
            add_generation_prompt=self.add_generation_prompt,
        )
        return text, frames, learn_ranges, start, target_stop, included_events

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.context_mode == "event_stream_window":
            text, frames, learn_ranges, frame_start, frame_stop, included_events = (
                self._build_event_stream_window(sample)
            )
        else:
            frames = self.load_frames(sample["feature_path"], timestamp=sample["timestamp"])
            frame_stop = None
            frame_start = None
            included_events = 1
            text, learn_ranges = self.build_text_and_ranges(
                frames=frames,
                emotion=sample["emotion"],
                add_generation_prompt=self.add_generation_prompt,
            )
        evaluation_kwargs = {
            "sample_id": sample["sample_id"],
            "event_index": sample["event_index"],
            "timestamp": sample["timestamp"],
            "emotion": sample["emotion"],
            "frame_start": frame_start,
            "frame_stop": frame_stop,
            "included_events": included_events,
        }
        return text, frames, learn_ranges, index, evaluation_kwargs
