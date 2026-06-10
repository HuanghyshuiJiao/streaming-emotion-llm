"""Dataset utilities aligned with the original ``data/stream.py`` pattern."""

import math
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


def _num_frames(frames: torch.Tensor | dict[str, torch.Tensor]) -> int:
    if isinstance(frames, dict):
        return int(frames["vision"].shape[0])
    return int(frames.shape[0])


def _slice_frames(
    frames: torch.Tensor | dict[str, torch.Tensor],
    start: int | None = None,
    stop: int | None = None,
):
    if isinstance(frames, dict):
        return {key: value[start:stop] for key, value in frames.items()}
    return frames[start:stop]


def _index_select_frames(
    frames: torch.Tensor | dict[str, torch.Tensor],
    dim: int,
    index: torch.Tensor,
):
    if isinstance(frames, dict):
        return {key: value.index_select(dim, index) for key, value in frames.items()}
    return frames.index_select(dim, index)


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
        feature_path: str | Path | dict,
        *,
        timestamp: float | None = None,
        start_frame: int | None = None,
        stop_frame: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        frames = self.load_feature_record(feature_path)

        if start_frame is not None or stop_frame is not None:
            start = 0 if start_frame is None else max(int(start_frame), 0)
            stop = _num_frames(frames) if stop_frame is None else min(int(stop_frame), _num_frames(frames))
            return _slice_frames(frames, start, stop)

        if self.context_mode == "prefix_until_event":
            if timestamp is None:
                raise ValueError("prefix_until_event requires an event timestamp.")
            stop = min(max(int(float(timestamp) * self.fps) + 1, 1), _num_frames(frames))
            start = max(stop - self.max_num_frames, 0) if self.max_num_frames else 0
            return _slice_frames(frames, start, stop)

        if self.context_mode != "full_video":
            raise ValueError(f"Unsupported context mode: {self.context_mode}")

        if self.max_num_frames and _num_frames(frames) > self.max_num_frames:
            idx = torch.linspace(0, _num_frames(frames) - 1, self.max_num_frames).long()
            frames = _index_select_frames(frames, 0, idx)
        return frames

    def load_feature_record(self, record_or_path: str | Path | dict):
        if not isinstance(record_or_path, dict):
            return _load_feature_tensor(Path(record_or_path))

        vision_frames = _load_feature_tensor(Path(record_or_path["feature_path"]))
        face_feature_path = record_or_path.get("face_feature_path")
        if not face_feature_path:
            return vision_frames

        face_frames = _load_feature_tensor(Path(face_feature_path))
        if face_frames.ndim == 2:
            face_frames = face_frames[:, None]
        if face_frames.shape[0] != vision_frames.shape[0]:
            stop = min(face_frames.shape[0], vision_frames.shape[0])
            vision_frames = vision_frames[:stop]
            face_frames = face_frames[:stop]
        return {"vision": vision_frames, "face": face_frames}

    def build_text_and_ranges(
        self,
        *,
        frames: torch.Tensor,
        emotion: str,
        add_generation_prompt: bool = False,
    ):
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "stream", "num_frames": _num_frames(frames), "learn": False},
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
        timestamp_alignment: str = "ceil",
        first_stream_learn: str = "skip_first",
        trailing_stream: str = "drop",
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
        self.timestamp_alignment = timestamp_alignment
        self.first_stream_learn = first_stream_learn
        self.trailing_stream = trailing_stream
        if self.context_mode == "full_video_stream":
            self.samples = list(iter_jsonl(self.manifest_path))
        else:
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
        position = float(timestamp) * self.fps
        if self.timestamp_alignment == "ceil":
            stop = int(math.ceil(position)) + 1
        elif self.timestamp_alignment == "floor":
            stop = int(math.floor(position))
        elif self.timestamp_alignment == "floor_plus_one":
            stop = int(position) + 1
        else:
            raise ValueError(f"Unsupported timestamp alignment: {self.timestamp_alignment}")
        return min(max(stop, 1), num_frames)

    def _stream_learn_value(self, num_frames: int, *, is_first_stream: bool):
        if num_frames <= 0:
            return False
        if not is_first_stream:
            return True
        if self.first_stream_learn == "all":
            return True
        if self.first_stream_learn == "none":
            return False
        if self.first_stream_learn == "skip_first":
            return False if num_frames == 1 else num_frames - 1
        raise ValueError(f"Unsupported first stream learn policy: {self.first_stream_learn}")

    def _build_event_stream_window(self, sample: dict):
        all_frames = self.load_feature_record(sample)
        target_stop = self._frame_stop_for_timestamp(sample["timestamp"], _num_frames(all_frames))
        start = max(target_stop - self.max_num_frames, 0) if self.max_num_frames else 0
        frames = _slice_frames(all_frames, start, target_stop)

        conversation = []
        cursor = start
        included_events = 0
        target_event = sample["events"][sample["event_index"]]
        events = sample["events"][: sample["event_index"] + 1]
        for event in events:
            emotion = str(event.get("emotion", "")).strip()
            if not emotion:
                continue
            event_stop = self._frame_stop_for_timestamp(event["timestamp"], _num_frames(all_frames))
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
                        "learn": True,
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

    def _build_full_video_stream(self, record: dict):
        all_frames = self.load_feature_record(record)
        if self.max_num_frames and _num_frames(all_frames) > self.max_num_frames:
            frames = _slice_frames(all_frames, None, self.max_num_frames)
        else:
            frames = all_frames

        raw_events = [
            event
            for event in record.get("events", [])
            if str(event.get("emotion", "")).strip()
        ]
        raw_events = sorted(raw_events, key=lambda event: float(event.get("timestamp", 0.0)))
        events_by_stop: dict[int, str] = {}
        for event in raw_events:
            event_stop = self._frame_stop_for_timestamp(event["timestamp"], _num_frames(frames))
            events_by_stop[event_stop] = str(event.get("emotion", "")).strip()
        events = sorted(events_by_stop.items())

        conversation = []
        cursor = 0
        included_events = 0
        for event_stop, emotion in events:
            if event_stop > cursor:
                conversation.append(
                    {
                        "role": "stream",
                        "num_frames": event_stop - cursor,
                        "learn": self._stream_learn_value(
                            event_stop - cursor,
                            is_first_stream=len(conversation) == 0,
                        ),
                    }
                )
                cursor = event_stop
            if not self.add_generation_prompt:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": emotion,
                        "learn": True,
                    }
                )
            included_events += 1

        if _num_frames(frames) > cursor:
            if self.trailing_stream == "drop":
                frames = _slice_frames(frames, None, cursor)
            elif self.trailing_stream in {"learn", "unlearn"}:
                conversation.append(
                    {
                        "role": "stream",
                        "num_frames": _num_frames(frames) - cursor,
                        "learn": self.trailing_stream == "learn",
                    }
                )
            else:
                raise ValueError(f"Unsupported trailing stream policy: {self.trailing_stream}")

        text, learn_ranges = self.build_text_and_ranges_from_conversation(
            conversation,
            add_generation_prompt=self.add_generation_prompt,
        )
        return text, frames, learn_ranges, included_events

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.context_mode == "full_video_stream":
            text, frames, learn_ranges, included_events = self._build_full_video_stream(sample)
            evaluation_kwargs = {
                "sample_id": sample["sample_id"],
                "event_index": None,
                "timestamp": None,
                "emotion": None,
                "frame_start": 0,
                "frame_stop": _num_frames(frames),
                "included_events": included_events,
            }
            return text, frames, learn_ranges, index, evaluation_kwargs
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
