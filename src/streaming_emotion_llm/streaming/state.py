from dataclasses import dataclass, field


@dataclass
class StreamState:
    video_windows: list = field(default_factory=list)
    audio_windows: list = field(default_factory=list)
    generated_events: list = field(default_factory=list)

    def reset(self) -> None:
        self.video_windows.clear()
        self.audio_windows.clear()
        self.generated_events.clear()
