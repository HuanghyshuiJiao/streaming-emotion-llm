"""Online inference runner for video and future audio-video streams."""


class StreamingInferenceRunner:
    def __init__(self, config: dict):
        self.config = config

    def run_video(self, input_path: str) -> list[dict]:
        raise NotImplementedError("Implement streaming video inference here.")
