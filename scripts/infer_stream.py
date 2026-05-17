import argparse

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.inference.streaming_runner import StreamingInferenceRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming emotion inference.")
    parser.add_argument("--config", required=True, help="Path to an inference YAML config.")
    parser.add_argument("--input", required=True, help="Input video or stream path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runner = StreamingInferenceRunner(config.values)
    runner.run_video(args.input)


if __name__ == "__main__":
    main()
