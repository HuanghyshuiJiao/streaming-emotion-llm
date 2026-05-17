import argparse

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.training.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a streaming emotion LLM experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config.values)


if __name__ == "__main__":
    main()
