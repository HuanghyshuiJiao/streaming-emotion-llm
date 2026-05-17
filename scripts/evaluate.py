import argparse

from streaming_emotion_llm.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate streaming emotion models.")
    parser.add_argument("--config", required=True, help="Path to an evaluation YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raise NotImplementedError(f"Evaluation runner is not implemented yet: {config.path}")


if __name__ == "__main__":
    main()
