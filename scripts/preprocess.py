import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess manifests, frames, audio, or features.")
    parser.add_argument("--config", required=True, help="Path to a preprocessing config.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise NotImplementedError("Preprocessing pipeline is not implemented yet.")


if __name__ == "__main__":
    main()
