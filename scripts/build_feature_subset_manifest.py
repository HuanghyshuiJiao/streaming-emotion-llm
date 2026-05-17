import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test manifests using only samples with precomputed features."
    )
    parser.add_argument("--input", default="data/manifests/all_valid.jsonl")
    parser.add_argument(
        "--feature-dir",
        default="data/processed/features/siglip_large_384_2fps_1plus3x3",
    )
    parser.add_argument("--output-dir", default="data/manifests/feature_subset")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    available = []
    for record in records:
        feature_path = feature_dir / f"{record['sample_id']}.pt"
        if feature_path.exists():
            item = dict(record)
            item["feature_path"] = feature_path.as_posix()
            item["feature_dir"] = feature_dir.as_posix()
            item["feature_mode"] = "full_video"
            available.append(item)

    random.Random(args.seed).shuffle(available)
    if args.limit is not None:
        available = available[: args.limit]

    train_end = int(len(available) * args.train_ratio)
    val_end = train_end + int(len(available) * args.val_ratio)
    splits = {
        "train": available[:train_end],
        "val": available[train_end:val_end],
        "test": available[val_end:],
    }

    write_jsonl(output_dir / "all.jsonl", available)
    for name, split_records in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", split_records)
        print(f"{name}={len(split_records)}")
    print(f"available_with_features={len(available)}")
    print(f"feature_dir={feature_dir}")


if __name__ == "__main__":
    main()
