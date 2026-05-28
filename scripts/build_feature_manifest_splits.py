import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add feature_path fields to existing train/val/test manifests."
    )
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument(
        "--feature-dir",
        default="data/processed/features/siglip_large_384_2fps_1plus3x3",
    )
    parser.add_argument("--output-dir", default="data/manifests/full_features")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def add_feature_paths(records: list[dict], feature_dir: Path) -> tuple[list[dict], list[dict]]:
    available = []
    missing = []
    for record in records:
        feature_path = feature_dir / f"{record['sample_id']}.pt"
        if not feature_path.exists():
            missing.append(record)
            continue
        item = dict(record)
        item["feature_path"] = feature_path.as_posix()
        item["feature_dir"] = feature_dir.as_posix()
        item["feature_mode"] = "full_video"
        available.append(item)
    return available, missing


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)

    all_available = []
    all_missing = []
    for split in args.splits:
        records = read_jsonl(manifest_dir / f"{split}.jsonl")
        available, missing = add_feature_paths(records, feature_dir)
        write_jsonl(output_dir / f"{split}.jsonl", available)
        all_available.extend(available)
        all_missing.extend({"split": split, **record} for record in missing)
        print(f"{split}={len(available)} missing_features={len(missing)}")

    write_jsonl(output_dir / "all.jsonl", all_available)
    if all_missing:
        write_jsonl(output_dir / "missing_features.jsonl", all_missing)
    print(f"available_with_features={len(all_available)}")
    print(f"missing_features={len(all_missing)}")
    print(f"feature_dir={feature_dir}")


if __name__ == "__main__":
    main()
