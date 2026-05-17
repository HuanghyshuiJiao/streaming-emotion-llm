import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect response annotation JSON files.")
    parser.add_argument(
        "--annotations-dir",
        default="data/annotations/responses",
        help="Directory containing per-clip response annotation .txt files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_dir = Path(args.annotations_dir)

    valid_files = 0
    invalid_files = []
    event_count = 0
    emotion_counts = Counter()
    field_counts = Counter()

    for path in sorted(annotations_dir.glob("*.txt")):
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            invalid_files.append((path.name, exc.lineno, exc.colno, exc.msg))
            continue

        valid_files += 1
        for event in events:
            event_count += 1
            emotion_counts[event.get("emotion", "")] += 1
            field_counts.update(event.keys())

    print(f"annotation_files={valid_files + len(invalid_files)}")
    print(f"valid_json_files={valid_files}")
    print(f"invalid_json_files={len(invalid_files)}")
    print(f"events={event_count}")
    print("\nfields:")
    for field, count in field_counts.most_common():
        print(f"  {field}: {count}")

    print("\ntop_emotions:")
    for emotion, count in emotion_counts.most_common(30):
        print(f"  {emotion}: {count}")

    if invalid_files:
        print("\ninvalid_files:")
        for name, line, col, msg in invalid_files[:100]:
            print(f"  {name}:{line}:{col} {msg}")


if __name__ == "__main__":
    main()
