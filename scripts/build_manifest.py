import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a JSONL manifest from videos and annotations.")
    parser.add_argument("--videos-dir", default="data/raw/videos")
    parser.add_argument("--annotations-dir", default="data/annotations/responses")
    parser.add_argument("--output", default="data/manifests/all_valid.jsonl")
    parser.add_argument("--invalid-output", default="data/manifests/invalid_annotations.txt")
    parser.add_argument("--missing-output", default="data/manifests/videos_missing_annotations.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    annotations_dir = Path(args.annotations_dir)
    output_path = Path(args.output)
    invalid_path = Path(args.invalid_output)
    missing_path = Path(args.missing_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotation_paths = {path.stem: path for path in annotations_dir.glob("*.txt")}
    video_paths = {path.stem: path for path in videos_dir.glob("*.mp4")}

    valid_count = 0
    event_count = 0
    invalid = []

    with output_path.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(set(video_paths) & set(annotation_paths)):
            annotation_path = annotation_paths[sample_id]
            try:
                events = json.loads(annotation_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                invalid.append(f"{annotation_path.name}:{exc.lineno}:{exc.colno} {exc.msg}")
                continue

            record = {
                "sample_id": sample_id,
                "video_path": str(video_paths[sample_id].as_posix()),
                "annotation_path": str(annotation_path.as_posix()),
                "events": events,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            valid_count += 1
            event_count += len(events)

    missing_annotations = sorted(set(video_paths) - set(annotation_paths))
    invalid_path.write_text("\n".join(invalid) + ("\n" if invalid else ""), encoding="utf-8")
    missing_path.write_text(
        "\n".join(missing_annotations) + ("\n" if missing_annotations else ""),
        encoding="utf-8",
    )

    print(f"wrote={output_path}")
    print(f"valid_samples={valid_count}")
    print(f"events={event_count}")
    print(f"invalid_annotations={len(invalid)}")
    print(f"videos_missing_annotations={len(missing_annotations)}")


if __name__ == "__main__":
    main()
