import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.inference.generation import (
    load_streaming_model,
    normalize_emotion,
    stream_autoregressive_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-video streaming autoregressive output.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.725)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_feature_tensor(path: str | Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_record_frames(record: dict) -> torch.Tensor | dict[str, torch.Tensor]:
    vision_frames = load_feature_tensor(record["feature_path"])
    face_feature_path = record.get("face_feature_path")
    if not face_feature_path:
        return vision_frames
    face_frames = load_feature_tensor(face_feature_path)
    if face_frames.ndim == 2:
        face_frames = face_frames[:, None]
    if face_frames.shape[0] != vision_frames.shape[0]:
        stop = min(face_frames.shape[0], vision_frames.shape[0])
        vision_frames = vision_frames[:stop]
        face_frames = face_frames[:stop]
    return {"vision": vision_frames, "face": face_frames}


def num_frames(frames: torch.Tensor | dict[str, torch.Tensor]) -> int:
    if isinstance(frames, dict):
        return int(frames["vision"].shape[0])
    return int(frames.shape[0])


def gold_events(record: dict) -> list[dict]:
    return [
        {
            "event_index": event_index,
            "timestamp": float(event.get("timestamp", 0.0)),
            "emotion": normalize_emotion(str(event.get("emotion", ""))),
        }
        for event_index, event in enumerate(record.get("events", []))
        if str(event.get("emotion", "")).strip()
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config).values
    model, tokenizer, device = load_streaming_model(config, args.checkpoint)

    manifest = read_manifest(config["data"][f"{args.split}_manifest"])
    total = min(args.limit, len(manifest)) if args.limit > 0 else len(manifest)
    fps = float(config["data"].get("streaming_window", {}).get("fps", 2.0))
    output_path = (
        Path(args.output)
        if args.output
        else Path(config["experiment"]["output_dir"])
        / f"{args.split}_video_stream_predictions_threshold_{str(args.threshold).replace('.', '')}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_gold = 0
    total_pred = 0
    zero_pred = 0
    with torch.no_grad(), output_path.open("w", encoding="utf-8") as handle:
        for index in tqdm(range(total), desc=f"video-stream {args.split}"):
            record = manifest[index]
            frames = load_record_frames(record)
            predictions = stream_autoregressive_features(
                model=model,
                tokenizer=tokenizer,
                frames=frames,
                device=device,
                fps=fps,
                max_new_tokens=args.max_new_tokens,
                frame_token_interval_threshold=args.threshold,
                max_frames=args.max_frames if args.max_frames > 0 else None,
            )
            gold = gold_events(record)
            row = {
                "index": index,
                "sample_id": record.get("sample_id", ""),
                "frame_count": num_frames(frames),
                "gold_events": gold,
                "predictions": predictions,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            total_gold += len(gold)
            total_pred += len(predictions)
            zero_pred += int(len(predictions) == 0)
            print(
                f"{index:03d} {row['sample_id']} frames={row['frame_count']} "
                f"gold={len(gold)} pred={len(predictions)} "
                f"times={[round(item['timestamp'], 2) for item in predictions[:12]]}",
                flush=True,
            )

    print(
        f"split={args.split} videos={total} total_gold_events={total_gold} "
        f"total_stream_predictions={total_pred} "
        f"avg_pred_per_video={total_pred / total if total else 0.0:.2f} "
        f"zero_prediction_videos={zero_pred}"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
