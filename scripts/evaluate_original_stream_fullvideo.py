import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.data.data_collator import get_data_collator
from streaming_emotion_llm.data.stream import StreamingEmotionDataset
from streaming_emotion_llm.inference.generation import load_streaming_model, get_base_model
from streaming_emotion_llm.prompts.templates import EMOTION_TOKEN_PROMPT


def move_batch_to_device(batch: dict, device: str) -> dict:
    moved = {}
    for key, value in batch.items():
        if key == "evaluation_kwargs":
            continue
        if isinstance(value, dict):
            moved[key] = {
                sub_key: sub_value.to(device) if hasattr(sub_value, "to") else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original-style stream_evaluate for full-video streams.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config).values
    model, tokenizer, device = load_streaming_model(config, args.checkpoint)
    base_model = get_base_model(model)
    data_config = config["data"]
    streaming_window = data_config.get("streaming_window", {})
    fps = float(streaming_window.get("fps", 2.0))
    dataset = StreamingEmotionDataset(
        data_config[f"{args.split}_manifest"],
        tokenizer,
        is_training=False,
        system_prompt=EMOTION_TOKEN_PROMPT,
        max_num_frames=int(streaming_window.get("max_num_frames", 1200)),
        fps=fps,
        context_mode=streaming_window.get("mode", "full_video_stream"),
        add_generation_prompt=False,
        timestamp_alignment=streaming_window.get("timestamp_alignment", "ceil"),
        first_stream_learn=streaming_window.get("first_stream_learn", "skip_first"),
        trailing_stream=streaming_window.get("trailing_stream", "drop"),
    )
    collator = get_data_collator(tokenizer=tokenizer)
    total = min(args.limit, len(dataset)) if args.limit > 0 else len(dataset)
    output_path = (
        Path(args.output)
        if args.output
        else Path(config["experiment"]["output_dir"])
        / f"{args.split}_original_stream_metrics_threshold_{str(args.threshold).replace('.', '')}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sums = {"lm_ppl": 0.0, "frame_diff": 0.0, "time_diff": 0.0, "fluency": 0.0, "lm_correctness": 0.0}
    with torch.no_grad(), output_path.open("w", encoding="utf-8") as handle:
        for index in tqdm(range(total), desc=f"original-stream {args.split}"):
            text, frames, learn_ranges, _, meta = dataset[index]
            batch = collator([(text, frames, learn_ranges, index, meta)])
            batch = move_batch_to_device(batch, device)
            metrics = base_model.stream_evaluate(
                batch["input_ids"],
                batch["labels"],
                batch["frames"],
                frame_token_interval_threshold=args.threshold,
            )
            row = {
                "index": index,
                "sample_id": meta["sample_id"],
                "included_events": meta["included_events"],
                "lm_ppl": float(metrics[0].item()),
                "frame_diff": float(metrics[1].item()),
                "time_diff": float(metrics[1].item() / fps),
                "fluency": float(metrics[2].item()),
                "lm_correctness": float(metrics[3].item()),
            }
            for key in sums:
                sums[key] += row[key]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    means = {key: sums[key] / total if total else 0.0 for key in sums}
    print(f"split={args.split} videos={total} threshold={args.threshold}")
    print(
        f"lm_ppl={means['lm_ppl']:.4f} "
        f"time_diff={means['time_diff']:.4f}s "
        f"frame_diff={means['frame_diff']:.4f} "
        f"fluency={means['fluency']:.4f} "
        f"lm_correctness={means['lm_correctness']:.4f}"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
