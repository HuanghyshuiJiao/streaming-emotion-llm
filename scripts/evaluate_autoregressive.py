import argparse
import json
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.inference.generation import (
    build_generation_dataset,
    decode_generated_emotion,
    greedy_generate_ids,
    load_streaming_model,
    normalize_emotion,
    token_overlap_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate emotion labels with true autoregressive generation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config).values
    model, tokenizer, device = load_streaming_model(config, args.checkpoint)
    dataset = build_generation_dataset(config, tokenizer, split=args.split)

    output_path = (
        Path(args.output)
        if args.output
        else Path(config["experiment"]["output_dir"])
        / f"{args.split}_autoregressive_predictions.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = min(args.limit, len(dataset)) if args.limit > 0 else len(dataset)
    rows = []
    exact_correct = 0
    token_correct = 0
    token_total = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index in tqdm(range(total), desc=f"autoregressive {args.split}"):
            text, frames, _, _, meta = dataset[index]
            generated_ids = greedy_generate_ids(
                model=model,
                tokenizer=tokenizer,
                text=text,
                frames=frames,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            prediction = decode_generated_emotion(tokenizer, generated_ids)
            normalized_prediction = normalize_emotion(prediction)
            normalized_gold = normalize_emotion(meta["emotion"])
            item_exact = normalized_prediction == normalized_gold
            item_token_correct, item_token_total = token_overlap_metrics(
                tokenizer,
                normalized_prediction,
                normalized_gold,
            )
            exact_correct += int(item_exact)
            token_correct += item_token_correct
            token_total += item_token_total
            row = {
                **meta,
                "prompt": text,
                "generated_text": prediction,
                "normalized_prediction": normalized_prediction,
                "normalized_gold": normalized_gold,
                "emotion_exact_match": item_exact,
                "emotion_token_correct": item_token_correct,
                "emotion_token_total": item_token_total,
                "emotion_token_accuracy": (
                    item_token_correct / item_token_total if item_token_total else 0.0
                ),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    exact_accuracy = exact_correct / total if total else 0.0
    token_accuracy = token_correct / token_total if token_total else 0.0
    print(
        f"split={args.split} total={total} "
        f"autoregressive_exact_match={exact_accuracy:.4f} "
        f"autoregressive_token_accuracy={token_accuracy:.4f}"
    )
    if rows:
        print(
            "mean_item_token_accuracy="
            f"{mean(row['emotion_token_accuracy'] for row in rows):.4f}"
        )
    for row in rows[:10]:
        print(
            f"{row['sample_id']}#{row['event_index']} "
            f"gold={row['normalized_gold']} pred={row['normalized_prediction']} "
            f"raw={row['generated_text']!r}"
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
