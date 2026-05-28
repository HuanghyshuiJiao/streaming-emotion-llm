import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.data.data_collator import get_data_collator
from streaming_emotion_llm.data.stream import StreamingEmotionDataset
from streaming_emotion_llm.inference.generation import load_streaming_model, normalize_emotion
from streaming_emotion_llm.prompts.templates import EMOTION_TOKEN_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-forcing eval for full-video streams.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_emotion_token_positions(
    *,
    tokenizer,
    text: str,
    emotion: str,
    search_start: int,
) -> tuple[list[int], int]:
    start = text.lower().find(emotion.lower(), search_start)
    if start < 0:
        return [], search_start
    stop = start + len(emotion)
    tokenized = tokenizer(
        [text],
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    positions = []
    for index, (token_start, token_stop) in enumerate(tokenized.offset_mapping[0].tolist()):
        if token_stop <= start or token_start >= stop:
            continue
        if token_stop > token_start:
            positions.append(index)
    return positions, stop


def main() -> None:
    args = parse_args()
    config = load_config(args.config).values
    model, tokenizer, device = load_streaming_model(config, args.checkpoint)
    model.eval()

    data_config = config["data"]
    streaming_window = data_config.get("streaming_window", {})
    manifest = read_manifest(data_config[f"{args.split}_manifest"])
    dataset = StreamingEmotionDataset(
        data_config[f"{args.split}_manifest"],
        tokenizer,
        is_training=False,
        system_prompt=EMOTION_TOKEN_PROMPT,
        max_num_frames=int(streaming_window.get("max_num_frames", 1200)),
        fps=float(streaming_window.get("fps", 2.0)),
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
        else Path(config["experiment"]["output_dir"]) / f"{args.split}_teacher_forcing_fullvideo.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loss_sum = 0.0
    stream_correct = 0
    stream_total = 0
    lm_correct = 0
    lm_total = 0
    emotion_exact = 0
    emotion_total = 0
    rows = []

    with torch.no_grad(), output_path.open("w", encoding="utf-8") as handle:
        for index in tqdm(range(total), desc=f"teacher-forcing {args.split}"):
            text, frames, learn_ranges, _, meta = dataset[index]
            batch = collator([(text, frames, learn_ranges, index, meta)])
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
                if key != "evaluation_kwargs"
            }
            outputs = model(**batch, return_dict=True, use_cache=False)
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            logits = outputs.logits

            valid_mask = labels != -100
            if valid_mask.any():
                loss = torch.nn.functional.cross_entropy(logits[valid_mask], labels[valid_mask])
                loss_sum += float(loss.item())

            stream_mask = valid_mask & (input_ids == model.config.v_placeholder_id)
            pred_ids = logits.argmax(dim=-1)
            if stream_mask.any():
                item_stream_correct = (
                    pred_ids[stream_mask] == model.config.frame_token_interval_id
                ).sum().item()
                item_stream_total = int(stream_mask.sum().item())
            else:
                item_stream_correct = 0
                item_stream_total = 0
            stream_correct += item_stream_correct
            stream_total += item_stream_total

            lm_mask = valid_mask & ~stream_mask
            item_lm_correct = (pred_ids[lm_mask] == labels[lm_mask]).sum().item()
            item_lm_total = int(lm_mask.sum().item())
            lm_correct += item_lm_correct
            lm_total += item_lm_total

            record = manifest[index]
            search_start = 0
            event_rows = []
            for event in record.get("events", []):
                gold = normalize_emotion(str(event.get("emotion", "")))
                if not gold:
                    continue
                positions, search_start = find_emotion_token_positions(
                    tokenizer=tokenizer,
                    text=text,
                    emotion=str(event.get("emotion", "")).strip(),
                    search_start=search_start,
                )
                if not positions:
                    continue
                gold_token_ids = [int(input_ids[0, pos].item()) for pos in positions if pos > 0]
                pred_token_ids = [int(pred_ids[0, pos - 1].item()) for pos in positions if pos > 0]
                prediction = normalize_emotion(tokenizer.decode(pred_token_ids, skip_special_tokens=True))
                exact = prediction == gold
                emotion_exact += int(exact)
                emotion_total += 1
                event_rows.append(
                    {
                        "timestamp": float(event.get("timestamp", 0.0)),
                        "gold": gold,
                        "teacher_forced_prediction": prediction,
                        "exact": exact,
                    }
                )

            row = {
                "index": index,
                "sample_id": record.get("sample_id", ""),
                "loss": float(loss.item()) if valid_mask.any() else 0.0,
                "stream_interval_accuracy": item_stream_correct / item_stream_total
                if item_stream_total
                else 0.0,
                "lm_token_accuracy": item_lm_correct / item_lm_total if item_lm_total else 0.0,
                "events": event_rows,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"split={args.split} videos={total}")
    print(f"teacher_forcing_loss={loss_sum / total if total else 0.0:.4f}")
    print(
        f"stream_interval_accuracy={stream_correct}/{stream_total}="
        f"{stream_correct / stream_total if stream_total else 0.0:.4f}"
    )
    print(f"lm_token_accuracy={lm_correct}/{lm_total}={lm_correct / lm_total if lm_total else 0.0:.4f}")
    print(
        f"emotion_exact={emotion_exact}/{emotion_total}="
        f"{emotion_exact / emotion_total if emotion_total else 0.0:.4f}"
    )
    for row in rows[:5]:
        label = " | ".join(f"L@{event['timestamp']:.1f}={event['gold']}" for event in row["events"])
        pred = " | ".join(
            f"P@{event['timestamp']:.1f}={event['teacher_forced_prediction']}"
            for event in row["events"]
        )
        print(f"{row['sample_id']}")
        print(f"label: {label}")
        print(f"pred : {pred}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
