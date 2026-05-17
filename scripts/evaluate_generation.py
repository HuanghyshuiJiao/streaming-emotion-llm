import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.data.stream import StreamingEmotionDataset
from streaming_emotion_llm.models.live_llama import build_live_llama
from streaming_emotion_llm.prompts.templates import EMOTION_TOKEN_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate streaming emotion-token predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--frame-token-interval-threshold",
        type=float,
        default=0.0,
        help="Optional probability threshold for treating frame interval predictions as continue.",
    )
    return parser.parse_args()


def normalize_emotion(text: str) -> str:
    text = text.strip().lower()
    text = text.splitlines()[0] if text else text
    for stop in [".", ",", ";", ":", "</s>"]:
        text = text.split(stop)[0]
    return text.strip().strip('"').strip("'")


def get_base_model(model):
    return getattr(getattr(model, "base_model", model), "model", model)


def build_labeled_batch(tokenizer, text: str, learn_ranges: list[range], device: str):
    tokenized = tokenizer(
        [text],
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    labels = torch.full_like(tokenized.input_ids, -100, dtype=torch.long)
    offset_mapping = tokenized.offset_mapping[0]
    input_ids = tokenized.input_ids[0]
    item_labels = labels[0]
    for learn_range in learn_ranges:
        start = torch.nonzero(offset_mapping[:, 0] == learn_range.start).flatten()[0].item()
        if offset_mapping[:, 0][-1] >= learn_range.stop:
            stop = torch.nonzero(offset_mapping[:, 0] == learn_range.stop).flatten()[0].item()
        else:
            stop = len(input_ids)
        item_labels[start - 1 : stop - 1] = input_ids[start:stop]
        item_labels[item_labels >= len(tokenizer) - 1] = tokenizer.eos_token_id
    tokenized.pop("offset_mapping")
    tokenized["labels"] = labels
    return tokenized.to(device)


def interval_accuracy(model, input_ids, frames, labels) -> tuple[int, int]:
    frame_token_interval_id = getattr(model.config, "frame_token_interval_id", None)
    if frame_token_interval_id is None:
        return 0, 0
    outputs = model(
        input_ids=input_ids,
        frames=frames,
        return_dict=True,
        use_cache=False,
    )
    stream_mask = (input_ids == model.config.v_placeholder_id) & (labels != -100)
    if not stream_mask.any():
        return 0, 0
    predictions = outputs.logits.argmax(dim=-1)
    gold_continue = labels[stream_mask] == frame_token_interval_id
    pred_continue = predictions[stream_mask] == frame_token_interval_id
    correct = (gold_continue == pred_continue).sum().item()
    return correct, gold_continue.numel()


def find_token_positions_for_last_text(
    offset_mapping: torch.Tensor,
    text: str,
    target: str,
) -> list[int]:
    start = text.lower().rfind(target.lower())
    if start < 0:
        return []
    stop = start + len(target)
    positions = []
    for index, (token_start, token_stop) in enumerate(offset_mapping.tolist()):
        if token_stop <= start or token_start >= stop:
            continue
        if token_stop > token_start:
            positions.append(index)
    return positions


def emotion_token_metrics(
    *,
    model,
    tokenizer,
    text: str,
    frames: torch.Tensor,
    emotion: str,
    device: str,
) -> dict:
    tokenized = tokenizer(
        [text],
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    token_positions = find_token_positions_for_last_text(
        tokenized.offset_mapping[0],
        text,
        emotion,
    )
    if not token_positions:
        return {
            "emotion_prediction": "",
            "normalized_prediction": "",
            "normalized_gold": normalize_emotion(emotion),
            "emotion_token_correct": 0,
            "emotion_token_total": 0,
            "emotion_token_accuracy": 0.0,
            "emotion_exact_match": False,
        }

    input_ids = tokenized.input_ids.to(device)
    frames = frames.to(device=device, dtype=torch.bfloat16)
    outputs = model(input_ids=input_ids, frames=frames, return_dict=True, use_cache=False)

    gold_ids = []
    pred_ids = []
    for token_position in token_positions:
        if token_position == 0:
            continue
        logit_position = token_position - 1
        gold_id = int(input_ids[0, token_position].item())
        pred_id = int(outputs.logits[0, logit_position].argmax(dim=-1).item())
        gold_ids.append(gold_id)
        pred_ids.append(pred_id)

    correct = sum(int(pred_id == gold_id) for pred_id, gold_id in zip(pred_ids, gold_ids))
    total = len(gold_ids)
    prediction = tokenizer.decode(pred_ids, skip_special_tokens=True)
    normalized_prediction = normalize_emotion(prediction)
    normalized_gold = normalize_emotion(emotion)
    return {
        "emotion_prediction": prediction,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "emotion_token_correct": correct,
        "emotion_token_total": total,
        "emotion_token_accuracy": correct / total if total else 0.0,
        "emotion_exact_match": normalized_prediction == normalized_gold,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    config = load_config(args.config).values
    model_config = config["model"]
    llm_config = model_config["llm"]
    vision_config = model_config["vision_encoder"]
    projector_config = model_config["projector"]
    data_config = config["data"]
    streaming_window = data_config["streaming_window"]

    model, tokenizer = build_live_llama(
        is_training=False,
        llm_pretrained=llm_config["name_or_path"],
        resume_from_checkpoint=args.checkpoint,
        attn_implementation=llm_config.get("attn_implementation", "sdpa"),
        torch_dtype=torch.bfloat16,
        local_files_only=bool(llm_config.get("local_files_only", True)),
        vision_pretrained=vision_config.get("name_or_path"),
        frame_resolution=int(vision_config.get("frame_size", 384)),
        frame_token_cls=bool(vision_config.get("frame_token_cls", True)),
        frame_token_pooled=vision_config.get("frame_token_pooled", [3, 3]),
        frame_num_tokens=int(vision_config.get("frame_num_tokens", 10)),
        frame_token_interval=",",
        stream_loss_weight=1.0,
        vision_hidden_size=int(projector_config.get("input_size", 1024)),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    base_model = get_base_model(model)
    if not hasattr(base_model.generation_config, "_original_object_hash"):
        base_model.generation_config._original_object_hash = hash(
            base_model.generation_config
        )

    manifest_key = f"{args.split}_manifest"
    dataset = StreamingEmotionDataset(
        data_config[manifest_key],
        tokenizer,
        is_training=False,
        system_prompt=EMOTION_TOKEN_PROMPT,
        max_num_frames=int(streaming_window.get("max_num_frames", 64)),
        fps=float(streaming_window.get("fps", 2.0)),
        context_mode=streaming_window.get("mode", "prefix_until_event"),
        add_generation_prompt=False,
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path(config["experiment"]["output_dir"]) / f"{args.split}_streaming_predictions.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = min(args.limit, len(dataset)) if args.limit > 0 else len(dataset)
    rows = []
    interval_correct = 0
    interval_total = 0
    emotion_exact_correct = 0
    emotion_token_correct = 0
    emotion_token_total = 0
    with torch.no_grad(), output_path.open("w", encoding="utf-8") as handle:
        for index in tqdm(range(total), desc=f"stream-evaluating {args.split}"):
            text, frames, learn_ranges, _, meta = dataset[index]
            batch = build_labeled_batch(tokenizer, text, learn_ranges, device)
            frames = frames.to(device=device, dtype=torch.bfloat16)
            stream_metrics = base_model.stream_evaluate(
                batch.input_ids,
                batch.labels,
                frames,
                frame_token_interval_threshold=args.frame_token_interval_threshold,
            )
            item_interval_correct, item_interval_total = interval_accuracy(
                base_model,
                batch.input_ids,
                frames,
                batch.labels,
            )
            interval_correct += item_interval_correct
            interval_total += item_interval_total
            emotion_metrics = emotion_token_metrics(
                model=base_model,
                tokenizer=tokenizer,
                text=text,
                frames=frames,
                emotion=meta["emotion"],
                device=device,
            )
            emotion_exact_correct += int(emotion_metrics["emotion_exact_match"])
            emotion_token_correct += emotion_metrics["emotion_token_correct"]
            emotion_token_total += emotion_metrics["emotion_token_total"]
            row = {
                **meta,
                **emotion_metrics,
                "lm_ppl": float(stream_metrics[0].item()),
                "frame_diff": float(stream_metrics[1].item()),
                "fluency": float(stream_metrics[2].item()),
                "lm_correctness": float(stream_metrics[3].item()),
                "interval_correct": item_interval_correct,
                "interval_total": item_interval_total,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    emotion_exact_accuracy = emotion_exact_correct / total if total else 0.0
    emotion_token_accuracy = (
        emotion_token_correct / emotion_token_total if emotion_token_total else 0.0
    )
    interval_acc = interval_correct / interval_total if interval_total else 0.0
    print(
        f"split={args.split} total={total} "
        f"emotion_exact_match={emotion_exact_accuracy:.4f} "
        f"emotion_token_accuracy={emotion_token_accuracy:.4f}"
    )
    print(
        "streaming "
        f"frame_diff={mean([row['frame_diff'] for row in rows]):.4f} "
        f"fluency={mean([row['fluency'] for row in rows]):.4f} "
        f"lm_correctness={mean([row['lm_correctness'] for row in rows]):.4f} "
        f"lm_ppl={mean([row['lm_ppl'] for row in rows]):.4f}"
    )
    print(f"interval_accuracy {interval_correct}/{interval_total}={interval_acc:.4f}")
    for row in rows[:10]:
        print(
            f"{row['sample_id']}#{row['event_index']} "
            f"gold={row['normalized_gold']} pred={row['normalized_prediction']}"
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
