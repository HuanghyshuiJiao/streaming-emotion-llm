import argparse
import json
from pathlib import Path

import gradio as gr
import torch

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.data.data_collator import get_data_collator
from streaming_emotion_llm.data.stream import StreamingEmotionDataset
from streaming_emotion_llm.inference.generation import (
    build_generation_dataset,
    decode_generated_emotion,
    greedy_generate_ids,
    load_streaming_model,
    normalize_emotion,
    stream_autoregressive_features,
    token_overlap_metrics,
)
from streaming_emotion_llm.prompts.templates import EMOTION_TOKEN_PROMPT


MODEL_CACHE = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for streaming emotion LLM.")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp2_r32_32videos.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/event_stream_original_interval_r32_fullvideo_subset_overfit_tinyllama_siglip_rtx4060_8gb/final",
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def get_model_and_dataset(config_path: str, checkpoint: str, split: str):
    key = (str(config_path), str(checkpoint), split)
    if key not in MODEL_CACHE:
        config = load_config(config_path).values
        model, tokenizer, device = load_streaming_model(config, checkpoint)
        dataset = build_generation_dataset(config, tokenizer, split=split)
        MODEL_CACHE[key] = (config, model, tokenizer, device, dataset)
    return MODEL_CACHE[key]


def sample_count(config_path: str, split: str) -> int:
    config = load_config(config_path).values
    manifest = config["data"][f"{split}_manifest"]
    return len(read_manifest(manifest))


def load_feature_tensor(path: str | Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def build_teacher_forcing_dataset(config: dict, tokenizer, split: str):
    data_config = config["data"]
    streaming_window = data_config.get("streaming_window", {})
    return StreamingEmotionDataset(
        data_config[f"{split}_manifest"],
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


def predict(config_path: str, checkpoint: str, split: str, sample_index: int, max_new_tokens: int):
    config, model, tokenizer, device, dataset = get_model_and_dataset(config_path, checkpoint, split)
    index = max(0, min(int(sample_index), len(dataset) - 1))
    text, frames, _, _, meta = dataset[index]
    generated_ids = greedy_generate_ids(
        model=model,
        tokenizer=tokenizer,
        text=text,
        frames=frames,
        device=device,
        max_new_tokens=int(max_new_tokens),
    )
    raw_prediction = decode_generated_emotion(tokenizer, generated_ids)
    pred = normalize_emotion(raw_prediction)
    gold = normalize_emotion(meta["emotion"])
    token_correct, token_total = token_overlap_metrics(tokenizer, pred, gold)
    exact = pred == gold

    manifest_records = read_manifest(config["data"][f"{split}_manifest"])
    video_path = ""
    if 0 <= index < len(dataset.samples):
        sample_id = dataset.samples[index]["sample_id"]
        for record in manifest_records:
            if record["sample_id"] == sample_id:
                video_path = record.get("video_path", "")
                break

    metrics = {
        "sample_id": meta["sample_id"],
        "event_index": meta["event_index"],
        "timestamp": meta["timestamp"],
        "gold": gold,
        "prediction": pred,
        "exact": exact,
        "token_accuracy": token_correct / token_total if token_total else 0.0,
        "token_correct": token_correct,
        "token_total": token_total,
        "raw_prediction": raw_prediction,
    }
    table = [[
        meta["sample_id"],
        meta["event_index"],
        f"{float(meta['timestamp']):.2f}s",
        gold,
        pred,
        "yes" if exact else "no",
        f"{metrics['token_accuracy']:.2%}",
    ]]
    return video_path if Path(video_path).exists() else None, table, metrics, text


def run_video_stream(
    config_path: str,
    checkpoint: str,
    split: str,
    video_index: int,
    max_new_tokens: int,
    frame_token_interval_threshold: float,
    max_frames: int,
):
    config, model, tokenizer, device, _ = get_model_and_dataset(config_path, checkpoint, split)
    manifest_records = read_manifest(config["data"][f"{split}_manifest"])
    if not manifest_records:
        return None, [], [], {"error": "empty manifest"}

    index = max(0, min(int(video_index), len(manifest_records) - 1))
    record = manifest_records[index]
    fps = float(config["data"].get("streaming_window", {}).get("fps", 2.0))
    frames = load_feature_tensor(record["feature_path"])
    predictions = stream_autoregressive_features(
        model=model,
        tokenizer=tokenizer,
        frames=frames,
        device=device,
        fps=fps,
        max_new_tokens=int(max_new_tokens),
        frame_token_interval_threshold=float(frame_token_interval_threshold),
        max_frames=int(max_frames) if int(max_frames) > 0 else None,
    )

    gold_events = [
        {
            "event_index": event_index,
            "timestamp": float(event.get("timestamp", 0.0)),
            "emotion": normalize_emotion(str(event.get("emotion", ""))),
        }
        for event_index, event in enumerate(record.get("events", []))
        if str(event.get("emotion", "")).strip()
    ]
    tf_rows, tf_metrics, _ = teacher_forcing_event_rows(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        split=split,
        index=index,
    )
    tf_by_event = {row[0]: row for row in tf_rows}
    prediction_rows = [
        [
            item["frame_index"],
            f"{item['timestamp']:.2f}s",
            item["prediction"],
            item["raw_prediction"],
        ]
        for item in predictions
    ]
    gold_rows = [
        [
            item["event_index"],
            f"{item['timestamp']:.2f}s",
            item["emotion"],
            tf_by_event.get(item["event_index"], ["", "", "", "", ""])[3],
            tf_by_event.get(item["event_index"], ["", "", "", "", ""])[4],
        ]
        for item in gold_events
    ]
    metrics = {
        "sample_id": record.get("sample_id"),
        "video_index": index,
        "num_feature_frames": int(frames.shape[0]),
        "num_gold_events": len(gold_events),
        "num_stream_predictions": len(predictions),
        "fps": fps,
        "max_frames": int(max_frames),
        "teacher_forcing_emotion_exact": tf_metrics.get("emotion_exact"),
        "teacher_forcing_interval_accuracy": tf_metrics.get("stream_interval_accuracy"),
    }
    video_path = record.get("video_path", "")
    return video_path if Path(video_path).exists() else None, prediction_rows, gold_rows, metrics


def teacher_forcing_event_rows(
    *,
    config: dict,
    model,
    tokenizer,
    device,
    split: str,
    index: int,
) -> tuple[list[list], dict, str]:
    dataset = build_teacher_forcing_dataset(config, tokenizer, split)
    manifest_records = read_manifest(config["data"][f"{split}_manifest"])
    if not manifest_records:
        return [], {"error": "empty manifest"}, ""

    index = max(0, min(int(index), len(dataset) - 1))
    text, frames, learn_ranges, _, meta = dataset[index]
    collator = get_data_collator(tokenizer=tokenizer)
    batch = collator([(text, frames, learn_ranges, index, meta)])
    batch = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key != "evaluation_kwargs"
    }
    with torch.no_grad():
        outputs = model(**batch, return_dict=True, use_cache=False)

    labels = batch["labels"]
    input_ids = batch["input_ids"]
    pred_ids = outputs.logits.argmax(dim=-1)
    valid_mask = labels != -100
    stream_mask = valid_mask & (input_ids == model.config.v_placeholder_id)
    lm_mask = valid_mask & ~stream_mask

    record = manifest_records[index]
    search_start = 0
    rows = []
    exact_count = 0
    found_count = 0
    for event_index, event in enumerate(record.get("events", [])):
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
            rows.append([event_index, f"{float(event.get('timestamp', 0.0)):.2f}s", gold, "", "not found"])
            continue
        gold_token_ids = [int(input_ids[0, pos].item()) for pos in positions if pos > 0]
        pred_token_ids = [int(pred_ids[0, pos - 1].item()) for pos in positions if pos > 0]
        prediction = normalize_emotion(tokenizer.decode(pred_token_ids, skip_special_tokens=True))
        exact = prediction == gold
        exact_count += int(exact)
        found_count += 1
        rows.append(
            [
                event_index,
                f"{float(event.get('timestamp', 0.0)):.2f}s",
                gold,
                prediction,
                "yes" if exact else "no",
            ]
        )

    metrics = {
        "sample_id": record.get("sample_id"),
        "video_index": index,
        "frame_stop": meta.get("frame_stop"),
        "included_events": meta.get("included_events"),
        "found_events": found_count,
        "emotion_exact": exact_count / found_count if found_count else 0.0,
        "stream_interval_accuracy": float(
            (pred_ids[stream_mask] == model.config.frame_token_interval_id).float().mean().item()
        )
        if stream_mask.any()
        else 0.0,
        "lm_token_accuracy": float((pred_ids[lm_mask] == labels[lm_mask]).float().mean().item())
        if lm_mask.any()
        else 0.0,
    }
    return rows, metrics, text


def run_teacher_forcing_video(
    config_path: str,
    checkpoint: str,
    split: str,
    video_index: int,
):
    config, model, tokenizer, device, _ = get_model_and_dataset(config_path, checkpoint, split)
    rows, metrics, text = teacher_forcing_event_rows(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        split=split,
        index=int(video_index),
    )
    manifest_records = read_manifest(config["data"][f"{split}_manifest"])
    if not manifest_records:
        return None, [], metrics, text
    index = max(0, min(int(video_index), len(manifest_records) - 1))
    record = manifest_records[index]
    video_path = record.get("video_path", "")
    return video_path if Path(video_path).exists() else None, rows, metrics, text


def build_app(default_config: str, default_checkpoint: str):
    with gr.Blocks(title="Streaming Emotion LLM") as demo:
        gr.Markdown("# Streaming Emotion LLM")
        with gr.Row():
            config_path = gr.Textbox(label="Config", value=default_config)
            checkpoint = gr.Textbox(label="Checkpoint", value=default_checkpoint)

        with gr.Tabs():
            with gr.Tab("Event"):
                with gr.Row():
                    split = gr.Dropdown(["train", "val", "test"], value="val", label="Split")
                    sample_index = gr.Number(value=0, precision=0, label="Event sample index")
                    max_new_tokens = gr.Slider(1, 16, value=8, step=1, label="Max new tokens")
                run = gr.Button("Generate event")
                with gr.Row():
                    video = gr.Video(label="Source video")
                    table = gr.Dataframe(
                        headers=[
                            "sample_id",
                            "event",
                            "timestamp",
                            "label",
                            "prediction",
                            "exact",
                            "token_acc",
                        ],
                        label="Prediction",
                    )
                metrics = gr.JSON(label="Metrics")
                prompt = gr.Textbox(label="Prompt", lines=12)

                run.click(
                    predict,
                    inputs=[config_path, checkpoint, split, sample_index, max_new_tokens],
                    outputs=[video, table, metrics, prompt],
                )

            with gr.Tab("Video Stream"):
                with gr.Row():
                    stream_split = gr.Dropdown(["train", "val", "test"], value="val", label="Split")
                    video_index = gr.Number(value=0, precision=0, label="Video index")
                    stream_max_new_tokens = gr.Slider(1, 16, value=8, step=1, label="Max new tokens")
                with gr.Row():
                    interval_threshold = gr.Slider(
                        0.0,
                        1.0,
                        value=0.85,
                        step=0.025,
                        label="Interval threshold",
                    )
                    max_frames = gr.Number(
                        value=0,
                        precision=0,
                        label="Max frames (0 = full video)",
                    )
                run_stream = gr.Button("Run video stream")
                stream_video = gr.Video(label="Source video")
                with gr.Row():
                    stream_predictions = gr.Dataframe(
                        headers=["frame", "timestamp", "prediction", "raw"],
                        label="Autoregressive stream predictions",
                    )
                    gold_events = gr.Dataframe(
                        headers=["event", "timestamp", "gold", "TF pred", "TF exact"],
                        label="Gold event timeline + teacher forcing",
                    )
                stream_metrics = gr.JSON(label="Stream metrics")

                run_stream.click(
                    run_video_stream,
                    inputs=[
                        config_path,
                        checkpoint,
                        stream_split,
                        video_index,
                        stream_max_new_tokens,
                        interval_threshold,
                        max_frames,
                    ],
                    outputs=[stream_video, stream_predictions, gold_events, stream_metrics],
                )

            with gr.Tab("Teacher Forcing"):
                with gr.Row():
                    tf_split = gr.Dropdown(["train", "val", "test"], value="train", label="Split")
                    tf_video_index = gr.Number(value=0, precision=0, label="Video index")
                run_tf = gr.Button("Run teacher forcing")
                tf_video = gr.Video(label="Source video")
                tf_rows = gr.Dataframe(
                    headers=["event", "timestamp", "label", "teacher-forced pred", "exact"],
                    label="Label vs prediction",
                )
                tf_metrics = gr.JSON(label="Teacher-forcing metrics")
                tf_prompt = gr.Textbox(label="Full prompt", lines=12)

                run_tf.click(
                    run_teacher_forcing_video,
                    inputs=[config_path, checkpoint, tf_split, tf_video_index],
                    outputs=[tf_video, tf_rows, tf_metrics, tf_prompt],
                )
    return demo


def main() -> None:
    args = parse_args()
    demo = build_app(args.config, args.checkpoint)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
