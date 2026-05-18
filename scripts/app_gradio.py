import argparse
import json
from pathlib import Path

import gradio as gr

from streaming_emotion_llm.config import load_config
from streaming_emotion_llm.inference.generation import (
    build_generation_dataset,
    decode_generated_emotion,
    greedy_generate_ids,
    load_streaming_model,
    normalize_emotion,
    token_overlap_metrics,
)


MODEL_CACHE = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for streaming emotion LLM.")
    parser.add_argument(
        "--config",
        default="configs/experiments/fullvideo_lora_r32_tinyllama_siglip_rtx4060_8gb.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/fullvideo_event_stream_tinyllama_siglip_lora_r32_rtx4060_8gb/checkpoint-462",
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


def build_app(default_config: str, default_checkpoint: str):
    with gr.Blocks(title="Streaming Emotion LLM") as demo:
        gr.Markdown("# Streaming Emotion LLM")
        with gr.Row():
            config_path = gr.Textbox(label="Config", value=default_config)
            checkpoint = gr.Textbox(label="Checkpoint", value=default_checkpoint)
        with gr.Row():
            split = gr.Dropdown(["train", "val", "test"], value="val", label="Split")
            sample_index = gr.Number(value=0, precision=0, label="Event sample index")
            max_new_tokens = gr.Slider(1, 16, value=8, step=1, label="Max new tokens")
        run = gr.Button("Generate")
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
    return demo


def main() -> None:
    args = parse_args()
    demo = build_app(args.config, args.checkpoint)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
