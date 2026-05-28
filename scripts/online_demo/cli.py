"""Original VideoLLM-online-style CLI benchmark adapted for emotion streaming."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import tqdm
import transformers

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import LiveInfer


logger = transformers.logging.get_logger("original-online")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark original-style online streaming.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True, help="Pre-sampled frame tensor .pt file.")
    parser.add_argument("--threshold", type=float, default=0.725)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def summarize_costs(timecosts: list[float]) -> dict:
    if not timecosts:
        return {}
    sorted_costs = sorted(timecosts)

    def pct(value: float) -> float:
        index = round((value / 100.0) * (len(sorted_costs) - 1))
        return sorted_costs[max(0, min(len(sorted_costs) - 1, index))]

    return {
        "avg_frame_cost": statistics.mean(timecosts),
        "p50_frame_cost": pct(50),
        "p90_frame_cost": pct(90),
        "p95_frame_cost": pct(95),
    }


def main(liveinfer: LiveInfer, args: argparse.Namespace):
    src_video_path = args.video
    save_history_path = args.output

    liveinfer.load_video(src_video_path)

    timecosts = []
    frame_total = (
        min(args.max_frames, liveinfer.num_video_frames)
        if args.max_frames > 0
        else liveinfer.num_video_frames
    )
    pbar = tqdm.tqdm(
        total=frame_total,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]",
    )
    history = {
        "video_path": src_video_path,
        "frame_fps": liveinfer.frame_fps,
        "threshold": liveinfer.frame_token_interval_threshold,
        "conversation": [],
    }

    if liveinfer.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for i in range(frame_total):
        if liveinfer.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        liveinfer.input_video_stream(i / liveinfer.frame_fps)
        query, response = liveinfer()
        if liveinfer.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        timecosts.append(end_time - start_time)
        fps = (i + 1) / sum(timecosts)
        pbar.set_postfix_str(f"Average Processing FPS: {fps:.1f}")
        pbar.update(1)
        if query:
            history["conversation"].append(
                {
                    "role": "user",
                    "content": query,
                    "time": liveinfer.video_time,
                    "fps": fps,
                    "cost": timecosts[-1],
                }
            )
            print(query)
        if response:
            history["conversation"].append(
                {
                    "role": "assistant",
                    "content": response,
                    "normalized": liveinfer.normalize_response(response),
                    "time": liveinfer.video_time,
                    "fps": fps,
                    "cost": timecosts[-1],
                }
            )
            print(response)
        if not query and not response:
            history["conversation"].append(
                {"time": liveinfer.video_time, "fps": fps, "cost": timecosts[-1]}
            )
    if liveinfer.device == "cuda":
        torch.cuda.synchronize()
    wall_time = time.perf_counter() - wall_start
    pbar.close()

    summary = {
        "frames": frame_total,
        "wall_time": wall_time,
        "average_processing_fps": frame_total / wall_time if wall_time else 0.0,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024**3)
            if liveinfer.device == "cuda"
            else 0.0
        ),
        **summarize_costs(timecosts),
    }
    history["summary"] = summary
    print(json.dumps(summary, indent=2))

    if save_history_path:
        output_path = Path(save_history_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"The conversation history has been saved to {output_path}.")


if __name__ == "__main__":
    parsed_args = parse_args()
    liveinfer = LiveInfer(
        config_path=parsed_args.config,
        checkpoint=parsed_args.checkpoint,
        frame_token_interval_threshold=parsed_args.threshold,
        max_new_tokens=parsed_args.max_new_tokens,
    )
    main(liveinfer, parsed_args)
