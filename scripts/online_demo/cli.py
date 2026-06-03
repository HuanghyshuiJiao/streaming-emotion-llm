"""Original VideoLLM-online-style CLI benchmark adapted for emotion streaming."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument(
        "--fps-window",
        type=int,
        default=10,
        help="Window size for rolling FPS in the per-frame curve.",
    )
    parser.add_argument(
        "--curve-output",
        default=None,
        help="Optional CSV path for per-frame latency/FPS curve data.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional PNG path for a latency/FPS curve plot. Requires matplotlib.",
    )
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


def rolling_fps(costs: list[float], window: int) -> float:
    if not costs:
        return 0.0
    window = max(1, int(window))
    recent_costs = costs[-window:]
    total = sum(recent_costs)
    return len(recent_costs) / total if total else 0.0


def write_curve_csv(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "video_time",
        "wall_time",
        "frame_cost",
        "instant_fps",
        "rolling_fps",
        "cumulative_fps",
        "triggered",
        "response",
        "normalized_response",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_curve_plot(path: str | Path, rows: list[dict]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not rows:
        return False

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    times = [row["video_time"] for row in rows]
    cumulative = [row["cumulative_fps"] for row in rows]
    rolling = [row["rolling_fps"] for row in rows]
    costs_ms = [row["frame_cost"] * 1000 for row in rows]
    trigger_times = [row["video_time"] for row in rows if row["triggered"]]

    fig, fps_axis = plt.subplots(figsize=(8, 4.5), dpi=160)
    fps_axis.plot(times, rolling, label="Rolling FPS", linewidth=2)
    fps_axis.plot(times, cumulative, label="Cumulative FPS", linewidth=2, alpha=0.75)
    for trigger_time in trigger_times:
        fps_axis.axvline(trigger_time, color="#d95f02", alpha=0.2, linewidth=1)
    fps_axis.set_xlabel("Video time (s)")
    fps_axis.set_ylabel("Processing FPS")
    fps_axis.grid(True, alpha=0.25)
    fps_axis.legend(loc="upper left")

    cost_axis = fps_axis.twinx()
    cost_axis.plot(times, costs_ms, label="Frame cost", color="#7570b3", alpha=0.35)
    cost_axis.set_ylabel("Frame cost (ms)")

    fig.suptitle("Online Streaming Throughput")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def main(liveinfer: LiveInfer, args: argparse.Namespace):
    src_video_path = args.video
    save_history_path = args.output

    liveinfer.load_video(src_video_path)

    timecosts = []
    frame_metrics = []
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
        "frame_metrics": frame_metrics,
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
        frame_cost = end_time - start_time
        timecosts.append(frame_cost)
        cumulative_fps = (i + 1) / sum(timecosts)
        instant_fps = 1 / frame_cost if frame_cost else 0.0
        current_rolling_fps = rolling_fps(timecosts, args.fps_window)
        normalized_response = liveinfer.normalize_response(response)
        frame_metric = {
            "frame_index": i,
            "video_time": liveinfer.video_time,
            "wall_time": end_time - wall_start,
            "frame_cost": frame_cost,
            "instant_fps": instant_fps,
            "rolling_fps": current_rolling_fps,
            "cumulative_fps": cumulative_fps,
            "triggered": response is not None,
            "response": response or "",
            "normalized_response": normalized_response or "",
        }
        frame_metrics.append(frame_metric)
        pbar.set_postfix_str(
            f"Avg FPS: {cumulative_fps:.1f}, Rolling FPS: {current_rolling_fps:.1f}"
        )
        pbar.update(1)
        if query:
            history["conversation"].append(
                {
                    "role": "user",
                    "content": query,
                    "time": liveinfer.video_time,
                    "cumulative_fps": cumulative_fps,
                    "rolling_fps": current_rolling_fps,
                    "cost": frame_cost,
                }
            )
            print(query)
        if response:
            history["conversation"].append(
                {
                    "role": "assistant",
                    "content": response,
                    "normalized": normalized_response,
                    "time": liveinfer.video_time,
                    "cumulative_fps": cumulative_fps,
                    "rolling_fps": current_rolling_fps,
                    "cost": frame_cost,
                }
            )
            print(response)
        if not query and not response:
            history["conversation"].append(
                {
                    "time": liveinfer.video_time,
                    "cumulative_fps": cumulative_fps,
                    "rolling_fps": current_rolling_fps,
                    "cost": frame_cost,
                }
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
    if frame_metrics:
        summary["final_cumulative_fps"] = frame_metrics[-1]["cumulative_fps"]
        summary["final_rolling_fps"] = frame_metrics[-1]["rolling_fps"]
        summary["min_rolling_fps"] = min(item["rolling_fps"] for item in frame_metrics)
        summary["max_rolling_fps"] = max(item["rolling_fps"] for item in frame_metrics)
    history["summary"] = summary
    print(json.dumps(summary, indent=2))

    if save_history_path:
        output_path = Path(save_history_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"The conversation history has been saved to {output_path}.")

    curve_output = args.curve_output
    if curve_output is None and save_history_path:
        curve_output = str(Path(save_history_path).with_suffix("")) + "_fps_curve.csv"
    if curve_output:
        write_curve_csv(curve_output, frame_metrics)
        print(f"The FPS curve data has been saved to {curve_output}.")

    plot_output = args.plot_output
    if plot_output is None and save_history_path:
        plot_output = str(Path(save_history_path).with_suffix("")) + "_fps_curve.png"
    if plot_output:
        if write_curve_plot(plot_output, frame_metrics):
            print(f"The FPS curve plot has been saved to {plot_output}.")
        else:
            print("Skipped FPS curve plot because matplotlib is unavailable or no frames were run.")


if __name__ == "__main__":
    parsed_args = parse_args()
    liveinfer = LiveInfer(
        config_path=parsed_args.config,
        checkpoint=parsed_args.checkpoint,
        frame_token_interval_threshold=parsed_args.threshold,
        max_new_tokens=parsed_args.max_new_tokens,
    )
    main(liveinfer, parsed_args)
