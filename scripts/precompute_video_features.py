import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import transformers
import cv2
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from streaming_emotion_llm.models.vision_live import build_live_vision


try:
    torchvision.set_video_backend("video_reader")
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute full-video SigLIP features for streaming emotion training."
    )
    parser.add_argument("--videos-dir", default="data/raw/videos")
    parser.add_argument(
        "--output-dir",
        default="data/processed/features/siglip_large_384_2fps_1plus3x3",
    )
    parser.add_argument("--manifest", default="data/manifests/all_valid.jsonl")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vision-pretrained", default="google/siglip-large-patch16-384")
    parser.add_argument("--frame-token-cls", action="store_true", default=True)
    parser.add_argument("--frame-token-pooled", type=int, nargs=2, default=[3, 3])
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick subset runs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_and_pad(frames: torch.Tensor, resolution: int) -> torch.Tensor:
    frames = frames.float()
    _, _, height, width = frames.shape
    if width >= height:
        new_width = resolution
        new_height = max(1, round(height * resolution / width))
    else:
        new_height = resolution
        new_width = max(1, round(width * resolution / height))

    frames = F.interpolate(
        frames,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )
    pad_left = (resolution - new_width) // 2
    pad_right = resolution - new_width - pad_left
    pad_top = (resolution - new_height) // 2
    pad_bottom = resolution - new_height - pad_top
    return F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom))


def sample_video(path: Path, fps: float, resolution: int) -> torch.Tensor:
    try:
        reader = torchvision.io.VideoReader(str(path), "video")
        frames = []
        next_time = 0.0
        for frame in reader:
            pts = float(frame["pts"])
            if pts + 1e-6 < next_time:
                continue
            frames.append(frame["data"])
            next_time += 1.0 / fps
    except Exception:
        try:
            video, _, info = torchvision.io.read_video(str(path), pts_unit="sec", output_format="TCHW")
            source_fps = float(info.get("video_fps", fps))
            step = max(1, int(round(source_fps / fps)))
            frames = [frame for frame in video[::step]]
        except Exception:
            frames = sample_video_opencv(path, fps)
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return resize_and_pad(torch.stack(frames), resolution)


def sample_video_opencv(path: Path, fps: float) -> list[torch.Tensor]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or math.isnan(source_fps) or source_fps <= 0:
        source_fps = fps
    step = max(1, int(round(source_fps / fps)))

    frames = []
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous())
        frame_idx += 1
    capture.release()
    return frames


def load_manifest_video_ids(manifest_path: Path) -> list[str]:
    sample_ids = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample_ids.append(json.loads(line)["sample_id"])
    return sorted(set(sample_ids))


def output_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def save_metadata(output_dir: Path, metadata: dict) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = load_manifest_video_ids(manifest_path)
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    vision_config = {
        "name_or_path": args.vision_pretrained,
        "frame_token_cls": args.frame_token_cls,
        "frame_token_pooled": args.frame_token_pooled,
    }
    vision_model, vision_encode = build_live_vision(vision_config)
    vision_model.to(args.device).eval()

    metadata = {
        "vision_pretrained": args.vision_pretrained,
        "fps": args.fps,
        "resolution": args.resolution,
        "frame_token_cls": args.frame_token_cls,
        "frame_token_pooled": args.frame_token_pooled,
        "dtype": args.dtype,
        "videos": {},
    }
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["videos"].update(existing.get("videos", {}))

    tensor_dtype = output_dtype(args.dtype)
    failures = []
    for sample_id in tqdm(sample_ids, desc="precompute video features"):
        video_path = videos_dir / f"{sample_id}.mp4"
        save_path = output_dir / f"{sample_id}.pt"
        if save_path.exists() and not args.overwrite:
            continue
        if not video_path.exists():
            failures.append({"sample_id": sample_id, "error": "missing_video"})
            continue

        try:
            frames = sample_video(video_path, args.fps, args.resolution)
            embeds = []
            with torch.inference_mode(), torch.cuda.amp.autocast(
                enabled=str(args.device).startswith("cuda")
            ):
                for batch in frames.split(args.batch_size):
                    batch = batch.to(args.device, non_blocking=True)
                    embeds.append(vision_encode(vision_model, batch).cpu())
            embeds = torch.cat(embeds).to(tensor_dtype)
            torch.save(embeds, save_path)

            duration = (len(embeds) - 1) / args.fps if len(embeds) > 1 else 0.0
            metadata["videos"][sample_id] = {
                "path": save_path.as_posix(),
                "video_path": video_path.as_posix(),
                "num_feature_frames": int(len(embeds)),
                "duration_sec": duration,
                "feature_shape": list(embeds.shape),
            }
            save_metadata(output_dir, metadata)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": repr(exc)})

    if failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )
    save_metadata(output_dir, metadata)
    print(f"features_dir={output_dir}")
    print(f"encoded_videos={len(metadata['videos'])}")
    print(f"failures={len(failures)}")


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    main()
