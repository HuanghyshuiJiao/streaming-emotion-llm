import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACEXFORMER_ROOT = PROJECT_ROOT / "reference" / "facexformer"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FACEXFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(FACEXFORMER_ROOT))

from network import FaceXFormer  # noqa: E402
from scripts.precompute_video_features import sample_video  # noqa: E402

try:
    from facenet_pytorch import MTCNN  # noqa: E402
except ImportError:  # pragma: no cover - optional dependency
    MTCNN = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute one FaceXFormer face token per sampled video frame."
    )
    parser.add_argument("--videos-dir", default="data/raw/videos")
    parser.add_argument("--manifest", default="data/manifests/all_valid.jsonl")
    parser.add_argument(
        "--output-dir",
        default="data/processed/features/facexformer_2fps_face_token_raw256",
    )
    parser.add_argument("--model-path", default="reference/facexformer/ckpts/model.pt")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument(
        "--face-crop-mode",
        choices=["center", "mtcnn"],
        default="center",
        help="center keeps the original deterministic crop; mtcnn follows the FaceXFormer demo.",
    )
    parser.add_argument("--face-margin", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def read_manifest_video_ids(manifest_path: Path) -> list[str]:
    sample_ids = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sample_ids.append(json.loads(line)["sample_id"])
    return sorted(set(sample_ids))


def normalize_imagenet(frames: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
    return (frames / 255.0 - mean) / std


def center_crop_square(frames: torch.Tensor) -> torch.Tensor:
    _, _, height, width = frames.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frames[:, :, top : top + side, left : left + side]


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    array = frame.permute(1, 2, 0).cpu().byte().numpy()
    return Image.fromarray(array)


def adjust_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_width: int,
    image_height: int,
    margin_percentage: float = 50.0,
) -> tuple[int, int, int, int]:
    width = x_max - x_min
    height = y_max - y_min
    increase_width = width * (margin_percentage / 100.0) / 2
    increase_height = height * (margin_percentage / 100.0) / 2
    x_min = max(0, x_min - increase_width)
    y_min = max(0, y_min - increase_height)
    x_max = min(image_width, x_max + increase_width)
    y_max = min(image_height, y_max + increase_height)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def center_crop_pil(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def build_face_transform(face_size: int):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(face_size, face_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def prepare_faces(frames: torch.Tensor, face_size: int) -> torch.Tensor:
    # The current dataset is already face-centric. Keep this deterministic and
    # aligned with SigLIP sampling by using a center square crop per sampled frame.
    frames = center_crop_square(frames.float())
    frames = F.interpolate(
        frames,
        size=(face_size, face_size),
        mode="bicubic",
        align_corners=False,
    )
    return normalize_imagenet(frames)


def prepare_faces_mtcnn(
    frames: torch.Tensor,
    face_size: int,
    *,
    mtcnn,
    margin_percentage: float = 50.0,
) -> tuple[torch.Tensor, list[dict]]:
    transform = build_face_transform(face_size)
    face_tensors = []
    records = []
    for frame_index, frame in enumerate(frames):
        image = tensor_to_pil(frame)
        width, height = image.size
        boxes, probs = mtcnn.detect(image)
        selected_box = None
        selected_prob = None
        if boxes is not None and len(boxes) > 0:
            best_index = int(probs.argmax()) if probs is not None else 0
            selected_box = boxes[best_index]
            selected_prob = float(probs[best_index]) if probs is not None else None
            crop_box = adjust_bbox(*selected_box, width, height, margin_percentage)
            cropped = image.crop(crop_box)
            fallback = False
        else:
            crop_box = None
            cropped = center_crop_pil(image)
            fallback = True

        face_tensors.append(transform(cropped))
        records.append(
            {
                "frame_index": frame_index,
                "bbox": [float(value) for value in selected_box] if selected_box is not None else None,
                "prob": selected_prob,
                "crop_box": list(crop_box) if crop_box is not None else None,
                "fallback_center_crop": fallback,
            }
        )
    return torch.stack(face_tensors), records


def load_model(model_path: Path, device: str) -> FaceXFormer:
    model = FaceXFormer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict_backbone", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_mtcnn(args):
    if args.face_crop_mode != "mtcnn":
        return None
    if MTCNN is None:
        raise ImportError(
            "facenet-pytorch is required for --face-crop-mode mtcnn. "
            "Install it or use --face-crop-mode center."
        )
    return MTCNN(keep_all=True, device=args.device)


@torch.no_grad()
def extract_face_tokens(model: FaceXFormer, frames: torch.Tensor, device: str) -> torch.Tensor:
    model.multi_scale_features.clear()
    _ = model.backbone(frames.to(device))
    batch_size = model.multi_scale_features[-1].shape[0]
    all_hidden_states = ()
    for encoder_hidden_state, mlp in zip(model.multi_scale_features, model.linear_c):
        height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
        encoder_hidden_state = mlp(encoder_hidden_state)
        encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
        encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
        encoder_hidden_state = F.interpolate(
            encoder_hidden_state,
            size=model.multi_scale_features[0].size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        all_hidden_states += (encoder_hidden_state,)

    fused_states = model.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
    return fused_states.mean(dim=(2, 3)).cpu()


def save_metadata(output_dir: Path, metadata: dict) -> None:
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"FaceXFormer checkpoint not found: {model_path}. "
            "Download ckpts/model.pt from kartiknarayan/facexformer first."
        )

    sample_ids = read_manifest_video_ids(manifest_path)
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]

    model = load_model(model_path, args.device)
    mtcnn = build_mtcnn(args)
    target_dtype = output_dtype(args.dtype)
    metadata = {
        "model_path": model_path.as_posix(),
        "feature_type": "facexformer_fused_state_mean_pool",
        "feature_dim": 256,
        "fps": args.fps,
        "face_size": args.face_size,
        "dtype": args.dtype,
        "videos": {},
    }
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata["videos"].update(json.loads(metadata_path.read_text(encoding="utf-8")).get("videos", {}))

    failures = []
    for sample_id in tqdm(sample_ids, desc="precompute FaceXFormer"):
        video_path = videos_dir / f"{sample_id}.mp4"
        save_path = output_dir / f"{sample_id}.pt"
        if save_path.exists() and not args.overwrite:
            continue
        if not video_path.exists():
            failures.append({"sample_id": sample_id, "error": "missing_video"})
            continue
        try:
            frames = sample_video(video_path, args.fps, args.resolution)
            face_tokens = []
            face_detection_records = []
            for batch in frames.split(args.batch_size):
                if args.face_crop_mode == "mtcnn":
                    batch, detection_records = prepare_faces_mtcnn(
                        batch,
                        args.face_size,
                        mtcnn=mtcnn,
                        margin_percentage=args.face_margin,
                    )
                    face_detection_records.extend(detection_records)
                else:
                    batch = prepare_faces(batch, args.face_size)
                face_tokens.append(extract_face_tokens(model, batch, args.device))
            face_tokens = torch.cat(face_tokens).to(target_dtype)
            torch.save(face_tokens, save_path)
            metadata["videos"][sample_id] = {
                "path": save_path.as_posix(),
                "video_path": video_path.as_posix(),
                "num_feature_frames": int(face_tokens.shape[0]),
                "feature_shape": list(face_tokens.shape),
                "face_crop_mode": args.face_crop_mode,
                "face_detection_fallbacks": sum(
                    int(record["fallback_center_crop"]) for record in face_detection_records
                ),
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
    try:
        torchvision.set_video_backend("video_reader")
    except Exception:
        pass
    main()
