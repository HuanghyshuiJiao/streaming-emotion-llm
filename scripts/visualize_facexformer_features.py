import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.patches as patches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACEXFORMER_ROOT = PROJECT_ROOT / "reference" / "facexformer"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FACEXFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(FACEXFORMER_ROOT))

from network import FaceXFormer  # noqa: E402
from scripts.precompute_facexformer_features import (  # noqa: E402
    MTCNN,
    prepare_faces,
    prepare_faces_mtcnn,
)
from scripts.precompute_video_features import sample_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FaceXFormer intermediate outputs.")
    parser.add_argument("--video", default="data/raw/videos/vid_0198_clip8.mp4")
    parser.add_argument("--model-path", default="reference/facexformer/ckpts/model.pt")
    parser.add_argument("--output", default="outputs/facexformer_visualizations/vid_0198_clip8_fxf_viz.png")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--face-crop-mode", choices=["center", "mtcnn"], default="mtcnn")
    parser.add_argument("--face-margin", type=float, default=50.0)
    parser.add_argument("--times", type=float, nargs="+", default=[1.5, 26.0, 43.5])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(model_path: Path, device: str) -> FaceXFormer:
    model = FaceXFormer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict_backbone", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def denormalize_face(face: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (face.cpu() * std + mean).clamp(0, 1)


def to_image(frame: torch.Tensor) -> torch.Tensor:
    return (frame.cpu().float() / 255.0).permute(1, 2, 0).clamp(0, 1)


def minmax(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


@torch.no_grad()
def run_facexformer(model: FaceXFormer, faces: torch.Tensor, device: str):
    faces = faces.to(device)
    model.multi_scale_features.clear()
    _ = model.backbone(faces)
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
    image_pe = model.pe_layer((fused_states.shape[2], fused_states.shape[3])).unsqueeze(0)
    landmark, headpose, attributes, visibility, age, gender, race, seg = model.face_decoder(
        image_embeddings=fused_states,
        image_pe=image_pe,
    )
    face_token = fused_states.mean(dim=(2, 3))
    heatmap = fused_states.square().mean(dim=1).sqrt()
    return {
        "seg": seg.softmax(dim=1).argmax(dim=1).cpu(),
        "heatmap": heatmap.cpu(),
        "face_token": face_token.cpu(),
        "landmark": landmark.cpu(),
        "headpose": headpose.cpu(),
    }


def main() -> None:
    args = parse_args()
    try:
        torchvision.set_video_backend("video_reader")
    except Exception:
        pass

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = sample_video(video_path, args.fps, args.resolution)
    frame_indices = [
        min(max(int(round(time * args.fps)), 0), frames.shape[0] - 1)
        for time in args.times
    ]
    selected = frames[frame_indices]
    if args.face_crop_mode == "mtcnn":
        if MTCNN is None:
            raise ImportError("facenet-pytorch is required for --face-crop-mode mtcnn.")
        mtcnn = MTCNN(keep_all=True, device=args.device)
        faces, detection_records = prepare_faces_mtcnn(
            selected,
            args.face_size,
            mtcnn=mtcnn,
            margin_percentage=args.face_margin,
        )
    else:
        faces = prepare_faces(selected, args.face_size)
        detection_records = [
            {
                "bbox": None,
                "crop_box": None,
                "fallback_center_crop": True,
            }
            for _ in frame_indices
        ]

    model = load_model(Path(args.model_path), args.device)
    outputs = run_facexformer(model, faces, args.device)

    fig, axes = plt.subplots(
        len(frame_indices),
        4,
        figsize=(13, 3.3 * len(frame_indices)),
        constrained_layout=True,
    )
    if len(frame_indices) == 1:
        axes = axes[None, :]

    for row, (time, frame_index) in enumerate(zip(args.times, frame_indices)):
        original = to_image(selected[row])
        face_img = denormalize_face(faces[row]).permute(1, 2, 0)
        seg = outputs["seg"][row]
        heatmap = minmax(outputs["heatmap"][row])
        heatmap = F.interpolate(
            heatmap[None, None],
            size=(args.face_size, args.face_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Original frame\n{time:.1f}s / idx {frame_index}")
        bbox = detection_records[row].get("bbox")
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            axes[row, 0].add_patch(
                patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor="#00e5ff",
                    facecolor="none",
                )
            )
        axes[row, 1].imshow(face_img)
        crop_title = "MTCNN face crop" if args.face_crop_mode == "mtcnn" else "Center crop"
        if detection_records[row].get("fallback_center_crop"):
            crop_title += "\nfallback"
        else:
            crop_title += "\nFXF input"
        axes[row, 1].set_title(crop_title)
        axes[row, 2].imshow(face_img)
        axes[row, 2].imshow(seg, alpha=0.45, cmap="tab20")
        axes[row, 2].set_title("Face parsing\nsegmentation")
        axes[row, 3].imshow(face_img)
        axes[row, 3].imshow(heatmap, alpha=0.55, cmap="magma")
        axes[row, 3].set_title("Fused feature\nactivation heatmap")

        for col in range(4):
            axes[row, col].axis("off")

    fig.suptitle(
        "FaceXFormer Visualization: face crop, segmentation, and fused feature heatmap",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=220)
    print(f"wrote={output_path}")
    print(f"sample={video_path}")
    print(f"frames={frame_indices}")
    print(f"face_token_shape={tuple(outputs['face_token'].shape)}")
    print(
        "fallback_center_crop="
        f"{sum(int(record['fallback_center_crop']) for record in detection_records)}/"
        f"{len(detection_records)}"
    )


if __name__ == "__main__":
    main()
