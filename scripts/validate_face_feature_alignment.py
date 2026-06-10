import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that optional face-token features align with SigLIP features."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--siglip-feature-dir",
        default="data/processed/features/siglip_large_384_2fps_1plus3x3",
    )
    parser.add_argument(
        "--face-feature-dir",
        default="data/processed/features/facexformer_2fps_face_token",
    )
    parser.add_argument("--expected-face-tokens", type=int, default=1)
    parser.add_argument("--expected-siglip-hidden-size", type=int, default=1024)
    parser.add_argument("--expected-face-hidden-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def face_shape(face: torch.Tensor) -> tuple[int, int, int]:
    if face.ndim == 2:
        return int(face.shape[0]), 1, int(face.shape[1])
    if face.ndim == 3:
        return int(face.shape[0]), int(face.shape[1]), int(face.shape[2])
    raise ValueError(f"Expected face tensor shape [T, D] or [T, N, D], got {tuple(face.shape)}")


def validate_record(
    record: dict,
    *,
    siglip_feature_dir: Path,
    face_feature_dir: Path,
    expected_face_tokens: int,
    expected_siglip_hidden_size: int,
    expected_face_hidden_size: int,
) -> dict:
    sample_id = record["sample_id"]
    siglip_path = Path(record.get("feature_path") or siglip_feature_dir / f"{sample_id}.pt")
    face_path = Path(record.get("face_feature_path") or face_feature_dir / f"{sample_id}.pt")
    row = {
        "sample_id": sample_id,
        "siglip_path": siglip_path.as_posix(),
        "face_path": face_path.as_posix(),
        "ok": False,
        "errors": [],
    }

    if not siglip_path.exists():
        row["errors"].append("missing_siglip_feature")
        return row
    if not face_path.exists():
        row["errors"].append("missing_face_feature")
        return row

    siglip = load_tensor(siglip_path)
    face = load_tensor(face_path)
    face_t, face_tokens, face_dim = face_shape(face)

    row.update(
        {
            "siglip_shape": list(siglip.shape),
            "face_shape": list(face.shape),
            "siglip_frames": int(siglip.shape[0]),
            "face_frames": face_t,
            "face_tokens": face_tokens,
            "face_hidden_size": face_dim,
        }
    )

    if siglip.ndim != 3:
        row["errors"].append("siglip_shape_not_T_N_D")
    if siglip.shape[0] != face_t:
        row["errors"].append("frame_count_mismatch")
    if siglip.shape[-1] != expected_siglip_hidden_size:
        row["errors"].append("siglip_hidden_size_mismatch")
    if face_dim != expected_face_hidden_size:
        row["errors"].append("face_hidden_size_mismatch")
    if face_tokens != expected_face_tokens:
        row["errors"].append("face_token_count_mismatch")
    if not torch.isfinite(siglip.float()).all():
        row["errors"].append("siglip_has_nan_or_inf")
    if not torch.isfinite(face.float()).all():
        row["errors"].append("face_has_nan_or_inf")

    row["ok"] = not row["errors"]
    return row


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.manifest))
    if args.limit > 0:
        records = records[: args.limit]

    rows = [
        validate_record(
            record,
            siglip_feature_dir=Path(args.siglip_feature_dir),
            face_feature_dir=Path(args.face_feature_dir),
            expected_face_tokens=args.expected_face_tokens,
            expected_siglip_hidden_size=args.expected_siglip_hidden_size,
            expected_face_hidden_size=args.expected_face_hidden_size,
        )
        for record in tqdm(records, desc="validate face features")
    ]
    ok_count = sum(int(row["ok"]) for row in rows)
    bad_rows = [row for row in rows if not row["ok"]]

    print(f"records={len(rows)}")
    print(f"ok={ok_count}")
    print(f"bad={len(bad_rows)}")
    for row in bad_rows[:20]:
        print(f"{row['sample_id']}: {', '.join(row['errors'])}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
