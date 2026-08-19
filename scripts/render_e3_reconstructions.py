from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def tensor_image(value: torch.Tensor, scale: int) -> Image.Image:
    """Convert a unit-range single-channel tensor to a nearest-neighbour PNG image."""
    pixels = value.detach().cpu().squeeze().clamp(0.0, 1.0).numpy()
    pixels = np.rint(pixels * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def render_pair(source: Path, destination: Path, scale: int) -> None:
    """Render one saved reference/reconstruction pair with its recorded metrics."""
    payload = torch.load(source, map_location="cpu", weights_only=False)
    reference = tensor_image(payload["reference"], scale)
    reconstruction = tensor_image(payload["reconstruction"], scale)
    row = payload["row"]
    gap, header, footer = 16, 38, 48
    canvas = Image.new(
        "RGB",
        (reference.width + reconstruction.width + gap, header + reference.height + footer),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 4), "Hidden reference", fill="black")
    draw.text((reference.width + gap, 4), "Gradient reconstruction", fill="black")
    canvas.paste(reference.convert("RGB"), (0, header))
    canvas.paste(reconstruction.convert("RGB"), (reference.width + gap, header))
    summary = (
        f"true={row['true_target_label']} inferred={row['inferred_label']}  "
        f"MSE={row['mse']:.4f}  PSNR={row['psnr']:.2f}  SSIM={row['ssim']:.3f}"
    )
    draw.text((0, header + reference.height + 12), summary, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render saved E3 tensor pairs as PNGs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results" / "e3" / "reconstructions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "e3" / "images",
    )
    parser.add_argument("--scale", type=int, default=8)
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("scale must be positive")
    sources = sorted(args.input_dir.glob("*.pt"))
    if not sources:
        parser.error(f"no reconstruction tensors found in {args.input_dir}")
    for source in sources:
        destination = args.output_dir / f"{source.stem}.png"
        render_pair(source, destination, args.scale)
        print(destination.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
