"""Predict native-resolution tissue labels from images without reference masks."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import historical
from final_b import ROOT, load_model, save_json, sha256
from image_preprocessing import preprocess_image, restore_logits
from tiling import fuse_global_tiles


@torch.inference_mode()
def predict_labels(model, rgb, device, mode="global-tiles"):
    tensor, metadata = preprocess_image(rgb, 256)
    logits = restore_logits(model(tensor.unsqueeze(0).to(device))[0], metadata).cpu()
    if mode in ("global-tiles", "area"):
        logits = fuse_global_tiles(
            model, rgb, logits, device, image_size=512, model_input_size=256,
            tile_overlap=0.25, global_weight=0.5, batch_size=2, amp=False)
    return logits.argmax(0).numpy().astype(np.uint8)


def predict_area_labels(model, rgb, device):
    """Align the image using the predicted foreground orientation."""
    labels = predict_labels(model, rgb, device, "area")
    ys, xs = np.where(labels > 0)
    rotate = bool(xs.size and int(ys.max() - ys.min()) > int(xs.max() - xs.min()))
    if rotate:
        rotated = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        labels = predict_labels(model, rotated, device, "area")
        labels = cv2.rotate(labels, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return labels, 90 if rotate else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/seed_1005.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("global", "global-tiles", "area"), default="area")
    parser.add_argument("--ruler-length-px", type=float)
    parser.add_argument("--ruler-length-mm", type=float, default=30.0)
    args = parser.parse_args()
    if args.ruler_length_px is not None and (
        not math.isfinite(args.ruler_length_px) or args.ruler_length_px <= 0
        or not math.isfinite(args.ruler_length_mm) or args.ruler_length_mm <= 0
    ):
        parser.error("Ruler lengths must be finite and greater than zero")
    torch.set_num_threads(4)
    historical.set_seed(1005)
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error(f"Cannot read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    model, metadata = load_model(args.checkpoint, args.device)
    if args.mode == "area":
        labels, rotation = predict_area_labels(model, rgb, args.device)
    else:
        labels = predict_labels(model, rgb, args.device, args.mode)
        rotation = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(args.output)
    counts = {name: int((labels == index).sum())
              for index, name in enumerate(historical.CFG.CLASSES)}
    report = dict(
        input_image=args.image.name, image_sha256=sha256(args.image),
        checkpoint_sha256=sha256(args.checkpoint), **metadata, mode=args.mode,
        mask_sha256=sha256(args.output), native_hw=list(labels.shape),
        reference_mask_used=False, class_pixel_counts=counts,
        wound_pixels=int((labels > 0).sum()), orientation_rotation_degrees=rotation)
    if args.ruler_length_px is not None:
        report.update(ruler_length_px=args.ruler_length_px,
                      ruler_length_mm=args.ruler_length_mm,
                      wound_area_mm2=report["wound_pixels"] * (
                          args.ruler_length_mm / args.ruler_length_px) ** 2)
    save_json(args.output.with_suffix(".json"), report)
    print(args.output)


if __name__ == "__main__":
    main()
