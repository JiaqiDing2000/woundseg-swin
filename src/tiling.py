"""Combine global and tiled image predictions in probability space."""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn.functional as functional

from image_preprocessing import _to_tensor


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _fraction(value, name, include_one):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0
            or (value > 1 if include_one else value >= 1)):
        upper = "1]" if include_one else "1)"
        raise ValueError(f"{name} must be a finite number in [0, {upper}")
    return float(value)


def tile_grid(height, width, image_size=512, tile_overlap=0.25):
    """Return native-pixel (top, left, height, width) tiles covering every pixel."""
    height = _positive_integer(height, "height")
    width = _positive_integer(width, "width")
    size = _positive_integer(image_size, "image_size")
    overlap = _fraction(tile_overlap, "tile_overlap", include_one=False)
    stride = max(1, int(round(size * (1.0 - overlap))))

    def starts(length):
        last = max(0, length - size)
        positions = list(range(0, last + 1, stride))
        if positions[-1] != last:
            positions.append(last)
        return positions

    return [(top, left, min(size, height - top), min(size, width - left))
            for top in starts(height) for left in starts(width)]


def _prepare_tile(rgb, box, image_size):
    top, left, height, width = box
    tile = rgb[top:top + height, left:left + width]
    pad_top, pad_left = (image_size - height) // 2, (image_size - width) // 2
    if height != image_size or width != image_size:
        tile = np.pad(tile, ((pad_top, image_size - height - pad_top),
                            (pad_left, image_size - width - pad_left), (0, 0)),
                      mode="reflect" if min(height, width) > 1 else "edge")
    return _to_tensor(tile), (pad_top, pad_left)


@torch.inference_mode()
def fuse_global_tiles(model, rgb_uint8, global_logits_native, device,
                      image_size=512, tile_overlap=0.25, global_weight=0.5,
                      batch_size=2, amp=False, model_input_size=None):
    """Return native-size CHW log-probabilities on CPU.

    Fuse ``global_weight * softmax(global) + (1-global_weight) * mean(tile_probs)``.
    ``image_size`` sets the native tile width; ``model_input_size`` sets its
    inference resolution. Global logits must match the image dimensions.
    """
    if model.training:
        raise ValueError("fuse_global_tiles requires model.eval()")
    if (not isinstance(rgb_uint8, np.ndarray) or rgb_uint8.dtype != np.uint8
            or rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3):
        raise ValueError("rgb_uint8 must be a uint8 HxWx3 RGB array")
    height, width = rgb_uint8.shape[:2]
    size = _positive_integer(image_size, "image_size")
    input_size = size if model_input_size is None else _positive_integer(model_input_size, "model_input_size")
    batch_size = _positive_integer(batch_size, "batch_size")
    weight = _fraction(global_weight, "global_weight", include_one=True)
    boxes = tile_grid(height, width, size, tile_overlap)
    if (not torch.is_tensor(global_logits_native) or global_logits_native.ndim != 3
            or tuple(global_logits_native.shape[-2:]) != (height, width)
            or global_logits_native.shape[0] < 1):
        raise ValueError("global_logits_native must be CHW with native RGB dimensions")
    global_cpu = global_logits_native.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(global_cpu).all():
        raise ValueError("global_logits_native contains non-finite values")
    if weight == 1.0:
        return torch.log_softmax(global_cpu, dim=0)

    device = torch.device(device)
    classes = global_cpu.shape[0]
    totals = torch.zeros((classes, height, width), dtype=torch.float32)
    counts = torch.zeros((height, width), dtype=torch.float32)
    for start in range(0, len(boxes), batch_size):
        batch_boxes = boxes[start:start + batch_size]
        prepared = [_prepare_tile(rgb_uint8, box, size) for box in batch_boxes]
        inputs = torch.stack([item[0] for item in prepared]).to(device)
        if input_size != size:
            inputs = functional.interpolate(inputs, size=(input_size, input_size),
                                            mode="bilinear", align_corners=False, antialias=True)
        with torch.autocast(device_type=device.type,
                            enabled=bool(amp) and device.type == "cuda"):
            logits = model(inputs)
        if (not torch.is_tensor(logits) or logits.ndim != 4
                or logits.shape[0] != len(batch_boxes) or logits.shape[1] != classes):
            raise ValueError("Tile model output must be BCHW with the global class count")
        logits = logits.float()
        if tuple(logits.shape[-2:]) != (size, size):
            logits = functional.interpolate(logits, size=(size, size), mode="bilinear",
                                            align_corners=False)
        probabilities = torch.softmax(logits, dim=1).cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("Tile model output contains non-finite probabilities")
        for index, (top, left, tile_height, tile_width) in enumerate(batch_boxes):
            pad_top, pad_left = prepared[index][1]
            native = probabilities[index, :, pad_top:pad_top + tile_height,
                                   pad_left:pad_left + tile_width]
            totals[:, top:top + tile_height, left:left + tile_width].add_(native)
            counts[top:top + tile_height, left:left + tile_width].add_(1.0)

    if torch.any(counts == 0):
        raise RuntimeError("Internal error: native tile grid did not cover every pixel")
    totals.div_(counts.unsqueeze(0)).mul_(1.0 - weight)
    if weight > 0:
        totals.add_(torch.softmax(global_cpu, dim=0), alpha=weight)
    return totals.clamp_min_(torch.finfo(torch.float32).tiny).log_()
