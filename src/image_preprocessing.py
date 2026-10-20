"""Image-only letterboxing and restoration used for the final-B area analysis."""
from __future__ import annotations
import numpy as np
from PIL import Image
NORMALIZE_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
NORMALIZE_STD = np.array([0.229,0.224,0.225], dtype=np.float32)

def _letterbox(rgb, size):
    if not isinstance(size, int) or size < 1:
        raise ValueError("image_size must be a positive integer")
    array = np.asarray(rgb.convert("RGB") if isinstance(rgb, Image.Image) else rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("Expected a uint8 RGB HxWx3 image or PIL image")
    h, w = array.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("Cannot preprocess an empty image")
    scale = min(size / h, size / w)
    rh, rw = max(1, min(size, round(h * scale))), max(1, min(size, round(w * scale)))
    resized = np.asarray(Image.fromarray(array).resize((rw, rh), Image.Resampling.BILINEAR))
    top, left = (size - rh) // 2, (size - rw) // 2
    pad = ((top, size - rh - top), (left, size - rw - left), (0, 0))
    padded = np.pad(resized, pad, mode="reflect" if min(rh, rw) > 1 else "edge")
    meta = {"original_hw": [h, w], "resized_hw": [rh, rw], "pad_yx": [top, left], "input_hw": [size, size]}
    return padded, meta


def _to_tensor(rgb):
    import torch
    normalized = (rgb.astype(np.float32) / 255.0 - NORMALIZE_MEAN) / NORMALIZE_STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def preprocess_image(rgb, size=256):
    """Image-only preprocessing for validation and deployment, returning CHW/meta."""
    padded, meta = _letterbox(rgb, size)
    return _to_tensor(padded), meta


def restore_logits(logits, meta):
    """Remove letterbox padding and bilinearly restore CHW or BCHW logits."""
    import torch.nn.functional as functional
    if logits.ndim not in (3, 4):
        raise ValueError("logits must have CHW or BCHW dimensions")
    unbatched = logits.ndim == 3
    values = logits.unsqueeze(0) if unbatched else logits
    input_hw = tuple(meta["input_hw"])
    if tuple(values.shape[-2:]) != input_hw:
        values = functional.interpolate(values.float(), size=input_hw, mode="bilinear", align_corners=False)
    top, left = meta["pad_yx"]
    rh, rw = meta["resized_hw"]
    values = values[..., top:top + rh, left:left + rw]
    values = functional.interpolate(values.float(), size=tuple(meta["original_hw"]), mode="bilinear", align_corners=False)
    return values.squeeze(0) if unbatched else values


