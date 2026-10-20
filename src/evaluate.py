"""Evaluate final B on the reference-mask-defined ROI benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import historical
from final_b import ROOT, load_model, save_json, sha256, verify_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/internal")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/seed_1005.pth")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    historical.set_seed(1005)
    manifest = verify_dataset(args.dataset)
    dataset = historical.WoundDataset(
        str(args.dataset), args.split, historical._val_tf(256))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    model, checkpoint_metadata = load_model(args.checkpoint, args.device)
    meter = historical.SegMetrics()
    per_image = []
    with torch.inference_mode():
        for images, labels, paths in loader:
            prediction = model(images.to(args.device)).argmax(1).cpu()
            meter.update(prediction, labels)
            for pred, target, path in zip(prediction, labels, paths):
                individual = historical.SegMetrics()
                individual.update(pred, target)
                per_image.append(dict(
                    image=str(Path(path).resolve().relative_to(args.dataset.resolve())),
                    confusion_matrix=individual._cm.tolist(), **individual.compute()))
            print(f"Evaluated {len(per_image)}/{len(dataset)}", flush=True)
    save_json(args.output, dict(
        metric_domain="historical_gt_roi_256", split=args.split,
        reference_mask_used_for_framing=True, images=len(dataset),
        dataset_fingerprint=manifest["fingerprint"],
        checkpoint_sha256=sha256(args.checkpoint), **checkpoint_metadata,
        confusion_matrix=meter._cm.tolist(), **meter.compute(), per_image=per_image))
    print(meter.log_str(split=args.split))
    print(args.output)


if __name__ == "__main__":
    main()
