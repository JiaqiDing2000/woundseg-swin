"""Evaluate a final-model checkpoint with its dataset-specific preprocessing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import core
from model import ROOT, load_model, save_json, sha256, verify_dataset, read_config, configure_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rat.json")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the dataset configuration evaluation batch size")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    core.set_seed(1005)
    config = read_config(args.config)
    configure_dataset(config)
    manifest = verify_dataset(args.dataset, config)
    dataset = core.WoundDataset(
        str(args.dataset), args.split, core._val_tf(256))
    loader = DataLoader(dataset, batch_size=args.batch_size or config["evaluation_batch_size"], shuffle=False,
                        num_workers=args.workers)
    model, checkpoint_metadata = load_model(args.checkpoint, args.device, config)
    meter = core.SegMetrics()
    per_image = []
    with torch.inference_mode():
        for images, labels, paths in loader:
            prediction = model(images.to(args.device)).argmax(1).cpu()
            meter.update(prediction, labels)
            for pred, target, path in zip(prediction, labels, paths):
                individual = core.SegMetrics()
                individual.update(pred, target)
                per_image.append(dict(
                    image=str(Path(path).resolve().relative_to(args.dataset.resolve())),
                    confusion_matrix=individual._cm.tolist(), **individual.compute()))
            print(f"Evaluated {len(per_image)}/{len(dataset)}", flush=True)
    save_json(args.output, dict(
        metric_domain=config["metric_domain"], split=args.split,
        dataset=config["dataset"], evaluation_context=manifest.get("evaluation_context", {}),
        reference_mask_used_for_framing=(config["dataset"] == "rat"), images=len(dataset),
        dataset_fingerprint=manifest.get("fingerprint"),
        checkpoint_sha256=sha256(args.checkpoint), **checkpoint_metadata,
        confusion_matrix=meter._cm.tolist(), **meter.compute(), per_image=per_image))
    print(meter.log_str(split=args.split))
    print(args.output)


if __name__ == "__main__":
    main()
