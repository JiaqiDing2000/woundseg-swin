"""Final-B configuration, checkpoint loading, loss and balanced sampling."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import historical

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "final_b.json"
MODEL = "dasp_ag_swinunet"


def read_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify_dataset(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    expected = read_config()["dataset_fingerprint"]
    if manifest["fingerprint"] != expected:
        raise ValueError("Dataset fingerprint differs from the released split")
    for split, count in {"train": 286, "valid": 30, "test": 35}.items():
        if len(manifest["splits"][split]) != count:
            raise ValueError(f"Unexpected number of images in {split}")
        for row in manifest["splits"][split]:
            for key in ("image", "mask"):
                path = (root / row[key]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"Dataset path outside root: {row[key]}")
                if sha256(path) != row[key + "_sha256"]:
                    raise ValueError(f"Dataset checksum mismatch: {row[key]}")
    return manifest


class CETversky(nn.Module):
    """Class-weighted CE and batch-averaged, five-class Tversky loss."""

    def __init__(self, config=None):
        super().__init__()
        config = config or read_config()
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(config["ce_weights"]), ignore_index=255,
            label_smoothing=config["label_smoothing"])
        self.register_buffer("fp_weights", torch.tensor(config["tversky_fp_weights"]))
        self.register_buffer("fn_weights", torch.tensor(config["tversky_fn_weights"]))

    def forward(self, logits, targets):
        valid = targets != 255
        if not valid.any():
            return logits.sum() * 0.0, dict(ce=0.0, dice=0.0, focal=0.0, tversky=0.0)
        clean = targets.masked_fill(~valid, 0)
        mask = valid.unsqueeze(1).to(logits.dtype)
        probs = logits.softmax(1) * mask
        one_hot = F.one_hot(clean, num_classes=5).permute(0, 3, 1, 2).to(logits.dtype) * mask
        axes = (0, 2, 3)
        tp = (probs * one_hot).sum(axes)
        fp = (probs * (1 - one_hot)).sum(axes)
        fn = ((1 - probs) * one_hot).sum(axes)
        tv = 1 - ((tp + 1.0) / (
            tp + self.fp_weights * fp + self.fn_weights * fn + 1.0)).mean()
        ce = self.ce(logits, targets)
        # Zero-valued slots preserve the training logger's metric schema.
        return 0.5 * ce + 0.5 * tv, dict(
            ce=float(ce.detach()), dice=0.0, focal=0.0, tversky=float(tv.detach()))


def build_model(pretrained=True):
    model = historical.DASP_AG_SwinUNet(pretrained=pretrained)
    for branch, old_rate, rate in zip(("c2", "c3", "c4"), (6, 12, 18), (1, 2, 3)):
        conv = getattr(model.aspp, branch)[0]
        if conv.dilation != (old_rate, old_rate) or conv.padding != (old_rate, old_rate):
            raise ValueError(f"Unexpected ASPP branch: {branch}")
        conv.dilation = (rate, rate)
        conv.padding = (rate, rate)
    return model


def load_model(checkpoint, device):
    model = build_model(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"], strict=True)
    metadata = {key: state[key] for key in ("epoch", "best_miou")}
    del state
    return model.to(device).eval(), metadata


def sampling_audit(root):
    """Compute probabilities from the unaugmented 256-pixel training ROIs."""
    dataset = historical.WoundDataset(str(root), "train", historical._val_tf(256))
    counts = []
    for index in range(len(dataset)):
        _, labels, _ = dataset[index]
        values = labels.numpy()
        counts.append(np.bincount(values[values != 255].reshape(-1), minlength=5))
    counts = np.asarray(counts, dtype=np.int64)
    probabilities = np.full(len(counts), 0.4 / len(counts), dtype=np.float64)
    pools = {}
    threshold = math.ceil(0.005 * 256 * 256)
    for class_id, name in ((2, "Granulation"), (3, "Necrotic"), (4, "Pus")):
        eligible = np.flatnonzero(counts[:, class_id] >= threshold)
        fallback = not len(eligible)
        if fallback:
            eligible = np.flatnonzero(counts[:, class_id] > 0)
        if not len(eligible):
            raise ValueError(f"Training set contains no {name} images")
        probabilities[eligible] += 0.2 / len(eligible)
        pools[name] = dict(class_id=class_id, indices=eligible.tolist(),
                           fallback_to_any_positive=bool(fallback))
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise ValueError("Sampling probabilities do not sum to one")
    return dict(
        threshold_pixels=threshold, pools=pools,
        samples=[dict(image=str(Path(image).resolve().relative_to(Path(root).resolve())),
                      roi_class_pixels=count.tolist(), sampling_probability=float(probability))
                 for image, count, probability in zip(dataset.imgs, counts, probabilities)])


def configure_training(root, output, audit):
    root, output = Path(root).resolve(), Path(output).resolve()
    historical.CFG.BASE_DIR = str(output)
    historical.CFG.DATASET_DIR = str(root)
    for attribute, name in (("CHECKPOINT_DIR", "checkpoints"), ("LOG_DIR", "logs"),
                            ("RESULTS_DIR", "results"), ("FIGURES_DIR", "figures")):
        setattr(historical.CFG, attribute, str(output / name))
    historical.CombinedLoss = CETversky
    historical.build_model = lambda name, num_classes=5: build_model(pretrained=True)
    original_loaders = historical.build_loaders
    probabilities = {
        str((root / row["image"]).resolve()): row["sampling_probability"]
        for row in audit["samples"]
    }

    def balanced_loaders(dataset_root, batch_size, image_size, num_workers=8, seed=42):
        train, valid, test = original_loaders(
            dataset_root, batch_size, image_size, num_workers, seed)
        paths = [str(Path(path).resolve()) for path in train.dataset.imgs]
        if set(paths) != set(probabilities):
            raise ValueError("Training membership differs from the sampling audit")
        sampler = WeightedRandomSampler(
            [probabilities[path] for path in paths], len(paths), replacement=True,
            generator=torch.Generator().manual_seed(seed + 200000))
        train = DataLoader(
            train.dataset, batch_size=batch_size, sampler=sampler, drop_last=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
            worker_init_fn=historical._seed_worker, generator=train.generator)
        return train, valid, test

    historical.build_loaders = balanced_loaders
    return historical
