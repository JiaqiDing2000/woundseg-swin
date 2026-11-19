"""Dataset configuration, checkpoint loading, loss and balanced sampling."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import core

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rat.json"
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


def verify_dataset(root, config=None):
    config = config or read_config()
    root = Path(root).resolve()
    manifest_path = ROOT / "datasets" / config["dataset"] / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = config.get("dataset_fingerprint")
    if expected and manifest.get("fingerprint") != expected:
        raise ValueError("Dataset fingerprint differs from the released split")
    for split, count in config["split_counts"].items():
        rows = manifest["splits"][split]
        if len(rows) != count:
            raise ValueError(f"Unexpected number of images in {split}")
        for row in rows:
            for key in ("image", "mask"):
                path = (root / row[key]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"Dataset path outside root: {row[key]}")
                if sha256(path) != row[key + "_sha256"]:
                    raise ValueError(f"Dataset checksum mismatch: {row[key]}")
        dataset_class = RatDataset if config["dataset"] == "rat" else HumanDataset
        kwargs = {} if config["dataset"] == "rat" else dict(variant=config["dataset_variant"], valid_fraction=config["validation_fraction"])
        actual = dataset_class(str(root), split, **kwargs)
        expected_paths = {str((root / row["image"]).resolve()) for row in rows}
        if {str(Path(path).resolve()) for path in actual.imgs} != expected_paths:
            raise ValueError(f"Dataset image membership differs from manifest: {split}")
        for image in actual.imgs:
            relative = str(Path(image).resolve().relative_to(root))
            expected_mask = next(row["mask"] for row in rows if row["image"] == relative)
            if Path(actual._mask_path(image)).resolve() != (root / expected_mask).resolve():
                raise ValueError(f"Mask pairing differs from manifest: {relative}")
    return manifest


class CETversky(nn.Module):
    """Class-weighted CE and batch-averaged, five-class Tversky loss."""

    def __init__(self, config=None):
        super().__init__()
        config = config or read_config()
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(config["ce_weights"]), ignore_index=255,
            label_smoothing=config["label_smoothing"])
        self.loss_weights = config["loss_weights"]
        self.smoothing = config["tversky_smoothing"]
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
        tv = 1 - ((tp + self.smoothing) / (
            tp + self.fp_weights * fp + self.fn_weights * fn + self.smoothing)).mean()
        ce = self.ce(logits, targets)
        # Zero-valued slots preserve the training logger's metric schema.
        return self.loss_weights["ce"] * ce + self.loss_weights["tversky"] * tv, dict(
            ce=float(ce.detach()), dice=0.0, focal=0.0, tversky=float(tv.detach()))


def build_model(pretrained=True, config=None):
    config = config or read_config()
    model = core.DASP_AG_SwinUNet(
        pretrained=pretrained, num_classes=len(config["class_names"]),
        backbone=config["backbone"], img_size=config["image_size"])
    for branch, old_rate, rate in zip(("c2", "c3", "c4"), (6, 12, 18), config["aspp_rates"]):
        conv = getattr(model.aspp, branch)[0]
        if conv.dilation != (old_rate, old_rate) or conv.padding != (old_rate, old_rate):
            raise ValueError(f"Unexpected ASPP branch: {branch}")
        conv.dilation = (rate, rate)
        conv.padding = (rate, rate)
    return model


def load_model(checkpoint, device, config=None):
    config = config or read_config()
    model = build_model(pretrained=False, config=config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("dataset", config["dataset"]) != config["dataset"]:
        raise ValueError("Checkpoint and dataset configuration do not match")
    if state.get("aspp_rates", config["aspp_rates"]) != config["aspp_rates"]:
        raise ValueError("Checkpoint and ASPP configuration do not match")
    model.load_state_dict(state["model_state"], strict=True)
    metadata = {key: state[key] for key in ("epoch", "best_miou", "seed") if key in state}
    del state
    return model.to(device).eval(), metadata


from human_data import HumanDataset, HumanLoss
RatDataset = core.WoundDataset
_original_loaders = core.build_loaders


def configure_dataset(config):
    core.CFG.CLASSES = list(config["class_names"])
    core.CFG.NUM_CLASSES = len(config["class_names"])
    core.WoundDataset = RatDataset if config["dataset"] == "rat" else partial(
        HumanDataset, variant=config["dataset_variant"], valid_fraction=config["validation_fraction"])


def sampling_audit(root, config=None):
    """Compute probabilities from the unaugmented 256-pixel training ROIs."""
    config = config or read_config()
    dataset = core.WoundDataset(str(root), "train", core._val_tf(config["image_size"]))
    counts = []
    for index in range(len(dataset)):
        _, labels, _ = dataset[index]
        values = labels.numpy()
        counts.append(np.bincount(values[values != 255].reshape(-1), minlength=5))
    counts = np.asarray(counts, dtype=np.int64)
    probabilities = np.full(len(counts), config["sampling_mixture"]["uniform"] / len(counts), dtype=np.float64)
    pools = {}
    threshold = config["sampling_minimum_roi_pixels"]
    for class_id, name in ((2, "Granulation"), (3, "Necrotic"), (4, "Pus")):
        eligible = np.flatnonzero(counts[:, class_id] >= threshold)
        fallback = not len(eligible)
        if fallback:
            eligible = np.flatnonzero(counts[:, class_id] > 0)
        if not len(eligible):
            raise ValueError(f"Training set contains no {name} images")
        probabilities[eligible] += config["sampling_mixture"][name] / len(eligible)
        pools[name] = dict(class_id=class_id, indices=eligible.tolist(),
                           fallback_to_any_positive=bool(fallback))
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise ValueError("Sampling probabilities do not sum to one")
    return dict(
        threshold_pixels=threshold, pools=pools,
        samples=[dict(image=str(Path(image).resolve().relative_to(Path(root).resolve())),
                      roi_class_pixels=count.tolist(), sampling_probability=float(probability))
                 for image, count, probability in zip(dataset.imgs, counts, probabilities)])


def configure_training(root, output, audit, config=None):
    config = config or read_config()
    configure_dataset(config)
    for field, key in {
        "IMAGE_SIZE": "image_size", "BATCH_SIZE": "batch_size",
        "NUM_EPOCHS": "maximum_epochs", "LEARNING_RATE": "learning_rate",
        "WEIGHT_DECAY": "weight_decay", "WARMUP_EPOCHS": "warmup_epochs",
        "WARMUP_START_FACTOR": "warmup_start_factor", "COSINE_ETA_MIN": "cosine_eta_min",
        "GRAD_ACCUMULATION_STEPS": "gradient_accumulation_steps",
        "CLIP_GRAD_NORM": "gradient_clip_norm", "EARLY_STOPPING_PATIENCE": "early_stopping_patience"
    }.items():
        setattr(core.CFG, field, config[key])
    core.CFG.FOCAL_GAMMA = config.get("focal_gamma", 2.0)
    root, output = Path(root).resolve(), Path(output).resolve()
    core.CFG.BASE_DIR = str(output)
    core.CFG.DATASET_DIR = str(root)
    for attribute, name in (("CHECKPOINT_DIR", "checkpoints"), ("LOG_DIR", "logs"),
                            ("RESULTS_DIR", "results"), ("FIGURES_DIR", "figures")):
        setattr(core.CFG, attribute, str(output / name))
    def human_loss():
        weights = config["loss_weights"]
        loss = HumanLoss(num_classes=len(config["class_names"]),
                         alpha=weights["ce"], beta=weights["dice"], gamma=weights["focal"])
        loss.ce = nn.CrossEntropyLoss(weight=torch.tensor(config["ce_weights"]),
                                      ignore_index=255, label_smoothing=config["label_smoothing"])
        return loss
    core.CombinedLoss = (lambda: CETversky(config)) if config["dataset"] == "rat" else human_loss
    core.build_model = lambda name, num_classes=None: build_model(pretrained=True, config=config)
    original_loaders = _original_loaders
    core.build_loaders = original_loaders
    if config["dataset"] == "human":
        return core
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
            [probabilities[path] for path in paths], config["sampling_draws_per_epoch"],
            replacement=config["sampling_with_replacement"],
            generator=torch.Generator().manual_seed(seed + config["sampler_seed_offset"]))
        train = DataLoader(
            train.dataset, batch_size=batch_size, sampler=sampler, drop_last=config["drop_last"],
            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
            worker_init_fn=core._seed_worker, generator=train.generator)
        return train, valid, test

    core.build_loaders = balanced_loaders
    return core
