"""Human dataset reader and loss for the four-class configuration."""
from __future__ import annotations
import os
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import albumentations as A
from core import CFG, IMAGE_EXTS

class HumanDataset(Dataset):
    """
    Reader for the human dataset layout:

        <root>/<variant>/Images/{TrainVal,Test}/<id>.png
        <root>/<variant>/Annotations/{TrainVal,Test}/<id>.png

    The dataset only ships TrainVal and Test on disk. We carve a deterministic
    `valid` split off the END of TrainVal (sorted by filename) — last
    `0.15` of files.
    """

    _FOLDER = {"train": "TrainVal", "valid": "TrainVal", "test": "Test"}

    def __init__(self, root: str, split: str = "train",
                 transform: A.Compose | None = None,
                 variant: str = "Original",
                 valid_fraction: float = 0.15) -> None:
        assert split in ("train", "valid", "test"), f"unknown split: {split}"
        assert variant == "Original", "The final human configuration uses Original images"
        self.transform = transform
        self.split     = split

        folder         = self._FOLDER[split]
        self.img_dir   = os.path.join(root, variant, "Images",      folder)
        self.mask_dir  = os.path.join(root, variant, "Annotations", folder)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask  dir not found: {self.mask_dir}")

        all_imgs = sorted(p for p in glob.glob(os.path.join(self.img_dir, "*"))
                          if Path(p).suffix.lower() in IMAGE_EXTS)
        if not all_imgs:
            raise RuntimeError(f"No images in {self.img_dir}")

        if split in ("train", "valid"):
            n_total = len(all_imgs)
            n_valid = max(1, int(round(n_total * valid_fraction)))
            if split == "train":
                self.imgs = all_imgs[:n_total - n_valid]
            else:
                self.imgs = all_imgs[n_total - n_valid:]
        else:
            self.imgs = all_imgs

        if not self.imgs:
            raise RuntimeError(f"Empty split {split} for {self.img_dir}")
        print(f"  [{split:5s}]  {len(self.imgs)} images   ({variant})")

    def _mask_path(self, img_path: str) -> str:
        stem = Path(img_path).stem
        c = os.path.join(self.mask_dir, stem + ".png")
        if os.path.exists(c):
            return c
        cands = glob.glob(os.path.join(self.mask_dir, stem + ".*"))
        if cands:
            return cands[0]
        raise FileNotFoundError(f"No annotation for {img_path} in {self.mask_dir}")

    def __len__(self) -> int: return len(self.imgs)

    def __getitem__(self, idx: int):
        ip = self.imgs[idx]
        mp = self._mask_path(ip)

        img  = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE).astype(np.int32)

        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        mask[(mask >= CFG.NUM_CLASSES) & (mask != CFG.IGNORE_INDEX)] = CFG.IGNORE_INDEX
        mask = mask.astype(np.uint8)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"]

        return img, mask.long(), ip

class DiceLoss(nn.Module):
    def __init__(self, num_classes=CFG.NUM_CLASSES, ignore_index=CFG.IGNORE_INDEX,
                 smooth=1.0) -> None:
        super().__init__()
        self.C = num_classes; self.ig = ignore_index; self.s = smooth

    def forward(self, logits, targets):
        valid   = targets != self.ig
        t_clean = targets.clone(); t_clean[~valid] = 0
        probs   = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        oh = F.one_hot(t_clean, C).permute(0, 3, 1, 2).float()
        v4 = valid.unsqueeze(1).float()
        probs, oh = probs * v4, oh * v4
        d      = (0, 2, 3)
        inter  = (probs * oh).sum(dim=d)
        card   = (probs + oh).sum(dim=d)
        dice_c = (2 * inter + self.s) / (card + self.s)
        pres   = (oh.sum(dim=(0, 2, 3)) > 0)
        return 1.0 - dice_c[pres].mean() if pres.any() else logits.sum() * 0

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = CFG.FOCAL_GAMMA,
                 alpha: list | None = None,
                 ignore_index: int = CFG.IGNORE_INDEX,
                 num_classes: int = CFG.NUM_CLASSES) -> None:
        super().__init__()
        self.gamma = gamma
        self.ig    = ignore_index
        if alpha is None:
            alpha = [1.0] * num_classes
        assert len(alpha) == num_classes
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        valid     = targets != self.ig
        t_clean   = targets.clone(); t_clean[~valid] = 0
        log_pt = log_probs.gather(1, t_clean.unsqueeze(1)).squeeze(1)
        pt     = log_pt.exp()
        focal  = (1.0 - pt).clamp(min=0).pow(self.gamma)
        alpha_w = self.alpha.to(logits.device)[t_clean]
        loss   = -alpha_w * focal * log_pt
        loss   = loss * valid.float()
        denom  = valid.float().sum().clamp(min=1.0)
        return loss.sum() / denom

class HumanLoss(nn.Module):
    def __init__(self, num_classes=CFG.NUM_CLASSES, alpha=CFG.CE_WEIGHT,
                 beta=CFG.DICE_WEIGHT, gamma=CFG.FOCAL_WEIGHT,
                 ignore_index=CFG.IGNORE_INDEX) -> None:
        super().__init__()
        self.a = alpha
        self.b = beta
        self.c = gamma

        if num_classes == 5:
            weights_array = [0.5, 4.0, 1.0, 2.0, 3.0]
        elif num_classes == 4:
            weights_array = [0.5, 2.0, 2.0, 2.0]
        else:
            weights_array = [1.0] * num_classes
        assert len(weights_array) == num_classes, \
            f"class weights ({len(weights_array)}) must equal num_classes ({num_classes})"
        class_weights = torch.tensor(weights_array, dtype=torch.float32)

        self.ce  = nn.CrossEntropyLoss(weight=class_weights,
                                       ignore_index=ignore_index,
                                       reduction="mean",
                                       label_smoothing=0.05)
        self.dc  = DiceLoss(num_classes, ignore_index)
        self.fl  = FocalLoss(gamma=CFG.FOCAL_GAMMA, alpha=None,
                             ignore_index=ignore_index,
                             num_classes=num_classes)

    def forward(self, logits, targets):
        ce_loss    = self.ce(logits, targets)
        dice_loss  = self.dc(logits, targets)
        focal_loss = self.fl(logits, targets)

        total_loss = self.a * ce_loss + self.b * dice_loss + self.c * focal_loss

        return total_loss, {
            "ce":    ce_loss.item(),
            "dice":  dice_loss.item(),
            "focal": focal_loss.item(),
        }
