"""DASP-AG-SwinUNet model and training functions."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")
import argparse
import csv
import datetime
import glob
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

EPS = 1e-7
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_RGB = [(64,64,64), (80,160,220), (220,80,80), (50,180,80), (255,200,50)]

class CFG:
    BASE_DIR       = str(Path(__file__).resolve().parents[1] / "outputs")
    DATASET_DIR    = str(Path(__file__).resolve().parents[1] / "datasets" / "internal")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR        = os.path.join(BASE_DIR, "logs")
    RESULTS_DIR    = os.path.join(BASE_DIR, "results")
    FIGURES_DIR    = os.path.join(BASE_DIR, "figures")

    CLASSES = [
        "Background",
        "Epithelial",
        "Granulation",
        "Necrotic",
        "Pus",
    ]

    NUM_CLASSES  = len(CLASSES)
    IGNORE_INDEX = 255

    IMAGE_SIZE              = 256
    BATCH_SIZE              = 8
    NUM_EPOCHS              = 100
    LEARNING_RATE           = 1e-4
    WEIGHT_DECAY            = 1e-4
    WARMUP_EPOCHS           = 5
    GRAD_ACCUMULATION_STEPS = 2
    CLIP_GRAD_NORM          = 5.0
    CE_WEIGHT               = 0.5
    DICE_WEIGHT             = 0.5
    FOCAL_WEIGHT            = 0.5
    FOCAL_GAMMA             = 2.0
    EARLY_STOPPING_PATIENCE = 20
    USE_AMP                 = True   # Overridden by FP32 training in phase_train.

    # ImageNet normalisation
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    MODELS = [
        "unet",
        "unetplusplus",
        "deeplabv3plus",
        "fpn",
        "pspnet",
        "manet",
        "pan",
        "segformer",
        "dasp_ag_swinunet",
    ]

    MODEL_CONFIGS = {
        "unet":             {"arch": "Unet",          "encoder_name": "resnet50",       "encoder_weights": "imagenet"},
        "unetplusplus":     {"arch": "UnetPlusPlus",  "encoder_name": "efficientnet-b4","encoder_weights": "imagenet"},
        "deeplabv3plus":    {"arch": "DeepLabV3Plus", "encoder_name": "resnet101",      "encoder_weights": "imagenet"},
        "fpn":              {"arch": "FPN",           "encoder_name": "resnet50",       "encoder_weights": "imagenet"},
        "pspnet":           {"arch": "PSPNet",        "encoder_name": "resnet50",       "encoder_weights": "imagenet"},
        "manet":            {"arch": "MAnet",         "encoder_name": "resnet50",       "encoder_weights": "imagenet"},
        "pan":              {"arch": "PAN",           "encoder_name": "resnet34",       "encoder_weights": "imagenet"},
        "segformer":        {"pretrained_model_name": "nvidia/mit-b2"},
        "dasp_ag_swinunet": {"backbone": "swinv2_small_window8_256"},
    }

    HEAVY_MODELS = {"segformer", "deeplabv3plus", "unetplusplus", "dasp_ag_swinunet"}

    SEEDS = [1001, 1002, 1003, 1004, 1005, 1006]

    @classmethod
    def make_dirs(cls) -> None:
        for d in [cls.DATASET_DIR, cls.CHECKPOINT_DIR, cls.LOG_DIR,
                  cls.RESULTS_DIR, cls.FIGURES_DIR]:
            os.makedirs(d, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def get_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler();  ch.setFormatter(fmt); logger.addHandler(ch)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"))
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


class CSVLogger:
    def __init__(self, path: str) -> None:
        self.path = path; self._init = False
    def log(self, row: dict) -> None:
        need_header = not os.path.exists(self.path) or not self._init
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if need_header: w.writeheader()
            w.writerow(row)
        self._init = True


class EarlyStopping:
    def __init__(self, patience: int = CFG.EARLY_STOPPING_PATIENCE) -> None:
        self.patience = patience; self.best = None; self.counter = 0
    def step(self, score: float) -> bool:
        if self.best is None or score > self.best:
            self.best = score; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


class Timer:
    def __init__(self) -> None: self._t = time.time()
    def elapsed(self) -> str:
        s = int(time.time() - self._t); h, r = divmod(s, 3600); m, s = divmod(r, 60)
        return f"{h:02d}h {m:02d}m {s:02d}s"


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


def build_scheduler(opt, warmup_epochs: int = CFG.WARMUP_EPOCHS,
                    total_epochs: int = CFG.NUM_EPOCHS):
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                                                total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs - warmup_epochs,
                                                         eta_min=CFG.LEARNING_RATE * 1e-3)
    return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine],
                                                  milestones=[warmup_epochs])


def _train_tf(sz: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=sz),
        A.PadIfNeeded(sz, sz, border_mode=cv2.BORDER_REFLECT_101,
                      mask_value=CFG.IGNORE_INDEX),
        A.RandomCrop(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5,
                           border_mode=cv2.BORDER_CONSTANT, value=0,
                           mask_value=CFG.IGNORE_INDEX),
        A.ElasticTransform(alpha=120, sigma=6, p=0.3,
                           border_mode=cv2.BORDER_CONSTANT, value=0,
                           mask_value=CFG.IGNORE_INDEX),
        A.GridDistortion(p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(0.3, 0.3, p=1.0),
            A.HueSaturationValue(10, 20, 10, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.6),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1,
                        fill_value=0, mask_fill_value=CFG.IGNORE_INDEX, p=0.2),
        A.Normalize(mean=CFG.MEAN, std=CFG.STD),
        ToTensorV2(),
    ])


def _val_tf(sz: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=sz),
        A.PadIfNeeded(sz, sz, border_mode=cv2.BORDER_REFLECT_101,
                      mask_value=CFG.IGNORE_INDEX),
        A.Normalize(mean=CFG.MEAN, std=CFG.STD),
        ToTensorV2(),
    ])


class WoundDataset(Dataset):
    """Load paired images and masks, with a reference-mask-defined square ROI."""
    def __init__(self, root: str, split: str = "train",
                 transform: A.Compose | None = None) -> None:
        self.transform = transform
        base_dir = os.path.join(root, split)

        if os.path.isdir(os.path.join(base_dir, "images")):
            self.img_dir  = os.path.join(base_dir, "images")
            self.mask_dir = os.path.join(base_dir, "masks")
        else:
            self.img_dir  = base_dir
            self.mask_dir = base_dir

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")

        self.imgs = sorted(p for p in glob.glob(os.path.join(self.img_dir, "*"))
                           if Path(p).suffix.lower() in IMAGE_EXTS
                           and "_mask" not in Path(p).stem)

        if not self.imgs:
            raise RuntimeError(f"No images in {self.img_dir}")
        print(f"  [{split:5s}]  {len(self.imgs)} images")

    def _mask_path(self, img_path: str) -> str:
        stem = Path(img_path).stem
        for suf in [".png", "_mask.png", ".jpg"]:
            c = os.path.join(self.mask_dir, stem + suf)
            if os.path.exists(c): return c
        cands = glob.glob(os.path.join(self.mask_dir, stem + "*"))
        if cands: return cands[0]
        raise FileNotFoundError(f"No mask for: {img_path}")

    def __len__(self) -> int: return len(self.imgs)

    def __getitem__(self, idx: int):
        ip = self.imgs[idx]
        mp = self._mask_path(ip)

        img  = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE).astype(np.int32)

        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        foreground_mask = (mask > 0) & (mask != CFG.IGNORE_INDEX)

        if foreground_mask.any():
            # Expand the mask bounding box by 40%, keeping the square within the image.
            y_idx, x_idx = np.where(foreground_mask)
            y0, y1 = int(y_idx.min()), int(y_idx.max())
            x0, x1 = int(x_idx.min()), int(x_idx.max())

            box_h = max(1, y1 - y0)
            box_w = max(1, x1 - x0)

            h, w = img.shape[:2]
            side = int(round(max(box_h, box_w) * 1.4))
            side = max(side, CFG.IMAGE_SIZE)
            side = min(side, h, w)

            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2
            half = side // 2
            y_min = cy - half
            x_min = cx - half
            y_max = y_min + side
            x_max = x_min + side
            if y_min < 0:
                y_min, y_max = 0, side
            elif y_max > h:
                y_min, y_max = h - side, h
            if x_min < 0:
                x_min, x_max = 0, side
            elif x_max > w:
                x_min, x_max = w - side, w

            img  = img[y_min:y_max, x_min:x_max]
            mask = mask[y_min:y_max, x_min:x_max]

        mask[(mask >= CFG.NUM_CLASSES) & (mask != CFG.IGNORE_INDEX)] = CFG.IGNORE_INDEX
        mask = mask.astype(np.uint8)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"]

        return img, mask.long(), ip


def build_loaders(root: str, batch_size: int, image_size: int,
                  num_workers: int = 8, seed: int = 42
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0,
              worker_init_fn=_seed_worker, generator=g)
    tr = DataLoader(WoundDataset(root, "train", _train_tf(image_size)),
                    batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    va = DataLoader(WoundDataset(root, "valid", _val_tf(image_size)),
                    batch_size=batch_size, shuffle=False, **kw)
    te = DataLoader(WoundDataset(root, "test",  _val_tf(image_size)),
                    batch_size=batch_size, shuffle=False, **kw)
    return tr, va, te


def _up(t: torch.Tensor) -> torch.Tensor:
    return F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)


def _to_nchw(t: torch.Tensor, expected_c: int) -> torch.Tensor:
    if t.dim() == 4 and t.shape[1] != expected_c and t.shape[-1] == expected_c:
        t = t.permute(0, 3, 1, 2).contiguous()
    return t


def _basic_dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class ASPPModule(nn.Module):
    """Parallel atrous convolutions and pooled context at the encoder bottleneck."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.c2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
                                nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.c3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
                                nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.c4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
                                nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        feat_pool = F.interpolate(self.pool(x), size=(h, w), mode='bilinear', align_corners=False)
        return self.project(torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x), feat_pool], dim=1))


class AttentionGate(nn.Module):
    """Gate a skip connection using decoder features."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DASP_AG_SwinUNet(nn.Module):
    """SwinV2 encoder, ASPP bottleneck and attention-gated skip connections."""
    def __init__(self, num_classes: int = CFG.NUM_CLASSES,
                 backbone: str = "swinv2_small_window8_256",
                 img_size: int = CFG.IMAGE_SIZE, pretrained: bool = True) -> None:
        super().__init__()
        import timm
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, features_only=True,
            img_size=img_size, out_indices=(0, 1, 2, 3),
        )
        self._chs = list(self.encoder.feature_info.channels())

        self.dec3 = _basic_dec_block(self._chs[3] + self._chs[2], self._chs[2])
        self.dec2 = _basic_dec_block(self._chs[2] + self._chs[1], self._chs[1])
        self.dec1 = _basic_dec_block(self._chs[1] + self._chs[0], self._chs[0])
        self.dec0 = _basic_dec_block(self._chs[0], 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        self.aspp = ASPPModule(self._chs[3], self._chs[3])
        self.ag3 = AttentionGate(F_g=self._chs[3], F_l=self._chs[2], F_int=self._chs[2] // 2)
        self.ag2 = AttentionGate(F_g=self._chs[2], F_l=self._chs[1], F_int=self._chs[1] // 2)
        self.ag1 = AttentionGate(F_g=self._chs[1], F_l=self._chs[0], F_int=self._chs[0] // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        f0, f1, f2, f3 = [_to_nchw(f, c) for f, c in zip(self.encoder(x), self._chs)]

        f3_aspp = self.aspp(f3)

        g3 = _up(f3_aspp)
        d3 = self.dec3(torch.cat([g3, self.ag3(g=g3, x=f2)], dim=1))
        g2 = _up(d3)
        d2 = self.dec2(torch.cat([g2, self.ag2(g=g2, x=f1)], dim=1))
        g1 = _up(d2)
        d1 = self.dec1(torch.cat([g1, self.ag1(g=g1, x=f0)], dim=1))

        return F.interpolate(self.head(self.dec0(_up(d1))), (H, W),
                             mode="bilinear", align_corners=False)


def count_params(m: nn.Module) -> str:
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return f"{n:,} ({n / 1e6:.1f} M)"


class SegMetrics:
    def __init__(self, num_classes=CFG.NUM_CLASSES, ignore_index=CFG.IGNORE_INDEX,
                 names: list | None = None) -> None:
        self.C  = num_classes; self.ig = ignore_index
        self.nm = names or CFG.CLASSES
        self._cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self): self._cm[:] = 0

    def update(self, preds, targets):
        if isinstance(preds,   torch.Tensor): preds   = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()
        p = preds.ravel().astype(np.int64); t = targets.ravel().astype(np.int64)
        v = t != self.ig; p = np.clip(p[v], 0, self.C - 1); t = t[v]
        flat = t * self.C + p
        self._cm += np.bincount(flat, minlength=self.C ** 2).reshape(self.C, self.C)

    def compute(self) -> dict:
        cm = self._cm.astype(np.float64)
        per, iou_l, dsc_l, pre_l, rec_l, spe_l, f2_l = {}, [], [], [], [], [], []

        for c in range(self.C):
            TP = cm[c, c]
            FP = cm[:, c].sum() - TP
            FN = cm[c, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            pres = (TP + FN + FP) > 0

            pre  = TP / (TP + FP + EPS)
            rec  = TP / (TP + FN + EPS)
            spe  = TN / (TN + FP + EPS)
            dsc  = 2 * TP / (2 * TP + FP + FN + EPS)
            f2   = 5 * TP / (5 * TP + 4 * FN + FP + EPS)
            iou  = TP / (TP + FP + FN + EPS)
            acc  = (TP + TN) / (TP + TN + FP + FN + EPS)

            nm = self.nm[c] if c < len(self.nm) else f"class_{c}"
            per[nm] = {
                "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
                "Precision":   float(pre)  if pres else float("nan"),
                "Recall":      float(rec)  if pres else float("nan"),
                "Specificity": float(spe),
                "DSC":         float(dsc)  if pres else float("nan"),
                "F2":          float(f2)   if pres else float("nan"),
                "IoU":         float(iou)  if pres else float("nan"),
                "Accuracy":    float(acc),
            }
            if pres:
                iou_l.append(iou); dsc_l.append(dsc); pre_l.append(pre)
                rec_l.append(rec); spe_l.append(spe); f2_l.append(f2)

        def _m(lst): return float(np.mean([x for x in lst if not np.isnan(x)])) if lst else float("nan")

        pixel_acc = cm.diagonal().sum() / (cm.sum() + EPS)
        freq      = cm.sum(axis=1) / (cm.sum() + EPS)
        fw_iou    = float(sum(
            freq[c] * cm[c, c] / (cm[:, c].sum() + cm[c, :].sum() - cm[c, c] + EPS)
            for c in range(self.C)))

        agg = {
            "mIoU": _m(iou_l), "mDSC": _m(dsc_l),
            "mPrecision": _m(pre_l), "mRecall": _m(rec_l),
            "mSpecificity": _m(spe_l), "mF2": _m(f2_l),
            "PixelAcc": float(pixel_acc), "MeanAcc": _m(rec_l), "FWIoU": fw_iou,
        }
        return {"per_class": per, "aggregates": agg}

    def log_str(self, epoch: int | None = None, split: str = "Val") -> str:
        r   = self.compute()
        agg = r["aggregates"]
        hdr = f"[{split}]" + (f" Epoch {epoch:03d}" if epoch is not None else "")
        sep = "=" * 90
        lines = [sep, f"{hdr}  Aggregate Metrics", sep,
                 f"  mIoU={agg['mIoU']:.4f}  mDSC={agg['mDSC']:.4f}  "
                 f"mPrecision={agg['mPrecision']:.4f}  mRecall={agg['mRecall']:.4f}",
                 f"  mSpecificity={agg['mSpecificity']:.4f}  mF2={agg['mF2']:.4f}  "
                 f"PixelAcc={agg['PixelAcc']:.4f}  MeanAcc={agg['MeanAcc']:.4f}  "
                 f"FWIoU={agg['FWIoU']:.4f}", "",
                 f"{'Class':<26}{'IoU':>8}{'DSC':>8}{'Prec':>8}{'Rec':>8}"
                 f"{'Spec':>8}{'F2':>8}{'Acc':>8}", "-" * 90]
        for nm, m in r["per_class"].items():
            def f(v): return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "  N/A "
            lines.append(f"{nm:<26}{f(m['IoU']):>8}{f(m['DSC']):>8}"
                         f"{f(m['Precision']):>8}{f(m['Recall']):>8}"
                         f"{f(m['Specificity']):>8}{f(m['F2']):>8}{f(m['Accuracy']):>8}")
        lines.append(sep)
        return "\n".join(lines)


def _plot_cm(cm: np.ndarray, names: list, path: str, title: str) -> None:
    rows = cm.sum(axis=1, keepdims=True).astype(float)
    norm = np.where(rows > 0, cm / (rows + EPS), 0.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(norm, annot=True, fmt=".2f", xticklabels=names, yticklabels=names,
                cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def _plot_cm_counts(cm: np.ndarray, names: list, path: str, title: str) -> None:
    cm_int = cm.astype(np.int64)
    fig, ax = plt.subplots(figsize=(8, 7))
    annot = np.array([[f"{v/1000:.1f}k" if v >= 1000 else str(v)
                       for v in row] for row in cm_int])
    sns.heatmap(cm_int, annot=annot, fmt="", xticklabels=names, yticklabels=names,
                cmap="Oranges", ax=ax, annot_kws={"size": 10})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def _mask_to_rgb(mask: np.ndarray, ignore_color=(255, 255, 255)) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(CFG.NUM_CLASSES, len(CLASS_RGB))):
        rgb[mask == c] = CLASS_RGB[c]
    rgb[mask == CFG.IGNORE_INDEX] = ignore_color
    return rgb


def _visualize_test_predictions(model, test_loader, device, use_amp,
                                 model_name: str, fig_dir: str,
                                 max_images: int | None = None) -> None:
    out_dir = os.path.join(fig_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    mean = np.array(CFG.MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(CFG.STD,  dtype=np.float32).reshape(1, 1, 3)

    legend_patches = [
        mpatches.Patch(color=[c / 255 for c in CLASS_RGB[i]], label=CFG.CLASSES[i])
        for i in range(CFG.NUM_CLASSES)
    ]

    model.eval()
    n_done = 0
    with torch.no_grad():
        for imgs, masks, paths in test_loader:
            imgs_d = imgs.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(imgs_d)
            preds = logits.argmax(1).cpu().numpy()
            imgs_cpu  = imgs.cpu().numpy()
            masks_cpu = masks.cpu().numpy()

            for i in range(imgs_cpu.shape[0]):
                img_t = imgs_cpu[i].transpose(1, 2, 0)
                img_rgb = (img_t * std + mean).clip(0.0, 1.0)
                img_uint8 = (img_rgb * 255).astype(np.uint8)

                gt   = masks_cpu[i].astype(np.int32)
                pred = preds[i].astype(np.int32)

                gt_rgb   = _mask_to_rgb(gt)
                pred_rgb = _mask_to_rgb(pred)
                overlay  = (img_uint8.astype(np.float32) * 0.55
                            + pred_rgb.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(img_uint8);   axes[0].set_title("Original");      axes[0].axis("off")
                axes[1].imshow(gt_rgb);      axes[1].set_title("Ground Truth");  axes[1].axis("off")
                axes[2].imshow(pred_rgb);    axes[2].set_title("Prediction");    axes[2].axis("off")
                axes[3].imshow(overlay);     axes[3].set_title("Overlay");       axes[3].axis("off")

                fig.legend(handles=legend_patches, loc="lower center",
                           ncol=CFG.NUM_CLASSES, fontsize=10,
                           bbox_to_anchor=(0.5, -0.02))
                plt.suptitle(f"{model_name}  |  {Path(paths[i]).name}", fontsize=12, y=1.02)
                plt.tight_layout()

                stem = Path(paths[i]).stem
                plt.savefig(os.path.join(out_dir, f"{stem}.png"),
                            dpi=120, bbox_inches="tight")
                plt.close(fig)

                n_done += 1
                if max_images is not None and n_done >= max_images:
                    return


def _plot_per_class(results: dict, model_name: str, save_dir: str) -> None:
    pc = results["per_class"]; names = list(pc.keys())
    for metric in ["IoU", "DSC", "Precision", "Recall", "Specificity", "F2"]:
        vals = [pc[n].get(metric, float("nan")) for n in names]
        fig, ax = plt.subplots(figsize=(9, 4))
        cols = ["steelblue" if not np.isnan(v) else "lightgray" for v in vals]
        ax.bar(names, [0 if np.isnan(v) else v for v in vals], color=cols)
        ax.set_ylim(0, 1.1); ax.set_ylabel(metric)
        ax.set_title(f"{model_name} — Per-class {metric}")
        for i, v in enumerate(vals):
            if not np.isnan(v): ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        plt.xticks(rotation=20, ha="right"); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{model_name}_{metric}.png"), dpi=150)
        plt.close(fig)


def _train_epoch(model, loader, crit, opt, device, scaler, use_amp, accum, epoch, logger):
    model.train(); meter = SegMetrics()
    tl = td = tc = tf_ = 0.0; opt.zero_grad()
    pbar = tqdm(loader, desc=f"Train E{epoch:03d}", leave=False, ncols=130)
    for step, (imgs, masks, _) in enumerate(pbar):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits       = model(imgs)
            loss, sub    = crit(logits, masks)
            scaled_loss  = loss / accum
        scaler.scale(scaled_loss).backward()
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP_GRAD_NORM)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        meter.update(logits.detach().argmax(1), masks)
        tl  += loss.item()
        tc  += sub["ce"]
        td  += sub["dice"]
        tf_ += sub.get("focal", 0.0)
        pbar.set_postfix(loss=f"{tl/(step+1):.4f}", ce=f"{tc/(step+1):.4f}",
                         dice=f"{td/(step+1):.4f}", focal=f"{tf_/(step+1):.4f}")
    n = len(loader)
    r = meter.compute()
    return {"loss": tl/n, "ce_loss": tc/n, "dice_loss": td/n,
            "focal_loss": tf_/n, **r["aggregates"]}


@torch.no_grad()
def _eval_epoch(model, loader, crit, device, use_amp, split, epoch, logger):
    model.eval(); meter = SegMetrics(); tl = 0.0
    pbar = tqdm(loader, desc=f"{split.capitalize()} E{epoch or 0:03d}", leave=False, ncols=110)
    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(imgs); loss, _ = crit(logits, masks)
        meter.update(logits.argmax(1), masks); tl += loss.item()
        pbar.set_postfix(loss=f"{tl/(pbar.n+1):.4f}")
    r = meter.compute()
    if logger: logger.info(meter.log_str(epoch=epoch, split=split.capitalize()))
    return {"loss": tl / len(loader), **r["aggregates"], "_res": r, "_meter": meter}


def phase_train(args) -> None:
    set_seed(args.seed)
    name       = args.model
    seed_tag   = f"seed_{args.seed}"
    batch_size = args.batch_size or (4 if name in CFG.HEAVY_MODELS else CFG.BATCH_SIZE)
    epochs     = args.epochs     or CFG.NUM_EPOCHS
    image_size = args.image_size or CFG.IMAGE_SIZE
    lr         = args.lr         or CFG.LEARNING_RATE

    ckpt_dir   = os.path.join(CFG.CHECKPOINT_DIR, name, seed_tag)
    log_dir    = os.path.join(CFG.LOG_DIR,        name, seed_tag)
    res_dir    = os.path.join(CFG.RESULTS_DIR,    name, seed_tag)
    fig_dir    = os.path.join(CFG.FIGURES_DIR,    name, seed_tag)
    for d in [ckpt_dir, log_dir, res_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    best_path = os.path.join(ckpt_dir, f"{name}_best.pth")
    if not args.force_retrain and os.path.exists(best_path):
        print(f"[SKIP] {name} {seed_tag}: best checkpoint exists at {best_path}")
        return

    run_name   = f"{name}_{seed_tag}"
    logger     = get_logger(run_name, log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir, f"{name}_metrics.csv"))
    tb         = SummaryWriter(os.path.join(CFG.LOG_DIR, "tensorboard", name, seed_tag))
    timer      = Timer()

    logger.info("=" * 70)
    logger.info(f"  MODEL      : {name}")
    logger.info(f"  SEED       : {args.seed}")
    logger.info(f"  EPOCHS     : {epochs}")
    logger.info(f"  BATCH SIZE : {batch_size}")
    logger.info(f"  IMAGE SIZE : {image_size}")
    logger.info(f"  LR         : {lr}")
    logger.info("=" * 70)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False
    logger.info(f"Device: {device}  |  AMP: {use_amp} (strict repro)")

    dataset_root = args.dataset_root or CFG.DATASET_DIR
    logger.info(f"Dataset: {dataset_root}")
    print("Loading datasets …")
    train_loader, val_loader, test_loader = build_loaders(
        dataset_root, batch_size, image_size, args.num_workers, seed=args.seed)

    model = build_model(name).to(device)
    logger.info(f"Parameters: {count_params(model)}")

    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, total_epochs=epochs)
    scaler    = GradScaler(enabled=use_amp)
    stopper   = EarlyStopping(CFG.EARLY_STOPPING_PATIENCE)

    best_miou = 0.0
    start_ep  = 1

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw  = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_ep  = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("best_miou", 0.0)
        logger.info(f"Resumed from {args.resume} (epoch {start_ep - 1})")

    logger.info("Starting training …")

    for epoch in range(start_ep, epochs + 1):
        tr  = _train_epoch(model, train_loader, criterion, optimizer,
                           device, scaler, use_amp, CFG.GRAD_ACCUMULATION_STEPS,
                           epoch, logger)
        val = _eval_epoch(model, val_loader, criterion, device, use_amp,
                          "val", epoch, logger)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        miou    = val["mIoU"]
        is_best = miou > best_miou
        best_miou = max(best_miou, miou)

        logger.info(
            f"Epoch {epoch:03d}/{epochs}  "
            f"TrainLoss={tr['loss']:.4f}  ValLoss={val['loss']:.4f}  "
            f"mIoU={miou:.4f}  mDSC={val['mDSC']:.4f}  "
            f"LR={lr_now:.2e}  Best={best_miou:.4f}"
            + ("  *** BEST ***" if is_best else "")
        )

        for k, v in tr.items():
            if isinstance(v, float): tb.add_scalar(f"train/{k}", v, epoch)
        for k, v in val.items():
            if isinstance(v, float): tb.add_scalar(f"val/{k}", v, epoch)
        tb.add_scalar("lr", lr_now, epoch)

        row = {"epoch": epoch, "lr": lr_now,
               **{f"train_{k}": v for k, v in tr.items()  if isinstance(v, (float, int))},
               **{f"val_{k}":   v for k, v in val.items() if isinstance(v, (float, int))}}
        csv_logger.log(row)

        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        state = {"epoch": epoch, "model_state": raw.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(),
                 "best_miou": best_miou}
        torch.save(state, os.path.join(ckpt_dir, f"{name}_last.pth"))
        if is_best:
            torch.save(state, os.path.join(ckpt_dir, f"{name}_best.pth"))

        if stopper.step(miou):
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    logger.info("Loading best checkpoint for test …")
    best_path = os.path.join(ckpt_dir, f"{name}_best.pth")
    if os.path.exists(best_path):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        ck  = torch.load(best_path, map_location=device)
        raw.load_state_dict(ck["model_state"])

    test = _eval_epoch(model, test_loader, criterion, device, use_amp,
                       "test", None, logger)

    results = {
        "model": name, "seed": args.seed, "best_val_mIoU": best_miou,
        "test_aggregates": test["_res"]["aggregates"],
        "test_per_class":  test["_res"]["per_class"],
    }
    save_json(results, os.path.join(res_dir, "test_results.json"))

    if "_meter" in test:
        meter = test["_meter"]
        _plot_cm(meter._cm, CFG.CLASSES,
                 os.path.join(fig_dir, "confusion_matrix.png"),
                 f"{name} (seed={args.seed}) — Confusion Matrix (Test, Row-Normalized)")
        _plot_cm_counts(meter._cm, CFG.CLASSES,
                        os.path.join(fig_dir, "confusion_matrix_counts.png"),
                        f"{name} (seed={args.seed}) — Confusion Matrix (Test, Raw Pixel Counts)")
        _plot_per_class(test["_res"], f"{name}_seed{args.seed}", fig_dir)

    logger.info("Saving test-image visualizations …")
    _visualize_test_predictions(model, test_loader, device, use_amp,
                                model_name=f"{name}/seed_{args.seed}", fig_dir=fig_dir)

    logger.info(f"Done.  Time: {timer.elapsed()}  |  Best val mIoU: {best_miou:.4f}")
    logger.info(f"Results → {res_dir}")
    tb.close()


