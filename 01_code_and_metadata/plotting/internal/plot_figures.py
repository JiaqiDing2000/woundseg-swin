"""Redraw the internal segmentation panels using the manuscript figure style."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.text import Text
import numpy as np


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
EXPORTS = BASE.parents[1] / "figures"
MM = 1 / 25.4
SEEDS = tuple(range(1001, 1007))
CLASSES = ("Background", "Epithelial", "Granulation", "Necrotic", "Pus")
TARGET = "dasp_ag_swinunet_aspp_small_rates"
MODEL_LABELS = {
    TARGET: "Ours",
    "unet": "UNet",
    "unetplusplus": "UNet++",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "pan": "PAN",
    "manet": "MAnet",
    "deeplabv3plus": "DeepLabV3+",
    "segformer": "SegFormer",
}
METRIC_COLORS = {
    "IoU": "#4DBBD5",
    "DSC": "#E64B35",
    "Precision": "#00A087",
    "Recall": "#3C5488",
}
BASELINE_COLORS = (
    "#E15759", "#F28E2B", "#59A14F", "#EDC948",
    "#B07AA1", "#76B7B2", "#FF9DA7", "#9CA0A4",
)
CAPTIONS = {
    "Fig2H_internal_confusion_ASPP123": (
        "Fig. 2H. Tissue segmentation on the internal wound dataset. "
        "The confusion matrix is row-normalized within each seed and then averaged "
        "over six training seeds (1001–1006). Rows denote reference labels and columns "
        "denote predictions. Evaluation uses the 35-image internal development evaluation subset "
        "with ground-truth-assisted 256 × 256 ROI preprocessing."
    ),
    "FigS2_internal_tissue_metrics_ASPP123": (
        "Fig. S2. Class-wise segmentation performance of ASPP-AG-SwinUNet with "
        "balanced sampling, CE + Tversky loss and ASPP dilation rates 1, 2 and 3. "
        "IoU, DSC and precision are computed from pooled test pixels separately for "
        "each seed. Circles denote the six training seeds (1001–1006); diamonds and "
        "whiskers denote their mean and sample standard deviation. Evaluation uses "
        "the 35-image internal development evaluation subset with ground-truth-assisted "
        "256 × 256 ROI preprocessing."
    ),
    "FigS3_internal_baseline_comparison_ASPP123": (
        "Fig. S3. Comparison of ASPP-AG-SwinUNet with eight segmentation baselines "
        "on the internal wound dataset. Ours uses balanced sampling, CE + Tversky "
        "loss and ASPP dilation rates 1, 2 and 3. mIoU is the unweighted average of "
        "the five class IoUs, including background, computed from pooled test pixels. "
        "Circles denote six training seeds per model (1001–1006); black horizontal "
        "lines denote the mean and vertical lines denote one sample standard "
        "deviation. Models are ordered by mean mIoU. All models use the same "
        "35-image internal development evaluation subset and ground-truth-assisted "
        "256 × 256 ROI evaluation."
    ),
    "FigS5_internal_confusion_counts_ASPP123": (
        "Fig. S5. Unnormalized confusion matrix for the internal wound dataset. "
        "Cells show absolute pixel counts for seed 1004, retaining the representative "
        "run specified in the manuscript. This run evaluates 2,293,760 pixels from "
        "the 35-image internal development evaluation subset after ground-truth-assisted "
        "256 × 256 ROI preprocessing. Rows denote reference labels and columns "
        "denote predictions; no row or column normalization is applied. Counts are "
        "abbreviated as k (×10³ pixels); exact integer counts are provided in the CSV."
    ),
    "Internal_tissue_metrics_with_recall_ASPP123": (
        "Class-wise segmentation performance including recall. The model and "
        "evaluation protocol are identical to Fig. S2. Circles denote six training "
        "seeds (1001–1006); diamonds and whiskers denote the mean and sample "
        "standard deviation. This four-metric layout adds recall to the original "
        "IoU, DSC and precision presentation."
    ),
}


def setup_style() -> None:
    for name in ("Arimo-Regular.ttf", "Arimo-Bold.ttf"):
        font_manager.fontManager.addfont(str(BASE / "fonts" / name))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arimo", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "wound-internal-aspp123-20261018",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty input: {name}")
    return rows


def class_values() -> dict[tuple[str, str], np.ndarray]:
    rows = read_csv("class_metrics_all_seeds.csv")
    values: dict[tuple[str, str], np.ndarray] = {}
    for tissue in CLASSES:
        for metric in METRIC_COLORS:
            group = [r for r in rows if r["class"] == tissue and r["metric"] == metric]
            group.sort(key=lambda r: int(r["seed"]))
            if tuple(int(r["seed"]) for r in group) != SEEDS:
                raise ValueError(f"Expected each of the six seeds: {tissue}, {metric}")
            scores = np.asarray([float(r["value_pct"]) for r in group])
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 100)).any():
                raise ValueError(f"Invalid percentage: {tissue}, {metric}")
            values[tissue, metric] = scores
    if len(rows) != len(CLASSES) * len(METRIC_COLORS) * len(SEEDS):
        raise ValueError("Unexpected class metric rows")
    return values


def tissue_figure(include_recall: bool = False) -> plt.Figure:
    metrics = tuple(METRIC_COLORS) if include_recall else ("IoU", "DSC", "Precision")
    values = class_values()
    fig, ax = plt.subplots(figsize=(150 * MM, 85 * MM))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.185, top=0.85)
    group_width = 0.70
    handles = []
    for j, metric in enumerate(metrics):
        color = METRIC_COLORS[metric]
        handles.append(Line2D([], [], color=color, marker="D", linestyle="None",
                              markersize=4.5, markeredgecolor="#333333",
                              markeredgewidth=0.5, label=metric))
        for i, tissue in enumerate(CLASSES):
            scores = values[tissue, metric]
            x = i + (j - (len(metrics) - 1) / 2) * group_width / len(metrics)
            jitter = np.linspace(-0.043, 0.043, len(SEEDS))
            ax.scatter(x + jitter, scores, color=color, s=13, alpha=0.48,
                       edgecolors="none", zorder=2)
            ax.errorbar(x, scores.mean(), yerr=scores.std(ddof=1), fmt="D",
                        color=color, markeredgecolor="#333333", markeredgewidth=0.5,
                        markersize=4.2, elinewidth=0.8, capsize=2, capthick=0.7,
                        zorder=3)
    ax.set_xlim(-0.55, len(CLASSES) - 0.45)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(np.arange(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Tissue", labelpad=7)
    ax.set_ylabel("Score (%)", labelpad=6)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.03),
              ncol=len(metrics), handletextpad=0.5, columnspacing=1.4, borderaxespad=0)
    return fig


def benchmark_figure() -> plt.Figure:
    rows = read_csv("benchmark_per_seed.csv")
    models = sorted({r["model"] for r in rows})
    if set(models) != set(MODEL_LABELS) or len(rows) != 54:
        raise ValueError(f"Expected target and eight baselines, six seeds each: {models}")
    data = {}
    for model in models:
        group = sorted((r for r in rows if r["model"] == model), key=lambda r: int(r["seed"]))
        if tuple(int(r["seed"]) for r in group) != SEEDS:
            raise ValueError(f"Incomplete seeds for {model}")
        scores = np.asarray([float(r["mIoU"]) * 100 for r in group])
        if not np.isfinite(scores).all():
            raise ValueError(f"Nonfinite scores for {model}")
        data[model] = scores
    order = sorted(models, key=lambda model: (-data[model].mean(), model))
    fig, ax = plt.subplots(figsize=(150 * MM, 90 * MM))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.295, top=0.955)
    color_index = 0
    for x, model in enumerate(order):
        scores = data[model]
        if model == TARGET:
            color = "#0056B3"
        else:
            color = BASELINE_COLORS[color_index]
            color_index += 1
        jitter = np.asarray([-0.14, -0.084, -0.028, 0.028, 0.084, 0.14])
        ax.scatter(x + jitter, scores, color=color, s=22, alpha=0.75,
                   edgecolors="none", zorder=2)
        mean, sd = scores.mean(), scores.std(ddof=1)
        ax.hlines(mean, x - 0.24, x + 0.24, color="black", linewidth=1.35, zorder=3)
        ax.vlines(x, mean - sd, mean + sd, color="black", linewidth=0.85, zorder=3)
    low = min(min(v.min(), v.mean() - v.std(ddof=1)) for v in data.values())
    high = max(max(v.max(), v.mean() + v.std(ddof=1)) for v in data.values())
    lower = min(55, 5 * np.floor((low - 1) / 5))
    upper = max(85, 5 * np.ceil((high + 1) / 5))
    ax.set_ylim(lower, upper)
    ax.set_yticks(np.arange(lower, upper + 1, 5))
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)), labels=[MODEL_LABELS[m] for m in order],
                  rotation=45, ha="right", rotation_mode="anchor")
    for label, model in zip(ax.get_xticklabels(), order):
        if model == TARGET:
            label.set_fontweight("bold")
            label.set_color("#0056B3")
    ax.set_xlabel("Model", labelpad=5)
    ax.set_ylabel("mIoU (%)", labelpad=6)
    return fig


def confusion_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Read long-form matrices and independently check their normalization."""
    count_rows = read_csv("confusion_counts_by_seed.csv")
    counts = np.full((len(SEEDS), len(CLASSES), len(CLASSES)), np.nan)
    for row in count_rows:
        s = SEEDS.index(int(row["seed"]))
        i = CLASSES.index(row["true_class"])
        j = CLASSES.index(row["predicted_class"])
        if np.isfinite(counts[s, i, j]):
            raise ValueError("Duplicate confusion count")
        counts[s, i, j] = float(row["pixels"])
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Incomplete or invalid confusion counts")
    if not np.all(counts.sum(axis=(1, 2)) == 2293760):
        raise ValueError("Unexpected evaluated pixel count")
    normalized = (counts / counts.sum(axis=2, keepdims=True)).mean(axis=0)
    supplied = np.full((len(CLASSES), len(CLASSES)), np.nan)
    for row in read_csv("mean_confusion_pct.csv"):
        i = CLASSES.index(row["true_class"])
        j = CLASSES.index(row["predicted_class"])
        if np.isfinite(supplied[i, j]):
            raise ValueError("Duplicate normalized confusion value")
        supplied[i, j] = float(row["mean_row_pct"]) / 100
    if not np.isfinite(supplied).all() or not np.allclose(supplied, normalized, atol=1e-10, rtol=0):
        raise ValueError("Normalized matrix differs from the per-seed count calculation")
    return normalized, counts[SEEDS.index(1004)]


def confusion_figure(normalized: bool) -> plt.Figure:
    proportions, counts = confusion_arrays()
    matrix = proportions if normalized else counts
    if normalized:
        fig = plt.figure(figsize=(85 * MM, 75 * MM))
        ax = fig.add_axes([0.215, 0.22, 0.62, 0.703])
        cax = fig.add_axes([0.875, 0.22, 0.025, 0.703])
        labels = ["Back-\nground", "Epith-\nelial", "Gran-\nulation", "Necr-\notic", "Pus"]
        font_size = 6.8
        formatter = lambda value: f"{value:.2f}"
        maximum = 1.0
    else:
        fig = plt.figure(figsize=(135 * MM, 110 * MM))
        ax = fig.add_axes([0.14, 0.205, 0.68, 0.745])
        cax = fig.add_axes([0.86, 0.205, 0.022, 0.745])
        labels = list(CLASSES)
        font_size = 6.8
        formatter = lambda value: "0" if value == 0 else f"{value / 1000:,.2f}k"
        maximum = float(matrix.max())
    edges = np.arange(len(CLASSES) + 1) - 0.5
    im = ax.pcolormesh(edges, edges, matrix, cmap="Blues", vmin=0,
                       vmax=maximum, shading="flat", rasterized=False)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, len(CLASSES) - 0.5)
    ax.set_ylim(len(CLASSES) - 0.5, -0.5)
    ax.set_xticks(range(len(CLASSES)), labels=labels)
    ax.set_yticks(range(len(CLASSES)), labels=labels)
    if not normalized:
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Predict", labelpad=6)
    ax.set_ylabel("True", labelpad=6)
    ax.tick_params(axis="both", which="major", length=0, pad=4)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="#929292", linestyle=(0, (4, 4)), linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#929292")
        spine.set_linewidth(0.5)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if matrix[i, j] > maximum * 0.52 else "#262222"
            ax.text(j, i, formatter(matrix[i, j]), ha="center", va="center",
                    fontsize=font_size, color=color)
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, pad=3, labelsize=6.5)
    if normalized:
        colorbar.set_ticks(np.linspace(0, 1, 6))
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
        fig.text(0.025, 0.975, "H", fontsize=11, fontweight="bold", va="top")
    else:
        colorbar.set_label("Pixel count (seed 1004)", fontsize=7.5, labelpad=5)
        colorbar.set_ticks([0, 500000, 1000000, 1500000])
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
    return fig


def export(fig: plt.Figure, stem: str, review: PdfPages) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    text_outside = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
            text_outside.append({"text": artist.get_text(), "bbox_pixels": [round(float(v), 3) for v in box.extents]})
    width, height = fig.get_size_inches()
    (EXPORTS / f"{stem}_export.json").write_text(json.dumps({
        "width_mm": round(float(width * 25.4), 4),
        "height_mm": round(float(height * 25.4), 4),
        "raster_dpi": 600,
        "preview_dpi": 180,
        "svg_text_editable": True,
        "font": "Arimo",
        "text_outside_figure": text_outside,
    }, indent=2) + "\n", encoding="utf-8")
    metadata = {"Title": stem, "Creator": "Matplotlib", "CreationDate": None, "ModDate": None}
    fig.savefig(EXPORTS / f"{stem}.pdf", metadata=metadata)
    fig.savefig(EXPORTS / f"{stem}.svg")
    fig.savefig(EXPORTS / f"{stem}.png", dpi=600)
    fig.savefig(EXPORTS / f"{stem}_preview.png", dpi=180)
    review.savefig(fig)
    (EXPORTS / f"{stem}_caption.txt").write_text(CAPTIONS[stem] + "\n", encoding="utf-8")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-only", action="store_true", help="Draw class metrics and model benchmark before the confusion matrices are available.")
    args = parser.parse_args()
    setup_style()
    EXPORTS.mkdir(parents=True, exist_ok=True)
    figures = [
        ("FigS2_internal_tissue_metrics_ASPP123", lambda: tissue_figure(False)),
        ("FigS3_internal_baseline_comparison_ASPP123", benchmark_figure),
        ("Internal_tissue_metrics_with_recall_ASPP123", lambda: tissue_figure(True)),
    ]
    if not args.metrics_only:
        figures.insert(0, ("Fig2H_internal_confusion_ASPP123", lambda: confusion_figure(True)))
        figures.insert(3, ("FigS5_internal_confusion_counts_ASPP123", lambda: confusion_figure(False)))
    review_name = "Internal_ASPP123_metrics_review.pdf" if args.metrics_only else "Internal_ASPP123_figures_review.pdf"
    with PdfPages(EXPORTS / review_name, metadata={"Title": "Internal wound segmentation: ASPP rates 1, 2, 3", "CreationDate": None, "ModDate": None}) as review:
        for stem, builder in figures:
            export(builder(), stem, review)
    if not args.metrics_only:
        (EXPORTS / "Figure_captions_EN.txt").write_text(
            "\n\n".join(CAPTIONS[stem] for stem, _ in figures) + "\n", encoding="utf-8")
    print(f"Exported {len(figures)} figures to {EXPORTS}")


if __name__ == "__main__":
    main()
