"""Export editable tables from the same source records used for the figures."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
DATA = HERE / "internal" / "data"
OUT = HERE.parent / "figures"
NAMES = {
    "dasp_ag_swinunet_aspp_small_rates": "ASPP-AG-SwinUNet (B)",
    "segformer": "SegFormer", "unet": "UNet", "unetplusplus": "UNet++",
    "deeplabv3plus": "DeepLabV3+", "fpn": "FPN", "pan": "PAN",
    "pspnet": "PSPNet", "manet": "MAnet",
}
for font in (HERE / "internal" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({"font.family": "Arimo", "font.size": 8,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def load(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def draw(stem, headings, rows, widths):
    fig, ax = plt.subplots(figsize=(183 / 25.4, (len(rows) * 7 + 15) / 25.4))
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headings, colWidths=widths,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CED6DF")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#E8EFF6")
            cell.get_text().set_weight("bold")
        elif r % 2:
            cell.set_facecolor("#F5F7FA")
        if c == 0:
            cell.get_text().set_ha("left")
    fig.tight_layout(pad=0.3)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}")
    fig.savefig(OUT / f"{stem}_preview.png", dpi=180)
    plt.close(fig)
    with (OUT / f"{stem}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headings)
        w.writerows(rows)


summary = sorted(load("benchmark_summary.csv"), key=lambda r: -float(r["mIoU_mean"]))
metrics = ["mIoU", "mDSC", "mPrecision", "mRecall"]
rows = [[NAMES[r["model"]]] + [
    f'{100 * float(r[m + "_mean"]):.2f} ± {100 * float(r[m + "_sd"]):.2f}'
    for m in metrics] for r in summary]
draw("Internal_benchmark_table", ["Model"] + [m + " (%)" for m in metrics],
     rows, [0.32, 0.17, 0.17, 0.17, 0.17])

summary = load("class_metric_summary.csv")
metrics = ["IoU", "DSC", "Precision", "Recall"]
rows = []
for tissue in ["Background", "Epithelial", "Granulation", "Necrotic", "Pus"]:
    row = [tissue]
    for metric in metrics:
        r = next(r for r in summary if r["class"] == tissue and r["metric"] == metric)
        row.append(f'{float(r["mean_pct"]):.2f} ± {float(r["sample_sd_pct"]):.2f}')
    rows.append(row)
draw("Internal_class_metrics_table", ["Tissue"] + [m + " (%)" for m in metrics],
     rows, [0.24, 0.19, 0.19, 0.19, 0.19])
