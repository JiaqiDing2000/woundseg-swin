#!/usr/bin/env python3
"""Present existing photographic overlays in a common grid-aligned frame."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def line_masks(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    r, g, b = rgb.transpose(2, 0, 1)
    blue = (r < 115) & (g > 125) & (b > 180) & (b - r > 85)
    orange = (r > 220) & (g > 125) & (g < 205) & (b < 115) & (r - b > 120)
    return blue, orange


def bbox(mask):
    y, x = np.nonzero(mask)
    if len(x) < 500:
        raise ValueError("Insufficient grid-line pixels to determine the display frame")
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def rotate_mask(mask, angle):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).rotate(
        angle, resample=Image.Resampling.NEAREST, expand=True)) > 0


def aligned_photo(root, record):
    source = root / record["source_image"]
    if sha256(source) != record["source_sha256"]:
        raise ValueError(f"Source image checksum mismatch: {source.name}")
    original = Image.open(source).convert("RGB")
    header = record["source_header_height_px"]
    photo = original.crop((0, header, original.width, original.height))
    blue, orange = line_masks(photo)
    angle = record["display_rotation_deg"]
    # All photograph pixels and baked overlays undergo the same transform.
    turned = photo.convert("RGBA").rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255, 0))
    blue_turned, orange_turned = rotate_mask(blue, angle), rotate_mask(orange, angle)
    bounds = bbox(blue_turned | orange_turned)
    blue_bounds = bbox(blue_turned)
    center = np.array([(blue_bounds[0] + blue_bounds[2]) / 2,
                       (blue_bounds[1] + blue_bounds[3]) / 2])
    span = max(blue_bounds[2] - blue_bounds[0], blue_bounds[3] - blue_bounds[1])
    extent = max(center[0] - bounds[0], bounds[2] - center[0],
                 center[1] - bounds[1], bounds[3] - center[1])
    half = int(np.ceil(extent + 0.045 * span))
    left, top = np.floor(center - half).astype(int)
    box = [int(left), int(top), int(left + 2 * half), int(top + 2 * half)]
    if not (box[0] <= bounds[0] and box[1] <= bounds[1]
            and box[2] >= bounds[2] and box[3] >= bounds[3]):
        raise ValueError("A grid boundary would be cropped")
    cropped = turned.crop(box)
    if np.asarray(cropped)[..., 3].min() != 255:
        raise ValueError("Display crop extends beyond the available photographic field")
    target = root / "panels" / f"{record['pair']}_photo.png"
    cropped.convert("RGB").save(target, dpi=(300, 300))

    rad = np.deg2rad(angle)
    rotation = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    points = np.asarray(record["geometry_points_px"], dtype=float)
    after = points @ rotation.T
    before_distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    after_distances = np.linalg.norm(after[:, None] - after[None, :], axis=2)
    error = float(np.max(np.abs(before_distances - after_distances)))
    if error > 1e-9:
        raise ValueError("Joint display transform did not preserve relative geometry")
    return {
        "pair": record["pair"], "animal": record["animal"], "date": record["date"],
        "iou_percent": record["iou_percent"], "dice_percent": record["dice_percent"],
        "iou_label": f"{record['iou_percent']:.2f}%",
        "source_image": record["source_image"], "source_sha256": sha256(source),
        "source_header_height_px": header,
        "display_rotation_deg": angle,
        "rotated_image_size_px": list(turned.size), "crop_box_after_rotation_px": box,
        "display_rotation_matrix": rotation.tolist(),
        "max_pairwise_distance_change_px": error,
        "all_detected_grid_lines_inside_crop": True,
        "source_photo_covers_entire_crop": True,
        "native_panel_size_px": list(cropped.size),
        "panel_path": str(target.relative_to(root)), "panel_sha256": sha256(target),
        "registration_gate_pass": record["registration_gate_pass"],
    }


def draw_plate(root, records, stem, columns, width_mm, height_mm):
    rows = (len(records) + columns - 1) // columns
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    centered_five = len(records) == 5 and columns == 3
    grid = fig.add_gridspec(rows, columns * 2 if centered_five else columns,
                           left=0.02, right=0.98, top=0.94,
                           bottom=0.155 if len(records) == 1 else 0.085,
                           wspace=0.06, hspace=0.26)
    effective_dpi = []
    for index, record in enumerate(records):
        if centered_five:
            start = 2 * index if index < 3 else 1 + 2 * (index - 3)
            ax = fig.add_subplot(grid[0 if index < 3 else 1, start:start + 2])
        else:
            ax = fig.add_subplot(grid[index // columns, index % columns])
        img = Image.open(root / record["panel_path"])
        ax.imshow(img, interpolation="none")
        ax.set_aspect("equal")
        ax.axis("off")
        day = record["date"]
        prefix = f"{chr(65 + index)}   " if len(records) > 1 else ""
        label = f"{prefix}{record['animal']} · {day[:4]}-{day[4:6]}-{day[6:]}"
        ax.set_title(label, fontsize=7.4, loc="left", pad=4, weight="bold")
        ax.text(0.5, -0.025, f"Overlay IoU  {record['iou_label']}",
                transform=ax.transAxes, va="top", ha="center", fontsize=8.5)
        fig.canvas.draw()
        actual_width_inch = ax.get_position().width * fig.get_figwidth()
        effective_dpi.append(img.width / actual_width_inch)
    for index in range(len(records), len(records) if centered_five else rows * columns):
        ax = fig.add_subplot(grid[index // columns, index % columns])
        ax.axis("off")
    handles = [Line2D([0], [0], color="#00B4FF", lw=1.6),
               Line2D([0], [0], color="#FFA032", lw=1.6)]
    fig.legend(handles, ["Recommended grid", "Placement grid"], ncol=2, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.008),
               fontsize=6.7 if len(records) == 1 else 8)
    exports = {}
    for suffix in ("svg", "pdf", "png"):
        target = root / "exports" / f"{stem}.{suffix}"
        fig.savefig(target, dpi=600, facecolor="white", metadata={"Creator": "Wound study"})
        exports[str(target.relative_to(root))] = sha256(target)
    preview = root / "exports" / f"{stem}_preview.png"
    fig.savefig(preview, dpi=180, facecolor="white")
    plt.close(fig)
    return {"stem": stem, "width_mm": width_mm, "height_mm": height_mm,
            "source_photo_dpi_at_final_size": effective_dpi,
            "exports": exports, "preview": str(preview.relative_to(root))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    for name in ("panels", "exports", "data"):
        (root / name).mkdir(parents=True, exist_ok=True)
    sources = json.loads((root / "sources" / "overlay_records.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    records = [aligned_photo(root, record) for record in sources]
    figures = [draw_plate(root, records, "FigR_photographic_overlays", 3, 180, 137),
               draw_plate(root, records[:2], "FigR_photographic_overlays_AB", 2, 132, 83),
               draw_plate(root, records[2:], "FigR_photographic_overlays_CDE", 3, 180, 76)]
    for record in records:
        stem = f"{record['pair']}_{record['date']}_{record['animal']}_IoU_{record['iou_label'][:-1]}"
        figures.append(draw_plate(root, [record], stem, 1, 64, 76))
    with (root / "data" / "overlay_values.csv").open("w", newline="") as handle:
        fields = ["pair", "date", "animal", "iou_percent", "iou_label", "dice_percent",
                  "registration_gate_pass"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: r[key] for key in fields} for r in records)
    report = {"purpose": "Display-only update of five existing photo overlays",
              "metrics_source": "source.best_original; unchanged from supplied composites",
              "processing": "Remove report header; rotate complete photo composite; square crop",
              "photograph_and_overlay_share_one_transform": True,
              "grid_layers_redrawn": False, "iou_reestimated_from_raster": False,
              "records": records, "figures": figures,
              "source_records_sha256": sha256(root / "sources" / "overlay_records.json"),
              "generator_sha256": sha256(Path(__file__))}
    (root / "processing_record.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(root), "iou_labels": [r["iou_label"] for r in records],
                      "minimum_source_dpi": min(min(f["source_photo_dpi_at_final_size"]) for f in figures),
                      "all_grid_boundaries_preserved": True}, indent=2))


if __name__ == "__main__":
    main()
