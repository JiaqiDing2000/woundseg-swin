#!/usr/bin/env python3
"""Reconstruct registered grid footprints and draw their geometric overlap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
FILL = "#DCEAF1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed_area(points):
    p = np.asarray(points, dtype=float)
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                        - p[:, 1] * np.roll(p[:, 0], -1)) / 2)


def area(points):
    return abs(signed_area(points)) if len(points) >= 3 else 0.0


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def intersection(subject, boundary):
    """Clip a convex polygon against each edge of a convex boundary."""
    polygon = np.asarray(subject, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    orientation = 1 if signed_area(boundary) > 0 else -1
    for a, b in zip(boundary, np.roll(boundary, -1, axis=0)):
        if len(polygon) == 0:
            break
        clipped = []
        for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
            fp = orientation * cross(b - a, p - a)
            fq = orientation * cross(b - a, q - a)
            if fp >= -1e-9:
                clipped.append(p)
            if (fp >= 0) != (fq >= 0):
                clipped.append(p + (q - p) * fp / (fp - fq))
        polygon = np.asarray(clipped, dtype=float)
    return polygon


def iou_percent(a, b):
    overlap = area(intersection(a, b))
    return 100 * overlap / (area(a) + area(b) - overlap)


def rotate_image_coordinates(points, angle_degrees, center):
    """Positive angles turn counterclockwise in image coordinates (y down)."""
    angle = np.deg2rad(angle_degrees)
    transform = np.array([[np.cos(angle), np.sin(angle)],
                          [-np.sin(angle), np.cos(angle)]])
    return (np.asarray(points) - center) @ transform.T + center


def restore_footprints(document):
    source = document["source"]
    metrics = document["metrics"]
    selected = source["best_original"]
    centers = np.array([cell["reference_center_px"] for cell in metrics["cells"]])
    placed_selected = np.array(metrics["actual_boundary_registered_px"])

    # v11 uses an integer side length and floor(side / 3) for the cell pitch.
    # Its internal 90-degree orientation is undone in the supplied coordinates.
    side = round(np.sqrt(metrics["footprint"]["reference_area_px2"]))
    residual = (side % 3) / 2
    cell_center_offset = rotate_image_coordinates(
        np.array([-residual, residual]), selected["reference_deg"], np.zeros(2))
    center = centers.mean(axis=0) - cell_center_offset
    if not np.allclose(center, np.round(center), atol=1e-7):
        raise ValueError("Reference grid center is inconsistent with v11 integer bounds")
    center = np.round(center)
    reference = center + side / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    placed_center = placed_selected.mean(axis=0)
    placed = rotate_image_coordinates(
        placed_selected, -selected["setup_deg"], placed_center)
    placed_cells = [rotate_image_coordinates(np.array(cell), -selected["setup_deg"], placed_center)
                    for cell in metrics["actual_cells_registered_px"]]
    reference_selected = rotate_image_coordinates(reference, selected["reference_deg"], center)
    baseline_iou = iou_percent(reference, placed)
    selected_iou = iou_percent(reference_selected, placed_selected)
    baseline_error = baseline_iou - source["observed"]["iou_percent"]
    selected_error = selected_iou - selected["iou_percent"]
    if max(abs(baseline_error), abs(selected_error)) >= 0.0001:
        raise ValueError("Reconstructed geometry does not reproduce the supplied scores")

    return {
        "date": source["date"], "animal": source["group"], "pair": source["job"],
        "registration_pass": source["registration_gate_pass"],
        "registration_reasons": source.get("registration_reasons", []),
        "plan_assignment": "provisional" if source["date"] == "20260926" else "supplied record",
        "reference_boundary_registered_px": reference.tolist(),
        "placed_boundary_registered_px": placed.tolist(),
        "placed_cells_registered_px": [cell.tolist() for cell in placed_cells],
        "reference_center_px": center.tolist(), "reference_side_px": side,
        "iou_percent": baseline_iou,
        "supplied_registered_iou_percent": source["observed"]["iou_percent"],
        "supplied_angle_search_iou_percent": selected["iou_percent"],
        "recomputed_angle_search_iou_percent": selected_iou,
        "baseline_difference_percentage_points": baseline_error,
        "angle_search_difference_percentage_points": selected_error,
        "recorded_reference_angle_deg": selected["reference_deg"],
        "recorded_placed_angle_deg": selected["setup_deg"],
        "H_after_to_before": document["H_after_to_before"],
        "display": {"common_rotation_deg": 0.0, "photo_processing": "none",
                    "geometry_translation_px": (-center).tolist(),
                    "uniform_geometry_scale": 1 / side,
                    "bounds_in_grid_side_units": [-0.65, 0.65, -0.65, 0.65]},
    }


def panel(ax, record, letter):
    center = np.array(record["reference_center_px"])
    side = record["reference_side_px"]
    normalize = lambda p: (np.array(p) - center) / side
    reference = normalize(record["reference_boundary_registered_px"])
    placed = normalize(record["placed_boundary_registered_px"])
    both = np.vstack([reference, placed])
    assert np.all(np.abs(both) < 0.65), "Display frame clips part of a footprint"
    ax.add_patch(Polygon(intersection(reference, placed), closed=True,
                         facecolor=FILL, edgecolor="none", zorder=1))
    pitch = side // 3
    for offset in (pitch, 2 * pitch):
        position = -0.5 + offset / side
        ax.plot([position, position], [-0.5, 0.5], color=BLUE, lw=0.65, alpha=0.7)
        ax.plot([-0.5, 0.5], [-position, -position], color=BLUE, lw=0.65, alpha=0.7)
    for cell in record["placed_cells_registered_px"]:
        ax.add_patch(Polygon(normalize(cell), closed=True, fill=False,
                             edgecolor=ORANGE, lw=0.55, alpha=0.65))
    ax.add_patch(Polygon(reference, closed=True, fill=False, edgecolor=BLUE, lw=1.65, zorder=4))
    ax.add_patch(Polygon(placed, closed=True, fill=False, edgecolor=ORANGE, lw=1.65, zorder=5))
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(0.65, -0.65)
    ax.set_aspect("equal")
    ax.axis("off")
    date = record["date"]
    ax.set_title(f"{letter}   {record['animal']} · {date[:4]}-{date[4:6]}-{date[6:]}",
                 fontsize=8.5, pad=2, loc="left", weight="bold")
    ax.text(0.5, -0.015, f"Grid-footprint IoU  {record['iou_percent']:.2f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5)


def draw(records, output, stem, width_mm, height_mm, columns, caption):
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(width_mm / 25.4, height_mm / 25.4), squeeze=False)
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.13, top=0.92, wspace=0.13, hspace=0.35)
    for i, ax in enumerate(axes.flat):
        if i < len(records):
            panel(ax, records[i], chr(65 + i))
        else:
            ax.axis("off")
    handles = [Line2D([0], [0], color=BLUE, lw=1.8), Line2D([0], [0], color=ORANGE, lw=1.8)]
    fig.legend(handles, ["Recommended grid", "Placed grid inferred from electrode centers"],
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=8)
    paths = []
    for suffix in ("pdf", "svg", "png"):
        path = output / "exports" / f"{stem}.{suffix}"
        fig.savefig(path, dpi=600, facecolor="white", metadata={"Creator": "Wound study"})
        paths.append(path)
    preview = output / "exports" / f"{stem}_preview.png"
    fig.savefig(preview, dpi=180, facecolor="white")
    plt.close(fig)
    (output / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")
    return {"stem": stem, "width_mm": width_mm, "height_mm": height_mm,
            "files": {str(p.relative_to(output)): digest(p) for p in paths},
            "preview": str(preview.relative_to(output))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path(__file__).parent / "sources")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    output = args.output.resolve()
    for directory in (output / "sources", output / "data", output / "exports"):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    records = []
    sources = []
    for index, path in enumerate(sorted(args.sources.glob("*.json")), 1):
        document = json.loads(path.read_text(encoding="utf-8"))
        if "source" not in document or "H_after_to_before" not in document:
            continue
        target = output / "sources" / f"pair_{index:02d}_source.json"
        if path.resolve() != target.resolve():
            shutil.copyfile(path, target)
        record = restore_footprints(document)
        record["source_json"] = str(target.relative_to(output))
        records.append(record)
        sources.append({"file": str(target.relative_to(output)), "sha256": digest(target)})
    if len(records) != 5:
        raise ValueError(f"Expected five supplied pairs; received {len(records)}")
    passed = [record for record in records if record["registration_pass"]]
    exploratory = [record for record in records if not record["registration_pass"]]
    common = ("Blue and orange outlines show the recommended and inferred placed 3 × 3 grid footprints, "
              "respectively; shading marks their intersection. IoU compares the complete footprint regions "
              "after image registration using features outside the device region. The placed footprint is "
              "inferred from the nine electrode centers. Geometry is reconstructed from the supplied coordinate "
              "records and displayed at a common normalized scale.")
    products = [draw(passed, output, "FigR_placement_registered", 180, 95, 2,
                     "Registered grid-footprint comparisons. " + common +
                     " Both selected pairs passed the registration diagnostics; their IoUs are 94.12% and 84.40%."),
                draw(exploratory, output, "FigR_placement_exploratory", 180, 77, 3,
                     "Exploratory grid-footprint comparisons. " + common +
                     " These three selected pairs did not pass all registration diagnostics and are shown for inspection. "
                     "The 26 September treatment-plan assignments are provisional."),
                draw(records, output, "FigR_placement_all_five", 180, 192, 2,
                     "Selected registered grid-footprint comparisons. " + common +
                     " Panels A and B passed the registration diagnostics. Panels C–E are exploratory; "
                     "the 26 September treatment-plan assignments are provisional. Five photo pairs represent "
                     "four animals, with G3-2 observed on two dates.")]
    (output / "data" / "registered_geometry.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    keys = ["date", "animal", "pair", "registration_pass", "plan_assignment", "iou_percent",
            "supplied_registered_iou_percent", "supplied_angle_search_iou_percent",
            "baseline_difference_percentage_points", "angle_search_difference_percentage_points"]
    with (output / "data" / "overlap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: record[key] for key in keys} for record in records)
    manifest = {
        "purpose": "Revised registered grid-footprint geometric comparisons for reviewer correspondence",
        "method": "Existing overlay geometry restored to the independently registered coordinate frame",
        "source_records": sources, "products": products,
        "numerical_checks": {
            "maximum_baseline_difference_percentage_points": max(abs(r["baseline_difference_percentage_points"]) for r in records),
            "maximum_angle_search_difference_percentage_points": max(abs(r["angle_search_difference_percentage_points"]) for r in records),
            "tolerance_percentage_points": 0.0001, "all_checks_passed": True,
            "full_footprints_inside_display": True,
        },
        "rendering": {"font": "DejaVu Sans", "raster_dpi": 600, "svg_editable_text": True},
        "photographs": {"raw_paired_photos_available": False,
                        "existing_composite_photos_retain_original_transform_and_metrics": True,
                        "photographic_panels_in_new_exports": False},
        "visual_review": {"status": "pending"},
        "script_sha256": digest(Path(__file__)),
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["numerical_checks"], indent=2))


if __name__ == "__main__":
    main()
