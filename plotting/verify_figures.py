"""Verify source statistics and SVG editability without rerunning inference."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plotting" / "internal" / "data"
OUT = ROOT / "figures"
NS = "{http://www.w3.org/2000/svg}"
CLASSES = ["Background", "Epithelial", "Granulation", "Necrotic", "Pus"]
SEEDS = list(range(1001, 1007))


def read(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


counts = np.zeros((6, 5, 5), dtype=np.int64)
for r in read("confusion_counts_by_seed.csv"):
    counts[SEEDS.index(int(r["seed"])), CLASSES.index(r["true_class"]),
           CLASSES.index(r["predicted_class"])] = int(r["pixels"])
assert np.all(counts.sum(axis=(1, 2)) == 2293760)
assert counts[3].diagonal().tolist() == [1924986, 5026, 255516, 50193, 6449]
normalized = (counts / counts.sum(axis=2, keepdims=True)).mean(axis=0)
for r in read("mean_confusion_pct.csv"):
    assert np.isclose(normalized[CLASSES.index(r["true_class"]),
                                CLASSES.index(r["predicted_class"])] * 100,
                      float(r["mean_row_pct"]), rtol=0, atol=1e-8)
rows = read("benchmark_per_seed.csv")
assert len(rows) == 54
for summary in read("benchmark_summary.csv"):
    group = [r for r in rows if r["model"] == summary["model"]]
    assert sorted(int(r["seed"]) for r in group) == SEEDS
    for m in ["mIoU", "mDSC", "mPrecision", "mRecall"]:
        values = np.array([float(r[m]) for r in group])
        assert np.isclose(values.mean(), float(summary[m + "_mean"]), rtol=0, atol=1e-10)
        assert np.isclose(values.std(ddof=1), float(summary[m + "_sd"]), rtol=0, atol=1e-10)
inventory = []
for p in sorted(OUT.glob("*.svg")):
    svg = ET.parse(p).getroot()
    text = list(svg.iter(NS + "text"))
    images = list(svg.iter(NS + "image"))
    assert len(text) > 0, p.name
    if p.name == "FigureS22_placement_overlays.svg":
        assert len(images) == 5
    else:
        assert not images, f"Numerical figure or diagram rasterized: {p.name}"
    for image in images:
        href = image.get("{http://www.w3.org/1999/xlink}href", image.get("href", ""))
        assert href.startswith("data:image/"), p.name
    inventory.append({
        "svg": p.name, "editable_text_elements": len(text),
        "embedded_raster_images": len(images),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    })
assert len(inventory) == 13
OUT.joinpath("qa").mkdir(exist_ok=True)
report = {
    "status": "passed", "svg_count": len(inventory),
    "source_statistics_checked": True,
    "six_seed_matrix_total_pixels": 2293760,
    "S5_seed": 1004, "S5_diagonal_counts": counts[3].diagonal().tolist(),
    "numerical_figures_have_no_embedded_raster": True,
    "placement_raster_images": 5,
    "figures": inventory,
    "visual_review_record": "visual_review.json",
}
(OUT / "qa" / "figure_checks.json").write_text(json.dumps(report, indent=2) + "\n")
with (OUT / "figure_inventory.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(inventory[0]))
    w.writeheader()
    w.writerows(inventory)
print(f"Passed: {len(inventory)} editable SVGs; source statistics and image embedding verified.")
