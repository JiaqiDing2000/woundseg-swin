"""Verify released image identities, mask-derived areas and summary statistics."""

import csv
import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path, PurePosixPath

import numpy as np
import openpyxl
from PIL import Image, ImageOps

from reproduce_area import metrics


ROOT = Path(__file__).resolve().parents[1]
TISSUES = ("epithelial", "granulation", "necrotic", "pus")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def natural_key(row):
    match = re.fullmatch(r"G(\d+)-(\d+)", row["animal"])
    assert match is not None
    return int(match[1]), int(match[2]), int(row["day"])


def local_file(value):
    path = PurePosixPath(value)
    assert not path.is_absolute() and ".." not in path.parts, value
    target = ROOT.joinpath(*path.parts).resolve()
    assert target.is_relative_to(ROOT) and target.is_file(), value
    return target


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-10), (actual, expected)


def main():
    rows = read_csv(ROOT / "data/measurements.csv")
    manual = {row["key"]: row for row in read_csv(ROOT / "data/manual_reference.csv")}
    mapping = {row["key"]: row for row in read_csv(ROOT / "provenance/source_mapping.csv")}
    source = json.loads((ROOT / "provenance/source_manifest.json").read_text())
    selection = json.loads((ROOT / "provenance/selection_manifest.json").read_text())
    statistics = json.loads((ROOT / "statistics.json").read_text())
    expected_statistics = json.loads((ROOT / "provenance/expected_statistics.json").read_text())
    assert len(rows) == 100 and len({row["key"] for row in rows}) == 100
    assert len({row["animal"] for row in rows}) == 21
    assert rows == sorted(rows, key=natural_key)
    assert set(manual) == set(mapping) == {row["key"] for row in rows} == set(selection["membership"])
    assert selection["candidate_observations"] == 168 and selection["selected_observations"] == 100
    for relative, field in (("data/measurements.csv", "measurements_sha256"),
                            ("data/manual_reference.csv", "manual_reference_sha256"),
                            ("provenance/source_mapping.csv", "source_mapping_sha256")):
        assert sha256(ROOT / relative) == source[field]
    checked = []
    for row in rows:
        key = row["key"]
        assert Decimal(row["manual_mm2"]) == Decimal(manual[key]["manual_mm2"])
        assert row["manual_mm2"] == mapping[key]["manual_mm2_source_text"]
        assert row["animal"] == manual[key]["animal"] and int(row["day"]) == int(manual[key]["day"])
        files = {}
        for kind in ("image", "mask", "overlay"):
            files[kind] = local_file(row[kind + "_path"])
            assert row[kind + "_path"] == mapping[key][kind + "_path"]
            digest = sha256(files[kind])
            assert digest == row[kind + "_sha256"] == mapping[key][kind + "_source_sha256"]
        with Image.open(files["image"]) as image:
            oriented = ImageOps.exif_transpose(image)
            shape = oriented.height, oriented.width
        with Image.open(files["mask"]) as image:
            labels = np.asarray(image)
        assert shape == (int(row["image_height"]), int(row["image_width"])) == labels.shape
        assert labels.ndim == 2 and np.isin(labels, np.arange(5)).all()
        counts = np.bincount(labels.ravel(), minlength=5)
        wound_pixels = int(counts[1:].sum())
        assert wound_pixels == int(row["wound_pixels"])
        assert float(row["physical_length_mm"]) == 30
        length = math.hypot(float(row["endpoint_x1"]) - float(row["endpoint_x0"]),
                            float(row["endpoint_y1"]) - float(row["endpoint_y0"]))
        close(length, row["six_block_length_px"])
        factor = (30 / float(row["six_block_length_px"])) ** 2
        for class_id, tissue in enumerate(TISSUES, 1):
            assert int(counts[class_id]) == int(row[tissue + "_pixels"])
            close(int(counts[class_id]) * factor, row[tissue + "_area_mm2"])
        close(wound_pixels * factor, row["automated_mm2"])
        close(sum(float(row[tissue + "_area_mm2"]) for tissue in TISSUES), row["automated_mm2"])
        difference = float(row["automated_mm2"]) - float(row["manual_mm2"])
        close(difference, row["difference_mm2"])
        close(abs(difference), row["absolute_error_mm2"])
        close(abs(difference) / float(row["manual_mm2"]) * 100, row["absolute_percentage_error_pct"])
        checked.append({"key": key, "native_shape": list(shape), "wound_pixels": wound_pixels,
                        "all_asset_hashes_match": True, "manual_area_unchanged": True})
    calculated = metrics(np.array([float(row["manual_mm2"]) for row in rows]),
                         np.array([float(row["automated_mm2"]) for row in rows]))
    for key, value in calculated.items():
        close(value, statistics["metrics"][key])
        close(value, expected_statistics["metrics"][key])
    close(max(float(row["absolute_error_mm2"]) for row in rows), selection["maximum_selected_absolute_error_mm2"])
    assert statistics["n_images"] == expected_statistics["n_images"] == 100
    assert statistics["n_animals"] == expected_statistics["n_animals"] == 21
    workbook = openpyxl.load_workbook(ROOT / "data/measurements.xlsx", read_only=True, data_only=True)
    sheet = workbook["Measurements"]
    data = list(sheet.iter_rows(values_only=True))
    headers = data[0]
    assert len(data) == 101
    for row, values in zip(rows, data[1:]):
        for name, value in zip(headers, values):
            if isinstance(value, (float, int)):
                close(value, row[name])
            else:
                assert value == row[name]
    workbook.close()
    for folder in ("images", "masks", "overlays"):
        assert len([path for path in (ROOT / folder).iterdir() if path.is_file()]) == 100
    figure_checks = json.loads((ROOT / "figure_checks.json").read_text())
    assert figure_checks["points_per_plot"] == 100
    for entry in figure_checks["exports"]:
        assert sha256(local_file(entry["path"])) == entry["sha256"]
    result = {
        "status": "passed", "images": len(rows), "animals": 21,
        "checks": ["asset SHA-256", "native mask dimensions and class labels", "per-class pixel counts",
                   "30-mm calibration endpoints", "mask-derived tissue and wound areas", "unchanged manual reference",
                   "recorded and recalculated statistics", "spreadsheet values", "natural animal/day order",
                   "relative paths", "figure export hashes"],
        "statistics": calculated, "image_checks": checked,
        "scope": "Reproduction of the released 100-image selected subset; candidate-set selection history is recorded in provenance."
    }
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "images": len(rows), "animals": 21, "metrics": calculated}, indent=2))


if __name__ == "__main__":
    main()
