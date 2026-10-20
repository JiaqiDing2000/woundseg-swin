# WoundSeg-Swin v1.0.0

This release provides the final B model with balanced whole-image sampling, cross-entropy plus Tversky loss, and ASPP dilation rates 1/2/3.

- Six original best checkpoints, seeds 1001–1006.
- Internal dataset: 351 RGB images and indexed masks, with the recorded 286/30/35 split.
- Final B and eight-baseline numerical comparisons over six seeds.
- Revised SVG figures and tables with editable sources.
- Wound-area numerical measurements and figures for the documented 100-image subset.
- Five supplied placement comparisons, geometry and SVG displays.

Code, configuration records, source tables and SVG files are available in the repository. Seven release assets contain the six original checkpoints and `internal_dataset.zip`. Download them and verify their hashes with:

```bash
python3 tools/download_assets.py --repo OWNER/woundseg-swin --tag v1.0.0
python3 tools/verify_release.py
```

Replace `OWNER` with this repository's owner. The ZIP extracts relative to the repository root. Original checkpoints belong in `weights/`. SHA-256 values are recorded in `release_assets_manifest.json`; a separate checksum attachment is not required.

Separate area/placement source photographs, area masks and PDF/XLSX exports are retained locally and are not part of these release assets.

The README and dataset/analysis records specify the benchmark ROI, development split, duplicates, area-subset selection and placement configurations. Numerical reproduction checks are provided in `environment/` and `provenance/`.
