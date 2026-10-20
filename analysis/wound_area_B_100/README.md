# Wound area: 100 selected images

The local research package contains 100 wound photographs, manual area measurements, native-resolution predicted masks, display overlays, and code to reproduce area conversion, descriptive statistics and figures. The GitHub edition distributes numerical CSV/JSON records, SVG figures and code. Separate photographs, masks, overlays and PDF/XLSX exports remain local.

## Contents

- `data/measurements.csv` (public) and `.xlsx` (local): one row per photograph, with manual and automated areas in mm², four tissue areas and pixel counts, 30-mm calibration endpoints, and relative file paths.
- `images/` (local): original photograph bytes, including original EXIF metadata.
- `masks/` (local): native-resolution label PNGs. Labels 0–4 are background, epithelial, granulation, necrotic and pus, respectively.
- `overlays/` (local): display previews. Tissue colors are blended over the photograph; the cyan segment marks the measured 30-mm span.
- `figures/`: public editable SVG; vector PDF, 600-dpi PNG, previews and the 100-overlay review PDF are local exports.
- `statistics.json` and `figure_caption.txt`: computed statistics and figure caption.
- `code/`: portable area, plotting and verification scripts.
- `provenance/`: selection method, source hashes, unmodified manual reference values and per-image source mapping.

## Methods and selection

The final model uses balanced image sampling, cross-entropy plus Tversky loss, and ASPP dilation rates 1/2/3. Seed 1005 was chosen by validation mIoU before area analysis. The companion repository provides its checkpoint at `../../weights/seed_1005.pth` and training/inference implementations under `../../src/`; see `../../CODE_USAGE.txt` for image-only inference commands. The released masks are sufficient to reproduce all area calculations in this subpackage.

The model was applied to 168 photographs with matched manual measurements and verified calibration. The 100 photographs with the lowest absolute automated-minus-manual area error were selected; ties were resolved by animal identifier and day. Statistics describe this error-selected subset, which includes repeated observations from 21 animals.

For each photograph, wound pixels are all non-background pixels in the predicted mask. Area is `wound_pixels × (30 / six_block_length_px)²`. The calibration length is the distance between the released endpoints in EXIF-oriented native-image coordinates. Four tissue areas use the same conversion. The model's masks were restored to these coordinates before saving; original photograph bytes are unchanged.

Scatter plots show automated versus manual area. Regression R² is the square of Pearson correlation for an ordinary least-squares fit with an intercept. `prediction_r2` additionally compares unadjusted automated areas with manual measurements and is reported separately. Difference plots show automated minus manual area against their mean. The mean ±1.96 sample SD lines describe the selected observations; they are not confidence intervals and do not account for repeated observations within animals.

## Reproduction

Use Python 3.9 or later in an environment with the listed dependencies:

```bash
python -m pip install -r requirements.txt
python code/reproduce_area.py --plots-only
```

This command reproduces statistics and area plots from the public CSV without photographs or masks. It writes a separate plots-only check record. All paths are relative to this directory.

With the complete local research package, `python code/reproduce_area.py` also generates the workbook and photographic review. `python code/verify_package.py` checks photograph/mask/overlay hashes, mask dimensions and pixel counts, calibration, manual reference values, workbooks and statistics. These two complete-package commands require the local binary files.

`SHA256SUMS` inventories the complete local subpackage. Use the repository-root `tools/verify_release.py` to verify publicly distributed files.

The image-selection criterion is recorded in `provenance/selection_manifest.json`; display order is natural animal/day order. Source mapping uses file hashes and relative package paths, without storage-system paths. No inference or pixel editing is performed by the reproduction scripts.
