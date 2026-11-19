# Wound-area data

This directory provides the 100 rat wound photographs used in the area analysis, their predicted tissue masks, paired manual/automated measurements and calibration values. The predictions use the rat ASPP-AG-SwinUNet model with seed 1005. **Masks are model predictions; manual reference measurements are the `manual_mm2` values in the table.**

Run the commands below from the repository root, with the [documented environment](../README.md#environment).

## Download and extract

```bash
python download_final_assets.py --group area
python -m zipfile -e Area_images/wound_images.zip Area_images
python -m zipfile -e Area_images/wound_masks.zip Area_images
```

The helper verifies both archives against [final_release_assets.json](../final_release_assets.json). Extraction creates `Area_images/images/` and `Area_images/masks/`. It does not require model weights. To check already downloaded archives:

```bash
python download_final_assets.py --group area --verify-only
```

## Files and columns

| File | Content |
| --- | --- |
| `image_index.csv` | Names and relative paths for 100 photographs; UTF-8 with a byte-order mark (`utf-8-sig`). |
| `measurements.csv` | One row per photograph, including manual/automated areas and the scale used for conversion. |
| `statistics.json` | Regression, error and descriptive difference statistics for these rows. |
| `mask_labels.json` | Integer labels, coordinate convention and area formula. |
| `images/<key>.jpg` | Photograph, after extracting the image archive. |
| `masks/<key>.png` | Corresponding class-ID prediction, after extracting the mask archive. |

Paths in `image_index.csv` and `measurements.csv` are relative to `Area_images/`. `key` identifies a photograph; `animal`, `group` and `day` describe the observation. Area columns ending in `_mm2` are in square millimeters. The measurement fields are:

- `manual_mm2`, `automated_mm2`: paired reference and predicted total wound areas.
- `difference_mm2`: automated minus manual; `absolute_error_mm2` and `absolute_percentage_error_pct`: absolute errors, with the percentage relative to the manual value.
- `wound_pixels`: total foreground pixels, including all tissue classes and connected components.
- `six_block_length_px`, `physical_length_mm`: measured pixel span and corresponding physical span (30 mm).
- `endpoint_x0`, `endpoint_y0`, `endpoint_x1`, `endpoint_y1`: calibration endpoints, with x as column and y as row.
- `epithelial_area_mm2`, `granulation_area_mm2`, `dark_discolored_area_mm2`, `pus_area_mm2`: class-specific predicted areas.

The 8-bit PNG label IDs are 0 = background, 1 = epithelial, 2 = granulation, 3 = dark-discolored tissue, and 4 = pus. The wound is the union of IDs 1–4; a separate binary-mask archive is not needed.

## Coordinates and area calculation

Masks and calibration endpoints align with the photographs **after applying EXIF orientation**, with the origin at the top left. With Pillow, use `ImageOps.exif_transpose`; leave the PNG masks unchanged. The prediction script reads photographs with OpenCV and returns masks to this image orientation.

For a foreground pixel count `N`, calibration length `L_px` and physical length `L_mm`:

```text
area_mm2 = N * (L_mm / L_px)**2
```

The same scale converts each tissue class's pixel count to its area. Do not resize an image or mask without adjusting the calibration scale. The supplied predictions and calibration values already use matching native dimensions.

The following example reads one released mask and checks its area against the measurement row:

```bash
python - <<'PY'
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

root = Path('Area_images')
with (root / 'measurements.csv').open(newline='') as handle:
    row = next(csv.DictReader(handle))
with Image.open(root / row['image_path']) as image:
    image_size = ImageOps.exif_transpose(image).size
with Image.open(root / row['mask_path']) as mask_image:
    mask = np.asarray(mask_image)
assert (mask.shape[1], mask.shape[0]) == image_size
pixels = int(np.count_nonzero((mask >= 1) & (mask <= 4)))
area = pixels * (float(row['physical_length_mm']) / float(row['six_block_length_px']))**2
assert pixels == int(row['wound_pixels'])
assert abs(area - float(row['automated_mm2'])) < 1e-9
print(row['key'], 'pixels:', pixels, 'area_mm2:', area)
PY
```

For this first row (`G1-1_D03`), the expected foreground count is 17,653 and the automated area is approximately 477.6099005325 mm². To generate a fresh prediction from its photograph, use the [root quick-start command](../README.md#quick-start-predict-a-rat-wound-mask).

## Reading the reported statistics

The paired table contains 100 images from 21 animals. This is the subset selected from 168 images with the smallest absolute area errors. Its results are descriptive for the selected sample, rather than an independent accuracy estimate. Multiple observations can belong to the same animal.

`ols_r2` is the coefficient of determination of the fitted ordinary least-squares regression, and `pearson_r` is the Pearson correlation. `bias_mm2` is the mean of automated-minus-manual differences. `sample_sd_mm2` uses `ddof=1`; the lower and upper descriptive limits are bias ± 1.96 sample SD, not confidence intervals.

The supplied rows give MAE = 8.5674 mm², RMSE = 9.9811 mm², bias = +2.6342 mm², and descriptive limits of −16.3301 to +21.5986 mm². Full precision is retained in [statistics.json](statistics.json). The area photograph collection is distinct from the [351-pair rat segmentation dataset](../WoundSegSwin_Model/README.md#rat-dataset).
