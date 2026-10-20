# Photographic overlay display

The GitHub edition includes self-contained photographic SVGs, numerical records and processing code. Separate JPEG/PNG sources and PDF exports remain in the local research package. Rebuilding the photographic panels requires those local source images.

The five supplied overlay composites are presented with the recommended-grid edges parallel to the display frame. The photograph, both grids and their intersection are processed as one raster image. The original overlay values are retained:

| Pair | Date | Animal | Overlay IoU |
|---|---|---|---:|
| 01 | 2026-09-27 | G3-2 | 94.48% |
| 02 | 2026-09-28 | G4-5 | 90.65% |
| 03 | 2026-09-24 | G4-1 | 90.32% |
| 04 | 2026-09-26 | G3-4 | 89.66% |
| 05 | 2026-09-26 | G3-2 | 87.03% |

`exports/` contains the combined plate, two smaller plates, and five individual panels as editable-label SVG, PDF and 600-dpi PNG. `panels/` contains the photographic crops at native resolution. SVG and PDF preserve the photographs as embedded rasters; the photos are not vector drawings. The stated 600 dpi is the figure export resolution, not a claim of increased photographic detail.

## Rebuild

```bash
python -m pip install -r requirements.txt
python build_overlay_display.py
```

## Source and processing record

The inputs are the study's existing composite JPEGs, retained byte-for-byte under `sources/` in the local research package. They already contain the photo, recommended grid, placement grid and intersection shading. The JSON source records identify the `best_original` selections from the existing angle-search outputs. These recorded overlay IoUs are distinct from the baseline registration-only scores in the earlier geometry comparison. This display update does not create a new placement measurement or estimate IoU from interpolated image pixels.

The 296-pixel report header is removed. Each complete photographic composite is rotated once by the negative of its recorded reference-grid angle, using bicubic interpolation, then cropped to a square containing both full grid boundaries and surrounding context. No layer is moved independently, no grid is redrawn, and no nonuniform scaling, perspective change, colour adjustment or local retouching is applied. The full source report remains available in each local original JPEG.

`processing_record.json` records the source hashes, common angles, crop bounds, output hashes, relative-geometry checks and effective source-image resolution. The joint rigid transform preserves relative positions and IoU analytically. Values and original registration checks are also preserved in `data/overlay_values.csv` and `sources/overlay_records.json`. Pairs 01–02 passed the original registration diagnostics; pairs 03–05 remain exploratory, with the two 26 September plan assignments provisional. Five photo pairs represent four animals.

These images illustrate the recorded two-dimensional grid overlays. No physical scale or delivered-dose measurement is inferred from the display crop. Separate unprocessed before/after photos are not included in this package. Source-photo reuse remains subject to the study owners' permissions.
