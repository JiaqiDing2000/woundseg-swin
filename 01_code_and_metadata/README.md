# WoundSeg-Swin

Training code, original model checkpoints, internal image annotations, numerical results and editable figures for the final B model: balanced whole-image sampling, cross-entropy plus Tversky loss, and ASPP dilation rates 1/2/3.

## Download

The Git repository contains code, configuration records, data manifests, numerical tables and SVG figures. Release `v1.0.0` provides seven assets: six checkpoints and `internal_dataset.zip`, containing the internal segmentation images and masks.

After cloning this repository, replace `OWNER` with its GitHub owner:

```bash
python3 tools/download_assets.py --repo OWNER/woundseg-swin
python3 tools/verify_release.py
```

For the internal dataset and one checkpoint:

```bash
python3 tools/download_assets.py --repo OWNER/woundseg-swin --groups internal weights --seed 1005
python3 tools/verify_release.py --scope available
```

The downloader requires Python 3.9 or later, verifies SHA-256 hashes and extracts each archive into its recorded directory. Each original checkpoint is 917,244,231 bytes; all six total 5.13 GiB. `release_assets_manifest.json` records download groups, filenames, sizes and hashes. When downloading assets through a browser, place the checkpoints in `weights/` and extract ZIP files at the repository root.

Area and placement numerical records, plotting code and SVG displays are included in Git. Their separate source photographs, area masks and companion PDF/XLSX files are retained in the local research package and are not distributed as release assets. Photographic SVGs contain their display images.

## Contents

| Directory | Contents |
|---|---|
| `src/` | Final B model, training, benchmark evaluation and image-only inference |
| `configs/final_b.json` | Recorded model and training settings |
| `datasets/internal/` | 351 image/mask pairs, split membership, class mapping and selection records |
| `weights/` | Original best checkpoints for seeds 1001–1006; hashes and metadata |
| `results/` | Six-seed Final B and eight-baseline results, class metrics and confusion counts |
| `figures/` | Replacement SVG figures and tables, numerical inputs and replacement index |
| `plotting/` | Editable figure specifications, numerical inputs and rendering scripts |
| `analysis/wound_area_B_100/` | Manual/automated measurements for 100 observations, selection records and plotting code |
| `analysis/placement/` | Five placement comparisons, geometric records and self-contained SVG displays |
| `environment/` | Runtime versions, source provenance and numerical reproduction checks |
| `provenance/` | Data/checkpoint source hashes and validation records |

Training, evaluation and inference commands are provided in [CODE_USAGE.txt](CODE_USAGE.txt). Exact environment versions and verification records accompany the code.

## Run the model

Python 3.10 was used for the recorded runs. Install PyTorch separately, then the remaining dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

For CPU inference, use the same PyTorch and torchvision versions from `https://download.pytorch.org/whl/cpu`. Full training requires a CUDA GPU.

Evaluate one released checkpoint:

```bash
python src/evaluate.py --checkpoint weights/seed_1005.pth --split test --device cpu --output outputs/seed_1005_test.json
```

Run image-only segmentation:

```bash
python src/predict.py --image example.jpg --checkpoint weights/seed_1005.pth --mode area --output outputs/example_mask.png --device cuda
```

Train the recorded model:

```bash
python src/train.py --seed 1005 --output outputs/retraining --workers 8
```

The label IDs are 0 Background, 1 Epithelial, 2 Granulation, 3 Necrotic and 4 Pus. The training recipe uses 40% uniform image sampling and 20% each from granulation-, necrotic- and pus-positive image pools; the complete loss weights, thresholds and optimizer settings are recorded in `configs/final_b.json`. ASPP rates must be set to 1/2/3 when constructing the model because dilation attributes are not stored in checkpoint tensors.

## Results and evaluation protocols

The six-seed final B comparison gives mIoU **73.50 ± 3.11%** and mDSC **82.16 ± 3.01%** (mean ± sample SD). `results/per_seed_results.csv` contains all 54 model/seed combinations for final B and eight baselines. Metrics in the source CSV and JSON files are fractions; multiply by 100 for percentages.

The internal benchmark uses 286/30/35 train/validation/test-directory records. The 351 records contain 342 unique decoded RGB photographs; duplicate groups and the development-stage split adjustments are documented in `datasets/internal/`. The benchmark uses the reference mask to define an expanded square ROI and resizes it to 256 pixels. Its results describe that recorded development comparison, not an untouched independent or animal-level holdout.

The wound-area analysis uses a separate image-only workflow with seed 1005, chosen by validation mIoU. It combines global and tiled probabilities, applies the recorded orientation rule and restores predictions to native image coordinates. The 100-image area subset was selected from 168 matched observations by absolute area error; its selection and all manual measurements are retained in the analysis provenance. Its statistics describe that subset.

Placement assets preserve the five supplied comparisons and their original scores. Independently registered geometry and the supplied photographic overlay configurations are identified separately in the placement README and source records. These are two-dimensional placement comparisons, not measurements of delivered tissue dose.

## Reproduce analyses and figures

Verify the downloaded files before rebuilding. Regenerating PDF/XLSX files or figure metadata can change file hashes even when the numerical results remain unchanged.

```bash
python -m pip install -r analysis/wound_area_B_100/requirements.txt
python analysis/wound_area_B_100/code/reproduce_area.py --plots-only
python -m pip install -r plotting/requirements.txt
python plotting/rebuild_figures.py
```

The default figure rebuild uses published numerical inputs and retains the supplied photographic SVG. Rebuilding photographic panels and checking source-image/mask hashes require the complete local research package; see `plotting/README.md`.

Consult [the figure replacement guide](figures/FIGURE_REPLACEMENT_GUIDE.md), [the machine-readable figure inventory](figures/figure_inventory.csv), and the analysis READMEs for individual outputs. Quantitative plots and diagrams use vector geometry. Original photographs remain embedded raster images in photographic SVG layouts.

## Release checks

The packaged seed-1005 checkpoint reproduces all recorded overall and per-class metrics on the 35-image benchmark exactly. Training ROI statistics and sampling probabilities match the recorded values for all 286 training images. The published loss values and gradients match the historical implementation. An image-only area inference check reproduces the original predicted-mask hash, foreground pixel count and calibrated area.

All six original checkpoint files and all internal image/mask files match their source hashes. `MANIFEST.sha256` covers the files distributed through Git and the seven release assets, including the extracted internal dataset. Numerical comparisons and figure source records are included. Packaging did not launch full retraining.

The repository provides the internal animal images used for the recorded segmentation comparison. Public human benchmark data retain their original distribution and licensing; they are not redistributed in the internal-dataset archive.
