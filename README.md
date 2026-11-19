# WoundSegSwin

Training, evaluation and image inference for the final ASPP-AG-SwinUNet wound segmentation models. The rat and human DFUTissue models are trained separately, with their own label sets, configurations and weights. Six training seeds (1001–1006) are provided for each dataset.

## Start here

| Task | Guide or files |
| --- | --- |
| Install and run an example | Quick start below |
| Prepare datasets, train or evaluate either model | [Model guide](WoundSegSwin_Model/README.md) |
| Use the 100 area-analysis photographs, masks and measurements | [Area data guide](Area_images/README.md) |
| Inspect final segmentation results | [Rat results](WoundSegSwin_Model/results/rat/) · [Human results](WoundSegSwin_Model/results/human/) |
| Download model weights and area archives | [Release v1.1.0](https://github.com/JiaqiDing2000/woundseg-swin/releases/tag/v1.1.0) |

## Environment

Commands in these guides use a Linux/Bash shell and run from the **repository root**. Use Python 3.10. Evaluation and prediction support CPU or CUDA; training requires a CUDA GPU.

```bash
git clone https://github.com/JiaqiDing2000/woundseg-swin.git
cd woundseg-swin
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install **one** PyTorch build. For CPU evaluation and prediction:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

For training or inference on a CUDA-compatible machine, the recorded build uses CUDA 12.4:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining pinned dependencies:

```bash
python -m pip install -r WoundSegSwin_Model/requirements.txt
```

See the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v260) for other supported builds. Package versions are recorded in [runtime_versions.json](WoundSegSwin_Model/environment/runtime_versions.json). Inference and evaluation load the supplied weights directly; the first training run may download the pretrained encoder used for initialization.

## Quick start: predict a rat wound mask

Download one model and the area photographs/masks, then extract the archives:

```bash
python download_final_assets.py --group rat --seed 1005
python download_final_assets.py --group area
python -m zipfile -e Area_images/wound_images.zip Area_images
python -m zipfile -e Area_images/wound_masks.zip Area_images

python WoundSegSwin_Model/src/predict.py \
  --config WoundSegSwin_Model/configs/rat.json \
  --checkpoint WoundSegSwin_Model/weights/rat/seed_1005.pth \
  --image Area_images/images/G1-1_D03.jpg \
  --mode area --device cpu \
  --output outputs/G1-1_D03_pred.png \
  --ruler-length-px 182.3869895082723 --ruler-length-mm 30
```

This writes a native-resolution class-ID PNG and an adjacent JSON file containing pixel counts, the calibrated total area, and file checksums. Use `--device cuda` for GPU inference. The ruler value above is specific to `G1-1_D03.jpg`; use the measured scale for each new photograph. Without a ruler value, the JSON reports pixel counts without a physical area.

The PNG stores integer tissue IDs, so it can appear nearly black in an ordinary image viewer. Its values are defined in the [model guide](WoundSegSwin_Model/README.md#labels-and-configurations).

## Downloads and file integrity

| Resource | Location | Local destination |
| --- | --- | --- |
| Rat weights, seeds 1001–1006 | `v1.1.0`, `rat_seed_*.pth` | `WoundSegSwin_Model/weights/rat/seed_*.pth` |
| Human weights, seeds 1001–1006 | `v1.1.0`, `human_seed_*.pth` | `WoundSegSwin_Model/weights/human/seed_*.pth` |
| 100 area photographs and predicted masks | `v1.1.0`, `area_wound_images.zip` and `area_wound_masks.zip` | `Area_images/wound_images.zip` and `Area_images/wound_masks.zip` |
| Rat segmentation images and annotations | [internal_dataset.zip, v1.0.0](https://github.com/JiaqiDing2000/woundseg-swin/releases/download/v1.0.0/internal_dataset.zip) | See [dataset preparation](WoundSegSwin_Model/README.md#rat-dataset) |
| Human images and annotations | [DFUTissue provider](https://github.com/uwm-bigdata/DFUTissueSegNet) | See [human dataset preparation](WoundSegSwin_Model/README.md#human-dataset) |

The download helper uses [final_release_assets.json](final_release_assets.json) to check file sizes and SHA256 hashes and place assets in the correct directories. It skips matching files and reports an error for mismatched existing files. ZIP archives are downloaded without extraction. The helper downloads weights and area archives; the rat training dataset is a separate download.

```bash
# All 12 weights and both area archives: approximately 3.75 GB in total.
python download_final_assets.py

# Only one human model.
python download_final_assets.py --group human --seed 1001

# Verify an already downloaded file without network access.
python download_final_assets.py --group rat --seed 1005 --verify-only
```

`--group` can be repeated. `--seed` filters weights only; area archives remain selected unless excluded by `--group`. An optional `--destination /path/to/folder` changes where the helper stores assets; adjust subsequent command paths accordingly. Each weight file is approximately 306 MB.

## Contents and support

- `WoundSegSwin_Model/`: model code, configurations, dependency versions, dataset manifests and final results.
- `Area_images/`: image index, paired area measurements, calibration values, mask labels and summary statistics.
- `download_final_assets.py` and `final_release_assets.json`: verified downloads for the published model and area assets.

The [model guide](WoundSegSwin_Model/README.md) explains the evaluation domains and metric conventions. The [area guide](Area_images/README.md) explains the selected-image sample, coordinate system and area calculation.

For usage questions, open a [GitHub issue](https://github.com/JiaqiDing2000/woundseg-swin/issues) with the command, configuration, seed and software versions. When reporting results, record the repository commit and the weight file's SHA256. Credit the [DFUTissue dataset publication](https://arxiv.org/abs/2406.16012) when using that dataset.

## License

Source code and documentation are released under the [MIT License](LICENSE). Third-party libraries and the DFUTissue dataset retain their respective licenses and terms; the code license does not relicense those materials.
