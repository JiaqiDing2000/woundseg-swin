# Model usage

Install the environment described in the [repository README](../README.md). **Run every command below from the repository root**, not from this subdirectory.

## Labels and configurations

| Class ID | Rat (`configs/rat.json`) | Human (`configs/human.json`) |
| --- | --- | --- |
| 0 | Background | Background |
| 1 | Epithelial | Fibrin |
| 2 | Granulation | Granulation |
| 3 | Dark-discolored tissue | Callus |
| 4 | Pus | — |

Rat class 3 uses the stored key `Necrotic` in code and result files; its manuscript display name is **Dark-discolored tissue**, as recorded in [label_map.csv](datasets/rat/label_map.csv). Training annotations use `255` for ignored pixels. Predictions are single-channel PNGs containing class IDs.

Use matching configurations and weights. The rat model has five output classes and ASPP rates 1/2/3; the human model has four classes and ASPP rates 6/12/18. The full recipes are in [rat.json](configs/rat.json) and [human.json](configs/human.json).

## Dataset preparation

Training and evaluation verify image and mask paths, pairings, membership and SHA256 hashes against the dataset manifests before running. Use the declared files and directory layout. Prediction on a new photograph does not require dataset annotations or a dataset manifest.

### Rat dataset

The released rat dataset contains 351 image–mask pairs: 286 training, 30 validation and 35 evaluation pairs. Download and extract the archive from the repository root:

```bash
mkdir -p data/rat
curl --fail --location \
  https://github.com/JiaqiDing2000/woundseg-swin/releases/download/v1.0.0/internal_dataset.zip \
  --output data/internal_dataset.zip
echo 'a057b71f3e753b3871316f02ba4fd8c0f80b4154db6ad4749bc0bff9cfa86d18  data/internal_dataset.zip' | sha256sum --check
python -m zipfile -e data/internal_dataset.zip data/rat
```

The dataset argument is then `data/rat/datasets/internal`, which contains `train/`, `valid/` and `test/`. The archive's name and extraction root are intentional. Its download metadata and expected file hashes are in [dataset_manifest.json](datasets/rat/dataset_manifest.json).

The recorded rat partition is a development evaluation split whose membership was informed by prior performance; it is not an independent test cohort. Benchmark preprocessing uses the annotation-derived wound bounding box, expanded by 40% and constrained to a square within the image, followed by resizing/padding to 256 pixels. This evaluation domain differs from annotation-free, native-image prediction.

### Human dataset

Obtain DFUTissue images and annotations from the [original provider](https://github.com/uwm-bigdata/DFUTissueSegNet). Human photographs and annotations are not redistributed in this repository. The [dataset publication](https://arxiv.org/abs/2406.16012) describes their source.

Prepare the four-class annotation files listed in [dataset_manifest.json](datasets/human/dataset_manifest.json), using this layout:

```text
data/DFUTissue/
└── Original/
    ├── Images/
    │   ├── TrainVal/
    │   └── Test/
    └── Annotations/
        ├── TrainVal/
        └── Test/
```

Each annotation uses the image's filename stem, with a `.png` extension and integer IDs from the label table above. Use the supplied manifest to check the exact filenames and file contents; filenames alone are insufficient. The loader expects these prepared annotations and does not convert color-coded or eight-class masks automatically.

`TrainVal` contains 94 images. Sorting filenames lexicographically assigns the first 80 to training and the final 14 to validation; `Test` contains 16 images. The [split manifest](datasets/human/split_manifest.csv) lists the expected membership. Human preprocessing resizes the longest side to 256 pixels and pads to 256×256. Dataset provenance and the scope of split verification are recorded in the dataset manifest.

## Train

The shared training entry point selects the dataset-specific recipe from `--config`. Training requires CUDA and uses FP32. Run one seed:

```bash
python WoundSegSwin_Model/src/train.py \
  --config WoundSegSwin_Model/configs/rat.json \
  --dataset data/rat/datasets/internal \
  --seed 1005 --workers 8 --output outputs/rat_training
```

For the human model:

```bash
python WoundSegSwin_Model/src/train.py \
  --config WoundSegSwin_Model/configs/human.json \
  --dataset data/DFUTissue \
  --seed 1001 --workers 8 --output outputs/human_training
```

To train all six rat seeds sequentially:

```bash
for seed in 1001 1002 1003 1004 1005 1006; do
  python WoundSegSwin_Model/src/train.py \
    --config WoundSegSwin_Model/configs/rat.json \
    --dataset data/rat/datasets/internal \
    --seed "$seed" --workers 8 --output outputs/rat_training
done
```

Use the human configuration, human dataset path and a separate output directory for six human runs. Both recipes use a batch size of 4, two gradient-accumulation steps, AdamW, a five-epoch warmup, cosine scheduling, up to 100 epochs, and early stopping after 20 epochs without improved validation mIoU. The highest validation mIoU selects the checkpoint. Rat training uses weighted sampling and CE–Tversky loss; human training uses shuffled sampling and CE–Dice–Focal loss.

The output directory receives `checkpoints/`, `results/`, `logs/` and `figures/`, organized by model and seed. If a best checkpoint already exists for a seed, training skips that run. Use a fresh output directory for a new training run. The published `.pth` assets contain model tensors and metadata; optimizer and scheduler states are not included, and the entry point does not offer training resumption from those assets.

## Evaluate released weights

Download the required weights first. For one rat seed:

```bash
python download_final_assets.py --group rat --seed 1005
python WoundSegSwin_Model/src/evaluate.py \
  --config WoundSegSwin_Model/configs/rat.json \
  --dataset data/rat/datasets/internal \
  --checkpoint WoundSegSwin_Model/weights/rat/seed_1005.pth \
  --split test --device cpu \
  --output outputs/rat_seed_1005_test.json
```

For one human seed:

```bash
python download_final_assets.py --group human --seed 1001
python WoundSegSwin_Model/src/evaluate.py \
  --config WoundSegSwin_Model/configs/human.json \
  --dataset data/DFUTissue \
  --checkpoint WoundSegSwin_Model/weights/human/seed_1001.pth \
  --split test --device cpu \
  --output outputs/human_seed_1001_test.json
```

Use `--split valid` for validation, or `--device cuda` for GPU evaluation. Default evaluation batch sizes are 4 for rat and 8 for human. The output JSON includes aggregate metrics, per-class metrics, accumulated and per-image confusion matrices, and the checkpoint checksum. Confusion-matrix rows are ground-truth classes and columns are predicted classes.

For all six rat seeds, download `--group rat` without `--seed`, then repeat the evaluation command with each seed and a distinct output filename. Use the corresponding human configuration and dataset for human evaluations.

## Predict on a new photograph

Rat image inference supports:

- `global`: one full-image, aspect-preserving model input.
- `global-tiles`: full-image prediction combined with native 512-pixel tiles, with 25% overlap and equal global/tile weighting.
- `area` (default for rat): global/tile inference with predicted-foreground orientation adjustment, restoring the mask to the input coordinates. This is the mode used for the released area data.

See the [root quick start](../README.md#quick-start-predict-a-rat-wound-mask) for a runnable rat example. Human inference uses `global` mode:

```bash
python WoundSegSwin_Model/src/predict.py \
  --config WoundSegSwin_Model/configs/human.json \
  --checkpoint WoundSegSwin_Model/weights/human/seed_1001.pth \
  --image /path/to/human_wound.jpg \
  --mode global --device cpu \
  --output outputs/human_prediction.png
```

Replace the example input path with an actual photograph. Outputs are a PNG label mask and an adjacent JSON report. Human labels are selected on the model grid, unpadded, then restored to the input dimensions with nearest-neighbor resizing. Prediction does not use a reference mask. Physical-area reporting requires a measured pixel span supplied as `--ruler-length-px`, with the corresponding length in millimeters supplied as `--ruler-length-mm`.

## Final result tables

Each dataset's results directory contains per-seed JSON records, `per_seed_metrics.csv`, `per_seed_class_metrics.csv`, `final_model_summary.csv` and `final_class_summary.csv`.

| Dataset | mIoU (%) | mDSC (%) | Seeds |
| --- | --- | --- | --- |
| Rat | 73.50 ± 3.11 | 82.16 ± 3.01 | 1001–1006 |
| Human DFUTissue | 74.47 ± 1.35 | 84.70 ± 1.01 | 1001–1006 |

Values above are means ± sample standard deviations over six seeds (`ddof=1`). Machine-readable metric values are fractions unless a unit explicitly says otherwise; multiply by 100 for percentages. Aggregate segmentation metrics include the background class. The summary tables describe six separate models, not an ensemble. A single-seed evaluation should be read against that seed's record, rather than against the six-seed mean.

## Common input problems

| Message or symptom | Check |
| --- | --- |
| Dataset checksum or membership mismatch | Match filenames, image/mask bytes and split folders to the selected dataset manifest; avoid resaving or resizing the released inputs. |
| Checkpoint/configuration mismatch | Use rat weights with `rat.json`, or human weights with `human.json`. |
| Training requires a CUDA GPU | Install a compatible CUDA PyTorch build and run on a GPU machine; CPU is supported for evaluation and prediction. |
| Model file is missing after cloning | Download the Release assets; weights are not stored in the Git checkout. |
| Label PNG looks black | Inspect integer values or apply a display palette; the PNG stores class IDs. |

Display every supported option with `python WoundSegSwin_Model/src/train.py --help`, `evaluate.py --help` or `predict.py --help` using the same full script-path prefix.
