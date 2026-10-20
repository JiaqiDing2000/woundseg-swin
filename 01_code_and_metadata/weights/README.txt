Final-B data and original checkpoints

Dataset: 351 RGB images + 351 indexed masks; 286/30/35 split; 52,513,735 image/mask bytes.
Weights: six original best checkpoints, 5,503,465,386 bytes total.
Each checkpoint is 917,244,231 bytes and contains model_state, optimizer_state, scheduler_state, epoch and best_miou. The model_state has 570 entries. Use model_state for inference; rebuild ASPP with rates [1, 2, 3].
The original checkpoint payload is retained byte-for-byte, including optimizer and scheduler state; this explains its size. Hashes are in weights/manifest.json and weights/SHA256SUMS.txt. The area workflow uses seed 1005, selected by validation mIoU; this is not a six-model ensemble.

Observed training environment: Python 3.10.12; PyTorch 2.6.0+cu124; CUDA container 12.4.1/cuDNN on Ubuntu 22.04. The code directory supplies the portable dependency description. No training was launched when packaging.

Publication metadata retain the performance-informed split, exact duplicates, six-seed variability and reference-mask-defined benchmark ROI. No independent holdout or animal-level split is asserted. These records allow readers to identify the exact inputs underlying the reported comparisons.

Only selected research inputs, checkpoints and numerical results are packaged. No scheduler credentials, shell history, personal accounts or unrelated files are included. RGB EXIF inventory is in provenance/image_metadata_check.json. Image bytes were not modified, and no inference is made about animal species beyond the source records.
