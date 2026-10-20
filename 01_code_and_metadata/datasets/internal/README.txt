Internal wound segmentation dataset

351 image/annotation pairs: 286 train, 30 valid and 35 test.
RGB photographs and indexed PNG masks are byte-identical to the training inputs.
The split directories contain images and their *_mask.png annotations together.
All manifest image/mask paths are relative to this directory.
class_mapping.csv: 0 Background, 1 Epithelial, 2 Granulation, 3 Necrotic, 4 Pus; ignore index 255 is reserved by the loader.

The released split was fixed before the six-seed final-B/baseline comparisons. Membership was adjusted using prior test performance: two test images moved to training and two epithelial-positive training images moved to the test directory. selection_audit.json and split_assignment.csv retain the selection rule and movements. This is a development comparison, not an untouched independent holdout.

351 files correspond to 342 unique decoded RGB photographs. The original duplicate policy retains nine duplicate groups, including six train/validation duplicate groups; masks can differ within a duplicate group. duplicate_groups.json records all members and hashes. Animal-level independence is not established by this image split. No image or annotation was edited for this release.

The historical benchmark uses a reference-mask-defined ROI plus its recorded margin, then resizes to 256 x 256. Area measurement uses a separate image-only inference workflow and must not be described as the same evaluation protocol.
