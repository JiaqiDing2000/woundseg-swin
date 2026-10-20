Final B and baseline comparison source results

per_seed_results.csv: 54 rows, 9 models x 6 seeds (1001-1006).
final_comparison_table.csv: arithmetic mean and sample SD (ddof=1) over all six seeds.
Final B is model dasp_ag_swinunet_aspp_small_rates. No other W69/W70/W71 variant or earlier final-model result is included in these comparison tables.
Metric values in the comparison CSVs are fractions (0-1), not percentages.
The evaluation pools pixels over 35 test-directory images, then macro-averages the five classes including Background. All models use the same GT-defined ROI benchmark. Model-training recipes differ as documented in the accompanying code.

final_b/ contains original test/validation result JSON files, provenance and epoch-level training metrics for all six final-B seeds. baselines/ contains the 48 original test result JSON files for the eight baseline models.
confusion_and_class_metrics/ contains exact 5 x 5 count matrices, per-seed class metrics and six-seed summaries. Each matrix was recovered by FP32 checkpoint reinference and checked against all original TP/TN/FP/FN and aggregate values. Rows are reference classes and columns are predicted classes.
Fig. S5 uses the single seed 1004 matrix; Fig. 2H uses the mean of six row-normalized matrices. These are different statistics and should retain their stated labels.
