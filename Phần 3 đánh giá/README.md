# Phase 3 - Compact evaluation notebook

This folder contains the final compact Phase 3 evaluation notebook: `workflow.ipynb`.

The notebook is evaluation-only. It reads existing Phase 2 `metrics.json`, `history.csv`, `predictions.csv`, and `relaxed_localization_summary.csv` artifacts, then writes selected figures and long error-analysis CSVs under `reports/`.

## Notebook Order

1. Scope + selected evidence table.
2. Stage 1 - GRU baseline and capacity screening.
3. Stage 2 - Attention-GRU narrowing.
4. Classifier diagnosis.
5. Stage 3 - Multi-output bottleneck.
6. Stage 4 - Final confirmation and ablation.
7. Localization and error analysis.
8. Final compact scorecard.
9. Final verification cell.

Small metric tables remain inline in the notebook. Long CSV output is limited to the three files listed below.

## Final Figure Set

The planned figure count is 15, under the assignment maximum of 16.

1. `fig01_gru_capacity_screening.png` - GRU validation/test macro F1 for `S1_G04`, `S1_G05`, `S1_G06`, `S1_G07`.
2. `fig02_gru_learning_curves.png` - GRU learning curves for `S1_G04` and `S1_G06` only.
3. `fig03_attention_test_metrics.png` - Attention-GRU test precision, recall, and F1 for `S2_A01`, `S2_A04`, `S2_A05`, `S2_A10`.
4. `fig04_attention_learning_curves.png` - Attention-GRU learning curves for `S2_A04` and `S2_A10` only.
5. `fig05_classifier_validation_to_test_gap.png` - Validation-to-test macro F1 gap for selected classifier candidates.
6. `fig06_final_classifier_confusion_matrix.png` - Square normalized confusion matrix for `S4_F02`.
7. `fig07_weak_pose_recall.png` - Per-pose recall bars for `S4_F02`.
8. `fig08_stage3_multioutput_bottleneck.png` - Stage 3 composite, cell F1, center score, and center error comparison.
9. `fig09_stage4_final_ablation.png` - Stage 4 final multi-output ablation comparison.
10. `fig10_cell_prediction_quality.png` - Square 5x5 support and accuracy heatmaps for `S4_F06`.
11. `fig11_relaxed_cell_accuracy.png` - Exact, within-1, within-2, and soft cell accuracy for `S4_F06`.
12. `fig12_center_error_distribution.png` - Center error histogram and ECDF for `S4_F06`.
13. `fig13_center_error_vs_cell_distance.png` - Center error against cell grid distance for `S4_F06`.
14. `fig14_error_examples_overview.png` - Compact error overview with top confusion pairs and high-confidence wrong predictions.
15. `fig15_final_model_scorecard.png` - Final classifier and multi-output scorecard.

## CSV Outputs

The notebook writes only these long diagnostic CSV files under `reports/tables/`:

- `classifier_wrong_predictions.csv`
- `cell_wrong_predictions.csv`
- `worst_center_errors.csv`

## Output Folders

```text
Phần 3 đánh giá/reports/figures/
Phần 3 đánh giá/reports/tables/
```

The notebook overwrites the listed output filenames when re-run; it does not delete Phase 1 or Phase 2 artifacts.
