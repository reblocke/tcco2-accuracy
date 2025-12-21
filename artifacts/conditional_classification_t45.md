# Conditional misclassification curves

Threshold (mmHg): 45.
Bin width: 1 (round).
Bootstrap draws: 1000 per subgroup (seed=202401).
Bootstrap mode: cluster_plus_withinstudy.

Each row corresponds to a PaCO2 bin with empirical count/weight.
TN/FP/FN/TP columns report bootstrap quantiles of conditional probabilities.

Columns: group, threshold, paco2_bin, count, weight,
tn_q025/tn_q50/tn_q975, fp_q025/fp_q50/fp_q975,
fn_q025/fn_q50/fn_q975, tp_q025/tp_q50/tp_q975.