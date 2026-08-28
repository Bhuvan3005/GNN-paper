# -*- coding: utf-8 -*-
"""
run_conv_comparison.py
======================
Ablation over the message-passing operator within the feature-node
formulation: GCN (ours) vs GAT vs GraphSAGE vs GIN.

Everything except the convolution is held fixed -- graph construction,
depth, hidden width, dropout, readout, optimizer, early stopping,
threshold tuning, folds and seeds. This answers the standard reviewer
question "why GCN rather than an attention-based or more expressive
operator?" without altering the research methodology.

Reported per operator: mean +/- std over 3 seeds x 5 folds, trainable
parameter count, and DeLong / McNemar tests against GCN on pooled
out-of-fold predictions.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from pipeline import load_data, TrainConfig, _aggregate
from conv_variants import ConvConfig, run_conv_cv, count_params
from run_external_stats import delong_roc_test

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123]
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC", "Specificity"]
OPERATORS = [
    ("GCN (Ours)", "gcn"),
    ("GAT", "gat"),
    ("GraphSAGE", "sage"),
    ("GIN", "gin"),
]


def main():
    X, y = load_data()
    print("=" * 78)
    print("GRAPH CONVOLUTION ABLATION (feature-node formulation)")
    print(f"Seeds {SEEDS} x 5-fold CV; only the operator varies")
    print("=" * 78)

    rows, oof_store, summaries = [], {}, {}
    for label, kind in OPERATORS:
        cfg = ConvConfig(conv=kind)
        folds, probs, preds, true_ref = [], {}, {}, None
        for sd in SEEDS:
            s, f, prob, pred, true = run_conv_cv(
                X, y, cfg, TrainConfig(), n_splits=5, seed=sd, return_oof=True)
            folds.extend(f)
            probs[sd], preds[sd] = prob, pred
            true_ref = true
        summ = _aggregate(folds)
        summaries[label] = summ
        oof_store[label] = (probs, preds, true_ref)
        n_par = count_params(cfg)
        rows.append({"Operator": label, "Params": n_par,
                     **{m: f"{summ[m][0]:.4f} ± {summ[m][1]:.4f}" for m in METRIC_KEYS}})
        print(f"  {label:<12} params={n_par:<6} Acc={summ['Accuracy'][0]:.4f}  "
              f"F1={summ['F1'][0]:.4f}  AUC={summ['ROC-AUC'][0]:.4f}  "
              f"MCC={summ['MCC'][0]:.4f}")

    table = pd.DataFrame(rows)
    table.to_csv("results/table_conv_comparison.csv", index=False)

    # ---- statistics vs GCN ----
    stat_rows = []
    g_probs, g_preds, true = oof_store["GCN (Ours)"]
    for label, _ in OPERATORS[1:]:
        o_probs, o_preds, _ = oof_store[label]
        dl_ps, mc_ps, d_aucs = [], [], []
        for sd in SEEDS:
            auc_g, auc_o, p = delong_roc_test(true, g_probs[sd], o_probs[sd])
            dl_ps.append(float(p)); d_aucs.append(float(auc_g - auc_o))
            gc = (g_preds[sd] == true).astype(int)
            oc = (o_preds[sd] == true).astype(int)
            b = int(np.sum((gc == 0) & (oc == 1)))
            c = int(np.sum((gc == 1) & (oc == 0)))
            tbl = [[int(np.sum((gc == 1) & (oc == 1))), c],
                   [b, int(np.sum((gc == 0) & (oc == 0)))]]
            mc_ps.append(float(mcnemar(tbl, exact=(b + c) < 25).pvalue))
        stat_rows.append({
            "Comparison": f"GCN vs {label}",
            "Mean ΔAUC (GCN−other)": round(float(np.mean(d_aucs)), 4),
            "DeLong p (median)": f"{float(np.median(dl_ps)):.4g}",
            "McNemar p (median)": f"{float(np.median(mc_ps)):.4g}",
        })
    stats = pd.DataFrame(stat_rows)
    stats.to_csv("results/table_conv_stats.csv", index=False)

    print("\n" + "=" * 78)
    print("OPERATOR COMPARISON")
    print("=" * 78)
    print(table[["Operator", "Params", "Accuracy", "F1", "ROC-AUC", "MCC"]]
          .to_string(index=False))
    print("\nStatistical comparison against GCN (pooled OOF, median over seeds):")
    print(stats.to_string(index=False))
    print("\nSaved: results/table_conv_comparison.csv, results/table_conv_stats.csv")


if __name__ == "__main__":
    main()
