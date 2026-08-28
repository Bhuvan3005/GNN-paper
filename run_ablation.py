# -*- coding: utf-8 -*-
"""
run_ablation.py
===============
Task 2 (their numbering): redesigned ablation study.

Every experiment answers a scientific question and is mapped to a
specific research contribution. All configurations are evaluated under
the identical 5-fold Stratified CV protocol, repeated over multiple
seeds; we report mean +/- std pooled across (seed x fold).

Groups
------
A. Graph topology  -> validates the graph-construction contribution
     corr+MST (Ours) | corr-only (no MST) | random | fully-connected | no-graph
B. Network depth   -> validates the 2-layer choice / over-smoothing
     1 layer | 2 layers (Ours)
C. Readout pooling -> validates global mean pooling
     mean (Ours) | max | add
D. Dropout         -> regularization sensitivity
     0.0 | 0.3 (Ours) | 0.5
E. Hidden width    -> capacity sensitivity
     16 | 32 (Ours) | 64
"""

import os
import numpy as np
import pandas as pd

from pipeline import (
    load_data, GCNConfig, TrainConfig, run_gcn_cv, TAU, _aggregate,
)

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123]    # 3 seeds x 5 folds = 15 runs/config (pooled mean±std)
N_SPLITS = 5
KEYS = ["Accuracy", "F1", "ROC-AUC", "MCC", "Recall", "Specificity"]

# (label, question, contribution, topology, GCNConfig kwargs)
EXPERIMENTS = [
    # --- A. Graph topology ---
    ("Full (Corr+MST)", "Does the full pipeline work?", "C1: feature-graph",
     "corr_mst", {}),
    ("w/o MST (Corr only)", "Do MST bridges add connectivity value?", "C1: MST",
     "corr_only", {}),
    ("Random graph", "Does the LEARNED topology matter vs. random edges?", "C1: topology",
     "random", {}),
    ("Fully connected", "Is SELECTIVE construction better than all-edges?", "C1: selectivity",
     "fully_connected", {}),
    ("No graph (indep. nodes)", "Does message passing help at all?", "C1: message-passing",
     "none", {}),
    # --- B. Depth ---
    ("1 GCN layer", "Is a second layer justified (over-smoothing)?", "C2: depth",
     "corr_mst", {"n_layers": 1}),
    # 2 layers == Full (reference)
    # --- C. Pooling ---
    ("Max pooling", "Is mean pooling the right readout?", "C3: readout",
     "corr_mst", {"pool": "max"}),
    ("Add pooling", "Is mean pooling the right readout?", "C3: readout",
     "corr_mst", {"pool": "add"}),
    # --- D. Dropout ---
    ("Dropout 0.0", "Is regularization necessary?", "C4: regularization",
     "corr_mst", {"dropout": 0.0}),
    ("Dropout 0.5", "Is heavier dropout better?", "C4: regularization",
     "corr_mst", {"dropout": 0.5}),
    # --- E. Hidden width ---
    ("Hidden 16", "Capacity sensitivity (smaller).", "C5: capacity",
     "corr_mst", {"hidden": 16}),
    ("Hidden 64", "Capacity sensitivity (larger).", "C5: capacity",
     "corr_mst", {"hidden": 64}),
]

# rows that ARE the full model under each group heading (for readability)
GROUP_REFERENCE = {
    "2 GCN layers (Ours)": ("C2: depth", "corr_mst", {}),
    "Mean pooling (Ours)": ("C3: readout", "corr_mst", {}),
    "Dropout 0.3 (Ours)": ("C4: regularization", "corr_mst", {}),
    "Hidden 32 (Ours)": ("C5: capacity", "corr_mst", {}),
}


def run_config(topology, cfg_kwargs):
    """Pool fold metrics across all seeds -> robust mean/std."""
    all_folds = []
    for seed in SEEDS:
        gcfg = GCNConfig(**cfg_kwargs)
        _, folds = run_gcn_cv(
            *DATA, gcfg=gcfg, tcfg=TrainConfig(),
            topology=topology, threshold=TAU,
            n_splits=N_SPLITS, seed=seed, tune_threshold=True)
        all_folds.extend(folds)
    return _aggregate(all_folds)


def main():
    global DATA
    DATA = load_data()
    print(f"Ablation over seeds={SEEDS}, {N_SPLITS}-fold CV "
          f"({len(SEEDS) * N_SPLITS} runs per config)")
    print("=" * 80)

    rows = []
    # Full model computed once and reused as the reference for later groups
    full_summary = run_config("corr_mst", {})
    full_row = {"Experiment": "Full (Corr+MST)", "Question": EXPERIMENTS[0][1],
                "Contribution": EXPERIMENTS[0][2]}
    full_row.update({k: f"{full_summary[k][0]:.4f} ± {full_summary[k][1]:.4f}" for k in KEYS})
    rows.append(full_row)
    print(f"{'Full (Corr+MST)':<26} "
          f"F1={full_summary['F1'][0]:.4f}  AUC={full_summary['ROC-AUC'][0]:.4f}  "
          f"MCC={full_summary['MCC'][0]:.4f}")

    summaries = {"Full (Corr+MST)": full_summary}
    for label, question, contrib, topo, kw in EXPERIMENTS[1:]:
        s = run_config(topo, kw)
        summaries[label] = s
        row = {"Experiment": label, "Question": question, "Contribution": contrib}
        row.update({k: f"{s[k][0]:.4f} ± {s[k][1]:.4f}" for k in KEYS})
        rows.append(row)
        print(f"{label:<26} F1={s['F1'][0]:.4f}  AUC={s['ROC-AUC'][0]:.4f}  "
              f"MCC={s['MCC'][0]:.4f}")

    abl = pd.DataFrame(rows)
    abl.to_csv("results/table_ablation.csv", index=False)

    # delta-from-full table (using pooled means)
    delta_rows = []
    full_means = {k: full_summary[k][0] for k in KEYS}
    for label, s in summaries.items():
        if label == "Full (Corr+MST)":
            continue
        d = {"Experiment": label}
        d.update({k: round(s[k][0] - full_means[k], 4) for k in KEYS})
        delta_rows.append(d)
    delta = pd.DataFrame(delta_rows)
    delta.to_csv("results/table_ablation_delta.csv", index=False)

    print("\nABLATION (mean ± std, pooled over seed × fold)")
    print(abl[["Experiment", "F1", "ROC-AUC", "MCC"]].to_string(index=False))
    print("\nDelta from Full model:")
    print(delta[["Experiment", "F1", "ROC-AUC", "MCC"]].to_string(index=False))
    print("\nSaved: results/table_ablation.csv, results/table_ablation_delta.csv")


if __name__ == "__main__":
    main()
