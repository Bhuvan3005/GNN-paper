# -*- coding: utf-8 -*-
"""
run_learnable_graph.py
======================
Prior-regularised learnable topology.

    L = L_cls + lambda1 * ||A - A0||_1 + lambda2 * L_conn

A0 is the fixed Pearson+MST graph, so optimisation STARTS at the paper's
existing topology and refines it. The sweep below validates each penalty
term separately rather than assuming it helps:

  * lambda1 = 0            -> free-form graph (no prior anchor)
  * lambda1 in {0.1,1,10}  -> increasing anchoring to Pearson
  * lambda2 = 0            -> no connectivity guarantee

Outputs
  results/table_learnable_graph.csv        performance of every variant
  results/table_learned_edges.csv          edges most strengthened / pruned
  figures/fig33_learned_vs_prior.png       learned A vs Pearson prior A0
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from pipeline import (
    FEATURES, N_FEATURES, NUMERIC_COLS, TAU, load_data, GCNConfig, TrainConfig,
    run_gcn_cv, _aggregate, build_adaptive_prior, build_complete_pairs,
)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

SEEDS = [42, 7, 123]
KEYS = ["Accuracy", "F1", "ROC-AUC", "MCC", "Recall", "Specificity"]

# (label, topology, kwargs)
VARIANTS = [
    ("Fixed Corr+MST (Ours, reference)", "corr_mst", {}),
    ("Learned, no regularisation (λ1=0, λ2=0)", "corr_mst_learned",
     {"lambda_l1": 0.0, "lambda_conn": 0.0}),
    ("Learned, λ1=0.1, λ2=1", "corr_mst_learned",
     {"lambda_l1": 0.1, "lambda_conn": 1.0}),
    ("Learned, λ1=1, λ2=1", "corr_mst_learned",
     {"lambda_l1": 1.0, "lambda_conn": 1.0}),
    ("Learned, λ1=10, λ2=1", "corr_mst_learned",
     {"lambda_l1": 10.0, "lambda_conn": 1.0}),
    ("Learned, λ1=1, λ2=0 (no connectivity)", "corr_mst_learned",
     {"lambda_l1": 1.0, "lambda_conn": 0.0}),
]


def run_variant(X, y, topology, kwargs, collect=None):
    all_folds = []
    for seed in SEEDS:
        _, folds = run_gcn_cv(
            X, y, gcfg=GCNConfig(**kwargs), tcfg=TrainConfig(),
            topology=topology, threshold=TAU, n_splits=5, seed=seed,
            tune_threshold=True, learned_graphs=collect)
        all_folds.extend(folds)
    return _aggregate(all_folds)


def prior_matrix(X, y, seed=42):
    """Mean Pearson prior A0 across the 5 folds (for comparison plots)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    priors = []
    for tr_idx, _ in skf.split(X, y):
        X_tr, _, _, _ = train_test_split(
            X.iloc[tr_idx], y.iloc[tr_idx], test_size=0.15,
            stratify=y.iloc[tr_idx], random_state=seed)
        X_tr = X_tr.copy()
        sc = MinMaxScaler(); X_tr[NUMERIC_COLS] = sc.fit_transform(X_tr[NUMERIC_COLS])
        _, _, a0, _ = build_adaptive_prior(X_tr, TAU)
        priors.append(a0.numpy())
    return np.mean(np.stack(priors), axis=0)


def main():
    X, y = load_data()
    print(f"Learnable-graph study | seeds={SEEDS} | 5-fold CV")
    print("=" * 78)

    rows, summaries = [], {}
    collected = {}
    for label, topo, kw in VARIANTS:
        bucket = [] if topo == "corr_mst_learned" else None
        s = run_variant(X, y, topo, kw, collect=bucket)
        summaries[label] = s
        if bucket:
            collected[label] = np.stack(bucket)
        row = {"Variant": label}
        row.update({k: f"{s[k][0]:.4f} ± {s[k][1]:.4f}" for k in KEYS})
        rows.append(row)
        print(f"{label:<42} F1={s['F1'][0]:.4f}  AUC={s['ROC-AUC'][0]:.4f}  "
              f"MCC={s['MCC'][0]:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/table_learnable_graph.csv", index=False)

    # ---- deltas vs the fixed-graph reference ----
    ref = summaries["Fixed Corr+MST (Ours, reference)"]
    drows = []
    for label, s in summaries.items():
        if label.startswith("Fixed"):
            continue
        drows.append({"Variant": label,
                      **{k: round(s[k][0] - ref[k][0], 4) for k in KEYS}})
    pd.DataFrame(drows).to_csv("results/table_learnable_graph_delta.csv", index=False)
    print("\nΔ vs fixed Corr+MST:")
    print(pd.DataFrame(drows)[["Variant", "F1", "ROC-AUC", "MCC"]].to_string(index=False))

    # ---- what did the graph learn? (default λ1=1, λ2=1) ----
    key = "Learned, λ1=1, λ2=1"
    if key in collected:
        W = collected[key].mean(axis=0)              # mean learned weight / pair
        A0 = prior_matrix(X, y)
        _, pair_index = build_complete_pairs()
        pi, pj = pair_index[0].tolist(), pair_index[1].tolist()
        delta = W - A0
        order = np.argsort(delta)
        recs = []
        for idx in list(order[-8:][::-1]) + list(order[:8]):
            recs.append({
                "Edge": f"{FEATURES[pi[idx]]}–{FEATURES[pj[idx]]}",
                "Pearson prior A0": round(float(A0[idx]), 4),
                "Learned A": round(float(W[idx]), 4),
                "Δ": round(float(delta[idx]), 4),
                "Direction": "strengthened" if delta[idx] > 0 else "pruned",
            })
        pd.DataFrame(recs).to_csv("results/table_learned_edges.csv", index=False)
        print("\nMost strengthened / pruned edges (learned vs Pearson prior):")
        print(pd.DataFrame(recs).to_string(index=False))

        # ---- figure ----
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].scatter(A0, W, s=28, alpha=0.75, color="#3498DB", edgecolor="white")
        lim = max(float(A0.max()), float(W.max())) * 1.08
        axes[0].plot([0, lim], [0, lim], "--", color="gray", lw=1,
                     label="A = A₀ (no change)")
        axes[0].set_xlabel("Pearson prior  A₀"); axes[0].set_ylabel("Learned  A")
        axes[0].set_title("Learned adjacency vs. Pearson prior\n(λ₁=1, λ₂=1)",
                          fontweight="bold")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        M = np.zeros((N_FEATURES, N_FEATURES))
        for idx, (i, j) in enumerate(zip(pi, pj)):
            M[i, j] = M[j, i] = delta[idx]
        v = float(np.abs(M).max())
        im = axes[1].imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[1].set_xticks(range(N_FEATURES)); axes[1].set_yticks(range(N_FEATURES))
        axes[1].set_xticklabels(FEATURES, rotation=90, fontsize=8)
        axes[1].set_yticklabels(FEATURES, fontsize=8)
        axes[1].set_title("Δ = learned − prior\n(red: strengthened, blue: pruned)",
                          fontweight="bold")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        plt.tight_layout()
        plt.savefig("figures/fig33_learned_vs_prior.png", dpi=150, bbox_inches="tight")
        print("\nSaved: figures/fig33_learned_vs_prior.png")

    print("\nSaved: results/table_learnable_graph.csv, "
          "table_learnable_graph_delta.csv, table_learned_edges.csv")


if __name__ == "__main__":
    main()
