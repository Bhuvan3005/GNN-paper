# -*- coding: utf-8 -*-
"""
run_graph_gap_bootstrap.py
==========================
Patient-level inference for the graph gap.

The seed-level Wilcoxon test in run_node_identity_external.py answers
"is the graph advantage reproducible across initialisations?" (yes:
15/15, p<1e-4). Reviewers will additionally ask the *patient-level*
question: "does it generalise to new patients?"

This script answers that with a paired bootstrap over patients. For each
cohort we average the predicted probabilities over seeds (a seed
ensemble, which is what one would actually deploy), then resample
patients with replacement B times and recompute

    Delta AUC = AUC(corr+MST) - AUC(no graph)
    Delta AUC = AUC(corr+MST) - AUC(fully connected)

reporting the 95% percentile interval. Sampling is PAIRED: the same
resampled patient indices are used for both models, so the comparison
removes patient-sampling noise common to both.

Scalar node encoding only -- the configuration used in the paper.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pipeline as P
from run_external import (
    EXT_FEATURES, EXT_CATEGORICAL, EXT_NUMERIC, COHORTS,
    patch_pipeline_features, load_cohort, complete_cases,
)
from run_node_identity import GCN_ID, make_graphs_id

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123, 2024, 5]
B = 2000
RNG = np.random.default_rng(12345)
TOPOS = {"corr_mst": "Corr+MST", "none": "No graph",
         "fully_connected": "Fully connected"}


def probs_for(X_dev, y_dev, ext_data, topology, seed, operator="gcn"):
    """Fit on Cleveland, return {cohort: prob vector}."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed)
    sc = MinMaxScaler()
    X_tr, X_val = X_tr.copy(), X_val.copy()
    X_tr[EXT_NUMERIC] = sc.fit_transform(X_tr[EXT_NUMERIC])
    X_val[EXT_NUMERIC] = sc.transform(X_val[EXT_NUMERIC])

    ei, _ = P.build_edge_index(X_tr, topology=topology,
                               threshold=P.TAU, seed=seed)
    g_tr = make_graphs_id(X_tr, y_tr, ei)
    g_val = make_graphs_id(X_val, y_val, ei)

    P.set_seed(seed)
    model = GCN_ID(hidden=32, dropout=0.30, id_dim=0, operator=operator)
    model = P.train_model(model, g_tr, g_val, P.TrainConfig())

    out = {}
    for cname, (Xe, ye) in ext_data.items():
        Xe = Xe.copy()
        Xe[EXT_NUMERIC] = sc.transform(Xe[EXT_NUMERIC])
        prob, _ = P.predict_probs(model, make_graphs_id(Xe, ye, ei))
        out[cname] = prob
    return out


def main():
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, _ = complete_cases(dev_raw, dev_y, EXT_FEATURES)

    ext_data, labels = {}, {}
    for cname, fname in COHORTS.items():
        d, yy = load_cohort(fname)
        Xe, ye, _ = complete_cases(d, yy, EXT_FEATURES)
        ext_data[cname] = (Xe, ye)
        labels[cname] = ye.values.astype(int)

    print(f"Patient-level paired bootstrap (B={B}) of the graph gap")
    print("Scalar node encoding, GCN operator, seed-ensembled probabilities")
    print("=" * 78)

    # seed-ensembled probabilities per topology per cohort
    ens = {t: {c: [] for c in COHORTS} for t in TOPOS}
    for topo in TOPOS:
        for seed in SEEDS:
            out = probs_for(X_dev, y_dev, ext_data, topo, seed)
            for c, p in out.items():
                ens[topo][c].append(p)
        for c in COHORTS:
            ens[topo][c] = np.mean(np.stack(ens[topo][c]), axis=0)
        print(f"  {TOPOS[topo]:<17} " + "  ".join(
            f"{c.split()[0]}={roc_auc_score(labels[c], ens[topo][c]):.4f}"
            for c in COHORTS))

    rows = []
    for other in ("none", "fully_connected"):
        for c in COHORTS:
            y = labels[c]
            a, b = ens["corr_mst"][c], ens[other][c]
            point = roc_auc_score(y, a) - roc_auc_score(y, b)
            deltas = []
            n = len(y)
            for _ in range(B):
                idx = RNG.integers(0, n, n)              # paired resample
                if len(np.unique(y[idx])) < 2:
                    continue
                deltas.append(roc_auc_score(y[idx], a[idx])
                              - roc_auc_score(y[idx], b[idx]))
            deltas = np.array(deltas)
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            rows.append({
                "Cohort": c,
                "Contrast": f"Corr+MST − {TOPOS[other]}",
                "n": n,
                "AUC Corr+MST": round(roc_auc_score(y, a), 4),
                f"AUC {TOPOS[other]}": round(roc_auc_score(y, b), 4),
                "ΔAUC": round(float(point), 4),
                "95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
                "Excludes 0": "yes" if (lo > 0 or hi < 0) else "no",
                "P(Δ>0)": round(float((deltas > 0).mean()), 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/table_graph_gap_bootstrap.csv", index=False)
    print("\nPATIENT-LEVEL PAIRED BOOTSTRAP")
    print(df.to_string(index=False))
    print("\nSaved: results/table_graph_gap_bootstrap.csv")


if __name__ == "__main__":
    main()
