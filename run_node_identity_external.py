# -*- coding: utf-8 -*-
"""
run_node_identity_external.py
=============================
The decisive test: does the feature-node graph contribute once nodes are
IDENTIFIABLE, measured where headroom actually exists (external cohorts)?

Internally the task is saturated (AUC ~0.91 for every model), so a null
graph effect there is uninformative. Under distribution shift there is
real headroom, and this is where an inductive bias should pay off.

Design
------
  node encoding : scalar (current)  |  scalar + identity embedding (d=8)
  topology      : no graph | corr+MST | fully connected
  cohorts       : Hungarian | Switzerland | VA Long Beach
  seeds         : 5

Everything is fit on Cleveland only (scaler, graph, weights, decision
threshold); each external cohort is scored exactly once per seed. The
transportable 8-feature set is used, as in run_external.py.

Quantity of interest: the GRAPH GAP

    Delta = AUC(corr+MST) - AUC(no graph)

per encoding, with a paired Wilcoxon test over matched seeds.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pipeline as P
from pipeline import metrics_from
from run_external import (
    EXT_FEATURES, EXT_CATEGORICAL, EXT_NUMERIC, COHORTS,
    patch_pipeline_features, load_cohort, complete_cases,
)
from run_node_identity import GCN_ID, make_graphs_id, eff_rank

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123, 2024, 5]
ID_DIM = 8
OPERATORS = ["gcn", "sage"]
TOPOLOGIES = ["none", "corr_mst", "fully_connected"]
TOPO_LABEL = {"none": "No graph", "corr_mst": "Corr+MST",
              "fully_connected": "Fully connected"}


def transport_once(X_dev, y_dev, ext_data, topology, id_dim, seed,
                   operator="gcn"):
    """Fit on Cleveland with one seed; score each external cohort once."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed)

    scaler = MinMaxScaler()
    X_tr = X_tr.copy(); X_val = X_val.copy()
    X_tr[EXT_NUMERIC] = scaler.fit_transform(X_tr[EXT_NUMERIC])
    X_val[EXT_NUMERIC] = scaler.transform(X_val[EXT_NUMERIC])

    edge_index, _ = P.build_edge_index(
        X_tr, topology=topology, threshold=P.TAU, seed=seed)

    g_tr = make_graphs_id(X_tr, y_tr, edge_index)
    g_val = make_graphs_id(X_val, y_val, edge_index)

    P.set_seed(seed)
    model = GCN_ID(hidden=32, dropout=0.30, id_dim=id_dim, operator=operator)
    model = P.train_model(model, g_tr, g_val, P.TrainConfig())

    vprob, vtrue = P.predict_probs(model, g_val)
    thr = P.best_threshold(vprob, vtrue)          # frozen on Cleveland

    model.eval()
    rank = float(np.mean([eff_rank(model.node_embeddings(g)) for g in g_val[:20]]))

    out = {}
    for cname, (Xe, ye) in ext_data.items():
        Xe = Xe.copy()
        Xe[EXT_NUMERIC] = scaler.transform(Xe[EXT_NUMERIC])
        g_e = make_graphs_id(Xe, ye, edge_index)
        prob, true = P.predict_probs(model, g_e)
        out[cname] = metrics_from(true, (prob >= thr).astype(int), prob)
    return out, rank


def main():
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    print(f"EXTERNAL node-identity factorial — {len(EXT_FEATURES)}-feature "
          f"transportable model, tau={P.TAU}")
    print(f"Encodings: scalar vs identity(d={ID_DIM}) | "
          f"Topologies: {list(TOPO_LABEL.values())} | seeds={SEEDS}")
    print("=" * 92)

    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, n_dev = complete_cases(dev_raw, dev_y, EXT_FEATURES)
    print(f"Cleveland development set: {n_dev} complete cases\n")

    ext_data = {}
    for cname, fname in COHORTS.items():
        d_raw, y_raw = load_cohort(fname)
        Xe, ye, n_c = complete_cases(d_raw, y_raw, EXT_FEATURES)
        ext_data[cname] = (Xe, ye)
        print(f"  {cname:<15} n={n_c}")
    print()

    # cell = (operator, id_dim, topology) -> cohort -> per-seed metric dicts
    results = {}
    ranks = {}
    for operator in OPERATORS:
        for id_dim in (0, ID_DIM):
            tag = "scalar" if id_dim == 0 else f"identity(d={id_dim})"
            for topo in TOPOLOGIES:
                per_cohort = {c: [] for c in COHORTS}
                rk = []
                for seed in SEEDS:
                    out, r = transport_once(X_dev, y_dev, ext_data, topo,
                                            id_dim, seed, operator=operator)
                    rk.append(r)
                    for c, m in out.items():
                        per_cohort[c].append(m)
                key = (operator, id_dim, topo)
                results[key] = per_cohort
                ranks[key] = float(np.mean(rk))
                aucs = {c: np.mean([m["ROC-AUC"] for m in v])
                        for c, v in per_cohort.items()}
                print(f"{operator.upper():<5} {tag:<16} {TOPO_LABEL[topo]:<17} "
                      + "  ".join(f"{c.split()[0]}={a:.4f}" for c, a in aucs.items())
                      + f"   rank={ranks[key]:.2f}")

    # ---------------- main table ----------------
    def cell_stats(v, key):
        return (float(np.mean([m[key] for m in v])), float(np.std([m[key] for m in v])))

    rows = []
    for operator in OPERATORS:
        for id_dim in (0, ID_DIM):
            tag = ("Scalar node (current)" if id_dim == 0
                   else f"+ Identity embedding (d={ID_DIM})")
            for topo in TOPOLOGIES:
                for c in COHORTS:
                    v = results[(operator, id_dim, topo)][c]
                    row = {
                        "Operator": operator.upper(),
                        "Node encoding": tag,
                        "Topology": TOPO_LABEL[topo],
                        "Cohort": c,
                        "Effective rank": round(ranks[(operator, id_dim, topo)], 2),
                    }
                    for k in ("ROC-AUC", "F1", "MCC"):
                        m, s = cell_stats(v, k)
                        row[k] = f"{m:.4f} ± {s:.4f}"
                    row["Specificity"] = f"{cell_stats(v, 'Specificity')[0]:.4f}"
                    rows.append(row)
    pd.DataFrame(rows).to_csv("results/table_node_identity_external.csv", index=False)

    # ---------------- graph gap, per cohort ----------------
    gap_rows = []
    for operator in OPERATORS:
        for id_dim in (0, ID_DIM):
            tag = ("Scalar node (current)" if id_dim == 0
                   else f"+ Identity embedding (d={ID_DIM})")
            for other, oname in (("none", "No graph"),
                                 ("fully_connected", "Fully connected")):
                for c in COHORTS:
                    a = np.array([m["ROC-AUC"] for m in
                                  results[(operator, id_dim, "corr_mst")][c]])
                    b = np.array([m["ROC-AUC"] for m in
                                  results[(operator, id_dim, other)][c]])
                    try:
                        _, p = wilcoxon(a, b)
                    except ValueError:
                        p = float("nan")
                    gap_rows.append({
                        "Operator": operator.upper(),
                        "Node encoding": tag,
                        "Contrast": f"Corr+MST − {oname}",
                        "Cohort": c,
                        "ΔROC-AUC": round(float(a.mean() - b.mean()), 4),
                        "Wins / 5": int((a > b).sum()),
                        "Wilcoxon p": round(float(p), 4),
                    })
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv("results/table_node_identity_external_gap.csv", index=False)

    # ---------------- pooled gap across cohorts ----------------
    pooled = []
    for operator in OPERATORS:
        for id_dim in (0, ID_DIM):
            tag = ("Scalar node (current)" if id_dim == 0
                   else f"+ Identity embedding (d={ID_DIM})")
            for other, oname in (("none", "No graph"),
                                 ("fully_connected", "Fully connected")):
                a, b = [], []
                for c in COHORTS:
                    a += [m["ROC-AUC"] for m in results[(operator, id_dim, "corr_mst")][c]]
                    b += [m["ROC-AUC"] for m in results[(operator, id_dim, other)][c]]
                a, b = np.array(a), np.array(b)
                try:
                    _, p = wilcoxon(a, b)
                except ValueError:
                    p = float("nan")
                pooled.append({
                    "Operator": operator.upper(),
                    "Node encoding": tag,
                    "Contrast": f"Corr+MST − {oname}",
                    "Mean ΔROC-AUC": round(float(a.mean() - b.mean()), 4),
                    "Wins / 15": int((a > b).sum()),
                    "Wilcoxon p (15 paired)": round(float(p), 5),
                })
    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv("results/table_node_identity_external_pooled.csv", index=False)

    print("\n" + "=" * 92)
    print("GRAPH GAP BY COHORT")
    print(gaps.to_string(index=False))
    print("\nPOOLED ACROSS COHORTS (the headline contrast)")
    print(pooled_df.to_string(index=False))
    print("\nSaved: results/table_node_identity_external.csv, "
          "table_node_identity_external_gap.csv, "
          "table_node_identity_external_pooled.csv")


if __name__ == "__main__":
    main()
