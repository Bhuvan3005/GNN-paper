# -*- coding: utf-8 -*-
"""
check_mst_activation_ext.py
===========================
Precondition check for the MST transport experiment.

The external study uses the reduced 8-feature transportable set, so the
graph has 8 nodes, NOT 13. Before asking whether MST bridges improve
external transport we must confirm that at a given tau the correlation-
only graph is actually DISCONNECTED on that 8-node feature set -- i.e.
that the MST has real work to do. If corr-only is already connected,
corr_mst == corr_only and the experiment is vacuous.

Reports, per tau, over the Cleveland development splits used by
run_external.transport_once: number of components and isolated nodes of
the correlation-only graph, and how many bridge edges MST adds.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pipeline as P
from run_external import (
    EXT_FEATURES, EXT_CATEGORICAL, EXT_NUMERIC, SEEDS,
    load_cohort, complete_cases, patch_pipeline_features,
)

TAUS = [0.10, 0.15, 0.20, 0.25, 0.30]


def corr_only_graph(corr, tau, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= tau:
                G.add_edge(i, j)
    return G


def main():
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    n = len(EXT_FEATURES)
    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, _ = complete_cases(dev_raw, dev_y, EXT_FEATURES)

    print("=" * 74)
    print(f"MST ACTIVATION CHECK — {n}-feature transportable set")
    print(f"Features: {EXT_FEATURES}")
    print("=" * 74)
    print(f"{'tau':>6} {'seed':>5} {'corr edges':>11} {'components':>11} "
          f"{'isolated':>9} {'MST bridges':>12}")
    print("-" * 74)

    summary = {}
    for tau in TAUS:
        comps, bridges, iso = [], [], []
        for seed in SEEDS:
            X_tr, _, y_tr, _ = train_test_split(
                X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed)
            X_tr = X_tr.copy()
            sc = MinMaxScaler()
            X_tr[EXT_NUMERIC] = sc.fit_transform(X_tr[EXT_NUMERIC])
            corr = X_tr[EXT_FEATURES].corr().values

            G = corr_only_graph(corr, tau, n)
            nc = nx.number_connected_components(G)
            n_iso = sum(1 for _, d in G.degree() if d == 0)

            G_mst = P._corr_mst_edge_set(corr, tau)
            n_bridge = G_mst.number_of_edges() - G.number_of_edges()

            comps.append(nc); bridges.append(n_bridge); iso.append(n_iso)
            print(f"{tau:>6.2f} {seed:>5} {G.number_of_edges():>11} "
                  f"{nc:>11} {n_iso:>9} {n_bridge:>12}")
        summary[tau] = (np.mean(comps), np.mean(bridges), np.mean(iso))
        print("-" * 74)

    print("\nSUMMARY (mean over seeds)")
    print(f"{'tau':>6} {'components':>12} {'isolated':>10} {'MST bridges':>13}  verdict")
    rows = []
    for tau, (c, b, i) in summary.items():
        connected_splits = sum(
            1 for seed in SEEDS
            if _components_for(X_dev, y_dev, tau, seed, n) == 1)
        verdict = ("MST ACTIVE — corr-only fragments" if c > 1.0001
                   else "vacuous — corr-only already connected")
        print(f"{tau:>6.2f} {c:>12.2f} {i:>10.2f} {b:>13.2f}  {verdict}")
        rows.append({
            "tau": tau,
            "Components before MST": round(c, 2),
            "Isolated nodes": round(i, 2),
            "MST bridges added": round(b, 2),
            "Connected before MST": f"{connected_splits}/{len(SEEDS)}",
            "MST status": "inert" if c <= 1.0001 else "active",
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(
        "results/table_mst_activation_ext.csv", index=False)
    print("\nSaved: results/table_mst_activation_ext.csv")


def _components_for(X_dev, y_dev, tau, seed, n):
    X_tr, _, _, _ = train_test_split(
        X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed)
    X_tr = X_tr.copy()
    sc = MinMaxScaler()
    X_tr[EXT_NUMERIC] = sc.fit_transform(X_tr[EXT_NUMERIC])
    corr = X_tr[EXT_FEATURES].corr().values
    return nx.number_connected_components(corr_only_graph(corr, tau, n))


if __name__ == "__main__":
    main()
