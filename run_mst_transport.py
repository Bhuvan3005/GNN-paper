# -*- coding: utf-8 -*-
"""
run_mst_transport.py
====================
Does the MST bridge set buy OUT-OF-DISTRIBUTION robustness?

MOTIVATION
----------
On Cleveland the internal topology ablation is essentially null: after
Holm correction, Corr+MST is indistinguishable from a no-graph model.
That leaves the graph-construction contribution unsupported.

The hypothesis tested here is that the graph's value is not in-distribution
accuracy but TRANSPORTABILITY: a connected feature graph propagates
information between all features, so when a cohort shifts, no feature is
stranded. Concretely:

    without MST -> graph disconnected -> external transport worse
    with MST    -> graph connected    -> transport improves

DESIGN
------
tau = 0.20 is used deliberately. At the paper's default tau = 0.15 on the
reduced 8-feature transportable set the correlation-only graph is already
nearly connected, so corr_mst == corr_only and the comparison is vacuous.
At tau = 0.20 the correlation-only graph fragments into ~2.8 components
with ~1.8 isolated nodes (see check_mst_activation_ext.py), so the MST
has real work to do.

Topologies compared (identical architecture, scaler, threshold protocol):
    corr_mst        correlation edges + MST bridges   (connected)
    corr_only       correlation edges only            (DISCONNECTED at 0.20)
    none            no edges                          (lower bound)
    fully_connected complete graph                    (upper bound on density)

Everything (scaler, graph, weights, decision threshold) is fit on
Cleveland only; each external cohort is scored once per seed.

Statistics
----------
  * Paired Wilcoxon across seeds  -> accounts for initialisation noise,
    the only thing that varies between two topologies on a fixed cohort.
  * DeLong on seed-averaged probabilities -> accounts for patient-level
    uncertainty on the fixed external cohort.
Both are reported because they answer different questions.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import pipeline as P
from run_external import (
    EXT_FEATURES, EXT_CATEGORICAL, EXT_NUMERIC, COHORTS,
    load_cohort, complete_cases, patch_pipeline_features,
)
from run_external_stats import delong_roc_test

os.makedirs("results", exist_ok=True)

TAU = 0.20                       # MST-active regime on the 8-feature set
SEEDS = [42, 7, 123, 2024, 5, 99, 314, 777, 1010, 2718]
TOPOLOGIES = ["corr_mst", "corr_only", "none", "fully_connected"]
LABEL = {
    "corr_mst": "Corr+MST (connected)",
    "corr_only": "Corr only (disconnected)",
    "none": "No graph",
    "fully_connected": "Fully connected",
}
METRIC_KEYS = ["Accuracy", "F1", "ROC-AUC", "MCC", "Recall", "Specificity"]


# ----------------------------------------------------------------------
def fit_and_transport(X_dev, y_dev, ext_data, topology, seed):
    """Fit one GCN on Cleveland with the given topology; score all cohorts.
    Returns (internal_val_auc, {cohort: (metrics, probs, y)}), plus the
    connectivity of the graph actually used."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed)

    scaler = MinMaxScaler()
    X_tr = X_tr.copy(); X_val = X_val.copy()
    X_tr[EXT_NUMERIC] = scaler.fit_transform(X_tr[EXT_NUMERIC])
    X_val[EXT_NUMERIC] = scaler.transform(X_val[EXT_NUMERIC])

    edge_index, _ = P.build_edge_index(X_tr, topology=topology,
                                       threshold=TAU, seed=seed)

    # record connectivity of the graph actually used
    n = len(EXT_FEATURES)
    G = nx.Graph(); G.add_nodes_from(range(n))
    if edge_index.numel():
        G.add_edges_from([(int(a), int(b)) for a, b in edge_index.t().tolist()])
    n_comp = nx.number_connected_components(G)

    g_tr = P.make_graphs(X_tr, y_tr, edge_index)
    g_val = P.make_graphs(X_val, y_val, edge_index)

    P.set_seed(seed)
    model = P.GCN(P.GCNConfig())
    model = P.train_model(model, g_tr, g_val, P.TrainConfig())

    vprob, vtrue = P.predict_probs(model, g_val)
    thr = P.best_threshold(vprob, vtrue)              # frozen on Cleveland
    internal_auc = roc_auc_score(vtrue, vprob) if len(set(vtrue)) > 1 else np.nan

    out = {}
    for name, (Xe, ye) in ext_data.items():
        Xe_s = Xe.copy()
        Xe_s[EXT_NUMERIC] = scaler.transform(Xe_s[EXT_NUMERIC])
        g_ext = P.make_graphs(Xe_s, ye, edge_index)
        prob, true = P.predict_probs(model, g_ext)
        m = P.metrics_from(true, (prob >= thr).astype(int), prob)
        out[name] = (m, prob, true)
    return internal_auc, out, n_comp, G.number_of_edges()


def main():
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    print("=" * 78)
    print(f"MST TRANSPORT EXPERIMENT — tau = {TAU} (MST-active regime)")
    print(f"{len(EXT_FEATURES)}-feature transportable set, {len(SEEDS)} seeds")
    print("=" * 78)

    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, n_dev = complete_cases(dev_raw, dev_y, EXT_FEATURES)
    print(f"Cleveland development: {n_dev} complete cases, "
          f"prevalence {100*y_dev.mean():.1f}%")

    ext_data = {}
    for name, fname in COHORTS.items():
        d_raw, y_raw = load_cohort(fname)
        Xe, ye, n_c = complete_cases(d_raw, y_raw, EXT_FEATURES)
        if n_c > 0 and ye.nunique() >= 2:
            ext_data[name] = (Xe, ye)
            print(f"  {name}: n={n_c}, prevalence {100*ye.mean():.1f}%")

    # ---------------- run every topology over every seed ----------------
    # store[topology][cohort] = list over seeds of (metrics, prob, true)
    store = {t: {c: [] for c in ext_data} for t in TOPOLOGIES}
    internal = {t: [] for t in TOPOLOGIES}
    conn = {t: [] for t in TOPOLOGIES}
    edges = {t: [] for t in TOPOLOGIES}

    for topo in TOPOLOGIES:
        print(f"\n[{LABEL[topo]}]")
        for seed in SEEDS:
            iauc, out, n_comp, n_edge = fit_and_transport(
                X_dev, y_dev, ext_data, topo, seed)
            internal[topo].append(iauc)
            conn[topo].append(n_comp)
            edges[topo].append(n_edge)
            for c, v in out.items():
                store[topo][c].append(v)
        print(f"  components={np.mean(conn[topo]):.2f}  "
              f"edges={np.mean(edges[topo]):.1f}  "
              f"internal val AUC={np.nanmean(internal[topo]):.4f}")

    # ---------------- main results table ----------------
    rows = []
    for topo in TOPOLOGIES:
        rows.append({
            "Topology": LABEL[topo],
            "Setting": "Cleveland (internal val)",
            "n": "—",
            "Components": f"{np.mean(conn[topo]):.2f}",
            "ROC-AUC": f"{np.nanmean(internal[topo]):.4f} ± {np.nanstd(internal[topo]):.4f}",
            **{k: "—" for k in METRIC_KEYS if k != "ROC-AUC"},
        })
        for c in ext_data:
            vals = {k: [m[k] for m, _, _ in store[topo][c]] for k in METRIC_KEYS}
            rows.append({
                "Topology": LABEL[topo],
                "Setting": c,
                "n": len(ext_data[c][1]),
                "Components": f"{np.mean(conn[topo]):.2f}",
                **{k: f"{np.mean(vals[k]):.4f} ± {np.std(vals[k]):.4f}"
                   for k in METRIC_KEYS},
            })
    df = pd.DataFrame(rows)
    df.to_csv("results/table_mst_transport.csv", index=False)

    print("\n" + "=" * 78)
    print("TRANSPORT BY TOPOLOGY (mean ± std over seeds)")
    print("=" * 78)
    print(df[["Topology", "Setting", "n", "Components", "ROC-AUC", "F1", "MCC"]]
          .to_string(index=False))

    # ---- persist raw per-seed probabilities for downstream analysis ----
    save = {}
    for topo in TOPOLOGIES:
        for c in ext_data:
            save[f"{topo}|{c}|prob"] = np.stack([p for _, p, _ in store[topo][c]])
            save[f"{topo}|{c}|true"] = store[topo][c][0][2]
    np.savez("results/mst_transport_probs.npz", **save)

    # ---------------- all contrasts against Corr+MST ----------------
    # Bootstrap CI over patients on the seed-averaged score, so the
    # interval reflects patient sampling rather than initialisation noise.
    rng = np.random.default_rng(0)
    B = 2000

    def boot_delta_ci(ytrue, pa, pb):
        n = len(ytrue)
        idx_pos = np.where(ytrue == 1)[0]
        idx_neg = np.where(ytrue == 0)[0]
        if len(idx_pos) < 2 or len(idx_neg) < 2:
            return float("nan"), float("nan")
        deltas = []
        for _ in range(B):
            bp = rng.choice(idx_pos, len(idx_pos), replace=True)
            bn = rng.choice(idx_neg, len(idx_neg), replace=True)
            bi = np.concatenate([bp, bn])
            yb = ytrue[bi]
            try:
                deltas.append(roc_auc_score(yb, pa[bi]) - roc_auc_score(yb, pb[bi]))
            except ValueError:
                continue
        if not deltas:
            return float("nan"), float("nan")
        return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

    stat_rows = []
    for c in ext_data:
        ytrue = store["corr_mst"][c][0][2]
        a_auc = np.array([m["ROC-AUC"] for m, _, _ in store["corr_mst"][c]])
        pa = np.mean(np.stack([p for _, p, _ in store["corr_mst"][c]]), axis=0)
        for other in ["corr_only", "none", "fully_connected"]:
            b_auc = np.array([m["ROC-AUC"] for m, _, _ in store[other][c]])
            pb = np.mean(np.stack([p for _, p, _ in store[other][c]]), axis=0)
            try:
                _, wp = wilcoxon(a_auc, b_auc)
            except ValueError:
                wp = float("nan")
            try:
                _aa, _ab, dp = delong_roc_test(ytrue, pa, pb)
            except Exception:
                dp = float("nan")
            lo, hi = boot_delta_ci(ytrue, pa, pb)
            stat_rows.append({
                "Cohort": c,
                "Contrast": f"Corr+MST vs {LABEL[other]}",
                "AUC Corr+MST": round(float(a_auc.mean()), 4),
                "AUC other": round(float(b_auc.mean()), 4),
                "ΔAUC": f"{a_auc.mean() - b_auc.mean():+.4f}",
                "95% CI (bootstrap)": f"[{lo:+.4f}, {hi:+.4f}]",
                "CI excludes 0": "yes" if (lo > 0 or hi < 0) else "no",
                "Wilcoxon p (seeds)": round(float(wp), 4),
                "DeLong p (patients)": f"{dp:.4g}",
            })
    stats = pd.DataFrame(stat_rows)

    # Holm correction across all contrasts, per test family
    for col, newcol in [("DeLong p (patients)", "Holm p (DeLong)"),
                        ("Wilcoxon p (seeds)", "Holm p (Wilcoxon)")]:
        p = pd.to_numeric(stats[col], errors="coerce").values
        order = np.argsort(np.nan_to_num(p, nan=1.0))
        m = len(p)
        adj = np.full(m, np.nan)
        running = 0.0
        for rank, i in enumerate(order):
            if np.isnan(p[i]):
                continue
            val = (m - rank) * p[i]
            running = max(running, val)
            adj[i] = min(1.0, running)
        stats[newcol] = np.round(adj, 4)
    stats["Significant (Holm, DeLong)"] = np.where(
        stats["Holm p (DeLong)"] < 0.05, "yes", "no")

    stats.to_csv("results/table_mst_transport_stats.csv", index=False)

    print("\n" + "=" * 78)
    print("ALL CONTRASTS AGAINST Corr+MST")
    print("=" * 78)
    print(stats[["Cohort", "Contrast", "AUC Corr+MST", "AUC other", "ΔAUC",
                 "95% CI (bootstrap)", "CI excludes 0", "Holm p (DeLong)",
                 "Significant (Holm, DeLong)"]].to_string(index=False))
    print("\nSaved: results/table_mst_transport.csv, "
          "results/table_mst_transport_stats.csv, "
          "results/mst_transport_probs.npz")


if __name__ == "__main__":
    main()
