# -*- coding: utf-8 -*-
"""
run_node_identity_sweep.py
===========================
Generalises run_node_identity.py's single-point test (id_dim = 8) into a
sweep over the identity-embedding dimension d in {0, 2, 4, 8, 16, 32},
requested by reviewers to check whether the smoothing-vs-relational-
reasoning reversal reported in the paper (Section "What message passing
over scalar nodes contributes") is an artefact of the single d=8 setting
that was originally tested, or holds consistently across dimension.

Protocol is UNCHANGED from run_node_identity.py: 3 seeds x 5 folds per
(id_dim, topology) cell, identical graph construction (tau=0.15, MST
augmentation), identical CV splits, identical training procedure. Only
the sweep range of id_dim is new.

Outputs
-------
results/table_node_identity_sweep.csv       -- full (id_dim, topology) grid
results/table_node_identity_gap_sweep.csv   -- graph-gap Wilcoxon test per d
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pipeline import load_data, _aggregate
from run_node_identity import run_cell, TOPOLOGIES, TOPO_LABEL

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123]
N_SPLITS = 5
ID_DIMS = [0, 2, 4, 8, 16, 32]     # 0 = scalar node (paper's current setting)


def main():
    X, y = load_data()
    print("Identity-embedding dimension sweep: d in", ID_DIMS)
    print(f"({len(SEEDS)} seeds x {N_SPLITS} folds = {len(SEEDS)*N_SPLITS} runs per cell, "
          f"{len(ID_DIMS)} dims x {len(TOPOLOGIES)} topologies = "
          f"{len(ID_DIMS)*len(TOPOLOGIES)} cells)")
    print("=" * 88)

    cells = {}
    rank_of = {}
    t_start = time.time()

    for id_dim in ID_DIMS:
        tag = "scalar" if id_dim == 0 else f"identity(d={id_dim})"
        for topo in TOPOLOGIES:
            folds, rk = [], []
            for seed in SEEDS:
                f, r = run_cell(X, y, topo, id_dim, seed)
                folds.extend(f)
                rk.append(r)
            cells[(id_dim, topo)] = folds
            rank_of[(id_dim, topo)] = float(np.mean(rk))
            s = _aggregate(folds)
            elapsed = time.time() - t_start
            print(f"[{elapsed:6.0f}s] {tag:<16} {TOPO_LABEL[topo]:<17} "
                  f"AUC={s['ROC-AUC'][0]:.4f}  F1={s['F1'][0]:.4f}  "
                  f"MCC={s['MCC'][0]:.4f}  rank(conv1)={rank_of[(id_dim, topo)]:.2f}")

    # ---------------- full grid table ----------------
    rows = []
    for id_dim in ID_DIMS:
        tag = "Scalar node (paper baseline)" if id_dim == 0 else f"Identity embedding (d={id_dim})"
        for topo in TOPOLOGIES:
            s = _aggregate(cells[(id_dim, topo)])
            rows.append({
                "id_dim": id_dim,
                "Node encoding": tag,
                "Topology": TOPO_LABEL[topo],
                "Effective rank (conv1)": round(rank_of[(id_dim, topo)], 2),
                "ROC-AUC": f"{s['ROC-AUC'][0]:.4f} \u00b1 {s['ROC-AUC'][1]:.4f}",
                "F1": f"{s['F1'][0]:.4f} \u00b1 {s['F1'][1]:.4f}",
                "MCC": f"{s['MCC'][0]:.4f} \u00b1 {s['MCC'][1]:.4f}",
                "Accuracy": f"{s['Accuracy'][0]:.4f} \u00b1 {s['Accuracy'][1]:.4f}",
            })
    df = pd.DataFrame(rows)
    df.to_csv("results/table_node_identity_sweep.csv", index=False)

    # ---------------- graph gap per dimension ----------------
    gap_rows = []
    for id_dim in ID_DIMS:
        tag = "Scalar node (paper baseline)" if id_dim == 0 else f"Identity embedding (d={id_dim})"
        for contrast, other in (("Corr+MST \u2212 No graph", "none"),
                                ("Corr+MST \u2212 Fully connected", "fully_connected")):
            a = np.array([m["ROC-AUC"] for m in cells[(id_dim, "corr_mst")]])
            b = np.array([m["ROC-AUC"] for m in cells[(id_dim, other)]])
            try:
                _, p = wilcoxon(a, b)
            except ValueError:
                p = float("nan")
            gap_rows.append({
                "id_dim": id_dim,
                "Node encoding": tag,
                "Contrast": contrast,
                "\u0394ROC-AUC": round(float(a.mean() - b.mean()), 4),
                "Wilcoxon p (15 paired folds)": round(float(p), 4),
                "Wins / 15": int((a > b).sum()),
            })
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv("results/table_node_identity_gap_sweep.csv", index=False)

    print("\n" + "=" * 88)
    print("FULL SWEEP GRID")
    print(df.to_string(index=False))
    print("\nGRAPH GAP BY DIMENSION (sign reversal check)")
    print(gaps.to_string(index=False))
    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    print("Saved: results/table_node_identity_sweep.csv, results/table_node_identity_gap_sweep.csv")


if __name__ == "__main__":
    main()
