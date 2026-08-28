# -*- coding: utf-8 -*-
"""
make_combined_tables.py
=======================
IEEE Access consolidation pass.

Several results were originally reported as a performance table plus a
separate significance/delta table. Each pair is logically one result, and
a 27-page manuscript carrying 55 floats reads as padded. This script
merges each pair into a SINGLE two-panel float.

Both of the original \\label commands are emitted inside the merged float,
so every \\ref in the manuscript keeps resolving -- to the same table
number -- and no prose has to be rewritten.

Run AFTER make_latex_tables.py:

    python make_latex_tables.py
    python make_combined_tables.py
"""

import os
import re
import pandas as pd

from make_latex_tables import (
    escape, _visible, DROP_COLS, RENAME, ALIGN_OVERRIDE,
)

OUT_DIR = "results/latex"

# Panel headers are shortened relative to the standalone tables. Stacking
# two panels in one float leaves less width, and a header such as
# "Sig. Holm (McNemar (median))" forces \resizebox to shrink the panel
# below IEEE's legibility floor. The qualifier moves into the caption.
PANEL_RENAME = {
    "table_conv_stats": {
        "Mean ΔAUC (GCN−other)": "Mean ΔAUC",
        "DeLong p (median)": "DeLong p",
        "McNemar p (median)": "McNemar p",
        "Holm p (DeLong (median))": "Holm p (DeLong)",
        "Sig. Holm (DeLong (median))": "Sig. (DeLong)",
        "Holm p (McNemar (median))": "Holm p (McNemar)",
        "Sig. Holm (McNemar (median))": "Sig. (McNemar)",
    },
    "table_knn_stats": {
        "Holm p (DeLong (median))": "Holm p (DeLong)",
        "Sig. Holm (DeLong (median))": "Sig. (DeLong)",
        "Holm p (McNemar (median))": "Holm p (McNemar)",
        "Sig. Holm (McNemar (median))": "Sig. (McNemar)",
    },
    "table_mst_tau_stats": {
        "Holm p (DeLong (median))": "Holm p (DeLong)",
        "Sig. Holm (DeLong (median))": "Sig. (DeLong)",
        "Holm p (McNemar (median))": "Holm p (McNemar)",
        "Sig. Holm (McNemar (median))": "Sig. (McNemar)",
    },
}

# merged-name -> (caption, [(source_table, panel_title), ...])
MERGES = {
    "combined_performance": (
        "Predictive performance under the identical five-fold protocol. "
        "Panel (a) reports the fold-wise mean $\\pm$ standard deviation; "
        "panel (b) reports pooled out-of-fold point estimates over all "
        "303 patients.",
        [("table_model_comparison", "(a) Fold-wise mean $\\pm$ std"),
         ("table_pooled_oof", "(b) Pooled out-of-fold point estimates")],
    ),
    "combined_ablation": (
        "Ablation study. Panel (a) reports absolute performance for each "
        "variant, pooled over 3 seeds $\\times$ 5 folds; panel (b) reports "
        "the change relative to the full model.",
        [("table_ablation", "(a) Absolute performance"),
         ("table_ablation_delta", "(b) Change relative to the full model")],
    ),
    "combined_conv": (
        "Convolution-operator comparison. Panel (a) reports performance for "
        "each operator; panel (b) reports the paired significance tests "
        "against GCN. All $p$-values in panel (b) are medians over the "
        "matched runs, with Holm correction applied within the family.",
        [("table_conv_comparison", "(a) Performance by operator"),
         ("table_conv_stats", "(b) Paired tests against GCN")],
    ),
    "combined_nodeid": (
        "Node-identity falsification test. Panel (a) reports performance "
        "and effective rank under each node encoding; panel (b) reports the "
        "internal graph gap paired over the 15 matched runs.",
        [("table_node_identity", "(a) Performance and effective rank"),
         ("table_node_identity_gap", "(b) Paired internal graph gap")],
    ),
    "combined_xai": (
        "Explainability evaluation. Panel (a) compares the three attribution "
        "methods across the quantitative metrics; panel (b) reports "
        "cross-method agreement.",
        [("table_xai_metrics", "(a) Attribution-method metrics"),
         ("table_xai_agreement", "(b) Cross-method agreement")],
    ),
    "combined_bootstrap": (
        "Paired patient-level bootstrap on the external cohorts. Panel (a) "
        "reports per-model ROC-AUC with percentile intervals; panel (b) "
        "reports the paired differences against the feature-node model.",
        [("table_external_bootstrap_auc", "(a) Bootstrap ROC-AUC"),
         ("table_external_bootstrap_delta", "(b) Paired differences")],
    ),
    "combined_msttau": (
        "Graph connectivity across the $\\tau$ regime. Panel (a) reports "
        "performance by topology and threshold; panel (b) reports the "
        "corresponding paired tests.",
        [("table_mst_tau_comparison", "(a) Performance by $\\tau$"),
         ("table_mst_tau_stats", "(b) Paired tests")],
    ),
    "combined_toposig": (
        "Topology significance. Panel (a) reports performance by topology; "
        "panel (b) reports the paired contrasts with multiplicity control.",
        [("table_topology_significance_perf", "(a) Performance by topology"),
         ("table_topology_significance", "(b) Paired contrasts")],
    ),
    "combined_knn": (
        "$k$-nearest-neighbour feature topology. Panel (a) reports "
        "performance across $k$; panel (b) reports the paired tests against "
        "the Pearson+MST graph.",
        [("table_knn_topology", "(a) Performance across $k$"),
         ("table_knn_stats", "(b) Paired tests vs. Pearson+MST")],
    ),
    # ---- second consolidation pass ----
    "combined_extsetup": (
        "External cohort setup. Panel (a) reports cohort availability under "
        "the 8-feature transportable set; panel (b) gives the internal "
        "Cleveland reference for that reduced model, separating the cost of "
        "feature reduction from the cost of cohort shift.",
        [("table_external_cohorts", "(a) Cohort availability"),
         ("table_external_reference", "(b) Internal 8-feature reference")],
    ),
    "combined_extresult": (
        "External validation results. Panel (a) reports transported "
        "performance per cohort (mean $\\pm$ std over 5 seeds); panel (b) "
        "reports the corresponding DeLong and McNemar comparisons with Holm "
        "correction.",
        [("table_external_validation", "(a) Transported performance"),
         ("table_external_stats", "(b) Statistical comparison")],
    ),
    "combined_patientext": (
        "Transport of the patient-similarity formulation. Panel (a) reports "
        "its external performance; panel (b) contrasts it with the "
        "feature-node model per cohort.",
        [("table_patient_external", "(a) Patient-similarity, transported"),
         ("table_patient_vs_feature_external", "(b) Contrast per cohort")],
    ),
    "combined_patient": (
        "Feature-node versus patient-similarity formulation in "
        "distribution. Panel (a) reports the paired comparison; panel (b) "
        "reports the significance tests.",
        [("table_patient_vs_feature", "(a) Paired comparison"),
         ("table_patient_vs_feature_stats", "(b) Significance tests")],
    ),
    "combined_mstact": (
        "MST activation on the 13-feature graph. Panel (a) reports where the "
        "thresholded graph fragments; panel (b) reports the resulting "
        "connectivity.",
        [("table_mst_activation", "(a) Fragmentation by $\\tau$"),
         ("table_mst_connectivity", "(b) Resulting connectivity")],
    ),
    "combined_calib": (
        "Calibration and clinical utility. Panel (a) reports calibration of "
        "the predicted risks; panel (b) summarises the decision-curve "
        "analysis.",
        [("table_calibration", "(a) Calibration"),
         ("table_decision_curve_summary", "(b) Decision-curve summary")],
    ),
    "combined_repr": (
        "Learned representation. Panel (a) reports the effective rank of the "
        "node embeddings; panel (b) lists the fold-stable feature "
        "neighbourhoods.",
        [("table_representation_rank", "(a) Effective rank"),
         ("table_feature_neighbourhoods", "(b) Feature neighbourhoods")],
    ),
}


def label_of(table_name):
    """Recover the \\label already assigned to a generated table."""
    path = os.path.join(OUT_DIR, table_name + ".tex")
    if not os.path.exists(path):
        return None
    m = re.search(r"\\label\{([^}]+)\}", open(path, encoding="utf-8").read())
    return m.group(1) if m else None


def panel(table_name, title):
    """Render one source table as a captioned tabular panel (no float)."""
    df = pd.read_csv(os.path.join("results", table_name + ".csv"))
    for c in DROP_COLS.get(table_name, []):
        if c in df.columns:
            df = df.drop(columns=c)
    df = df.rename(columns=RENAME.get(table_name, {}))
    df = df.rename(columns=PANEL_RENAME.get(table_name, {}))

    cols = list(df.columns)
    align = ALIGN_OVERRIDE.get(table_name, "l" + "r" * (len(cols) - 1))
    header = " & ".join(escape(c) for c in cols) + " \\\\"
    body = [" & ".join(escape(row[c]) for c in cols) + " \\\\"
            for _, row in df.iterrows()]

    # Merged tables are always full width; scale any panel that overruns.
    widest = max(len(_visible(x)) for x in [header] + body)
    fit = ("p{" not in align) and widest > 118

    out = [f"\\textbf{{{title}}}\\\\[2pt]"]
    if fit:
        out.append("\\resizebox{\\textwidth}{!}{%")
    out += [f"\\begin{{tabular}}{{{align}}}", "\\toprule", header, "\\midrule"]
    out += body
    out += ["\\bottomrule", "\\end{tabular}"]
    if fit:
        out.append("}")
    return out


def build(name, caption, parts):
    labels = [label_of(t) for t in [p[0] for p in parts]]
    if any(l is None for l in labels):
        print(f"  SKIP {name}: missing source table")
        return None

    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}"]
    # Both original labels live here, so existing \ref commands still work.
    lines += [f"\\label{{{l}}}" for l in labels]
    lines.append("\\footnotesize")
    for i, (tbl, title) in enumerate(parts):
        if i:
            lines.append("\\vspace{7pt}")
        lines += panel(tbl, title)
    lines += ["\\end{table*}", ""]

    path = os.path.join(OUT_DIR, name + ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path, labels


def main():
    print("Merging paired tables into two-panel floats:")
    made = []
    for name, (caption, parts) in MERGES.items():
        r = build(name, caption, parts)
        if r:
            path, labels = r
            made.append(path)
            print(f"  {name:<24} <- {' + '.join(p[0] for p in parts)}")
            print(f"  {'':<24}    labels kept: {', '.join(labels)}")
    print(f"\n{len(made)} merged tables written to {OUT_DIR}/")
    print(f"Floats saved: {len(made)} (each pair now occupies one float)")


if __name__ == "__main__":
    main()
