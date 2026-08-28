# -*- coding: utf-8 -*-
"""
make_latex_tables.py
=====================
Converts every results/*.csv into a booktabs LaTeX table in
results/latex/*.tex, ready to \\input into the manuscript.

IEEE two-column layout gives a single column only ~252pt. Tables with
several "mean ± std" columns do not fit, so wide tables are emitted as
full-width `table*` floats, and purely descriptive columns (e.g. the
ablation's prose "Question"/"Contribution" fields, which are stated in
the running text) are dropped from the typeset version.
"""

import os
import glob
import re
import pandas as pd

os.makedirs("results/latex", exist_ok=True)

CAPTIONS = {
    "table_model_comparison": ("Model comparison under identical 5-fold "
                               "stratified cross-validation, repeated over three "
                               "seeds (mean $\\pm$ std).", "tab:comparison"),
    "table_statistical_tests": ("Statistical significance of the GCN vs. baseline "
                                "classifiers (McNemar on pooled OOF predictions; "
                                "Wilcoxon signed-rank on per-fold F1).", "tab:stats"),
    "table_ablation": ("Ablation study. Each experiment isolates one design "
                       "choice; results are mean $\\pm$ std pooled over "
                       "3 seeds $\\times$ 5 folds.", "tab:ablation"),
    "table_ablation_delta": ("Ablation deltas relative to the full model.",
                             "tab:ablation_delta"),
    "table_graph_statistics": ("Feature-graph statistics (mean $\\pm$ std across "
                               "the 5 CV folds).", "tab:graphstats"),
    "table_hyperparameters": ("Training and architecture hyperparameters.",
                              "tab:hyperparams"),
    "table_threshold_sensitivity": ("Sensitivity of the full model to the Pearson "
                                    "correlation threshold $\\tau$.", "tab:threshold"),
    "table_xai_metrics": ("Quantitative explainability comparison of three "
                          "attribution methods (mean over test patients).", "tab:xai"),
    "table_xai_agreement": ("Cross-method explanation agreement "
                            "(top-$k$ Jaccard and Spearman rank correlation).",
                            "tab:xai_agree"),
    "table_learnable_graph": ("Prior-regularised learnable topology versus the "
                             "fixed Pearson+MST graph (mean $\\pm$ std over "
                             "3 seeds $\\times$ 5 folds).", "tab:learnable"),
    "table_learnable_graph_delta": ("Change in performance relative to the fixed "
                                    "graph when the topology is learned.",
                                    "tab:learnabledelta"),
    "table_learnable_stats": ("Paired Wilcoxon signed-rank tests over the 15 "
                              "matched folds for the learnable-topology study.",
                              "tab:learnablestats"),
    "table_learned_edges": ("Edges most strengthened and most pruned by the "
                            "learned topology relative to the Pearson prior "
                            "$A_0$.", "tab:learnededges"),
    "table_pooled_oof": ("Pooled out-of-fold point metrics.", "tab:pooled"),
    # --- node-identity falsification test ---
    "table_node_identity": ("Node-identity falsification test. Adding a "
                            "learnable per-feature embedding raises the "
                            "effective rank only when the graph is removed; "
                            "with edges present, message passing smooths the "
                            "injected identity away and accuracy falls. "
                            "Mean $\\pm$ std over 3 seeds $\\times$ 5 folds.",
                            "tab:nodeid"),
    "table_node_identity_gap": ("Internal graph gap "
                                "$\\Delta=\\mathrm{AUC}(\\text{Corr+MST})-"
                                "\\mathrm{AUC}(\\text{control})$ under each node "
                                "encoding, paired over the 15 matched "
                                "(seed, fold) runs.", "tab:nodeidgap"),
    "table_node_identity_sweep": ("Identity-embedding dimension sweep "
                                 "($d\\in\\{0,2,4,8,16,32\\}$) across three "
                                 "topologies. Mean $\\pm$ std over 3 seeds "
                                 "$\\times$ 5 folds.", "tab:nodeidsweep"),
    "table_node_identity_gap_sweep": ("Internal graph gap by identity-embedding "
                                     "dimension, Holm-corrected across all 12 "
                                     "(dimension, contrast) hypotheses jointly.",
                                     "tab:nodeidgapsweep"),
    # --- tau-regime analysis of graph connectivity ---
    "table_mst_activation_ext": ("MST activation on the reduced 8-feature "
                                 "transportable set. Unlike the 13-feature "
                                 "graph, this graph already fragments at "
                                 "$\\tau=0.15$.", "tab:mstactext"),
    "table_mst_transport": ("External transport by graph topology at "
                            "$\\tau=0.20$ (mean $\\pm$ std over 10 seeds). "
                            "Internal and external orderings dissociate.",
                            "tab:msttransport"),
    "table_mst_transport_stats": ("Pairwise contrasts against Corr+MST at "
                                  "$\\tau=0.20$, with paired patient-level "
                                  "bootstrap intervals and Holm-adjusted "
                                  "DeLong tests.", "tab:msttransportstats"),
    "table_mst_trend": ("Page's trend test for the ordered alternative "
                        "AUC(no graph) $<$ AUC(corr only) $<$ AUC(corr+MST). "
                        "Blocks are seeds; the test quantifies reproducibility "
                        "across initialisations, not patient-level "
                        "generalisation.", "tab:msttrend"),
    "table_mst_selectivity": ("Selectivity control. A fully connected graph is "
                              "also a single component, isolating selectivity "
                              "from connectivity.", "tab:mstselect"),
    "table_dataset_summary": ("UCI Cleveland heart-disease dataset summary "
                              "(303 patients, 13 features).", "tab:datasetsummary"),
    "table_feature_descriptives": ("Descriptive statistics of the 13 clinical "
                                   "features.", "tab:features"),
    "table_runtime_memory": ("Wall-clock time and peak memory for one 5-fold "
                             "CV pass (single seed, CPU).", "tab:runtime"),
    # --- external validation ---
    "table_external_cohorts": ("External cohort availability. Complete-case counts "
                               "use the 8-feature transportable set; \\texttt{ca}, "
                               "\\texttt{thal} and \\texttt{slope} are not recorded "
                               "outside Cleveland.", "tab:extcohorts"),
    "table_external_reference": ("Internal reference for the reduced 8-feature model "
                                 "on Cleveland, separating the cost of feature "
                                 "reduction from the cost of cohort shift.", "tab:extref"),
    "table_external_validation": ("External validation of the transported 8-feature "
                                  "models (mean $\\pm$ std over 5 seeds). Scaler, "
                                  "graph, weights and decision threshold are fit on "
                                  "Cleveland only.", "tab:external"),
    "table_external_stats": ("Statistical comparison on the external cohorts. "
                             "DeLong's test compares correlated ROC curves; "
                             "McNemar's test compares thresholded predictions.",
                             "tab:extstats"),
    # --- deployment characterisation ---
    "table_deployment": ("Deployment characteristics of the feature-node "
                         "model. Inference is inductive: patients are scored "
                         "individually with no cohort present.",
                         "tab:deploy"),
    "table_calibration": ("Calibration of predicted risks. Brier score and "
                          "expected calibration error (ECE, 10 bins), "
                          "out-of-fold on Cleveland and on each external "
                          "cohort. Lower is better.", "tab:calib"),
    "table_decision_curve_summary": ("Net benefit at representative risk "
                                     "thresholds $p_t$, against the treat-all "
                                     "strategy.", "tab:dca"),
    # --- bootstrap uncertainty on the external cohorts ---
    "table_external_bootstrap_auc": ("External ROC-AUC with nested bootstrap "
                                     "95\\% confidence intervals ($B=2000$), "
                                     "propagating both patient-sampling and "
                                     "model-initialisation uncertainty.",
                                     "tab:extboot"),
    "table_external_bootstrap_delta": ("Paired bootstrap 95\\% confidence "
                                       "intervals for $\\Delta$AUC against the "
                                       "feature-node GCN. An interval "
                                       "containing zero indicates the "
                                       "advantage is not established on that "
                                       "cohort.", "tab:extbootdelta"),
    # --- formulation comparison: feature-node vs patient-similarity ---
    "table_patient_knn_sweep": ("Patient-similarity baseline swept over the "
                                "neighbourhood size $k$ (mean $\\pm$ std over "
                                "3 seeds $\\times$ 5 folds). $k$ is selected on "
                                "Cleveland and never re-tuned externally.",
                                "tab:knnsweep"),
    "table_patient_vs_feature": ("Feature-node versus patient-similarity graph "
                                 "formulation under the identical "
                                 "cross-validation protocol.", "tab:formulation"),
    "table_patient_vs_feature_stats": ("Statistical comparison of the two "
                                       "formulations on pooled out-of-fold "
                                       "predictions, per seed. Rows are seed "
                                       "replicates of a single hypothesis, not "
                                       "a family of hypotheses, so no "
                                       "multiplicity correction is applied.",
                                       "tab:formstats"),
    "table_patient_external": ("External transport of the conventional "
                               "patient-similarity GNN (mean $\\pm$ std over "
                               "5 seeds).", "tab:patext"),
    "table_patient_vs_feature_external": ("External ROC-AUC of the two "
                                          "formulations. Positive $\\Delta$AUC "
                                          "favours the feature-node model.",
                                          "tab:formext"),
    # --- convolution operator ablation ---
    "table_conv_comparison": ("Ablation over the message-passing operator. "
                              "Graph construction, depth, width, readout and "
                              "protocol are held fixed; only the convolution "
                              "varies.", "tab:convs"),
    "table_conv_stats": ("Statistical comparison of alternative operators "
                         "against GCN on pooled out-of-fold predictions.",
                         "tab:convstats"),
    # --- graph-construction rule: global cut-off vs. per-node k-NN ---
    "table_mst_activation": ("Activation of the MST augmentation as a function "
                             "of $\\tau$. Below $\\tau=0.20$ the thresholded "
                             "correlation graph is already connected and the MST "
                             "adds no edges; above it the MST supplies the "
                             "bridges that keep the graph connected.",
                             "tab:mstactivation"),
    "table_knn_structure": ("Structure of the feature graph under a global "
                            "correlation cut-off versus per-node $k$-NN "
                            "construction (across the 5 folds).", "tab:knnstruct"),
    "table_knn_topology": ("Global correlation cut-off versus adaptive per-node "
                           "$k$-NN graph construction (mean $\\pm$ std over "
                           "3 seeds $\\times$ 5 folds).", "tab:knn"),
    "table_knn_stats": ("Paired Wilcoxon signed-rank tests over the 15 matched "
                        "folds comparing $k$-NN construction against the "
                        "Pearson+MST graph.", "tab:knnstats"),
    "table_mst_connectivity": ("What the MST augmentation changes. At "
                               "$\\tau=0.15$ it is inert; at $\\tau=0.20$ it "
                               "restores a single connected component in every "
                               "fold.", "tab:mstconn"),
    "table_mst_tau_comparison": ("Correlation graph with and without MST "
                                 "augmentation, at a threshold where the MST is "
                                 "inert ($\\tau=0.15$) and one where it is "
                                 "active ($\\tau=0.20$).", "tab:msttau"),
    "table_mst_tau_stats": ("Paired tests for the MST augmentation. Where the "
                            "MST is active it restores connectivity without "
                            "changing any metric significantly.",
                            "tab:msttaustats"),
    # --- what message passing over scalar nodes contributes ---
    "table_feature_neighbourhoods": ("Fold-stable feature neighbourhoods: "
                                     "neighbours retained in all five folds. "
                                     "These are the relations along which a "
                                     "feature's representation is contextualised.",
                                     "tab:neighbourhoods"),
    "table_representation_rank": ("Participation-ratio effective rank of the "
                                  "$13\\times32$ node-embedding matrix on the "
                                  "trained model (mean $\\pm$ std over test "
                                  "patients). With one scalar per node the "
                                  "layer-1 pre-activation is rank-one in "
                                  "direction by construction.", "tab:rank"),
    "table_topology_significance": ("Paired Wilcoxon tests, Holm-corrected over "
                                    "nine hypotheses, isolating what the graph "
                                    "topology contributes.", "tab:toposig"),
    "table_topology_significance_perf": ("Predictive performance of the topology "
                                         "variants tested in "
                                         "Table~\\ref{tab:toposig}.",
                                         "tab:toposigperf"),
}

# Tables that must span both columns.
WIDE = {
    "table_model_comparison", "table_statistical_tests", "table_ablation",
    "table_ablation_delta", "table_threshold_sensitivity", "table_pooled_oof",
    "table_runtime_memory", "table_xai_metrics", "table_external_cohorts",
    "table_external_reference", "table_external_validation", "table_external_stats",
    "table_feature_descriptives", "table_xai_agreement",
    "table_external_bootstrap_auc", "table_external_bootstrap_delta",
    "table_calibration", "table_decision_curve_summary",
    "table_patient_knn_sweep", "table_patient_vs_feature",
    "table_patient_vs_feature_stats", "table_patient_external",
    "table_conv_comparison",
    "table_knn_topology", "table_knn_stats", "table_mst_activation",
    "table_mst_tau_comparison", "table_mst_tau_stats",
    # Widened by the Holm adjusted-p / verdict columns; these no longer fit
    # a single IEEE column and overflow badly if left as `table`.
    "table_conv_stats", "table_learnable_graph", "table_learnable_stats",
    "table_learnable_graph_delta",
    "table_topology_significance", "table_feature_neighbourhoods",
    # tau-regime analysis
    "table_mst_transport", "table_mst_transport_stats", "table_mst_trend",
    "table_mst_selectivity", "table_mst_activation_ext",
    # node-identity falsification test
    "table_node_identity", "table_node_identity_gap",
    "table_node_identity_sweep", "table_node_identity_gap_sweep",
}

# Columns to omit from the typeset table (prose that lives in the text).
DROP_COLS = {
    "table_ablation": ["Question", "Contribution"],
    # Precision is recoverable from the other columns and costs width.
    "table_model_comparison": ["Precision"],
    "table_pooled_oof": ["Precision"],
    "table_external_validation": ["Accuracy", "Precision"],
    "table_external_reference": ["Precision"],
    # 'Regime' is prose that the surrounding text already states.
    "table_mst_tau_stats": ["Regime"],
    "table_learnable_stats": ["Question"],
    # Stated in the caption instead of repeated on every row.
    "table_patient_vs_feature_stats": ["Multiplicity"],
    # The question each comparison answers is stated in the running text.
    "table_topology_significance": ["Question"],
    # Discrimination (ROC-AUC) carries the transport argument; the
    # threshold-dependent metrics are reported in the text where relevant.
    "table_mst_transport": ["Accuracy", "Recall", "Specificity"],
    # Raw p-values are superseded by the Holm-adjusted columns.
    "table_mst_transport_stats": ["Wilcoxon p (seeds)", "DeLong p (patients)",
                                  "Holm p (Wilcoxon)"],
    "table_representation_rank": ["Max possible"],
    # Accuracy and F1 track MCC here; rank and ROC-AUC carry the argument.
    "table_node_identity": ["Accuracy", "F1"],
    "table_node_identity_sweep": ["Accuracy"],
}

# Long CSV headers -> compact typeset headers.
RENAME = {
    "table_statistical_tests": {
        "McNemar_b(GCN-wrong,other-right)": "$b$",
        "McNemar_c(GCN-right,other-wrong)": "$c$",
        "McNemar_stat": "McNemar stat",
        "McNemar_p": "McNemar $p$",
        "Wilcoxon_p(foldF1)": "Wilcoxon $p$",
    },
    "table_external_stats": {
        "AUC (GCN)": "AUC$_{\\mathrm{GCN}}$",
        "AUC (other)": "AUC$_{\\mathrm{other}}$",
        "DeLong p": "DeLong $p$",
        "McNemar p": "McNemar $p$",
    },
}

# Tables the character heuristic under-estimates (wide numeric columns with
# many decimals), confirmed overfull in the compiled log.
FORCE_FIT = {
    "table_threshold_sensitivity",
}

ALIGN_OVERRIDE = {
    # Both columns wrap: an `l` first column is as wide as its longest label
    # and pushes these past the 8.8cm single-column width.
    "table_deployment": "p{3.1cm}p{4.6cm}",
    "table_hyperparameters": "p{3.1cm}p{4.6cm}",
    "table_dataset_summary": "p{3.1cm}p{4.6cm}",
    "table_feature_descriptives": "llrrrrrr",
}


def _visible(row):
    """Approximate printed width of a LaTeX table row, in characters.

    Control sequences render as one glyph or none, and $ { } & \\\\ are
    markup rather than ink, so counting raw characters badly overestimates
    width. This is only a heuristic used to decide whether a table needs
    scaling, not a typesetting calculation."""
    s = row.replace("\\\\", "")
    s = re.sub(r"\\[a-zA-Z]+", "x", s)      # macro -> ~1 glyph
    s = re.sub(r"[${}]", "", s)             # math delimiters/grouping
    s = s.replace("&", "  ")                # column gap
    return s


def escape(s):
    s = str(s)
    # protect already-mathmode / macro content produced upstream
    if "$" in s or "\\texttt" in s:
        return s
    for a, b in [("±", "$\\pm$"), ("τ", "$\\tau$"), ("↓", "$\\downarrow$"),
                 ("×", "$\\times$"), ("−", "$-$"), ("→", "$\\to$"),
                 ("Δ", "$\\Delta$"),
                 # Greek letters must reach pdfTeX in math mode: a bare
                 # \lambda / \alpha in text mode is a hard compile error.
                 ("λ", "$\\lambda$"), ("α", "$\\alpha$"), ("μ", "$\\mu$"),
                 ("σ", "$\\sigma$"), ("β", "$\\beta$"), ("ρ", "$\\rho$"),
                 ("γ", "$\\gamma$"), ("θ", "$\\theta$"), ("λ", "$\\lambda$"),
                 # dashes
                 ("—", "---"), ("–", "--"),
                 ("&", "\\&"), ("%", "\\%"),
                 ("_", "\\_"), ("#", "\\#")]:
        s = s.replace(a, b)
    return s


def to_latex(csv_path):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    if name not in CAPTIONS:
        return None
    df = pd.read_csv(csv_path)

    for c in DROP_COLS.get(name, []):
        if c in df.columns:
            df = df.drop(columns=c)
    df = df.rename(columns=RENAME.get(name, {}))

    caption, label = CAPTIONS[name]
    cols = list(df.columns)
    align = ALIGN_OVERRIDE.get(name, "l" + "r" * (len(cols) - 1))
    env = "table*" if name in WIDE else "table"
    size = "\\footnotesize" if name in WIDE else "\\small"

    header = " & ".join(escape(c) for c in cols) + " \\\\"
    body = [" & ".join(escape(row[c]) for c in cols) + " \\\\"
            for _, row in df.iterrows()]

    # Fit-to-width. A p{} column wraps, so only fixed-width columns are
    # measured. Budget is the printable width in characters at the chosen
    # size; anything past it is scaled down with \resizebox so that no table
    # ever runs into the margin (IEEE reviewers flag overfull boxes).
    if name in FORCE_FIT:
        fit = True
    elif "p{" not in align:
        widest = max(len(_visible(x)) for x in [header] + body)
        budget = 118 if env == "table*" else 53
        fit = widest > budget
    else:
        fit = False

    lines = [
        f"\\begin{{{env}}}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
    ]
    if fit:
        lines.append("\\resizebox{" +
                     ("\\textwidth" if env == "table*" else "\\columnwidth") +
                     "}{!}{%")
    lines += [f"\\begin{{tabular}}{{{align}}}", "\\toprule", header, "\\midrule"]
    lines += body
    lines += ["\\bottomrule", "\\end{tabular}"]
    if fit:
        lines.append("}")
    lines += [f"\\end{{{env}}}", ""]

    out = f"results/latex/{name}.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def main():
    made = []
    for csv_path in sorted(glob.glob("results/*.csv")):
        r = to_latex(csv_path)
        if r:
            made.append(r)
    print(f"Generated {len(made)} LaTeX tables:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
