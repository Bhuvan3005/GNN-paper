# -*- coding: utf-8 -*-
"""
run_external_figures.py
========================
Figures + LaTeX tables for the external-validation study.

fig31_external_auc.png    grouped AUC bars per cohort with seed-std error bars
fig32_external_shift.png  internal->external degradation, and the
                          specificity collapse of the baselines under shift
"""

import os
import re
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

os.makedirs("figures", exist_ok=True)
os.makedirs("results/latex", exist_ok=True)

COLORS = {"GCN (Ours)": "#2ECC71", "Logistic Regression": "#3498DB",
          "Random Forest": "#9B59B6"}
MODELS = ["GCN (Ours)", "Logistic Regression", "Random Forest"]


def parse_ms(s):
    m = re.match(r"\s*([\-\d.]+)\s*(?:±|\+/-)\s*([\-\d.]+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (float(s), 0.0)


def esc(s):
    s = str(s)
    for a, b in [("±", "$\\pm$"), ("Δ", "$\\Delta$"), ("_", "\\_"),
                 ("%", "\\%"), ("&", "\\&")]:
        s = s.replace(a, b)
    return s


def write_table(name, header, rows, caption, label, align=None):
    align = align or ("l" * len(header))
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\small", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(esc(h) for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(esc(x) for x in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(f"results/latex/{name}.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote results/latex/" + name + ".tex")


def main():
    ext = pd.read_csv("results/table_external_validation.csv")
    info = pd.read_csv("results/table_external_cohorts.csv")
    stats = pd.read_csv("results/table_external_stats.csv")
    ref = pd.read_csv("results/table_external_reference.csv")

    cohorts = list(dict.fromkeys(ext["Cohort"]))

    # ---------------- fig31: AUC per cohort ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(len(cohorts)); width = 0.26
    for i, mdl in enumerate(MODELS):
        means, stds = [], []
        for c in cohorts:
            row = ext[(ext.Cohort == c) & (ext.Model == mdl)].iloc[0]
            m, s = parse_ms(row["ROC-AUC"]); means.append(m); stds.append(s)
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               label=mdl, color=COLORS[mdl], alpha=0.9, edgecolor="white")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x + width); ax.set_xticklabels(cohorts, fontsize=9)
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 1.0)
    ax.set_title("External validation: ROC-AUC by cohort\n"
                 "(mean $\\pm$ std over 5 seeds)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)

    # ---------------- specificity collapse ----------------
    ax2 = axes[1]
    for i, mdl in enumerate(MODELS):
        means, stds = [], []
        for c in cohorts:
            row = ext[(ext.Cohort == c) & (ext.Model == mdl)].iloc[0]
            m, s = parse_ms(row["Specificity"]); means.append(m); stds.append(s)
        ax2.bar(x + i * width, means, width, yerr=stds, capsize=4,
                label=mdl, color=COLORS[mdl], alpha=0.9, edgecolor="white")
    ax2.set_xticks(x + width); ax2.set_xticklabels(cohorts, fontsize=9)
    ax2.set_ylabel("Specificity"); ax2.set_ylim(0, 1.05)
    ax2.set_title("Specificity under distribution shift\n"
                  "(Random Forest collapses toward majority-class prediction)", fontsize=11)
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig31_external_auc.png", dpi=170, bbox_inches="tight")
    plt.close()

    # ---------------- fig32: degradation ----------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    internal_auc, internal_sd = parse_ms(ref.iloc[0]["ROC-AUC"])
    labels = ["Cleveland\n(internal CV)"] + [c.replace(" ", "\n") for c in cohorts]
    vals, errs = [internal_auc], [internal_sd]
    for c in cohorts:
        m, s = parse_ms(ext[(ext.Cohort == c) & (ext.Model == "GCN (Ours)")].iloc[0]["ROC-AUC"])
        vals.append(m); errs.append(s)
    bars = ax.bar(labels, vals, yerr=errs, capsize=5,
                  color=["#16A085"] + ["#2ECC71"] * len(cohorts),
                  alpha=0.9, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(internal_auc, color="#16A085", ls="--", alpha=0.6,
               label="Cleveland internal reference")
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 1.0)
    ax.set_title("Transportability of the feature-node GCN\n"
                 "(8-feature transportable model)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig32_external_shift.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("wrote figures/fig31_external_auc.png, figures/fig32_external_shift.png")
    # LaTeX tables for these CSVs are emitted centrally by make_latex_tables.py
    # (which handles full-width `table*` layout for the wide ones).


if __name__ == "__main__":
    main()
