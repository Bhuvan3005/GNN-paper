# Feature-Node Graph Representation for Cardiovascular Risk Prediction

Reproducibility package for the manuscript

> **Beyond Accuracy: A Feature-Node Graph Representation for Interpretable,
> Deployable and Transportable Cardiovascular Risk Prediction**

Every table and figure in the paper is regenerated end-to-end by the code in
this repository. No reported number is transcribed by hand.

---

## What the method is

Each patient is an **independent graph** whose 13 nodes are the clinical
features of the UCI Cleveland cohort and whose edges encode feature–feature
Pearson dependence, thresholded at `τ = 0.15` and augmented with a minimum
spanning tree to guarantee a single connected component. A two-layer GCN with
global mean pooling scores each patient graph.

This is the **opposite** of the usual clinical GNN: nodes are features, not
patients, so inference is inductive — a single patient is scored with no cohort
present.

## Headline finding

Under a leakage-controlled protocol shared identically with four baselines, the
model is **statistically indistinguishable** from Logistic Regression
(McNemar *p* = 0.70) and Random Forest (*p* = 0.77). We report this parity as a
finding, not an improvement; the case for the formulation rests on
representation, deployability and transportability instead.

---

## Setup

```bash
pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn networkx scipy statsmodels captum
```

Runs on CPU. Python 3.13, PyTorch 2.x, PyTorch Geometric 2.7, scikit-learn 1.7.

## Reproducing the results

Run from the repository root, in this order:

```bash
python run_main.py
```

| Stage | Script | Produces |
|---|---|---|
| Core CV + baselines + significance | `run_main.py` | model comparison, pooled OOF, McNemar/Wilcoxon |
| Graph statistics + hyperparameters | `run_graphstats.py` | graph topology table |
| Ablation (topology, depth, pooling, dropout, width) | `run_ablation.py` | ablation + deltas |
| Correlation-threshold sweep | `run_sensitivity.py` | τ sensitivity |
| Explainability (GNNExplainer, IG, Saliency) | `run_xai.py` | XAI metrics, agreement, deletion/insertion |
| Convolution operators (GAT, SAGE, GIN) | `run_conv_comparison.py` | operator comparison |
| External validation (Hungarian, Switzerland, VA) | `run_external.py` | transport tables |
| Patient-similarity GNN comparison | `run_patient_comparison.py` | inductive-vs-transductive contrast |
| Deployment characteristics | `run_deployment.py` | latency, size, calibration |
| Multiple-comparison control | `apply_holm.py` | Holm-adjusted columns |

`pipeline.py` holds the shared core: fold-specific graph construction, in-fold
scaling, the GCN, training, and metrics. Fixed seeds (42, 7, 123) throughout.

## Building the manuscript

```bash
python make_latex_tables.py && python make_combined_tables.py && bash paper/build.sh
```

`make_latex_tables.py` emits one LaTeX table per `results/*.csv`;
`make_combined_tables.py` merges paired result/significance tables into
two-panel floats and **must run after it**.

To typeset with the official IEEE Access class, drop `ieeeaccess.cls` beside
`paper/main.tex` — the file auto-detects it and needs no edits. Without it the
build falls back to `IEEEtran`.

## Layout

```
pipeline.py            shared core (graph construction, GCN, CV, metrics)
run_*.py               one experiment each, writes results/*.csv
make_latex_tables.py   CSV -> LaTeX
make_combined_tables.py  merges paired tables into two-panel floats
results/               all result tables (CSV + latex/)
figures/               all manuscript figures
paper/                 main.tex, references.bib, build.sh, main.pdf
data_external/         Hungarian / Switzerland / VA cohorts
archive/               original exploratory notebook export
```

## Data

UCI Machine Learning Repository, Heart Disease (DOI 10.24432/C52P4X).
Cleveland (n = 303) is used for development; Hungarian, Switzerland and
Long-Beach-VA are held out entirely for external validation and touched once.
External cohorts support only 8 of the 13 features, which bounds the transport
claims — this is stated explicitly in the paper.

## Note on synthetic data

`make_synthetic.py` regenerates the synthetic cohorts (seed 42) used for the
mechanism experiments. The generated CSVs are not committed because they are
derived artefacts and reproduce byte-for-byte.
