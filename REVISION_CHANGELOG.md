# Revision changelog — response to reviewer-requested fixes

Date: 2026-08-27. Scope: five action items from the reviewer note, implemented
against `paper/main.tex`, `paper/references.bib`, `apply_holm.py`,
`make_latex_tables.py`, and the experiment results under `results/`.
`paper/main.pdf` was rebuilt with `tectonic` after every change; the final
build has zero undefined references/citations and zero overfull boxes
(21 pages, up from 19 before this revision).

## 1. Added XGBoost and LightGBM baselines under the identical protocol

`run_main.py`'s baseline factory was extended with `XGBClassifier` and
`LGBMClassifier`, run under the same 5-fold x 3-seed `StratifiedKFold` splits,
in-fold `StandardScaler`, and inner-validation threshold tuning already used
for the other baselines. Updated results:

| Model | ROC-AUC |
|---|---|
| GCN (Ours) | 0.9056 +/- 0.0408 |
| Logistic Regression | 0.9025 +/- 0.0385 |
| Random Forest | 0.9044 +/- 0.0336 |
| XGBoost | 0.8845 +/- 0.0332 |
| LightGBM | 0.8817 +/- 0.0355 |
| Gradient Boosting | 0.8721 +/- 0.0390 |
| MLP | 0.8680 +/- 0.0320 |

McNemar and Wilcoxon tests (Holm-corrected) show no statistically significant
difference between the GCN and either boosted-tree baseline (GCN vs XGBoost:
Holm p = 1.0 on both tests; GCN vs LightGBM: Holm p = 1.0 and 0.606). Results,
Discussion, abstract, contributions list, and the Baselines subsection in
`main.tex` were updated to describe six baselines instead of four, and two new
verified BibTeX entries (`chen2016xgboost`, `ke2017lightgbm`) were added.

**Note on baseline-number drift:** re-running Random Forest/Logistic
Regression under the current environment initially produced numbers that
drifted slightly from the paper's original cached values. Root cause: the
installed scikit-learn version (1.9) was newer than the version the
manuscript's Reproducibility section declares. Rather than renumber the whole
paper, scikit-learn was pinned back to the declared version for this run;
after pinning, Random Forest and Gradient Boosting reproduced the original
cached numbers exactly. No `requirements.txt`/environment-lock file exists in
the repo -- the paper's Reproducibility section remains the sole authoritative
version declaration.

Artifacts: `table_model_comparison.csv`, `table_statistical_tests.csv`,
`fig_baseline_comparison.png`.

## 2. Identity-embedding dimension sweep (d = 0, 2, 4, 8, 16, 32)

The single previously-reported identity-embedding data point (d=8) was
generalized into a sweep over d in {2,4,8,16,32} against the scalar (d=0)
baseline, holding topology (no graph / Corr+MST / fully connected) and the
3-seed x 5-fold protocol fixed. Finding: the direction of the "graph beats
no-graph" advantage seen at d=0 (delta ROC-AUC = +0.0089, Corr+MST vs
no-graph) reverses sign once nodes carry an identity embedding (e.g. d=4:
-0.0081; d=32: -0.0605), and the graph-augmented runs become increasingly
unstable at higher dimension (ROC-AUC std under Corr+MST grows from 0.041 at
d=0 to 0.185 at d=32, versus 0.037 for the no-graph condition at the same
dimension). None of the individual dimension x contrast comparisons reach
Holm-corrected significance, but the *direction* of the effect is consistent
across all five swept dimensions once identity is present.

The two new sweep tables (`table_node_identity_sweep.csv`,
`table_node_identity_gap_sweep.csv`) were registered as a new Holm-corrected
statistical family in `apply_holm.py` (12 dimension x contrast hypotheses),
added to `make_latex_tables.py` as wide (`table*`) floats, and the mechanism/
falsification-test paragraph in `main.tex` was rewritten to report the full
sweep instead of the single d=8 point.

Artifacts: `table_node_identity_sweep.csv`, `table_node_identity_gap_sweep.csv`,
`fig_identity_sweep.png`.

## 3. tau-regime wording softened to "suggestive evidence"

All instances of overclaiming language around the MST/tau-regime sparse-
topology claim ("we establish...", "confirms...") were replaced with the
reviewer's proposed hedged phrasing ("suggestive evidence... may be more
favorable") in the Discussion, the tau-regime section's own concluding
sentence, and the Introduction's contributions bullet. The Abstract did not
state this claim strongly to begin with and required no change.

## 4. "External validation" renamed to "cross-site validation"

The "External validation" section title and all supporting mentions
(limitations list, contributions list, running text) were renamed to a
"cross-site" framing that explicitly discloses that the Hungarian,
Switzerland, and Long-Beach-VA cohorts share the same underlying UCI
heart-disease collection lineage as the primary Cleveland cohort, rather than
implying independent external data sources. All existing statistics were
preserved unchanged.

## 5. Related Work expanded with verified recent literature

Real, DOI-verified 2023-2026 papers (via OpenAlex/CrossRef) on clinical/
biomedical GNNs, patient-similarity graphs, feature-node/feature-graph
approaches, tabular Transformer/attention architectures, and explainable
tabular ML were added: 9 new BibTeX entries in `references.bib` (37 total, up
from 28), plus 2 more for XGBoost/LightGBM (see item 1) -- 39 entries in
total. Two new paragraphs were added to the Related Work section citing this
literature and giving concrete, citation-backed support for the "feature-node
graphs are under-explored" claim, replacing what had been an unsupported
assertion.

## Build verification

`tectonic -X compile main.tex --keep-logs` -- 0 undefined citations/
references, 0 overfull boxes, 21 pages, `main.pdf` regenerated at
`paper/main.pdf`.
