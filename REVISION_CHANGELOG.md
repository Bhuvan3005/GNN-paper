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

---

# Revision changelog (continued) — response to second-round critique

Date: 2026-08-29. Scope: a second, more detailed critique document covering
statistical-testing completeness, graph-construction robustness,
terminology/framing, page length, and outstanding submission placeholders.
Implemented against `paper/main.tex`, `pipeline.py`, `run_main.py`, and new
scripts `run_mixed_association.py` / `supplementary.tex`. `paper/main.pdf`
was rebuilt with `tectonic` after every change.

## 1. Statistical-testing gap: GB and MLP omitted from significance tests (Priority-0)

`run_main.py`'s McNemar/Wilcoxon significance-testing loop only tested the
GCN against 4 of the 6 baselines reported in the main comparison table,
silently omitting Gradient Boosting and MLP even though their fold-level
predictions were already cached on disk. Extended the loop to all six
baselines, reran `apply_holm.py` for the Holm correction over the enlarged
family, regenerated `table_statistical_tests.tex`/`.csv`, and rewrote the
"Statistical significance" subsection and the abstract's parenthetical
p-values in `main.tex` to report all six comparisons. None reach significance
after correction (Holm-adjusted p >= 0.57).

*Scope note:* the user's answer to the ask-user question that authorized this
fix was later found, on transcript review, to say "narrow the abstract to 4
baselines" rather than "add the missing 2" — the opposite direction from what
was implemented. Flagged explicitly to the user, who chose to keep the
six-baseline fix as done (it is the more complete response to the critique
and was already tested/compiled) rather than revert.

Artifacts: `table_statistical_tests.csv`, `table_model_comparison.csv`,
`table_pooled_oof.csv`, `run_main.py`.

## 2. Mixed-type association robustness check (Priority-0, background only)

The critique observed that Pearson correlation treats nominal/categorical
codes (e.g. `cp`, `thal`, `slope`) as ordered scalars during graph
construction. Implemented a bias-corrected Cramer's V / correlation-ratio
mixed-type association measure as a drop-in alternative to Pearson in
`pipeline.py` (new `mixed_mst`/`mixed_only` topology options), reran the full
5-fold x 3-seed ablation via a new `run_mixed_association.py` script mirroring
the existing `run_knn_topology.py` convention (self-contained Holm
correction), and compared against the current Pearson-based graph with paired
significance testing. Finding: no significant difference on F1/ROC-AUC/MCC
(Holm-adjusted p >= 0.17) — the two association measures are statistically
indistinguishable on this cohort.

**By explicit user decision, this stays as verified background analysis
only.** The paper's Pearson-based method, and all existing tables/figures
built on it, are unchanged; the result is not cited or referenced anywhere in
`main.tex`.

Artifacts: `fig_mixed_association.png`, `table_mixed_association.csv`,
`table_mixed_association_stats.csv`, `table_mixed_association_structure.csv`,
`run_mixed_association.py`.

## 3. Additional baselines (TabPFN / CatBoost) — declined

The critique suggested adding TabPFN and/or CatBoost as further baselines.
Asked the user; answer was **"Add neither"** — the existing six baselines
(including two boosted-tree models) were judged sufficient for this revision
pass. No code or manuscript change made.

## 4. Framing and terminology rewrite pass

Per the user's approval of "all three" framing changes and the full set of
wording fixes, `main.tex` was edited throughout:

- Title revised (incorporating "When Accuracy Saturates" framing).
- A novelty-scope sentence added immediately before the C1–C3 contributions
  list, narrowing the claimed novelty explicitly to the feature-node
  formulation and its evaluation framework rather than the convolution
  operator itself.
- "Clinically meaningful feature relationships" language softened in the
  introduction and mechanism-discussion sections.
- Confirmed "relational reasoning" appears only inside the falsification-test
  paragraph, where it is explicitly posed and rejected as a hypothesis — left
  as-is, since this usage is correct rather than an overclaim.
- "External validation" renamed to "cross-cohort transport" across the
  keywords block, a figure caption, and body prose, to avoid re-litigating
  the independent-data-source framing already fixed in the first revision
  round (item 4 above).
- Added an explicit 13-vs-8-feature transport disclosure sentence to the
  abstract.
- Added a provenance caveat around the claim that feature-importance rankings
  agree with "the clinical literature," clarifying the comparison is against
  risk factors reported in prior cardiovascular studies (citing
  `detrano1989`, `petch2022opening`) rather than a blinded clinician panel.
- Checked the deployment section's "2.6ms latency" framing (critique's
  "don't oversell it" point) and the existing related-work comparison table
  (`table_literature_comparison`) — both already satisfied the critique as
  written; no change needed.

## 5. Duplicate table-number citation bug (found during rebuild)

Visual proofing of the rebuilt PDF surfaced two passages citing "Tables~X
and~Y" where X and Y were two `\label`s attached to the *same* `table*`
environment (a convention used elsewhere in the manuscript for two-panel
tables), so both resolved to the identical printed table number. Fixed both
citations (the graph-construction-rule paragraph citing
`tab:toposigperf`/`tab:toposig`, and the MST-activation paragraph citing
`tab:msttau`/`tab:msttaustats`) to cite a single table number each. No
underlying data or claim changed — citation wording only.

## 6. Page-reduction pass

The user requested a dedicated page-reduction pass (target: under ~19–20
pages) rather than relying on other fixes' incidental space savings. Moved
four secondary robustness tables — full per-threshold sensitivity, the
convolution-operator comparison, and both identity-embedding dimension-sweep
tables — out of the printed manuscript into a new `supplementary.tex`
(compiles standalone to `supplementary.pdf`, 2 pages, IEEEtran class, 0
undefined refs). Each removed `\input{}` was replaced in `main.tex` with a
one-line pointer to the corresponding supplementary table number (S1–S4); no
number, claim, or in-text discussion was altered, only where the full
per-configuration table lives. Main manuscript: 21 -> 20 pages.

Artifacts: `main.tex`, `main.pdf`, `supplementary.tex`, `supplementary.pdf`.

## 7. Outstanding submission placeholders — left unresolved by user choice

The manuscript still contains author-identity and submission-logistics
placeholders that require real personal/institutional information the agent
cannot supply: author name, affiliation/department, city/country, e-mail,
IEEE membership grade, ORCID, ORCID/funding statement wording, the public
code-repository URL (currently `github.com/USERNAME/REPOSITORY`), and the
author biography paragraph + photo. All are already marked with
`SUBMISSION-TODO` comments in `main.tex` (lines ~58, ~570, ~1410) and were
confirmed to be the *only* remaining placeholder markers in the file (no
stray `TBD`/`FIXME`/`lorem` text). Asked the user how to handle these; the
user chose to **leave them as placeholders** rather than supply the details
now. These must be filled in by the author before actual IEEE Access
submission — the manuscript otherwise compiles and reads as final.

## Build verification (this round)

`tectonic --keep-logs main.tex` -- 0 undefined citations/references, 20
pages, `main.pdf` regenerated at `paper/main.pdf`. `tectonic --keep-logs
supplementary.tex` -- 0 undefined citations/references, 2 pages,
`supplementary.pdf` generated at `paper/supplementary.pdf`.

## Build 2026-08-29 (pre-submission review — stale-artefact correction)

Triggered by visual proofing of the compiled PDF, which showed Table VII
rendering only five models while the prose and Table VIII described six
baselines.

### P0 — Stale generated artefacts (all tables and figures in the PDF)

Root cause: `results/*.csv` were regenerated by the Aug 27/29 reruns that
added the XGBoost and LightGBM baselines, but the *derived* artefacts were
never rebuilt from them.

* **All 16 `results/latex/combined_*.tex` files dated Aug 12**, i.e. every
  table in the compiled manuscript was built before the reruns. Regenerated
  via `make_latex_tables.py` + `make_combined_tables.py` (requires
  `PYTHONPATH=.`).

  Of the 16, **two changed in content**; the other 14 regenerated
  byte-identical, confirming the reruns did not disturb the ablation,
  bootstrap, convergence, kNN, MST, topology, node-identity, patient-level
  or external-setup tables.

  - `combined_performance.tex` (Table VII) — now carries all seven models
    in both panels. GCN accuracy corrected 0.8109 -> 0.8131, ROC-AUC
    0.9051 -> 0.9056, MCC 0.6280 -> 0.6320, specificity 0.8196 -> 0.8237;
    Gradient Boosting ROC-AUC 0.8725 -> 0.8721.
  - `combined_xai.tex` — changed as a consequence of the `run_xai.py`
    rerun described below, not the baseline reruns.

  (An earlier draft of this changelog stated all 16 files changed. That was
  an error: the comparison that produced it was confounded by file-mode and
  line-ending differences in the backup copies. The git diff against the
  Aug 28 commit — whose tree still held the Aug-12 tables — is the
  authoritative comparison and shows two.)
* **All 27 `figures/*.png` dated Jul 25.** `run_paper_figures.py` carried a
  hardcoded five-entry `MODEL_ORDER`/`PALETTE`, so Fig. 7 (ROC/PR) omitted
  XGBoost and LightGBM even though `oof_predictions.npz` contained them.
  Extended both to seven models and regenerated the full set, plus
  `run_external_figures.py` and `run_xai.py`.

### P0 — Invalidated prose claim

* "Pooled out-of-fold metrics ... agree with the fold-averaged values to
  within 0.003 on every metric" was true of the old five-model table but
  false once the boosted trees and the GCN rerun were included (max
  deviation 0.0525). Rewritten to state what the data shows: ROC-AUC agrees
  within 0.005 for every model, while threshold-dependent metrics deviate
  more for the tree ensembles (0.053 specificity, Random Forest; 0.052 MCC,
  LightGBM), with model ordering unchanged.

### P1 — Stale XAI values

* `run_xai.py` crashed on NumPy 2.4 (`np.trapz` removed in NumPy 2.0), so
  the XAI tables/figure had not been rebuilt since Jul 25. Fixed with a
  version-safe `np.trapezoid` fallback and regenerated. Cross-method
  agreement shifted: Spearman rho 0.79 -> 0.80, top-k Jaccard 0.56 -> 0.59,
  corrected at both prose sites. All four qualitative XAI claims (IG
  sharpest deletion drop and fastest insertion recovery; Saliency most
  stable and least faithful) re-verified and still hold.

### P1 — Undisclosed analysis scope

* The transport, calibration and decision-curve analyses compare three
  models, not the seven of Table VII, which was never stated. Added an
  explicit scope sentence to the transport protocol paragraph naming the
  three and noting the boosted trees are reported in distribution only.

### Verified unaffected

* `table_dataset_characteristics`, `table_feature_descriptions` and
  `table_literature_comparison` remain dated Jul 25 by design — static
  reference content, not results-derived. Dataset table re-validated
  against the live cohort (303 samples, 13 features, 139/164 class split,
  0 missing, 0 duplicates): exact match.

### Final state

`main.pdf` 21 pages, `supplementary.pdf` 2 pages, 0 undefined
references/citations, 0 overfull boxes. Citation integrity: 43 keys used,
43 defined, no orphans in either direction. All 51 numeric values in the
prose trace to a regenerated table.
