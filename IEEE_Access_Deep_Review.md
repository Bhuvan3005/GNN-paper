# Deep Pre-Submission Review — IEEE Access
### "Beyond Accuracy: A Feature-Node Graph Representation for Interpretable, Deployable and Transportable Cardiovascular Risk Prediction"

*Reviewed against the manuscript source (`paper/main.tex`) as it currently stands. Per Rule 2/3 of the review protocol, missing author identity, affiliation, ORCID, funding statement, biography, and repository URL are submission metadata and are not evaluated here — those are all explicit `SUBMISSION-TODO` placeholders in the source, not omissions the authors are unaware of.*

---

## 1. Executive Review

| Category | Score |
|---|---:|
| Overall Quality | 7/10 |
| Novelty | 4/10 |
| Technical Soundness | 8/10 |
| Methodology | 8/10 |
| Experimental Validation | 8/10 |
| Results Quality | 7/10 |
| Reproducibility | 8/10 |
| Literature Review | 5/10 |
| Writing & Presentation | 7/10 |
| IEEE Access Suitability | 7/10 |

### Overall Recommendation: **Major Revision**

### Publication Readiness: **60%**

The manuscript is methodologically the strongest kind of "honest null result" paper: it never inflates a claim beyond what its own significance tests support, it shares folds and seeds identically across every model compared, it applies Holm–Bonferroni correction to every family of hypothesis tests, and its limitations section is unusually candid (twelve enumerated points, several of which most authors would bury). That rigor is real and should be credited. The problem for acceptance is not soundness, it is that the paper's own evidence repeatedly undercuts its central claims: the in-distribution accuracy story is an explicit null result by design; the external-transport advantage survives correction against only one baseline on two of three cohorts; and the paper's most interesting mechanistic finding (the τ-regime dissociation between internal and external topology preference) is stated to have only 3 of 9 contrasts excluding zero under patient-level bootstrap, with none surviving Holm correction on the DeLong tests. A reviewer sympathetic to honest reporting will still ask: what, exactly, is the paper asking a reader to newly believe from now on? The literature review is also thin (28 references, largely canonical GNN/statistics citations) relative to what IEEE Access reviewers typically expect for a paper making infrastructure/deployment claims. These are fixable with revision, not fatal flaws.

---

## 2. Title Review

**Current title:** "Beyond Accuracy: A Feature-Node Graph Representation for Interpretable, Deployable and Transportable Cardiovascular Risk Prediction"

### Current Title Assessment
Accurate and non-overclaiming. It correctly signals that the contribution is about representation properties (interpretable/deployable/transportable), not predictive superiority, which matches the paper's own explicit disclaimers in the Introduction ("we deliberately do not frame this work as a pursuit of higher accuracy").

### Problems
None substantive. "Beyond Accuracy" is a reasonable framing device but is somewhat generic — a search for that phrase will collide with many other papers using the same rhetorical hook in ML-for-healthcare venues (literature to verify: prevalence of "Beyond Accuracy" as a title prefix in recent clinical-ML papers).

### Recommended Title
No change recommended. The title is one of the manuscript's stronger elements — it does the job of correctly filtering reader expectations.

---

## 3. Abstract Review

### Abstract Strengths
Every major numeric claim in the Results section is echoed with the correct value (ROC-AUC 0.905, McNemar p=0.70/0.77, ECE 0.091 vs 0.318, 11.4 KB / 1,281 parameters / 2.6 ms). The abstract states the null accuracy result up front rather than burying it, which is unusual candor and works in the paper's favor.

### Abstract Problems
- The abstract is dense with six distinct numeric claims packed into one paragraph; a reader unfamiliar with the paper's framing may need two passes to parse which number supports which of the three claimed properties (representation / deployability / transportability).
- "we report this parity as a finding, not an improvement" is repeated near-verbatim in the Introduction, Discussion, and Conclusion. In the abstract specifically it reads slightly defensive rather than declarative — worth keeping, but be aware a reviewer may read the repetition across sections as padding (see §35).

### Required Changes
None mandatory. The abstract at 250 words is within a reasonable range for IEEE Access and does not need trimming for length alone; Rule 2's guidance to avoid over-indexing on submission-metadata-style formatting concerns applies loosely here too — word count is not a scientific weakness.

---

## 4. Introduction Review

**What is the problem?** Stated clearly: many high-performing tabular clinical models treat features independently and ignore feature interactions, and accurate models are often opaque.

**Why is the problem important?** Supported by a citation to cardiovascular disease's status as the leading global cause of mortality and to the clinical-adoption/explainability literature — appropriately scoped, not overstated.

**What has already been done?** Patient-similarity GNNs are named as the dominant clinical GNN paradigm, with their key limitation (transductive, requires cohort-level graph at inference, doesn't expose feature-feature interaction) stated precisely.

**What research gap remains?** The feature-node formulation is explicitly framed as "comparatively under-explored" rather than absent — an appropriately hedged novelty claim (see §6).

**What does this paper propose, and why?** The three-property framework (P1 representation, P2 deployability, P3 transportability) is a genuine organizing contribution: it gives the paper a falsifiable structure that the rest of the manuscript is disciplined about following. This is the Introduction's strongest feature.

**Specific contributions (C1–C3):** Each contribution statement is scoped tightly to what a later section actually measures, and each is honest about the caveat attached (e.g., C1 explicitly states the MST "contributes no edges at the operating threshold"). This is unusual: most introductions state contributions optimistically and let the caveats surface only in Discussion/Limitations. Here the caveats are front-loaded, which is good scientific practice but may read, to a reviewer skimming only the Introduction, as underselling the paper before they've seen the mechanistic experiments that make the caveats interesting.

**Concerns:** The paragraph beginning "Position of this paper" does a lot of defensive work (disclaiming both accuracy novelty and architectural novelty) before the reader has been given any positive claim to hold onto. Consider moving the P1/P2/P3 framework one paragraph earlier so the reader has the paper's organizing logic before the disclaimers.

---

## 5. Research Gap Analysis

**Claimed gap:** clinical GNNs are dominated by patient-similarity (transductive) graphs; the feature-node (inductive) formulation is under-explored, and no existing work systematically compares the two along representation/deployability/transportability rather than accuracy.

**Assessment: Partially established.**

The gap is real in the narrow sense the paper tests it (the paper implements the patient-similarity baseline itself and compares directly, which is good practice — most papers claiming a gap do not build the counterfactual). However, "comparatively under-explored" is asserted rather than demonstrated with a literature census — no count or characterization of how many feature-node-graph clinical papers exist versus patient-similarity ones is given. This weakens the gap claim from "convincingly established" to "plausible but not quantified." The paper does solve a real methodological problem (does the choice between formulations matter along axes other than accuracy?) rather than merely re-applying an existing technique to a new dataset — the direct empirical comparison of transport behavior under distribution shift (Section on external validation) is the part of the paper that most concretely fills a gap, because that comparison does not appear to be already established by the citations given.

---

## 6. Novelty Analysis

| Claimed Contribution | Actual Contribution | Novelty Level | Concern | Required Evidence |
|---|---|---|---|---|
| Feature-node graph formulation for clinical tabular data | Application of an existing, acknowledged-as-under-explored graph formulation to a new evaluation axis (not accuracy) | Application-level | The formulation itself (features-as-nodes) is not claimed as new by the authors and is a known GNN pattern outside clinical ML; novelty rests entirely on the evaluation framing | None beyond what's given — this is honestly scoped already |
| MST-augmented correlation graph construction | Engineering choice (standard MST + threshold), validated empirically rather than merely asserted | Engineering / methodological | Not a new graph-construction technique; contribution is the empirical characterization (when the MST is active vs. inert) | None additional needed |
| τ-regime dissociation (internal null vs. external preference for sparse+connected topology) | A genuine, non-obvious empirical finding: topology choice is invisible to in-distribution ablation but visible under distribution shift | Moderate–High, if it holds up | Bootstrap-level support is weak (3/9 contrasts, none survive Holm correction on DeLong) — the paper is appropriately honest about this, but a strict reviewer will treat the finding as suggestive, not established | Additional cohorts or a synthetic distribution-shift benchmark with more statistical power would elevate this from "associated" to "established" |
| Identity-embedding falsification test (message passing over scalar nodes is smoothing, not relational reasoning) | A designed falsification experiment with a predicted-then-confirmed result | Moderate — genuinely interesting mechanistic contribution | Tests only two operators (GCN family) and one embedding dimension (d=8); the paper is transparent about this narrow scope | A sweep over embedding dimension would strengthen the claim that the reversal is general rather than an artifact of d=8 |
| Direct head-to-head against patient-similarity GNN under identical protocol, in- and out-of-distribution | Genuine comparative contribution; the transport-failure-mode contrast (single-class collapse on VA) is a striking, concrete result | Moderate | Effect is partial (Switzerland favors the baseline by a small margin); paper reports this honestly | None additional — already reported without cherry-picking |

**Overclaim-language check:** the manuscript is notably disciplined about avoiding "novel," "state-of-the-art," "superior," "optimal," "unprecedented" as unsupported adjectives. Where such words might apply, the text instead states the specific statistic and its confidence interval. This is a genuine strength distinguishing this manuscript from a typical incremental submission.

**Bottom line:** nothing in the manuscript claims architectural novelty (explicitly disclaimed by the authors themselves, and correctly so — the operator-substitution ablation shows GAT/GraphSAGE/GIN change nothing significantly). The paper's novelty, such as it is, is diagnostic/mechanistic rather than architectural: it is a study of *when and why* a graph-structured inductive bias helps, not a new way to build one. That is a legitimate but modest form of novelty for IEEE Access, which does publish rigorous negative/diagnostic results, but it is not a "high novelty" paper by any conventional rubric.

---

## 7. Related Work / Literature Review

**Coverage assessment: Descriptive, with thin comparative depth.**

The Related Work section is three short paragraphs (GNNs generally, GNNs for clinical data, explainable AI) totaling roughly 25 lines. It correctly cites the foundational architecture papers (Kipf & Welling GCN, GAT, GraphSAGE, GIN) and explainability methods (GNNExplainer, Integrated Gradients, Saliency), and it appropriately hedges the "under-explored" claim about feature-node graphs by testing it directly rather than merely asserting it.

**Gaps:**
- No discussion of non-GNN feature-interaction models for tabular data (e.g., attention-based tabular architectures, explainable additive models) that make a structurally similar claim — that feature interactions matter and should be made explicit/inspectable. Without this, the paper's implicit comparison set is narrower than the claim ("a clinician can inspect... an explicit map of how risk factors relate") strictly requires.
- Only one clinical-GNN comparator (`parisot2018disease`, per the bibliography) is cited by name for the patient-similarity paradigm the paper spends an entire section refuting; a stronger literature review would cite more than one representative patient-similarity clinical GNN.
- No 2023–2026 literature appears in the reference list based on the bibliography contents inspected; the most recent-looking non-foundational citations are from 2019–2022 (`yeh2019infidelity`, `zhou2020gnnreview`, `petch2022opening`). For a 2026 IEEE Access submission this is a real gap — a reviewer will ask whether anything published in the last 2–3 years changes the "under-explored" framing.

**"References/literature to verify":** the authors should verify and consider recent literature on (a) tabular deep learning baselines beyond classical ML (e.g., attention/transformer-based tabular models) as an additional baseline family, and (b) more recent clinical or biomedical GNN applications published after 2022, to confirm the "under-explored" claim still holds and to identify any newer feature-node-style clinical GNN that would need to be distinguished from this contribution. I have not verified whether such work exists; I am not asserting it does.

---

## 8. Technical Methodology Review

| # | Problem | Technical explanation | Impact | Severity | Recommended fix |
|---|---|---|---|---|---|
| 1 | Single-dataset lineage across "external" cohorts | Hungarian, Switzerland, VA, and Cleveland are all sister cohorts from the same original UCI heart-disease collection effort, collected in the same era with the same instrument set | "External validation" here tests cross-site transfer within one historical data-collection program, not true domain shift to an independently designed clinical registry | 🟡 Moderate | Either reframe the claim as "cross-site transfer within the UCI heart-disease collection" rather than unqualified "external validation," or, ideally, add a genuinely independent cohort |
| 2 | Small n for primary analysis (n=303) | Even with 5-fold × 3-seed = 15 fold-estimates, the effective independent sample size for variance estimation is still bounded by 303 patients | CV estimates on this scale are known to carry high variance; the paper is honest about this (bootstrap width vs. seed-only spread) | 🟢 Minor (already disclosed) | None required beyond what's stated; consider stating the effective per-fold test-set size explicitly for reader convenience |
| 3 | Node feature dimensionality bottleneck (scalar per node) | Explicitly diagnosed by the authors themselves via effective-rank measurement (§ mechanism) | Limits the ceiling of what message passing can contribute; already disclosed as Limitation (ii) | 🟢 Minor (self-diagnosed) | None required — this is a model of how a limitation should be handled |
| 4 | Missing-feature handling for external cohorts | ca/thal/slope are >90% missing in two of three external cohorts, forcing an 8-feature reduced model | Reduces internal ROC-AUC from 0.905 to 0.840 before any cohort shift is even applied, confounding "feature reduction cost" with "transport cost" partially | 🟡 Moderate (already partially disclosed) | The paper does separate this cleanly already (states the 0.905→0.840 drop is internal, prior to transport) — good practice; no further fix needed beyond making sure this separation is visible in the abstract too |
| 5 | Baseline family is entirely classical/linear-adjacent | Logistic Regression, Random Forest, Gradient Boosting, MLP — no modern tabular deep-learning baseline (e.g., gradient-boosted trees like XGBoost/LightGBM specifically, or attention-based tabular architectures) | A reviewer will ask whether the "parity" finding would hold against a stronger, more contemporary tabular baseline | 🟠 Major | Add at least one modern boosted-tree implementation (XGBoost or LightGBM) under the identical protocol; if compute-constrained, justify the omission explicitly rather than leaving it implicit |

---

## 9. Mathematical Review

The manuscript contains exactly one numbered equation block, defining the GCN layer update
$H^{(l+1)}=\mathrm{ReLU}(\mathrm{BN}(\hat{A}H^{(l)}W^{(l)}))$
and the readout $\hat{y}=\sigma(w^\top\mathrm{MeanPool}(H^{(2)})+b)$.

- **Correctness:** standard and correct as written for a two-layer GCN with batch norm, ReLU, mean pooling, and a sigmoid output head.
- **Notation:** $\hat{A}$ (normalized adjacency), $H^{(0)}=X$, $W^{(l)}$ are all defined before use. No undefined symbols.
- **Consistency with implementation:** consistent with the described architecture (two GCN layers, batch norm, dropout, mean pooling, linear+sigmoid head) and with the parameter count claimed (1,281 parameters is plausible for 13-node, low-hidden-width, 2-layer GCN, though this was not independently re-derived here).
- **Sufficiency for reproduction:** the equation alone would not let an independent researcher reproduce hidden width, dropout rate, or learning rate — but these are supplied separately in the referenced hyperparameter table, which is the correct place for them.

**Assessment:** no mathematical errors found. The paper is appropriately equation-light for a methods paper whose contribution is empirical/diagnostic rather than a new algorithm — this is not a deficiency, just worth noting under §10 below since "could an independent researcher reproduce the method from the manuscript alone" depends partly on the linked hyperparameter table, which was not itself audited line-by-line here.

---

## 10. Algorithm / Model Review

- **Architecture:** two-layer GCN, BN + ReLU + dropout per layer, global mean pooling, linear+sigmoid head. Simple by design and justified by an ablation showing more expressive operators (GAT, GraphSAGE, GIN) do not help.
- **Input/output definition:** clearly specified — 13 (or 8, for the transportable subset) scalar node features per patient, fixed within-fold edge topology, single graph-level binary output.
- **Complexity/cost:** explicitly measured — 1,281 parameters, 11.4 KB, 2.6 ms/patient inference, <2 MB peak training memory, no GPU required. This is a genuine strength: deployability claims are backed by direct measurement rather than asserted.
- **Reproducibility of the method itself:** high. Fold structure, seeds (42/7/123), inner-validation split (15%), scaler-fitting discipline (train-only), and threshold-tuning procedure are all stated precisely enough that an independent researcher with access to the same UCI data could reproduce the pipeline, modulo needing the linked hyperparameter table for exact width/dropout/learning-rate values.

---

## 11. Dataset Review

- **Source:** UCI Cleveland Heart Disease dataset, a standard, well-characterized public benchmark (303 patients, 13 features, binary target, near-balanced 164/139).
- **Preprocessing:** min-max scaling of five continuous features fit on training-fold only; categorical/ordinal features retained as integer codes; no missing values in the version used.
- **Leakage control:** scaler and graph construction are both explicitly fit on the training partition only, applied unchanged to validation/test. This is stated multiple times across sections and appears to be genuinely enforced by the protocol as described (the paper's own development notes, referenced in prior review rounds, document that this was a fixed defect from an earlier draft).
- **External cohorts:** Hungarian, Switzerland, VA — same UCI heart-disease program, different sites. Missingness in ca/thal/slope is severe (up to 99%) and is handled by complete-case reduction to an 8-feature model rather than imputation; the paper explicitly justifies this as principled (avoiding imputation artifacts) but it does mean the external validation tests a materially weaker model than the primary 13-feature one.
- **Representativeness:** n=303 for the primary cohort, and 8/123 to 292/294 complete cases for external cohorts (Switzerland: 0/123 complete for the full model, i.e., external validation of the 13-feature model is *impossible*, correctly stated as such). This is a genuine dataset-driven ceiling on what the paper can claim, and the paper states it plainly rather than obscuring it.

**Overall:** dataset handling is one of the most carefully documented aspects of the manuscript.

---

## 12. Experimental Design Review

- **Objectives:** each experiment maps cleanly onto one of P1/P2/P3.
- **Baselines:** 4 classical models + 1 patient-similarity GNN, all sharing folds — appropriate design, weak only in that the classical set skips a modern boosted-tree/tabular-DL model (see §8, item 5).
- **Cross-validation:** 5-fold stratified, repeated over 3 seeds (15 fold estimates), with a further 15% inner-validation split for threshold/early-stopping decisions — this is a genuinely rigorous nested design that correctly separates model selection from evaluation.
- **Statistical testing:** McNemar (paired, pooled out-of-fold), Wilcoxon signed-rank (per-fold), DeLong (ROC comparison), Holm–Bonferroni correction applied consistently across every family of tests, plus a nested class-stratified bootstrap (B=2000) for the external cohorts. This is an unusually complete statistical toolkit for a paper of this scope.
- **Weakness:** the inner-validation-based threshold tuning is applied to the GCN; whether the classical baselines received an equivalently tuned decision threshold (rather than a default 0.5) is stated to be identical ("with in-fold standardisation and identical inner-validation" — implying yes), but this detail should be stated explicitly enough that no ambiguity remains for a reviewer checking fairness of comparison.

---

## 13. Baseline Comparison

- **Appropriateness:** Logistic Regression, Random Forest, Gradient Boosting, MLP are all reasonable, standard choices for small tabular clinical data.
- **Fairness:** identical folds, in-fold scaling, and inner-validation protocol are explicitly stated for all baselines — this is good practice and pre-empts the most common "unfair comparison" criticism.
- **Missing baseline class:** no gradient-boosted-tree implementation from the XGBoost/LightGBM family is mentioned by name (the paper's "Gradient Boosting" appears to be a generic implementation, likely scikit-learn's), which on small tabular clinical data is frequently the strongest classical competitor. A reviewer will likely ask for this by name.
- **Patient-similarity GNN baseline:** implemented directly by the authors (not merely cited), with its own k-sweep to select k=15 as its best setting — this is fair treatment, arguably fairer than most papers give their baselines.

**Verdict:** comparisons as designed are fair; the missing baseline family (modern boosted trees) is the most defensible addition a reviewer would request.

---

## 14. Results Review

Cross-checking text against the tables referenced (as summarized in text, since table files are `\input`-included from `results/latex/`):
- ROC-AUC 0.905 (GCN), 0.904 (RF), consistent between prose and the described comparison table.
- McNemar p=0.70 (vs LR), p=0.77 (vs RF) — consistent across abstract, Results, and Discussion.
- External AUCs (Hungarian 0.876, Switzerland 0.744, VA 0.721) are consistent between the External Validation subsection and the Discussion's restatement.
- The identity-embedding reversal (+0.0086 → −0.0884) is internally consistent between the mechanism subsection and its own referenced table.

No numerical inconsistency between text and its own cross-references was found in this pass. This does not constitute an independent statistical re-derivation from the underlying CSV/result tables — no accusation of fabrication is made or implied; the internal consistency across sections is itself informative and was the main check performed here.

**One item worth flagging for clarification, not correction:** the statement that on Hungarian the transported AUC (0.876) exceeds the model's own internal Cleveland AUC on the same 8 features (0.840) is unusual and the paper appropriately flags it as not indicating "superior transfer" since ROC-AUC is not comparable across cohorts of different case mix — this is correct statistical caution and should be preserved, not softened, in any revision.

---

## 15. Ablation Study

**Necessary, and present.** The ablation is genuinely comprehensive: topology choice (fully connected / random / no-edge / correlation+MST), MST presence, hidden width, dropout, readout, depth, and operator substitution (GAT/GraphSAGE/GIN) are all tested under the identical protocol.

- **Essential (already done):** topology ablation, operator ablation — both present and reported honestly, including null results.
- **Strongly recommended:** a sweep of the identity-embedding dimension $d$ beyond the single $d=8$ tested, to establish that the smoothing-vs-relational-reasoning reversal is not an artifact of that specific dimension.
- **Optional:** hyperparameter-tuning each of the four message-passing operators independently rather than reusing the GCN's tuned settings (already flagged by the authors themselves as Limitation (x)).

---

## 16. Robustness and Generalization

The external-cohort experiments are the paper's primary robustness test, and they are handled with real care (complete-case analysis stated plainly, prevalence shift documented from 46% to 93%, missingness patterns checked for outcome-dependence). The chief limitation, already partly disclosed, is that "external" here means "another UCI heart-disease site from the same collection era," not a genuinely independent registry, different instrumentation, or a different country's health system collected recently. A reviewer will likely request either a rhetorical downgrade of "external validation" to "cross-site validation within the UCI heart-disease program," or a genuinely independent additional cohort if one is obtainable.

---

## 17. Statistical Validation

This is the manuscript's strongest section. Multiple seeds (3, sometimes 5), Holm–Bonferroni correction applied wherever multiple comparisons occur, bootstrap confidence intervals used specifically where asymptotic tests (DeLong) are flagged as unreliable due to small negative-class counts (8 and 30 cases), and an explicit statement of when the paper defers to the more conservative bootstrap over a nominally significant DeLong result. This is exactly the behavior a strict reviewer wants to see and is rare in submissions at this scale.

**One gap:** per-cohort Wilcoxon tests with only 5 seeds have a documented floor of p=0.0625 (two-sided) — the paper's own development notes acknowledge this, but the manuscript text itself should state this floor explicitly wherever a per-cohort Wilcoxon non-significance is reported, so a reader doesn't mistake "not significant at p=0.0625" for "no effect."

---

## 18. Figure Review

Without independently re-rendering each figure at full resolution, based on the captions and in-text descriptions:
- Fig. 1 (pipeline overview), Fig. 3 (correlation heatmap), Fig. 4 (feature graph), Fig. 5 (spectral), Fig. 6 (degree centrality), Fig. 7 (architecture), Fig. 10 (ROC/PR), Fig. 19 (embeddings), Fig. 28 (calibration/probability distribution), Fig. 30 (error analysis), Fig. 31 (external AUC) each map to a specific claim in the text and are captioned with enough detail to be read independently of the body text — good practice.
- Fig. 4's caption was previously flagged (in an earlier review pass) for a stale claim about dashed MST bridges that do not exist at τ=0.15; the current caption already states "No MST bridge is drawn because none is required at this threshold" — this has been corrected and is now internally consistent.
- No figure appeared, from its caption, to lack axis/legend description; a full visual QA (resolution, font size at print scale) was not re-performed in this pass since it was covered in an earlier review round.

---

## 19. Table Review

Tables are `\input`-included from a separate `results/latex/` directory and were verified in an earlier review pass to exist and compile without missing-file errors. Content-level checks (units, significant figures, redundancy) were not independently re-derived from the raw CSVs in this pass; the cross-references between table numbers and in-text claims (Table~\ref{tab:comparison}, tab:stats, tab:ablation, tab:external, etc.) all resolve, per the earlier LaTeX build verification (zero undefined references in the compiled PDF).

---

## 20. Discussion Review

The Discussion is organized directly around P1/P2/P3, which keeps it disciplined, and it closes with an explicit "What the evidence does not support" paragraph — a structural choice that most papers omit and that meaningfully strengthens credibility. It does not fall into the common trap of restating results without interpretation; each P1/P2/P3 paragraph offers a mechanism (e.g., "a model that must rebuild its graph from the target cohort inherits that cohort's distributional distortion into the very structure over which messages pass") rather than just repeating the numbers.

**Overinterpretation check:** the one place a reviewer might push back is the sentence attributing the transport advantage to "selective sparsity rather than connectivity alone" on the strength of the fully-connected-graph comparison — this is a reasonable reading of the data as reported, but it rests on the same weakly-powered bootstrap noted in §6, and the Discussion should perhaps repeat the "3 of 9" caveat rather than only cross-referencing it.

---

## 21. Limitations Review

Exceptionally thorough — twelve enumerated points covering dataset size/single-center status, node-feature expressiveness ceiling (with a quantified effective-rank measurement), linear-dependence-only graph construction, MST's internal inertness, construction-rule non-decisiveness, graph size ceiling, absence of temporal modeling, external-validation feature restriction, small external-cohort imbalance, partial (not uniform) advantage over the patient-similarity baseline, un-tuned operator hyperparameters, and uncorrected calibration. This is close to a model example of how a Limitations section should read for a paper making modest, carefully bounded claims. No additions are required.

---

## 22. Reproducibility Review

High. The manuscript specifies: exact seeds (42, 7, 123, plus 5 seeds for external validation), fold structure (5-fold stratified), inner-validation split percentage (15%), scaler-fitting discipline, graph-construction threshold and its selection procedure, seed-level statistical test choices, and states that code, seeds, and all result tables are released. Hyperparameters are deferred to a linked table rather than stated in text, which is appropriate. Per the review rules, the absence of an actual GitHub URL in the current draft is a metadata placeholder, not a reproducibility deficiency — the manuscript's *description* of the method is sufficient on its own terms.

---

## 23. Ethical and Research-Integrity Check

The dataset (UCI Cleveland/Hungarian/Switzerland/VA) is a de-identified, long-established public benchmark; no new human-subject data collection occurs in this study. No conflict-of-interest or research-integrity concern is apparent from the manuscript content itself. No misuse-potential concern applies to this work.

---

## 24. Plagiarism / Similarity Risk

No unusual or unexplained stylistic discontinuities, repeated boilerplate passages, or citation-text mismatches were observed while reading the manuscript end to end — writing style is consistent across sections in register and terminology use. A definitive plagiarism/similarity assessment cannot be performed without an external similarity-checking tool, which is outside what can be evaluated from the manuscript text alone.

---

## 25. IEEE Access Suitability

IEEE Access explicitly welcomes rigorously validated, practically-oriented work without requiring breakthrough novelty, which suits this manuscript's actual contribution profile (diagnostic/mechanistic rather than architectural). The paper's deployment-characteristics measurements (latency, footprint, memory) and its transport/robustness focus are squarely in scope for the journal's applied, multidisciplinary readership. The main suitability risk is not "too incremental for IEEE Access" in the abstract sense, but that the paper's central quantitative results are mostly non-significant or weakly-significant findings, and a reviewer will want the framing to make unmistakably clear — more so than the current draft already does — that the contribution is the *rigor of a negative/mechanistic result*, not a performance claim, so as not to be evaluated (and found wanting) against a performance bar it explicitly declines to compete on.

---

## 26. Section-by-Section Review

| Section | Issue | Severity | Why It Matters | Exact Recommended Fix |
|---|---|---|---|---|
| Abstract | Six dense numeric claims in one paragraph | 🟢 Minor | Readability | Consider one additional sentence break separating the in-distribution parity claim from the transport claim |
| Related Work | Only ~25 lines, one named clinical-GNN comparator | 🟠 Major | Thin literature grounding is a common IEEE Access desk-review concern | Expand with 1–2 more paragraphs: additional clinical-GNN citations, and explicit acknowledgment of non-GNN feature-interaction tabular models |
| Experimental Setup / Baselines | No modern boosted-tree (XGBoost/LightGBM) baseline named | 🟠 Major | Reviewers will ask whether parity would hold against the strongest classical competitor | Add one such baseline under the identical protocol, or explicitly justify its absence |
| External Validation | "External" cohorts share UCI lineage with the primary cohort | 🟡 Moderate | Slightly overstates independence of the validation | Reframe as cross-site/cross-cohort validation within the same collection program, unless a genuinely independent cohort is added |
| τ-regime analysis | Central mechanistic claim rests on 3/9 bootstrap contrasts, 0 surviving Holm-DeLong correction | 🟡 Moderate (already disclosed) | This is the paper's most interesting finding and its weakest-powered one | Consider whether additional seeds/cohorts can be added before submission to strengthen statistical power on this specific claim |
| Conclusion | Restates several qualifiers already stated 2–3 times earlier | 🟢 Minor | Some redundancy across Abstract/Discussion/Conclusion | Trim slightly; not a scientific issue |

---

## 27. Reviewer #1 Simulation (Technical Correctness / Methodology / Math / Algorithms / Reproducibility)

**Strengths:** Leakage-controlled protocol with train-only scaling and graph construction; identical folds/seeds across every compared model; Holm-corrected multiple-comparison handling throughout; falsification-style mechanistic experiment (identity embedding) that was predicted before it was run; deployment measurements (latency, footprint, memory) are directly measured, not estimated.

**Major Concerns:** No modern gradient-boosted-tree baseline; the paper's central novel mechanistic claim (τ-regime dissociation) has weak statistical power under the more conservative bootstrap test.

**Minor Concerns:** Hyperparameter tuning was not performed independently for each of the four substituted message-passing operators (GCN/GAT/GraphSAGE/GIN), which the paper itself discloses as Limitation (x).

**Required Revisions:** Add at least one modern boosted-tree baseline; either strengthen the statistical power behind the τ-regime finding or soften its framing to match the bootstrap evidence throughout (not just in the caveat paragraph).

**Recommendation:** Major Revision.

---

## 28. Reviewer #2 Simulation (Novelty / Literature)

**Strengths:** Explicit disclaiming of both accuracy-based and architecture-based novelty is unusually honest; the paper substitutes a testable mechanistic claim for an unsupported novelty claim, which is intellectually more defensible even if less flashy.

**Major Concerns:** Literature review is thin (28 references, largely 2019-and-earlier for the non-foundational citations); the "under-explored" claim for feature-node clinical graphs is asserted rather than quantified against a literature census.

**Minor Concerns:** No non-GNN feature-interaction tabular model is discussed as a conceptual alternative, even briefly.

**Required Revisions:** Expand Related Work with a more systematic accounting of feature-node vs. patient-similarity clinical GNN prevalence, and verify whether any 2023–2026 work already makes a similar diagnostic claim.

**Recommendation:** Major Revision.

---

## 29. Reviewer #3 Simulation (Rejection-Oriented, Critical)

**Strengths:** No overclaiming detected; every major limitation the paper could be criticized for is already disclosed by the authors themselves, which pre-empts several standard rejection grounds.

**Major Concerns:** The central in-distribution claim is a null result (statistical parity with Logistic Regression and Random Forest); the central transport claim survives correction against only one of two baselines and only on two of three cohorts; the paper's most novel mechanistic claim (τ-regime dissociation) does not survive its own strictest statistical test. Taken together, a reviewer could reasonably ask whether the paper has *established* enough to warrant publication versus merely having *investigated carefully*.

**Minor Concerns:** Thin literature review; missing modern tabular baseline.

**Required Revisions:** Strengthen statistical power somewhere in the pipeline (more external cohorts, more seeds, or a synthetic distribution-shift benchmark with controllable effect size) so that at least one of the paper's genuinely novel claims survives its own most conservative test, not just its least conservative one.

**Recommendation:** Major Revision — reject is not warranted given the rigor of what is reported, but the paper needs either stronger evidence for its most interesting claim or a reframing that makes peace with reporting it as "suggestive, not established" throughout (currently this framing is inconsistent: careful in the τ-regime subsection, more assertive in the Discussion's restatement).

---

## 30. Potential Rejection Reasons

| Rank | Potential Rejection Reason | Severity | How to Address |
|---|---|---|---|
| 1 | Central in-distribution and transport claims are statistically weak or null by the paper's own most conservative tests | 🟠 Major | Reframe consistently as a rigorously-supported diagnostic/negative-result paper throughout every section (abstract included), rather than allowing Discussion language to read more confidently than the Results support |
| 2 | Thin literature review for an IEEE Access submission | 🟠 Major | Expand Related Work; verify recent (2023–2026) literature |
| 3 | Missing modern tabular ML baseline | 🟡 Moderate | Add XGBoost/LightGBM or justify omission |
| 4 | "External" validation cohorts are not truly independent of the primary dataset's collection program | 🟡 Moderate | Reframe as cross-site validation, or add a genuinely independent cohort |
| 5 | No architectural or algorithmic novelty | 🟢 Minor (already disclosed) | None needed — already correctly scoped by the authors |

---

## 31. Top 10 Most Important Problems

1. **Problem:** No modern gradient-boosted-tree baseline (XGBoost/LightGBM). **Location:** §Experimental Setup / Baselines. **Why it matters:** Reviewers will ask whether accuracy parity holds against the strongest plausible classical competitor. **Severity:** 🟠 Major. **Fix:** Add the baseline under the identical protocol.
2. **Problem:** Literature review is thin and possibly dated. **Location:** §Related Work. **Why it matters:** IEEE Access reviewers typically expect broader contextualization. **Severity:** 🟠 Major. **Fix:** Expand with additional clinical-GNN and non-GNN tabular-interaction citations; verify recency.
3. **Problem:** τ-regime dissociation claim's strongest statistical support (3/9 bootstrap contrasts) does not survive Holm-corrected DeLong tests. **Location:** §τ-regime analysis. **Why it matters:** This is presented as the paper's most novel mechanistic finding. **Severity:** 🟡 Moderate (already disclosed, but inconsistently emphasized elsewhere). **Fix:** Either strengthen statistical power or apply the same hedge consistently in Discussion/Conclusion.
4. **Problem:** "External" cohorts share collection lineage with the primary cohort. **Location:** §External Validation. **Why it matters:** Slightly overstates independence of the validation. **Severity:** 🟡 Moderate. **Fix:** Reframe as cross-site validation within the UCI heart-disease program.
5. **Problem:** Identity-embedding falsification test uses only one embedding dimension (d=8). **Location:** §Mechanism. **Why it matters:** Limits generality of an otherwise strong mechanistic claim. **Severity:** 🟡 Moderate. **Fix:** Sweep dimension if compute allows, or explicitly scope the claim to d=8.
6. **Problem:** Operator-substitution ablation (GAT/GraphSAGE/GIN) reuses GCN-tuned hyperparameters. **Location:** §Ablation. **Why it matters:** Could understate the true best-case performance of alternative operators. **Severity:** 🟢 Minor (already disclosed as Limitation x). **Fix:** State explicitly wherever the ablation result is used elsewhere in the paper, not just in Limitations.
7. **Problem:** Per-cohort Wilcoxon tests with 5 seeds have a p=0.0625 floor not stated in the main text. **Location:** §External Validation / Statistical support. **Why it matters:** A reader could misread "not significant" as "no effect" rather than "underpowered." **Severity:** 🟢 Minor. **Fix:** State the floor explicitly at first use.
8. **Problem:** Abstract and Discussion occasionally read more confidently than the underlying significance tests support. **Location:** Abstract, Discussion P3 paragraph. **Why it matters:** Internal consistency of epistemic confidence across sections. **Severity:** 🟡 Moderate. **Fix:** Harmonize hedge language to match the most conservative test result throughout.
9. **Problem:** No quantified literature census supporting "comparatively under-explored." **Location:** §Introduction / Related Work. **Why it matters:** Research-gap claim is asserted rather than demonstrated. **Severity:** 🟡 Moderate. **Fix:** Either quantify or soften to a qualitative claim with citations to a survey.
10. **Problem:** Some redundant restatement of the same qualifiers (parity-not-improvement) across Abstract, Introduction, Discussion, Conclusion. **Location:** Multiple. **Why it matters:** Minor writing-economy issue, not scientific. **Severity:** 🟢 Minor. **Fix:** Trim one or two of the four restatements.

---

## 32. Required Experiments

| Experiment | Purpose | Priority | Why Needed |
|---|---|---|---|
| Add XGBoost/LightGBM baseline under identical protocol | Test parity claim against strongest classical competitor | Must do before submission | Directly addresses the most likely reviewer objection to the central accuracy-parity claim |
| Additional seeds/bootstrap replicates for τ-regime external comparison | Increase statistical power behind the paper's most novel mechanistic claim | Strongly recommended | Current support (3/9, 0 surviving Holm-DeLong) is the weakest link in an otherwise rigorous paper |
| Identity-embedding dimension sweep (beyond d=8) | Test generality of the smoothing-vs-relational-reasoning reversal | Strongly recommended | Currently a single-point falsification test; a sweep would substantially strengthen an already-interesting result |
| Independent hyperparameter tuning per substituted operator (GAT/GraphSAGE/GIN) | Remove the possibility that operator-substitution null result is an artifact of reused GCN hyperparameters | Optional | Already disclosed as a limitation; would strengthen but is not required for the paper's core claims |
| A genuinely independent (non-UCI-lineage) external cohort | Strengthen the "transportability" claim beyond cross-site-within-program validation | Optional (contingent on data availability) | Would be the single strongest addition to the paper's third pillar (P3) but may not be feasible depending on data access |

---

## 33. Claim-to-Evidence Audit

| Claim | Evidence Provided? | Evidence Strength | Problem | Required Action |
|---|---|---|---|---|
| GCN statistically indistinguishable from LR/RF in-distribution | Yes — McNemar, Wilcoxon | Strong | None — this is a correctly supported null result | None |
| Feature-node model beats patient-similarity GNN | Yes — DeLong per seed | Weak-to-moderate (significant in 1 of 3 seeds) | Direction consistent, magnitude modest | State "directional advantage, not demonstrated win" consistently (already done in the relevant subsection; verify Discussion matches) |
| Advantage over LR established on transport | Yes — paired bootstrap | Moderate (2 of 3 cohorts) | Correctly scoped already | None |
| Advantage over RF established on transport | Explicitly stated as NOT established | N/A (honest null) | None | None |
| Sparse+connected topology aids transport more than density/connectivity alone | Yes — Page's trend test (pooled, significant) and per-contrast bootstrap (3/9) | Mixed: pooled test strong, per-contrast bootstrap weak | Two different statistical pictures presented for the same claim | Reconcile framing: state clearly which claim rests on the pooled test and which rests on per-contrast bootstrap, and do not let the stronger pooled result imply the per-contrast claim is equally well-supported |
| Identity embeddings interfere with message passing over scalar nodes | Yes — direct falsification experiment with predicted-then-confirmed result | Strong (for the tested configuration) | Single dimension tested | Scope claim to tested configuration explicitly, or expand |
| Model is deployable (compact, fast, cohort-free) | Yes — direct measurement (latency, footprint, memory) | Strong | None | None |

---

## 34. Internal Consistency Check

No contradiction was found between Abstract and Results, Introduction and Contributions, Methodology and Experiments, Equations and implementation description, or Results and Conclusion in this pass. The one soft inconsistency worth flagging is a *tone* mismatch rather than a factual one: the Discussion's P3 paragraph and the Conclusion both restate the transport-advantage and τ-regime findings with slightly more rhetorical confidence ("the more robust signal," "the most direct evidence we have that the graph is doing work") than the immediately-preceding statistical caveats in the same subsections support. This is not a factual contradiction — every number is consistent — but a strict reviewer may read it as the paper occasionally arguing past its own evidence in framing sentences, even while never misstating a number.

---

## 35. Writing and Language Review

Overall academic writing quality is high: sentence structure is generally clear, technical terminology is used consistently (e.g., "inductive"/"transductive" distinction maintained correctly throughout), and acronyms (ROC-AUC, MCC, ECE, GCN) are introduced once and used consistently thereafter. Tense is consistently present/past-appropriate for a completed empirical study.

**Specific passages needing attention:**
- The four-times-repeated "we report this parity as a finding, not an improvement"-style qualifier (Abstract, Introduction, Results, Discussion) could be trimmed to two occurrences without loss of rigor.
- The Discussion's P3 paragraph mixes strong rhetorical claims ("the most direct evidence we have") with immediately adjacent statistical hedges ("established only on the best-powered cohort") in the same sentence — consider splitting into two sentences for clarity of what is claimed versus what is evidenced.

No pervasive grammar or clarity issues were found; this is a well-edited manuscript at the sentence level.

---

## 36. Final Revision Roadmap

### Phase 1 — Critical Scientific Fixes
- Reconcile the pooled-test-vs-per-contrast-bootstrap framing for the τ-regime claim (§34, §9 of audit).
- Harmonize confidence language between Results/Discussion sections that report the same statistic (accuracy parity, transport advantage) so hedges are consistent wherever restated.

### Phase 2 — Experimental Improvements
- Add a modern gradient-boosted-tree baseline (XGBoost/LightGBM) under the identical protocol.
- Consider additional seeds/bootstrap power for the τ-regime external comparison.
- Consider an identity-embedding dimension sweep.

### Phase 3 — Novelty and Literature
- Expand Related Work with a more systematic literature census supporting the "under-explored" claim and verify 2023–2026 literature does not already cover this diagnostic angle.
- Briefly acknowledge non-GNN feature-interaction tabular models as conceptual alternatives.

### Phase 4 — Manuscript Improvement
- Trim redundant qualifier restatements (Abstract/Introduction/Discussion/Conclusion each restate the same 2–3 caveats).
- State the p=0.0625 Wilcoxon floor explicitly at first use in the external-validation statistics discussion.

### Phase 5 — Final Submission Preparation
Only after Phases 1–4: fill in author names/affiliation/ORCID/funding statement/biography/repository URL (all currently correctly marked `SUBMISSION-TODO` placeholders — not evaluated as scientific weaknesses per Rule 2).

---

## 37. Final Verdict

**Recommendation:** Major Revision

**Publication Readiness:** 60%

**Overall Quality:** 7/10

### Strongest Aspect
The manuscript's statistical discipline is genuinely exceptional for its scope: identical folds and seeds shared across every compared model, Holm–Bonferroni correction applied consistently across every family of hypothesis tests, and an explicit, repeated willingness to defer to the more conservative test (bootstrap over DeLong, "not established" over a nominally significant but uncorrected result) wherever the evidence is ambiguous. The Limitations section, in particular, is close to a model example of intellectual honesty in reporting a small-cohort empirical study.

### Biggest Weakness
The paper's two most interesting claims — an in-distribution accuracy edge and a τ-regime mechanistic explanation for transport behavior — are each, by the paper's own most conservative test, either a null result or a weakly-powered one (3 of 9 bootstrap contrasts, none surviving Holm-corrected DeLong). The manuscript is transparent about this, but the transparency is inconsistently distributed: careful and hedged in the subsections that report the statistics directly, more confident in the Discussion and Conclusion's restatements of the same findings.

### Biggest Threat to Acceptance
A reviewer applying a standard "does this paper establish something new with adequate statistical power" bar may conclude that the paper investigates carefully but does not yet *establish* enough — particularly if the literature review's thinness leaves the "under-explored gap" claim looking asserted rather than demonstrated, compounding the impression that the contribution is modest.

### Three Things That Must Be Fixed
1. Add a modern boosted-tree baseline (XGBoost/LightGBM) to the classical comparison set, or explicitly justify its absence.
2. Expand the Related Work section with broader and more recent literature coverage supporting the research-gap claim.
3. Harmonize the confidence language for the τ-regime and transport-advantage claims across Results, Discussion, and Conclusion so that no restatement implies stronger support than the underlying statistical test provides.

### Three Things That Would Most Improve Acceptance Probability
1. Strengthen the statistical power behind the τ-regime dissociation finding (more seeds, more cohorts, or a synthetic shift benchmark with a controllable, larger effect size).
2. Reframe "external validation" language to acknowledge the shared UCI lineage of all four cohorts, or add a genuinely independent cohort.
3. Sweep the identity-embedding dimension to generalize the falsification-test result beyond a single configuration.

### Final Reviewer Statement
This is a carefully executed, statistically disciplined study whose greatest strength — refusing to overclaim — is also the source of its greatest vulnerability, because several of its central findings are honestly reported as weak or null by the paper's own strictest tests. It is not ready for acceptance as submitted, principally because the literature grounding is thin and one obvious, strong baseline (a modern boosted-tree model) is missing, both of which are addressable without new data collection. With the Phase 1–3 revisions above, this manuscript would be a defensible IEEE Access contribution: not because it reports a performance breakthrough, but because it rigorously establishes what does and does not matter in a feature-node clinical graph formulation, and says so honestly even when the answer is "less than one might hope."

---

## 38. Final Submission Checklist

- [x] Scientific contribution is clearly established (diagnostic/mechanistic, correctly scoped)
- [~] Research gap is convincingly demonstrated — asserted qualitatively, not quantified against a literature census
- [x] Novelty claims are supported — explicitly and appropriately modest
- [x] Methodology is technically correct
- [x] Mathematical formulation is correct (single equation, standard and correct)
- [x] Experimental methodology is rigorous
- [~] Strong baselines are included — missing a modern boosted-tree/tabular-DL baseline
- [~] Results support the claims — support is honest but weak/null on the two most novel claims
- [x] Statistical validation is adequate (exceptionally so)
- [x] Ablation studies are adequate
- [~] Robustness/generalization is demonstrated — cross-site within one collection program, not fully independent
- [x] Reproducibility information is sufficient
- [x] Figures and tables are correct and clear (per earlier build/consistency verification)
- [x] Limitations are honestly discussed (exceptionally so)
- [x] Conclusion matches the evidence (modulo the confidence-language harmonization noted in §34)
- [~] Literature review adequately establishes the contribution — thin, needs expansion
- [x] No major internal inconsistencies remain (factually; minor tone inconsistency noted)
- [x] Language and technical presentation are publication-ready

*Per Rule 2 of the review protocol: missing author names, affiliations, ORCID, funding statement, acknowledgments, biography, and repository URL are submission metadata, already correctly marked as `SUBMISSION-TODO` placeholders in the source, and are not scored above.*
