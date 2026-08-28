# Review: "Beyond Accuracy: A Feature-Node Graph Representation for Interpretable,
# Deployable and Transportable Cardiovascular Risk Prediction" — IEEE Access readiness

**Scope of this review.** I read the full manuscript (`paper/main.tex`, 1,306 lines / 19 typeset
pages), the reference list (28 entries, all cited, none orphaned), and the development log
(`PAPER_NOTES.md`). I rebuilt the PDF from source with `tectonic` in an isolated environment to
verify it compiles cleanly, and ran static checks for citation/label consistency, missing files,
and unused assets. I did **not** re-run the underlying ML pipeline or re-derive any reported
numbers — that would require re-executing `pipeline.py`/`run_*.py` against the raw UCI data, which
is outside a manuscript review. The checks below are about publication readiness, not a
statistical audit of the results themselves.

## Bottom line

This is an unusually well-argued paper for the venue: it explicitly declines to claim an accuracy
win, states its null result on accuracy up front, and structures the entire contribution around
three falsifiable properties (representation, deployability, transportability) with a dedicated
mechanistic account (Section "What message passing over scalar nodes contributes") and a
falsification test for that account. That framing is a genuine strength for IEEE Access review,
where "we found a small, honestly-quantified effect and here is why" reads far better than an
inflated accuracy claim on n=303. The manuscript **compiles cleanly** and its internal
cross-referencing is consistent. It is **not yet submittable** only because of unresolved
placeholder fields that every IEEE Access submission requires — not because of any structural or
scientific defect I found.

## 1. Build and structural integrity — PASS

- Rebuilt with `tectonic` end-to-end (auto multi-pass): the final log has **zero** "Citation ...
  undefined" / "Reference ... undefined" warnings and **zero** overfull-hbox warnings. Only benign
  font-substitution notices from the `newtxtext`/`newtxmath` Times setup remain.
- Compiled length: **19 pages** under the `IEEEtran` fallback class. IEEE Access has no hard page
  limit but "strongly recommend[s] keeping the page count under 20 pages for ease of readability" —
  this paper sits right at that boundary, so it is fine as-is but has no headroom to grow.
- Citation/reference integrity: all 28 `references.bib` entries are cited (25 directly in
  `main.tex`, the remainder inside `\input`-ed table files such as
  `results/latex/table_literature_comparison.tex`); no orphaned bib entries, no dangling `\cite`.
- All `\input{}` table-file paths resolve; all `\includegraphics` files exist except
  `author_photo.jpg`, which is intentionally gated behind `\IfFileExists` so the document builds
  without it — expected until a headshot is supplied.
- Label/reference consistency: table labels referenced from `main.tex` (e.g. `tab:knnstats`,
  `tab:external`, `tab:msttau`) are defined inside the `\input`-ed files under `results/latex/`,
  not in `main.tex` itself — this resolves correctly at build time, confirmed by the clean log.
- The document auto-detects the true `ieeeaccess.cls`; until that file is dropped in, it falls back
  to `IEEEtran`, which the file's own header notes is "layout-compatible and suitable for internal
  review." **Before final submission, download the actual IEEE Access class from
  template-selector.ieee.org** — the layout, headers, and "INDEX TERMS"/DOI/history blocks differ
  cosmetically from `IEEEtran` and the editorial office will expect the genuine template.

## 2. Must-fix before submission (blocking)

The manuscript is functionally complete but carries four unresolved `SUBMISSION-TODO` markers.
None require new writing — they are metadata substitutions:

| Item | Current placeholder | Location |
|---|---|---|
| Author name, ORCID, IEEE membership grade | "First A. Author", no ORCID | `\author{...}` block, both `ieeeaccess`/`IEEEtran` branches |
| Affiliation and corresponding-author e-mail | "Institution Name, City 000000, Country", `author@institution.edu` | Same block, twice (address + `\corresp`) |
| Funding statement | Generic "received no specific grant" boilerplate | `\tfootnote{...}` — confirm this is actually true for your case, or replace it |
| Code/data repository URL | `https://github.com/USERNAME/REPOSITORY` | "Data and code availability" paragraph, §Reproducibility |
| Author biography and photograph | Generic templated bio ("B.E. degree ... Institution Name ... 20xx"), no photo file | End-of-document `IEEEbiography` block |

IEEE Access requires a real ORCID, institutional affiliation, and corresponding-author e-mail for
every author at submission — these are validated at desk-review, not just typeset placeholders, so
none of the above can go out as-is. The repository URL is load-bearing for your own claim: the
paper states multiple times that "every table and figure in this paper is regenerated end-to-end
by that code," so an unresolved GitHub link would undercut the reproducibility claim the paper is
making about itself.

## 3. Minor issues (non-blocking, worth fixing)

1. **Related Work formatting.** In the Related Work section, the "GNNs for clinical data." and
   "Explainable AI." bold mini-headers run together in the same paragraph with no line break
   between them in the source — the "Explainable AI." header starts mid-sentence after "and compare
   the two formulations both in-distribution and under transport... **Explainable AI.** Post-hoc
   attribution...". Insert a paragraph break so the third mini-header reads as a new paragraph like
   its two siblings.
2. **Abstract length.** The abstract is **263 words**. General IEEE editorial convention (and
   several IEEE society author guides) targets roughly 150–250 words, single paragraph, no
   citations — this abstract meets the single-paragraph/no-citation requirement but runs slightly
   over the usual ceiling. It is not a hard rule at IEEE Access specifically, but trimming by
   10–15% (e.g., tightening the ablation and external-validation sentences) would bring it in line
   with typical reviewer expectations and improve indexing-service display.
3. **Unused figure files.** Fifteen PNGs in `figures/` are not referenced by any
   `\includegraphics` in `main.tex` (e.g. `fig02_dataset_distribution.png`, `fig08_loss.png`,
   `fig12_confusion.png`, `fig13_performance_bars.png`, `fig14_threshold.png`,
   `fig17_ablation.png`, `fig18_topology.png`, `fig20_hidden.png`, `fig21_dropout.png`,
   `fig22_pooling.png`, `fig23_feature_importance.png`, `fig25_example_explanations.png`,
   `fig27_agreement_heatmap.png`, `fig29_radar.png`, `fig32_external_shift.png`). This is not a
   defect — the paper correctly relies on tables for most of the ablation/threshold/hyperparameter
   results rather than plotting every one — but it's worth a deliberate pass: either cut these from
   the repository before the code-release link goes live, or promote 1–2 of the more informative
   ones (e.g. the ablation bar chart or the external-shift figure) into the manuscript if they'd
   strengthen the Discussion/External Validation sections beyond what the tables already show.
4. **AI-use disclosure.** Some IEEE journals (e.g. the IEEE Open Access Journal of Power and
   Energy) now require a citation/disclosure for any section drafted with AI assistance. I could
   not confirm whether IEEE Access currently imposes the same requirement — check the current
   Submission Guidelines page at submission time and add a disclosure statement if required.

## 4. Content and argument review (non-blocking, reviewer-facing observations)

These are not defects but points a peer reviewer is likely to probe, since the paper is unusually
candid about its own limitations already:

- The paper's central rhetorical move — reporting accuracy parity as a *finding* rather than
  hedging it — is well supported internally (McNemar/DeLong tests, Holm–Bonferroni correction
  across every family of hypotheses) and is consistent with the Limitations section, which already
  anticipates most objections a reviewer would raise (small single-centre cohort, rank-one node
  representation, Pearson-only dependence, MST's limited internal effect, imbalanced external
  cohorts). This internal consistency is a strength — the Discussion and Limitations sections do
  not contradict the Results, which is not always true of papers built up incrementally.
- The externally-validated model is explicitly restricted to eight features because `ca`,
  `thal`, and `slope` are absent from the other UCI cohorts (Limitation vii) — this is disclosed
  clearly and is unlikely to draw a desk-reject, but expect a reviewer to ask why the 13-feature
  model isn't the one being transported, given transportability is one of the three headline
  properties.
- The falsification test in §"What message passing over scalar nodes contributes" (identity
  embeddings reversing the sign of the graph gap) is a genuinely strong piece of evidence for the
  paper's mechanistic claim and is worth foregrounding — if the abstract needs trimming (point 2
  above), this is one of the findings I would keep rather than cut, since it's the paper's most
  distinctive methodological contribution beyond the headline parity result.

## Summary checklist for the author

- [ ] Replace all 5 `SUBMISSION-TODO` placeholder fields (author identity, ORCID, affiliation,
      e-mail, funding statement, repository URL, biography/photo)
- [ ] Download and drop in the official `ieeeaccess.cls` before final compilation
- [ ] Fix the missing paragraph break before "Explainable AI." in Related Work
- [ ] Optionally trim the abstract to ~220 words
- [ ] Decide on the 15 unused figure files (prune from repo release or promote into the paper)
- [ ] Confirm current IEEE Access policy on AI-authorship disclosure before submission

No scientific, statistical, or LaTeX-structural defect blocks submission once the placeholder
fields above are filled in.
