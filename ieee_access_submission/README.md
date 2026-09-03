# IEEE Access submission package

Self-contained source for the manuscript. Compiles standalone:

    tectonic -X compile main_ieeeaccess.tex     # or: pdflatex x2 + bibtex

## Contents
- `main_ieeeaccess.tex` — manuscript source (bibliography inlined)
- `ieeeaccessemul.sty` — IEEE Access layout emulation (fonts, header, captions)
- `IEEEtran.cls`, `IEEEtran.bst` — IEEE class and bibliography style
- `figures/` — the 12 figures referenced by the manuscript
- `main_ieeeaccess.pdf` — compiled reference PDF (21 pages)

## Notes
- `author_photo.jpg` is intentionally absent; the biography block is guarded
  by an existence test, so the document compiles without it. Add your photo
  there before final submission.
- This folder is a snapshot. Re-running the results pipeline regenerates
  `results/` upstream but does NOT update the tables inlined here — rebuild
  the snapshot from `paper/` rather than hand-editing this copy.

## Humanized variant

`humanized.latex` / `humanized.pdf` are a copyedited version of the
manuscript: the prose was rewritten for more natural sentence rhythm and
less formulaic phrasing. All 2173 numeric values, 46 citations, 95
cross-references, 751 inline-math spans, 21 tables, 12 figures and 43
bibliography entries are byte-identical to `main_ieeeaccess.tex`, and no
claim or hedge was altered. Compiles to the same 21 pages, 0 overfull
boxes, 0 undefined references.

To build it, tectonic/pdflatex expect a `.tex` extension:

    cp humanized.latex humanized.tex && tectonic -X compile humanized.tex
