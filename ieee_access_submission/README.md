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
