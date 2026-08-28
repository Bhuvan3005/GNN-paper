#!/usr/bin/env bash
# Build the manuscript.
#
# Two supported toolchains:
#
#   Tectonic (self-contained, no TeX install needed; what was used here)
#       tectonic -X compile main.tex
#
#   Classic TeX Live / MiKTeX
#       pdflatex main && bibtex main && pdflatex main && pdflatex main
#
# Regenerate the result tables first if any experiment has been re-run:
#       python ../make_latex_tables.py      # one .tex per results/*.csv
#       python ../make_combined_tables.py   # merge paired tables into
#                                           # two-panel floats (must run
#                                           # after make_latex_tables.py)
#
# main.tex loads newtxtext/newtxmath so that Times is used under either
# engine; without it, XeTeX-based builds silently fall back to Latin Modern.

set -e
cd "$(dirname "$0")"

if command -v tectonic >/dev/null 2>&1; then
    tectonic -X compile main.tex --keep-logs
elif command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode main
    bibtex main || true
    pdflatex -interaction=nonstopmode main
    pdflatex -interaction=nonstopmode main
else
    echo "No LaTeX engine found. Install tectonic or a TeX distribution." >&2
    exit 1
fi

echo
echo "--- overfull boxes ---"
python find_overfull.py || true
echo
echo "--- undefined refs/citations ---"
grep -c "Citation.*undefined\|LaTeX Warning: Reference" main.log || echo 0
ls -la main.pdf
