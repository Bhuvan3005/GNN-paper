# -*- coding: utf-8 -*-
"""Map Overfull \\hbox warnings in main.log to the output page they occur on,
so column-width problems can be located in the PDF."""
import re
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

log = open("main.log", encoding="utf-8", errors="replace").read()
pat = re.compile(r"Overfull \\hbox \(([0-9.]+)pt too wide\) in (\w+) at lines (\d+)--(\d+)")

rows = []
for m in pat.finditer(log):
    pt = float(m.group(1))
    kind, l1, l2 = m.group(2), m.group(3), m.group(4)
    before = log[: m.start()]
    page = len(re.findall(r"\[\d+", before))
    rows.append((pt, page, kind, f"{l1}--{l2}"))

rows.sort(reverse=True)
print(f"{len(rows)} overfull boxes (page = approx. output page)\n")
print(f"{'width':>9}  {'page':>4}  {'kind':<12} lines")
for pt, page, kind, lines in rows:
    flag = "   <-- FIX" if pt > 20 else ""
    print(f"{pt:8.1f}pt  {page:>4}  {kind:<12} {lines}{flag}")
