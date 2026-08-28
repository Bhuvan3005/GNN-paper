# -*- coding: utf-8 -*-
"""Static consistency check: labels, refs, citations, figure/table counts."""
import re
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
tex = open("main.tex", encoding="utf-8").read()


def expand(t):
    def rep(m):
        fp = os.path.normpath(m.group(1) + ".tex")
        if os.path.exists(fp):
            return open(fp, encoding="utf-8").read()
        return ""
    return re.sub(r"\\input\{([^}]+)\}", rep, t)


full = expand(tex)

labels = re.findall(r"\\label\{([^}]+)\}", full)
refs = set(re.findall(r"\\ref\{([^}]+)\}", full))
cites = set()
for c in re.findall(r"\\cite\{([^}]+)\}", full):
    cites |= {x.strip() for x in c.split(",")}
bib = set(re.findall(r"@\w+\{([^,]+),",
                     open("references.bib", encoding="utf-8").read()))

dups = sorted({l for l in labels if labels.count(l) > 1})

print("figure envs in main.tex  :", len(re.findall(r"\\begin\{figure", tex)))
print("  full-width (figure*)   :", len(re.findall(r"\\begin\{figure\*", tex)))
print("table envs (incl inputs) :", len(re.findall(r"\\begin\{table", full)))
print("  full-width (table*)    :", len(re.findall(r"\\begin\{table\*", full)))
print("labels total             :", len(labels))
print("duplicate labels         :", dups or "none")
print("undefined refs           :", sorted(refs - set(labels)) or "none")
print("defined but never \\ref'd :", sorted(set(labels) - refs) or "none")
print("cited but missing in bib :", sorted(cites - bib) or "none")
print("uncited bib entries      :", sorted(bib - cites) or "none")
