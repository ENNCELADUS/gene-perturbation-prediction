"""Render <tex ...>LaTeX</tex> placeholders of an SVG template to vector paths.

usage: python docs/figures/tex_to_svg.py docs/figures/gene_knowledge_model.template.svg "docs/Gene-Knowledge Cell-Set Model Workflow.svg"
Needs latex (helvet, preview) and dvisvgm.
Attributes: x, y (px), size (px font size), anchor=middle|start, v=base|mid, fill.
y is the baseline (v=base) or the vertical centre (v=mid).
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PREAMBLE = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb}
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{0pt}
\begin{document}
"""
BP_PER_EM = 9.963  # 10pt in bp

tpl = Path(sys.argv[1]).read_text()
snips = list(re.finditer(r"<tex ([^>]*)>(.*?)</tex>", tpl, re.S))
work = Path(tempfile.mkdtemp())
body = "".join(f"\\begin{{preview}}{m.group(2)}\\end{{preview}}\n" for m in snips)
(work / "s.tex").write_text(PREAMBLE + body + "\\end{document}\n")
run = subprocess.run(["latex", "-interaction=nonstopmode", "s.tex"], cwd=work, capture_output=True, text=True)
if run.returncode:
    sys.exit(run.stdout[-3000:])
subprocess.run(["dvisvgm", "--no-fonts", "--bbox=preview", "--page=1-", "s.dvi"], cwd=work, check=True, capture_output=True)

pages = sorted(work.glob("s-*.svg"), key=lambda f: int(f.stem.split("-")[1]))
assert len(pages) == len(snips), (len(pages), len(snips))
out, last = [], 0
for k, m in enumerate(snips, 1):
    a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    svg = pages[k - 1].read_text()
    vb = [float(v) for v in re.search(r"viewBox='([^']*)'", svg).group(1).split()]
    inner = svg[svg.index("<defs>"): svg.rindex("</svg>")]
    inner = re.sub(r"id='([^']*)'", rf"id='t{k}-\1'", inner).replace("xlink:href='#", f"xlink:href='#t{k}-")
    s = float(a["size"]) / BP_PER_EM
    w, h = vb[2] * s, vb[3] * s
    x = float(a["x"]) - (w / 2 if a.get("anchor", "middle") == "middle" else 0)
    y = float(a["y"]) - (h / 2 if a.get("v") == "mid" else -vb[1] * s)
    fill = a.get("fill", "#1a1a1a")
    out += [tpl[last: m.start()],
            f'<svg x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" viewBox="{" ".join(f"{v:g}" for v in vb)}" overflow="visible" fill="{fill}">{inner}</svg>']
    last = m.end()
out.append(tpl[last:])
res = "".join(out)
if "xmlns:xlink" not in res[:300]:
    res = res.replace('<svg xmlns="http://www.w3.org/2000/svg"', '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"', 1)
Path(sys.argv[2]).write_text(res)
print(f"{len(snips)} formulas")
