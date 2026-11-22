"""Convert the two investigation reports (Markdown) to Overleaf-ready LaTeX (pdfLaTeX-compatible).

    python benchmarks/md_to_latex.py      # writes reports/overleaf/{report1.tex,report2.tex,figures/}

Handles the Markdown subset used in the reports: headings, paragraphs, **bold**, *italic*, `code`,
[links](url), tables, images, bullet / numbered lists, fenced code blocks, horizontal rules.
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overleaf"
FIGS = OUT / "figures"

UNI = {
    "µ": r"\textmu{}", "×": r"$\times$", "≈": r"$\approx$", "→": r"$\rightarrow$", "←": r"$\leftarrow$",
    "≲": r"$\lesssim$", "±": r"$\pm$", "−": r"$-$", "—": "---", "–": "--", "σ": r"$\sigma$", "²": r"$^2$",
    "¹": r"$^1$", "⁴": r"$^4$", "⁵": r"$^5$", "⁰": r"$^0$", "✅": r"\checkmark{}", "〃": r"\textquotedbl{}",
    "∘": r"$\circ$", "’": "'", "“": "``", "”": "''", "…": r"\ldots{}", "↔": r"$\leftrightarrow$",
    "ᵀ": r"$^{T}$", "✗": r"$\times$", "³": r"$^3$", "↔": r"$\leftrightarrow$", "≥": r"$\geq$", "≤": r"$\leq$", "·": r"$\cdot$", "…": r"\ldots{}",
}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s: str) -> str:
    out = []
    for ch in s:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in UNI:
            out.append(UNI[ch])
        else:
            out.append(ch)
    return "".join(out)


def inline(s: str) -> str:
    """Escape text and convert inline Markdown, protecting code spans and links first."""
    tokens = []

    def keep(latex):
        tokens.append(latex)
        return f"\x00{len(tokens) - 1}\x00"

    s = re.sub(r"`([^`]+)`", lambda m: keep(r"\texttt{" + esc(m.group(1)) + "}"), s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: keep(r"\href{" + m.group(2).replace("%", r"\%").replace("#", r"\#") + "}{" + inline(m.group(1)) + "}")
               if m.group(2).startswith("http") else keep(inline(m.group(1))), s)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], s)
    return s


def table(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    header, body = rows[0], rows[2:]
    n = len(header)
    spec = "l" + "l" * (n - 1)
    out = [r"\begin{table}[H]\centering\small", r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
           r"\begin{tabular}{" + spec + "}", r"\toprule",
           " & ".join(inline(h) for h in header) + r" \\", r"\midrule"]
    for r in body:
        r = (r + [""] * n)[:n]
        out.append(" & ".join(inline(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return out


def convert(md: str, title: str) -> str:
    lines = md.splitlines()
    out = []
    i = 0
    in_list = None
    first_h1 = True

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{%s}" % in_list)
            in_list = None

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            close_list()
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].startswith("```"):
                code.append(lines[j])
                j += 1
            out.append(r"\begin{lstlisting}")
            out += code
            out.append(r"\end{lstlisting}")
            i = j + 1
            continue
        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            close_list()
            j = i
            block = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                block.append(lines[j])
                j += 1
            out += table(block)
            i = j
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", ln.strip())
        if m:
            close_list()
            src = Path(m.group(2)).name
            out += [r"\begin{figure}[H]\centering", r"\includegraphics[width=\linewidth]{figures/" + src + "}",
                    r"\caption{" + inline(m.group(1)) + "}", r"\end{figure}"]
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            close_list()
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and first_h1:
                first_h1 = False
            else:
                cmd = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection"}[level]
                text = re.sub(r"^\d+(\.\d+)*\.?\s+", "", text)  # drop manual numbering
                out.append("\\%s{%s}" % (cmd, inline(text)))
            i += 1
            continue
        if re.match(r"^\s*---+\s*$", ln):
            close_list()
            i += 1
            continue
        m = re.match(r"^(\s*)([*-]|\d+\.)\s+(.*)", ln)
        if m:
            kind = "enumerate" if m.group(2)[0].isdigit() else "itemize"
            if in_list != kind:
                close_list()
                out.append(r"\begin{%s}" % kind)
                in_list = kind
            item = m.group(3)
            j = i + 1
            while j < len(lines) and lines[j].startswith("   ") and lines[j].strip() and not re.match(r"^\s*([*-]|\d+\.)\s", lines[j]):
                item += " " + lines[j].strip()
                j += 1
            out.append(r"\item " + inline(item))
            i = j
            continue
        if not ln.strip():
            close_list()
            out.append("")
            i += 1
            continue
        close_list()
        para = [ln.strip()]
        j = i + 1
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#|\||!\[|```|\s*([*-]|\d+\.)\s)", lines[j]):
            para.append(lines[j].strip())
            j += 1
        out.append(inline(" ".join(para)))
        out.append("")
        i = j
    close_list()
    body = "\n".join(out)
    return PREAMBLE.replace("@TITLE@", inline(title)) + body + "\n\\end{document}\n"


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx,booktabs,float,amsmath,amssymb,textcomp,xcolor}
\usepackage{listings}
\usepackage[hidelinks]{hyperref}
\lstset{basicstyle=\ttfamily\small,breaklines=true,frame=single,columns=fullflexible,keepspaces=true}
\setlength{\parskip}{0.5em}\setlength{\parindent}{0pt}
\title{@TITLE@}
\author{MegaFold PDL investigation --- NCSA DeltaAI, 1$\times$ NVIDIA GH200}
\date{November 2026}
\begin{document}
\maketitle
"""

if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "results" / "figures").glob("*.png"):
        shutil.copy(f, FIGS / f.name)
    jobs = [("REPORT_1_PDL_INVESTIGATION.md", "report1.tex",
             "Report 1: CUDA Programmatic Dependent Launch for a MegaFold Training Step on one GH200"),
            ("REPORT_2_MAKING_PDL_IMPACTFUL.md", "report2.tex",
             "Report 2: What Would Make PDL Impactful for MegaFold?"),
            ("REPORT_3_CROSS_LANGUAGE_PDL.md", "report3.tex",
             "Report 3: Cross-language PDL (Triton, ATen, cuBLAS, CUDA C++) for a MegaFold Training Step")]
    for src, dst, title in jobs:
        md = (ROOT / "reports" / src).read_text()
        (OUT / dst).write_text(convert(md, title))
        print("wrote", OUT / dst)
    shutil.make_archive(str(ROOT / "reports" / "overleaf_project"), "zip", OUT)
    print("wrote", ROOT / "reports" / "overleaf_project.zip")
