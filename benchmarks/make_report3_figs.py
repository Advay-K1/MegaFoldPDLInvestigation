"""Figures for Report 3 (cross-language PDL)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results" / "report3"
FIG = ROOT / "results" / "figures"
INK, INK2, GRID, SURF, REF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29b"
BLUE, ORANGE, YELLOW = "#2a78d6", "#eb6834", "#eda100"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10, "axes.titlecolor": INK,
                     "axes.axisbelow": True})

# ---------------- Figure 1: reachability of the real step's boundaries (baseline trace)
c = json.loads((R3 / "census_iscuda_fix_baseline.json").read_text())["reachability"]
ORDER = [("queued, secondary Triton (reachable today)", "queued, next = Triton (PDL reachable today)", BLUE, None),
         ("queued, secondary ATen (needs cross-language PDL)", "queued, next = ATen (needs cross-language PDL)", ORANGE, None),
         ("queued, secondary cuBLAS/other library (closed source)", "queued, next = cuBLAS (closed source)", YELLOW, None),
         ("queued, memcpy/memset involved (never PDL)", "queued, memcpy/memset (never PDL)", INK2, "///"),
         ("CPU-bound: secondary not yet enqueued when primary ended", "CPU-bound: next kernel not yet launched", REF, None)]
fig, axes = plt.subplots(2, 1, figsize=(10, 3.9), gridspec_kw=dict(hspace=1.0))
for ax, metric, title, unit in ((axes[0], "boundaries", "Kernel boundaries in one training step (168,273)", ""),
                                (axes[1], "gap_ms", "GPU idle gap at those boundaries (9.36 s, traced)", " ms")):
    tot = sum(c[k][metric] for k, *_ in ORDER)
    left = 0.0
    for k, lab, col, hatch in ORDER:
        v = c[k][metric]; w = 100 * v / tot
        ax.barh(0, w, left=left, color=col, edgecolor=SURF, linewidth=2, hatch=hatch, height=0.6, label=lab)
        if w > 4:
            txt = f"{v:,.0f}{unit}" if metric == "boundaries" else f"{v:,.0f} ms"
            ax.text(left + w / 2, 0, f"{txt}\n{w:.1f}%", ha="center", va="center", fontsize=8.5,
                    color=SURF if col in (BLUE, INK2, ORANGE) else INK)
        left += w
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("% of total")
    ax.set_title(title, loc="left", fontweight="bold"); ax.grid(axis="y", visible=False)
q_aten = c[ORDER[1][0]]["gap_ms"]
axes[1].text(0, -0.75, f"Queued boundaries with an ATen secondary carry only {q_aten:.0f} ms of gap "
             f"({100 * q_aten / sum(c[k]['gap_ms'] for k, *_ in ORDER):.2f}%): the most cross-language PDL could remove.",
             fontsize=8.5, color=INK, transform=axes[1].get_yaxis_transform())
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)
fig.savefig(FIG / "r3_boundary_reachability.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: dependent chains in a CUDA graph
x = json.loads((R3 / "cross_language.json").read_text())["E3_dependent_chains_cuda_graph"]
hz = json.loads((R3 / "hazard.json").read_text())
names = list(x)
MODES = [("off", "PDL off", REF), ("on, implicit trigger", "PDL, implicit trigger", YELLOW),
         ("on, early trigger", "PDL, early trigger", ORANGE)]
fig, ax = plt.subplots(figsize=(10, 4.2))
h = 0.26
for j, (m, lab, col) in enumerate(MODES):
    for i, n in enumerate(names):
        if m not in x[n]:
            continue
        v = x[n][m]["us_per_kernel"]
        unsafe = "shim" in n and m == "on, early trigger"
        ax.barh(i + (j - 1) * h, v, height=h - 0.03, color=col, edgecolor=SURF, linewidth=2,
                hatch="xx" if unsafe else None, label=lab if i == 0 else None)
        tag = f"{v:.2f} us"
        if m != "off":
            tag += f"  ({x[n][m]['speedup_vs_off_pct']:+.0f}%)"
        if unsafe:
            tag += "  UNSAFE: wrong results in hazard test"
        ax.text(v + 0.02, i + (j - 1) * h, tag, va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace(" (unmodified, PDL attr forced by shim, no wait)", "\n(ATen + LD_PRELOAD shim, no wait)")
                    .replace(" (PDL attr forced by shim, no wait)", "\n(LD_PRELOAD shim, no wait)") for n in names],
                   fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0, 3.2)
ax.set_xlabel("time per kernel in a 256-kernel dependent chain, CUDA graph (us)")
ax.set_title("Cross-language PDL boundaries: Triton <-> CUDA C++ <-> ATen (1x GH200)", loc="left", fontweight="bold")
ax.legend(loc="upper right", frameon=False)
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "r3_cross_language_chains.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote figures")
