"""Figures for Report 2 (what would make PDL impactful)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "results" / "report2"
FIG = ROOT / "results" / "figures"
INK, INK2, GRID, SURF, REF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29b"
SER = [("cuda_graph | PDL off", "CUDA graph, PDL off", "#2a78d6"),
       ("cuda_graph | PDL on, early trigger", "CUDA graph, PDL early trigger", "#eb6834"),
       ("cuda_graph | PDL on, implicit trigger", "CUDA graph, PDL implicit trigger", "#eda100"),
       ("eager | PDL off", "eager launches, PDL off", REF)]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10, "axes.titlecolor": INK,
                     "axes.axisbelow": True})

d = json.loads((R2 / "regime_sweep.json").read_text())
shapes = list(dict.fromkeys(r["shape"] for r in d["results"]))
fig, axes = plt.subplots(1, len(shapes), figsize=(12, 3.9))
for ax, sh in zip(axes, shapes):
    rows = [r for r in d["results"] if r["shape"] == sh]
    works = [r["work"] for r in rows]
    for key, label, col in SER:
        ys = [r[key] for r in rows]
        ax.plot(works, ys, color=col, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=label)
    ax.set_xscale("symlog", linthresh=16)
    ax.set_yscale("log")
    ax.set_title(sh.replace(" (MegaFold-like row-wise)", " (MegaFold-like)").replace(" (few-us kernels)", ""), loc="left", fontweight="bold")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(v)[0] in "25" else ""))
    ax.set_xlabel("work per element (dependent FMAs)")
    ax.set_xticks(works, [str(w) for w in works])
axes[0].set_ylabel("µs per kernel (256-kernel dependent chain)")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Where PDL pays off: short kernels with few CTAs, launched from a CUDA graph", x=0.01, ha="left",
             fontweight="bold", color=INK, fontsize=11.5)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
fig.savefig(FIG / "r2_regime_sweep.png", dpi=160)
print("wrote", FIG / "r2_regime_sweep.png")
