"""Figures for Report 4 (compiling the activation-checkpointed layers)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R3, R4, FIG = ROOT / "results" / "report3", ROOT / "results" / "report4", ROOT / "results" / "figures"
INK, INK2, GRID, SURF, REF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29b"
BLUE, ORANGE, YELLOW = "#2a78d6", "#eb6834", "#eda100"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10, "axes.titlecolor": INK,
                     "axes.axisbelow": True})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1, 1.4], wspace=0.35))
# panel 1: time per step index (mean of the 2 repetitions); every variant shares the step-to-step pattern
V = [("off", "baseline (checkpointed layers eager)", REF), ("opaque", "layers compiled inside checkpoints", BLUE),
     ("opaque_pdl", "layers compiled + PDL", ORANGE)]
for v, lab, col in V:
    runs = [json.loads((R4 / f"e2e_{v}_r{r}.json").read_text())["step_ms_all"][2:] for r in (1, 2)]
    ys = [(a + b) / 2e3 for a, b in zip(*runs)]
    xs = list(range(2, 2 + len(ys)))
    a1.plot(xs, ys, color=col, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label=lab)
a1.set_xlabel("training step (after 2 warm-up steps)"); a1.set_ylabel("step time (s), mean of 2 runs")
a1.set_ylim(0, None); a1.legend(loc="lower left", frameon=False, fontsize=8)
a1.set_title("Step time by step index", loc="left", fontweight="bold")
# panel 2: GPU operations per step by kind
base = json.loads((R3 / "census_iscuda_fix_baseline.json").read_text())["launch_api_by_family"]
opq = json.loads((R4 / "census_opaque.json").read_text())["launch_api_by_family"]
K = [("aten", "ATen eager", REF), ("triton_inductor", "Inductor Triton", BLUE), ("triton_megafold", "MegaFold Triton", YELLOW),
     ("cublas", "cuBLAS", INK2), ("memset", "memset/memcpy", GRID)]
for j, (name, d) in enumerate((("baseline", base), ("layers compiled", opq))):
    left = 0
    for k, lab, col in K:
        v = sum(d.get(k, {}).values()) + (sum(d.get("memcpy", {}).values()) if k == "memset" else 0)
        a2.barh(j, v, left=left, color=col, edgecolor=SURF, linewidth=2, height=0.6, label=lab if j == 0 else None)
        if v > 9000:
            a2.text(left + v / 2, j, f"{v/1e3:.0f}k", ha="center", va="center", fontsize=8.5,
                    color=SURF if col in (INK2, BLUE) else INK)
        left += v
    a2.text(left + 1500, j, f"{left/1e3:.0f}k", va="center", fontsize=8.5, color=INK)
a2.set_yticks([0, 1]); a2.set_yticklabels(["baseline", "layers compiled"]); a2.invert_yaxis()
a2.set_xlabel("GPU operations in one traced step"); a2.grid(axis="y", visible=False); a2.set_xlim(0, 190000)
a2.set_title("Kernel mix per step (nsys)", loc="left", fontweight="bold")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, frameon=False, fontsize=8)
fig.savefig(FIG / "r4_compile_checkpointed.png", dpi=160, bbox_inches="tight")
print("wrote figure")
