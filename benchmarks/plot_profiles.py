"""Render the nsys kernel timeline of one training step per variant.

    python benchmarks/plot_profiles.py name=path/to/<run>_cuda_gpu_trace.csv [name=...]

For each run writes results/figures/profile_<name>.png:
  top    - the whole traced step, every GPU op as a mark coloured by kernel family; blank = GPU idle
  bottom - a 3 ms zoom anchored at the same logical point in every run (the middle `_attn_fwd`
           launch); an op that starts before its predecessor ends goes on lane 2, so PDL overlap is
           directly visible.
and results/profile_summary.json with per-run counts, busy/idle and overlap statistics.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# kernel families, fixed categorical order (reference palette, light mode)
FAMS = [
    ("megafold_evoattention", "MegaFold EvoFlash-3D attention", "#2a78d6"),
    ("megafold_layernorm_linear", "MegaFold LayernormLinear", "#eb6834"),
    ("megafold_swiglu", "MegaFold SwiGLU", "#1baf7a"),
    ("inductor_triton", "Inductor Triton", "#eda100"),
    ("cublas_gemm", "cuBLAS / CUTLASS GEMM", "#e87ba4"),
    ("other_aten", "ATen eager kernels", "#008300"),
    ("memcpy_memset", "memcpy / memset", "#6250d6"),
]
COL = {k: c for k, _, c in FAMS}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def family(nm: str) -> str:
    n = nm.lower()
    if "_attn_" in n:
        return "megafold_evoattention"
    if "swiglu" in n:
        return "megafold_swiglu"
    if re.search(r"_(first|second)_(forward|backward)_kernel", n):
        return "megafold_layernorm_linear"
    if n.startswith("triton_"):
        return "inductor_triton"
    if any(s in n for s in ("gemm", "cutlass", "xmma", "cublas", "nvjet", "sm80_", "sm90_")):
        return "cublas_gemm"
    if "memcpy" in n or "memset" in n:
        return "memcpy_memset"
    return "other_aten"


def load(path):
    rows = csv.DictReader(open(path))
    cols = rows.fieldnames
    cs = next(c for c in cols if c.startswith("Start"))
    cd = next(c for c in cols if c.startswith("Duration"))
    sc = 1.0 if "(ns)" in cs else 1e3
    ev = []
    for r in rows:
        if r[cd]:
            s = float(r[cs]) * sc
            ev.append((s, s + float(r[cd]) * sc, r["Name"]))
    ev.sort()
    return ev


def summarize(ev):
    t0, t1 = ev[0][0], max(e for _, e, _ in ev)
    busy, cs, ce = 0.0, ev[0][0], ev[0][1]
    for s, e, _ in ev[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    fam_n, fam_t = defaultdict(int), defaultdict(float)
    for s, e, n in ev:
        f = family(n)
        fam_n[f] += 1
        fam_t[f] += e - s
    ov = [ev[i - 1][1] - ev[i][0] for i in range(1, len(ev)) if ev[i][0] < ev[i - 1][1]]
    return dict(ops=len(ev), span_s=(t1 - t0) / 1e9, gpu_busy_s=busy / 1e9, gpu_idle_s=(t1 - t0 - busy) / 1e9,
                gpu_idle_frac=1 - busy / (t1 - t0), overlaps=len(ov), overlap_total_ms=sum(ov) / 1e6,
                count_by_family=dict(fam_n), time_s_by_family={k: v / 1e9 for k, v in fam_t.items()})


def plot(name, ev, summ, anchor_idx):
    t0 = ev[0][0]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 6.6), gridspec_kw=dict(height_ratios=[1, 1.25, 1.25], hspace=0.75))
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURF)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_yticks([])
    fig.patch.set_facecolor(SURF)

    # whole step: one row, ops coloured by family (rasterised: ~1.7e5 marks)
    by = defaultdict(list)
    for s, e, n in ev:
        by[family(n)].append(((s - t0) / 1e9, max((e - s) / 1e9, 2e-6)))
    for k, _, c in FAMS:
        if by[k]:
            ax1.broken_barh(by[k], (0, 1), facecolors=c, linewidth=0, rasterized=True)
    ax1.set_xlim(0, summ["span_s"])
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("time in step (s)", color=INK2, fontsize=9)
    ax1.set_title(f"{name}: one training step on the GPU  -  {summ['ops']:,} ops, busy {summ['gpu_busy_s']:.2f} s, "
                  f"idle {summ['gpu_idle_s']:.2f} s ({100 * summ['gpu_idle_frac']:.0f}%), "
                  f"PDL overlaps {summ['overlaps']:,}", loc="left", fontsize=10.5, color=INK, fontweight="bold")

    def zoom(ax, a, width_ns, label, min_w):
        b = a + width_ns
        lanes = defaultdict(list)
        prev_end = -1
        n_ov = 0
        for s, e, n in ev:
            if e < a or s > b:
                prev_end = max(prev_end, e)
                continue
            lane = 1 if s < prev_end else 0
            n_ov += lane
            lanes[(lane, family(n))].append(((s - a) / 1e3, max((e - s) / 1e3, min_w)))
            prev_end = max(prev_end, e)
        for (lane, k), bars in lanes.items():
            ax.broken_barh(bars, (lane * 1.3, 1), facecolors=COL[k], edgecolor=SURF, linewidth=0.6)
        ax.set_xlim(0, width_ns / 1e3)
        ax.set_ylim(-0.2, 2.5)
        ax.set_yticks([0.5, 1.8], ["in order", "overlaps\npredecessor"], fontsize=8, color=INK2)
        ax.set_xlabel(f"µs  ({label}; {n_ov} ops overlap their predecessor)", color=INK2, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)

    # zoom A: 3 ms from just before the middle _attn_fwd launch
    idx = [i for i, (_, _, n) in enumerate(ev) if n.startswith("_attn_fwd")]
    a = ev[idx[len(idx) // 2]][0] - 0.3e6 if idx else ev[len(ev) // 2][0]
    zoom(ax2, a, 3e6, "3 ms window at the middle EvoFlash-3D forward launch", 0.15)
    # zoom B: 200 us around the same op index in every run (median PDL overlap of the PDL-all run)
    i0 = min(anchor_idx, len(ev) - 1)
    zoom(ax3, ev[i0][0] - 60e3, 200e3, f"200 µs window at op #{i0:,}, same ops in every run", 0.05)
    present = [k for k, _, _ in FAMS if summ["count_by_family"].get(k)]
    fig.legend(handles=[Patch(color=COL[k], label=lbl) for k, lbl, _ in FAMS if k in present],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.19)
    out = FIG / f"profile_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    summary_path = ROOT / "results" / "profile_summary.json"
    allsum = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    loaded = [(a.split("=", 1)[0], load(a.split("=", 1)[1])) for a in sys.argv[1:]]
    # anchor for zoom B: op index of the median PDL overlap in the PDL-all run (or saved value)
    for name, ev in loaded:
        if name == "pdl_all":
            ov = [i for i in range(1, len(ev)) if ev[i][0] < ev[i - 1][1]]
            if ov:
                allsum["_anchor_op_index"] = ov[len(ov) // 2]
    anchor = allsum.get("_anchor_op_index", 0)
    for name, ev in loaded:
        s = summarize(ev)
        allsum[name] = s
        print(name, plot(name, ev, s, anchor), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if not isinstance(v, dict)})
    summary_path.write_text(json.dumps(allsum, indent=2))
