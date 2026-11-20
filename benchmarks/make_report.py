"""Build REPORT.md + figures for the MegaFold PDL investigation from the result files.

    python benchmarks/make_report.py            # reads results/, writes REPORT.md and results/figures/*.png
"""

import csv
import json
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# reference categorical palette (light mode), fixed order: baseline, PDL-all, PDL-tiled
C = {"baseline": "#2a78d6", "pdl_all": "#eb6834", "pdl_tiled": "#1baf7a", "pdl_implicit": "#eda100"}
LABEL = {"baseline": "Baseline", "pdl_all": "PDL (all kernels)", "pdl_tiled": "PDL (tiled kernels only)",
         "pdl_implicit": "PDL (all kernels, implicit trigger)"}
VARIANTS = ("pdl_all", "pdl_implicit", "pdl_tiled")
# (baseline rep, variant rep) pairs: each variant run is paired with an adjacent baseline run
PAIRS = {v: [(1, 1), (2, 2)] for v in VARIANTS}  # interleaved: B, all, implicit, tiled per repetition
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.axisbelow": True,
})


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, d=1):
    return f"{x:,.{d}f}"


# ---------------------------------------------------------------- end-to-end
base_primary = load(R / "baseline" / "baseline.json")
e2e = {"baseline": {r: load(R / f"final_baseline_r{r}.json") for r in (1, 2)}}
e2e.update({v: {r: load(R / f"final_{v}_r{r}.json") for r in (1, 2)} for v in VARIANTS})
e2e = {v: {r: x for r, x in runs.items() if x} for v, runs in e2e.items()}


def timed(run):
    return run["step_ms_all"][run["warmup"]:]


paired = {}  # variant -> list of per-step ratios vs baseline of the same rep
for v in VARIANTS:
    ratios = []
    for rb, rx in PAIRS[v]:
        if rb in e2e["baseline"] and rx in e2e.get(v, {}):
            ratios += [xi / bi for bi, xi in zip(timed(e2e["baseline"][rb]), timed(e2e[v][rx]))]
    paired[v] = ratios

# per-step chart (mean over reps)
if e2e["baseline"]:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for v in ("baseline",) + VARIANTS:
        runs = list(e2e.get(v, {}).values())
        if not runs:
            continue
        steps = list(range(runs[0]["warmup"], runs[0]["warmup"] + len(timed(runs[0]))))
        mean = [st.mean(t) / 1e3 for t in zip(*[timed(r) for r in runs])]
        ax.plot(steps, mean, color=C[v], lw=2, marker="o", ms=8, mec=SURF, mew=2, label=LABEL[v])
    ax.set_xlabel("training step index (same RNG stream in every variant)")
    ax.set_ylabel("step time (s)")
    ax.set_title("MegaFold training step, bs=1, 384 tokens, torch.compile, 1x GH200\n(mean over runs per variant)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "e2e_step_time.png", dpi=160)
    plt.close(fig)

# ---------------------------------------------------------------- kernel variants
KV = R / "kernel_variants"
kv_names = {"k1_pdl_all": "PDL, all kernels", "k2_pdl_all_nolaunch": "PDL instructions only (no PDL launch)",
            "k3_pdl_all_implicit": "PDL, all kernels, implicit trigger", "k4_pdl_tiled": "PDL, tiled kernels only"}
kv = {k: load(KV / f"cmp_{k}.json") for k in kv_names}
kv = {k: v for k, v in kv.items() if v}
if kv:
    base_ms = next(iter(kv.values()))["A"]["fwd_bwd_ms_median"]
    rows = [("Baseline", base_ms, C["baseline"])] + [
        (kv_names[k], v["B"]["fwd_bwd_ms_median"], C["pdl_tiled"] if k == "k4_pdl_tiled" else C["pdl_all"])
        for k, v in kv.items()]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ys = range(len(rows))[::-1]
    ax.barh(list(ys), [r[1] for r in rows], height=0.5, color=[r[2] for r in rows])
    for y, (name, ms, _) in zip(ys, rows):
        ax.text(ms + 0.3, y, f"{ms:.2f} ms ({100 * (ms / base_ms - 1):+.1f}%)" if name != "Baseline" else f"{ms:.2f} ms",
                va="center", color=INK, fontsize=9)
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.set_xlabel("forward + backward, median of 20 (ms)")
    ax.set_title("Fused-kernel stack (8 blocks, n=384, d=128), eager")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "kernel_variants.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- per-kernel durations (nsys)
def kern_sum(p):
    d = {}
    if not Path(p).exists():
        return d
    for r in csv.DictReader(open(p)):
        k = re.sub(r"\(.*", "", r["Name"])
        if k.startswith("_"):  # MegaFold Triton kernels
            d[k] = float(next(r[c] for c in r if c.startswith("Avg"))) / 1e3
    return d


NS = R / "nsys"
ks = {v: kern_sum(NS / f"kernels_{v}_cuda_gpu_kern_sum.csv") for v in ("baseline", "pdl", "pdl_tiled")}
ROWWISE = {"_first_forward_kernel", "_second_backward_kernel", "_swiglu_forward_kernel", "_swiglu_backward_kernel"}
if ks["baseline"] and ks["pdl"]:
    names = sorted(ks["baseline"], key=lambda k: (k not in ROWWISE, -ks["baseline"][k]))
    fig, ax = plt.subplots(figsize=(8.5, 4))
    w = 0.26
    series = [("baseline", "baseline"), ("pdl", "pdl_all")] + ([("pdl_tiled", "pdl_tiled")] if ks["pdl_tiled"] else [])
    for i, (src, col) in enumerate(series):
        xs = [j + (i - (len(series) - 1) / 2) * (w + 0.02) for j in range(len(names))]
        ax.bar(xs, [ks[src].get(n, 0) for n in names], width=w, color=C[col], label=LABEL[col])
    ax.set_xticks(range(len(names)), [n.strip("_").replace("_kernel", "") + ("*" if n in ROWWISE else "") for n in names],
                  rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("avg duration per launch (µs)")
    ax.set_title("MegaFold Triton kernels: per-launch duration (* = one CTA per row, grid = n²)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "per_kernel_duration.png", dpi=160)
    plt.close(fig)

# ---------------------------------------------------------------- where the step time goes (baseline nsys)
g_base = load(R / "baseline" / "nsys" / "baseline_gaps.json")
elig = load(R / "baseline" / "nsys" / "baseline_pdl_eligibility.json")
g_all, g_tiled = load(NS / "pdl_all_gaps.json"), load(NS / "pdl_tiled_gaps.json")
FAMN = {"other_aten": "ATen eager kernels", "megafold_evoattention": "MegaFold EvoFlash-3D attention",
        "megafold_layernorm_linear": "MegaFold LayernormLinear", "cublas_gemm": "cuBLAS / CUTLASS GEMM",
        "megafold_swiglu": "MegaFold SwiGLU", "inductor_triton": "Inductor Triton", "memcpy_memset": "memcpy / memset"}
if g_base:
    items = sorted(g_base["kernel_time_ms_by_family"].items(), key=lambda kv: -kv[1])
    items = [("GPU idle between ops", g_base["gpu_idle_ms"])] + [(FAMN.get(k, k), v) for k, v in items]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ys = range(len(items))[::-1]
    ax.barh(list(ys), [v / 1e3 for _, v in items], height=0.55,
            color=["#a3a29b"] + [C["baseline"]] * (len(items) - 1))
    tot = g_base["span_ms"]
    for y, (n, v) in zip(ys, items):
        ax.text(v / 1e3 + 0.08, y, f"{v / 1e3:.2f} s ({100 * v / tot:.0f}%)", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(ys), [n for n, _ in items])
    ax.set_xlabel("seconds in one traced baseline step")
    ax.set_title("Where one baseline training step goes (nsys, single stream)")
    ax.set_xlim(0, items[0][1] / 1e3 * 1.3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "baseline_step_breakdown.png", dpi=160)
    plt.close(fig)

json.dump(dict(paired={k: dict(n=len(v), mean=st.mean(v), median=st.median(v), stdev=st.stdev(v) if len(v) > 1 else 0)
                       for k, v in paired.items() if v}),
          open(R / "paired_summary.json", "w"), indent=2)
print("figures written to", FIG)
for k, v in paired.items():
    if v:
        print(k, "paired step-time ratio vs baseline: mean", round(st.mean(v), 4), "median", round(st.median(v), 4), "n", len(v))
