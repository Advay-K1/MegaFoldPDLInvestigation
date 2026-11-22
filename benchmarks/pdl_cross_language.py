"""Cross-language PDL: can a Triton kernel and a non-Triton kernel (ATen C++, cuBLAS, hand-written CUDA C++)
form a Programmatic-Dependent-Launch pair?

PDL is not a language feature. A boundary P -> S overlaps only if
  (a) S is *launched* with cudaLaunchAttributeProgrammaticStreamSerialization (host side, per launch), and
  (b) S executes griddepcontrol.wait before reading P's output (device side, inside S's code);
  optionally P executes griddepcontrol.launch_dependents to release S before P finishes.
Triton's gdc_wait()/gdc_launch_dependents() are inline asm for exactly these PTX instructions; CUDA C++ can
emit the same asm. Experiments (run with LD_PRELOAD=benchmarks/pdl_xlang/libpdl_shim.so):

  E1 "who listens": P = 1-CTA Triton kernel that calls gdc_launch_dependents first and then runs ~30 us;
     S = an independent kernel of each kind. If S's launch carries the PDL attribute, S runs *under* P
     (time(P;S) ~ max instead of sum). Captured in a CUDA graph.
  E2 "who triggers": P = a long kernel of each kind; S = a PDL-launched CUDA C++ probe that records
     %globaltimer at entry and after griddepcontrol.wait. A large gap = P released S early.
  E3 dependent chains (256 kernels, each reads the previous output) alternating Triton with ATen / CUDA C++,
     in a CUDA graph: time per boundary and bitwise correctness vs PDL off, over many randomized trials.
     Includes forcing the PDL attribute onto unmodified ATen kernels with the LD_PRELOAD shim (they cannot wait).
  E4 the same Triton <-> CUDA C++ chain launched eagerly (as in the real training step).

    LD_PRELOAD=$PWD/benchmarks/pdl_xlang/libpdl_shim.so python benchmarks/pdl_cross_language.py \
        --out results/report3/cross_language.json
"""

import argparse
import ctypes
import json
import os
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent / "pdl_xlang"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--chain", type=int, default=256)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--trials", type=int, default=200, help="randomized correctness trials per chain config")
args = ap.parse_args()

ext = load(name="pdl_xlang_ext", sources=[str(HERE / "pdl_ext.cu")], build_directory=str(HERE / "build"),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
try:
    shim = ctypes.CDLL(str(HERE / "libpdl_shim.so"))
    SHIM = bool(shim.pdl_shim_loaded()) and "libpdl_shim" in os.environ.get("LD_PRELOAD", "")
except OSError:
    shim, SHIM = None, False
shim_set = (lambda on: shim.pdl_shim_set(int(on))) if SHIM else (lambda on: None)
shim_count = (lambda: shim.pdl_shim_count()) if SHIM else (lambda: -1)


# ---------------------------------------------------------------- Triton kernels
@triton.jit
def tri_axpb(src, dst, n, a, b, WAIT: tl.constexpr, TRIG: tl.constexpr, WORK: tl.constexpr, BLOCK: tl.constexpr):
    if WAIT:
        tl.extra.cuda.gdc_wait()
    if TRIG:
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(src + offs, mask=m)
    for _ in range(WORK):
        v = v * 0.999 + 0.001
    tl.store(dst + offs, v * a + b, mask=m)


def t_launch(src, dst, a, b, pdl, wait, trig, work=0, block=512):
    n = src.numel()
    tri_axpb[(triton.cdiv(n, block),)](src, dst, n, a, b, WAIT=wait, TRIG=trig, WORK=work, BLOCK=block,
                                       num_warps=4, launch_pdl=pdl)


def c_launch(src, dst, a, b, pdl, wait, trig, work=0, block=256):
    n = src.numel()
    ext.axpb(src, dst, a, b, pdl, wait, trig, work, triton.cdiv(n, block), block)


def graph_time(run, reps=None, per=1):
    """median time (us) of one replay of `run` captured in a CUDA graph, divided by `per`."""
    reps = reps or args.reps
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        run(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            run()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / per)
    return statistics.median(ts), g


def eager_time(run, per=1):
    run(); torch.cuda.synchronize()
    ts = []
    for _ in range(args.reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / per)
    return statistics.median(ts)


R = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
         cuda=torch.version.cuda, shim_loaded=SHIM)
R["stream_attribute_pss"] = {"cudaError": int(ext.try_stream_attr()),
                             "note": "0 = accepted as a stream attribute; non-zero = only valid per launch"}
print("stream attr:", R["stream_attribute_pss"], "shim:", SHIM, flush=True)
dev = "cuda"

# ---------------------------------------------------------------- E1: who listens (secondary side)
K = 20  # (P;S) pairs per graph
p_src = torch.rand(512, device=dev); p_dst = torch.empty_like(p_src)
P_WORK = 6000


def P_trig():  # 1 CTA, releases its dependents immediately, then ~tens of us of dependent FMAs
    t_launch(p_src, p_dst, 1.0, 0.0, pdl=False, wait=False, trig=True, work=P_WORK, block=512)


big = torch.rand(1 << 24, device=dev); big_o = torch.empty_like(big)       # 64 MiB
rows = big.view(4096, 4096)
row_o = torch.empty(4096, device=dev)
A16 = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16); B16 = torch.randn_like(A16); C16 = torch.empty_like(A16)
A32 = torch.randn(2048, 2048, device=dev); B32 = torch.randn_like(A32); C32 = torch.empty_like(A32)
S_KINDS = {
    "Triton, launch_pdl=True": lambda: t_launch(big, big_o, 1.0, 1.0, pdl=True, wait=False, trig=False),
    "Triton, launch_pdl=False": lambda: t_launch(big, big_o, 1.0, 1.0, pdl=False, wait=False, trig=False),
    "CUDA C++, cudaLaunchKernelEx + PDL attr": lambda: c_launch(big, big_o, 1.0, 1.0, pdl=True, wait=False, trig=False),
    "CUDA C++, no attr": lambda: c_launch(big, big_o, 1.0, 1.0, pdl=False, wait=False, trig=False),
    "ATen add (vectorized_elementwise_kernel)": lambda: torch.add(big, 1.0, out=big_o),
    "ATen sum over rows (reduce_kernel)": lambda: torch.sum(rows, dim=1, out=row_o),
    "ATen copy_": lambda: big_o.copy_(big),
    "cuBLAS bf16 GEMM 4096^3": lambda: torch.mm(A16, B16, out=C16),
    "cuBLAS fp32 GEMM 2048^3": lambda: torch.mm(A32, B32, out=C32),
}
SHIMMED = ["ATen add (vectorized_elementwise_kernel)", "ATen sum over rows (reduce_kernel)", "ATen copy_",
           "cuBLAS bf16 GEMM 4096^3"]
torch.backends.cuda.matmul.allow_tf32 = False
tP, _ = graph_time(lambda: [P_trig() for _ in range(K)], per=K)
e1 = {"P alone (us)": tP}
cases = [(k, False) for k in S_KINDS] + ([(k, True) for k in SHIMMED] if SHIM else [])
for name, use_shim in cases:
    S = S_KINDS[name]
    label = name + (" + LD_PRELOAD PDL shim" if use_shim else "")
    shim_set(use_shim); c0 = shim_count()
    tS, _ = graph_time(lambda: [S() for _ in range(K)], per=K)
    tPS, _ = graph_time(lambda: [(P_trig(), S()) for _ in range(K)], per=K)
    conv = shim_count() - c0
    shim_set(False)
    hidden = (tP + tS - tPS) / min(tP, tS)
    e1[label] = dict(S_us=round(tS, 2), P_then_S_us=round(tPS, 2), overlap_frac_of_shorter=round(hidden, 3),
                     runs_under_P=hidden > 0.5, shim_converted_launches=conv if use_shim else None)
    print("E1", label, e1[label], flush=True)
R["E1_who_listens"] = e1

# ---------------------------------------------------------------- E2: who triggers (primary side)
probe_out = torch.zeros(2, dtype=torch.int64, device=dev)
LA16 = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16); LB16 = torch.randn_like(LA16)
LC16 = torch.empty_like(LA16)
SKA = torch.randn(256, 65536, device=dev, dtype=torch.bfloat16); SKB = torch.randn(65536, 256, device=dev, dtype=torch.bfloat16)
SKC = torch.empty(256, 256, device=dev, dtype=torch.bfloat16)
huge = torch.rand(1 << 27, device=dev); huge_o = torch.empty_like(huge)   # 512 MiB
P_KINDS = {
    "Triton 1-CTA, gdc_launch_dependents first (positive control)":
        lambda: t_launch(p_src, p_dst, 1.0, 0.0, pdl=False, wait=False, trig=True, work=P_WORK * 4, block=512),
    "Triton 1-CTA, no trigger (negative control)":
        lambda: t_launch(p_src, p_dst, 1.0, 0.0, pdl=False, wait=False, trig=False, work=P_WORK * 4, block=512),
    "CUDA C++ 1-CTA, launch_dependents first":
        lambda: ext.axpb(p_src, p_dst, 1.0, 0.0, False, False, True, P_WORK * 4, 1, 512),
    "ATen add, 512 MiB": lambda: torch.add(huge, 1.0, out=huge_o),
    "ATen sum over rows, 512 MiB": lambda: torch.sum(huge.view(32768, 4096), dim=1),
    "cuBLAS bf16 GEMM 8192^3": lambda: torch.mm(LA16, LB16, out=LC16),
    "cuBLAS bf16 GEMM 256x256, K=65536 (few CTAs)": lambda: torch.mm(SKA, SKB, out=SKC),
}
e2 = {}
for name, P in P_KINDS.items():
    gaps = []
    for _ in range(10):
        P(); ext.probe(probe_out); torch.cuda.synchronize()
        t0, t1 = probe_out.tolist()
        gaps.append((t1 - t0) / 1e3)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); P(); e.record(); torch.cuda.synchronize()
    e2[name] = dict(P_us=round(s.elapsed_time(e) * 1e3, 1), probe_wait_us_median=round(statistics.median(gaps), 2),
                    triggers_early=statistics.median(gaps) > 2.0)
    print("E2", name, e2[name], flush=True)
R["E2_who_triggers"] = e2

# ---------------------------------------------------------------- E3: dependent chains in a CUDA graph
N_SMALL = 132 * 512    # one wave of 512-element CTAs for Triton (132 CTAs), few-us kernels
x0 = torch.empty(N_SMALL, device=dev); bufA = torch.empty_like(x0); bufB = torch.empty_like(x0)


def aten_step(src, dst):
    torch.add(src, 0.001, out=dst)  # ATen vectorized_elementwise_kernel


def make_chain(pattern, pdl, trig, shim_aten):
    """pattern: string over {T, C, A}; repeated to args.chain kernels. Every kernel reads the previous output."""
    seq = (pattern * args.chain)[:args.chain]

    def run():
        t_launch(x0, bufA, 0.999, 0.001, pdl=False, wait=False, trig=False)  # chain head (not PDL)
        src, dst = bufA, bufB
        for k in seq:
            if k == "T":
                t_launch(src, dst, 0.999, 0.001, pdl=pdl, wait=pdl, trig=pdl and trig)
            elif k == "C":
                c_launch(src, dst, 0.999, 0.001, pdl=pdl, wait=pdl, trig=pdl and trig)
            else:
                shim_set(pdl and shim_aten)
                aten_step(src, dst)
                shim_set(False)
            src, dst = dst, src
        return src
    return run


def chain_result(run_graph, out_buf, trials):
    """replay the graph on `trials` fresh random inputs; return list of outputs' checksums + mismatch count vs ref."""
    outs = []
    gen = torch.Generator(dev)
    for t in range(trials):
        gen.manual_seed(1000 + t)
        x0.copy_(torch.rand(N_SMALL, device=dev, generator=gen))
        run_graph()
        outs.append(out_buf.clone())
    torch.cuda.synchronize()
    return outs


CHAINS = {
    "Triton -> Triton": "T",
    "Triton -> CUDA C++ (PDL-aware)": "TC",
    "CUDA C++ -> CUDA C++ (PDL-aware)": "C",
    "Triton -> ATen (unmodified)": "TA",
    "Triton -> ATen (unmodified, PDL attr forced by shim, no wait)": "TA*",
    "ATen -> ATen (PDL attr forced by shim, no wait)": "A*",
}
MODES = {"off": (False, False), "on, implicit trigger": (True, False), "on, early trigger": (True, True)}
e3 = {}
for cname, pat in CHAINS.items():
    shim_aten = pat.endswith("*")
    pat = pat.rstrip("*")
    if shim_aten and not SHIM:
        continue
    row = {}
    refs = None
    for mname, (pdl, trig) in MODES.items():
        if pat == "A" and mname == "on, early trigger":
            continue  # ATen kernels have no trigger
        run = make_chain(pat, pdl, trig, shim_aten)
        out_buf = bufA if args.chain % 2 == 0 else bufB
        t, g = graph_time(run, per=args.chain + 1)
        outs = chain_result(g.replay, out_buf, args.trials)
        if refs is None:
            refs = outs
        mism = sum(not torch.equal(o, r) for o, r in zip(outs, refs))
        row[mname] = dict(us_per_kernel=round(t, 3), mismatching_trials=mism, trials=args.trials)
    off = row["off"]["us_per_kernel"]
    for mname in row:
        row[mname]["speedup_vs_off_pct"] = round(100 * (off / row[mname]["us_per_kernel"] - 1), 1)
    e3[cname] = row
    print("E3", cname, json.dumps(row), flush=True)
R["E3_dependent_chains_cuda_graph"] = e3

# ---------------------------------------------------------------- E4: eager launches
e4 = {}
for cname, pat in (("Triton -> CUDA C++ (PDL-aware)", "TC"), ("Triton -> Triton", "T")):
    row = {}
    for mname, (pdl, trig) in MODES.items():
        row[mname] = round(eager_time(make_chain(pat, pdl, trig, False), per=args.chain + 1), 3)
    e4[cname] = row
    print("E4", cname, row, flush=True)
R["E4_eager_us_per_kernel"] = e4

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(R, indent=1))
print("wrote", args.out)
