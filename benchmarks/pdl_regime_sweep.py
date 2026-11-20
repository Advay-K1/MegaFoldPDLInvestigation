"""Where does PDL pay off? A chain of truly dependent Triton kernels, PDL off/on, eager vs CUDA graph.

Each kernel reads buffer A and writes buffer B (then the next kernel reads B, writes A, ...), so every
kernel depends on its predecessor. We sweep
  * per-kernel work (inner loop iterations -> kernel duration from ~2 us to ~200 us),
  * grid shape: 1 wave of fat CTAs (132 CTAs, one per SM) vs. many tiny CTAs (one 128-wide row each,
    like MegaFold's row-wise LayerNorm/SwiGLU kernels),
  * PDL: off / on with implicit trigger / on with early gdc_launch_dependents,
  * launch: eager stream launches vs. the same chain captured in a CUDA graph.
Reported: average time per kernel in the chain (= total chain time / chain length).

    python benchmarks/pdl_regime_sweep.py --out results/report2/regime_sweep.json
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

ap = argparse.ArgumentParser()
ap.add_argument("--chain", type=int, default=256)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--out", required=True)
args = ap.parse_args()


@triton.jit
def chain_kernel(src, dst, n_rows, WORK: tl.constexpr, BLOCK: tl.constexpr, ROWS_PER_CTA: tl.constexpr,
                 USE_PDL: tl.constexpr, EARLY: tl.constexpr):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY:
            tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    for r in range(ROWS_PER_CTA):
        row = pid * ROWS_PER_CTA + r
        if row < n_rows:
            x = tl.load(src + row * BLOCK + cols)
            acc = x
            for _ in range(WORK):  # dependent FMA chain sets the kernel duration
                acc = acc * 0.999 + 0.001
            tl.store(dst + row * BLOCK + cols, acc)


BLOCK = 128
SHAPES = {
    # name: (n_rows, rows_per_cta)  -> grid = n_rows / rows_per_cta; each row = 128 fp32
    "132 fat CTAs x 4 rows (few-us kernels)": (132 * 4, 4),
    "8,448 one-row CTAs (few-us kernels)": (8448, 1),
    "67,584 one-row CTAs (MegaFold-like row-wise)": (67584, 1),
}
WORKS = [0, 16, 128, 512]
PDL_MODES = {"off": (False, False, False), "on, implicit trigger": (True, True, False), "on, early trigger": (True, True, True)}


def make_chain(n_rows, rpc, work, use_pdl, launch_pdl, early, a, b):
    grid = (triton.cdiv(n_rows, rpc),)

    def run():
        src, dst = a, b
        for _ in range(args.chain):
            chain_kernel[grid](src, dst, n_rows, WORK=work, BLOCK=BLOCK, ROWS_PER_CTA=rpc,
                               USE_PDL=use_pdl, EARLY=early, num_warps=4, launch_pdl=launch_pdl)
            src, dst = dst, src

    return run


def time_eager(run):
    run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / args.chain)
    return statistics.median(ts)


def time_graph(run):
    run(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        run()  # warm on capture stream
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=st):
            run()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(args.reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / args.chain)
    return statistics.median(ts)


results = []
for shape, (n_rows, rpc) in SHAPES.items():
    a = torch.randn(n_rows, BLOCK, device="cuda")
    b = torch.empty_like(a)
    for work in WORKS:
        row = dict(shape=shape, work=work)
        ref = None
        for pname, (use_pdl, launch_pdl, early) in PDL_MODES.items():
            for launch, timer in (("eager", time_eager), ("cuda_graph", time_graph)):
                run = make_chain(n_rows, rpc, work, use_pdl, launch_pdl, early, a, b)
                try:
                    row[f"{launch} | PDL {pname}"] = timer(run)
                except Exception as ex:  # noqa: BLE001
                    row[f"{launch} | PDL {pname}"] = None
                    row.setdefault("errors", []).append(f"{launch}/{pname}: {type(ex).__name__}: {ex}"[:300])
            # correctness: PDL output must equal the non-PDL output
            a.copy_(torch.randn(n_rows, BLOCK, device="cuda", generator=torch.Generator("cuda").manual_seed(0)))
            make_chain(n_rows, rpc, work, use_pdl, launch_pdl, early, a, b)()
            torch.cuda.synchronize()
            out = (a if args.chain % 2 == 0 else b).clone()
            if ref is None:
                ref = out
            row[f"bitwise equal to PDL off | {pname}"] = bool(torch.equal(out, ref))
        results.append(row)
        print(json.dumps(row), flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(dict(chain=args.chain, reps=args.reps, gpu=torch.cuda.get_device_name(),
                                          torch=torch.__version__, triton=triton.__version__, results=results), indent=2))
