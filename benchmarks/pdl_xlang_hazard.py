"""Is it safe to force the PDL launch attribute onto unmodified ATen kernels (LD_PRELOAD shim), which cannot
execute griddepcontrol.wait? Widen the race window: the Triton primary does WORK dependent FMAs per element
*before* its stores, with an early (gdc_launch_dependents at entry) or implicit trigger; the secondary reads
the primary's output. Secondary = ATen add with the shim (no wait) vs. a PDL-aware CUDA C++ kernel (waits).
Counts trials whose output differs from the PDL-off reference.

    LD_PRELOAD=$PWD/benchmarks/pdl_xlang/libpdl_shim.so python benchmarks/pdl_xlang_hazard.py \
        --out results/report3/hazard.json
"""
import argparse
import ctypes
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent / "pdl_xlang"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--trials", type=int, default=200)
args = ap.parse_args()
ext = load(name="pdl_xlang_ext", sources=[str(HERE / "pdl_ext.cu")], build_directory=str(HERE / "build"),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
shim = ctypes.CDLL(str(HERE / "libpdl_shim.so"))


@triton.jit
def tri_axpb(src, dst, n, a, b, WAIT: tl.constexpr, TRIG: tl.constexpr, WORK: tl.constexpr, BLOCK: tl.constexpr):
    if WAIT:
        tl.extra.cuda.gdc_wait()
    if TRIG:
        tl.extra.cuda.gdc_launch_dependents()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(src + offs, mask=m)
    for _ in range(WORK):
        v = v * 0.999 + 0.001
    tl.store(dst + offs, v * a + b, mask=m)


N = 132 * 512 * 8
x = torch.empty(N, device="cuda"); y = torch.empty_like(x); z = torch.empty_like(x)
res = {}
for work in (0, 64, 512, 4096):
    for trig_name, trig in (("implicit trigger", False), ("early trigger", True)):
        for sec in ("ATen add + shim (no wait)", "CUDA C++ PDL-aware (waits)"):
            def run(pdl):
                tri_axpb[(triton.cdiv(N, 512),)](x, y, N, 1.0, 1.0, WAIT=False, TRIG=pdl and trig, WORK=work,
                                                  BLOCK=512, num_warps=4)
                if sec.startswith("ATen"):
                    shim.pdl_shim_set(int(pdl)); torch.add(y, 1.0, out=z); shim.pdl_shim_set(0)
                else:
                    ext.axpb(y, z, 1.0, 1.0, pdl, pdl, False, 0, triton.cdiv(N, 256), 256)
            graphs = {}
            for pdl in (False, True):
                st = torch.cuda.Stream()
                with torch.cuda.stream(st):
                    run(pdl); torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=st):
                        run(pdl)
                graphs[pdl] = g
            bad = 0; max_err = 0.0
            for t in range(args.trials):
                x.normal_(); y.fill_(float("nan")); z.zero_()   # stale y = NaN: a racing read is visible
                graphs[False].replay(); torch.cuda.synchronize(); ref = z.clone()
                y.fill_(float("nan")); z.zero_()
                graphs[True].replay(); torch.cuda.synchronize()
                if not torch.equal(z, ref):
                    bad += 1
                    max_err = max(max_err, (z != ref).float().mean().item())
            key = f"WORK={work} | {trig_name} | {sec}"
            res[key] = dict(bad_trials=bad, trials=args.trials, worst_frac_wrong_elements=round(max_err, 4))
            print(key, res[key], flush=True)
Path(args.out).write_text(json.dumps(res, indent=1))
