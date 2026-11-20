"""Correctness + speed check of MegaFold's hand-written Triton kernels with and without PDL.

Runs (eagerly, no torch.compile) a pairformer-shaped stack of blocks built only from MegaFold's
fused kernels -- LayernormLinear -> EvoFlash-3D triangle attention -> FusedTransition (SwiGLU) --
forward + backward, with a fixed seed. Run it once per mode, then compare:

    python benchmarks/pdl_kernel_check.py --out results/kernels_baseline.pt
    MEGAFOLD_PDL=1 python benchmarks/pdl_kernel_check.py --out results/kernels_pdl.pt
    python benchmarks/pdl_kernel_check.py --compare results/kernels_baseline.pt results/kernels_pdl.pt
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

p = argparse.ArgumentParser()
p.add_argument("--out")
p.add_argument("--compare", nargs=2)
p.add_argument("--n", type=int, default=384)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--iters", type=int, default=20)
args = p.parse_args()

import torch  # noqa: E402

if args.compare:
    a, b = (torch.load(f) for f in args.compare)
    report = {"max_abs_diff": {}, "bitwise_equal": {}}
    for k in a["tensors"]:
        x, y = a["tensors"][k].float(), b["tensors"][k].float()
        report["max_abs_diff"][k] = (x - y).abs().max().item()
        report["bitwise_equal"][k] = bool(torch.equal(a["tensors"][k], b["tensors"][k]))
    report["all_bitwise_equal"] = all(report["bitwise_equal"].values())
    report["A"], report["B"] = a["timing"], b["timing"]
    report["speedup_B_over_A"] = a["timing"]["fwd_bwd_ms_median"] / b["timing"]["fwd_bwd_ms_median"]
    print(json.dumps(report, indent=2))
    Path(args.compare[1]).with_name("kernels_compare.json").write_text(json.dumps(report, indent=2))
    sys.exit(0 if report["all_bitwise_equal"] else 1)

from einops import rearrange  # noqa: E402

from megafold.model import pdl  # noqa: E402
from megafold.model.FusedEvoAttention.evoattention import TritonEvoformer  # noqa: E402
from megafold.model.FusedLayernormLinear.fused_layernorm_linear import LayernormLinear  # noqa: E402
from megafold.model.FusedTransition.fused_transition import FusedTransition  # noqa: E402

mode = "pdl" if pdl.PDL_ENABLED else "baseline"
torch.manual_seed(0)
dev = "cuda"
n, d, h = args.n, args.dim, args.heads
dh = d // h


class Block(torch.nn.Module):
    """Triangle-attention-like block, using only MegaFold fused kernels for LN/linear/attn/SwiGLU."""

    def __init__(self):
        super().__init__()
        self.to_qkvb = LayernormLinear(d, 3 * d + h, has_linear_bias=False)
        self.to_out = torch.nn.Linear(d, d, bias=False)
        self.transition = FusedTransition(dim=d)

    def forward(self, x, mask):  # x: (1, n, n, d)
        qkvb = self.to_qkvb(x)
        q, k, v, bias = qkvb.split([d, d, d, h], dim=-1)
        q, k, v = (rearrange(t, "b i j (h e) -> b i j h e", h=h).bfloat16().contiguous() for t in (q, k, v))
        bias = rearrange(bias, "b i j h -> b 1 h i j").bfloat16().contiguous()
        o = TritonEvoformer(q, k, v, mask, bias)
        x = x + self.to_out(rearrange(o, "b i j h e -> b i j (h e)").to(x.dtype))
        return x + self.transition(x)


blocks = torch.nn.ModuleList([Block() for _ in range(args.depth)]).to(dev)
for prm in blocks.parameters():  # LayernormLinear leaves its params uninitialised
    torch.nn.init.normal_(prm, std=0.02) if prm.dim() > 1 else torch.nn.init.ones_(prm)
x0 = torch.randn(1, n, n, d, device=dev)
mask = torch.zeros(1, n, 1, 1, n, device=dev, dtype=torch.bfloat16)


def run():
    x = x0.clone().requires_grad_()
    y = x
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for blk in blocks:
            y = blk(y, mask)
    loss = y.float().pow(2).mean()
    loss.backward()
    return x, y, loss


# correctness snapshot (first call also JIT-compiles every kernel)
blocks.zero_grad(set_to_none=True)
x, y, loss = run()
torch.cuda.synchronize()
tensors = {"y": y.detach().cpu(), "dx": x.grad.cpu(), "loss": loss.detach().cpu()}
for i, blk in enumerate(blocks):
    tensors[f"grad_to_qkvb_w_{i}"] = blk.to_qkvb.linear_weight.grad.cpu() if hasattr(blk.to_qkvb, "linear_weight") \
        else next(blk.to_qkvb.parameters()).grad.cpu()

# timing
ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)]
for _ in range(3):
    blocks.zero_grad(set_to_none=True); run()
torch.cuda.synchronize()
torch.cuda.profiler.start()  # nsys --capture-range=cudaProfilerApi: timed loop only
for s, e in ev:
    blocks.zero_grad(set_to_none=True)
    s.record(); run(); e.record()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
ms = [s.elapsed_time(e) for s, e in ev]
timing = dict(fwd_bwd_ms_median=statistics.median(ms), fwd_bwd_ms_mean=statistics.mean(ms),
              fwd_bwd_ms_stdev=statistics.stdev(ms), n=n, dim=d, heads=h, depth=args.depth, iters=args.iters)
timing["pdl"] = pdl.describe()
print(mode, json.dumps(timing))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({"mode": mode, "tensors": tensors, "timing": timing}, args.out)
