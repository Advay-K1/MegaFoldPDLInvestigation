"""Minimal repro: MegaFold SwiGLU / LayernormLinear row kernels vs. a PyTorch reference, fp32 and bf16.

    MEGAFOLD_PDL=0 python benchmarks/pdl_repro_swiglu.py
    MEGAFOLD_PDL=1 MEGAFOLD_PDL_LAUNCH=0 python benchmarks/pdl_repro_swiglu.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from megafold.model import pdl  # noqa: E402
from megafold.model.FusedLayernormLinear.fused_layernorm_linear import LayernormLinear  # noqa: E402
from megafold.model.FusedTransition.fused_transition import FusedSwiGLU  # noqa: E402

print("pdl:", pdl.describe(), "| triton cache:", os.environ.get("TRITON_CACHE_DIR"))
torch.manual_seed(0)
for dtype in (torch.float32, torch.bfloat16):
    for shape in [(4, 384, 3204), (1, 384, 384, 512), (1, 384, 1536 * 2)]:
        x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        y = FusedSwiGLU()(x)
        a, b = x.float().chunk(2, dim=-1)
        ref = F.silu(b) * a
        bad = int((~torch.isfinite(y)).sum())
        err = (y.float() - ref).abs().max().item() if bad == 0 else float("nan")
        g = torch.randn_like(y)
        (gx,) = torch.autograd.grad(y, x, g)
        xr = x.detach().float().requires_grad_()
        ar, br = xr.chunk(2, dim=-1)
        (gr,) = torch.autograd.grad(F.silu(br) * ar, xr, g.float())
        gbad = int((~torch.isfinite(gx)).sum())
        gerr = (gx.float() - gr).abs().max().item() if gbad == 0 else float("nan")
        print(f"swiglu {str(dtype):15s} {str(shape):22s} fwd non-finite={bad:9d} max_err={err:.2e} | "
              f"bwd non-finite={gbad:9d} max_err={gerr:.2e}")
    for K, N in [(128, 128), (384, 768), (768, 384)]:
        m = LayernormLinear(K, N, has_linear_bias=False).cuda()
        x = torch.randn(1, 384, 384 if K == 128 else 1, K, device="cuda", dtype=dtype)
        y = m(x)
        ref = F.layer_norm(x.float(), (K,), m.WEIGHT.float(), m.BIAS.float(), 1e-5) @ m.linear_weight.float()
        bad = int((~torch.isfinite(y)).sum())
        err = (y.float() - ref).abs().max().item() if bad == 0 else float("nan")
        print(f"lnlin  {str(dtype):15s} K={K:4d} N={N:4d} rows={x.numel() // K:7d}  fwd non-finite={bad:9d} max_err={err:.2e}")
