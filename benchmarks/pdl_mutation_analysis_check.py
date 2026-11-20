"""Check Inductor's user-kernel mutation analysis on MegaFold's PDL-instrumented kernels.

Prints, for each kernel, which pointer args Inductor believes are written. With PDL instrumentation
and without the fix in megafold/model/pdl.py this degrades to 'every input mutated'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from torch._higher_order_ops.triton_kernel_wrap import identify_mutated_tensors  # noqa: E402

from megafold.model import pdl  # noqa: E402
from megafold.model.FusedTransition import fused_transition as ft  # noqa: E402
from megafold.model.FusedLayernormLinear import fused_layernorm_linear as fl  # noqa: E402

print("pdl:", pdl.describe())
x = torch.randn(1536, 512, device="cuda"); y = torch.empty(1536, 256, device="cuda"); dx = torch.empty_like(x)
print("swiglu fwd :", identify_mutated_tensors(ft._swiglu_forward_kernel, dict(x_ptr=x, y_ptr=y, stride_y=256, D=512, BLOCK_SIZE=256), {}))
print("swiglu bwd :", identify_mutated_tensors(ft._swiglu_backward_kernel, dict(dy_ptr=y, x_ptr=x, dx_ptr=dx, stride=256, D=256, BLOCK_SIZE=256), {}))
X = torch.randn(1536, 128, device="cuda"); M = torch.empty(1536, device="cuda"); Rs = torch.empty(1536, device="cuda")
print("LN stats   :", identify_mutated_tensors(fl._first_forward_kernel, dict(X_ptr=X, X_row_stride=128, Mean_ptr=M, Mean_row_stride=1, RSTD_ptr=Rs, RSTD_row_stride=1, K=128, eps=1e-5, BLOCK_SIZE=128), {}))
