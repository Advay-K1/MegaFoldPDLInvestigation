# Copyright 2026 UIUC SSAIL
# Part of MegaFold project
#
# Licensed under the MIT License. See LICENSE file in the project root
# for full license text.

"""CUDA Programmatic Dependent Launch (PDL) switch for MegaFold's hand-written Triton kernels.

PDL (Hopper+, CUDA >= 12.3) lets kernel N+1 on a stream be *launched* while kernel N is still
running. Kernel N+1's CTAs get scheduled once every CTA of kernel N has executed
`griddepcontrol.launch_dependents` (or exited), run their prologue, and block in
`griddepcontrol.wait` until kernel N has fully completed and its memory is visible.
See https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html

Contract used by every MegaFold kernel when PDL is on:
  * the kernel is launched with `launch_pdl=LAUNCH_PDL` (sets
    cudaLaunchAttributeProgrammaticStreamSerialization on the launch);
  * its first statement is `pdl_prologue()`, which does `gdc_wait()` *before any global
    load or store* (so no RAW/WAR hazard with the predecessor is possible) and then
    `gdc_launch_dependents()` so the next PDL kernel can be scheduled as early as possible.
    Triggering early is safe: a dependent's `gdc_wait()` always waits for this grid's full
    completion and memory flush, the trigger only affects *scheduling*.

The switch is read once, at import time, from MEGAFOLD_PDL=1. When off, `USE_PDL` is a
False constexpr, so the `gdc_*` instructions are compiled out and launches are ordinary:
the kernels are identical to the original MegaFold kernels.
"""

import os

import torch
import triton
import triton.language as tl


def _pdl_requested() -> bool:
    return os.environ.get("MEGAFOLD_PDL", "0") == "1"


def _pdl_supported() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


PDL_ENABLED: bool = _pdl_requested() and _pdl_supported()
if _pdl_requested() and not PDL_ENABLED:
    print("[megafold.pdl] MEGAFOLD_PDL=1 but device is not CUDA sm_90+; PDL disabled.")

# --- knobs (only read when MEGAFOLD_PDL=1) -------------------------------------------------------
# MEGAFOLD_PDL_KERNELS = all | tiled
#   all   : every MegaFold Triton kernel uses PDL.
#   tiled : only the tiled kernels (EvoFlash-3D attention x4, LayernormLinear matmul fwd/bwd) use PDL;
#           the row-wise kernels launched with one tiny CTA per row (LN statistics, LN dX, SwiGLU fwd/bwd,
#           grid = (M,) with M = n*n ~ 1.5e5) do not. Those pay the per-CTA griddepcontrol.wait cost
#           on ~1.5e5 CTAs each, which measured far larger than the launch latency PDL hides.
# MEGAFOLD_PDL_LAUNCH = 1 | 0     0 = compile the gdc_* instructions but launch normally (diagnostic:
#                                 isolates the codegen cost from the scheduling effect).
# MEGAFOLD_PDL_TRIGGER = early | implicit
#   early    : gdc_launch_dependents() right after gdc_wait() (successor may be scheduled ASAP);
#   implicit : no explicit trigger; successor is released as each CTA exits (CUDA default).
PDL_KERNELS = os.environ.get("MEGAFOLD_PDL_KERNELS", "all")
PDL_TRIGGER = os.environ.get("MEGAFOLD_PDL_TRIGGER", "early")
_LAUNCH = os.environ.get("MEGAFOLD_PDL_LAUNCH", "1") == "1"
assert PDL_KERNELS in ("all", "tiled"), PDL_KERNELS
assert PDL_TRIGGER in ("early", "implicit"), PDL_TRIGGER

_rowwise_on = PDL_ENABLED and PDL_KERNELS == "all"

# Python-side launch flags, passed as `launch_pdl=` to `kernel[grid](...)`.
LAUNCH_PDL: bool = PDL_ENABLED and _LAUNCH
LAUNCH_PDL_ROWWISE: bool = _rowwise_on and _LAUNCH
# Device-side flags: global constexprs readable from inside @triton.jit functions.
USE_PDL = tl.constexpr(PDL_ENABLED)
USE_PDL_ROWWISE = tl.constexpr(_rowwise_on)
EARLY_TRIGGER = tl.constexpr(PDL_TRIGGER == "early")


def describe() -> dict:
    return dict(enabled=PDL_ENABLED, kernels=PDL_KERNELS if PDL_ENABLED else None,
                launch=_LAUNCH if PDL_ENABLED else None, trigger=PDL_TRIGGER if PDL_ENABLED else None)


@triton.jit
def pdl_prologue():
    """Tiled kernels: wait for the predecessor grid, then (optionally) release the successor grid."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def pdl_prologue_rowwise():
    """Row-wise (one-CTA-per-row) kernels: same contract, gated separately by MEGAFOLD_PDL_KERNELS."""
    if USE_PDL_ROWWISE:
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()


def _patch_inductor_mutation_analysis() -> None:
    """Keep torch.compile correct for PDL-instrumented user Triton kernels.

    Under torch.compile, Dynamo traces MegaFold's Triton launches into the graph and Inductor
    statically analyses each kernel's TTIR to find which pointer arguments it writes
    (`torch._higher_order_ops.triton_kernel_wrap.analyze_kernel_mutations`). Any non-pure
    `tt.elementwise_inline_asm` makes that analysis bail out ("assuming every input is mutated").
    `gdc_wait` / `gdc_launch_dependents` are exactly such ops (non-pure on purpose, so they are not
    reordered or eliminated). On this model the bail-out path produced uninitialised outputs (NaN
    loss), even with PDL launches disabled.

    An inline-asm op with *no operands* receives no pointer, so it cannot mutate any kernel
    parameter: drop exactly those ops before the analysis. Inline asm that takes operands still
    triggers the conservative bail-out.
    """
    try:
        from torch._higher_order_ops import triton_kernel_wrap as tkw
    except ImportError:  # pragma: no cover
        return
    # Hook the TTIR -> op-list parse (called once per analysis) rather than
    # `analyze_kernel_mutations`, which is a memoizing object with a `.reset()` the caller relies on.
    if getattr(tkw.ttir_to_functions, "_megafold_pdl_patched", False):
        return
    orig = tkw.ttir_to_functions

    def ttir_to_functions(ttir_module):
        functions = orig(ttir_module)
        for body in functions.values():
            for key, ops in body.items():
                body[key] = [op for op in ops if not (op.name == "tt.elementwise_inline_asm" and len(op.args) == 0)]
        return functions

    ttir_to_functions._megafold_pdl_patched = True
    tkw.ttir_to_functions = ttir_to_functions

if PDL_ENABLED:
    _patch_inductor_mutation_analysis()
