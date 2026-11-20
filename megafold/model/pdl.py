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

# Python-side launch flag, passed as `launch_pdl=` to `kernel[grid](...)`.
LAUNCH_PDL: bool = PDL_ENABLED
# Device-side flag, a global constexpr readable from inside @triton.jit functions.
USE_PDL = tl.constexpr(PDL_ENABLED)


@triton.jit
def pdl_prologue():
    """Wait for the predecessor grid, then let the successor grid get scheduled."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
