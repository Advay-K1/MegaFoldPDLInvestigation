"""Why does torch.compile not compile MegaFold's activation-checkpointed layers, and which fix works?

A PairformerStack (the trunk's real per-block shapes: n = 384, dim_single 384, dim_pairwise 128, 16 heads,
use_optimized_evo="triton", bf16 autocast) with `depth` blocks, forward + backward, in four variants:
  eager            : as written (checkpoint around every layer), no compile
  compile_outer    : torch.compile(stack)  -- what the training step does today (whole model compiled)
  compile_inner    : each layer module compiled, the *compiled* layer wrapped by MegaFold's checkpoint
  compile_no_ac    : torch.compile(stack) with checkpointing switched off (memory reference)
Reported per variant: time per fwd+bwd, CUDA kernels launched per fwd+bwd (torch.profiler), Dynamo graph
breaks, peak memory, max |diff| of output and grads vs eager.

    python benchmarks/compile_checkpoint_probe.py --depth 4 --out results/report4/probe.json
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from collections import Counter


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TYPECHECK", "False")
import torch  # noqa: E402
import torch._dynamo  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--n", type=int, default=384)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--variants", default="eager,eager_orig,compile_outer,compile_inner,compile_no_ac")
ap.add_argument("--out", required=True)
args = ap.parse_args()

from megafold.model.megafold import PairformerStack  # noqa: E402
from megafold.distributed.parallel_info import setup_2d_parallel_groups  # noqa: E402
import torch.distributed as dist  # noqa: E402

os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29533")
if not dist.is_initialized():
    dist.init_process_group("nccl", rank=0, world_size=1)
setup_2d_parallel_groups(1, 1)
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 4096


import megafold.model.attention as _attn_mod  # noqa: E402
import megafold.model.FusedEvoAttention.evoattention as _evo  # noqa: E402
import megafold.model.FusedLayernormLinear.fused_layernorm_linear as _lnl  # noqa: E402
import megafold.model.FusedTransition.fused_transition as _swi  # noqa: E402
OPAQUE_DEFAULT = _swi.OPAQUE_KERNELS


def set_opaque(on):
    """switch MegaFold's kernels between the original autograd Functions and the custom-op versions"""
    _attn_mod.TritonEvoformer = (_evo.EvoformerAttentionOpaque if on else _evo.EvoformerAttention).apply
    _lnl._LNL_FUNCTION = _lnl.LayernormLinearFunctionOpaque if on else _lnl.LayernormLinearFunction
    _swi.OPAQUE_KERNELS = on


def build(checkpoint):
    torch.manual_seed(0)
    m = PairformerStack(dim_single=384, dim_pairwise=128, depth=args.depth, pair_bias_attn_dim_head=64,
                        pair_bias_attn_heads=16, dropout_row_prob=0.0, checkpoint=checkpoint).cuda().train()
    for d in m.modules():  # PairwiseBlock has its own 0.25 row/col dropout; compiled and eager RNG streams
        if isinstance(d, torch.nn.Dropout):  # differ, so dropout must be off for a numerical comparison
            d.p = 1e-12  # keeps the dropout ops (graph structure of the real config); p = 0 trips the min-cut partitioner
    return m


g = torch.Generator(device="cuda").manual_seed(0)
S0 = torch.randn(1, args.n, 384, device="cuda", generator=g)
P0 = torch.randn(1, args.n, args.n, 128, device="cuda", generator=g)
MASK = torch.ones(1, args.n, dtype=torch.bool, device="cuda")


def step(fn, mod):
    s = S0.clone().requires_grad_(); p = P0.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        so, po = fn(single_repr=s, pairwise_repr=p, mask=MASK, use_optimized_evo="triton")
    loss = so.float().square().mean() + po.float().square().mean()
    loss.backward()
    grads = [s.grad, p.grad] + [q.grad for q in mod.parameters()]
    return so.detach().float(), po.detach().float(), grads


def variant(name):
    set_opaque(OPAQUE_DEFAULT)
    if name == "eager":
        mod = build(True); return mod, mod
    if name == "eager_orig":  # original (non-custom-op) kernels, eager: checks the opaque path bit-for-bit
        set_opaque(False)
        mod = build(True); return mod, mod
    if name == "compile_outer":
        mod = build(True); return mod, torch.compile(mod, dynamic=False)
    if name == "compile_no_ac":
        mod = build(False); return mod, torch.compile(mod, dynamic=False)
    if name.startswith("compile_no_ac_budget"):  # compile-native activation checkpointing: memory budget
        torch._functorch.config.activation_memory_budget = float(name.split("budget")[1])
        mod = build(False); return mod, torch.compile(mod, dynamic=False)
    if name == "compile_outer_must_recompute":  # MegaFold's checkpoints, but every op inside forced to recompute
        from functools import partial
        from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
        import megafold.model.megafold as _mf
        policy = lambda ctx, op, *a, **k: CheckpointPolicy.MUST_RECOMPUTE  # noqa: E731
        _mf.checkpoint = partial(checkpoint, use_reentrant=False,
                                 context_fn=partial(create_selective_checkpoint_contexts, policy))
        mod = build(True); return mod, torch.compile(mod, dynamic=False)
    if name == "compile_inner":
        mod = build(True)
        # compile each layer module; MegaFold's to_checkpointed_layers still wraps every (compiled) layer
        # call in non-reentrant torch.utils.checkpoint, so recomputation re-runs the compiled forward.
        for i, triple in enumerate(mod.layers):
            for j in range(len(triple)):
                triple[j] = torch.compile(triple[j], dynamic=False)
        return mod, mod
    raise ValueError(name)


res = dict(depth=args.depth, n=args.n, torch=torch.__version__, gpu=torch.cuda.get_device_name())
ref = None
for name in args.variants.split(","):
    torch._dynamo.reset(); torch._dynamo.utils.counters.clear()
    torch._functorch.config.activation_memory_budget = 1.0
    import megafold.model.megafold as _mf, megafold.tensor_typing as _tt
    _mf.checkpoint = _tt.checkpoint
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    mod, fn = variant(name)
    t0 = time.perf_counter()
    try:
        out = step(fn, mod); torch.cuda.synchronize()
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc()
        res[name] = dict(error=f"{type(ex).__name__}: {ex}"[:1500]); print(name, res[name], flush=True)
        continue
    first = time.perf_counter() - t0
    for _ in range(2):
        step(fn, mod)
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.iters):
        torch.cuda.synchronize(); t = time.perf_counter(); step(fn, mod); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(fn, mod); torch.cuda.synchronize()
    kinds = Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            nm = e.name
            k = ("memcpy/memset" if ("Memcpy" in nm or "Memset" in nm) else
                 "triton (Inductor)" if nm.startswith("triton_") else
                 "triton (MegaFold)" if ("_attn_" in nm or "swiglu" in nm or "_forward_kernel" in nm or "_backward_kernel" in nm) else
                 "cuBLAS/CUTLASS" if any(x in nm.lower() for x in ("gemm", "cutlass", "nvjet", "xmma", "sm90")) else
                 "ATen eager")
            kinds[k] += 1
    gb = dict(torch._dynamo.utils.counters["graph_break"])
    so, po, grads = out
    if ref is None:
        ref = out
    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
    pairs = [(a, b) for a, b in zip(grads, ref[2]) if a is not None and b is not None]
    diff = dict(single_max_abs=float((so - ref[0]).abs().max()), pairwise_max_abs=float((po - ref[1]).abs().max()),
                single_rel_l2=rel(so, ref[0]), pairwise_rel_l2=rel(po, ref[1]),
                pairwise_max_abs_ref=float(ref[1].abs().max()),
                grads_rel_l2_worst=max(rel(a, b) for a, b in pairs),
                grads_rel_l2_median=statistics.median(rel(a, b) for a, b in pairs))
    res[name] = dict(first_call_s=round(first, 1), ms_per_fwd_bwd=round(statistics.median(ts), 2),
                     kernels_per_fwd_bwd=sum(kinds.values()), kernels_by_kind=dict(kinds),
                     peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     graph_breaks=sum(gb.values()), graph_break_reasons=dict(Counter(gb).most_common(8)),
                     max_abs_diff_vs_eager=diff)
    print(name, json.dumps(res[name]), flush=True)
    del mod, fn, out

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
