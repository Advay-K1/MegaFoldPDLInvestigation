"""Single-GPU MegaFold training-step benchmark (baseline vs. CUDA Programmatic Dependent Launch).

One "training step" mirrors `megafold/trainer.py` for the 1x1 config:
    loss = model(**batch, dtype=bf16, return_loss_breakdown=True, call_confidence_head=...)
    loss.backward(); clip_grad_norm_(10.0); Adam(lr=1e-4).step(); zero_grad()

The model is wrapped with `torch.compile(model)` exactly as the trainer does with
`use_torch_compile=True`. Inputs are a fixed synthetic protein-like example (batch size 1,
`--seq-len` tokens) collated with the repo's own `collate_inputs_to_batched_atom_input`, so
the step is shape-stable and the two variants see bit-identical inputs.

PDL is selected with the environment variable MEGAFOLD_PDL=1, which must be set *before*
MegaFold is imported (see `megafold/model/pdl.py`). With --mode pdl this script also turns on
`torch._inductor.config.triton.enable_pdl` so Inductor-generated Triton kernels use PDL too.

Usage:
    TORCH_COMPILE_DEBUG=1 python benchmarks/train_step_bench.py --mode baseline --out results/baseline.json
    TORCH_COMPILE_DEBUG=1 MEGAFOLD_PDL=1 python benchmarks/train_step_bench.py --mode pdl --out results/pdl.json
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # megafold resolves data/ccd_data relative to the repo root

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["baseline", "pdl"], required=True)
parser.add_argument("--config", default="configs/megafold_1x1.yaml")
parser.add_argument("--seq-len", type=int, default=384)
parser.add_argument("--msa-depth", type=int, default=512)
parser.add_argument("--num-templates", type=int, default=4)
parser.add_argument("--warmup", type=int, default=3, help="steps excluded from statistics (first one compiles)")
parser.add_argument("--iters", type=int, default=10, help="timed steps after warmup")
parser.add_argument("--confidence-head", action="store_true",
                    help="call the confidence head every step (trainer does it every 10th step)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--nvtx-step", type=int, default=-1,
                    help="if >=0, wrap this step index in cudaProfilerStart/Stop (for nsys --capture-range)")
parser.add_argument("--ac-compile", choices=["none", "inner"], default="none",
                    help="inner: torch.compile every layer of each activation-checkpointed stack and keep the "
                         "checkpoint loops eager (needs MEGAFOLD_OPAQUE_KERNELS=1); Report 4")
parser.add_argument("--inductor-pdl", type=int, default=1, help="with --mode pdl: also enable Inductor PDL codegen")
parser.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "cg-selective"],
                    help="reduce-overhead = Inductor CUDA graphs (cudagraph trees)")
parser.add_argument("--explain", action="store_true", help="dump torch._dynamo.explain graph-break census and exit")
parser.add_argument("--no-checkpoint", action="store_true", help="disable all activation checkpointing in the model config")
parser.add_argument("--nvtx-modules", action="store_true", help="NVTX ranges per top-level submodule forward + bwd/opt")
parser.add_argument("--out", required=True)
args = parser.parse_args()

want_pdl = args.mode == "pdl"
env_pdl = os.environ.get("MEGAFOLD_PDL", "0") == "1"
assert want_pdl == env_pdl, "--mode pdl requires MEGAFOLD_PDL=1 (and baseline requires it unset)"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

torch.set_float32_matmul_precision("high")  # same as train.py
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 4096

if want_pdl and args.inductor_pdl:
    import torch._inductor.config as inductor_config

    inductor_config.triton.enable_pdl = True

    # Bug fix for torch 2.10's experimental Inductor PDL codegen: `_handle_pdl_after_load` calls
    # `cse.generate(...)` without `shape=`, so any kernel with a load crashes with
    # "AssertionError: TritonCSEVariable must have shape". The generated value is the scalar `0`,
    # so its shape is (). Otherwise identical to torch/_inductor/codegen/triton.py.
    from torch._inductor.codegen import triton as _inductor_triton
    from torch._inductor.utils import DelayMaybeLine

    def _handle_pdl_after_load(self, launch_buffer, result_var):
        GDC_LAUNCH = "tl.extra.cuda.gdc_launch_dependents()"
        if self.inside_reduction:
            launch_buffer = self.post_loop_combine
        if _inductor_triton.enable_pdl_codegen():
            current_load_index = self._load_index
            launch_if_last_load = DelayMaybeLine(
                lambda: current_load_index == self._load_index,
                f"0; {GDC_LAUNCH} # gdc launch for {result_var}",
            )
            self.cse.generate(launch_buffer, launch_if_last_load, dtype=torch.int32, shape=())

    _inductor_triton.TritonKernel._handle_pdl_after_load = _handle_pdl_after_load

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", str(29500 + random.randint(0, 1000)))
dist.init_process_group("nccl", rank=0, world_size=1)
torch.cuda.set_device(0)

from megafold.configs import MegaFoldConfig  # noqa: E402
from megafold.distributed.parallel_info import setup_2d_parallel_groups  # noqa: E402
from megafold.inputs import (  # noqa: E402
    CONSTRAINT_DIMS,
    CONSTRAINTS_MASK_VALUE,
    DEFAULT_NUM_MOLECULE_MODS,
    IS_MOLECULE_TYPES,
    AtomInput,
    collate_inputs_to_batched_atom_input,
)
from megafold.model import pdl as megafold_pdl  # noqa: E402
from megafold.utils.model_utils import exclusive_cumsum  # noqa: E402

setup_2d_parallel_groups(data_parallel_size=1, sequence_parallel_size=1)
assert megafold_pdl.PDL_ENABLED == want_pdl

if os.environ.get("MEGAFOLD_NAN_TRACE") == "1":
    # Diagnostic: report the first fused-kernel call whose inputs are finite but outputs are not.
    from megafold.model.FusedEvoAttention.evoattention import EvoformerAttention
    from megafold.model.FusedLayernormLinear.fused_layernorm_linear import LayernormLinearFunction
    from megafold.model.FusedTransition.fused_transition import FusedSwiGLUFunction

    _reported = set()

    def _fin(ts):
        return all(bool(torch.isfinite(t).all()) for t in ts if torch.is_tensor(t) and t.is_floating_point())

    def _wrap(cls, which):
        orig = getattr(cls, which)

        def f(ctx, *a, **k):
            out = orig(ctx, *a, **k)
            outs = out if isinstance(out, tuple) else (out,)
            if _fin(a) and not _fin(outs):
                key = (cls.__name__, which, tuple(tuple(t.shape) for t in a if torch.is_tensor(t)))
                if key not in _reported:
                    _reported.add(key)
                    print(f"[nan-trace] {cls.__name__}.{which}: finite inputs -> non-finite outputs; "
                          f"in={[(tuple(t.shape), str(t.dtype), t.stride()) for t in a if torch.is_tensor(t)]} "
                          f"out={[(tuple(t.shape), int((~torch.isfinite(t)).sum())) for t in outs if torch.is_tensor(t)]}",
                          flush=True)
            return out

        setattr(cls, which, staticmethod(f))

    for _c in (EvoformerAttention, LayernormLinearFunction, FusedSwiGLUFunction):
        _wrap(_c, "forward")
        _wrap(_c, "backward")
    torch._dynamo.config.suppress_errors = False


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_example(n: int, msa_depth: int, num_templates: int, dim_atom_inputs: int) -> AtomInput:
    """A single-chain, protein-like example with n tokens (~9 atoms/token, like a real crop)."""
    g = torch.Generator().manual_seed(1234)
    molecule_atom_lens = torch.randint(6, 13, (n,), generator=g)
    m = int(molecule_atom_lens.sum())
    offsets = exclusive_cumsum(molecule_atom_lens)

    is_molecule_types = torch.zeros(n, IS_MOLECULE_TYPES, dtype=torch.bool)
    is_molecule_types[:, 0] = True  # protein
    idx = torch.arange(n)
    zeros = torch.zeros(n, dtype=torch.long)
    additional_molecule_feats = torch.stack([idx, idx, zeros, zeros, zeros], dim=-1)

    token_bonds = torch.zeros(n, n, dtype=torch.bool)
    token_bonds[idx[:-1], idx[1:]] = True
    token_bonds[idx[1:], idx[:-1]] = True

    center = offsets + molecule_atom_lens // 2
    return AtomInput(
        atom_inputs=torch.randn(m, dim_atom_inputs, generator=g),
        atompair_inputs=torch.randn(m, m, 5, generator=g),
        molecule_ids=torch.randint(0, 20, (n,), generator=g),
        token_bonds=token_bonds,
        molecule_atom_lens=molecule_atom_lens,
        additional_molecule_feats=additional_molecule_feats,
        additional_msa_feats=torch.randn(msa_depth, n, 2, generator=g),
        additional_token_feats=torch.randn(n, 33, generator=g),
        is_molecule_types=is_molecule_types,
        is_molecule_mod=torch.zeros(n, DEFAULT_NUM_MOLECULE_MODS, dtype=torch.bool),
        templates=torch.randn(num_templates, n, n, 108, generator=g),
        template_mask=torch.ones(num_templates, dtype=torch.bool),
        msa=torch.randn(msa_depth, n, 32, generator=g),
        msa_mask=torch.ones(msa_depth, dtype=torch.bool),
        atom_pos=torch.randn(m, 3, generator=g) * 10.0,
        molecule_atom_indices=center,
        distogram_atom_indices=center,
        resolved_labels=torch.ones(m, dtype=torch.long),
        chains=torch.tensor([0, -1]),
        atom_ids=torch.randint(0, 47, (m,), generator=g),          # config: num_atom_embeds=47
        atompair_ids=torch.randint(0, 11, (m, m), generator=g),    # config: num_atompair_embeds=11
        # constraints [pocket, contact, docking] -> 1+1+4 dims, all 'no constraint' (mask value)
        token_constraints=torch.full((n, n, sum(CONSTRAINT_DIMS.values())), CONSTRAINTS_MASK_VALUE),
    )


seed_all(args.seed)
cfg = MegaFoldConfig.from_yaml_file(args.config, ["model"])
if args.no_checkpoint:
    for k in ("checkpoint_input_embedding", "checkpoint_trunk_pairformer", "checkpoint_diffusion_module",
              "checkpoint_distogram_head", "checkpoint_confidence_head"):
        setattr(cfg, k, False)
model = cfg.create_instance().cuda().train()
if os.environ.get("MEGAFOLD_FALLBACK_RANDOM") == "1":
    # compiled code otherwise draws its own seeds from the global generator, shifting every later random
    # draw (diffusion noise, augmentation) -> eager and compiled runs see different noise. With fallback_random
    # Inductor calls the same ATen random ops as eager, so losses are comparable step by step.
    import torch._inductor.config as _ic
    _ic.fallback_random = True
if os.environ.get("MEGAFOLD_NO_DROPOUT") == "1":
    # numerical comparisons between eager and compiled layers: their dropout RNG streams differ. p = 1e-12
    # keeps the dropout ops in the graph (p = 0 changes the graph) while never dropping anything.
    for _d in model.modules():
        if isinstance(_d, torch.nn.Dropout):
            _d.p = 1e-12
num_params = sum(p.numel() for p in model.parameters())

batch = collate_inputs_to_batched_atom_input(
    [make_example(args.seq_len, args.msa_depth, args.num_templates, cfg.dim_atom_inputs)],
    atoms_per_window=model.atoms_per_window,
)
batch = batch.cuda() if hasattr(batch, "cuda") else batch
batch_kwargs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.dict().items()}
num_atoms = int(batch_kwargs["molecule_atom_lens"].sum())

def compile_checkpointed_layers(model):
    """Report 4: compile the layers *inside* MegaFold's activation checkpoints. Each checkpointed stack
    (MSAModule, PairformerStack, TemplateEmbedder, DiffusionTransformer, ...) loops over `self.layers` /
    `self.pairformer_stack` and wraps every layer call in torch.utils.checkpoint. We replace each leaf layer
    module with torch.compile(layer) and mark the to_checkpointed_* loop itself torch.compiler.disable, so
    the outer compiled model does not trace through the checkpoints (doing so keeps ~1.8x the activations
    and runs out of memory on the full model) while every layer still runs as compiled Inductor code."""
    from torch import nn
    stacks = []
    for m in model.modules():
        meths = [a for a in dir(type(m)) if a.startswith("to_checkpointed")]
        if meths and getattr(m, "checkpoint", False):
            stacks.append((m, meths))
    n = 0

    def comp(ml):
        nonlocal n
        for i, ch in enumerate(ml):
            if isinstance(ch, nn.ModuleList):
                comp(ch)
            elif ch is not None:
                ml[i] = torch.compile(ch, dynamic=False); n += 1

    for m, meths in stacks:
        for attr in ("layers", "pairformer_stack"):
            if isinstance(getattr(m, attr, None), nn.ModuleList):
                comp(getattr(m, attr))
        for a in meths:
            setattr(m, a, torch.compiler.disable(getattr(m, a)))
    return dict(stacks=[type(m).__name__ for m, _ in stacks], compiled_layers=n)


ac_compile_info = None
if args.ac_compile == "inner":
    assert megafold_pdl.OPAQUE_KERNELS, "--ac-compile inner requires MEGAFOLD_OPAQUE_KERNELS=1"
    ac_compile_info = compile_checkpointed_layers(model)
    print("[ac-compile]", ac_compile_info, flush=True)

if args.compile_mode == "reduce-overhead":
    # CUDA graphs: the stochastic frame averaging in the diffusion module runs a 3x3 torch.linalg.eigh
    # (cuSOLVER), which fails inside a captured CUDA graph (CUSOLVER_STATUS_INTERNAL_ERROR). Keep that tiny
    # op out of the compiled/captured regions.
    import frame_averaging_pytorch.frame_averaging as _fa
    _fa.FrameAverage.forward = torch.compiler.disable(_fa.FrameAverage.forward)

if args.nvtx_modules:
    import torch.cuda.nvtx as nvtx

    def _hook_pair(name):
        def pre(m, a):  # must return None (a non-None return replaces the module's args)
            nvtx.range_push(name)

        def post(m, a, o):
            nvtx.range_pop()

        return pre, post

    targets = [(n, m) for n, m in model.named_children()]
    targets += [(f"diffusion_module.{n}", m) for n, m in model.diffusion_module.named_children()]
    for n, m in targets:
        pre, post = _hook_pair(n)
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# dynamic=False: with the default automatic-dynamic shapes, Dynamo 2.10 crashes while tracing MegaFold's
# user-defined Triton launches once a dim turns symbolic ("SymNodeVariable has no attribute value").
# The benchmark input is fixed, so static graphs are the natural setting anyway.
if args.compile_mode == "cg-selective":
    # CUDA graphs only on the heavy, static, host-sync-free submodules; loss / alignment code stays eager
    # (it contains cuSOLVER SVD/eigh and .item() syncs that cannot be captured).
    CG_TARGETS = [(model, "pairformer"), (model, "msa_module"), (model, "template_embedder"),
                  (model.diffusion_module, "atom_encoder"), (model.diffusion_module, "token_transformer"),
                  (model.diffusion_module, "atom_decoder")]
    for parent, attr in CG_TARGETS:
        setattr(parent, attr, torch.compile(getattr(parent, attr), dynamic=False, mode="reduce-overhead"))
    compiled = model
else:
    compiled = torch.compile(model, dynamic=False, mode=None if args.compile_mode == "default" else args.compile_mode)


def train_step(step: int):
    loss, breakdown = compiled(
        **batch_kwargs,
        dtype=torch.bfloat16,
        return_loss_breakdown=True,
        call_confidence_head=args.confidence_head,
    )
    if args.nvtx_modules: torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if args.nvtx_modules: torch.cuda.nvtx.range_pop(); torch.cuda.nvtx.range_push("clip+optimizer")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    if args.nvtx_modules: torch.cuda.nvtx.range_pop()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


if args.explain:
    import collections
    seed_all(args.seed)
    # same static-shape setting as torch.compile(dynamic=False) (explain() has no `dynamic` flag)
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.assume_static_by_default = True
    ex = torch._dynamo.explain(model)(**batch_kwargs, dtype=torch.bfloat16, return_loss_breakdown=True,
                                     call_confidence_head=args.confidence_head)
    reasons = collections.Counter()
    where = collections.Counter()
    for br in ex.break_reasons:
        reasons[str(br.reason).split("\n")[0][:160]] += 1
        if br.user_stack:
            fr = br.user_stack[-1]
            where[f"{Path(fr.filename).name}:{fr.lineno} {fr.name}"] += 1
    rep = dict(graph_count=ex.graph_count, graph_break_count=ex.graph_break_count, op_count=ex.op_count,
               top_reasons=reasons.most_common(25), top_locations=where.most_common(25))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2)[:4000])
    sys.exit(0)

# Report 3: force the PDL launch attribute onto every runtime (<<<>>>) launch, i.e. every eager ATen kernel,
# via the LD_PRELOAD shim benchmarks/pdl_xlang/libpdl_shim.so. ATen kernels cannot griddepcontrol.wait, so
# this is only defensible when no kernel triggers early (MEGAFOLD_PDL_TRIGGER=implicit, --inductor-pdl 0).
_shim = None
if os.environ.get("MEGAFOLD_PDL_SHIM_ATEN") == "1":
    import ctypes
    assert "libpdl_shim" in os.environ.get("LD_PRELOAD", ""), "MEGAFOLD_PDL_SHIM_ATEN=1 needs LD_PRELOAD shim"
    assert not (want_pdl and args.inductor_pdl), "Inductor PDL kernels trigger early: use --inductor-pdl 0"
    _shim = ctypes.CDLL(str(ROOT / "benchmarks" / "pdl_xlang" / "libpdl_shim.so"))
    _shim.pdl_shim_count.restype = ctypes.c_long
    _shim.pdl_shim_set(1)
shim_counts = []

step_times, losses = [], []
total = args.warmup + args.iters
for step in range(total):
    seed_all(args.seed + step)  # identical RNG stream per step in both variants
    if args.compile_mode in ("reduce-overhead", "cg-selective"):
        torch.compiler.cudagraph_mark_step_begin()
    if step == args.nvtx_step:
        torch.cuda.profiler.start()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = train_step(step)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if step == args.nvtx_step:
        torch.cuda.profiler.stop()
    step_times.append(dt)
    losses.append(float(loss))
    if _shim is not None:
        shim_counts.append(_shim.pdl_shim_count())
    print(f"[{args.mode}] step {step:3d}  {dt * 1e3:9.1f} ms  loss {float(loss):.5f}", flush=True)
    if step == 0:
        torch.cuda.reset_peak_memory_stats()

timed = step_times[args.warmup:]
result = dict(
    mode=args.mode,
    pdl_enabled=megafold_pdl.PDL_ENABLED,
    inductor_enable_pdl=bool(want_pdl and args.inductor_pdl),
    megafold_pdl=megafold_pdl.describe(),
    torch_compile_debug=os.environ.get("TORCH_COMPILE_DEBUG", "0"),
    compile_mode=args.compile_mode,
    seq_len=args.seq_len,
    num_atoms=num_atoms,
    msa_depth=args.msa_depth,
    num_templates=args.num_templates,
    batch_size=1,
    confidence_head=args.confidence_head,
    num_params=num_params,
    warmup=args.warmup,
    iters=args.iters,
    first_step_s=step_times[0],
    step_ms_mean=statistics.mean(timed) * 1e3,
    step_ms_median=statistics.median(timed) * 1e3,
    step_ms_stdev=(statistics.stdev(timed) * 1e3) if len(timed) > 1 else 0.0,
    step_ms_min=min(timed) * 1e3,
    step_ms_all=[t * 1e3 for t in step_times],
    losses=losses,
    peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30,
    gpu=torch.cuda.get_device_name(0),
    torch=torch.__version__,
    cuda=torch.version.cuda,
    triton=__import__("triton").__version__,
    host=os.uname().nodename,
    source_tree=str(ROOT),
    git_commit=__import__('subprocess').run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'],
                                            capture_output=True, text=True).stdout.strip(),
    slurm_job=os.environ.get("SLURM_JOB_ID"),
    ac_compile=args.ac_compile, ac_compile_info=ac_compile_info,
    aten_pdl_shim=_shim is not None,
    aten_pdl_shim_launches_per_step=[b - a for a, b in zip(shim_counts, shim_counts[1:])],
)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("step_ms_all", "losses")}, indent=2))
dist.destroy_process_group()
