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
parser.add_argument("--inductor-pdl", type=int, default=1, help="with --mode pdl: also enable Inductor PDL codegen")
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
model = cfg.create_instance().cuda().train()
num_params = sum(p.numel() for p in model.parameters())

batch = collate_inputs_to_batched_atom_input(
    [make_example(args.seq_len, args.msa_depth, args.num_templates, cfg.dim_atom_inputs)],
    atoms_per_window=model.atoms_per_window,
)
batch = batch.cuda() if hasattr(batch, "cuda") else batch
batch_kwargs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.dict().items()}
num_atoms = int(batch_kwargs["molecule_atom_lens"].sum())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# dynamic=False: with the default automatic-dynamic shapes, Dynamo 2.10 crashes while tracing MegaFold's
# user-defined Triton launches once a dim turns symbolic ("SymNodeVariable has no attribute value").
# The benchmark input is fixed, so static graphs are the natural setting anyway.
compiled = torch.compile(model, dynamic=False)


def train_step(step: int):
    loss, breakdown = compiled(
        **batch_kwargs,
        dtype=torch.bfloat16,
        return_loss_breakdown=True,
        call_confidence_head=args.confidence_head,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


step_times, losses = [], []
total = args.warmup + args.iters
for step in range(total):
    seed_all(args.seed + step)  # identical RNG stream per step in both variants
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
)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("step_ms_all", "losses")}, indent=2))
dist.destroy_process_group()
