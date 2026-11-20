#!/bin/bash
# Full PDL investigation driver, run inside one GPU allocation:
#   srun --jobid=<holder> --overlap -n1 bash scripts/run_investigation.sh
# BASE_TREE = checkout with the original kernels (baseline), this tree = PDL kernels.
set -uo pipefail
PDL_TREE=$(cd "$(dirname "$0")/.." && pwd)
BASE_TREE=${BASE_TREE:-$PDL_TREE/../MegaFold}
cd "$PDL_TREE"
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1; conda activate base; source ~/venvs/megafold/bin/activate
export TYPECHECK=False CXX=g++ CC=gcc
TAG=${SLURM_JOB_ID:-local}_$$_$(date +%s)
mkdir -p results/kernel_variants results/nsys

echo "##### phase 1: kernel-level variants"
declare -A KV=(
  [k0_baseline]="MEGAFOLD_PDL=0"
  [k1_pdl_all]="MEGAFOLD_PDL=1 MEGAFOLD_PDL_KERNELS=all"
  [k2_pdl_all_nolaunch]="MEGAFOLD_PDL=1 MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_LAUNCH=0"
  [k3_pdl_all_implicit]="MEGAFOLD_PDL=1 MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_TRIGGER=implicit"
  [k4_pdl_tiled]="MEGAFOLD_PDL=1 MEGAFOLD_PDL_KERNELS=tiled"
)
for v in k0_baseline k1_pdl_all k2_pdl_all_nolaunch k3_pdl_all_implicit k4_pdl_tiled; do
  env ${KV[$v]} TRITON_CACHE_DIR=/tmp/tc_${v}_$TAG python benchmarks/pdl_kernel_check.py --out results/kernel_variants/$v.pt | tail -1
done
for v in k1_pdl_all k2_pdl_all_nolaunch k3_pdl_all_implicit k4_pdl_tiled; do
  python benchmarks/pdl_kernel_check.py --compare results/kernel_variants/k0_baseline.pt results/kernel_variants/$v.pt > results/kernel_variants/cmp_$v.json
  python3 -c "import json;r=json.load(open('results/kernel_variants/cmp_$v.json'));print('$v', 'median_ms', round(r['B']['fwd_bwd_ms_median'],3), 'speedup', round(r['speedup_B_over_A'],4), 'y_bitwise', r['bitwise_equal']['y'])"
done
env ${KV[k4_pdl_tiled]} TRITON_CACHE_DIR=/tmp/tc_k4_pdl_tiled_$TAG nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o results/nsys/kernels_pdl_tiled \
  python benchmarks/pdl_kernel_check.py --iters 5 --out /tmp/kc_nsys_tiled_$TAG.pt >/dev/null
nsys stats --force-export=true --report cuda_gpu_trace --format csv --output results/nsys/kernels_pdl_tiled results/nsys/kernels_pdl_tiled.nsys-rep >/dev/null
nsys stats --report cuda_gpu_kern_sum --format csv --output results/nsys/kernels_pdl_tiled results/nsys/kernels_pdl_tiled.nsys-rep >/dev/null
python benchmarks/analyze_nsys.py results/nsys/kernels_pdl_tiled_cuda_gpu_trace.csv --out results/nsys/kernels_pdl_tiled_gaps.json >/dev/null

echo "##### phase 2: end-to-end training step, interleaved (cold caches, TORCH_COMPILE_DEBUG=1)"
for rep in 1 2; do
  (cd "$BASE_TREE" && SLURM_SUBMIT_DIR="$BASE_TREE" OUT_TAG=e2e_baseline_r$rep bash scripts/bench_train_step.sh baseline --warmup 2 --iters 10 > results/logs/e2e_baseline_r$rep.log 2>&1)
  echo "rep $rep baseline exit=$?"; cp "$BASE_TREE/results/e2e_baseline_r$rep.json" results/ 2>/dev/null
  SLURM_SUBMIT_DIR="$PDL_TREE" OUT_TAG=e2e_pdl_all_r$rep MEGAFOLD_PDL_KERNELS=all bash scripts/bench_train_step.sh pdl --warmup 2 --iters 10 > results/logs/e2e_pdl_all_r$rep.log 2>&1
  echo "rep $rep pdl_all exit=$?"
  SLURM_SUBMIT_DIR="$PDL_TREE" OUT_TAG=e2e_pdl_tiled_r$rep MEGAFOLD_PDL_KERNELS=tiled bash scripts/bench_train_step.sh pdl --warmup 2 --iters 10 > results/logs/e2e_pdl_tiled_r$rep.log 2>&1
  echo "rep $rep pdl_tiled exit=$?"
done

echo "##### phase 3: nsys of one training step, PDL variants"
SLURM_SUBMIT_DIR="$PDL_TREE" OUT_TAG=pdl_all MEGAFOLD_PDL_KERNELS=all bash scripts/nsys_train_step.sh pdl > results/logs/nsys_pdl_all.log 2>&1; echo "nsys pdl_all exit=$?"
SLURM_SUBMIT_DIR="$PDL_TREE" OUT_TAG=pdl_tiled MEGAFOLD_PDL_KERNELS=tiled bash scripts/nsys_train_step.sh pdl > results/logs/nsys_pdl_tiled.log 2>&1; echo "nsys pdl_tiled exit=$?"
echo "##### done"
