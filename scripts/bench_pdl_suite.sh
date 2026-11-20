#!/bin/bash
# PDL side of the investigation, one allocation:
#  1) kernel-level correctness/speed check of MegaFold's Triton kernels, PDL off vs on
#  2) full training-step benchmark with PDL (same settings as the baseline job)
#SBATCH --account=bcjw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=mf_pdl
#SBATCH --output=results/logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
conda activate base
source ~/venvs/megafold/bin/activate
export TYPECHECK=False
export CXX=g++ CC=gcc   # PrgEnv sets CXX=CC (Cray wrapper), which breaks Inductor's C++ codegen
TAG=${SLURM_JOB_ID:-local}_$$_$(date +%s)
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv

if [ "${SKIP_KERNELS:-0}" != 1 ]; then
echo "=== kernel check: baseline (PDL off), run A and run B (noise floor: atomics make bwd order-dependent)"
TRITON_CACHE_DIR=/tmp/tc_kbase_$TAG  env -u MEGAFOLD_PDL python benchmarks/pdl_kernel_check.py --out results/kernels_baseline.pt
TRITON_CACHE_DIR=/tmp/tc_kbase_$TAG  env -u MEGAFOLD_PDL python benchmarks/pdl_kernel_check.py --out results/kernels_baseline_B.pt
echo "=== kernel check: PDL on"
TRITON_CACHE_DIR=/tmp/tc_kpdl_$TAG MEGAFOLD_PDL=1 python benchmarks/pdl_kernel_check.py --out results/kernels_pdl.pt
echo "=== compare baseline A vs baseline B (noise floor)"
python benchmarks/pdl_kernel_check.py --compare results/kernels_baseline.pt results/kernels_baseline_B.pt > results/kernels_compare_noise.json || true
cat results/kernels_compare_noise.json
echo "=== compare baseline A vs PDL"
python benchmarks/pdl_kernel_check.py --compare results/kernels_baseline.pt results/kernels_pdl.pt > results/kernels_compare.json || true
cat results/kernels_compare.json

echo "=== PTX check: kernels containing griddepcontrol (expect 0 for baseline, all MegaFold kernels for PDL)"
for m in kbase kpdl; do
  files=$(find /tmp/tc_${m}_$TAG -name '*.ptx')
  tot=$(echo "$files" | grep -c . || true)
  hits=$(grep -l "griddepcontrol" $files 2>/dev/null || true)
  echo "$m: $(echo -n "$hits" | grep -c . || true) / $tot PTX files contain griddepcontrol"
  echo "$hits" | grep . | xargs -r -n1 basename | sed 's/\.ptx$//' | sort | uniq -c || true
done | tee results/ptx_griddepcontrol.txt

echo "=== nsys of kernel microbenchmark (both modes)"
mkdir -p results/nsys
for m in baseline pdl; do
  if [ $m = pdl ]; then E="MEGAFOLD_PDL=1"; C=/tmp/tc_kpdl_$TAG; else E="MEGAFOLD_PDL=0"; C=/tmp/tc_kbase_$TAG; fi
  env $E TRITON_CACHE_DIR=$C nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
     -o results/nsys/kernels_$m python benchmarks/pdl_kernel_check.py --iters 5 --out /tmp/kc_nsys_${m}_$TAG.pt >/dev/null
  nsys stats --force-export=true --report cuda_gpu_trace --format csv --output results/nsys/kernels_$m results/nsys/kernels_$m.nsys-rep >/dev/null
  nsys stats --report cuda_gpu_kern_sum --format csv --output results/nsys/kernels_$m results/nsys/kernels_$m.nsys-rep >/dev/null
  python benchmarks/analyze_nsys.py results/nsys/kernels_${m}_cuda_gpu_trace.csv --out results/nsys/kernels_${m}_gaps.json > /dev/null
done
fi

echo "=== training step: PDL"
bash scripts/bench_train_step.sh pdl "$@"
