#!/bin/bash
# Nsight Systems trace of ONE steady-state training step (after 2 warm-up steps), baseline or PDL.
#   sbatch scripts/nsys_train_step.sh baseline
#   sbatch scripts/nsys_train_step.sh pdl
#SBATCH --account=bcjw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=mf_nsys
#SBATCH --output=results/logs/%x_%j.out

set -euo pipefail
MODE=${1:-baseline}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
conda activate base
source ~/venvs/megafold/bin/activate

export TYPECHECK=False
export CXX=g++ CC=gcc   # PrgEnv sets CXX=CC (Cray wrapper), which breaks Inductor's C++ codegen
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=/tmp/compile_debug_nsys_${MODE}_${SLURM_JOB_ID:-local}_$$_$(date +%s)
export TRITON_CACHE_DIR=/tmp/triton_cache_nsys_${MODE}_${SLURM_JOB_ID:-local}_$$_$(date +%s)
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_nsys_${MODE}_${SLURM_JOB_ID:-local}_$$_$(date +%s)
if [ "$MODE" = "pdl" ]; then export MEGAFOLD_PDL=1; else unset MEGAFOLD_PDL; fi

mkdir -p results/nsys
OUT=results/nsys/${MODE}
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true -o "$OUT" \
  python benchmarks/train_step_bench.py --mode "$MODE" --warmup 2 --iters 1 --nvtx-step 2 \
    --out "results/nsys/${MODE}_run.json"

nsys stats --force-export=true --report cuda_gpu_trace --format csv --output "$OUT" "$OUT.nsys-rep" >/dev/null
nsys stats --report cuda_api_sum --format csv --output "$OUT" "$OUT.nsys-rep" >/dev/null
python benchmarks/analyze_nsys.py "${OUT}_cuda_gpu_trace.csv" --out "results/nsys/${MODE}_gaps.json"
