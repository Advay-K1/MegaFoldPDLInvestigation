#!/bin/bash
# Benchmark one MegaFold training step (bs=1, seq_len=384, torch.compile, TORCH_COMPILE_DEBUG=1)
# on a single GH200 of NCSA DeltaAI.
#   sbatch scripts/bench_train_step.sh baseline
#   sbatch scripts/bench_train_step.sh pdl
#SBATCH --account=bcjw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=mf_step
#SBATCH --output=results/logs/%x_%j.out

set -euo pipefail
MODE=${1:-baseline}
shift || true
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
conda activate base
source ~/venvs/megafold/bin/activate

export TYPECHECK=False          # required by trainer.py when torch.compile is used
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=$PWD/results/compile_debug/$MODE
export TRITON_CACHE_DIR=/tmp/triton_cache_${MODE}_${SLURM_JOB_ID:-local}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_${MODE}_${SLURM_JOB_ID:-local}
if [ "$MODE" = "pdl" ]; then export MEGAFOLD_PDL=1; else unset MEGAFOLD_PDL; fi

nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
python benchmarks/train_step_bench.py --mode "$MODE" --out "results/${MODE}.json" "$@"
