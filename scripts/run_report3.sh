#!/bin/bash
# Report 3 (cross-language PDL) GPU experiments. Run inside a holder job:
#   srun --jobid=<holder> --overlap -n1 bash scripts/run_report3.sh
set -eo pipefail
cd /u/akadam2/MegaFold-pdl
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1; conda activate base; source ~/venvs/megafold/bin/activate
export CXX=g++ CC=gcc CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9 TORCH_CUDA_ARCH_LIST=9.0a
export TRITON_CACHE_DIR=/tmp/triton_r3_$SLURM_JOB_ID
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
LD_PRELOAD=$PWD/benchmarks/pdl_xlang/libpdl_shim.so python benchmarks/pdl_cross_language.py \
    --out results/report3/cross_language.json "$@"
