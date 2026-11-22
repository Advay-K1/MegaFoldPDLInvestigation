#!/bin/bash
# Report 3 end-to-end: does cross-language PDL (MegaFold Triton kernels + *every* eager ATen kernel made a PDL
# secondary by the LD_PRELOAD shim, implicit triggers only) change the real training step?
#   srun --jobid=<holder> --overlap -n1 bash scripts/run_report3_e2e.sh
set -uo pipefail
T=/u/akadam2/MegaFold-pdl; cd $T
SHIM=$T/benchmarks/pdl_xlang/libpdl_shim.so
for rep in 1 2; do
  SLURM_SUBMIT_DIR=$T OUT_TAG=report3/e2e_pdl_off_r$rep bash scripts/bench_train_step.sh baseline --warmup 2 --iters 8 \
      > results/logs/r3_e2e_off_r$rep.log 2>&1; echo "rep $rep off exit=$?"
  LD_PRELOAD=$SHIM MEGAFOLD_PDL_SHIM_ATEN=1 MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_TRIGGER=implicit SLURM_SUBMIT_DIR=$T \
      OUT_TAG=report3/e2e_xlang_r$rep bash scripts/bench_train_step.sh pdl --inductor-pdl 0 --warmup 2 --iters 8 \
      > results/logs/r3_e2e_xlang_r$rep.log 2>&1; echo "rep $rep xlang exit=$?"
done
echo "##### done"
