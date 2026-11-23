#!/bin/bash
# Report 4: the training step with MegaFold's kernels as opaque custom ops (MEGAFOLD_OPAQUE_KERNELS=1), so
# torch.compile compiles *through* the activation-checkpointed layers; then PDL on top.
#   srun --jobid=<holder> --overlap -n1 bash scripts/run_report4_e2e.sh [variants...]
set -uo pipefail
T=/u/akadam2/MegaFold-pdl; cd $T
VARIANTS=${*:-"off opaque opaque_pdl"}
for rep in ${REPS:-1 2}; do
  for v in $VARIANTS; do
    case $v in
      off)        E=""; M=baseline; X="" ;;
      opaque)     E="MEGAFOLD_OPAQUE_KERNELS=1"; M=baseline; X="--ac-compile inner" ;;
      opaque_pdl) E="MEGAFOLD_OPAQUE_KERNELS=1 MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_TRIGGER=implicit"; M=pdl; X="--ac-compile inner" ;;
      opaque_pdl_early) E="MEGAFOLD_OPAQUE_KERNELS=1 MEGAFOLD_PDL_KERNELS=tiled MEGAFOLD_PDL_TRIGGER=early"; M=pdl; X="--ac-compile inner" ;;
    esac
    env $E SLURM_SUBMIT_DIR=$T OUT_TAG=report4/e2e_${v}_r$rep bash scripts/bench_train_step.sh $M --warmup 2 --iters ${ITERS:-8} $X \
        > results/logs/r4_e2e_${v}_r$rep.log 2>&1; echo "rep $rep $v exit=$?"
  done
done
echo "##### done"
# numerical check with dropout effectively off: losses of opaque+compiled layers vs baseline
if [ "${CHECK:-1}" = "1" ]; then
  for v in off opaque; do
    case $v in off) E=""; X="" ;; opaque) E="MEGAFOLD_OPAQUE_KERNELS=1"; X="--ac-compile inner" ;; esac
    env $E MEGAFOLD_NO_DROPOUT=1 SLURM_SUBMIT_DIR=$T OUT_TAG=report4/nodrop_${v} bash scripts/bench_train_step.sh baseline \
        --warmup 1 --iters 2 $X > results/logs/r4_nodrop_${v}.log 2>&1; echo "nodrop $v exit=$?"
  done
fi
echo "##### all done"
