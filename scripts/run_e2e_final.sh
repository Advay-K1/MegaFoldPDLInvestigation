#!/bin/bash
# Final end-to-end measurements (after the Inductor mutation-analysis fix), one GPU allocation:
#   srun --jobid=<holder> --overlap -n1 bash scripts/run_e2e_final.sh
# Variants are interleaved within each repetition; every run compiles from cold caches with
# TORCH_COMPILE_DEBUG=1, and every step index uses the same RNG stream in every variant.
set -uo pipefail
PDL_TREE=$(cd "$(dirname "$0")/.." && pwd); BASE_TREE=${BASE_TREE:-$PDL_TREE/../MegaFold}
cd "$PDL_TREE"; unset MEGAFOLD_NAN_TRACE
for rep in 1 2; do
  (cd "$BASE_TREE" && SLURM_SUBMIT_DIR="$BASE_TREE" OUT_TAG=final_baseline_r$rep bash scripts/bench_train_step.sh baseline --warmup 2 --iters 10 > "$PDL_TREE/results/logs/final_baseline_r$rep.log" 2>&1)
  echo "rep $rep baseline exit=$?"; cp "$BASE_TREE/results/final_baseline_r$rep.json" results/
  for v in all implicit tiled; do
    case $v in
      all)      E="MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_TRIGGER=early" ;;
      implicit) E="MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_TRIGGER=implicit" ;;
      tiled)    E="MEGAFOLD_PDL_KERNELS=tiled MEGAFOLD_PDL_TRIGGER=early" ;;
    esac
    env $E SLURM_SUBMIT_DIR="$PDL_TREE" OUT_TAG=final_pdl_${v}_r$rep bash scripts/bench_train_step.sh pdl --warmup 2 --iters 10 > results/logs/final_pdl_${v}_r$rep.log 2>&1
    echo "rep $rep pdl_$v exit=$?"
  done
done
echo "##### e2e done"
for v in all implicit; do
  case $v in all) E="MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_TRIGGER=early" ;; implicit) E="MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_TRIGGER=implicit" ;; esac
  env $E SLURM_SUBMIT_DIR="$PDL_TREE" OUT_TAG=pdl_$v bash scripts/nsys_train_step.sh pdl > results/logs/nsys_pdl_$v.log 2>&1; echo "nsys pdl_$v exit=$?"
done
echo "##### all done"
