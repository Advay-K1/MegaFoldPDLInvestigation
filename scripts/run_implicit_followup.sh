#!/bin/bash
# Follow-up: end-to-end runs of the best kernel-level PDL config (all kernels, implicit trigger),
# each paired with an adjacent fresh baseline run.
set -uo pipefail
PDL_TREE=$(cd "$(dirname "$0")/.." && pwd); BASE_TREE=${BASE_TREE:-$PDL_TREE/../MegaFold}
cd "$PDL_TREE"
for rep in 1 2; do
  b=$((rep + 2))
  (cd "$BASE_TREE" && SLURM_SUBMIT_DIR="$BASE_TREE" OUT_TAG=e2e_baseline_r$b bash scripts/bench_train_step.sh baseline --warmup 2 --iters 10 > results/logs/e2e_baseline_r$b.log 2>&1)
  echo "baseline r$b exit=$?"; cp "$BASE_TREE/results/e2e_baseline_r$b.json" results/ 2>/dev/null
  SLURM_SUBMIT_DIR="$PDL_TREE" OUT_TAG=e2e_pdl_implicit_r$rep MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_TRIGGER=implicit \
    bash scripts/bench_train_step.sh pdl --warmup 2 --iters 10 > results/logs/e2e_pdl_implicit_r$rep.log 2>&1
  echo "pdl_implicit r$rep exit=$?"
done
echo "##### followup done"
