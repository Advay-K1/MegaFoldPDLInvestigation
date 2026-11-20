#!/bin/bash
# Report 2 experiments (what would make PDL impactful), one GPU allocation:
#   srun --jobid=<holder> --overlap -n1 bash scripts/run_report2.sh
set -uo pipefail
PDL_TREE=$(cd "$(dirname "$0")/.." && pwd); BASE_TREE=${BASE_TREE:-$PDL_TREE/../MegaFold}
cd "$PDL_TREE"
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1; conda activate base; source ~/venvs/megafold/bin/activate
export TYPECHECK=False CXX=g++ CC=gcc
mkdir -p results/report2 results/logs
TAG=${SLURM_JOB_ID:-local}_$$_$(date +%s)

echo "##### R2-1: PDL regime sweep (dependent Triton chain; eager vs CUDA graph)"
TRITON_CACHE_DIR=/tmp/tc_sweep_$TAG python benchmarks/pdl_regime_sweep.py --out results/report2/regime_sweep.json > results/logs/r2_sweep.log 2>&1
echo "sweep exit=$?"

echo "##### R2-2: graph-break census (torch._dynamo.explain on one forward)"
SLURM_SUBMIT_DIR="$PDL_TREE" OUT_TAG=report2/graph_breaks bash scripts/bench_train_step.sh baseline --explain > results/logs/r2_explain.log 2>&1 \
  || MEGAFOLD_PDL=0 python benchmarks/train_step_bench.py --mode baseline --explain --out results/report2/graph_breaks.json >> results/logs/r2_explain.log 2>&1
echo "explain exit=$?"

echo "##### R2-3: CUDA graphs (torch.compile mode=reduce-overhead), interleaved baseline / PDL"
E_PDL="MEGAFOLD_PDL_KERNELS=all MEGAFOLD_PDL_TRIGGER=implicit"
(cd "$BASE_TREE" && SLURM_SUBMIT_DIR="$BASE_TREE" OUT_TAG=cg_smoke bash scripts/bench_train_step.sh baseline --warmup 1 --iters 2 --compile-mode reduce-overhead > "$PDL_TREE/results/logs/r2_cg_smoke.log" 2>&1)
rc=$?; echo "cg smoke exit=$rc"; cp "$BASE_TREE/results/cg_smoke.json" results/report2/ 2>/dev/null
if [ $rc -eq 0 ]; then
  for rep in 1 2; do
    (cd "$BASE_TREE" && SLURM_SUBMIT_DIR="$BASE_TREE" OUT_TAG=cg_baseline_r$rep bash scripts/bench_train_step.sh baseline --warmup 3 --iters 10 --compile-mode reduce-overhead > "$PDL_TREE/results/logs/r2_cg_baseline_r$rep.log" 2>&1)
    echo "cg baseline r$rep exit=$?"; cp "$BASE_TREE/results/cg_baseline_r$rep.json" results/report2/ 2>/dev/null
    env $E_PDL SLURM_SUBMIT_DIR="$PDL_TREE" OUT_TAG=report2/cg_pdl_r$rep bash scripts/bench_train_step.sh pdl --warmup 3 --iters 10 --compile-mode reduce-overhead > results/logs/r2_cg_pdl_r$rep.log 2>&1
    echo "cg pdl r$rep exit=$?"
  done
fi
echo "##### report2 done"
