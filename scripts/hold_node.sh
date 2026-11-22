#!/bin/bash
#SBATCH --account=bcjw-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=mf_hold
#SBATCH --output=results/logs/%x_%j.out
# Holds one GH200 so benchmark steps can be run inside it with `srun --jobid=$ID --overlap ...`.
sleep 7190
