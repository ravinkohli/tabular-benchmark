#!/bin/bash
#SBATCH --job-name=tabular_benchmark_gpu
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --error=/work/dlclarge2/rkohli-results_tab-bench/sweep_logs/%j_gpu.err
#SBATCH --output=/work/dlclarge2/rkohli-results_tab-bench/sweep_logs/%j_gpu.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=600

source ~/anaconda3/bin/activate gael_eval-env
wandb agent $wandb_id/$project/$sweep_id