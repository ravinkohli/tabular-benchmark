#!/bin/bash
#SBATCH --error=/work/dlclarge2/rkohli-results_tab-bench/sweep_logs/%j_cpu.err
#SBATCH --output=/work/dlclarge2/rkohli-results_tab-bench/sweep_logs/%j_cpu.out
#SBATCH --job-name=cpu_tabular_benchmark
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=400
#SBATCH --mem-per-cpu=2000

source ~/anaconda3/bin/activate gael_eval-env
wandb agent $wandb_id/$project/$sweep_id