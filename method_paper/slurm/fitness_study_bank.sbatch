#!/bin/bash
#SBATCH --job-name=mp_fs_bank
#SBATCH --output=results/logs/mp_fs_bank_%j.log
#SBATCH --error=results/logs/mp_fs_bank_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=03:00:00

# Fitness-vs-accuracy study, step 1: NEAT + equal-budget random search with
# the teacher-only fitness, 40 genomes picked across the fitness range,
# rescored on one shared probe set. Writes
# results/method_paper/cifar100/fitness_study/bank/{bank.csv,patterns.pt,bank.png}.
# Then submit the student runs with a dependency on this job:
#   B=$(sbatch --parsable method_paper/slurm/fitness_study_bank.sbatch)
#   sbatch --dependency=afterok:$B method_paper/slurm/fitness_study_train.sbatch

set -e

# Run from wherever sbatch was invoked (the repo root), with the
# machine-specific env activation in cluster_env.sh -- see cluster_env.sh.example.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f cluster_env.sh ]; then
  echo "cluster_env.sh not found in $SLURM_SUBMIT_DIR -- submit from the repo root and create it from cluster_env.sh.example" >&2
  exit 1
fi
source cluster_env.sh
mkdir -p results/logs

python method_paper/scripts/fitness_study.py bank --config method_paper/configs/cifar100.yaml
