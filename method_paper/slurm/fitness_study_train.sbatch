#!/bin/bash
#SBATCH --job-name=mp_fs_train
#SBATCH --output=results/logs/mp_fs_train_%A_%a.log
#SBATCH --error=results/logs/mp_fs_train_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=02:00:00
#SBATCH --array=0-82

# Fitness-vs-accuracy study, step 2: 60-epoch resnet20 students -- indices
# 0-2 = kd reference (3 seeds), 3-82 = one bank view each (40 genomes x 2
# seeds), ~30 min each. Needs the bank from fitness_study_bank.sbatch.
# Afterwards:
#   python method_paper/scripts/fitness_study.py analyze --config method_paper/configs/cifar100.yaml
# Check the job list with:
#   python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage fitness_study --list

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

python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage fitness_study --index ${SLURM_ARRAY_TASK_ID}
