#!/bin/bash
#SBATCH --job-name=mp_p3_abl
#SBATCH --output=results/logs/mp_p3_abl_%A_%a.log
#SBATCH --error=results/logs/mp_p3_abl_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=12:00:00
#SBATCH --array=0-20

# Phase 3 ablations: phase3.method on phase3.ablation_pair with each of the
# 7 override sets in phase3.ablations (re-evolution every 10/40 epochs,
# averaging all views, no jitter, random search, unscored random views,
# noise masks) x 3 seeds = 21 jobs, each in a tagged run directory. The
# "average" ablation uses all 5 views on every batch (~6x a kd epoch, ~5 h).
# Check the job list with:
#   python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage phase3_ablations --list

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

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage phase3_ablations --index ${SLURM_ARRAY_TASK_ID} ${LAUNCH_ARGS:-}
