#!/bin/bash
#SBATCH --job-name=mp_p1_base
#SBATCH --output=results/logs/mp_p1_base_%A_%a.log
#SBATCH --error=results/logs/mp_p1_base_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=48:00:00
#SBATCH --array=0-17

# Phase 1: corrected baselines (configs/cifar100.yaml's phase1 section) --
# 6 modes x 1 pair x 3 seeds. Needs the Phase 0 teacher checkpoints; submit
# with --dependency=afterok:<teacher job> if they're still training. Longer
# limit than Phase 0 because view_sampling=all with a 5-view ensemble adds
# five student forward/backward passes per batch (roughly 6x a kd epoch).
# Check against the Phase 0 kd students' epoch time before submitting.
# Check the job list with:
#   python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage phase1 --list

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

python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage phase1 --index ${SLURM_ARRAY_TASK_ID}
