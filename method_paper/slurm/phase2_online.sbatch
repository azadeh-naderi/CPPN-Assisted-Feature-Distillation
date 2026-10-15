#!/bin/bash
#SBATCH --job-name=mp_p2_online
#SBATCH --output=results/logs/mp_p2_online_%A_%a.log
#SBATCH --error=results/logs/mp_p2_online_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=12:00:00
#SBATCH --array=0-8

# Phase 2: student-aware online view evolution (configs/cifar100.yaml's
# phase2 section) -- 3 fitness modes x 1 pair x 3 seeds. Needs the Phase 0
# teacher checkpoints. One view per batch means ~2x a kd epoch (Phase 0:
# 0.25 min/epoch for resnet56->resnet20) plus 65 NEAT generations of
# 150 genomes spread over the run -- the pilot took ~1.9 h. Check the job list with:
#   python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage phase2 --list
#
# Tuning runs: pass extra launch.py arguments through LAUNCH_ARGS. Tune on a
# validation split, never on test accuracy, and tag every variant, e.g.
#   LAUNCH_ARGS="--set val_size=5000 --set online.gate.tau_low=0.5 --tag gate05" \
#       sbatch method_paper/slurm/phase2_online.sbatch
# (sbatch exports the submitting shell's environment by default.)

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

python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage phase2 --index ${SLURM_ARRAY_TASK_ID} ${LAUNCH_ARGS:-}
