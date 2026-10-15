#!/bin/bash
#SBATCH --job-name=mp_p3_main
#SBATCH --output=results/logs/mp_p3_main_%A_%a.log
#SBATCH --error=results/logs/mp_p3_main_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=12:00:00
#SBATCH --array=0-35

# Phase 3 main table: every benchmark pair (6) x (phase3.main_baselines +
# phase3.method) x 3 seeds = 36 jobs; runs that already exist (e.g. the
# development pair from Phases 1-2) exit immediately. Needs every pair's
# teacher, so run the teachers stage first if resnet110 isn't trained yet:
#   T=$(sbatch --parsable method_paper/slurm/phase0_teachers.sbatch)
#   sbatch --dependency=afterok:${T%%;*} method_paper/slurm/phase0_students.sbatch
#   sbatch --dependency=afterok:${T%%;*} method_paper/slurm/phase3_main.sbatch
# Set phase3.method / method_set in configs/cifar100.yaml to Gate 2's
# winner first. Check the job list with:
#   python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage phase3_main --list

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

python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage phase3_main --index ${SLURM_ARRAY_TASK_ID} ${LAUNCH_ARGS:-}
