#!/bin/bash
#SBATCH --job-name=mp_eval
#SBATCH --output=results/logs/mp_eval_%j.log
#SBATCH --error=results/logs/mp_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=08:00:00

# Phase 3: calibration, teacher agreement and CIFAR-100-C robustness for every
# finished student run (writes extra_metrics.json next to each summary.json;
# runs already evaluated are skipped). No training. Download CIFAR-100-C to
# data/CIFAR-100-C on a login node first -- see method_paper/README.md.
# Extra arguments (e.g. --runs 'resnet56__resnet20__*') via EVAL_ARGS.

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

python method_paper/scripts/evaluate_extra.py --config method_paper/configs/cifar100.yaml ${EVAL_ARGS:-}
