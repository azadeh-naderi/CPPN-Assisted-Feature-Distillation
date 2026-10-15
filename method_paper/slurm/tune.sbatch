#!/bin/bash
#SBATCH --job-name=mp_tune
#SBATCH --output=results/logs/mp_tune_%A_%a.log
#SBATCH --error=results/logs/mp_tune_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=12:00:00
#SBATCH --array=0-5

# Hyperparameter tuning on a validation split (configs/cifar100.yaml's tune
# section): index 0-2 = kd reference, 3-5 = the method, seeds 0-2. Results
# get __val5000[__<tag>] run directories and are compared by val_final in
# summarize.py -- never by test accuracy. Pass settings through LAUNCH_ARGS
# (sbatch exports the submitting shell's environment):
#
#   # once: the kd reference on the validation split
#   LAUNCH_ARGS="--set val_size=5000" sbatch --array=0-2 method_paper/slurm/tune.sbatch
#   # per variant: one tag each
#   LAUNCH_ARGS="--set val_size=5000 --set online.gate.tau_low=0.5 --tag gate05" \
#       sbatch --array=3-5 method_paper/slurm/tune.sbatch

set -e

case " ${LAUNCH_ARGS:-} " in
  *"val_size="*) ;;
  *) echo "tune.sbatch: LAUNCH_ARGS must include --set val_size=<N> (tune on validation, not test)" >&2; exit 1 ;;
esac

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

python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage tune --index ${SLURM_ARRAY_TASK_ID} ${LAUNCH_ARGS}
