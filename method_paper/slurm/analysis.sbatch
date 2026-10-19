#!/bin/bash
#SBATCH --job-name=mp_analysis
#SBATCH --output=results/logs/mp_analysis_%A_%a.log
#SBATCH --error=results/logs/mp_analysis_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=16:00:00
#SBATCH --array=0-20

# Analysis-paper controls (configs/cifar100.yaml's analysis section):
#   0-17  generality: kd_random_cppn / kd_evolved_cppn / kd_online_teacher on
#         wrn_40_2 -> wrn_16_2 and vgg13 -> mobilenetv2, 3 seeds
#         (kd_evolved_cppn uses 5 views per batch, ~4-5 h)
#   18-20 noise masks selected by the method's fitness, resnet56 -> resnet20
# Check the job list with:
#   python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage analysis --list

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

python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage analysis --index ${SLURM_ARRAY_TASK_ID}
