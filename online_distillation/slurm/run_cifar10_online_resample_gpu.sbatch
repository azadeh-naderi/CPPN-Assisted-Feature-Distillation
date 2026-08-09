#!/bin/bash
#SBATCH --job-name=cppn_online_rs
#SBATCH --output=results/logs/cppn_online_rs_%A_%a.log
#SBATCH --error=results/logs/cppn_online_rs_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=08:00:00
#SBATCH --array=0-2

# Attempt 2: per-epoch CPPN pattern resampling -- see
# online_distillation/configs/cifar10_resnet18_resample.yaml and
# EXPERIMENT_LOG.md. Starting at 3 seeds, matching attempt 1.

set -e

export MAMBA_ROOT_PREFIX=/project/ahoover/an57/micromamba
eval "$(/project/ahoover/an57/bin/micromamba shell hook -s bash)"
micromamba activate /project/ahoover/an57/envs/torch-cuda-mm

cd /project/ahoover/an57/CPPN-Assisted-Feature-Distillation
mkdir -p results/logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python online_distillation/scripts/run_online_sweep.py --config online_distillation/configs/cifar10_resnet18_resample.yaml --modes all --seeds ${SLURM_ARRAY_TASK_ID}
