#!/bin/bash
#SBATCH --job-name=cppn_only
#SBATCH --output=results/logs/cppn_only_%A_%a.log
#SBATCH --error=results/logs/cppn_only_%A_%a.err
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

# CPPN-only loss ablation -- see configs/datasets/cifar10_resnet18_cppn_only.yaml
# for rationale. Starting at 3 seeds, matching how every other new
# config/loss-composition experiment was first validated in this project.

set -e

export MAMBA_ROOT_PREFIX=/project/ahoover/an57/micromamba
eval "$(/project/ahoover/an57/bin/micromamba shell hook -s bash)"
micromamba activate /project/ahoover/an57/envs/torch-cuda-mm

cd /project/ahoover/an57/CPPN-Assisted-Feature-Distillation
mkdir -p results/logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python scripts/run_baseline_sweep.py --config configs/datasets/cifar10_resnet18_cppn_only.yaml --modes all --seeds ${SLURM_ARRAY_TASK_ID} --use-ensemble
