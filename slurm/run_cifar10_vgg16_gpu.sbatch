#!/bin/bash
#SBATCH --job-name=cppn_vgg16
#SBATCH --output=results/logs/cppn_vgg16_%A_%a.log
#SBATCH --error=results/logs/cppn_vgg16_%A_%a.err
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

# Different-architecture ablation (VGG16 instead of ResNet18) -- see
# configs/datasets/cifar10_vgg16.yaml for rationale. Starting at 3 seeds
# (array=0-2), same as every other architecture/dataset pairing's first
# real run in this project (FashionMNIST/LeNet, CIFAR-10/ResNet18) --
# hyperparameters are an untuned copy of the ResNet18 config, so check the
# teacher clears the from-scratch student baseline before scaling to more
# seeds or trusting any KD-mode comparison.

set -e

export MAMBA_ROOT_PREFIX=/project/ahoover/an57/micromamba
eval "$(/project/ahoover/an57/bin/micromamba shell hook -s bash)"
micromamba activate /project/ahoover/an57/envs/torch-cuda-mm

cd /project/ahoover/an57/CPPN-Assisted-Feature-Distillation
mkdir -p results/logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python scripts/run_baseline_sweep.py --config configs/datasets/cifar10_vgg16.yaml --modes all --seeds ${SLURM_ARRAY_TASK_ID} --use-ensemble
