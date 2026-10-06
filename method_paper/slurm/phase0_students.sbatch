#!/bin/bash
#SBATCH --job-name=mp_p0_stud
#SBATCH --output=results/logs/mp_p0_stud_%A_%a.log
#SBATCH --error=results/logs/mp_p0_stud_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=12:00:00
#SBATCH --array=0-23

# Phase 0, stage 2: student_only (4 student archs x 3 seeds) + kd (4 pairs x
# 3 seeds) = 24 jobs. The kd jobs need the teachers from
# phase0_teachers.sbatch, so submit this with a dependency on that job:
#   T=$(sbatch --parsable method_paper/slurm/phase0_teachers.sbatch)
#   sbatch --dependency=afterok:${T%%;*} method_paper/slurm/phase0_students.sbatch
# Check the job list with:
#   python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage students --list

set -e

export MAMBA_ROOT_PREFIX=/project/ahoover/an57/micromamba
eval "$(/project/ahoover/an57/bin/micromamba shell hook -s bash)"
micromamba activate /project/ahoover/an57/envs/torch-cuda-mm

cd /project/ahoover/an57/CPPN-Assisted-Feature-Distillation
mkdir -p results/logs

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python method_paper/scripts/launch.py --config method_paper/configs/cifar100.yaml --stage students --index ${SLURM_ARRAY_TASK_ID}
