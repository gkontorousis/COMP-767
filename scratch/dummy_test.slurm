#!/bin/bash
#SBATCH --job-name=dummy-test
#SBATCH --account=def-sreddy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192M
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-none}"

module load python/3.12
source ~/venvs/comp767/bin/activate
export PYTHONUNBUFFERED=1
python -u dummy_scoring_new_dataset.py
nvidia-smi || true

echo "Dummy job completed."
