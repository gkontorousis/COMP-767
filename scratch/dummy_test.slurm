#!/bin/bash
#SBATCH --job-name=dummy-test
#SBATCH --account=def-sreddy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192M
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-none}"

module load python/3.10
python dummy_scoring_new_dataset.py
nvidia-smi || true

echo "Dummy job completed."
