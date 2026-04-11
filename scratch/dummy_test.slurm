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

# Keep large HF / torch downloads on SCRATCH (not $HOME/.cache).
export HF_HOME="${SCRATCH}/huggingface"
export TORCH_HOME="${SCRATCH}/torch"
export TMPDIR="${SCRATCH}/tmp/${SLURM_JOB_ID:-job}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-none}"
echo "HF_HOME: ${HF_HOME}"
echo "TORCH_HOME: ${TORCH_HOME}"

module load python/3.12
source ~/venvs/comp767/bin/activate
export PYTHONUNBUFFERED=1
python -u dummy_scoring_new_dataset.py "$@"
nvidia-smi || true

echo "Dummy job completed."
