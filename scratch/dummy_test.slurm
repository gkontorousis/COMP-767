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

# Narval compute nodes are typically offline. Download the model once on a
# login node, then point MODEL_PATH at that local directory.
#
# Example (login node):
#   huggingface-cli download Qwen/Qwen2.5-7B --local-dir "${SCRATCH}/models/Qwen2.5-7B"
MODEL_PATH="${MODEL_PATH:-}"

# Optional: only used if you set it. Useful if you load by repo id from a cache
# instead of a dedicated --model-path directory.
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"

# Where the Python script writes its redirected stdout/stderr.
OUTPUT_FILE="${OUTPUT_FILE:-${SCRATCH}/dummy_scoring_new_dataset_output.txt}"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-none}"
echo "HF_HOME: ${HF_HOME}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "MODEL_PATH: ${MODEL_PATH:-<unset>}"
echo "MODEL_CACHE_DIR: ${MODEL_CACHE_DIR:-<unset>}"
echo "OUTPUT_FILE: ${OUTPUT_FILE}"

module load python/3.12
source ~/venvs/comp767/bin/activate
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"
echo "PWD: $(pwd)"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is not set."
  echo "Download the model on a login node, then re-submit with:"
  echo "  sbatch --export=ALL,MODEL_PATH=\"${SCRATCH}/models/Qwen2.5-7B\" scratch/dummy_test.slurm"
  exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is not a directory: ${MODEL_PATH}"
  exit 2
fi

# Hard-disable hub access on compute nodes (defense in depth).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CACHE_ARGS=()
if [[ -n "${MODEL_CACHE_DIR}" ]]; then
  CACHE_ARGS+=(--cache-dir "${MODEL_CACHE_DIR}")
fi

python -u dummy_scoring_new_dataset.py \
  --model-path "${MODEL_PATH}" \
  "${CACHE_ARGS[@]}" \
  --local-files-only \
  --output-file "${OUTPUT_FILE}" \
  "$@"
nvidia-smi || true

echo "Dummy job completed."
