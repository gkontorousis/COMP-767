#!/bin/bash
#SBATCH --job-name=mdl-science
#SBATCH --account=def-sreddy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40000M
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

export HF_HOME="${SCRATCH}/huggingface"
export TORCH_HOME="${SCRATCH}/torch"
export TMPDIR="${SCRATCH}/tmp/${SLURM_JOB_ID:-job}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

# Download the model once on a login node, then set MODEL_PATH to the parent
# directory that contains per-model folders.
#
# Example (login node):
#   mkdir -p "${SCRATCH}/models"
#   hf download Qwen/Qwen3-14B-Base --local-dir "${SCRATCH}/models/Qwen3-14B-Base"
#
# Submit with:
#   sbatch --export=ALL,MODEL_PATH="${SCRATCH}/models" \
#       scratch/mdl_scoring_science.slurm \
#       --model-name Qwen3-14B-Base
MODEL_PATH="${MODEL_PATH:-}"

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-none}"
echo "HF_HOME: ${HF_HOME}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "MODEL_PATH (models root): ${MODEL_PATH:-<unset>}"
echo "MODEL_CACHE_DIR: ${MODEL_CACHE_DIR:-<unset>}"
if [[ "$#" -gt 0 ]]; then
  echo "Forwarded sbatch args:"
  printf '  %q\n' "$@"
else
  echo "Forwarded sbatch args: <none>"
fi

module load python/3.12
source ~/venvs/comp767/bin/activate
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"
echo "PWD: $(pwd)"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is not set."
  echo "Download the model on a login node, then re-submit with:"
  echo "  sbatch --export=ALL,MODEL_PATH=\"\${SCRATCH}/models\" scratch/mdl_scoring_science.slurm --model-name Qwen3-14B-Base"
  exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is not a directory: ${MODEL_PATH}"
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CACHE_ARGS=()
if [[ -n "${MODEL_CACHE_DIR}" ]]; then
  CACHE_ARGS+=(--cache-dir "${MODEL_CACHE_DIR}")
fi

python -u scratch/dummy_mdl_scoring_science.py \
  --model-path "${MODEL_PATH}" \
  "${CACHE_ARGS[@]}" \
  --local-files-only \
  "$@"
nvidia-smi || true

echo "MDL ScienceWorld job completed."
