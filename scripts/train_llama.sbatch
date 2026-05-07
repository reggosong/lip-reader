#!/bin/bash
#SBATCH --job-name=vallr-llama-lora
#SBATCH --output=logs/vallr-llama-%j.out
#SBATCH --error=logs/vallr-llama-%j.err
#SBATCH --partition=pi_ppliang
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=65G
#SBATCH --time=12:00:00
# #SBATCH --nodelist=node2500   # uncomment to pin to a specific node
# ---------------------------------------------------------------------------
# Defaults follow the pi_ppliang group's interactive-session recipe
# (1 GPU, 15 CPUs, 65 GB RAM, 12 h). Tweak as needed:
#   --partition / -p   : pi_ppliang  -> the group's queue.
#                        mit_normal_gpu -> MIT's free-GPU pool.
#   --gres=gpu:N       : up to gpu:8 for the group (be considerate).
#   --time             : HH:MM:SS, max 168:00:00 (7 days). 12 h is plenty
#                        for a LoRA run on wikitext-2 with Llama-3.2-3B.
#   --cpus-per-task    : ~15 cores per GPU is the recommended ratio.
#   --mem              : up to ~250G per GPU; 65G is comfortable here.
# ---------------------------------------------------------------------------
#
# One-time setup (login node):
#     conda create -n vallr python=3.10 -y && conda activate vallr
#     # 1) torch matched to the cluster CUDA (skip if `module load` provides it):
#     pip install --index-url https://download.pytorch.org/whl/cu121 \
#         torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
#     # 2) the slim training deps (HF stack + phoneme utils):
#     pip install -r requirements-llm.txt
#     echo "hf_xxx" > ~/.hf_token && chmod 600 ~/.hf_token
#
# Submit with:
#     sbatch scripts/train_llama.sbatch
#
# Override defaults via env vars:
#     OUTPUT_DIR=/scratch/$USER/llama_lora_noisy \
#     NOISE_CONFIG=configs/noisy_llm.json \
#     HF_TOKEN=hf_xxx \
#     sbatch scripts/train_llama.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail

# -------- 0. Paths & user-tunable knobs -----------------------------------
# Repo root = parent of this script's directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Where checkpoints / logs land. Default to a per-job dir under the repo;
# on a real cluster you almost certainly want this on /scratch.
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/llama_lora_${SLURM_JOB_ID:-local}}"
NOISE_CONFIG="${NOISE_CONFIG:-configs/noisy_llm.json}"

mkdir -p "$REPO_ROOT/logs" "$OUTPUT_DIR"

# -------- 1. Modules / conda env ------------------------------------------
# Most clusters expose CUDA + Python via `module`. Edit these to match.
# Comment out the block if you manage everything via conda/venv only.
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load cuda/12.1 || true
    module load python/3.10 || true
fi

# Activate your project env. Pick ONE of the blocks below and delete the rest.

# --- (a) conda env named "vallr" ---
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate vallr
fi

# -------- 2. Hugging Face auth & caches -----------------------------------
# Llama-3.2-3B-Instruct is gated; you need a token with access. Export
# HF_TOKEN before submitting (or stash it in ~/.hf_token, see below).
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_token" ]]; then
    HF_TOKEN="$(cat "$HOME/.hf_token")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it or write it to ~/.hf_token." >&2
    exit 1
fi
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Park the HF cache somewhere with space (datasets + Llama-3.2-3B is ~7 GB).
# Override HF_HOME from outside if your cluster has a shared cache dir.
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# -------- 3. Runtime knobs Llama.py expects -------------------------------
export WANDB_MODE=disabled
export TRANSFORMERS_NO_TF=1
export LLAMA_ATTN_IMPL="${LLAMA_ATTN_IMPL:-sdpa}"   # fall back to "eager" if SDPA misbehaves
export TOKENIZERS_PARALLELISM=false

# -------- 4. Diagnostics --------------------------------------------------
echo "==================== job info ===================="
echo "Job ID         : ${SLURM_JOB_ID:-<no-slurm>}"
echo "Node           : $(hostname)"
echo "Repo root      : $REPO_ROOT"
echo "Output dir     : $OUTPUT_DIR"
echo "Noise config   : $NOISE_CONFIG"
echo "HF cache       : $HF_HOME"
echo "Python         : $(command -v python) ($(python --version 2>&1))"
echo "==================================================="
nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__,
      "| cuda available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

# -------- 5. Train --------------------------------------------------------
srun --unbuffered python Models/Llama.py \
    --noise_config "$NOISE_CONFIG" \
    --output_dir   "$OUTPUT_DIR"

echo "Training finished. Outputs in: $OUTPUT_DIR"


echo "hf_xxx_your_token" > ~/.hf_token
chmod 600 ~/.hf_token