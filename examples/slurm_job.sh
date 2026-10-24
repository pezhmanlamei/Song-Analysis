#!/bin/bash
# Example SLURM array job for a university GPU cluster (adjust partition/account/modules).
# 80,000 songs: e.g. 40 array tasks x ~2,000 songs each, each task with one GPU.
#SBATCH --job-name=songfeatures
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-40
#SBATCH --output=logs/songfeatures_%A_%a.log

module load ffmpeg 2>/dev/null || true
source ~/venvs/songfeatures/bin/activate

# Compute nodes often have no internet: run once on a login node beforehand
#   python -m songfeatures download-models --config config.yaml
export HF_HOME=$HOME/.cache/huggingface
export TORCH_HOME=$HOME/.cache/torch

N=40
# Each task processes every N-th song; results go to the same (shared) results folder.
# Resumable: resubmitting a task continues from its cache.
python -m songfeatures run \
  --manifest data/manifest.csv \
  --out results/ \
  --config config.yaml \
  --workers "$SLURM_CPUS_PER_TASK" \
  --shard "${SLURM_ARRAY_TASK_ID}/${N}"

# When ALL array tasks have finished, run once (CPU is enough):
#   python -m songfeatures merge --manifest data/manifest.csv --out results/ --config config.yaml
# e.g. as a dependent job:  sbatch --dependency=afterok:<ARRAY_JOB_ID> merge_job.sh
