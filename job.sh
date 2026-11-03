#!/bin/bash
export WANDB_API_KEY=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
ID=$(sbatch --parsable \
  -p c12m85-a100-1 \
  --account=csci_ga_3033-2026fa \
  --gres=gpu:1 \
  --cpus-per-task=12 \
  --mem=80G \
  --time=08:00:00 \
  --export=ALL \
  --wrap="uv run python -u -m main")
tail -F slurm-$ID.out