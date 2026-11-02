bash
#!/bin/bash
sbatch -p c24m170-a100-1 \
  --account=csci_ga_3033-2026fa \
  --gres=gpu:1 \
  --cpus-per-task=12 \
  --mem=80G \
  --time=08:00:00 \
  --wrap="uv run python -u -m main"