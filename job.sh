
#!/bin/bash
srun -p a100-4-std \
  --account=csci_ga_3033-2026fa \
  --gres=gpu:1 \
  --cpus-per-task=12 \
  --mem=80G \
  --time=08:00:00 \
uv run python -m main