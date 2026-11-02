
#!/bin/bash
#SBATCH -p c12m85-a100-1
#SBATCH --account=csci_ga_3033-2026fa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=08:00:00

export HF_TOKEN="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"
uv run python3 -m main