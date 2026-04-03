#!/bin/bash
#SBATCH --job-name=bart-base-ce
#SBATCH --gpus=a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sft_bart_base_ce_%j.out
#SBATCH --error=logs/sft_bart_base_ce_%j.err

mkdir -p logs

source .venv/bin/activate

python scripts/train.py \
    --model facebook/bart-base \
    --direction sar-to-non \
    --data_dir data/splits/sar_to_non_context_enhanced \
    --lr 3e-4 \
    --batch_size 16 \
    --epochs 5 \
    --max_length 128 \
    --output_dir outputs/bart-base-ce/sar-to-non \
    --seed 42
