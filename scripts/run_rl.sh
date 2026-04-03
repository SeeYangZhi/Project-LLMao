#!/bin/bash
#SBATCH --job-name=bart-rl-sar2non
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rl_%j.out
#SBATCH --error=logs/rl_%j.err

# Create log directory
mkdir -p logs

# Activate environment
source .venv/bin/activate

python scripts/train_rl.py \
    --sft_checkpoint checkpoints/bart-base/sar-to-non/final \
    --classifier_model loyongzhe/sarcasm-classifier-binary \
    --direction sar-to-non \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-5 \
    --kl_coeff 0.2 \
    --style_weight 0.5 \
    --max_length 128 \
    --output_dir outputs/bart-base-rl/sar-to-non \
    --log_interval 50 \
    --seed 42
