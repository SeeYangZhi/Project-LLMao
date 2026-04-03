#!/bin/bash
#SBATCH --job-name=bart-ce-rl
#SBATCH --gpus=a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rl_bart_base_ce_%j.out
#SBATCH --error=logs/rl_bart_base_ce_%j.err

mkdir -p logs

source .venv/bin/activate

python scripts/train_rl.py \
    --sft_checkpoint outputs/bart-base-ce/sar-to-non/final \
    --classifier_model loyongzhe/sarcasm-classifier-binary \
    --direction sar-to-non \
    --data_dir data/splits/sar_to_non_context_enhanced \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-5 \
    --kl_coeff 0.2 \
    --style_weight 0.5 \
    --max_length 128 \
    --output_dir outputs/bart-base-ce-rl/sar-to-non \
    --log_interval 50 \
    --seed 42
