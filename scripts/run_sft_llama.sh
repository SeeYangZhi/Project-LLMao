#!/bin/bash
#SBATCH --job-name=llama-sft
#SBATCH --gpus=h200-141:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sft_llama_%j.out
#SBATCH --error=logs/sft_llama_%j.err

mkdir -p logs

export TMPDIR=~/tmp
mkdir -p $TMPDIR

source .venv/bin/activate

python scripts/train_llama.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --data_dir data/splits/sar_to_non_context_enhanced \
    --lr 2e-4 \
    --batch_size 8 \
    --grad_accum 2 \
    --epochs 3 \
    --max_length 256 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --output_dir outputs/llama-3.2-1b-instruct/sar-to-non \
    --seed 42
