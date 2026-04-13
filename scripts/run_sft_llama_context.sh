#!/bin/bash
#SBATCH --job-name=llama-ctx-sft
#SBATCH --gpus=h200-141:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sft_llama_context_%j.out
#SBATCH --error=logs/sft_llama_context_%j.err

mkdir -p logs

export TMPDIR=~/tmp
mkdir -p $TMPDIR

source .venv/bin/activate

python scripts/train_llama_context.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --data_dir data/splits/sar_to_non_context_enhanced \
    --article_cache data/processed/intermediate/article_scrape_cache.jsonl \
    --lr 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --epochs 3 \
    --max_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --output_dir outputs/llama-3.2-1b-instruct-context/sar-to-non \
    --seed 42
