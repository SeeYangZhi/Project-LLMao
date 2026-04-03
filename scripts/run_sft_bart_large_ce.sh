#!/bin/bash
#SBATCH --job-name=bart-large-ce
#SBATCH --gpus=a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sft_bart_large_ce_%j.out
#SBATCH --error=logs/sft_bart_large_ce_%j.err

mkdir -p logs

source .venv/bin/activate

python -c "
import torch
from transformers import AutoModelForSeq2SeqLM
m = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')
print('dtype:', next(m.parameters()).dtype)
print('any nan:', any(torch.isnan(p).any().item() for p in m.parameters()))
" && python scripts/train.py \
    --model facebook/bart-large \
    --direction sar-to-non \
    --data_dir data/splits/sar_to_non_context_enhanced \
    --lr 1e-5 \
    --batch_size 8 \
    --no_mixed_precision \
    --epochs 5 \
    --max_length 128 \
    --output_dir outputs/bart-large-ce/sar-to-non \
    --seed 42
