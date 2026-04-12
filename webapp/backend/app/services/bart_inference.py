"""Local HuggingFace inference for BART-CE-RL model."""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import BART_CE_RL_PATH, BART_PREFIX

_model = None
_tokenizer = None
_device = None


def _load():
    global _model, _tokenizer, _device
    if _model is not None:
        return
    _device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _tokenizer = AutoTokenizer.from_pretrained(str(BART_CE_RL_PATH))
    _model = AutoModelForSeq2SeqLM.from_pretrained(str(BART_CE_RL_PATH)).to(_device)
    _model.eval()


async def generate(text: str) -> str:
    _load()
    input_text = BART_PREFIX + text
    encoded = _tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=128,
    ).to(_device)
    with torch.no_grad():
        generated = _model.generate(**encoded, max_length=128, num_beams=4)
    return _tokenizer.decode(generated[0], skip_special_tokens=True)


def is_loaded() -> bool:
    return _model is not None
