"""Local HuggingFace inference for Camille's T5-base joint model.

The joint model emits `strategy: <X> rewrite: <Y>` — we return only the rewrite
so the playground output matches the other models.
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import T5_JOINT_HF_ID, T5_JOINT_PREFIX

_model = None
_tokenizer = None
_device = None

_REWRITE_RE = re.compile(r"rewrite:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _load():
    global _model, _tokenizer, _device
    if _model is not None:
        return
    _device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _tokenizer = AutoTokenizer.from_pretrained(T5_JOINT_HF_ID)
    _model = AutoModelForSeq2SeqLM.from_pretrained(T5_JOINT_HF_ID).to(_device)
    _model.eval()


def _parse_joint(output: str) -> str:
    match = _REWRITE_RE.search(output)
    return match.group(1).strip() if match else output.strip()


async def generate(text: str) -> str:
    _load()
    input_text = T5_JOINT_PREFIX + text
    encoded = _tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=128,
    ).to(_device)
    with torch.no_grad():
        generated = _model.generate(**encoded, max_length=128, num_beams=4)
    decoded = _tokenizer.decode(generated[0], skip_special_tokens=True)
    return _parse_joint(decoded)


def is_loaded() -> bool:
    return _model is not None
