from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

from app.services import bart_inference, llama_inference, metric_computer

router = APIRouter(prefix="/api/generate", tags=["inference"])


class GenerateRequest(BaseModel):
    text: str
    model: str = "bart-ce-rl"


class GenerateResponse(BaseModel):
    input: str
    output: str
    model: str
    metrics: dict
    inference_time_ms: float


@router.post("", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    start = time.time()

    if req.model == "llama-3.2-1b":
        output = await llama_inference.generate(req.text)
    else:
        output = await bart_inference.generate(req.text)

    elapsed = (time.time() - start) * 1000
    metrics = metric_computer.compute_metrics(req.text, output)

    return GenerateResponse(
        input=req.text,
        output=output,
        model=req.model,
        metrics=metrics,
        inference_time_ms=round(elapsed, 1),
    )


@router.get("/models")
async def available_models():
    lmstudio_ok = await llama_inference.check_connection()
    return [
        {
            "name": "bart-ce-rl",
            "display": "BART CE+RL",
            "available": True,
            "loaded": bart_inference.is_loaded(),
            "note": "Local HuggingFace inference",
        },
        {
            "name": "llama-3.2-1b",
            "display": "LLaMA 3.2 1B",
            "available": lmstudio_ok,
            "loaded": lmstudio_ok,
            "note": "Via LMStudio" if lmstudio_ok else "LMStudio not connected",
        },
    ]
