from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from transformers import pipeline

from . import config

summariser = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global summariser
    print(f"Loading model: {config.MODEL_NAME} ...")
    summariser = pipeline("summarization", model=config.MODEL_NAME)
    print("Model loaded.")
    yield
    summariser = None


app = FastAPI(title="Text Summariser API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummariseRequest(BaseModel):
    text: str
    max_words: Optional[int] = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input text must not be empty")
        return v


class SummariseResponse(BaseModel):
    summary: str
    model: str
    took_ms: float


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


@app.post("/summarise", response_model=SummariseResponse)
async def summarise(req: SummariseRequest):
    if summariser is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = req.text.strip()

    # Truncate very long inputs to stay within model limits
    if len(text) > config.MAX_INPUT_CHARS:
        text = text[: config.MAX_INPUT_CHARS]

    max_words = req.max_words if req.max_words is not None else config.DEFAULT_MAX_WORDS
    max_words = _clamp(max_words, config.MIN_MAX_WORDS, config.MAX_MAX_WORDS)

    # Rough token estimate: 1 word ≈ 1.3 tokens
    max_length = int(max_words * 1.3)
    min_length = max(10, max_length // 4)

    start = time.perf_counter()
    result = summariser(text, max_length=max_length, min_length=min_length, do_sample=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    summary_text = result[0]["summary_text"]

    return SummariseResponse(
        summary=summary_text,
        model=config.MODEL_NAME,
        took_ms=round(elapsed_ms, 1),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": summariser is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host=config.HOST, port=config.PORT, reload=True)
