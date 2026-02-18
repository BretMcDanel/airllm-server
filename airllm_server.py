import asyncio
import os
import time
import uuid
import logging
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from air_llm import AutoModelForCausalLM
import torch

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("airllm-server")

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DEFAULT_MODEL = os.getenv("AIRLLM_DEFAULT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
API_KEY = os.getenv("AIRLLM_API_KEY")

AVAILABLE_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

app = FastAPI()

# Global model registry + lock
models: dict[str, object] = {}
models_lock = asyncio.Lock()


# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool | None = False
    max_tokens: int | None = 512


class ChatResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[ChatResponseChoice]
    usage: Usage


# ---------------------------------------------------------
# Auth
# ---------------------------------------------------------
def check_api_key(authorization: str | None):
    if API_KEY is None:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
async def get_or_load_model(model_name: str):
    async with models_lock:
        if model_name in models:
            return models[model_name]

        log.info(f"[model] Loading model: {model_name}")
        try:
            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                use_flash_attention=True,
                device_map="auto",
            )
        except Exception as e:
            log.exception(f"[model] Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Model load error: {str(e)}")

        models[model_name] = llm
        log.info(f"[model] Model loaded: {model_name}")
        return llm


@app.on_event("startup")
async def startup_event():
    await get_or_load_model(DEFAULT_MODEL)
    log.info(f"[startup] Default model ready: {DEFAULT_MODEL}")


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(models.keys()),
    }


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def extract_prompt(messages: list[Message]) -> str:
    user_messages = [m.content for m in messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user messages provided")
    return user_messages[-1]


def estimate_tokens(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------
async def stream_generator(llm, prompt: str, max_tokens: int, request_id: str):
    start = time.time()
    try:
        output = llm.generate(prompt, max_new_tokens=max_tokens)
    except Exception as e:
        log.exception(f"[{request_id}] Inference error: {e}")
        yield b'data: {"error": "inference_error"}\n\n'
        return
    end = time.time()

    log.info(f"[{request_id}] Inference completed in {end - start:.2f}s")

    words = output.split()
    chunks = []
    current = []

    for token in words:
        current.append(token)
        if len(current) >= 40:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))

    for i, chunk in enumerate(chunks):
        msg = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": chunk},
                    "finish_reason": None if i < len(chunks) - 1 else "stop",
                }
            ],
        }
        yield f"data: {msg}\n\n".encode()

    yield b"data: [DONE]\n\n"


# ---------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat(
    req: ChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    check_api_key(authorization)

    model_name = req.model or DEFAULT_MODEL
    request_id = str(uuid.uuid4())

    log.info(
        f"[{request_id}] /v1/chat/completions model={model_name} "
        f"stream={req.stream} from={request.client.host}"
    )

    llm = await get_or_load_model(model_name)
    prompt = extract_prompt(req.messages)
    max_tokens = req.max_tokens or 512

    if req.stream:
        return StreamingResponse(
            stream_generator(llm, prompt, max_tokens, request_id),
            media_type="text/event-stream",
        )

    start = time.time()
    try:
        output = llm.generate(prompt, max_new_tokens=max_tokens)
    except Exception as e:
        log.exception(f"[{request_id}] Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    end = time.time()

    log.info(f"[{request_id}] Inference completed in {end - start:.2f}s")

    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(output)
    total_tokens = prompt_tokens + completion_tokens

    reply = Message(role="assistant", content=output)

    return ChatResponse(
        id=request_id,
        choices=[ChatResponseChoice(index=0, message=reply)],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


# ---------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------
@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    check_api_key(authorization)

    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "loaded": m in models,
                "default": m == DEFAULT_MODEL,
            }
            for m in AVAILABLE_MODELS
        ],
    }


@app.post("/v1/models/load")
async def load_model_endpoint(model: str, authorization: str | None = Header(default=None)):
    check_api_key(authorization)

    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Unknown model")

    await get_or_load_model(model)
    return {"status": "loaded", "model": model}


@app.post("/v1/models/unload")
async def unload_model_endpoint(model: str, authorization: str | None = Header(default=None)):
    check_api_key(authorization)

    async with models_lock:
        if model not in models:
            return {"status": "not_loaded", "model": model}

        del models[model]
        torch.cuda.empty_cache()

    return {"status": "unloaded", "model": model}


@app.post("/v1/models/reload")
async def reload_model_endpoint(model: str, authorization: str | None = Header(default=None)):
    check_api_key(authorization)

    await unload_model_endpoint(model, authorization)
    await load_model_endpoint(model, authorization)

    return {"status": "reloaded", "model": model}


@app.get("/v1/models/vram")
async def vram_report(authorization: str | None = Header(default=None)):
    check_api_key(authorization)

    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = total - reserved

    return {
        "gpu": torch.cuda.get_device_name(0),
        "total_vram": total,
        "reserved_vram": reserved,
        "allocated_vram": allocated,
        "free_vram": free,
        "loaded_models": list(models.keys()),
    }


# ---------------------------------------------------------
# Run server
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "airllm_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
