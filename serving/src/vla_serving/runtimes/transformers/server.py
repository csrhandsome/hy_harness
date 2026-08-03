#!/usr/bin/env python3
"""Shared OpenAI-compatible server for Transformers vision-language models.

This wraps AutoModelForImageTextToText + AutoProcessor with
/v1/chat/completions. Model-specific values come from the model manifest.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.8
    stream: bool = False
    enable_thinking: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None


def _parse_content(content: Any) -> tuple[str, list[Image.Image]]:
    """Return (text, images) from OpenAI-style message content."""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise HTTPException(status_code=400, detail=f"unsupported content: {type(content)}")

    texts: list[str] = []
    images: list[Image.Image] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            texts.append(str(part.get("text") or ""))
        elif ptype in ("image_url", "image"):
            url = None
            if ptype == "image_url":
                image_url = part.get("image_url") or {}
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
            else:
                url = part.get("image")
            if not url:
                continue
            images.append(_load_image(str(url)))
    return "\n".join(texts).strip(), images


def _load_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        # data:image/jpeg;base64,....
        try:
            b64 = url.split(",", 1)[1]
        except IndexError as e:
            raise HTTPException(status_code=400, detail="invalid data URL") from e
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if url.startswith("http://") or url.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    # local path
    if not os.path.exists(url):
        raise HTTPException(status_code=400, detail=f"image not found: {url}")
    return Image.open(url).convert("RGB")


def build_app(
    model_path: str, model_id: str, served_name: str, device: str, dtype: str
) -> FastAPI:
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    print(f"[{model_id}] loading {model_path} on {device} ({dtype}) ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    chat_tmpl = os.path.join(model_path, "chat_template.jinja")
    if os.path.isfile(chat_tmpl):
        processor.chat_template = open(chat_tmpl, encoding="utf-8").read()

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device).eval()
    print(f"[{model_id}] ready")

    app = FastAPI(title=f"{model_id} (transformers)")
    state = {"model": model, "processor": processor, "served_name": served_name, "model_id": model_id, "device": device}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state["served_name"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": state["model_id"],
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> JSONResponse:
        if req.stream:
            raise HTTPException(status_code=400, detail="streaming not supported yet")

        enable_thinking = True
        if req.enable_thinking is not None:
            enable_thinking = req.enable_thinking
        elif req.chat_template_kwargs and "enable_thinking" in req.chat_template_kwargs:
            enable_thinking = bool(req.chat_template_kwargs["enable_thinking"])

        # Build HF chat messages (last user turn drives generation; keep history as text).
        hf_messages: list[dict[str, Any]] = []
        for msg in req.messages:
            text, images = _parse_content(msg.content)
            content: list[dict[str, Any]] = []
            for img in images:
                content.append({"type": "image", "image": img})
            if text:
                content.append({"type": "text", "text": text})
            if not content:
                content.append({"type": "text", "text": ""})
            hf_messages.append({"role": msg.role, "content": content})

        processor = state["processor"]
        model = state["model"]
        inputs = processor.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        ).to(model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": req.max_tokens,
            "use_cache": True,
            "temperature": req.temperature,
            "do_sample": req.temperature > 0,
        }
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)

        in_len = inputs["input_ids"].shape[1]
        out_ids = generated[0][in_len:]
        text = processor.decode(out_ids, skip_special_tokens=True)

        created = int(time.time())
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": created,
                "model": req.model or state["served_name"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": int(in_len),
                    "completion_tokens": int(out_ids.numel()),
                    "total_tokens": int(in_len + out_ids.numel()),
                },
            }
        )

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "tencent/HY-Embodied-0.5"))
    ap.add_argument("--model-id", default=os.environ.get("SERVING_MODEL_ID", "transformers-model"))
    ap.add_argument("--served-name", default=os.environ.get("SERVED_NAME", "hy_embodied_0_5"))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dtype", default=os.environ.get("DTYPE", "bfloat16"))
    args = ap.parse_args()

    app = build_app(args.model, args.model_id, args.served_name, args.device, args.dtype)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
