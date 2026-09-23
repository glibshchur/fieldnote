"""Thin wrapper over the Anthropic SDK, plus a scripted fake for tests."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

# Claude downsamples anything larger; resizing first saves upload time and tokens.
MAX_IMAGE_SIDE = 1568
FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}


@dataclass
class LLMResult:
    text: str
    stop_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def complete(self, system: str, content: list[dict], max_tokens: int = 2048) -> LLMResult: ...


class ClaudeLLM:
    def __init__(self, model: str, effort: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def complete(self, system: str, content: list[dict], max_tokens: int = 2048) -> LLMResult:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.model in FALLBACK_MODELS:
            # On a safety decline, the API re-runs the request on a fallback model server-side.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
            resp = self.client.beta.messages.create(**kwargs)
        else:
            resp = self.client.messages.create(**kwargs)

        if resp.stop_reason == "refusal":
            return LLMResult("The model declined to answer this request.", "refusal")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResult(
            text=text.strip(),
            stop_reason=resp.stop_reason,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class ScriptedLLM:
    """Returns canned responses in order; records every request. For tests."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, list[dict]]] = []

    def complete(self, system: str, content: list[dict], max_tokens: int = 2048) -> LLMResult:
        self.calls.append((system, content))
        text = self.responses.pop(0) if self.responses else ""
        return LLMResult(text, "end_turn")


def image_block(path: str | Path) -> dict:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)  # PNG keeps chart lines and digits crisp
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}
