"""LLM backends (Anthropic, Gemini) behind one interface, plus a scripted fake for tests.

The pipeline builds Anthropic-style content blocks ({"type": "text"} / {"type": "image"});
each backend converts them to its own wire format."""

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


class GeminiLLM:
    """Google Gemini via the google-genai SDK. Reads GEMINI_API_KEY from the environment."""

    def __init__(
        self, model: str, thinking_level: str | None = None, media_resolution: str = "high"
    ):
        import os

        from google import genai
        from google.genai import types

        # Without a timeout a dropped connection (e.g. a Wi-Fi switch) hangs the eval forever.
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=60_000,  # ms
                retry_options=types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=30.0),
            ),
        )
        self.model = model
        self.thinking_level = thinking_level
        # Charts need the high-resolution image tokenizer; low loses axis tick labels.
        self.media_resolution = f"MEDIA_RESOLUTION_{media_resolution.upper()}"

    def _parts(self, content: list[dict]) -> list:
        from google.genai import types

        parts = []
        for block in content:
            if block["type"] == "text":
                parts.append(types.Part.from_text(text=block["text"]))
            elif block["type"] == "image":
                src = block["source"]
                parts.append(
                    types.Part.from_bytes(
                        data=base64.standard_b64decode(src["data"]),
                        mime_type=src["media_type"],
                        media_resolution={"level": self.media_resolution},
                    )
                )
        return parts

    def complete(self, system: str, content: list[dict], max_tokens: int = 2048) -> LLMResult:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            # Plain generation: no tools, so skip the SDK's automatic function calling.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if self.thinking_level:
            config.thinking_config = types.ThinkingConfig(
                thinking_level=self.thinking_level.upper()
            )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=self._parts(content))],
            config=config,
        )
        finish = resp.candidates[0].finish_reason if resp.candidates else None
        finish_name = getattr(finish, "value", finish)
        if finish_name in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST") or not resp.candidates:
            return LLMResult("The model declined to answer this request.", "refusal")
        usage = resp.usage_metadata
        return LLMResult(
            text=(resp.text or "").strip(),
            stop_reason=finish_name,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )


def make_llm(provider: str, model: str, effort: str | None = None) -> LLM:
    """`effort` maps to Claude's output_config effort or Gemini's thinking level."""
    if provider == "anthropic":
        return ClaudeLLM(model, effort)
    if provider == "gemini":
        return GeminiLLM(model, thinking_level=effort)
    raise ValueError(f"unknown LLM provider: {provider}")


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
