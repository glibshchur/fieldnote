"""The caption-only baseline: "a text pipeline wearing a costume".

Every image (figure crops, page renders, photos) is captioned by a strong
vision model, the captions are embedded as text, and the answer model only
ever sees the captions. The captioner is told to transcribe values and read
curves, so this is a fair, strong baseline rather than a straw man.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from fieldnote.answer.prompts import CAPTION_SYSTEM
from fieldnote.encoders import TextEncoder
from fieldnote.index.store import TEXT_VEC, ChunkStore
from fieldnote.llm import LLM, image_block, text_block
from fieldnote.types import Chunk


class CaptionCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.data[row["key"]] = row["caption"]

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, caption: str) -> None:
        self.data[key] = caption
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "caption": caption}) + "\n")


def _key(model: str, image_path: str) -> str:
    return model + ":" + hashlib.sha1(Path(image_path).read_bytes()).hexdigest()


def caption_chunks(
    chunks: list[Chunk], llm: LLM, model_name: str, cache: CaptionCache, workers: int = 8
) -> list[Chunk]:
    """Return a copy of `chunks` where every image chunk's text is its caption."""
    todo = [c for c in chunks if c.modality == "image" and c.image_path]
    keys = {c.id: _key(model_name, c.image_path) for c in todo}

    def run(c: Chunk) -> tuple[str, str]:
        hint = f"Source: {c.source_type} from {c.source}" + (f", page {c.page}" if c.page else "")
        res = llm.complete(CAPTION_SYSTEM, [image_block(c.image_path), text_block(hint)], 1024)
        return c.id, res.text

    missing = [c for c in todo if cache.get(keys[c.id]) is None]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for cid, caption in pool.map(run, missing):
            cache.put(keys[cid], caption)

    out = []
    for c in chunks:
        if c.id in keys:
            out.append(replace(c, text=cache.get(keys[c.id]) or ""))
        else:
            out.append(c)
    return out


def build_caption_index(
    store: ChunkStore, captioned: list[Chunk], text_encoder: TextEncoder
) -> int:
    """Everything, captions included, goes on the text vector."""
    store.recreate(text_encoder.dim, text_encoder.dim)
    vecs = text_encoder.encode_documents([c.text or c.source for c in captioned])
    store.upsert(captioned, TEXT_VEC, vecs)
    return len(captioned)
