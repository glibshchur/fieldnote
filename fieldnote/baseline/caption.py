"""The caption-only baseline: "a text pipeline wearing a costume".

Every image (figure crops, page renders, photos) is captioned by a strong
vision model, the captions are embedded as text, and the answer model only
ever sees the captions. The captioner is told to transcribe values and read
curves, so this is a fair, strong baseline rather than a straw man.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from tqdm import tqdm

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


@dataclass
class CaptionStats:
    requested: int = 0
    cached: int = 0
    done: int = 0
    failed: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def caption_chunks(
    chunks: list[Chunk],
    llm: LLM,
    model_name: str,
    cache: CaptionCache,
    workers: int = 8,
    include_pages: bool = False,
    limit: int | None = None,
) -> tuple[list[Chunk], CaptionStats]:
    """Return a copy of `chunks` where every image chunk's text is its caption.

    Resumable: captions are cached by (model, image hash), so a rerun only pays for what is
    missing. A failed image is skipped and reported, never fatal. Page renders are excluded
    unless include_pages is set (figure crops already cover the visual evidence).
    """
    kept = [c for c in chunks if include_pages or c.source_type != "manual_page"]
    todo = [c for c in kept if c.modality == "image" and c.image_path]
    keys = {c.id: _key(model_name, c.image_path) for c in todo}
    stats = CaptionStats(requested=len(todo))
    missing = [c for c in todo if cache.get(keys[c.id]) is None]
    stats.cached = len(todo) - len(missing)
    if limit is not None:  # pilot on a random, reproducible sample (not the first N pages)
        missing = random.Random(0).sample(missing, min(limit, len(missing)))
    lock = threading.Lock()

    def run(c: Chunk) -> None:
        hint = f"Source: {c.source_type} from {c.source}" + (f", page {c.page}" if c.page else "")
        try:
            res = llm.complete(CAPTION_SYSTEM, [image_block(c.image_path), text_block(hint)], 2048)
        except Exception as e:  # keep going; the cache makes a rerun pick up the rest
            with lock:
                stats.failed.append(f"{c.id}: {type(e).__name__}: {e}"[:200])
            return
        if not res.text or res.stop_reason == "refusal":
            with lock:
                stats.failed.append(f"{c.id}: empty or refused")
            return
        with lock:
            cache.put(keys[c.id], res.text)
            stats.done += 1
            stats.input_tokens += res.input_tokens
            stats.output_tokens += res.output_tokens

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(run, missing), total=len(missing), desc="captions", disable=not missing))
    stats.seconds = time.perf_counter() - t0

    out = []
    for c in kept:
        out.append(replace(c, text=cache.get(keys[c.id]) or "") if c.id in keys else c)
    return out, stats


def build_caption_index(
    store: ChunkStore, captioned: list[Chunk], text_encoder: TextEncoder
) -> int:
    """Everything, captions included, goes on the text vector."""
    store.recreate(text_encoder.dim, text_encoder.dim)
    vecs = text_encoder.encode_documents([c.text or c.source for c in captioned])
    store.upsert(captioned, TEXT_VEC, vecs)
    return len(captioned)
