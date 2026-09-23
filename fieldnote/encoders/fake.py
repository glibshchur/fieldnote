"""Deterministic hashing encoders for tests and CI. No model downloads.

The image encoder hashes the tokens of an image's file name (plus an optional
sidecar .txt), which is enough to exercise the full pipeline end to end.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from fieldnote.encoders import l2_normalize


def _hash_embed(texts: list[str], dim: int) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[row, h % dim] += 1.0
        if not out[row].any():
            out[row, 0] = 1.0
    return l2_normalize(out)


class HashingTextEncoder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return _hash_embed(texts, self.dim)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return _hash_embed(texts, self.dim)


class HashingImageEncoder:
    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode_images(self, paths: list[str]) -> np.ndarray:
        texts = []
        for p in paths:
            path = Path(p)
            sidecar = path.with_suffix(".txt")
            extra = sidecar.read_text() if sidecar.exists() else ""
            texts.append(path.stem.replace("_", " ") + " " + extra)
        return _hash_embed(texts, self.dim)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return _hash_embed(texts, self.dim)
