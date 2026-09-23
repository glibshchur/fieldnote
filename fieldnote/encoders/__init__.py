"""Encoders. Each modality keeps its own encoder; nothing is captioned into text.

Model-backed encoders import torch lazily so the core package (and CI) runs
without the `ml` extra installed.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class ImageEncoder(Protocol):
    """Images and text queries embedded into one shared space (SigLIP/CLIP style)."""

    dim: int

    def encode_images(self, paths: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)
