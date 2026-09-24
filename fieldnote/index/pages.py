"""Multi-vector page index with exact MaxSim search.

3,444 pages x ~700 patches x 128 dims is about 0.6 GB in fp16, small enough to hold in one
padded tensor and score exhaustively on the GPU in well under a second. That keeps the index
exact and dependency-free (no approximate multi-vector store needed at this scale).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fieldnote.types import Chunk


class PageIndex:
    def __init__(self, embeddings: torch.Tensor, lengths: torch.Tensor, chunks: list[Chunk]):
        self.embeddings = embeddings  # (N, P_max, D), zero-padded
        self.lengths = lengths  # (N,)
        self.chunks = chunks
        self._device_embs: torch.Tensor | None = None

    @classmethod
    def build(cls, chunks: list[Chunk], page_embs: list[torch.Tensor]) -> PageIndex:
        n, p_max, d = len(page_embs), max(e.shape[0] for e in page_embs), page_embs[0].shape[1]
        embs = torch.zeros((n, p_max, d), dtype=torch.float16)
        for i, e in enumerate(page_embs):
            embs[i, : e.shape[0]] = e
        lengths = torch.tensor([e.shape[0] for e in page_embs])
        return cls(embs, lengths, chunks)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": self.embeddings, "lengths": self.lengths}, directory / "pages.pt")
        with (directory / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_payload()) + "\n")

    @classmethod
    def load(cls, directory: Path) -> PageIndex:
        data = torch.load(directory / "pages.pt")
        chunks = [
            Chunk.from_payload(json.loads(line))
            for line in (directory / "chunks.jsonl").read_text().splitlines()
        ]
        return cls(data["embeddings"], data["lengths"], chunks)

    def maxsim(self, query: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """Score every page: sum over query tokens of the best patch similarity."""
        if self._device_embs is None or self._device_embs.device.type != device:
            self._device_embs = self.embeddings.to(device)
        q = query.to(device, self._device_embs.dtype)  # (T, D)
        sims = torch.einsum("td,npd->ntp", q, self._device_embs)  # (N, T, P)
        # Padding rows are zero vectors, but zero can beat negative similarities: mask them.
        pad = (
            torch.arange(sims.shape[-1], device=device)[None, :] >= self.lengths.to(device)[:, None]
        )
        sims = sims.masked_fill(pad[:, None, :], float("-inf"))
        return sims.max(dim=-1).values.float().sum(dim=-1)  # (N,)

    def search(self, query: torch.Tensor, k: int, device: str = "cpu") -> list[tuple[Chunk, float]]:
        scores = self.maxsim(query, device)
        top = torch.topk(scores, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(top.values, top.indices, strict=True)]
