"""Late fusion, step 1: run one search per modality in parallel."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from fieldnote.index.store import IMAGE_VEC, TEXT_VEC, ChunkStore
from fieldnote.types import Candidate, Chunk


@dataclass
class Lane:
    name: str  # "text", "image", "audio", "page"
    vector_name: str
    modalities: tuple[str, ...]
    k: int
    embed_query: Callable[[str], np.ndarray] | None = None
    # A lane can bring its own search (e.g. the multi-vector page index) instead of Qdrant.
    search: Callable[[str, int], list[tuple[Chunk, float]]] | None = None


class ParallelRetriever:
    def __init__(self, store: ChunkStore, lanes: list[Lane]):
        self.store = store
        self.lanes = [lane for lane in lanes if lane.k > 0]
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.lanes)))

    def _run_lane(self, lane: Lane, question: str) -> list[Candidate]:
        if lane.search is not None:
            hits = lane.search(question, lane.k)
        else:
            vec = lane.embed_query(question)
            hits = self.store.search(lane.vector_name, vec, lane.k, modalities=lane.modalities)
        return [
            Candidate(chunk=c, score=s, retriever=lane.name, rank=r)
            for r, (c, s) in enumerate(hits, start=1)
        ]

    def retrieve(self, question: str) -> dict[str, list[Candidate]]:
        futures = {
            lane.name: self._pool.submit(self._run_lane, lane, question) for lane in self.lanes
        }
        return {name: f.result() for name, f in futures.items()}


def native_lanes(
    text_encoder, image_encoder, k_text: int, k_image: int, k_audio: int
) -> list[Lane]:
    """Fieldnote: images are searched by their own pixels in SigLIP space."""
    embed_text = lambda q: text_encoder.encode_queries([q])[0]  # noqa: E731
    embed_image = lambda q: image_encoder.encode_queries([q])[0]  # noqa: E731
    return [
        Lane("text", TEXT_VEC, ("text",), k_text, embed_text),
        Lane("image", IMAGE_VEC, ("image",), k_image, embed_image),
        Lane("audio", TEXT_VEC, ("audio",), k_audio, embed_text),
    ]


def page_lane(page_index, page_encoder, k: int, device: str) -> Lane:
    """Late-interaction page retrieval (ColQwen2.5 + MaxSim) over full-page renders."""

    def search(question: str, k: int) -> list[tuple[Chunk, float]]:
        return page_index.search(page_encoder.encode_queries([question])[0], k, device)

    return Lane("page", "", ("image",), k, search=search)


def caption_lanes(text_encoder, k_text: int, k_image: int, k_audio: int) -> list[Lane]:
    """Baseline: images are searched through their captions on the text vector."""
    embed_text = lambda q: text_encoder.encode_queries([q])[0]  # noqa: E731
    return [
        Lane("text", TEXT_VEC, ("text",), k_text, embed_text),
        Lane("image", TEXT_VEC, ("image",), k_image, embed_text),
        Lane("audio", TEXT_VEC, ("audio",), k_audio, embed_text),
    ]
