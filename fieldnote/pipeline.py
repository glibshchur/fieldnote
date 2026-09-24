"""End-to-end question answering: retrieve -> fuse -> rerank -> synthesize -> cite."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from fieldnote.answer.crop import citation_image
from fieldnote.answer.synthesize import Answer, Citation, synthesize
from fieldnote.config import Settings
from fieldnote.index.store import ChunkStore, make_client
from fieldnote.llm import LLM
from fieldnote.retrieve.fuse import rrf_fuse
from fieldnote.retrieve.rerank import Reranker
from fieldnote.retrieve.search import ParallelRetriever, caption_lanes, native_lanes
from fieldnote.types import Candidate

Mode = Literal["native", "caption"]


@dataclass
class Retrieval:
    per_lane: dict[str, list[Candidate]]
    fused: list[Candidate]
    reranked: list[Candidate]
    timings: dict[str, float]


class Fieldnote:
    def __init__(
        self,
        settings: Settings,
        retriever: ParallelRetriever,
        reranker: Reranker,
        llm: LLM,
        mode: Mode = "native",
    ):
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.mode = mode
        self.last_retrieval: Retrieval | None = None

    def retrieve(self, question: str) -> Retrieval:
        t0 = time.perf_counter()
        per_lane = self.retriever.retrieve(question)
        t1 = time.perf_counter()
        fused = rrf_fuse(per_lane)
        reranked = self.reranker.rerank(question, fused)
        t2 = time.perf_counter()
        return Retrieval(per_lane, fused, reranked, {"search_s": t1 - t0, "rerank_s": t2 - t1})

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        r = self.last_retrieval = self.retrieve(question)
        evidence = r.reranked[: self.settings.top_n]
        t1 = time.perf_counter()
        text, cited, stop, usage = synthesize(
            self.llm, question, evidence, use_images=self.mode == "native"
        )
        t2 = time.perf_counter()

        citations = []
        for sid in cited:
            cand = evidence[int(sid[1:]) - 1]
            img = citation_image(
                cand.chunk, self.settings.data_dir / "manuals", self.settings.crops_dir
            )
            citations.append(Citation(sid, cand.chunk.citation_label(), cand, img))

        timings = dict(r.timings)
        timings.update({"answer_s": t2 - t1, "total_s": time.perf_counter() - t0})
        return Answer(question, text, citations, evidence, stop, timings, usage)


def collection_for(settings: Settings, mode: Mode) -> str:
    return settings.collection if mode == "native" else f"{settings.collection}_caption"


def load_pipeline(
    settings: Settings, mode: Mode = "native", reranker: str | None = None
) -> Fieldnote:
    """Build the real pipeline with model-backed encoders (needs the `ml` extra)."""
    from fieldnote.encoders.text import SentenceTransformerEncoder
    from fieldnote.llm import make_llm
    from fieldnote.retrieve.rerank import make_reranker

    store = ChunkStore(
        make_client(settings.qdrant_path, settings.qdrant_url), collection_for(settings, mode)
    )
    text_enc = SentenceTransformerEncoder(settings.text_model, settings.device)
    if mode == "native":
        from fieldnote.encoders.image import SiglipEncoder

        image_enc = SiglipEncoder(settings.image_model, settings.device)
        lanes = native_lanes(
            text_enc, image_enc, settings.k_text, settings.k_image, settings.k_audio
        )
    else:
        lanes = caption_lanes(text_enc, settings.k_text, settings.k_image, settings.k_audio)
    return Fieldnote(
        settings,
        ParallelRetriever(store, lanes),
        make_reranker(
            reranker or settings.reranker,
            settings.reranker_model,
            settings.device,
            settings.reranker_revision,
        ),
        make_llm(settings.llm_provider, settings.answer_model, settings.answer_effort),
        mode,
    )
