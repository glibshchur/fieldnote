"""End-to-end question answering: retrieve -> fuse -> rerank -> synthesize -> cite."""

from __future__ import annotations

import time
from collections.abc import Callable
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
from fieldnote.types import Candidate, Chunk

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
        k = self.settings.rerank_k
        reranked = self.reranker.rerank(question, fused[:k]) + fused[k:]
        t2 = time.perf_counter()
        return Retrieval(per_lane, fused, reranked, {"search_s": t1 - t0, "rerank_s": t2 - t1})

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        r = self.last_retrieval = self.retrieve(question)
        te = time.perf_counter()
        evidence = prepare_evidence(
            r.reranked,
            self.settings.top_n,
            self.retriever.store.page_chunks,
            self.settings.expand_page_figures,
            self.settings.max_page_renders,
        )
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
        timings.update(
            {"expand_s": t1 - te, "answer_s": t2 - t1, "total_s": time.perf_counter() - t0}
        )
        return Answer(question, text, citations, evidence, stop, timings, usage)


def prepare_evidence(
    reranked: list[Candidate],
    top_n: int,
    page_chunks: Callable[[str, int, str], list[Chunk]],
    expand_limit: int,
    max_page_renders: int,
) -> list[Candidate]:
    """Head of the reranked list, with figure crops preferred over full-page renders.

    A ColQwen hit is a whole page. Keep at most `max_page_renders` of those, fill the freed
    slots with the next non-page candidates, then attach figure crops so the model can read
    the chart without a stack of full-page images crowding it out.
    """
    selected: list[Candidate] = []
    pages_kept = 0
    for cand in reranked:
        if len(selected) >= top_n:
            break
        if cand.chunk.source_type == "manual_page":
            if pages_kept >= max_page_renders:
                continue
            pages_kept += 1
        selected.append(cand)

    # Keep the capped page render. Dropping it in favor of a crop hid diagrams whose
    # answer is the page itself (a wiring figure the detector split into the wrong region).
    added = expand_page_figures(selected, page_chunks, expand_limit)
    return selected + added


def expand_page_figures(
    evidence: list[Candidate], page_chunks: Callable[[str, int, str], list[Chunk]], limit: int
) -> list[Candidate]:
    """Sibling expansion: text often wins retrieval for a page whose answer lives in a chart
    (the chart's title and axis labels are in the text layer, the curve is not). For each
    text chunk in the evidence, add figure crops from the same page, largest first."""
    if limit <= 0:
        return []
    have = {c.chunk.id for c in evidence}
    added: list[Candidate] = []
    for cand in evidence:
        c = cand.chunk
        # Text chunks and page renders both point at a page whose crops are sharper evidence.
        if c.source_type not in ("manual_text", "manual_page") or c.page is None:
            continue
        figures = page_chunks(c.source, c.page, "manual_figure")
        figures.sort(key=lambda f: -_area(f.bbox))
        for fig in figures:
            if fig.id in have:
                continue
            have.add(fig.id)
            added.append(Candidate(chunk=fig, score=0.0, retriever="page_expansion"))
            if len(added) >= limit:
                return added
    return added


def _area(bbox) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox else 0.0


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
        if (
            settings.page_lane
            and settings.k_page > 0
            and (settings.page_index_dir / "pages.pt").exists()
        ):
            from fieldnote.encoders.colpali import ColQwenEncoder
            from fieldnote.index.pages import PageIndex
            from fieldnote.retrieve.search import page_lane

            page_enc = ColQwenEncoder(settings.page_model, settings.device)
            lanes.append(
                page_lane(
                    PageIndex.load(settings.page_index_dir),
                    page_enc,
                    settings.k_page,
                    page_enc.device,
                )
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
            settings.reranker_image_px,
            use_images=mode == "native",
        ),
        make_llm(settings.llm_provider, settings.answer_model, settings.answer_effort),
        mode,
    )
