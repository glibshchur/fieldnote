"""Core data types shared across ingestion, retrieval, answering and evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Literal

Modality = Literal["text", "image", "audio"]
SourceType = Literal["manual_text", "manual_figure", "manual_page", "photo", "call"]


@dataclass
class Chunk:
    """One retrievable unit. Every chunk keeps its modality tag and enough
    provenance to render a citation (page + crop, photo, or call timestamp)."""

    modality: Modality
    source_type: SourceType
    source: str  # file name: manual PDF, photo, or call recording
    text: str = ""  # text content, transcript window, or photo notes
    page: int | None = None  # 1-indexed page number for manual chunks
    bbox: tuple[float, float, float, float] | None = None  # PDF points (x0, y0, x1, y1)
    image_path: str | None = None  # crop / page render / photo on disk
    start_s: float | None = None  # call window start (seconds)
    end_s: float | None = None
    metadata: dict = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            key = f"{self.source_type}|{self.source}|{self.page}|{self.bbox}|{self.start_s}"
            self.id = hashlib.sha1(key.encode()).hexdigest()[:16]

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox) if self.bbox else None
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> Chunk:
        data = dict(payload)
        if data.get("bbox"):
            data["bbox"] = tuple(data["bbox"])
        return cls(**data)

    def citation_label(self) -> str:
        if self.source_type == "call" and self.start_s is not None:
            return f"{self.source} @ {format_ts(self.start_s)}"
        if self.page is not None:
            return f"{self.source}, p.{self.page}"
        return self.source


@dataclass
class Candidate:
    """A chunk returned by one retriever, carried through fusion and reranking."""

    chunk: Chunk
    score: float
    retriever: str  # "text", "image" or "audio"
    rank: int = 0
    fused_score: float = 0.0
    rerank_score: float | None = None


def format_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
