"""HTTP API: POST /ask returns the answer, citations and links to citation images."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fieldnote.config import get_settings
from fieldnote.pipeline import Fieldnote, load_pipeline

settings = get_settings()
app = FastAPI(title="Fieldnote", version="0.1.0")
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=settings.artifacts_dir), name="artifacts")


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    id: str
    label: str
    source_type: str
    page: int | None
    start_s: float | None
    image_url: str | None


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    timings: dict[str, float]


@lru_cache(maxsize=1)
def pipeline() -> Fieldnote:
    return load_pipeline(settings)


def _url(path: str | None) -> str | None:
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(settings.artifacts_dir.resolve())
    except ValueError:
        return None
    return f"/artifacts/{rel.as_posix()}"


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    ans = pipeline().ask(req.question)
    return AskResponse(
        answer=ans.text,
        citations=[
            CitationOut(
                id=c.sid,
                label=c.label,
                source_type=c.candidate.chunk.source_type,
                page=c.candidate.chunk.page,
                start_s=c.candidate.chunk.start_s,
                image_url=_url(c.image_path),
            )
            for c in ans.citations
        ],
        timings=ans.timings,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
