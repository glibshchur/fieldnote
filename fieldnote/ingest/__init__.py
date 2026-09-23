"""Ingestion: turn manuals, photos and call recordings into Chunks."""

from __future__ import annotations

from fieldnote.config import Settings
from fieldnote.ingest.audio import Transcriber, ingest_calls
from fieldnote.ingest.pdf import PdfIngestOptions, ingest_pdf
from fieldnote.ingest.photos import ingest_photos
from fieldnote.types import Chunk


def ingest_corpus(
    settings: Settings,
    transcriber: Transcriber | None = None,
    pdf_opts: PdfIngestOptions | None = None,
) -> list[Chunk]:
    data = settings.data_dir
    chunks: list[Chunk] = []

    manuals = data / "manuals"
    if manuals.exists():
        for pdf in sorted(manuals.glob("*.pdf")):
            chunks.extend(ingest_pdf(pdf, settings.crops_dir, settings.pages_dir, pdf_opts))

    photos = data / "photos"
    if photos.exists():
        chunks.extend(ingest_photos(photos, settings.photos_dir))

    calls = data / "calls"
    if calls.exists() and any(calls.iterdir()):
        if transcriber is None:
            from fieldnote.ingest.audio import FasterWhisperTranscriber

            transcriber = FasterWhisperTranscriber(settings.whisper_model)
        chunks.extend(ingest_calls(calls, settings.artifacts_dir / "transcripts", transcriber))

    return chunks


def save_chunks(chunks: list[Chunk], path) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_payload()) + "\n")


def load_chunks(path) -> list[Chunk]:
    import json
    from pathlib import Path

    return [Chunk.from_payload(json.loads(line)) for line in Path(path).read_text().splitlines()]


__all__ = [
    "ingest_corpus",
    "ingest_pdf",
    "ingest_photos",
    "ingest_calls",
    "save_chunks",
    "load_chunks",
]
