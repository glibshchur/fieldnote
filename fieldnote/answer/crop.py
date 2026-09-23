"""Citation images: every manual citation comes back with a picture of its evidence."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from fieldnote.types import Chunk


def citation_image(chunk: Chunk, manuals_dir: Path, crops_dir: Path, dpi: int = 160) -> str | None:
    """Return a path to an image the technician can check the answer against.

    Figures, pages and photos already have one. A text chunk gets its page
    region rendered on demand, so the citation shows the paragraph in context.
    Calls have no image; their citation is a timestamp.
    """
    if chunk.image_path:
        return chunk.image_path
    if chunk.source_type != "manual_text" or chunk.page is None or chunk.bbox is None:
        return None
    out = crops_dir / f"{Path(chunk.source).stem}_p{chunk.page:03d}_text_{chunk.id}.png"
    if out.exists():
        return str(out)
    pdf = manuals_dir / chunk.source
    if not pdf.exists():
        return None
    crops_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf) as doc:
        page = doc[chunk.page - 1]
        x0, y0, x1, y1 = chunk.bbox
        clip = pymupdf.Rect(x0 - 12, y0 - 12, x1 + 12, y1 + 12) & page.rect
        page.get_pixmap(clip=clip, dpi=dpi).save(out)
    return str(out)
