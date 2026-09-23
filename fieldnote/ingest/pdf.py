"""Manual PDF ingestion.

Each page yields three kinds of chunks:
  * manual_text:   text blocks grouped into ~900 character chunks, with a bbox
  * manual_figure: figure/table regions (raster images, clusters of vector
                   drawings, detected tables) cropped to PNG at high DPI
  * manual_page:   a render of the whole page (fallback + page-level retrieval)

Figures are indexed as images, never captioned. That is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from fieldnote.types import Chunk

Rect = tuple[float, float, float, float]

TEXT_CHUNK_CHARS = 900
MIN_FIGURE_SIDE = 60.0  # points
MIN_VECTOR_ITEMS = 6  # a chart or diagram has many strokes; an underline has one
CLUSTER_GAP = 10.0  # merge drawings closer than this (points)
CROP_MARGIN = 18.0  # include axis labels and legends that sit just outside the strokes
PAGE_COVER_LIMIT = 0.9  # ignore page frames/borders


@dataclass
class PdfIngestOptions:
    crop_dpi: int = 200
    page_dpi: int = 110
    index_pages: bool = True
    detect_tables: bool = True
    tile_fallback: bool = True  # 2x2 tiles when a page has no detected figures but has drawings


def ingest_pdf(
    pdf_path: Path, crops_dir: Path, pages_dir: Path, opts: PdfIngestOptions | None = None
) -> list[Chunk]:
    opts = opts or PdfIngestOptions()
    crops_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            chunks.extend(_ingest_page(page, pdf_path, crops_dir, pages_dir, opts))
    return chunks


def _ingest_page(
    page: pymupdf.Page, pdf_path: Path, crops_dir: Path, pages_dir: Path, opts: PdfIngestOptions
) -> list[Chunk]:
    source = pdf_path.name
    stem = pdf_path.stem
    pno = page.number + 1
    chunks: list[Chunk] = []

    chunks.extend(_text_chunks(page, source))

    regions = detect_figure_regions(page, detect_tables=opts.detect_tables)
    if not regions and opts.tile_fallback and len(page.get_drawings()) >= MIN_VECTOR_ITEMS:
        regions = [(r, "tile") for r in _tiles(page.rect)]

    for i, (rect, kind) in enumerate(regions):
        clip = _expand(rect, CROP_MARGIN, page.rect)
        out = crops_dir / f"{stem}_p{pno:03d}_{kind}{i}.png"
        page.get_pixmap(clip=pymupdf.Rect(clip), dpi=opts.crop_dpi).save(out)
        chunks.append(
            Chunk(
                modality="image",
                source_type="manual_figure",
                source=source,
                text=_text_in(page, clip),
                page=pno,
                bbox=clip,
                image_path=str(out),
                metadata={"region_kind": kind, "region_index": i},
            )
        )

    if opts.index_pages:
        out = pages_dir / f"{stem}_p{pno:03d}.png"
        page.get_pixmap(dpi=opts.page_dpi).save(out)
        r = page.rect
        chunks.append(
            Chunk(
                modality="image",
                source_type="manual_page",
                source=source,
                page=pno,
                bbox=(r.x0, r.y0, r.x1, r.y1),
                image_path=str(out),
            )
        )
    return chunks


def _text_chunks(page: pymupdf.Page, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[str] = []
    box: Rect | None = None

    def flush() -> None:
        nonlocal buf, box
        text = " ".join(buf).strip()
        if text:
            chunks.append(
                Chunk(
                    modality="text",
                    source_type="manual_text",
                    source=source,
                    text=text,
                    page=page.number + 1,
                    bbox=box,
                )
            )
        buf, box = [], None

    for x0, y0, x1, y1, text, _no, btype in page.get_text("blocks", sort=True):
        if btype != 0:  # image block
            continue
        text = " ".join(text.split())
        if not text:
            continue
        if buf and sum(len(t) for t in buf) + len(text) > TEXT_CHUNK_CHARS:
            flush()
        buf.append(text)
        box = _union(box, (x0, y0, x1, y1))
    flush()
    return chunks


def detect_figure_regions(page: pymupdf.Page, detect_tables: bool = True) -> list[tuple[Rect, str]]:
    """Find chart, diagram, photo and table regions on a page."""
    page_area = page.rect.width * page.rect.height
    found: list[tuple[Rect, str]] = []

    # 1. Raster images placed on the page (scanned charts, photos, diagrams).
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            found.append(((r.x0, r.y0, r.x1, r.y1), "image"))

    # 2. Vector drawings (plotted curves, wiring diagrams) clustered into regions.
    # Weight each drawing by its path segments: one path can hold a whole chart grid.
    drawings = [d for d in page.get_drawings() if d.get("rect") is not None]
    rects = [(d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1) for d in drawings]
    weights = [len(d.get("items", [])) or 1 for d in drawings]
    for rect, count in _cluster(rects, CLUSTER_GAP, weights):
        if count >= MIN_VECTOR_ITEMS:
            found.append((rect, "figure"))

    # 3. Tables (PyMuPDF's detector); many manual tables are the only home of a spec value.
    if detect_tables:
        try:
            for t in page.find_tables().tables:
                found.append((tuple(t.bbox), "table"))
        except Exception:  # table detection is best-effort
            pass

    kept: list[tuple[Rect, str]] = []
    for rect, kind in found:
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if w < MIN_FIGURE_SIDE or h < MIN_FIGURE_SIDE:
            continue
        if (w * h) / page_area > PAGE_COVER_LIMIT:
            continue
        kept.append((rect, kind))
    return _merge_overlapping(kept)


def _cluster(
    rects: list[Rect], gap: float, weights: list[int] | None = None
) -> list[tuple[Rect, int]]:
    """Union-find over rectangles that overlap once expanded by `gap`.
    Returns each cluster's bbox and total weight."""
    weights = weights or [1] * len(rects)
    parent = list(range(len(rects)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    expanded = [(r[0] - gap, r[1] - gap, r[2] + gap, r[3] + gap) for r in rects]
    order = sorted(range(len(rects)), key=lambda i: expanded[i][0])
    for a_idx, i in enumerate(order):
        for j in order[a_idx + 1 :]:
            if expanded[j][0] > expanded[i][2]:
                break
            if _overlaps(expanded[i], expanded[j]):
                parent[find(i)] = find(j)

    groups: dict[int, tuple[Rect, int]] = {}
    for i, r in enumerate(rects):
        root = find(i)
        box, n = groups.get(root, (None, 0))
        groups[root] = (_union(box, r), n + weights[i])
    return list(groups.values())


def _merge_overlapping(regions: list[tuple[Rect, str]]) -> list[tuple[Rect, str]]:
    merged: list[tuple[Rect, str]] = []
    for rect, kind in sorted(regions, key=lambda x: -_area(x[0])):
        for i, (m, mkind) in enumerate(merged):
            if _overlaps(m, rect):
                merged[i] = (_union(m, rect), mkind)
                break
        else:
            merged.append((rect, kind))
    return sorted(merged, key=lambda x: (x[0][1], x[0][0]))


def _text_in(page: pymupdf.Page, rect: Rect) -> str:
    """Whole words touching the region (clip-based extraction cuts words in half)."""
    words = [w for w in page.get_text("words", sort=True) if _overlaps(rect, tuple(w[:4]))]
    return " ".join(w[4] for w in words)


def _tiles(r: pymupdf.Rect) -> list[Rect]:
    mx, my = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    return [(r.x0, r.y0, mx, my), (mx, r.y0, r.x1, my), (r.x0, my, mx, r.y1), (mx, my, r.x1, r.y1)]


def _expand(r: Rect, m: float, bounds: pymupdf.Rect) -> Rect:
    return (
        max(bounds.x0, r[0] - m),
        max(bounds.y0, r[1] - m),
        min(bounds.x1, r[2] + m),
        min(bounds.y1, r[3] + m),
    )


def _union(a: Rect | None, b: Rect) -> Rect:
    if a is None:
        return tuple(b)  # type: ignore[return-value]
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _overlaps(a: Rect, b: Rect) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
