"""Job-site photo ingestion.

Photos are embedded as images. Optional technician notes from
data/photos/metadata.csv (columns: file, notes) are stored as text so they can
be shown in citations, but retrieval of photos runs on the image vector.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageOps

from fieldnote.types import Chunk

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SIDE = 1600


def load_notes(photos_dir: Path) -> dict[str, str]:
    meta = photos_dir / "metadata.csv"
    if not meta.exists():
        return {}
    with meta.open(newline="") as f:
        return {row["file"]: row.get("notes", "") for row in csv.DictReader(f)}


def ingest_photos(photos_dir: Path, out_dir: Path) -> list[Chunk]:
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = load_notes(photos_dir)
    chunks: list[Chunk] = []
    for path in sorted(photos_dir.iterdir()):
        if path.suffix.lower() not in PHOTO_EXTS:
            continue
        out = out_dir / f"{path.stem}.jpg"
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            im.save(out, quality=88)
        chunks.append(
            Chunk(
                modality="image",
                source_type="photo",
                source=path.name,
                text=notes.get(path.name, ""),
                image_path=str(out),
            )
        )
    return chunks
