"""Indexing: embed chunks with the encoder for their modality and store them."""

from __future__ import annotations

from fieldnote.encoders import ImageEncoder, TextEncoder
from fieldnote.index.store import IMAGE_VEC, TEXT_VEC, ChunkStore
from fieldnote.types import Chunk


def build_index(
    store: ChunkStore,
    chunks: list[Chunk],
    text_encoder: TextEncoder,
    image_encoder: ImageEncoder,
) -> dict[str, int]:
    """Native multimodal index: text/audio chunks on the text vector, images on the image vector."""
    store.recreate(text_encoder.dim, image_encoder.dim)
    text_chunks = [c for c in chunks if c.modality in ("text", "audio")]
    image_chunks = [c for c in chunks if c.modality == "image"]
    if text_chunks:
        vecs = text_encoder.encode_documents([c.text for c in text_chunks])
        store.upsert(text_chunks, TEXT_VEC, vecs)
    if image_chunks:
        vecs = image_encoder.encode_images([c.image_path for c in image_chunks])
        store.upsert(image_chunks, IMAGE_VEC, vecs)
    return {"text": len(text_chunks), "image": len(image_chunks)}


__all__ = ["ChunkStore", "build_index", "TEXT_VEC", "IMAGE_VEC"]
