"""Qdrant store with one named vector per embedding space.

  * "text"  vector: manual text chunks and call transcript windows
  * "image" vector: manual figures, page renders and job-site photos

Every point carries a `modality` and `source_type` payload so each retriever
searches only its own slice of the index.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

import numpy as np
from qdrant_client import QdrantClient, models

from fieldnote.types import Chunk

TEXT_VEC = "text"
IMAGE_VEC = "image"


def make_client(path: str | None = None, url: str | None = None) -> QdrantClient:
    if url:
        return QdrantClient(url=url)
    if path in (None, ":memory:"):
        return QdrantClient(location=":memory:")
    return QdrantClient(path=path)


class ChunkStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    def recreate(self, text_dim: int, image_dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={
                TEXT_VEC: models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
                IMAGE_VEC: models.VectorParams(size=image_dim, distance=models.Distance.COSINE),
            },
        )
        for key in ("modality", "source_type", "source"):
            try:
                self.client.create_payload_index(
                    self.collection, key, models.PayloadSchemaType.KEYWORD
                )
            except Exception:  # local mode has no payload indexes; filters still work
                pass

    def upsert(
        self, chunks: list[Chunk], vector_name: str, vectors: np.ndarray, batch: int = 256
    ) -> None:
        assert len(chunks) == len(vectors)
        for i in range(0, len(chunks), batch):
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                        vector={vector_name: v.tolist()},
                        payload=c.to_payload(),
                    )
                    for c, v in zip(chunks[i : i + batch], vectors[i : i + batch], strict=True)
                ],
            )

    def search(
        self,
        vector_name: str,
        query: np.ndarray,
        k: int,
        modalities: Iterable[str] | None = None,
        source_types: Iterable[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        must = []
        if modalities:
            must.append(
                models.FieldCondition(key="modality", match=models.MatchAny(any=list(modalities)))
            )
        if source_types:
            must.append(
                models.FieldCondition(
                    key="source_type", match=models.MatchAny(any=list(source_types))
                )
            )
        res = self.client.query_points(
            self.collection,
            query=query.tolist(),
            using=vector_name,
            limit=k,
            query_filter=models.Filter(must=must) if must else None,
            with_payload=True,
        )
        return [(Chunk.from_payload(p.payload), float(p.score)) for p in res.points]

    def count(self) -> int:
        return self.client.count(self.collection).count
