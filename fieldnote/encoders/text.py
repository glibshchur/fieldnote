"""Text encoder (sentence-transformers). Used for manual text and call transcripts."""

from __future__ import annotations

import numpy as np

from fieldnote.encoders import resolve_device

# BGE English models retrieve better when queries carry this instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=resolve_device(device))
        get_dim = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )
        self.dim = get_dim()
        self.batch_size = batch_size
        name = model_name.lower()
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in name and "m3" not in name else ""

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            [self.query_prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
