"""Late fusion, step 3: rerank the merged candidate pool on one scale.

A text-only cross-encoder cannot score a chart image, so the default reranker
is multimodal (jina-reranker-m0 scores query-text and query-image pairs with
one model). RRF order is the fallback and the "rerank off" ablation.
"""

from __future__ import annotations

from typing import Protocol

from fieldnote.types import Candidate


class Reranker(Protocol):
    def rerank(self, question: str, candidates: list[Candidate]) -> list[Candidate]: ...


class RRFOrder:
    """No model: keep the fused order."""

    def rerank(self, question: str, candidates: list[Candidate]) -> list[Candidate]:
        return sorted(candidates, key=lambda c: c.fused_score, reverse=True)


class JinaM0Reranker:
    def __init__(self, model_name: str = "jinaai/jina-reranker-m0", device: str = "auto"):
        import torch
        from transformers import AutoModel

        from fieldnote.encoders import resolve_device

        dev = resolve_device(device)
        dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
        self.model = (
            AutoModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
            .to(dev)
            .eval()
        )

    def rerank(self, question: str, candidates: list[Candidate]) -> list[Candidate]:
        images = [c for c in candidates if c.chunk.modality == "image" and c.chunk.image_path]
        texts = [c for c in candidates if c not in images]
        if texts:
            scores = self.model.compute_score(
                [[question, c.chunk.text] for c in texts], max_length=1024, doc_type="text"
            )
            for c, s in zip(texts, _as_list(scores), strict=True):
                c.rerank_score = float(s)
        if images:
            scores = self.model.compute_score(
                [[question, c.chunk.image_path] for c in images], max_length=2048, doc_type="image"
            )
            for c, s in zip(images, _as_list(scores), strict=True):
                c.rerank_score = float(s)
        return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)


def _as_list(scores) -> list[float]:
    return [scores] if isinstance(scores, float) else list(scores)


def make_reranker(kind: str, model_name: str, device: str = "auto") -> Reranker:
    if kind == "jina":
        return JinaM0Reranker(model_name, device)
    if kind in ("rrf", "none"):
        return RRFOrder()
    raise ValueError(f"unknown reranker: {kind}")
