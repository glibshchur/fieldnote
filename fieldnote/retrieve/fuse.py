"""Late fusion, step 2: merge per-modality result lists.

Raw similarity scores from different encoders are not comparable (SigLIP
cosine vs BGE cosine live on different scales), so lists are merged by rank
with reciprocal rank fusion. The reranker then scores the merged pool on a
single scale.
"""

from __future__ import annotations

from fieldnote.types import Candidate

RRF_K = 60


def rrf_fuse(lists: dict[str, list[Candidate]], k: int = RRF_K) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for candidates in lists.values():
        for c in candidates:
            contribution = 1.0 / (k + c.rank)
            if c.chunk.id in merged:
                merged[c.chunk.id].fused_score += contribution
            else:
                c.fused_score = contribution
                merged[c.chunk.id] = c
    return sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)
