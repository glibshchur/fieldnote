"""ColQwen2.5 page encoder (late interaction, ColBERT-style multi-vector).

SigLIP embeds a whole image into one vector, which works for photos but loses the fine
structure of dense technical pages (wiring diagrams, image-only tables). ColQwen2.5 embeds a
page as ~700 patch vectors and a query as one vector per token; relevance is MaxSim: for
each query token, its best-matching patch, summed. That is what lets a question about "the
PS1 and PS2 terminals" land on the one schematic that draws them.
"""

from __future__ import annotations

import torch
from PIL import Image

from fieldnote.encoders import resolve_device


class ColQwenEncoder:
    def __init__(self, model_name: str = "vidore/colqwen2.5-v0.2", device: str = "auto"):
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        self.device = resolve_device(device)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float16
        if self.device == "cpu":
            dtype = torch.float32
        self.dtype = dtype
        self.model = ColQwen2_5.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_pages(self, paths: list[str], batch_size: int = 4) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for i in range(0, len(paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
            batch = self.processor.process_images(images).to(self.device)
            embs = self.model(**batch)  # (B, n_tokens, 128), padded
            mask = batch["attention_mask"].bool()
            out.extend(e[m].to("cpu", torch.float16) for e, m in zip(embs, mask, strict=True))
        return out

    @torch.no_grad()
    def encode_pages_fp32(self, paths: list[str]) -> list[torch.Tensor]:
        """Re-encode in full precision on CPU: a few dense pages overflow fp16 on MPS."""
        model = self.model.to("cpu", torch.float32)
        try:
            out = []
            for p in paths:
                batch = self.processor.process_images([Image.open(p).convert("RGB")])
                emb = model(**batch)[0]
                mask = batch["attention_mask"][0].bool()
                out.append(emb[mask].to(torch.float16))
            return out
        finally:
            self.model = model.to(self.device, self.dtype)

    @torch.no_grad()
    def encode_queries(self, texts: list[str]) -> list[torch.Tensor]:
        batch = self.processor.process_queries(texts).to(self.device)
        embs = self.model(**batch)
        mask = batch["attention_mask"].bool()
        return [e[m].to("cpu", torch.float16) for e, m in zip(embs, mask, strict=True)]
