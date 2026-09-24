"""SigLIP image encoder: images and text queries in one embedding space.

A text question retrieves a pressure chart directly by its pixels, with no
caption in between.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from fieldnote.encoders import l2_normalize, resolve_device


class SiglipEncoder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = batch_size
        self.dim = int(self.encode_queries(["probe"]).shape[1])

    def encode_images(self, paths: list[str]) -> np.ndarray:
        from tqdm import tqdm

        out = []
        steps = range(0, len(paths), self.batch_size)
        for i in tqdm(steps, desc="images", disable=len(paths) <= 256):
            images = [Image.open(p).convert("RGB") for p in paths[i : i + self.batch_size]]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            with self.torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            out.append(feats.float().cpu().numpy())
        return l2_normalize(np.concatenate(out)) if out else np.zeros((0, self.dim))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        # SigLIP was trained with max_length padding; other padding hurts retrieval.
        inputs = self.processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return l2_normalize(feats.float().cpu().numpy())
