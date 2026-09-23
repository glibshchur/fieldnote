"""Runtime settings, read from environment variables with sensible local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("FIELDNOTE_DATA_DIR", "data")))
    artifacts_dir: Path = field(
        default_factory=lambda: Path(_env("FIELDNOTE_ARTIFACTS_DIR", "artifacts"))
    )

    # Qdrant: local on-disk mode by default, a server when QDRANT_URL is set.
    qdrant_path: str = field(default_factory=lambda: _env("FIELDNOTE_QDRANT_PATH", ".qdrant"))
    qdrant_url: str | None = field(default_factory=lambda: os.environ.get("QDRANT_URL"))
    collection: str = field(default_factory=lambda: _env("FIELDNOTE_COLLECTION", "fieldnote"))

    # Encoders
    text_model: str = field(
        default_factory=lambda: _env("FIELDNOTE_TEXT_MODEL", "BAAI/bge-base-en-v1.5")
    )
    image_model: str = field(
        default_factory=lambda: _env("FIELDNOTE_IMAGE_MODEL", "google/siglip-so400m-patch14-384")
    )
    whisper_model: str = field(default_factory=lambda: _env("FIELDNOTE_WHISPER_MODEL", "small"))
    device: str = field(default_factory=lambda: _env("FIELDNOTE_DEVICE", "auto"))

    # Retrieval: k per modality sums to the total candidate pool (k=20 in the ship gate).
    k_text: int = field(default_factory=lambda: int(_env("FIELDNOTE_K_TEXT", "8")))
    k_image: int = field(default_factory=lambda: int(_env("FIELDNOTE_K_IMAGE", "8")))
    k_audio: int = field(default_factory=lambda: int(_env("FIELDNOTE_K_AUDIO", "4")))
    top_n: int = field(default_factory=lambda: int(_env("FIELDNOTE_TOP_N", "5")))
    reranker: str = field(default_factory=lambda: _env("FIELDNOTE_RERANKER", "jina"))
    reranker_model: str = field(
        default_factory=lambda: _env("FIELDNOTE_RERANKER_MODEL", "jinaai/jina-reranker-m0")
    )

    # LLM
    answer_model: str = field(
        default_factory=lambda: _env("FIELDNOTE_ANSWER_MODEL", "claude-opus-5")
    )
    answer_effort: str = field(default_factory=lambda: _env("FIELDNOTE_ANSWER_EFFORT", "low"))
    caption_model: str = field(
        default_factory=lambda: _env("FIELDNOTE_CAPTION_MODEL", "claude-opus-5")
    )
    judge_model: str = field(default_factory=lambda: _env("FIELDNOTE_JUDGE_MODEL", "claude-opus-5"))

    @property
    def k_total(self) -> int:
        return self.k_text + self.k_image + self.k_audio

    @property
    def crops_dir(self) -> Path:
        return self.artifacts_dir / "crops"

    @property
    def pages_dir(self) -> Path:
        return self.artifacts_dir / "pages"

    @property
    def photos_dir(self) -> Path:
        return self.artifacts_dir / "photos"


def get_settings() -> Settings:
    return Settings()
