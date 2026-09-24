"""Runtime settings, read from environment variables with sensible local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Local secrets and overrides live in .env (gitignored); real env vars take precedence.
load_dotenv(override=False)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


DEFAULT_MODELS = {"anthropic": "claude-opus-5", "gemini": "gemini-3.8-flash"}


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
    # jina-reranker-m0 ships custom modeling code (trust_remote_code); pin the reviewed revision.
    reranker_revision: str = field(
        default_factory=lambda: _env(
            "FIELDNOTE_RERANKER_REVISION", "94bfe0aeb2d4dd7978362699cddd5893d4e0adc8"
        )
    )

    # LLM: provider "anthropic" or "gemini". Models default per provider.
    llm_provider: str = field(default_factory=lambda: _env("FIELDNOTE_LLM_PROVIDER", "anthropic"))
    answer_model: str = field(default_factory=lambda: _env("FIELDNOTE_ANSWER_MODEL", ""))
    answer_effort: str = field(default_factory=lambda: _env("FIELDNOTE_ANSWER_EFFORT", "low"))
    caption_model: str = field(default_factory=lambda: _env("FIELDNOTE_CAPTION_MODEL", ""))
    judge_model: str = field(default_factory=lambda: _env("FIELDNOTE_JUDGE_MODEL", ""))

    def __post_init__(self) -> None:
        default = DEFAULT_MODELS.get(self.llm_provider, "")
        for name in ("answer_model", "caption_model", "judge_model"):
            if not getattr(self, name):
                object.__setattr__(self, name, default)

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
