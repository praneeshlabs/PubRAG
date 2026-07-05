"""
app/core/config.py

Centralised application configuration.

Loads all environment variables at startup, provides typed defaults,
and raises a clear error when required secrets are missing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env file before anything else reads os.environ
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Single source of truth for every tuneable parameter in the system.

    Mandatory environment variables
    ──────────────────────────────
    ANTHROPIC_API_KEY : Anthropic API key for Claude calls.
    NCBI_EMAIL        : Email registered with NCBI for Entrez usage policy.
    """

    # ── Secrets (loaded from environment) ──────────────────────────────────
    ANTHROPIC_API_KEY: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    NCBI_EMAIL: str = field(
        default_factory=lambda: os.getenv("NCBI_EMAIL", "researcher@example.com")
    )

    # LLM settings:
    LLM_MODEL: str = "claude-opus-4-6"
    MAX_TOKENS: int = 2048

    # Embedding settings:
    # NeuML/pubmedbert-base-embeddings is trained on PubMed abstracts and
    # substantially outperforms general-purpose embeddings on biomedical tasks.
    EMBEDDING_MODEL: str = "NeuML/pubmedbert-base-embeddings"

    # Vector store:
    COLLECTION_NAME: str = "pubmed_papers"

    # Retrieval defaults
    DEFAULT_NUM_PAPERS: int = 20   # Papers fetched from PubMed
    DEFAULT_TOP_K: int = 5         # Candidates retrieved from vector store
    DEFAULT_RERANK_TOP_N: int = 3  # Documents kept after FlashRank reranking

    # Chunking 
    CHUNK_SIZE: int = 768
    CHUNK_OVERLAP: int = 64

    # Entrez / NCBI 
    ENTREZ_RATE_LIMIT_SLEEP: float = 0.2  # seconds between API calls

    def __post_init__(self) -> None:
        """Validate required fields and log effective configuration."""
        self._validate()
        self._log_config()

    def _validate(self) -> None:
        """Raise ValueError if required secrets are absent."""
        missing: list[str] = []
        if not self.ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY")
        if not self.NCBI_EMAIL or self.NCBI_EMAIL == "researcher@example.com":
            logger.warning(
                "NCBI_EMAIL is unset or default. "
                "NCBI requires a real email for Entrez access. "
                "Set NCBI_EMAIL in your .env file."
            )
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env and fill in the values."
            )

    def _log_config(self) -> None:
        """Log non-sensitive configuration at startup."""
        logger.info(
            "Config loaded | model=%s | embedding=%s | chunk_size=%d",
            self.LLM_MODEL,
            self.EMBEDDING_MODEL,
            self.CHUNK_SIZE,
        )


# Module-level singleton — import this directly in other modules.
# Using a factory makes it easy to override in tests.
def get_config() -> Config:
    """Return a validated Config instance."""
    return Config()
