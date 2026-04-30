"""
Embedding service for RLM MCP Server.

Supports multiple backends:
- openai: Uses OpenAI Embeddings API (default, requires OPENAI_API_KEY)
- disabled: No embeddings (keyword search only)

Provides cosine similarity for vector comparison.
"""

import os
import logging
import math
from typing import Optional

logger = logging.getLogger("rlm-mcp.embeddings")

# Default embedding model for OpenAI
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
# Dimension for text-embedding-3-small
DEFAULT_DIMENSION = 1536


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses pure Python (no numpy required at runtime).
    """
    if len(v1) != len(v2):
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


class EmbeddingService:
    """Manages text embeddings with pluggable backends.

    Modes:
    - "openai": Uses OpenAI API (text-embedding-3-small by default)
    - "disabled": Returns empty embeddings, semantic search disabled
    """

    def __init__(self, mode: Optional[str] = None):
        self.mode = (mode or os.getenv("RLM_EMBEDDING_MODE", "openai")).lower()
        self._client = None
        self._model = os.getenv("RLM_EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL)
        self._dimension = DEFAULT_DIMENSION

        if self.mode == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set, falling back to disabled mode")
                self.mode = "disabled"
            else:
                try:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=api_key)
                    logger.info(f"EmbeddingService initialized: openai ({self._model})")
                except ImportError:
                    logger.warning("openai package not available, falling back to disabled")
                    self.mode = "disabled"

        if self.mode == "disabled":
            logger.info("EmbeddingService initialized: disabled")

    @property
    def is_available(self) -> bool:
        """Whether embeddings are available."""
        return self.mode != "disabled"

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        if not texts:
            return []

        if self.mode == "disabled":
            return [[] for _ in texts]

        if self.mode == "openai":
            return self._embed_openai(texts)

        return [[] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        if not query or self.mode == "disabled":
            return []

        results = self.embed_texts([query])
        return results[0] if results else []

    def similarity(self, v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        return _cosine_similarity(v1, v2)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API.

        Batches by both count (max 2048 texts/call) and token budget
        (max 250k tokens/call, leaving margin under the 300k API cap).
        """
        if not self._client:
            return [[] for _ in texts]

        # Truncate individual texts up front (~8000 chars ≈ 2000 tokens)
        truncated = [t[:8000] for t in texts]

        # Pack into sub-batches that respect both limits
        MAX_BATCH = 2048
        MAX_TOKENS = 250_000
        sub_batches: list[list[str]] = []
        cur: list[str] = []
        cur_tokens = 0
        for t in truncated:
            n = max(1, len(t) // 4)  # rough char→token estimate
            if cur and (len(cur) >= MAX_BATCH or cur_tokens + n > MAX_TOKENS):
                sub_batches.append(cur)
                cur, cur_tokens = [], 0
            cur.append(t)
            cur_tokens += n
        if cur:
            sub_batches.append(cur)

        all_embeddings: list[list[float]] = []
        try:
            for batch in sub_batches:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                )
                for item in response.data:
                    all_embeddings.append(item.embedding)
            return all_embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return [[] for _ in texts]


# Singleton
_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton EmbeddingService instance."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
