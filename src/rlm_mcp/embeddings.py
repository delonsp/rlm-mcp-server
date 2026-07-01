"""
Embedding service for RLM MCP Server.

Supports multiple backends:
- openai: Uses OpenAI Embeddings API (default, requires OPENAI_API_KEY)
- disabled: No embeddings (keyword search only)

Provides cosine similarity for vector comparison.
"""

import os
import re
import logging
import math
from typing import Optional

logger = logging.getLogger("rlm-mcp.embeddings")

# Default embedding model for OpenAI
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
# Dimension for text-embedding-3-small
DEFAULT_DIMENSION = 1536

# Caracteres de controle C0 exceto \t \n \r — lixo de extração de PDF (páginas-
# imagem cujos bytes vazam no "texto"). A OpenAI Embeddings API REJEITA strings
# com NUL (\x00): sem remover, o embed do chunk inteiro falha → o índice vetorial
# não constrói → busca semântica/hybrid cai p/ keyword silenciosamente.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_text(text: str) -> str:
    """Remove NUL bytes e caracteres de controle não-imprimíveis (preserva \\t \\n \\r)."""
    if not text:
        return text
    return _CONTROL_CHARS_RE.sub("", text)


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
        # Última falha de API (não token-cap). Setada em _embed_call, limpa no
        # início de embed_query — permite ao search distinguir "0 resultados" de
        # "API caiu" em vez de mostrar "No results" durante um outage.
        self.last_error: Optional[str] = None

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

        # Zera o marcador antes: se ficar setado depois, foi ESTA query que falhou.
        self.last_error = None
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
        truncated = [sanitize_text(t)[:8000] for t in texts]

        all_embeddings: list[list[float]] = []
        for batch in _pack_batches(truncated):
            all_embeddings.extend(self._embed_call(batch))
        return all_embeddings

    def _embed_call(self, batch: list[str]) -> list[list[float]]:
        """Uma chamada à API; se estourar o cap de tokens, divide e re-tenta.

        A estimativa char→token de _pack_batches é heurística — se ainda assim
        um lote real passar de 300k tokens (conteúdo que tokeniza pior que o
        previsto), a API devolve 400 max_tokens_per_request. Dividir ao meio e
        recursar converge para lotes válidos em O(log n) chamadas extras, para
        QUALQUER razão chars/token (bug 2026-06-06: lotes de 1953 chunks do
        ReCODE chegavam com 380-409k tokens reais e falhavam TODOS — vars
        ficavam sem embedding ou com cobertura parcial silenciosa).
        """
        try:
            response = self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            # Ordena por .index: a API não garante que data venha na ordem
            # do input, e o resultado é casado posicionalmente com os chunks.
            return [item.embedding
                    for item in sorted(response.data, key=lambda d: d.index)]
        except Exception as e:
            msg = str(e)
            token_cap = ("max_tokens_per_request" in msg
                         or "maximum request size" in msg)
            if token_cap and len(batch) > 1:
                mid = len(batch) // 2
                logger.warning(
                    f"Batch de {len(batch)} estourou o cap de tokens da API; "
                    f"dividindo em {mid}+{len(batch) - mid} e re-tentando"
                )
                return self._embed_call(batch[:mid]) + self._embed_call(batch[mid:])
            # Falha de UM sub-batch não descarta os demais: preenche vazios
            # só para este batch (mantém alinhamento) e segue. Antes, qualquer
            # falha jogava fora todos os embeddings já computados.
            logger.error(f"OpenAI embedding error (batch de {len(batch)}): {e}")
            # Marca a falha p/ o search distinguir outage de "0 resultados".
            self.last_error = msg
            return [[] for _ in batch]


def _pack_batches(texts: list[str], max_batch: int = 2048,
                  max_tokens: int = 250_000) -> list[list[str]]:
    """Agrupa textos em lotes sob os limites da API (contagem E tokens).

    Estimativa char→token: len//2 (2 chars/token). Conservadora de propósito:
    PT com acentos + termos médicos + resíduo de extração de PDF tokenizam a
    ~2,4-2,6 chars/token (medido live 2026-06-06) — a estimativa antiga (//4)
    subdimensionava e o lote real passava de 300k tokens. O custo do
    conservadorismo é só mais chamadas HTTP; o _embed_call ainda cobre o caso
    de conteúdo patológico via split-retry.
    """
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for t in texts:
        n = max(1, len(t) // 2)
        if cur and (len(cur) >= max_batch or cur_tokens + n > max_tokens):
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(t)
        cur_tokens += n
    if cur:
        batches.append(cur)
    return batches


# Singleton
_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton EmbeddingService instance."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
