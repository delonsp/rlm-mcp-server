"""
Vector index for semantic search in RLM MCP Server.

Chunks text into overlapping segments, embeds them,
and provides similarity-based search.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .embeddings import get_embedding_service, _cosine_similarity, sanitize_text

logger = logging.getLogger("rlm-mcp.vector_index")

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 50  # overlap between chunks


@dataclass
class ChunkInfo:
    """A text chunk with its metadata."""
    chunk_index: int
    text: str
    line_start: int
    line_end: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class VectorSearchResult:
    """A single search result from vector search."""
    chunk_text: str
    line_start: int
    line_end: int
    score: float
    chunk_index: int


class VectorIndex:
    """Vector index for a single variable's text content.

    Chunks text, embeds chunks, and supports similarity search.
    """

    def __init__(self, var_name: str):
        self.var_name = var_name
        self.chunks: list[ChunkInfo] = []
        self.total_chars: int = 0
        self.total_lines: int = 0

    def build(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> bool:
        """Build the vector index for the given text.

        Args:
            text: Full text to index
            chunk_size: Characters per chunk
            overlap: Overlap between chunks

        Returns:
            True if successfully built with embeddings
        """
        if not text:
            return False

        # Saneia NUL/controle (lixo de extração de PDF) ANTES de chunkar: chunks
        # ficam limpos no índice (display da busca semântica) e o input ao embed
        # é válido (a OpenAI rejeita NUL). Ver embeddings.sanitize_text.
        text = sanitize_text(text)

        self.total_chars = len(text)
        self.total_lines = text.count('\n') + 1

        # Create chunks
        self.chunks = _chunk_text(text, chunk_size, overlap)

        if not self.chunks:
            return False

        # Get embeddings
        service = get_embedding_service()
        if not service.is_available:
            logger.warning(f"Embeddings disabled, vector index for '{self.var_name}' has no vectors")
            return False

        chunk_texts = [c.text for c in self.chunks]
        embeddings = service.embed_texts(chunk_texts)

        if len(embeddings) != len(self.chunks):
            logger.error(f"Embedding count mismatch: {len(embeddings)} vs {len(self.chunks)} chunks")
            return False

        for chunk, emb in zip(self.chunks, embeddings):
            chunk.embedding = emb

        # Filter out chunks with empty embeddings
        valid_chunks = [c for c in self.chunks if c.embedding]
        if not valid_chunks:
            logger.warning(f"No valid embeddings for '{self.var_name}'")
            return False

        logger.info(f"Vector index built for '{self.var_name}': {len(valid_chunks)}/{len(self.chunks)} chunks embedded")
        return True

    def search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        """Search the index for similar chunks.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of VectorSearchResult sorted by similarity (descending)
        """
        service = get_embedding_service()
        if not service.is_available:
            return []

        query_embedding = service.embed_query(query)
        if not query_embedding:
            return []

        # Compute similarities
        scored = []
        for chunk in self.chunks:
            if not chunk.embedding:
                continue
            score = _cosine_similarity(query_embedding, chunk.embedding)
            scored.append((chunk, score))

        # Sort by score descending
        scored.sort(key=lambda x: -x[1])

        # Return top_k
        results = []
        for chunk, score in scored[:top_k]:
            results.append(VectorSearchResult(
                chunk_text=chunk.text,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                score=score,
                chunk_index=chunk.chunk_index,
            ))

        return results

    def get_stats(self) -> dict:
        """Return index statistics."""
        embedded = sum(1 for c in self.chunks if c.embedding)
        return {
            "var_name": self.var_name,
            "total_chunks": len(self.chunks),
            "embedded_chunks": embedded,
            "total_chars": self.total_chars,
            "total_lines": self.total_lines,
        }

    def to_serializable(self) -> dict:
        """Convert to serializable dict for persistence."""
        return {
            "var_name": self.var_name,
            "total_chars": self.total_chars,
            "total_lines": self.total_lines,
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "embedding": c.embedding,
                }
                for c in self.chunks
            ],
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "VectorIndex":
        """Reconstruct from serialized dict."""
        vi = cls(var_name=data["var_name"])
        vi.total_chars = data.get("total_chars", 0)
        vi.total_lines = data.get("total_lines", 0)
        vi.chunks = [
            ChunkInfo(
                chunk_index=c["chunk_index"],
                text=c["text"],
                line_start=c["line_start"],
                line_end=c["line_end"],
                embedding=c.get("embedding", []),
            )
            for c in data.get("chunks", [])
        ]
        return vi


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkInfo]:
    """Split text into overlapping chunks with line number tracking.

    Args:
        text: Text to chunk
        chunk_size: Target characters per chunk
        overlap: Characters of overlap between consecutive chunks

    Returns:
        List of ChunkInfo objects
    """
    if not text:
        return []

    chunks = []
    lines = text.split('\n')

    # Build a char-offset to line-number map
    line_offsets = []  # (start_char, line_num)
    char_pos = 0
    for i, line in enumerate(lines):
        line_offsets.append((char_pos, i))
        char_pos += len(line) + 1  # +1 for \n

    def _char_to_line(pos: int) -> int:
        """Find line number for a character position."""
        for j in range(len(line_offsets) - 1, -1, -1):
            if pos >= line_offsets[j][0]:
                return line_offsets[j][1]
        return 0

    # Create chunks with overlap
    step = max(1, chunk_size - overlap)
    chunk_idx = 0

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        if not chunk_text.strip():
            continue

        line_start = _char_to_line(start)
        line_end = _char_to_line(end - 1) if end > start else line_start

        chunks.append(ChunkInfo(
            chunk_index=chunk_idx,
            text=chunk_text,
            line_start=line_start,
            line_end=line_end,
        ))
        chunk_idx += 1

        if end >= len(text):
            break

    return chunks


# Cache of vector indices in memory
_vector_indices: dict[str, VectorIndex] = {}


def get_vector_index(var_name: str) -> Optional[VectorIndex]:
    """Get vector index from cache."""
    return _vector_indices.get(var_name)


def set_vector_index(var_name: str, index: VectorIndex):
    """Save vector index to cache."""
    _vector_indices[var_name] = index


def clear_vector_index(var_name: str):
    """Remove vector index from cache."""
    _vector_indices.pop(var_name, None)


def clear_all_vector_indices():
    """Clear all vector indices from cache."""
    _vector_indices.clear()
