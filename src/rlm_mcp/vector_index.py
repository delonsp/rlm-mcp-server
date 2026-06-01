"""
Vector index for semantic search in RLM MCP Server.

Chunks text into overlapping segments, embeds them,
and provides similarity-based search.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .embeddings import get_embedding_service, _cosine_similarity, sanitize_text

logger = logging.getLogger("rlm-mcp.vector_index")

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 50  # overlap between chunks


# --- Boilerplate chunk classification (reference lists / chapter headers / page
# markers). Conservative by design: only a chunk that is PREDOMINANTLY boilerplate
# is flagged — a single stray marker inside prose never trips it. The flag is
# COMPUTED from the chunk text (in _chunk_text and on from_serializable), never
# persisted, so existing indexes pick it up on load without re-embedding. It is
# used to DOWN-WEIGHT (not drop) such chunks in semantic search via the
# RLM_BOILERPLATE_PENALTY env. See plans/20260601-filter-boilerplate-chunks.md ---

_MIN_BOILERPLATE_LINES = 3        # need at least this many non-empty lines to judge
_REFERENCE_LINE_FRACTION = 0.6    # >= this fraction of lines are strong citations
_HEADER_CONTENT_FRACTION = 0.5    # >= this fraction of lines are headers/markers

# A line counts as a reference ONLY with a STRONG citation co-signal: a DOI or a
# Vancouver-style "year;volume:page". Bare line-numbering ("1. ") and a bare
# "et al" are deliberately NOT signals — they fire on recipes, clinical protocols
# (e.g. ReCODE steps) and narrative prose that merely names authors. (Red-team
# 2026-06-01 on the real corpus: numbering-alone + et-al-alone produced 23 false
# positives, the worst error class.) Cost: misses APA/ABNT/book refs that lack a
# DOI or a year;vol:page — a tolerable false-negative, since the flag only
# down-weights a chunk's semantic score, never drops it.
_RE_CITATION_STRONG = re.compile(
    r"\bdoi:\s*10\.\d"                          # "doi:10.1038/..."
    r"|\bdoi\.org/10\.\d"                       # "https://doi.org/10.1038/..."
    r"|\b(?:19|20)\d{2}\s*;\s*\d+\s*:\s*\d",    # Vancouver "2016;8:1250"
    re.IGNORECASE,
)
# Page markers — PT extraction "--- Página 12 ---" and EN "--- page 12 ---" — and
# chapter headers. The all-caps-title rule was DROPPED: it fired on lab panels,
# code constants, staging glossaries ("ESTÁGIO I/II") and emphatic prose (9 of the
# red-team false positives). Structural headers are still caught via these markers.
_RE_PAGE_MARKER = re.compile(r"-{2,}\s*(?:p[áa]gina|page)\s*\d+", re.IGNORECASE)
_RE_CHAPTER_HEADER = re.compile(r"^\s*(?:Chapter|Cap[íi]tulo)\s+\d+", re.IGNORECASE)


def _classify_boilerplate(text: str) -> bool:
    """True if a chunk is predominantly a reference list or a header/page marker.

    Conservative: requires a DENSITY of STRONG boilerplate signals across the
    chunk's lines, never a single stray marker. Short or ambiguous chunks default
    to prose (False). The result only down-weights the chunk's semantic score, so a
    false negative just keeps current behaviour and a false positive only mildly
    demotes one chunk — never data loss. Tuned against an adversarial red-team of
    the real corpus to drive false positives ~to zero (see _RE_CITATION_STRONG).
    """
    if not text:
        return False

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < _MIN_BOILERPLATE_LINES:
        return False

    # Reference list: fraction of lines carrying a strong citation co-signal.
    ref_lines = sum(1 for ln in lines if _RE_CITATION_STRONG.search(ln))
    if ref_lines / len(lines) >= _REFERENCE_LINE_FRACTION:
        return True

    # Header/marker block: fraction of lines that are page markers / chapter headers.
    header_lines = sum(
        1 for ln in lines
        if _RE_PAGE_MARKER.search(ln) or _RE_CHAPTER_HEADER.match(ln)
    )
    if header_lines / len(lines) >= _HEADER_CONTENT_FRACTION:
        return True

    return False


def _boilerplate_penalty() -> float:
    """Score multiplier for boilerplate chunks in semantic search.

    Read from RLM_BOILERPLATE_PENALTY (default 1.0 = disabled, no effect). Values
    outside [0.0, 1.0] are rejected back to 1.0 with a warning — a multiplier > 1
    would boost boilerplate, which is never intended.
    """
    raw = os.getenv("RLM_BOILERPLATE_PENALTY", "1.0").strip()
    try:
        val = float(raw)
    except ValueError:
        logger.warning("RLM_BOILERPLATE_PENALTY=%r inválido; usando 1.0 (desligado)", raw)
        return 1.0
    if not (0.0 <= val <= 1.0):
        logger.warning("RLM_BOILERPLATE_PENALTY=%r fora de [0,1]; usando 1.0", raw)
        return 1.0
    return val


_BOILERPLATE_PENALTY = _boilerplate_penalty()


@dataclass
class ChunkInfo:
    """A text chunk with its metadata."""
    chunk_index: int
    text: str
    line_start: int
    line_end: int
    embedding: list[float] = field(default_factory=list)
    # Computed (not persisted): True for reference-list / header / page-marker
    # chunks, used to down-weight them in semantic search. See _classify_boilerplate.
    is_boilerplate: bool = False


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
            # Down-weight reference/header chunks (no-op when penalty == 1.0, the
            # default — production scoring stays byte-identical until flipped).
            if chunk.is_boilerplate and _BOILERPLATE_PENALTY != 1.0:
                score *= _BOILERPLATE_PENALTY
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
                # Recompute from text on load: persisted indexes gain the flag
                # without any schema change or re-embedding.
                is_boilerplate=_classify_boilerplate(c["text"]),
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
            is_boilerplate=_classify_boilerplate(chunk_text),
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
