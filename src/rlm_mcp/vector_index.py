"""
Vector index for semantic search in RLM MCP Server.

Chunks text into overlapping segments, embeds them,
and provides similarity-based search.
"""

import bisect
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from .embeddings import get_embedding_service, _cosine_similarity, sanitize_text

logger = logging.getLogger("rlm-mcp.vector_index")

# numpy é opcional por design (degrada para o caminho Python puro se ausente),
# mas em produção é o que dá a economia de ~8x de RAM: os embeddings ficam numa
# matriz float32 contígua em vez de N listas de floats boxed do CPython.
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - produção sempre tem numpy
    np = None
    _HAS_NUMPY = False
    logger.warning(
        "numpy ausente: índice vetorial cai no fallback list[float] (RAM ~8x maior, "
        "busca mais lenta). Instale numpy para o caminho otimizado."
    )

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
_MIN_CITATION_TOKENS = 3          # >= this many strong citation tokens => reference list
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

    # Reference list. Two complementary signals, both immune to the v1 false
    # positives (recipes / clinical protocols / "et al" prose never carry these
    # tokens):
    #  (a) an ABSOLUTE count of strong citation tokens — catches Vancouver lists
    #      whose entries WRAP across lines, so the "year;vol:page" token lands on
    #      only ~half the lines and the per-line ratio stays under threshold (live
    #      finding 2026-06-01: real ref chunks L10626/L11566 were missed by ratio);
    #  (b) the per-line ratio — catches short blocks of one-line entries.
    ref_token_count = sum(len(_RE_CITATION_STRONG.findall(ln)) for ln in lines)
    ref_lines = sum(1 for ln in lines if _RE_CITATION_STRONG.search(ln))
    if ref_token_count >= _MIN_CITATION_TOKENS or ref_lines / len(lines) >= _REFERENCE_LINE_FRACTION:
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
    """A text chunk with its metadata.

    O embedding NÃO mora aqui: vive em VectorIndex como uma única matriz
    float32 contígua (ver VectorIndex._ingest_embeddings). Guardar 1536 floats
    boxed do Python por chunk custava ~8x a RAM de uma linha float32 e era o
    dreno de memória #1 do servidor.
    """
    chunk_index: int
    text: str
    line_start: int
    line_end: int
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
        # Store de embeddings, alinhado 1:1 com self.chunks.
        #  - numpy: matriz float32 contígua (N, dim) L2-normalizada; linhas de
        #    chunks sem embedding ficam todo-zero.
        #  - fallback (sem numpy): lista de vetores crus por chunk ([] = ausente).
        self._matrix = None                                  # Optional[np.ndarray]
        self._vectors: Optional[list[list[float]]] = None
        self._has_vec: list[bool] = []
        self._dim: int = 0
        # Fingerprint do texto-fonte (detecta rebind → invalida cache stale).
        # Vazio = desconhecido. Ver indexer.fingerprint_source.
        self.source_fingerprint: str = ""

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

        # Fingerprint do texto CRU (antes de sanitizar) — a checagem de staleness
        # no search compara contra repl.variables[var], que também é cru.
        from .indexer import fingerprint_source
        self.source_fingerprint = fingerprint_source(text)

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

        self._ingest_embeddings(embeddings)

        embedded = sum(self._has_vec)
        if embedded == 0:
            logger.warning(f"No valid embeddings for '{self.var_name}'")
            return False

        logger.info(f"Vector index built for '{self.var_name}': {embedded}/{len(self.chunks)} chunks embedded")
        return True

    def _ingest_embeddings(self, vectors: list[list[float]]) -> None:
        """Funde vetores por-chunk (alinhados 1:1 com self.chunks) no store.

        Caminho numpy: uma matriz float32 contígua L2-normalizada — ~8x menos
        RAM que N listas de floats boxed, e a busca vira um produto matriz-vetor.
        Sem numpy, mantém as listas cruas e usa o loop de cosseno em Python puro
        (mais lento, mas nunca quebra). Cosseno é invariante a escala e os
        embeddings da OpenAI já são unit-norm, então normalizar aqui não muda
        ranking nenhum.
        """
        n = len(self.chunks)
        # Alinhamento defensivo: a contagem de chunks é a fonte da verdade.
        if len(vectors) != n:
            vectors = (list(vectors) + [[]] * n)[:n]
        self._has_vec = [bool(v) for v in vectors]
        self._dim = next((len(v) for v in vectors if v), 0)

        if not _HAS_NUMPY:
            self._vectors = [list(v) if v else [] for v in vectors]
            self._matrix = None
            return

        self._vectors = None
        if self._dim == 0:
            self._matrix = None
            return
        mat = np.zeros((n, self._dim), dtype=np.float32)
        for i, v in enumerate(vectors):
            if v and len(v) == self._dim:
                mat[i] = v
        # L2-normaliza as linhas in-place; linhas todo-zero continuam zero.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        np.divide(mat, norms, out=mat, where=norms > 0)
        self._matrix = mat

    def _vector_at(self, i: int) -> list[float]:
        """Vetor cru do chunk i como list[float] ([] se ausente). Para persistência."""
        if i < 0 or i >= len(self._has_vec) or not self._has_vec[i]:
            return []
        if self._matrix is not None:
            return self._matrix[i].astype(float).tolist()
        if self._vectors is not None:
            return list(self._vectors[i])
        return []

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

        n_vec = sum(self._has_vec)
        if n_vec == 0:
            return []
        # Down-weight de chunks de referência/cabeçalho: no-op quando == 1.0 (o
        # default — scoring de produção fica idêntico até flipar).
        penalize = _BOILERPLATE_PENALTY != 1.0

        # --- Caminho numpy: cosseno = matriz_normalizada @ query_normalizada ---
        if self._matrix is not None:
            q = np.asarray(query_embedding, dtype=np.float32)
            if q.shape[0] != self._matrix.shape[1]:
                return []
            qn = float(np.linalg.norm(q))
            if qn == 0.0:
                return []
            scores = self._matrix @ (q / qn)
            if penalize:
                pen = np.fromiter(
                    (_BOILERPLATE_PENALTY if c.is_boilerplate else 1.0 for c in self.chunks),
                    dtype=np.float32, count=len(self.chunks),
                )
                scores = scores * pen
            # Exclui chunks sem vetor real (linha zero pontuaria 0): -inf nunca rankeia.
            has = np.fromiter(self._has_vec, dtype=bool, count=len(self._has_vec))
            scores = np.where(has, scores, -np.inf)
            k = min(top_k, n_vec)
            order = np.argsort(-scores, kind="stable")[:k]
            results = []
            for idx in order:
                i = int(idx)
                c = self.chunks[i]
                results.append(VectorSearchResult(
                    chunk_text=c.text,
                    line_start=c.line_start,
                    line_end=c.line_end,
                    score=float(scores[i]),
                    chunk_index=c.chunk_index,
                ))
            return results

        # --- Fallback Python puro: loop de cosseno sobre os vetores crus ---
        scored = []
        for i, vec in enumerate(self._vectors or []):
            if not vec:
                continue
            score = _cosine_similarity(query_embedding, vec)
            if penalize and self.chunks[i].is_boilerplate:
                score *= _BOILERPLATE_PENALTY
            scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        results = []
        for i, score in scored[:top_k]:
            c = self.chunks[i]
            results.append(VectorSearchResult(
                chunk_text=c.text,
                line_start=c.line_start,
                line_end=c.line_end,
                score=score,
                chunk_index=c.chunk_index,
            ))
        return results

    def get_stats(self) -> dict:
        """Return index statistics."""
        embedded = sum(self._has_vec)
        return {
            "var_name": self.var_name,
            "total_chunks": len(self.chunks),
            "embedded_chunks": embedded,
            "total_chars": self.total_chars,
            "total_lines": self.total_lines,
        }

    def persist_payload(self) -> list[dict]:
        """Linhas para persistence.save_embeddings().

        Persiste TODOS os chunks — os sem vetor entram com embedding vazio. Antes
        só os chunks com vetor eram salvos: no restart, from_persisted reconstruía
        `total_chunks` apenas a partir das linhas salvas, então `embedded == total`
        e o warning `emb:X/Y ⚠️parcial` (feito justamente p/ pegar essa classe de
        bug) sumia — um build parcial (falha de rede num batch) virava "100%"
        permanente e invisível. Chunk sem vetor custa ~um texto no DB (sem blob de
        floats) e restaura a observabilidade + o auto-rebuild de ensure_embeddings.
        """
        payload = []
        for i, c in enumerate(self.chunks):
            has_vec = i < len(self._has_vec) and self._has_vec[i]
            payload.append({
                "chunk_index": c.chunk_index,
                "chunk_text": c.text,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "embedding": self._vector_at(i) if has_vec else [],
            })
        return payload

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
                    "embedding": self._vector_at(i),
                }
                for i, c in enumerate(self.chunks)
            ],
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "VectorIndex":
        """Reconstruct from serialized dict."""
        vi = cls(var_name=data["var_name"])
        vi.total_chars = data.get("total_chars", 0)
        vi.total_lines = data.get("total_lines", 0)
        chunk_dicts = data.get("chunks", [])
        vi.chunks = [
            ChunkInfo(
                chunk_index=c["chunk_index"],
                text=c["text"],
                line_start=c["line_start"],
                line_end=c["line_end"],
                # Recompute from text on load: persisted indexes gain the flag
                # without any schema change or re-embedding.
                is_boilerplate=_classify_boilerplate(c["text"]),
            )
            for c in chunk_dicts
        ]
        vi._ingest_embeddings([c.get("embedding", []) for c in chunk_dicts])
        return vi

    @classmethod
    def from_persisted(cls, var_name: str, text_value, loaded_chunks: list[dict]) -> "VectorIndex":
        """Reconstrói a partir das linhas de persistence.load_embeddings().

        Este é o caminho REAL de restore no startup (from_serializable não roda
        em runtime). loaded_chunks trazem 'chunk_text' (nome da coluna do DB) e
        um 'embedding' cru; os embeddings são fundidos na matriz float32 compacta
        e as listas por-linha são descartadas — o pico de RAM no boot deixa de
        ser O(N listas de floats) e passa a O(matriz).
        """
        vi = cls(var_name=var_name)
        vi.total_chars = len(text_value) if isinstance(text_value, str) else 0
        vi.total_lines = text_value.count('\n') + 1 if isinstance(text_value, str) else 0
        # Seed do fingerprint com o texto restaurado — assim uma busca logo após
        # o restart NÃO invalida um índice válido (só invalida se o var mudar).
        from .indexer import fingerprint_source
        vi.source_fingerprint = fingerprint_source(text_value)
        vi.chunks = [
            ChunkInfo(
                chunk_index=c["chunk_index"],
                text=c["chunk_text"],
                line_start=c["line_start"],
                line_end=c["line_end"],
                is_boilerplate=_classify_boilerplate(c["chunk_text"]),
            )
            for c in loaded_chunks
        ]
        vi._ingest_embeddings([c.get("embedding", []) for c in loaded_chunks])
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

    # Char-offset de início de cada linha (line i começa em line_starts[i]).
    # Ascendente por construção → busca binária no lookup abaixo.
    line_starts = []  # start_char de cada linha
    char_pos = 0
    for line in lines:
        line_starts.append(char_pos)
        char_pos += len(line) + 1  # +1 for \n

    def _char_to_line(pos: int) -> int:
        """Linha (0-idx) que contém a posição de caractere `pos`.

        bisect_right - 1 = maior índice cujo start_char <= pos. Antes isto era
        um scan linear de trás-p/-frente por chunk → O(chunks × linhas): num var
        de 11 MB (~24k chunks × ~150k linhas) eram bilhões de iterações Python a
        cada build de embeddings, tudo dentro do dispatch lock.
        """
        j = bisect.bisect_right(line_starts, pos) - 1
        return j if j >= 0 else 0

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
