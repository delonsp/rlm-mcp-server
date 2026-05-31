"""
Indexação automática de texto para RLM MCP Server.

Cria índices semânticos automaticamente ao carregar documentos grandes,
permitindo buscas rápidas sem varrer o texto todo.
"""

import os
import re
import math
import logging
import threading
import unicodedata
from collections import defaultdict, Counter
from typing import Optional
from dataclasses import dataclass, field

from .stopwords import STOPWORDS

logger = logging.getLogger("rlm-mcp.indexer")

# Termos padrão para indexação (pode ser expandido)
DEFAULT_INDEX_TERMS = {
    # Emoções/Estados mentais
    'medo', 'ansiedade', 'raiva', 'tristeza', 'depressão', 'alegria',
    'culpa', 'vergonha', 'humilhação', 'indignação', 'ressentimento',
    'ciúme', 'inveja', 'orgulho', 'arrogância', 'timidez',

    # Relações
    'família', 'pai', 'mãe', 'filho', 'filha', 'irmão', 'irmã',
    'parceiro', 'marido', 'esposa', 'amigo', 'chefe',

    # Trabalho/Sociedade
    'trabalho', 'empresa', 'negócio', 'dinheiro', 'pobreza', 'riqueza',
    'sucesso', 'fracasso', 'responsabilidade', 'dever', 'tarefa',
    'poder', 'liderança', 'autoridade', 'controle',

    # Sintomas físicos comuns
    'dor', 'cefaleia', 'febre', 'fraqueza', 'cansaço', 'insônia',
    'náusea', 'vômito', 'diarreia', 'constipação', 'tosse',
    'palpitação', 'tremor', 'paralisia', 'convulsão',

    # Partes do corpo
    'cabeça', 'olho', 'ouvido', 'nariz', 'boca', 'garganta',
    'coração', 'pulmão', 'estômago', 'fígado', 'rim',
    'osso', 'músculo', 'pele', 'sangue', 'nervo',

    # Modalidades
    'frio', 'calor', 'manhã', 'noite', 'repouso', 'movimento',
}


@dataclass
class TextIndex:
    """Índice semântico de um texto."""

    var_name: str
    total_chars: int
    total_lines: int
    terms: dict = field(default_factory=dict)  # termo -> [{"linha": int, "contexto": str}]
    structure: dict = field(default_factory=dict)  # capítulos, seções, etc.
    custom_terms: list = field(default_factory=list)  # termos adicionais indexados

    # --- BM25 (runtime-only; NÃO serializado em to_dict/from_dict; lazy-build) ---
    bm25_postings: dict = field(default_factory=dict, repr=False)   # term -> [(seg_id, tf)]
    bm25_doc_len: list = field(default_factory=list, repr=False)    # seg_id -> nº tokens
    bm25_segments: list = field(default_factory=list, repr=False)   # seg_id -> (l_start, l_end) 0-idx
    bm25_avgdl: float = field(default=0.0, repr=False)
    bm25_n: int = field(default=0, repr=False)
    _bm25_built: bool = field(default=False, repr=False)
    _bm25_degraded: bool = field(default=False, repr=False)
    _lines: list = field(default_factory=list, repr=False)          # cache de linhas p/ snippet

    def search(self, term: str, limit: int = 10, source_text: str = None,
               context_chars: int = 100) -> list[dict]:
        """Busca um termo no índice. Falls back to live scan if term not pre-indexed.

        Args:
            term: Term to search
            limit: Max results
            source_text: Original text for live fallback (if term not in pre-built index)
            context_chars: Context chars for live scan results
        """
        term_lower = term.lower()
        if term_lower in self.terms:
            return self.terms[term_lower][:limit]

        # Live search fallback: scan source text and cache results
        if source_text is not None:
            matches = _live_scan_term(source_text, term_lower, context_chars)
            if matches:
                self.terms[term_lower] = matches  # cache for future lookups
            return matches[:limit]

        return []

    def search_multiple(self, terms: list[str], require_all: bool = False,
                        source_text: str = None) -> dict:
        """
        Busca múltiplos termos.

        Args:
            terms: Lista de termos para buscar
            require_all: Se True, retorna apenas linhas com TODOS os termos
            source_text: Original text for live fallback

        Returns:
            {termo: [matches]} ou {linha: [termos]} se require_all
        """
        if not require_all:
            result = {}
            for t in terms:
                hits = self.search(t, source_text=source_text)
                if hits:
                    result[t] = hits
            return result

        # Buscar linhas que têm todos os termos
        line_terms = defaultdict(set)
        for term in terms:
            for match in self.search(term, source_text=source_text):
                line_terms[match['linha']].add(term.lower())

        # Filtrar linhas com todos os termos
        all_terms_set = set(t.lower() for t in terms)
        result = {}
        for linha, found_terms in line_terms.items():
            if found_terms == all_terms_set:
                result[linha] = list(found_terms)

        return result

    # =========================================================================
    # BM25 (Okapi) — ranking de relevância em granularidade de segmento.
    # Campos runtime-only (acima); build idempotente e thread-safe; lazy na 1ª
    # busca. Ver plans/20260529-bm25-sentence-level.md.
    # =========================================================================

    def build_bm25(self, source_text: str, target_tokens: int = None) -> bool:
        """Constrói o índice invertido BM25 a partir do texto-fonte.

        Idempotente (gated por `_bm25_built`/`_bm25_degraded`). Thread-safe:
        builda em estruturas LOCAIS e faz atribuição atômica no fim, sob um lock
        por var — duas buscas concorrentes no mesmo índice não corrompem os dicts.

        Returns:
            True se o índice BM25 está utilizável; False se degradou p/ legacy.
        """
        if self._bm25_built or self._bm25_degraded:
            return self._bm25_built
        lock = _get_bm25_lock(self.var_name)
        with lock:
            # double-checked: outra thread pode ter buildado enquanto esperávamos
            if self._bm25_built or self._bm25_degraded:
                return self._bm25_built
            if not source_text:
                self._bm25_built = True  # nada a indexar; evita rebuild loop
                return True

            target = target_tokens or _BM25_TARGET_TOKENS
            lines = source_text.split('\n')
            segments = _segment_lines(source_text, target)

            postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
            doc_len: list[int] = []
            seg_ranges: list[tuple[int, int]] = []
            total_tokens = 0
            est_postings = 0

            for ls, le in segments:
                seg_text = "\n".join(lines[ls:le + 1])
                toks = _bm25_tokenize(seg_text)
                if not toks:
                    continue
                seg_id = len(seg_ranges)
                seg_ranges.append((ls, le))
                tf = Counter(toks)
                doc_len.append(len(toks))
                total_tokens += len(toks)
                for term, freq in tf.items():
                    postings[term].append((seg_id, freq))
                est_postings += len(tf)
                # Guard de memória: aborta e DEGRADA p/ legacy (sem cap silencioso)
                if est_postings > _BM25_MAX_POSTINGS:
                    logger.warning(
                        f"BM25 '{self.var_name}': postings ({est_postings}) > teto "
                        f"({_BM25_MAX_POSTINGS}); DEGRADANDO p/ keyword legacy."
                    )
                    self._bm25_degraded = True
                    return False

            n = len(seg_ranges)
            if n == 0:
                self._bm25_built = True  # texto sem tokens úteis; nada a ranquear
                return True

            # Atribuição atômica — _bm25_built por ÚLTIMO (publica o índice pronto)
            self.bm25_postings = dict(postings)
            self.bm25_doc_len = doc_len
            self.bm25_segments = seg_ranges
            self.bm25_avgdl = total_tokens / n
            self.bm25_n = n
            self._lines = lines
            self._bm25_built = True
            logger.info(
                f"BM25 '{self.var_name}': {n} segmentos, {len(postings)} termos, "
                f"~{est_postings} postings, avgdl={self.bm25_avgdl:.1f}"
            )
            return True

    def search_bm25(self, query_terms: list[str], source_text: str = None,
                    limit: int = 20, offset: int = 0,
                    k1: float = None, b: float = None,
                    require_all: bool = False) -> list[dict]:
        """Busca BM25: retorna segmentos ranqueados por relevância.

        Args:
            query_terms: termos do usuário (podem ser multi-palavra; juntados num bag)
            source_text: texto-fonte (necessário p/ lazy-build)
            limit/offset: paginação sobre a lista ranqueada
            k1/b: parâmetros Okapi (default env)
            require_all: pós-filtro — só segmentos contendo TODOS os tokens da query

        Returns:
            [{"line", "line_end", "score", "text", "_overlap_text"}] ordenado desc
        """
        if not _BM25_ENABLED:
            return []
        self.build_bm25(source_text)
        if self._bm25_degraded or not self._bm25_built or self.bm25_n == 0:
            return []

        k1 = _BM25_K1 if k1 is None else k1
        b = _BM25_B if b is None else b

        q_tokens: list[str] = []
        for t in query_terms:
            q_tokens.extend(_bm25_tokenize(t))
        q_set = set(q_tokens)
        if not q_set:
            return []

        N = self.bm25_n
        avgdl = self.bm25_avgdl or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in q_set:
            postings = self.bm25_postings.get(term)
            if not postings:
                continue
            n_q = len(postings)  # nº de segmentos que contêm o termo
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            for seg_id, tf in postings:
                dl = self.bm25_doc_len[seg_id]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[seg_id] += idf * (tf * (k1 + 1.0)) / denom
        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: -x[1])

        # require_all: pós-filtro por interseção de segmentos (preserva ranking BM25)
        if require_all and len(q_set) > 1:
            seg_sets = []
            for term in q_set:
                postings = self.bm25_postings.get(term)
                seg_sets.append({sid for sid, _ in postings} if postings else set())
            common = set.intersection(*seg_sets) if seg_sets else set()
            ranked = [(sid, sc) for sid, sc in ranked if sid in common]

        page = ranked[offset:offset + limit]
        hits = []
        for seg_id, sc in page:
            ls, le = self.bm25_segments[seg_id]
            raw = "\n".join(self._lines[ls:le + 1]) if self._lines else ""
            hits.append({
                "line": ls,
                "line_end": le,
                "score": sc,
                "text": _normalize_snippet(raw),
                "_overlap_text": raw,
            })
        return hits

    def get_stats(self) -> dict:
        """Retorna estatísticas do índice."""
        return {
            "var_name": self.var_name,
            "total_chars": self.total_chars,
            "total_lines": self.total_lines,
            "indexed_terms": len(self.terms),
            "total_occurrences": sum(len(v) for v in self.terms.values()),
            "top_terms": sorted(
                [(k, len(v)) for k, v in self.terms.items()],
                key=lambda x: -x[1]
            )[:20]
        }

    def to_dict(self) -> dict:
        """Serializa o índice para persistência."""
        return {
            "var_name": self.var_name,
            "total_chars": self.total_chars,
            "total_lines": self.total_lines,
            "terms": self.terms,
            "structure": self.structure,
            "custom_terms": self.custom_terms
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TextIndex":
        """Reconstrói índice a partir de dict."""
        return cls(
            var_name=data["var_name"],
            total_chars=data["total_chars"],
            total_lines=data["total_lines"],
            terms=data.get("terms", {}),
            structure=data.get("structure", {}),
            custom_terms=data.get("custom_terms", [])
        )


def _live_scan_term(text: str, term_lower: str, context_chars: int = 100) -> list[dict]:
    """Scan text for a term not in the pre-built index. Returns matches like the indexed format."""
    matches = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if term_lower in line.lower():
            # Avoid duplicate lines
            if matches and matches[-1]['linha'] == i:
                continue
            matches.append({
                'linha': i,
                'contexto': line[:context_chars].strip()
            })
    return matches


def create_index(
    text: str,
    var_name: str,
    additional_terms: list[str] = None,
    context_chars: int = 100
) -> TextIndex:
    """
    Cria um índice semântico para um texto.

    Args:
        text: Texto para indexar (None tratado como string vazia)
        var_name: Nome da variável associada
        additional_terms: Termos adicionais para indexar além dos padrão
        context_chars: Caracteres de contexto ao redor do termo

    Returns:
        TextIndex com o índice criado
    """
    # Tratar None como string vazia
    if text is None:
        text = ""

    logger.info(f"Criando índice para '{var_name}' ({len(text):,} chars)")

    # Combinar termos padrão + adicionais
    terms_to_index = DEFAULT_INDEX_TERMS.copy()
    if additional_terms:
        terms_to_index.update(t.lower() for t in additional_terms)

    # Inicializar índice
    index = TextIndex(
        var_name=var_name,
        total_chars=len(text),
        total_lines=len(text.splitlines()),
        custom_terms=additional_terms or []
    )

    # Indexar cada linha
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()

        for term in terms_to_index:
            if term in line_lower:
                if term not in index.terms:
                    index.terms[term] = []

                # Evitar duplicatas muito próximas
                if index.terms[term] and index.terms[term][-1]['linha'] == i:
                    continue

                index.terms[term].append({
                    'linha': i,
                    'contexto': line[:context_chars].strip()
                })

    # Detectar estrutura do documento (capítulos, seções)
    index.structure = _detect_structure(text)

    logger.info(f"Índice criado: {len(index.terms)} termos, {sum(len(v) for v in index.terms.values())} ocorrências")
    return index


def _detect_structure(text: str) -> dict:
    """Detecta estrutura do documento (capítulos, seções, remédios)."""
    structure = {
        "headers": [],
        "capitulos": [],
        "remedios": []
    }

    # Tratar None como string vazia
    if text is None:
        text = ""

    lines = text.split('\n')

    for i, line in enumerate(lines):
        # Headers markdown
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            structure["headers"].append({
                "linha": i,
                "nivel": level,
                "titulo": title[:100]
            })

        # Padrão de capítulo numérico (ex: "4.8 Ferrum")
        match = re.match(r'^(\d+\.\d+)\s+([A-Z][a-zA-Z]+)', line)
        if match:
            structure["capitulos"].append({
                "linha": i,
                "numero": match.group(1),
                "titulo": match.group(2)
            })

        # Padrão de remédio (ex: "Quadro de Ferrum metallicum")
        match = re.match(r'Quadro de (\w+(?:\s+\w+)?)', line)
        if match:
            structure["remedios"].append({
                "linha": i,
                "nome": match.group(1)
            })

    return structure


def auto_index_if_large(text: str, var_name: str, min_chars: int = 100000) -> Optional[TextIndex]:
    """
    Cria índice automaticamente se o texto for grande o suficiente.

    Args:
        text: Texto para potencialmente indexar (None tratado como string vazia)
        var_name: Nome da variável
        min_chars: Tamanho mínimo para indexar automaticamente

    Returns:
        TextIndex se indexado, None se texto pequeno
    """
    # Tratar None como string vazia
    if text is None:
        text = ""

    if len(text) >= min_chars:
        return create_index(text, var_name)
    return None


# Cache de índices em memória
_indices_cache: dict[str, TextIndex] = {}


def get_index(var_name: str) -> Optional[TextIndex]:
    """Retorna índice do cache."""
    return _indices_cache.get(var_name)


def set_index(var_name: str, index: TextIndex):
    """Salva índice no cache."""
    _indices_cache[var_name] = index


def clear_index(var_name: str):
    """Remove índice do cache."""
    _indices_cache.pop(var_name, None)


def clear_all_indices():
    """Limpa todo o cache de índices."""
    _indices_cache.clear()


# =============================================================================
# BM25 — configuração e helpers module-level (parsing igual aos _DAMPENING_*)
# =============================================================================

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_BM25_ENABLED = os.getenv("RLM_BM25_ENABLED", "true").lower() in ("true", "1", "yes")
_BM25_TARGET_TOKENS = _env_int("RLM_BM25_TARGET_TOKENS", 120)
_BM25_K1 = _env_float("RLM_BM25_K1", 1.2)
_BM25_B = _env_float("RLM_BM25_B", 0.75)
# Tokens com len < este são dropados. Default 2 (casa Matryoshka/_extract_key_terms).
# Limitação conhecida: derruba "vitamina D", "K2" → baixar p/ 1 via env se preciso.
_BM25_MIN_TOKEN_LEN = _env_int("RLM_BM25_MIN_TOKEN_LEN", 2)
# Teto de postings antes de degradar p/ legacy (backstop, não cap silencioso).
_BM25_MAX_POSTINGS = _env_int("RLM_BM25_MAX_POSTINGS", 5_000_000)
# Janela (em linhas) p/ fundir segmento BM25 e chunk semântico cujos ranges quase
# se tocam. 0 = só overlap estrito; default 2 ponte near-miss de fronteiras.
_BM25_FUSION_WINDOW = _env_int("RLM_BM25_FUSION_WINDOW", 2)

# Locks por var p/ build BM25 thread-safe (lazy-build sob concorrência FastAPI).
_bm25_locks: dict[str, threading.Lock] = {}
_bm25_locks_guard = threading.Lock()


def _get_bm25_lock(var_name: str) -> threading.Lock:
    """Lock dedicado por var (cria sob guarda na 1ª vez)."""
    lock = _bm25_locks.get(var_name)
    if lock is None:
        with _bm25_locks_guard:
            lock = _bm25_locks.get(var_name)
            if lock is None:
                lock = threading.Lock()
                _bm25_locks[var_name] = lock
    return lock


# Sentinel de fronteira de var no índice combinado de coleção (header
# "===...\n=== VARIÁVEL: x ===\n===..."). Força fim de segmento → nenhum
# segmento BM25 cruza fronteira de var (mapeamento linha→var intacto).
_COLLECTION_SENTINEL_RE = re.compile(r"^(={3,}\s*$|=== VARI[ÁA]VEL:)")
_WS_RE = re.compile(r"\s+")


def _fold_accents(text: str) -> str:
    """NFKD + remove combining marks → match acento-insensitive (esperado em PT:
    'câncer' e 'cancer' colapsam; robusto a OCR sujo)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _bm25_tokenize(text: str) -> list[str]:
    """Tokeniza p/ BM25: lowercase + accent-fold, split no _KEY_TERM_RE (reusado),
    dropa STOPWORDS e tokens < _BM25_MIN_TOKEN_LEN. Mantém FREQUÊNCIA (lista)."""
    if not text:
        return []
    folded = _fold_accents(text.lower())
    words = _KEY_TERM_RE.split(folded)
    return [w for w in words
            if len(w) >= _BM25_MIN_TOKEN_LEN and w not in STOPWORDS]


def _segment_lines(text: str, target_tokens: int = None) -> list[tuple[int, int]]:
    """Empacota linhas consecutivas em segmentos de ~target_tokens tokens.

    Fecha o segmento ao: (a) atingir o alvo de tokens, (b) encontrar linha em
    branco com conteúdo acumulado, (c) encontrar sentinel de fronteira de var.
    Sentinels e linhas em branco NÃO entram em segmento. Linha única acima do
    alvo vira segmento próprio. Convenção 0-indexed; retorna [(l_start, l_end)]
    inclusivo (casa com vector_index/_live_scan_term).
    """
    target = target_tokens or _BM25_TARGET_TOKENS
    lines = text.split('\n')
    segments: list[tuple[int, int]] = []
    seg_start: Optional[int] = None
    acc_tokens = 0

    def _close(end_idx: int):
        nonlocal seg_start, acc_tokens
        if seg_start is not None:
            segments.append((seg_start, end_idx))
            seg_start = None
            acc_tokens = 0

    for i, line in enumerate(lines):
        if _COLLECTION_SENTINEL_RE.match(line):
            _close(i - 1)
            continue
        if not line.strip():
            _close(i - 1)
            continue
        if seg_start is None:
            seg_start = i
            acc_tokens = 0
        acc_tokens += len(_bm25_tokenize(line))
        if acc_tokens >= target:
            _close(i)

    if seg_start is not None:
        segments.append((seg_start, len(lines) - 1))
    return segments


def _normalize_snippet(text: str) -> str:
    """Colapsa runs de whitespace p/ display (só o snippet; NÃO toca tokenização
    nem _overlap_text)."""
    return _WS_RE.sub(" ", text).strip()


def _legacy_to_ranked(keyword_results: dict) -> list[dict]:
    """Converte o dict legacy {termo:[matches]} p/ a lista ranqueada do BM25.

    Usado quando o BM25 degrada/desabilita mas ainda queremos alimentar a fusão
    RRF (e o formatter BM25) com o keyword legacy. Dedup por linha, ordem de 1ª
    ocorrência (não há score BM25 → score 0.0).
    """
    out: list[dict] = []
    seen = set()
    for matches in keyword_results.values():
        for m in matches:
            linha = m["linha"]
            if linha in seen:
                continue
            seen.add(linha)
            ctx = m.get("contexto", "")
            out.append({
                "line": linha,
                "line_end": linha,
                "score": 0.0,
                "text": ctx,
                "_overlap_text": ctx,
            })
    return out


# =============================================================================
# Hybrid Search (keyword + semantic with Reciprocal Rank Fusion)
# =============================================================================

def hybrid_search(
    var_name: str,
    terms: list[str],
    mode: str = "keyword",
    require_all: bool = False,
    limit: int = 20,
    offset: int = 0,
    source_text: str = None,
) -> dict:
    """Perform keyword, semantic, or hybrid search.

    Args:
        var_name: Variable name to search
        terms: Search terms
        mode: "keyword" (default), "semantic", or "hybrid"
        require_all: For keyword mode, require all terms in same line
        limit: Max results
        offset: Pagination offset
        source_text: Original text for live keyword fallback

    Returns:
        dict with:
            - "mode": actual mode used
            - "keyword_results": keyword search results (if applicable)
            - "semantic_results": list of {chunk_text, line_start, line_end, score} (if applicable)
            - "hybrid_results": RRF-fused results (if hybrid mode)
            - "stats": index stats
    """
    from .vector_index import get_vector_index

    result = {
        "mode": mode,
        "keyword_results": None,   # dict legacy {termo:[matches]} — só frase literal (handler)
        "keyword_ranked": None,    # lista ranqueada BM25 — fonte de verdade default
        "semantic_results": None,
        "hybrid_results": None,
        "stats": {},
    }

    keyword_index = get_index(var_name)
    if keyword_index is None and source_text:
        # Cria índice on-the-fly; BM25 é buildado lazy na 1ª busca
        keyword_index = create_index(source_text, var_name)
        set_index(var_name, keyword_index)

    def _keyword_leg(lim: int, off: int, req_all: bool) -> Optional[list[dict]]:
        """Perna keyword como lista ranqueada (BM25, ou legacy convertido se BM25
        desabilitado/degradado). Pagina por (off, lim)."""
        if keyword_index is None:
            return None
        result["stats"]["keyword"] = keyword_index.get_stats()
        if _BM25_ENABLED and source_text:
            ranked = keyword_index.search_bm25(
                terms, source_text, limit=lim, offset=off, require_all=req_all
            )
            if not keyword_index._bm25_degraded:
                return ranked
        # BM25 off ou degradou (guard de memória) → legacy substring convertido
        legacy = keyword_index.search_multiple(
            terms, require_all=False, source_text=source_text
        )
        return _legacy_to_ranked(legacy)[off:off + lim]

    # --- Semantic leg ---
    semantic_results = None
    vector_index = get_vector_index(var_name)
    if vector_index and mode in ("semantic", "hybrid"):
        query_text = " ".join(terms)
        raw_results = vector_index.search(query_text, top_k=limit + offset)
        semantic_results = [
            {
                "chunk_text": r.chunk_text,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "score": r.score,
                "chunk_index": r.chunk_index,
            }
            for r in raw_results
        ]
        result["stats"]["vector"] = vector_index.get_stats()

    # --- Dispatch por modo ---
    if mode == "keyword":
        # Página direta do BM25 (offset/limit já aplicados na perna)
        result["keyword_ranked"] = _keyword_leg(limit, offset, require_all)

    elif mode == "semantic":
        if semantic_results is not None:
            result["semantic_results"] = semantic_results[offset:offset + limit]
        else:
            # Sem índice vetorial → fallback keyword
            result["mode"] = "keyword (fallback)"
            result["keyword_ranked"] = _keyword_leg(limit, offset, require_all)

    elif mode == "hybrid":
        # Candidatos não-paginados de cada perna; o RRF pagina no fim
        bm25_for_fusion = _keyword_leg(limit + offset, 0, False) or []
        sem_for_fusion = semantic_results or []
        if bm25_for_fusion or sem_for_fusion:
            result["hybrid_results"] = _reciprocal_rank_fusion(
                bm25_for_fusion, sem_for_fusion, terms, limit, offset
            )
            if not sem_for_fusion:
                result["mode"] = (
                    "hybrid (no embeddings)" if not vector_index
                    else "hybrid (no semantic hits)"
                )
        else:
            result["mode"] = "keyword (no results)"

    return result


# =============================================================================
# Gravity dampening (insight portado do Matryoshka / Ori-Mnemos)
# Rebaixa "cosine ghosts": resultados de score alto que não contêm nenhum
# termo da query no texto. Ablation-validated no Ori-Mnemos (P@5 delta -0.256).
# =============================================================================

_DAMPENING_ENABLED = os.getenv("RLM_GRAVITY_DAMPENING", "true").lower() in ("true", "1", "yes")
_DAMPENING_PENALTY = float(os.getenv("RLM_DAMPENING_PENALTY", "0.5"))
_DAMPENING_THRESHOLD_RATIO = float(os.getenv("RLM_DAMPENING_THRESHOLD_RATIO", "0.3"))

# Stopwords mínimas (PT+EN) para extração de termos — evita que uma stopword
# na query satisfaça o overlap trivialmente.
_QUERY_STOPWORDS = {
    "a", "o", "e", "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "um", "uma", "que", "com", "por", "para", "se", "os", "as", "ao", "aos", "à",
    "às", "ou", "the", "of", "and", "to", "in", "is", "it", "for", "on", "with", "at", "by",
}

_KEY_TERM_RE = re.compile(r"[^0-9a-zà-ÿ]+", re.UNICODE)


def _extract_key_terms(text: str) -> set[str]:
    """Extrai termos de conteúdo (lowercase, sem stopwords, len>1)."""
    if not text:
        return set()
    words = _KEY_TERM_RE.split(text.lower())
    return {w for w in words if len(w) > 1 and w not in _QUERY_STOPWORDS}


def _apply_gravity_dampening(
    results: list[dict],
    terms: list[str],
    score_key: str = "rrf_score",
    text_key: str = "_overlap_text",
    penalty: float = None,
    threshold_ratio: float = None,
) -> list[dict]:
    """Aplica gravity dampening in-place sobre uma lista de resultados.

    Multiplica por `penalty` o score de qualquer resultado cujo score esteja
    acima de `threshold_ratio * max_score` E cujo texto não compartilhe nenhum
    termo da query. Threshold adaptativo funciona em qualquer escala de score.
    Lados keyword nunca são afetados (o contexto sempre contém o termo).
    """
    if not _DAMPENING_ENABLED or not results:
        return results
    penalty = _DAMPENING_PENALTY if penalty is None else penalty
    threshold_ratio = _DAMPENING_THRESHOLD_RATIO if threshold_ratio is None else threshold_ratio

    query_terms: set[str] = set()
    for t in terms:
        query_terms |= _extract_key_terms(t)
    if not query_terms:
        return results

    max_score = max((r.get(score_key, 0.0) for r in results), default=0.0)
    if max_score <= 0:
        return results
    threshold = max_score * threshold_ratio

    for r in results:
        if r.get(score_key, 0.0) <= threshold:
            continue
        text_terms = _extract_key_terms(r.get(text_key, ""))
        if query_terms.isdisjoint(text_terms):
            r[score_key] = r.get(score_key, 0.0) * penalty
    return results


def _reciprocal_rank_fusion(
    bm25_hits: list[dict],
    semantic_results: list[dict],
    terms: list[str],
    limit: int = 20,
    offset: int = 0,
    k: int = 60,
) -> list[dict]:
    """Funde keyword(BM25) e semantic via Reciprocal Rank Fusion.

    RRF é rank-based: cada perna contribui 1/(k+rank) — o *valor* do score BM25
    não entra na fusão, só ordena a perna keyword. A fusão se dá por SOBREPOSIÇÃO
    de range [line_start, line_end] (não `line` exato): segmentos BM25 (~120 tok) e
    chunks semânticos (512 chars) quase nunca têm line_start idêntico, então keyar
    por linha exata degeneraria em concatenação (P0 da crítica Codex). Entradas
    cujos ranges se sobrepõem (ou ficam dentro de _BM25_FUSION_WINDOW linhas)
    fundem-se: a fundida herda o menor line_start como chave de display e acumula
    1/(k+rank) de cada perna.

    Args:
        bm25_hits: lista ranqueada do search_bm25 [{line, line_end, score, text, _overlap_text}]
        semantic_results: [{chunk_text, line_start, line_end, score}] do vetorial
        terms: termos originais da query (p/ dampening)
        limit/offset: paginação
        k: constante RRF (default 60)

    Returns:
        Lista fundida ordenada por rrf_score desc.
    """
    # Entradas unificadas de ambas as pernas (rank = posição na lista da perna).
    entries: list[dict] = []
    for rank, h in enumerate(bm25_hits):
        ls = h["line"]
        entries.append({
            "line_start": ls,
            "line_end": h.get("line_end", ls),
            "rrf": 1.0 / (k + rank),
            "source": "keyword",
            "text": h.get("text", ""),
            "overlap": h.get("_overlap_text", h.get("text", "")),
        })
    for rank, sr in enumerate(semantic_results):
        ls = sr["line_start"]
        entries.append({
            "line_start": ls,
            "line_end": sr.get("line_end", ls),
            "rrf": 1.0 / (k + rank),
            "source": "semantic",
            "text": sr["chunk_text"][:200],
            "overlap": sr["chunk_text"],
        })

    if not entries:
        return []

    # Merge por overlap: sorted por line_start → sweep contra o último cluster
    # (o cluster acumula o maior line_end visto, então qualquer intervalo que
    # sobreponha um cluster anterior também sobrepõe o corrente).
    entries.sort(key=lambda e: e["line_start"])
    fused: list[dict] = []
    for e in entries:
        if fused and e["line_start"] <= fused[-1]["line_end"] + _BM25_FUSION_WINDOW:
            f = fused[-1]
            f["rrf_score"] += e["rrf"]
            f["sources"].add(e["source"])
            f["line_end"] = max(f["line_end"], e["line_end"])
            f["_overlap_text"] += "\n" + e["overlap"]
            # Display: preferir texto da perna keyword (contém o termo literal)
            if f["_text_source"] != "keyword" and e["source"] == "keyword":
                f["text"] = e["text"]
                f["_text_source"] = "keyword"
        else:
            fused.append({
                "line": e["line_start"],
                "line_end": e["line_end"],
                "rrf_score": e["rrf"],
                "sources": {e["source"]},
                "text": e["text"],
                "_text_source": e["source"],
                "_overlap_text": e["overlap"],
            })

    # Gravity dampening antes do sort (rebaixa cosine ghosts)
    fused = _apply_gravity_dampening(fused, terms)

    sorted_results = sorted(fused, key=lambda x: -x["rrf_score"])

    # Serializável: sources set→list; remover helpers internos
    for r in sorted_results:
        r["sources"] = list(r["sources"])
        r.pop("_overlap_text", None)
        r.pop("_text_source", None)

    return sorted_results[offset:offset + limit]
