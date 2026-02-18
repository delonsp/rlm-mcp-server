"""
Indexação automática de texto para RLM MCP Server.

Cria índices semânticos automaticamente ao carregar documentos grandes,
permitindo buscas rápidas sem varrer o texto todo.
"""

import re
import logging
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass, field

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
        "keyword_results": None,
        "semantic_results": None,
        "hybrid_results": None,
        "stats": {},
    }

    # Keyword search
    keyword_results = None
    keyword_index = get_index(var_name)
    if mode in ("keyword", "hybrid"):
        if not keyword_index and source_text:
            # Create index on-the-fly
            keyword_index = create_index(source_text, var_name)
            set_index(var_name, keyword_index)
        if keyword_index:
            keyword_results = keyword_index.search_multiple(
                terms, require_all=require_all, source_text=source_text
            )
            result["keyword_results"] = keyword_results
            result["stats"]["keyword"] = keyword_index.get_stats()

    # Semantic search
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
        result["semantic_results"] = semantic_results[offset:offset + limit]
        result["stats"]["vector"] = vector_index.get_stats()

    # Hybrid fusion with RRF
    has_keyword = bool(keyword_results) if isinstance(keyword_results, dict) else False
    has_semantic = bool(semantic_results)

    if mode == "hybrid" and has_keyword and has_semantic:
        result["hybrid_results"] = _reciprocal_rank_fusion(
            keyword_results, semantic_results, terms, limit, offset
        )
    elif mode == "hybrid" and has_semantic and not has_keyword:
        # Keyword returned empty but semantic has results — promote semantic to hybrid
        result["hybrid_results"] = [
            {
                "line": sr["line_start"],
                "rrf_score": sr["score"],
                "text": sr["chunk_text"][:100],
                "sources": ["semantic"],
            }
            for sr in semantic_results
        ]
    elif mode == "hybrid" and has_keyword and not has_semantic:
        # Semantic unavailable, keyword has results — promote keyword to hybrid
        fused = []
        for term, matches in keyword_results.items():
            for m in matches:
                fused.append({
                    "line": m["linha"],
                    "rrf_score": 0.0,
                    "text": m.get("contexto", ""),
                    "sources": ["keyword"],
                })
        # Deduplicate by line, keep first occurrence
        seen = set()
        deduped = []
        for f in fused:
            if f["line"] not in seen:
                seen.add(f["line"])
                deduped.append(f)
        result["hybrid_results"] = deduped[offset:offset + limit]
    elif mode == "semantic" and not vector_index:
        # Fallback: no vector index, try keyword
        if keyword_index:
            result["mode"] = "keyword (fallback)"
            result["keyword_results"] = keyword_index.search_multiple(terms, require_all=require_all)
    elif mode == "hybrid" and not vector_index and not has_keyword:
        # Fallback: nothing available
        result["mode"] = "keyword (no embeddings)"

    return result


def _reciprocal_rank_fusion(
    keyword_results: dict,
    semantic_results: list[dict],
    terms: list[str],
    limit: int = 20,
    offset: int = 0,
    k: int = 60,
) -> list[dict]:
    """Combine keyword and semantic results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each result list.

    Args:
        keyword_results: {term: [matches]} from keyword search
        semantic_results: [{chunk_text, line_start, line_end, score}] from vector search
        terms: Original search terms
        limit: Max results
        offset: Pagination offset
        k: RRF constant (default 60)

    Returns:
        List of fused results sorted by RRF score
    """
    rrf_scores: dict[int, dict] = {}  # line_number -> {score, sources, text}

    # Score keyword results by line
    rank = 0
    for term, matches in keyword_results.items():
        for match in matches:
            line = match["linha"]
            if line not in rrf_scores:
                rrf_scores[line] = {
                    "line": line,
                    "rrf_score": 0.0,
                    "text": match.get("contexto", ""),
                    "sources": set(),
                }
            rrf_scores[line]["rrf_score"] += 1.0 / (k + rank)
            rrf_scores[line]["sources"].add("keyword")
            rank += 1

    # Score semantic results by line
    for rank, sr in enumerate(semantic_results):
        line = sr["line_start"]
        if line not in rrf_scores:
            rrf_scores[line] = {
                "line": line,
                "rrf_score": 0.0,
                "text": sr["chunk_text"][:100],
                "sources": set(),
            }
        rrf_scores[line]["rrf_score"] += 1.0 / (k + rank)
        rrf_scores[line]["sources"].add("semantic")

    # Sort by RRF score
    sorted_results = sorted(rrf_scores.values(), key=lambda x: -x["rrf_score"])

    # Convert sources set to list for JSON serialization
    for r in sorted_results:
        r["sources"] = list(r["sources"])

    return sorted_results[offset:offset + limit]
