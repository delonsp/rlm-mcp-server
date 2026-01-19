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

    def search(self, term: str, limit: int = 10) -> list[dict]:
        """Busca um termo no índice."""
        term_lower = term.lower()
        if term_lower in self.terms:
            return self.terms[term_lower][:limit]
        return []

    def search_multiple(self, terms: list[str], require_all: bool = False) -> dict:
        """
        Busca múltiplos termos.

        Args:
            terms: Lista de termos para buscar
            require_all: Se True, retorna apenas linhas com TODOS os termos

        Returns:
            {termo: [matches]} ou {linha: [termos]} se require_all
        """
        if not require_all:
            return {t: self.search(t) for t in terms if self.search(t)}

        # Buscar linhas que têm todos os termos
        line_terms = defaultdict(set)
        for term in terms:
            for match in self.search(term):
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
