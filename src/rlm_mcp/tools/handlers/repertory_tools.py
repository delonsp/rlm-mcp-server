"""
Handler do rlm_repertorio — repertorização homeopática sobre a var
`kent_repertorio` (default). Router com actions: buscar_rubrica,
repertorizar, info. A lógica de domínio (parser/índice/ranking) vive em
rlm_mcp.repertory (módulo puro); aqui só validação de argumentos e formato.
"""

import logging

from ... import repertory
from ... import response_formatter as fmt
from ..base import error_response, text_response
from ..context import ToolContext

logger = logging.getLogger("rlm-http")

DEFAULT_SOURCE = "kent_repertorio"
MAX_LIMIT = 50


def _get_index(arguments: dict, ctx: ToolContext):
    """Resolve a var fonte e devolve (index, cached, None) ou (None, _, erro)."""
    source_var = arguments.get("source_var") or DEFAULT_SOURCE
    value = ctx.repl.variables.get(source_var)
    if value is None:
        return None, False, error_response(
            f"Erro: variável '{source_var}' não encontrada no REPL. "
            f"Carregue o repertório (ex: rlm_load_s3) antes de repertorizar."
        )
    if not isinstance(value, str):
        return None, False, error_response(
            f"Erro: variável '{source_var}' não é texto (tipo {type(value).__name__})."
        )
    index, cached = repertory.get_repertory_index(source_var, value)
    return index, cached, None


def rlm_repertorio(arguments: dict, ctx: ToolContext) -> dict:
    action = arguments.get("action", "info")

    if action not in ("buscar_rubrica", "repertorizar", "info"):
        return error_response(
            f"Ação desconhecida: {action}. Use: buscar_rubrica, repertorizar, info"
        )

    index, cached, err = _get_index(arguments, ctx)
    if err is not None:
        return err

    if action == "info":
        return text_response(fmt.format_repertory_info(index, cached))

    if action == "buscar_rubrica":
        query = (arguments.get("query") or "").strip()
        if not query:
            return error_response("Erro: 'query' é obrigatória para buscar_rubrica.")
        limit = max(1, min(int(arguments.get("limit", 10)), MAX_LIMIT))
        offset = max(0, int(arguments.get("offset", 0)))
        matches, total, fuzzy_note = repertory.search_rubrics(
            index, query, limit=limit, offset=offset
        )
        return text_response(fmt.format_repertory_search(
            matches, total, query, index.source_var,
            offset=offset, fuzzy_note=fuzzy_note,
        ))

    # action == "repertorizar"
    refs = arguments.get("rubrics") or []
    if not isinstance(refs, list) or not refs:
        return error_response(
            "Erro: 'rubrics' (lista de IDs var:L### ou textos de rubrica) é "
            "obrigatória para repertorizar. Use buscar_rubrica para obter os IDs."
        )
    if len(refs) > 30:
        return error_response("Erro: máximo de 30 rubricas por repertorização.")
    entries, errors, fuzzy_notes = repertory.resolve_rubric_refs(index, refs)
    if errors:
        return error_response("Erro ao resolver rubricas:\n- " + "\n- ".join(errors))
    # dedup preservando ordem (mesma rubrica 2x não deve dobrar score)
    seen, unique = set(), []
    for e in entries:
        if e.line_no not in seen:
            seen.add(e.line_no)
            unique.append(e)
    sort = arguments.get("sort", "coverage")
    if sort not in ("coverage", "score"):
        return error_response(f"Erro: sort '{sort}' inválido (use coverage ou score).")
    limit = max(1, min(int(arguments.get("limit", 20)), MAX_LIMIT))
    result = repertory.repertorize(index, unique, sort=sort)
    out = fmt.format_repertorization(result, index, limit=limit)
    if fuzzy_notes:
        # ref textual só casou via fuzzy — avisa que o texto pedido NÃO existe
        # literalmente e foi corrigido (não substitui rubrica clínica em silêncio)
        out = "⚠️ Correção fuzzy em ref textual: " + "; ".join(fuzzy_notes) + "\n" + out
    return text_response(out)
