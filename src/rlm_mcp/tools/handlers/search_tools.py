"""
Handlers de busca: rlm_search_index (keyword/semantic/hybrid sobre uma var)
e rlm_search_code (símbolos via tree-sitter).

Corpos movidos verbatim do call_tool monolítico (http_server).
"""

import logging

from ... import response_formatter as fmt
from ... import code_parser
from ...indexer import get_index, set_index, create_index, hybrid_search
from ...services.persistence_service import ensure_embeddings
from ..context import ToolContext

logger = logging.getLogger("rlm-http")


def rlm_search_index(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    var_name = arguments["var_name"]
    terms = arguments["terms"]
    mode = arguments.get("mode", "keyword")
    require_all = arguments.get("require_all", False)
    limit = arguments.get("limit", 20)
    offset = arguments.get("offset", 0)
    max_results = arguments.get("max_results", 30)

    # Verificar se variável existe
    if var_name not in repl.variables:
        return {
            "content": [
                {"type": "text", "text": f"Erro: Variável '{var_name}' não encontrada no REPL."}
            ],
            "isError": True
        }

    try:
        source_var = repl.variables.get(var_name)
        source_str = source_var if isinstance(source_var, str) else None

        if mode in ("semantic", "hybrid"):
            # Lazy-build embeddings se faltarem (var criado via rlm_execute
            # ou cujo embed falhou no load). Persiste server-side: o custo
            # é pago uma vez, e mesmo que o client dê timeout no 1º build
            # grande, os embeddings ficam salvos e a 2ª busca volta rápida.
            if source_str:
                ensure_embeddings(var_name, source_str)
            # Use hybrid search (supports keyword, semantic, hybrid)
            search_result = hybrid_search(
                var_name, terms, mode=mode,
                require_all=require_all,
                limit=limit, offset=offset,
                source_text=source_str,
            )
            text = fmt.format_hybrid_search(
                search_result, terms, var_name,
                offset=offset, limit=limit,
                max_results=max_results,
            )
            return {"content": [{"type": "text", "text": text}]}
        else:
            # Keyword: BM25-ranked por default. Substring legacy fica só
            # para FRASE literal (termo com espaço → match exato). require_all
            # (sem frase) vai pelo BM25 com pós-filtro de interseção.
            index = get_index(var_name)
            if not index and not source_str:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro: Variável '{var_name}' não possui índice e não é texto."}
                    ],
                    "isError": True
                }

            is_phrase = any(' ' in t.strip() for t in terms)
            if not is_phrase:
                search_result = hybrid_search(
                    var_name, terms, mode="keyword",
                    require_all=require_all,
                    limit=limit, offset=offset,
                    source_text=source_str,
                )
                text = fmt.format_hybrid_search(
                    search_result, terms, var_name,
                    offset=offset, limit=limit,
                    max_results=max_results,
                )
                return {"content": [{"type": "text", "text": text}]}

            # Frase literal → substring legacy
            if not index and source_str:
                index = create_index(source_str, var_name)
                set_index(var_name, index)

            results = index.search_multiple(terms, require_all=require_all,
                                             source_text=source_str)

            # Apply global cap (max_results) across all terms
            total_available = 0
            if not require_all and isinstance(results, dict):
                total_available = sum(len(v) for v in results.values())
                if total_available > max_results:
                    capped = {}
                    count = 0
                    for term, matches in results.items():
                        if count >= max_results:
                            break
                        take = min(len(matches), max_results - count)
                        capped[term] = matches[:take]
                        count += take
                    results = capped

            total_results = len(results) if results else 0

            text = fmt.format_search_response(
                results, terms, require_all, total_results,
                offset, limit,
                max_results=max_results, total_available=total_available,
            )
            return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro na busca: {e}"}
            ],
            "isError": True
        }


def rlm_search_code(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    var_name = arguments["var_name"]
    query = arguments.get("query")
    kind = arguments.get("kind")
    include_source = arguments.get("include_source", False)
    language_hint = arguments.get("language")
    code_limit = arguments.get("limit", 20)
    code_offset = arguments.get("offset", 0)
    max_source_lines = arguments.get("max_source_lines", 5)

    if var_name not in repl.variables:
        return {
            "content": [{"type": "text", "text": f"Erro: Variável '{var_name}' não encontrada no REPL."}],
            "isError": True,
        }

    value = repl.variables[var_name]
    if not isinstance(value, str):
        return {
            "content": [{"type": "text", "text": f"Erro: Variável '{var_name}' não é texto (tipo: {type(value).__name__})."}],
            "isError": True,
        }

    # Check if we already have a parsed CodeStructure in metadata
    meta_key = f"_code_structure_{var_name}"
    structure = repl.variables.get(meta_key)

    if not structure or not isinstance(structure, code_parser.CodeStructure):
        # Parse on-the-fly
        lang = language_hint
        if not lang:
            # Try to detect from variable metadata source
            meta = repl.variable_metadata.get(var_name)
            source_hint = meta.source if meta else ""
            # Source may contain filename info like "s3:bucket/path/file.py" or "file:/data/file.py"
            lang = code_parser.detect_language(source_hint or var_name, value)

        if not lang:
            return {
                "content": [{"type": "text", "text": f"Erro: Não foi possível detectar a linguagem de '{var_name}'. Especifique o parâmetro 'language'."}],
                "isError": True,
            }

        if not code_parser.is_available():
            return {
                "content": [{"type": "text", "text": "Erro: tree-sitter não está instalado no servidor."}],
                "isError": True,
            }

        structure = code_parser.parse(value, lang)
        if not structure:
            return {
                "content": [{"type": "text", "text": f"Erro: Falha ao parsear '{var_name}' como {lang}. Gramática pode não estar instalada."}],
                "isError": True,
            }

        # Cache the structure
        repl.variables[meta_key] = structure

    results = structure.search(
        query=query,
        kind=kind,
        include_source=include_source,
        source_code=value,
    )

    # Apply pagination
    total_matched = len(results)
    results = results[code_offset:code_offset + code_limit]

    text = fmt.format_search_code(
        results, var_name, structure.language,
        query=query, kind=kind, total_symbols=len(structure.symbols),
        limit=code_limit, offset=code_offset,
        max_source_lines=max_source_lines, total_matched=total_matched,
    )
    return {"content": [{"type": "text", "text": text}]}
