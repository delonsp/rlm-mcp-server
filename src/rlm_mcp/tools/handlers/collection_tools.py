"""
Handlers de coleções: rlm_collection (router consolidado), create/add/
list/info/rebuild/delete e rlm_search_collection (BM25 + fallback híbrido
+ auto-tokenize phrase-trap).

Corpos movidos verbatim do call_tool monolítico (http_server).
"""

import logging

from ... import response_formatter as fmt
from ...collection_builder import build_collection_combined
from ...indexer import (
    get_index, set_index, clear_index, create_index,
    tokenize_for_fallback, tokenized_collection_scan,
    parse_quoted_terms, format_fallback_banner,
)
from ...persistence import get_persistence
from ..context import ToolContext

logger = logging.getLogger("rlm-http")


def rlm_collection(arguments: dict, ctx: ToolContext) -> dict:
    action = arguments.get("action", "list")
    if action == "create":
        name = "rlm_collection_create"
        arguments = {"name": arguments.get("name", ""), "description": arguments.get("description")}
    elif action == "add":
        name = "rlm_collection_add"
        arguments = {"collection": arguments.get("name", ""), "vars": arguments.get("vars", [])}
    elif action == "list":
        name = "rlm_collection_list"
    elif action == "info":
        name = "rlm_collection_info"
        arguments = {"name": arguments.get("name", "")}
    elif action == "rebuild":
        name = "rlm_collection_rebuild"
        arguments = {"name": arguments.get("name", "")}
    elif action == "delete":
        name = "rlm_collection_delete"
        arguments = {"name": arguments.get("name", "")}
    elif action == "search":
        name = "rlm_search_collection"
        arguments = {"collection": arguments.get("name", ""), "terms": arguments.get("terms", []), "limit": arguments.get("limit", 10), "offset": arguments.get("offset", 0)}
    else:
        return {"content": [{"type": "text", "text": f"Ação desconhecida: {action}"}], "isError": True}
    # Dispatch to the original handler
    return ctx.call_tool(name, arguments, ctx.client_id)


def rlm_collection_create(arguments: dict, ctx: ToolContext) -> dict:
    try:
        persistence = get_persistence()
        coll_name = arguments["name"]
        description = arguments.get("description")

        success = persistence.create_collection(coll_name, description)

        if not success:
            return {
                "content": [
                    {"type": "text", "text": f"Erro: Falha ao criar coleção '{coll_name}' - verifique logs do servidor"}
                ],
                "isError": True
            }

        text = f"✅ Coleção '{coll_name}' criada"
        if description:
            text += f"\nDescrição: {description}"

        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao criar coleção: {e}"}
            ],
            "isError": True
        }


def rlm_collection_add(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    try:
        persistence = get_persistence()
        coll_name = arguments["collection"]
        var_names = arguments["vars"]

        # Verificar se variáveis existem
        missing = [v for v in var_names if v not in repl.variables]
        if missing:
            return {
                "content": [
                    {"type": "text", "text": f"Erro: Variáveis não encontradas: {', '.join(missing)}"}
                ],
                "isError": True
            }

        added = persistence.add_to_collection(coll_name, var_names)

        # === OPÇÃO C: Criar índice combinado da coleção ===
        # Obter TODAS as variáveis da coleção (não só as novas)
        all_vars = persistence.get_collection_vars(coll_name)

        # Texto combinado + mapping via builder único (line-mapping P0)
        combined_text, var_mapping, _ = build_collection_combined(
            all_vars, repl.variables
        )

        if combined_text:
            combined_var_name = f"_coll_{coll_name}_combined"

            # Salvar variável combinada no REPL
            repl.variables[combined_var_name] = combined_text

            # Forçar criação de índice (min_chars=0)
            combined_index = create_index(combined_text, combined_var_name)
            set_index(combined_var_name, combined_index)

            # Salvar mapeamento como metadado
            repl.variables[f"_coll_{coll_name}_mapping"] = var_mapping

        text = f"✅ {added} variável(is) adicionada(s) à coleção '{coll_name}'"
        text += f"\nVariáveis: {', '.join(var_names)}"

        if combined_text:
            text += f"\n\n🔍 Índice combinado atualizado: {len(combined_text):,} chars indexados"
            text += f"\n   Variáveis no índice: {len(all_vars)}"

        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao adicionar à coleção: {e}"}
            ],
            "isError": True
        }


def rlm_collection_list(arguments: dict, ctx: ToolContext) -> dict:
    try:
        persistence = get_persistence()
        collections = persistence.list_collections()

        text = fmt.format_collection_list(collections)
        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao listar coleções: {e}"}
            ],
            "isError": True
        }


def rlm_collection_info(arguments: dict, ctx: ToolContext) -> dict:
    try:
        persistence = get_persistence()
        coll_name = arguments["name"]

        info = persistence.get_collection_info(coll_name)
        if not info:
            return {
                "content": [
                    {"type": "text", "text": f"Coleção '{coll_name}' não encontrada."}
                ],
                "isError": True
            }

        text = fmt.format_collection_info(info)
        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao obter info da coleção: {e}"}
            ],
            "isError": True
        }


def rlm_collection_rebuild(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    try:
        persistence = get_persistence()
        coll_name = arguments["name"]

        # Obter variáveis da coleção
        all_vars = persistence.get_collection_vars(coll_name)
        if not all_vars:
            return {
                "content": [
                    {"type": "text", "text": f"Coleção '{coll_name}' vazia ou não existe."}
                ],
                "isError": True
            }

        # Texto combinado + mapping via builder único (line-mapping P0)
        combined_text, var_mapping, vars_included = build_collection_combined(
            all_vars, repl.variables
        )

        if not combined_text:
            return {
                "content": [
                    {"type": "text", "text": f"Nenhuma variável de texto encontrada na coleção '{coll_name}'."}
                ],
                "isError": True
            }

        combined_var_name = f"_coll_{coll_name}_combined"

        # Salvar variável combinada no REPL
        repl.variables[combined_var_name] = combined_text

        # Forçar criação de índice (min_chars=0)
        combined_index = create_index(combined_text, combined_var_name)
        set_index(combined_var_name, combined_index)

        # Salvar mapeamento como metadado
        repl.variables[f"_coll_{coll_name}_mapping"] = var_mapping

        stats = combined_index.get_stats()
        text = f"✅ Índice combinado da coleção '{coll_name}' reconstruído!"
        text += f"\n\n📊 Estatísticas:"
        text += f"\n   Variáveis incluídas: {vars_included}/{len(all_vars)}"
        text += f"\n   Tamanho total: {len(combined_text):,} caracteres"
        text += f"\n   Termos indexados: {stats['indexed_terms']}"
        text += f"\n   Total de ocorrências: {stats['total_occurrences']}"
        text += f"\n\n🔍 Agora use: rlm_search_collection(collection=\"{coll_name}\", terms=[...])"

        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao reconstruir índice: {e}"}
            ],
            "isError": True
        }


def rlm_collection_delete(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    try:
        persistence = get_persistence()
        coll_name = arguments["name"]
        if not coll_name:
            return {"content": [{"type": "text", "text": "Erro: informe o nome da coleção."}],
                    "isError": True}

        if persistence.get_collection_info(coll_name) is None:
            return {"content": [{"type": "text",
                                 "text": f"Coleção '{coll_name}' não existe."}],
                    "isError": True}

        if not persistence.delete_collection(coll_name):
            return {"content": [{"type": "text",
                                 "text": f"Erro ao remover a coleção '{coll_name}' (ver logs)."}],
                    "isError": True}

        # Limpa os artefatos em memória (combinado, mapping e índice) —
        # as VARIÁVEIS membras ficam intactas (delete remove só a
        # associação, como o persistence documenta).
        combined_var_name = f"_coll_{coll_name}_combined"
        repl.variables.pop(combined_var_name, None)
        repl.variables.pop(f"_coll_{coll_name}_mapping", None)
        clear_index(combined_var_name)

        text = (f"🗑️ Coleção '{coll_name}' removida (associação + índice combinado).\n"
                f"As variáveis membras NÃO foram apagadas — use rlm_clear para isso.")
        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro ao remover coleção: {e}"}],
            "isError": True
        }


def rlm_search_collection(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    try:
        persistence = get_persistence()
        coll_name = arguments["collection"]
        raw_terms = arguments["terms"]
        limit = arguments.get("limit", 10)
        offset = arguments.get("offset", 0)
        # Guard #3: snippet configurável (default 150; clamp p/ não inflar contexto).
        snippet_len = max(40, min(400, int(arguments.get("snippet_len", 150))))
        # Guard (c): termo entre aspas = busca exata explícita → não tokeniza.
        terms, quoted_flags, all_quoted = parse_quoted_terms(raw_terms)

        # Obter variáveis da coleção
        var_names = persistence.get_collection_vars(coll_name)
        if not var_names:
            return {
                "content": [
                    {"type": "text", "text": f"Coleção '{coll_name}' vazia ou não existe."}
                ],
                "isError": True
            }

        # === OPÇÃO C: Tentar usar índice combinado primeiro ===
        combined_var_name = f"_coll_{coll_name}_combined"
        combined_index = get_index(combined_var_name)
        mapping_var = f"_coll_{coll_name}_mapping"

        all_results = {}
        bm25_ranked_hits = []  # ordem GLOBAL de relevância (p/ paginação)
        terms_via_index = []
        terms_via_fallback = []
        indexed_terms_count = 0
        used_bm25 = False
        is_phrase_coll = any(' ' in t.strip() for t in terms)

        # === BM25 sobre o índice combinado (cobre TODO o vocabulário) ===
        # Substitui o split indexado-vs-fulltext. Frase literal cai no legacy.
        if (combined_index and not is_phrase_coll
                and mapping_var in repl.variables
                and combined_var_name in repl.variables):
            var_mapping = repl.variables[mapping_var]
            combined_text = repl.variables[combined_var_name]
            bm25_hits = combined_index.search_bm25(
                terms, combined_text, limit=(limit + offset + 50), offset=0
            )
            if combined_index._bm25_built and not combined_index._bm25_degraded:
                used_bm25 = True
                index_stats = combined_index.get_stats()
                indexed_terms_count = index_stats.get('indexed_terms', 0)
                label = " ".join(terms)
                for h in bm25_hits:
                    # BM25 é 0-indexed; var_mapping é 1-indexed (convenção
                    # full-text correta) → +1. Sentinel garante que o segmento
                    # não cruza fronteira de var.
                    mapped = var_mapping.get(h["line"] + 1)
                    if not mapped:
                        continue
                    orig_var, orig_linha = mapped
                    all_results.setdefault(orig_var, {}).setdefault(label, []).append({
                        'linha': orig_linha,
                        'contexto': h.get("text", "")[:snippet_len],
                    })
                    # Lista plana na ordem do ranking BM25 — a exibição
                    # pagina por ela (não por bucket var→termo)
                    bm25_ranked_hits.append({
                        'var': orig_var,
                        'linha': orig_linha,
                        'contexto': h.get("text", "")[:snippet_len],
                    })

        if not used_bm25 and combined_index and mapping_var in repl.variables:
            # Usar índice combinado + fallback híbrido (legacy)
            var_mapping = repl.variables[mapping_var]
            index_stats = combined_index.get_stats()
            indexed_terms_count = index_stats.get('indexed_terms', 0)
            available_terms = set(combined_index.terms.keys())

            # Separar termos indexados vs não-indexados
            for term in terms:
                if term.lower() in available_terms:
                    terms_via_index.append(term)
                else:
                    terms_via_fallback.append(term)

            # Buscar termos indexados via índice
            if terms_via_index:
                results = combined_index.search_multiple(terms_via_index, require_all=False)
                if results:
                    for term, matches in results.items():
                        for m in matches:
                            # Matches do TextIndex são 0-indexed
                            # (create_index/_live_scan_term); var_mapping
                            # é 1-indexed → +1 (era o "bug 0-vs-1").
                            linha_combined = m['linha'] + 1
                            if linha_combined in var_mapping:
                                orig_var, orig_linha = var_mapping[linha_combined]
                                if orig_var not in all_results:
                                    all_results[orig_var] = {}
                                if term not in all_results[orig_var]:
                                    all_results[orig_var][term] = []
                                all_results[orig_var][term].append({
                                    'linha': orig_linha,
                                    'contexto': m['contexto']
                                })

            # Buscar termos não-indexados via full-text
            if terms_via_fallback and combined_var_name in repl.variables:
                combined_text = repl.variables[combined_var_name]
                for term in terms_via_fallback:
                    term_lower = term.lower()
                    for line_num, line in enumerate(combined_text.split('\n'), start=1):
                        if term_lower in line.lower():
                            if line_num in var_mapping:
                                orig_var, orig_linha = var_mapping[line_num]
                                if orig_var not in all_results:
                                    all_results[orig_var] = {}
                                if term not in all_results[orig_var]:
                                    all_results[orig_var][term] = []
                                all_results[orig_var][term].append({
                                    'linha': orig_linha,
                                    'contexto': line.strip()
                                })
        elif not used_bm25:
            # Fallback total: buscar em índices individuais ou full-text
            terms_via_fallback = terms[:]
            for var_name in var_names:
                index = get_index(var_name)
                if index:
                    results = index.search_multiple(terms, require_all=False)
                    if results:
                        all_results[var_name] = results
                        terms_via_fallback = []  # Encontrou no índice

            # Se não encontrou em índices individuais, tenta full-text
            if not all_results and combined_var_name in repl.variables and mapping_var in repl.variables:
                combined_text = repl.variables[combined_var_name]
                var_mapping = repl.variables[mapping_var]
                for term in terms:
                    term_lower = term.lower()
                    for line_num, line in enumerate(combined_text.split('\n'), start=1):
                        if term_lower in line.lower():
                            if line_num in var_mapping:
                                orig_var, orig_linha = var_mapping[line_num]
                                if orig_var not in all_results:
                                    all_results[orig_var] = {}
                                if term not in all_results[orig_var]:
                                    all_results[orig_var][term] = []
                                all_results[orig_var][term].append({
                                    'linha': orig_linha,
                                    'contexto': line.strip()
                                })

        # === Auto-tokenize fallback (phrase-trap) — guardas a/b/c ===
        # A busca acima casa a FRASE literalmente (substring numa linha). Se a
        # frase não bateu mas havia frase, quebramos em tokens e re-buscamos
        # (AND antes de OR, word-boundary). Não dispara se o usuário pediu
        # busca exata via aspas (all_quoted).
        fallback_note = None
        if (not all_results and is_phrase_coll and not all_quoted
                and combined_var_name in repl.variables
                and mapping_var in repl.variables):
            tok_source = [t for t, q in zip(terms, quoted_flags) if not q]
            # Guard (c) mixed: termo QUOTED não é tokenizado E vira
            # filtro obrigatório no scan (intenção explícita de exato)
            required_literals = [t for t, q in zip(terms, quoted_flags) if q]
            tokens = tokenize_for_fallback(tok_source)
            if tokens:
                scan_results, scan_mode = tokenized_collection_scan(
                    repl.variables[combined_var_name],
                    repl.variables[mapping_var],
                    tokens,
                    snippet_len=snippet_len,
                    required_literals=required_literals,
                )
                if scan_results:
                    all_results = scan_results
                    used_bm25 = False
                    terms_via_index = []
                    terms_via_fallback = tokens
                    fallback_note = (scan_mode, tokens, " ".join(terms),
                                     required_literals)

        if not all_results:
            # Nenhum resultado nem no índice nem no fallback nem na tokenização
            text = f"Nenhum resultado para {terms} na coleção '{coll_name}'\n"
            if all_quoted:
                text += ("\n(busca exata por aspas — não tokenizei. Remova as aspas "
                         "para tentar por tokens.)")
            elif any(quoted_flags):
                text += ("\n(termo entre aspas é filtro EXATO obrigatório — nenhuma "
                         "linha satisfez o literal quoted junto com os tokens. "
                         "Remova as aspas para busca só por tokens.)")
            text += ("\n💡 A busca de coleção é LEXICAL (casa palavras/tokens). "
                     "Passe termos como array — [\"a\",\"b\"] — em vez de frase. "
                     "Para recall por significado/sinônimo/cross-idioma, use "
                     "rlm_search_index(var=..., mode=\"hybrid\") por fonte.")
        else:
            lines = [f"🔍 Busca em '{coll_name}': {', '.join(terms)}", ""]
            # Guard (b): transparência — deixa explícito que NÃO foi a busca exata.
            if fallback_note:
                _mode, _toks, _orig, _req = fallback_note
                banner = format_fallback_banner(_mode, _toks, _orig)
                if _req:
                    banner.append(
                        f"   🔒 Filtro exato obrigatório (aspas): {', '.join(_req)}")
                    banner.append("")
                lines = banner + lines

            # Stats de busca híbrida
            if used_bm25:
                lines.append(f"🔎 Ranking BM25 por relevância ({len(var_names)} vars na coleção)")
                lines.append("")
            elif terms_via_index and terms_via_fallback:
                lines.append(f"📊 Busca híbrida: {len(terms_via_index)} via índice, {len(terms_via_fallback)} via full-text")
                lines.append(f"   ✅ Indexados: {', '.join(terms_via_index)}")
                lines.append(f"   🔄 Full-text: {', '.join(terms_via_fallback)}")
                if indexed_terms_count:
                    lines.append(f"   ℹ️  {indexed_terms_count} termos disponíveis no índice")
                lines.append("")
            elif terms_via_fallback:
                lines.append(f"🔄 Busca full-text ({len(terms_via_fallback)} termos não indexados)")
                if indexed_terms_count:
                    lines.append(f"   ℹ️  {indexed_terms_count} termos disponíveis no índice")
                lines.append("")
            elif combined_index:
                lines.append(f"✅ Usando índice combinado ({len(var_names)} vars, {indexed_terms_count} termos)")
                lines.append("")

            if used_bm25 and bm25_ranked_hits:
                # Paginação GLOBAL na ordem de relevância (P1 Codex
                # 2026-06-02): paginar por bucket var→termo descartava
                # o ranking — limit=10 mostrava 10 POR var e a página 1
                # podia não conter os melhores hits da coleção.
                page = bm25_ranked_hits[offset:offset + limit]
                total_global = len(bm25_ranked_hits)
                start_idx = offset + 1 if page else 0
                lines.append(
                    f"  📌 '{', '.join(terms)}' (mostrando {start_idx}-"
                    f"{offset + len(page)} de {total_global}, ordem de relevância global)"
                )
                cur_var = None
                for h in page:
                    if h['var'] != cur_var:
                        cur_var = h['var']
                        lines.append(f"📄 {cur_var}:")
                    lines.append(f"      L{h['linha']}: {h['contexto'][:snippet_len]}...")
                lines.append("")
                n_docs = len({h['var'] for h in bm25_ranked_hits})
                lines.append(f"📊 Total: {total_global} ocorrências em {n_docs} documento(s)")
            else:
                for var_name, results in all_results.items():
                    lines.append(f"📄 {var_name}:")
                    for term, matches in results.items():
                        total_term = len(matches)
                        paginated = matches[offset:offset + limit]
                        start_idx = offset + 1 if paginated else 0
                        end_idx = offset + len(paginated)
                        lines.append(f"  📌 '{term}' ({total_term} ocorrências, mostrando {start_idx}-{end_idx})")
                        for m in paginated:
                            lines.append(f"      L{m['linha']}: {m['contexto'][:snippet_len]}...")
                    lines.append("")

                total_matches = sum(
                    sum(len(matches) for matches in results.values())
                    for results in all_results.values()
                )
                lines.append(f"📊 Total: {total_matches} ocorrências em {len(all_results)} documento(s)")
            if not fallback_note:
                lines.append(
                    "ℹ️ Busca lexical (tokens). Recall por significado/cross-idioma: "
                    "rlm_search_index(var=..., mode=\"hybrid\") por fonte."
                )
            text = "\n".join(lines)

        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro na busca: {e}"}
            ],
            "isError": True
        }
