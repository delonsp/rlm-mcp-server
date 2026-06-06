# Fix: P0 line-mapping da busca de coleção — plano consolidado

**Data**: 2026-06-06 · **Workflow**: /plan-code-codex + revisão item a item com 4 subagentes
**Bug**: busca de coleção citava `var:linha` deslocado (+2 por var de profundidade; medido live: +16 na 8ª var — sankaran L3062 reportada, real L3046). Texto certo, linha errada.

## Causa raiz (validada empiricamente)
`"\n".join(combined_parts)` insere um `\n` entre partes — cada parte ocupa `count('\n')+1` linhas no combinado. O header (4 `\n` = **5 linhas**) era contado como **4** (`header.count('\n')`) → off-by-1 POR header, acumulando. Builder **triplicado** com o mesmo bug: lifespan (:508), collection_add (:1739), collection_rebuild (:1852).

Bug secundário ("0-vs-1", também corrigido): matches do `TextIndex` (`create_index`/`_live_scan_term`) são **0-indexed**, e o consumer legacy (`search_multiple`, http_server ~:2001) fazia lookup direto no `var_mapping` 1-indexed sem `+1` → hits dropados/deslocados quando BM25 degrada.

## O que mudou
1. **`src/rlm_mcp/collection_builder.py`** (novo): `build_collection_combined(var_names, variables) -> (combined_text, var_mapping, vars_included)` — função pura, linha de início derivada das partes reais (`+= part.count('\n') + 1`), headers sem mapping, cheque de sanidade fail-loud (1ª linha mapeada de cada var conferida contra o combinado real).
2. **3 call sites** substituídos pelo builder (lifespan/add/rebuild), preservando mensagens e edge-cases de cada um (vazio: silent/msg/erro). `vars_included` na tupla atende o rebuild (achado de subagente: assinatura de 2 itens era insuficiente).
3. **Consumer legacy** (:~2010): `linha_combined = m['linha'] + 1`.
4. **Inalterados** (verificados corretos): BM25 `h["line"]+1` (segmentos 0-indexed confirmados), fulltext `enumerate(start=1)`, `tokenized_collection_scan` (1-indexed interno). Walk-forward do BM25 descartado: o sentinel regex cobre todas as linhas não-vazias do header e blank lines nunca iniciam segmento.

## Prova empírica
- Combined antigo × novo: **byte-idêntico** (zero mudança em ranking/snippets).
- Mapping antigo: 5/5 entradas erradas no cenário de teste; novo: 0/5. Shift da 2ª var (+2) reproduzido (12 vs real 14).
- Canário em teste: começos corretos 6/14/21/27 (antigo diria 5/12/18/23).

## Verificações dos subagentes (item a item)
- `_coll_*` **nunca persiste** no SQLite (memória apenas) → rollback-safe; lifespan rebuilda TODAS as coleções (list_collections sem filtro) DEPOIS do restore; uvicorn só aceita conexões pós-lifespan → deploy corrige `homeopatia` sem ação manual.
- Rebuild parcial silencioso se var não restaurada (pré-existente, fora do escopo).
- BM25 lazy: 1ª busca pós-deploy paga o build (pré-existente).
- **Fora do escopo (decisão pendente)**: display single-var (`rlm_search_index`) é 0-indexed CONSTANTE em todas as telas (formatter sem `+1`, produtores 0-indexed) — corrigir mudaria todos os `L####` já citados.

## Testes
- `tests/test_collection_builder.py` (8): invariante forte linha-a-linha com header real, cobertura inversa, canário aritmético, ordem, edge-cases.
- `tests/test_collection_search_line_mapping.py` (5): integração real via call_tool (persistence tmpdir + rebuild + search): BM25 cita L1 (antigo: L3), legacy +1, fulltext, invariante do mapping em memória.
- `tests/test_bm25.py`: convenção 0-indexed do search_bm25 (arquivo agora versionado).
- `tests/test_collection_phrase_fallback.py`: adversarial 1ª linha do tokenized scan.
- conftest: reset do singleton `_persistence` (achado de subagente).

## Pós-deploy (verificação live)
Busca na coleção `homeopatia` real → conferir que o texto da linha citada bate com `var.split('\n')[L-1]` via rlm_execute.

## Apêndices
- Plano original do Codex: /tmp/codex-plan-linemap.md
- Vereditos dos 4 subagentes: builder/persistência, consumers/convenções, deploy/lifespan, testes/canário.
