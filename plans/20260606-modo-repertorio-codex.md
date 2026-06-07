# Modo-repertório — plano consolidado (2026-06-06)

Repertorização homeopática no rlm-mcp. Fonte MVP: var `kent_repertorio` (Eizayaga ES,
4,28M chars, 138.738 linhas). Decisão do usuário: ranking BINÁRIO (CAPS=grau 3,
resto=grau 1; grau 2/itálico perdido na extração — campo `grade` fica no modelo p/
upgrade futuro via re-extração do PDF).

## Investigação live (NÃO re-derivar — validado 2026-06-06 contra produção)

Classificação por CONTEÚDO das 138.738 linhas (headings `#` são ruído de OCR em linhas
arbitrárias, NÃO estrutura):

| classe | linhas | regra |
|---|---|---|
| entry | 61.408 | `texto: rem, rem, REM.` — split no ÚLTIMO `:`, ≥60% do tail são tokens de remédio válidos |
| blank | 59.987 | vazia |
| continuation | 5.467 | lista pura de remédios (≥2 tokens, ≥70% válidos, sem `:`) — ANEXAR à entry anterior |
| prose | 4.986 | front matter/índices/perdidas |
| header | 4.160 | termina com `:` sem remédios → rubrica corrente |
| colon_prose | 1.857 | tem `:` mas tail não valida (front matter, glossário, perdas OCR) |
| crossref | 486 | `(ver X).` — pular |
| page | 387 | `--- Página N ---` — pular sem alterar contagem de linhas |

- Token de remédio: `[A-Za-z][A-Za-z-]{1,11}\.?` ; grau 3 sse token original isupper().
- Headers `(cont.)` (1.879) = continuação de rubrica após quebra de página — manter rubrica corrente.
- Front matter L1–~2315 (capa/prólogo/GLOSSÁRIO) — excluir do parse de rubricas; só inicia
  no primeiro capítulo real (`## PSIQUISMO` ~L2316). MAS harvest do glossário L311–2131:
  `ABBREV: nome completo` → mapa de display + cross-check de vocabulário.
- OCR: 224.868 tokens, 1.210 abrevs freq≥10 = 92,4% da massa. Micro-fixes: `1`→`i` em token,
  pontos internos removidos. Levenshtein CONSERVADOR: só freq≤3 → candidato ÚNICO no vocab
  estável; len≤4 → max dist 1 (calc↔carb dist 2!); empate → descarta. `aliea`→`ail` é dist 3
  (NÃO corrigível — descartar é correto).
- Duplo colon: `comiendo: mej.: graph.` → split no último `:`.

## Arquitetura (Codex, mantida)

- `src/rlm_mcp/repertory.py` — módulo PURO: dataclasses (slots=True), parser content-based,
  cache lazy thread-safe por fingerprint (sha256+len; auto-invalida se var recarregada;
  lock por var com double-check, padrão `indexer._get_bm25_lock`), canonicalização,
  busca, ranking. NÃO armazenar original_line por entry (derivar do texto na exibição).
- `src/rlm_mcp/tools/handlers/repertory_tools.py` — handler router `rlm_repertorio`
  (assinatura `(arguments, ctx)`), actions: `buscar_rubrica`, `repertorizar`, `info`.
- Registro: `handlers/__init__.py` (TOOL_HANDLERS) + schema em `tools/schemas.py`
  (estilo rlm_collection, required=["action"]).
- `response_formatter.py`: format_repertory_search / format_repertorization /
  format_repertory_info com get_verbosity() (produção = compact).

## Actions

- `buscar_rubrica(query, limit=10, offset=0)`: accent-fold + lower; mini-dict PT→ES
  (medo→temor etc.); match exact-token > substring; fuzzy SÓ como fallback se < limit
  resultados. Resultado: `kent_repertorio:L{n} CAPÍTULO > RUBRICA > texto | rem.(g), ...`
  + total p/ paginação. 0 hits → sugerir rlm_search_index hybrid.
- `repertorizar(rubrics=[ids], sort="coverage"|"score")`: IDs `var:L123` (texto aceito só
  com match único; ambíguo → erro listando candidatos). Tabela remédio×rubrica com grau,
  ranking coverage desc → score desc → nome (ou score-first se sort=score). Rodapé honesto:
  grau 2 indisponível na fonte.
- `info()`: stats do parse (entries, continuações anexadas, headers, perdas prose/colon_prose,
  descartados OCR, vocab estável, fingerprint curto, build ms). Contabilidade honesta de perdas.

## Citações (lição do P0 line-mapping)

`line_no` 1-indexed da var REAL; invariante testado: a linha citada contém o texto da entry.
Continuações: citar a linha da entry-base (range opcional depois).

## Testes

- `tests/test_repertory.py` + `tests/fixtures/kent_repertory_sample.txt` (verbatim com
  todos os fenômenos: CAPS, continuation, (cont.), crossref, página, duplo-colon, OCR noise,
  glossário, front matter).
- Casos: classificação, continuação anexada, grau 3, line-mapping exato, canonicalização
  conservadora (não corrige empate; len≤4 dist 1), ranking coverage/score, handler via
  hs.call_tool (var ausente → isError; compact e normal via monkeypatch get_verbosity).
- `.gitignore`: allowlist dos arquivos novos de teste.

## QA live (scripts/qa_extensive.py)

`sec_repertorio(c, r)` antes de cleanup: EXPECTED_TOOLS += rlm_repertorio; info (SKIP se
kent ausente); buscar_rubrica "abandono sentimiento" → acha ABANDONO/sentimiento de com
AUR grau 3 + linha; repertorizar 2 IDs → ranking estável com cobertura.

## Sequência

implementar → suite local (baseline 190) → workflow adversarial de review → commit →
push main (deploy Dokploy) → harness live (baseline 66) → repertorização real de validação.

---
Apêndice A: plano original do Codex em /tmp/codex-plan.md (L4008+).
Apêndice B: crítica estruturada na conversa de 2026-06-06 (sessão 50e467d4).
