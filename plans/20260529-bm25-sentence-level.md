# Plano: BM25 sentence/paragraph-level para o RLM MCP Server
Data: 2026-05-29
Spec de origem: conversa (sessão "BM25"); memória [[project-dampening-shipped]], [[project-lazy-embed-and-contamination]]

## Visão geral
Adicionar ranking BM25 (Okapi, k1=1.2, b=0.75) com índice invertido `term→[(seg_id, tf)]` em granularidade de segmento (~120 tokens, quebrando em linha em branco), morando **dentro do `TextIndex`**. Resolve as 2 otimizações pendentes: (1) resultados keyword ranqueados por relevância em vez de "encontrado"; (2) fim do vocabulário fixo de 64 termos — todo termo indexado. A perna keyword da fusão RRF hybrid passa a usar **rank de relevância BM25** (RRF é rank-based — o *valor* do score BM25 não entra na fusão, só ordena a perna keyword; o score aparece no display). Lazy-build no load (padrão lazy-embed já aprovado), sem persistir no SQLite.

**Achado de medição que reposiciona o plano** (dados reais, 3 corpora):
- `recode_ciencia`: 5.4M chars, **98.5% whitespace** de layout (pdfplumber multi-coluna) → 110k chars reais.
- `recode_protocolos_geral`: 11M chars, 30.9% whitespace + NUL bytes.
- `livro_seyfried`: 1.2M chars, 15.2% whitespace (limpo, EPUB).
- Consequências: (a) memória do índice BM25 é trivial (~0.1–1.5 MB/var — corpora são densos em whitespace, não em tokens); (b) tokenização BM25 é imune a whitespace (runs de espaço já são delimitadores); (c) o whitespace degrada o **semantic** (chunks de 512 chars com ~10 chars reais) — explica a "Otimização #1: scores 0.47–0.59". **Decisão do usuário: normalização dos vars armazenados é TAREFA SEPARADA** (não neste plano). BM25 normaliza só internamente (tokenização já é imune; snippet de display é colapsado).

## Decisões arquiteturais (confirmadas pelo usuário, FASE 2)
| # | Decisão | Escolha |
|---|---------|---------|
| D1 | Normalização de whitespace dos vars armazenados | **Tarefa separada depois** (BM25 só normaliza internamente p/ display) |
| D2 | Comportamento de `mode="keyword"` | **BM25-ranked vira o default**; substring scan = fallback p/ frase exata e `require_all` |
| D3 | Persistência do índice BM25 | **Lazy-rebuild no load** (não serializa; ~segundos na 1ª busca) |
| D4 | Granularidade do segmento | **~120 tokens**, quebrando em linha em branco, preservando mapeamento de linha |

## Requisitos rastreados da spec
| ID | Requisito | Como será atendido | Arquivo(s) |
|----|-----------|---------------------|------------|
| R1 | Ranquear keyword por relevância BM25 (k1=1.2, b=0.75) | `TextIndex.search_bm25()` com IDF `log((N-n+0.5)/(n+0.5)+1)` e TF-norm padrão | `indexer.py` |
| R2 | Eliminar vocabulário fixo de 64 termos | Índice invertido cobre todo o vocabulário do corpus (build a partir do texto) | `indexer.py` |
| R3 | Granularidade segmento (~120 tok), não linha | `_segment_lines(text, target_tokens=120)` empacota linhas consecutivas, quebra em linha em branco; preserva `(line_start, line_end)` originais | `indexer.py` |
| R4 | Perna keyword do RRF usa rank BM25 | `_reciprocal_rank_fusion` recebe lista ranqueada BM25 (não dict `{term:matches}`) | `indexer.py` |
| R5 | `mode="keyword"` default = BM25 | handler roteia keyword p/ `search_bm25`; fallback substring p/ frase/`require_all` | `http_server.py`, `indexer.py` |
| R6 | Dampening preservado | `_overlap_text` = texto cru do segmento (contém o termo → protegido); chamada de dampening intacta na fusão | `indexer.py` |
| R7 | Lazy-build, sem persistir | campos BM25 runtime-only (fora de `to_dict()`); `search_bm25` builda se `_bm25_built` falso | `indexer.py` |
| R8 | Coleções herdam BM25 | combined index é `TextIndex` → `search_bm25` funciona; busca de coleção roteada p/ BM25 | `http_server.py` |
| R9 | API dos 19 tools inalterada | assinatura de `rlm_search_index` igual; só muda formato interno do resultado keyword | `tools/schemas.py` (sem mudança) |
| R10 | Display mostra segmento + score | `format_bm25_results` (compact/verbose); snippet com whitespace colapsado | `response_formatter.py` |

## O que NÃO estamos fazendo (anti-scope creep)
- **NÃO** normalizar/re-salvar os vars armazenados nem re-embedar (D1 → tarefa separada).
- **NÃO** mexer em embeddings, chunking semântico ou `vector_index.py`.
- **NÃO** persistir BM25 no SQLite (D3).
- **NÃO** remover o índice legacy de 64 termos nem `_live_scan_term` (viram fallback).
- **NÃO** mudar assinatura dos 19 tools nem `tools/schemas.py`.
- **NÃO** trocar RRF para score-weighted (mantém rank-based `1/(k+rank)` — preserva matemática validada + dampening).
- **NÃO** corrigir contaminação NUL-byte dos 5 vars (separada; ver [[project-lazy-embed-and-contamination]]).

## Arquitetura

### Arquivos novos
- `src/rlm_mcp/stopwords.py` — set `STOPWORDS` PT+EN (~300 termos). Função única, sem deps. Reaproveita/expande o `_QUERY_STOPWORDS` do dampening.

### Arquivos modificados
- `src/rlm_mcp/indexer.py` (núcleo):
  - `TextIndex`: +campos runtime-only (NÃO em `to_dict()`/`from_dict()`): `bm25_postings: dict[str, list[tuple[int,int]]]`, `bm25_doc_len: list[int]`, `bm25_segments: list[tuple[int,int]]`, `bm25_avgdl: float`, `bm25_n: int`, `_bm25_built: bool=False`.
  - `_bm25_tokenize(text) -> list[str]` — `[w for w in _KEY_TERM_RE.split(text.lower()) if len(w)>1 and w not in STOPWORDS]` (mantém freq; reusa o regex Unicode do dampening).
  - `_segment_lines(text, target_tokens=120) -> list[tuple[int,int]]` — itera linhas (0-indexed, convenção atual), acumula tokens; fecha segmento ao atingir alvo OU em linha em branco com conteúdo acumulado; linha única > alvo vira segmento próprio (line_start==line_end). Preserva mapeamento de linha.
  - `TextIndex.build_bm25(source_text, target_tokens=120)` — segmenta, conta `Counter` por segmento, popula postings/doc_len/segments/avgdl/n; loga `n_seg`, vocab, postings, MB estimado; set `_bm25_built=True`. Guard: se postings estimados > teto sane (ex: 5M), loga e segue (sem cap silencioso).
  - `TextIndex.search_bm25(query_terms, source_text, limit=20, offset=0, k1=None, b=None) -> list[dict]` — builda se preciso; bag de query via `_bm25_tokenize`; acumula score por seg_id; ranqueia desc; fatia `[offset:offset+limit]`; cada hit `{"line":line_start,"line_end":...,"score":...,"text":_normalize_snippet(slice),"_overlap_text":raw_slice}`.
  - `_normalize_snippet(text)` — colapsa whitespace p/ display (só o snippet; NÃO toca tokenização nem `_overlap_text`).
  - `hybrid_search`: perna keyword passa a chamar `search_bm25` (lista ranqueada). Para `mode="keyword"`: retorna `result["keyword_ranked"]` (BM25) salvo se `require_all` ou termo-frase → fallback `search_multiple` em `result["keyword_results"]`. Para `mode="hybrid"`: passa lista BM25 ao RRF.
  - `_reciprocal_rank_fusion(bm25_hits: list, semantic_results, terms, ...)`: keyword leg agora itera `bm25_hits` (rank por ordem da lista), keying por `line`; `_overlap_text` do hit BM25; resto idêntico (dampening + sort). Branches de promoção (só-keyword / só-semantic) ajustados p/ lista BM25.
  - Env: `RLM_BM25_ENABLED` (true), `RLM_BM25_TARGET_TOKENS` (120), `RLM_BM25_K1` (1.2), `RLM_BM25_B` (0.75).
- `src/rlm_mcp/http_server.py`:
  - Handler `rlm_search_index` (L1349–1430): keyword mode roteia p/ `hybrid_search(mode="keyword")` (centraliza) OU chama `search_bm25` direto; usa `keyword_ranked` quando presente → `format_bm25_results`; senão fallback atual. `source_str` já disponível.
  - Busca de coleção (L1740–1810): rotear `combined_index.search_bm25(terms, combined_text, ...)`; remove dependência de `combined_index.terms.keys()` p/ split indexado-vs-fulltext (BM25 cobre tudo). Mapeamento de linha→(var,linha) via `_coll_*_mapping` preservado.
  - Restore (L368): `from_dict` reconstrói só keyword legacy; BM25 fica lazy (1ª busca builda). Sem mudança de serialização.
- `src/rlm_mcp/response_formatter.py`:
  - `format_bm25_results(ranked, terms, var_name, offset, limit, verbosity, max_results)` — compact `[kw:var | N hits | L123(2.45) | ...]`; verbose com L-ref, score e snippet[:120].
  - `format_hybrid_search`: +branch p/ `keyword_ranked` (delega a `format_bm25_results`).
- `src/rlm_mcp/services/persistence_service.py`: (opcional) helper `ensure_bm25(var_name, source_text)` espelhando `ensure_embeddings` — só se quisermos pré-aquecer; default é build dentro de `search_bm25` (sem helper).

### Schema/banco
Nenhuma mudança (D3 lazy-rebuild). `indices` table inalterada; `to_dict()` NÃO ganha chave BM25.

## Armadilhas (red herrings)
- `code_parser.py` (busca estrutural de código) — não relacionado; não tocar.
- `vector_index.py` chunking de 512 chars — tentador "consertar chunk", mas é a tarefa de normalização (D1, fora de escopo).
- `_live_scan_term` / `search_multiple` — manter como fallback; não deletar.
- `terms_count` em `save_index` (conta chaves do dict) — cosmético; não refatorar.
- Convenção de linha: sistema usa **0-indexed** (`enumerate`), Matryoshka usa 1-indexed — usar 0-indexed p/ casar com semantic/keyword.

## Critérios de verificação
- [ ] `python3 -c "import ast; ast.parse(open('src/rlm_mcp/indexer.py').read()); ast.parse(open('src/rlm_mcp/response_formatter.py').read()); ast.parse(open('src/rlm_mcp/stopwords.py').read())"` — sintaxe OK
- [ ] `python3 -c "import sys; sys.path.insert(0,'src'); from rlm_mcp.indexer import TextIndex, create_index; from rlm_mcp.stopwords import STOPWORDS; ..."` — imports OK
- [ ] Unit local: build_bm25 em texto sintético (3 segmentos, termo raro em 1) → segmento com termo raro ranqueia 1º; termo inexistente → []; `require_all` cai no fallback substring.
- [ ] Live: `search_bm25` em `recode_ciencia` p/ termo real → lista ranqueada por score (não "encontrado/não").
- [ ] Live: `mode="hybrid"` em var com embeddings → perna keyword com score BM25 na fusão; dampening ainda rebaixa ghost (controle PT-em-EN do [[project-dampening-shipped]]).
- [ ] Live: `mode="keyword"` default agora ranqueado; `require_all=True` ainda funciona (fallback).
- [ ] Live: busca de coleção retorna BM25; mapeamento var/linha intacto.
- [ ] Restart (redeploy) → 1ª busca BM25 reconstrói lazy (log "BM25 built"), sem erro.
- [ ] Memória pós-build de 3 vars grandes via `/health` < teto (esperado: +poucos MB).

## Estado final desejado
`rlm_search_index(mode="keyword")` retorna segmentos ranqueados por relevância BM25 (score visível), sem vocabulário fixo. `mode="hybrid"` funde keyword(BM25)+semantic via RRF com score real nas duas pernas, dampening ativo. Coleções idem. Índice BM25 reconstruído automaticamente e silenciosamente após restart na 1ª busca. Zero mudança na API dos tools, zero crescimento do SQLite, zero impacto em embeddings. `require_all`/frase exata continuam via substring.

## Checklist de implementação (ordem)
1. [ ] `stopwords.py` — set PT+EN — dep: nenhuma
2. [ ] `indexer.py`: `_bm25_tokenize`, `_segment_lines`, `_normalize_snippet` — dep: #1
3. [ ] `indexer.py`: campos BM25 em `TextIndex` + `build_bm25` — dep: #2
4. [ ] `indexer.py`: `search_bm25` — dep: #3
5. [ ] `indexer.py`: refatorar `_reciprocal_rank_fusion` p/ lista BM25 + branches de promoção — dep: #4
6. [ ] `indexer.py`: `hybrid_search` roteando keyword/hybrid p/ BM25 + fallback require_all/frase — dep: #5
7. [ ] `response_formatter.py`: `format_bm25_results` + branch em `format_hybrid_search` — dep: #4
8. [ ] `http_server.py`: handler keyword → BM25 + formatter — dep: #6,#7
9. [ ] `http_server.py`: busca de coleção → `search_bm25` — dep: #6
10. [ ] Env vars + defaults + guard de memória com log — dep: #3
11. [ ] Unit tests locais (sintético) — dep: #4
12. [ ] Smoke local (import) + commit + push (deploy Dokploy derruba MCP → reconectar c/ /mcp) — dep: todos
13. [ ] Validação live (tabela de critérios) — dep: #12

## Notas para o implementador
- Style guide: seguir convenções do `indexer.py` (docstrings PT, dataclass, type hints `list[...]`).
- Reusar `_KEY_TERM_RE` já existente (não criar regex novo); STOPWORDS no módulo novo.
- 0-indexed para linhas (casar com `_live_scan_term`/`vector_index`).
- NÃO incluir features extras (sem score-weighted RRF, sem persistência, sem normalização de vars).
- Build BM25 é idempotente e gated por `_bm25_built`; seguro chamar a cada busca.
- Deploy = push main; avisar usuário que o MCP cai e precisa `/mcp`.

---

## Refinamentos pós-crítica adversarial (Codex, 2026-05-29)
Crítica adversarial do Codex (read-only). Pontos genuínos incorporados; decisão do usuário: manter **RRF rank-based** + fix de chave (não score-weighted).

### P0 — corrigir antes de mergear
- **[RRF-KEY] Fusão por sobreposição de range, não `line` exato** (substitui parte de R4/R6). Segmentos BM25 (~120 tok) e chunks semânticos quase nunca têm `line_start` idêntico → keying por `line` exato faz a fusão degenerar em concatenação (pernas não se reforçam — já é problema latente hoje, BM25 piora). **Fix:** `_reciprocal_rank_fusion` funde keyword(BM25) e semantic quando seus ranges `[line_start, line_end]` **se sobrepõem** (ou ficam dentro de uma janela de N linhas). Implementação: ordenar hits por line_start, merge por overlap; a entrada fundida herda o menor line_start como chave de display e acumula `1/(k+rank)` de cada perna. Dampening continua sobre `_overlap_text` do item fundido.
- **[CONCORRÊNCIA] Build thread-safe** (reforça R7). FastAPI pode disparar 2 buscas no mesmo índice → ambas chamam `build_bm25` e mutam os dicts. **Fix:** build em estruturas **locais** e **atribuição atômica** dos campos (`self.bm25_postings = local_postings; ...; self._bm25_built = True` por último), protegido por um `threading.Lock` por var (dict module-level `_bm25_locks`). Nota: o lazy-embed (`ensure_embeddings`) tem a MESMA race latente hoje — aplicar o mesmo padrão lá (item da varredura de bugs).
- **[COLEÇÃO-FRONTEIRA] Quebra de segmento no sentinel de var** (reforça R8). O combined index concatena vars com header `=== VARIÁVEL: x ===`. **Fix:** `_segment_lines` força fim de segmento ao encontrar linha que casa o sentinel (regex `^=+$` ou `=== VARIÁVEL:`), garantindo que nenhum segmento cruze fronteira de var → mapeamento linha→(var,linha) intacto.

### P1 — incorporar no design
- **[REQUIRE_ALL] BM25 + pós-filtro, não substring** (revisa D2/R5). `require_all=True` NÃO cai pra substring (perderia ranking no caso multi-termo, o mais importante). **Fix:** roda BM25 normal e **filtra os segmentos que contêm todos os query tokens** (set de tokens do segmento ⊇ set de query tokens), preservando ordem BM25. Substring scan fica só para **frase literal** (termo com espaço quando o usuário quer match exato).
- **[TOKENIZER] Acento-folding + min-len configurável** (revisa R3). `.lower()` puro separa `câncer`/`cancer` e quebra em OCR sujo. **Fix:** normalizar NFKD + remover combining marks no index E na query (match acento-insensitive — esperado em PT). `MIN_TOKEN_LEN` via env (default 2, casa Matryoshka) **com flag explícita**: derruba tokens de 1 char tipo "vitamina **D**", "**K**2" — limitação conhecida do corpus biomédico; documentar e deixar configurável p/ baixar a 1 se necessário. NUL bytes já viram delimitador no regex (não aparecem em token).
- **[SNIPPET-CACHE] Cachear linhas no build** (revisa R10). NÃO fazer `source_text.splitlines()` por hit (O(texto)/hit). **Fix:** `build_bm25` guarda `self._lines = source_text.split('\n')` (runtime-only) uma vez; `search_bm25` fatia `self._lines[line_start:line_end+1]` por hit. Snippet de display = `_normalize_snippet`; `_overlap_text` = junção crua.
- **[GUARD-MEM] Degradar pra legacy, não "logar e seguir"** (revisa R7/guard). Se postings estimados > teto, **abortar BM25 naquele var** e marcar `_bm25_degraded=True` → buscas caem no keyword legacy (`search_multiple`), com log explícito. Sem cap silencioso, sem falsa segurança.
- **[WORDING] "rank de relevância", não "score na fusão"** (corrigido em Visão geral, R4). Honestidade: RRF rank-based → BM25 melhora a ORDEM da perna keyword; magnitude do score só no display.

### P2 — menores
- **[ENV] Onde lê:** env vars BM25 parseadas **module-level em `indexer.py`** (igual aos `_DAMPENING_*`), com parsing robusto (try/except float, bool via `.lower() in (...)`).
- **[CONTRATO] Dois formatos keyword:** `hybrid_search` retorna `keyword_ranked` (BM25, novo) OU `keyword_results` (dict legacy, só em require_all-via-substring/frase). Documentar no docstring qual é fonte de verdade: **BM25 = ranking/contagem/highlight default**; legacy = require_all-literal/frase.
- **[COMPAT] Mudança comportamental de display:** `format_bm25_results` muda o texto compact do keyword. Único consumidor é o Claude lendo a string (não parser) → risco baixo, mas registrar como mudança de comportamento no commit.
- **[TESTES] Expandir suite sintética:** +acento (`câncer`/`cancer`), stopword-only query (→ vazio gracioso), token curto (vitamina D), concorrência (2 builds simultâneos), fronteira de coleção, offset/limit, restart/from_dict (lazy rebuild).

### Reconhecido, sem mudar (mal-entendido/by-design)
- **Lazy trava 1ª busca (P0-2 Codex):** build BM25 é CPU-only e **mais rápido que o embed-build síncrono que já shipamos** e o usuário aceitou; tokens medidos são poucos. Mitigação leve: **prewarm opcional no lifespan** (build BM25 dos vars grandes já carregados, em background) + log de tempo de build. Não bloqueia.
- **Segmentação ruim em PDF (P1-13 Codex):** o cap de ~120 tokens **já limita o segmento** independente de linha-em-branco bizarra (blank line só permite quebra mais cedo). Robusto por design; validar empiricamente num var real no passo de validação.
- **query_terms ambíguo (P1-6 Codex):** contrato — `terms` é a lista do usuário (pode conter strings multi-palavra); `search_bm25` junta+tokeniza num bag (frase-ness perdida pro BM25, preservada só no fallback de frase literal). Esclarecido, sem mudança.
