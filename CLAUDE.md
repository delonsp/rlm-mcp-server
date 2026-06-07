# RLM MCP Server - Contexto do Projeto

## Arquitetura

Servidor MCP que roda em **Docker na VPS** (não local).

```
Local (Claude Code) ──── HTTPS ────► VPS (Docker: rlm-mcp-server)
                                          │
                                          ├── /data/ (volume read-only)
                                          ├── /persist/ (SQLite: variáveis + índices)
                                          ├── Minio/S3 (storage externo)
                                          └── REPL Python (variáveis em memória)
```

**Startup automático:** variáveis são restauradas do SQLite, metadata recriado, e índices de coleções reconstruídos automaticamente. Não requer intervenção manual após restart/redeploy.

## Deployment

- **Plataforma**: Dokploy na VPS (CI/CD: push `main` → deploy automático)
- **Domínio**: rlm.drsolution.online (via Traefik)
- **Rede**: dokploy-network
- Ver seção "Dokploy + Traefik" no final para troubleshooting

## Estrutura do Código

```
src/rlm_mcp/
├── http_server.py         # Servidor HTTP/SSE FastAPI (MCP protocol, lifespan auto-restore)
├── repl.py                # REPL: dispatch p/ sandbox subprocess ou in-process (legado)
├── sandbox_worker.py      # Isolamento do rlm_execute por subprocesso (forkserver)
├── response_formatter.py  # Formatação de respostas (compact/normal/verbose)
├── indexer.py             # Indexação keyword + live scan fallback + hybrid/RRF search
├── vector_index.py        # Índice vetorial para busca semântica (embeddings)
├── embeddings.py          # Cliente de embeddings (OpenAI API)
├── code_parser.py         # Parse estrutural com tree-sitter (busca em código)
├── persistence.py         # Persistência SQLite (variáveis + índices + coleções)
├── task_manager.py        # Tasks assíncronas (batch ops, PDFs grandes)
├── rate_limiter.py        # Sliding window rate limiting
├── pdf_parser.py          # Extração PDF (pdfplumber + Mistral OCR)
├── s3_client.py           # Cliente Minio/S3
├── llm_client.py          # Cliente para sub-chamadas LLM
├── services/
│   ├── persistence_service.py  # Auto-persist + index helper
│   └── s3_guard.py             # Validação config S3
└── tools/
    ├── base.py            # text_response/error_response helpers
    └── schemas.py         # Definições de tools MCP (20 tools)
```

## Tools Disponíveis (20 tools)

| Categoria | Tool | Descrição |
|-----------|------|-----------|
| **Dados** | `rlm_load_data` | Dados diretos (string) |
| | `rlm_load_file` | Arquivo do volume /data |
| | `rlm_load_s3` | Arquivo(s) do Minio/S3 (unitário ou batch) |
| **Execução** | `rlm_execute` | Código Python no REPL |
| **Gerenciamento** | `rlm_list_vars` | Lista variáveis (paginação: offset/limit) |
| | `rlm_var_info` | Info de uma variável |
| | `rlm_clear` | Limpa variáveis |
| | `rlm_memory` | Uso de memória + stats persistência |
| | `rlm_pin_var` | Protege variável do GC |
| **S3** | `rlm_list_buckets` | Lista buckets |
| | `rlm_list_s3` | Lista objetos (paginação) |
| | `rlm_upload_url` | Upload de URL para bucket |
| | `rlm_save_to_s3` | Salva variável(eis) no S3 |
| **Busca** | `rlm_search_index` | Keyword/semantic/hybrid (max_results, paginação) |
| | `rlm_search_code` | Busca estrutural em código (limit/offset/max_source_lines) |
| **Coleções** | `rlm_collection` | create, add, list, info, rebuild, search, delete |
| **Repertório** | `rlm_repertorio` | buscar_rubrica, repertorizar, info (homeopatia, fonte kent_repertorio) |
| **PDF/Tasks** | `rlm_process_pdf` | Extrai texto de PDF |
| | `rlm_task` | Gerencia tasks assíncronas |
| **Ajuda** | `rlm_help` | Documentação integrada |

## Busca — 3 Modos

### Keyword (padrão)
- 64 termos pré-indexados automaticamente (emoções, relações, corpo, etc.)
- **Qualquer outro termo** funciona via live scan + cache (transparente)
- `rlm_search_index(var_name="livro", terms=["karma", "perdão"])`

### Semantic (vetorial)
- Requer `OPENAI_API_KEY` no servidor
- `rlm_search_index(var_name="livro", terms=["por que o Criador se experimenta"], mode="semantic")`

### Hybrid (keyword + semantic via RRF)
- Combina as duas pernas com Reciprocal Rank Fusion
- **Dica:** usar termos separados no array para hybrid (`["karma", "perdão"]` em vez de `["karma perdão"]`)
- `rlm_search_index(var_name="livro", terms=["karma", "perdão"], mode="hybrid")`

### Parâmetros de controle de output
| Parâmetro | Tool | Default | Função |
|-----------|------|---------|--------|
| `max_results` | search_index | 30 | Cap global entre todos os termos |
| `limit` | search_code | 20 | Max símbolos retornados |
| `offset` | search_code | 0 | Paginação |
| `max_source_lines` | search_code | 5 | Max linhas de código por símbolo |

## Persistência e Indexação

### Auto-restore no startup (lifespan)
1. Restaura variáveis do SQLite → `repl.variables` + `variable_metadata`
2. Restaura índices keyword e embeddings vetoriais
3. Reconstrói índices combinados de todas as coleções (`_coll_*_combined`)

### Indexação automática
- Textos >= 100K chars: índice criado automaticamente com 64 termos padrão
- Textos menores: índice criado on-the-fly na primeira busca
- Termos fora do vocabulário pré-indexado: live scan + cache no `TextIndex.terms`

### Coleções
```python
rlm_collection(action="create", name="docs", description="Documentação")
rlm_collection(action="add", name="docs", vars=["doc1", "doc2"])
rlm_collection(action="search", name="docs", terms=["instalação"])
```
- Índice combinado reconstruído automaticamente no startup
- Busca usa índice combinado + full-text fallback para termos não indexados
- **Busca de coleção é LEXICAL (token), sem perna semântica.** Passar termos como array
  de palavras, não frase: `terms=["calor","ar","livre"]`, não `["pior no calor ar livre"]`
  (frase casa substring literal numa linha → quase zero). Quando uma frase não casa, o
  handler tokeniza e re-busca (AND→OR, word-boundary via `tokenized_collection_scan`) e
  **avisa que é fallback** (não a busca exata — preserva rastreabilidade de citação).
- Frase literal forçada: aspas no termo → `terms=['"open air"']`. `snippet_len` (default
  150, clamp 40–400) controla o trecho exibido.
- Recall por significado/sinônimo/cross-idioma **não existe na coleção**; usar
  `rlm_search_index(var=..., mode="hybrid")` por fonte. Decisão de NÃO portar semântica
  p/ coleção é empírica (benchmark Carcinosinum 2026-06): embedding multilíngue fraco no
  corpus homeopático erra o alvo cross-idioma; e as rubricas "perdidas" eram recuperáveis
  só com tokenização. Ver memória `project_collection_phrase_trap`.

## Variáveis de Ambiente

| Variável | Uso | Padrão |
|----------|-----|--------|
| `RLM_API_KEY` | Autenticação Bearer token | — |
| `OPENAI_API_KEY` | Embeddings + sub-chamadas LLM (fallback) | — |
| `DEEPSEEK_API_KEY` | Sub-chamadas LLM `llm_query` (precede OPENAI) | — |
| `RLM_LLM_BASE_URL` | Endpoint LLM OpenAI-compat | — |
| `RLM_SUB_MODEL` | Modelo das sub-chamadas | `gpt-4o-mini` |
| `MISTRAL_API_KEY` | OCR para PDFs escaneados | — |
| `MINIO_*` | Storage S3 | — |
| `RLM_PERSIST_DIR` | Diretório SQLite | `/persist` |
| `RLM_MAX_MEMORY_MB` | Memória total do REPL | `1024` |
| `RLM_MAX_VAR_SIZE_MB` | Limite por variável | `50` |
| `RLM_RESPONSE_VERBOSITY` | compact/normal/verbose | `compact` |
| `RLM_BOILERPLATE_PENALTY` | Multiplicador de score `[0,1]` p/ chunks de bibliografia/cabeçalho na busca semântica; `1.0` = desligado | `1.0` |
| `RLM_SANDBOX_MODE` | `subprocess` (seguro) / `inprocess` (INSEGURO) | `subprocess` |
| `RLM_EXECUTE_TIMEOUT` | Timeout do execute em s (wall-clock + killpg) | `60` |
| `RLM_SANDBOX_MEM_MB` | RLIMIT_AS do processo-filho | `2048` |
| `RLM_SANDBOX_LOCKDOWN` | Lockdown B2 do filho: `required`/`warn`/`off` (Linux-only) | `warn` |
| `RLM_SANDBOX_FS_LOCKDOWN` | Liga/desliga o Landlock (FS) do lockdown B2 | `true` |
| `RLM_SANDBOX_NET_LOCKDOWN` | Liga/desliga o seccomp (rede) do lockdown B2 | `true` |

## Upload de Arquivos

```bash
# Via Minio CLI (principal)
mc cp arquivo.csv minio/claude-code/data/
mc cp relatorio.pdf minio/claude-code/pdfs/

# Carregar no REPL
rlm_load_s3(key="data/arquivo.csv", name="dados", data_type="csv")
rlm_load_s3(key="pdfs/relatorio.pdf", name="doc", data_type="pdf")
```

Tipos: text, json, lines, csv, code, pdf, pdf_ocr. Bucket padrão: `claude-code`.

## Helper Functions no REPL

| Função | Retorno | Uso |
|--------|---------|-----|
| `buscar(texto, termo)` | `[{posicao, linha, contexto}]` | Busca case-insensitive |
| `contar(texto, termo)` | `{total, por_linha}` | Conta ocorrências |
| `extrair_secao(texto, inicio, fim)` | `[{conteudo, linha_inicio, linha_fim}]` | Extrai entre marcadores |
| `resumir_tamanho(bytes)` | `"1.5 MB"` | Humaniza bytes |

## Segurança

- **Rate limiting**: SSE 100/60s, uploads 10/60s (configurável via `RLM_SSE_RATE_LIMIT`, `RLM_UPLOAD_RATE_LIMIT`)
- **Memory protection**: auto-cleanup em 80% do limite, variáveis individuais max 50MB
- **Sandbox (rlm_execute)**: isolamento por **subprocesso** `forkserver` (`sandbox_worker.py`) — código do user roda em filho efêmero com env scrubado (sem chaves), FDs fechados, `setrlimit`+`killpg` (timeout funciona fora da main thread). Trust assimétrico: o retorno filho→pai nunca executa código (JSON + unpickler com allowlist de tipos → fecha RCE-reverso por pickle). Deny-list AST (`repl.validate_code`) é só a 1ª camada. **Só dados** persistem entre execuções (funções/objetos/mutação in-place, não). Break-glass inseguro: `RLM_SANDBOX_MODE=inprocess`.
- **Lockdown B2 (FS/rede)** (`sandbox_lockdown.py`): quando ativo (Linux + `RLM_SANDBOX_LOCKDOWN`≠`off`), o filho aplica **Landlock** (allowlist de FS: `/usr…` RO + `/dev/shm` RW; nega `/persist`/`/data`/resto) + **seccomp** (corte de rede: `socket`/`connect`/… → `EPERM`) logo antes do `exec`, via `ctypes` *unprivileged* (sem mudar a postura do container). Default `warn` (degrada p/ B1 + WARNING se Landlock/seccomp indisponível); `required` = fail-closed; `off` = break-glass. Resíduo pós-B2: dados que o pai já enviou seguem em memória do filho (por design); sem cgroup por-filho (CPU/mem via RLIMIT/deadline); em `warn`/`off` ou kernel sem Landlock, reverte ao resíduo B1 (escape lê `/persist`+`/data`).
- **CORS**: restrito via `RLM_CORS_ORIGINS`

## Observabilidade

- `/health` — status, memória, versão
- `/metrics` — request counts, latência (p50/p95/p99), tool calls
- `X-Request-Id` header em todas as respostas
- `RLM_LOG_FORMAT=json|text`, `RLM_LOG_LEVEL=INFO|DEBUG|WARNING`

## Dokploy + Traefik — Regras

1. **NÃO usar `container_name`** — causa conflito no redeploy
2. **Configurar domínios pela UI do Dokploy** (aba Domains)
3. **NÃO duplicar routers** HTTP + HTTPS — Dokploy gerencia redirecionamento
4. **Cache busting**: incrementar `CACHE_BUST` ARG no Dockerfile para forçar rebuild

| Erro | Solução |
|------|---------|
| 502 Bad Gateway | Verificar `traefik.docker.network=dokploy-network` |
| Container name conflict | Remover `container_name` do compose |
| Código antigo após deploy | Incrementar `CACHE_BUST` |
