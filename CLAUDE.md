# RLM MCP Server - Contexto do Projeto

## Arquitetura

Este é um servidor MCP que roda em **Docker na VPS** (não local).

```
Local (Claude Code) ──── SSH/HTTPS ────► VPS (Docker: rlm-mcp-server)
                                              │
                                              ├── /data/ (volume para arquivos)
                                              ├── Minio/S3 (storage externo)
                                              └── REPL Python (variáveis em memória)
```

## Deployment

- **Plataforma**: Dokploy na VPS
- **Domínio**: rlm.drsolution.online (via Traefik)
- **Rede**: dokploy-network
- **CI/CD**: Dokploy sincroniza automaticamente com o repositório GitHub
  - Push para `main` → Deploy automático
  - Não precisa de SSH manual para atualizar

## Como adicionar arquivos para processamento

### Método principal: Minio CLI (mc)

O CLI do Minio (`mc`) já está instalado e configurado.

### Estrutura de pastas no bucket `claude-code`

| Pasta | Tipo de arquivo | Extensões |
|-------|-----------------|-----------|
| `data/` | Dados estruturados | `.csv`, `.json`, `.txt`, `.xml` |
| `pdfs/` | Documentos PDF | `.pdf` |
| `logs/` | Arquivos de log | `.log` |

### Exemplos de upload

```bash
# Dados estruturados → data/
mc cp planilha.csv minio/claude-code/data/
mc cp config.json minio/claude-code/data/

# PDFs → pdfs/
mc cp relatorio.pdf minio/claude-code/pdfs/
mc cp documento-escaneado.pdf minio/claude-code/pdfs/

# Logs → logs/
mc cp app.log minio/claude-code/logs/

# Listar arquivos
mc ls minio/claude-code/ --recursive
```

### Exemplos de carregamento no Claude Code

```
# Dados estruturados
rlm_load_s3(key="data/planilha.csv", name="dados", data_type="csv")
rlm_load_s3(key="data/config.json", name="config", data_type="json")

# PDFs (auto-detecta se precisa OCR)
rlm_load_s3(key="pdfs/relatorio.pdf", name="doc", data_type="pdf")

# PDFs escaneados (força OCR - requer MISTRAL_API_KEY)
rlm_load_s3(key="pdfs/escaneado.pdf", name="doc", data_type="pdf_ocr")

# Logs
rlm_load_s3(key="logs/app.log", name="logs", data_type="lines")
```

**Nota:** O bucket padrão é `claude-code`, não precisa especificar.

### Alternativa: URL assinada (rlm_upload_url)
```
# Gera URL para upload direto (sem precisar de mc)
rlm_upload_url(bucket="docs", key="arquivo.pdf")

# Retorna URL para fazer upload via curl
curl -X PUT -T arquivo.pdf "URL_ASSINADA"
```

### Alternativa: SCP/SFTP para a VPS
```bash
# Do local para a VPS (se bind mount estiver configurado)
scp arquivo.pdf user@vps:/caminho/para/rlm-data/

# No container, estará em /data/arquivo.pdf
```

## Variáveis de Ambiente Necessárias

| Variável | Obrigatória | Uso |
|----------|-------------|-----|
| `RLM_API_KEY` | Recomendado | Autenticação do servidor |
| `OPENAI_API_KEY` | Para llm_query | Sub-chamadas LLM |
| `MISTRAL_API_KEY` | Para OCR | PDFs escaneados |
| `MINIO_*` | Opcional | Storage S3 |
| `RLM_PERSIST_DIR` | Opcional | Diretório SQLite (padrão: /persist) |

## Tools Disponíveis

### Carregamento de dados
- `rlm_load_data` - Dados diretos (string)
- `rlm_load_file` - Arquivo do volume /data (text, json, csv, lines, **pdf**, **pdf_ocr**)
- `rlm_load_s3` - Arquivo do Minio/S3 (text, json, csv, lines, **pdf**, **pdf_ocr**)

### Execução
- `rlm_execute` - Código Python no REPL

### Gerenciamento
- `rlm_list_vars` - Lista variáveis (suporta paginação: `offset`, `limit`)
- `rlm_var_info` - Info de uma variável
- `rlm_clear` - Limpa variáveis
- `rlm_memory` - Uso de memória

### S3/Minio
- `rlm_list_buckets` - Lista buckets
- `rlm_list_s3` - Lista objetos (suporta paginação: `offset`, `limit`)
- `rlm_upload_url` - Upload de URL para bucket (rate limited: 10/min)

### Busca e Persistência
- `rlm_search_index` - Busca termos no índice semântico (suporta paginação: `offset`, `limit`)
- `rlm_persistence_stats` - Estatísticas de variáveis/índices persistidos

### Coleções (Multi-assunto)
- `rlm_collection_create` - Cria coleção para agrupar variáveis
- `rlm_collection_add` - Adiciona variáveis a uma coleção
- `rlm_collection_list` - Lista todas as coleções
- `rlm_collection_info` - Info detalhada de uma coleção
- `rlm_search_collection` - Busca em TODAS as variáveis de uma coleção (suporta paginação: `offset`, `limit`)

### Processamento de PDF (duas etapas)
- `rlm_process_pdf` - Extrai texto de PDF e salva .txt no bucket (não bloqueia)
  ```
  # Etapa 1: Processar PDF grande (salva texto no bucket)
  rlm_process_pdf(key="pdfs/livro.pdf")  # → salva pdfs/livro.txt

  # Etapa 2: Carregar texto rápido para análise
  rlm_load_s3(key="pdfs/livro.txt", name="texto", data_type="text")
  ```

## Paginação

Endpoints que retornam listas grandes suportam paginação via `offset` e `limit`:

```python
# Listar variáveis (primeiras 10)
rlm_list_vars(offset=0, limit=10)

# Listar mais variáveis (próximas 10)
rlm_list_vars(offset=10, limit=10)

# Busca com paginação
rlm_search_index(var_name="texto", terms=["erro"], offset=0, limit=20)

# Listar objetos S3 com paginação
rlm_list_s3(bucket="claude-code", prefix="data/", offset=0, limit=50)

# Busca em coleção com paginação
rlm_search_collection(collection="docs", terms=["termo"], offset=0, limit=10)
```

## MCP Resources

O servidor expõe resources conforme a especificação MCP:

| Resource URI | Descrição |
|--------------|-----------|
| `rlm://variables` | Lista variáveis persistidas com metadados |
| `rlm://memory` | Uso de memória do REPL (total, usada, livre) |
| `rlm://collections` | Lista de coleções com contagem de variáveis |

Resources são acessíveis via `resources/list` e `resources/read` no protocolo MCP.

## Helper Functions no REPL

O REPL Python inclui funções auxiliares pré-definidas para facilitar análise de textos:

### `buscar(texto, termo)` → list[dict]
Busca um termo no texto (case-insensitive).
```python
# Retorna: [{'posicao': int, 'linha': int, 'contexto': str}]
resultados = buscar(meu_texto, "erro")
for r in resultados:
    print(f"Linha {r['linha']}: {r['contexto']}")
```

### `contar(texto, termo)` → dict
Conta ocorrências de um termo (case-insensitive).
```python
# Retorna: {'total': int, 'por_linha': {linha: count}}
stats = contar(meu_texto, "warning")
print(f"Total: {stats['total']}")
```

### `extrair_secao(texto, inicio, fim)` → list[dict]
Extrai seções entre marcadores (case-insensitive).
```python
# Retorna: [{'conteudo': str, 'posicao_inicio': int, 'posicao_fim': int, 'linha_inicio': int, 'linha_fim': int}]
secoes = extrair_secao(doc, "## Introdução", "## Conclusão")
for s in secoes:
    print(s['conteudo'][:200])
```

### `resumir_tamanho(bytes)` → str
Converte bytes para formato humanizado.
```python
# Retorna: string como "1.5 MB", "256 KB"
print(resumir_tamanho(1048576))  # → "1.0 MB"
print(resumir_tamanho(1536))     # → "1.5 KB"
```

## Estrutura do Código

```
src/rlm_mcp/
├── http_server.py   # Servidor HTTP/SSE (único servidor MCP)
├── repl.py          # REPL Python sandboxed
├── pdf_parser.py    # Extração de PDF (pdfplumber + Mistral OCR)
├── s3_client.py     # Cliente Minio/S3
├── llm_client.py    # Cliente para sub-chamadas LLM
├── persistence.py   # Persistência SQLite (variáveis + índices)
└── indexer.py       # Indexação semântica automática
```

## Persistência e Indexação

### Persistência automática (SQLite)
- Variáveis carregadas com `rlm_load_s3` ou `rlm_load_data` são **automaticamente persistidas**
- Ao reiniciar o servidor, variáveis são restauradas automaticamente
- Dados ficam em `/persist/rlm_data.db` (volume Docker)

### Indexação automática
- Textos com **100k+ caracteres** são **indexados automaticamente**
- Índice permite busca rápida por termos sem varrer o texto todo
- Termos padrão: emoções, relações, trabalho, sintomas físicos, partes do corpo
- Use `rlm_search_index(var_name, terms)` para buscar

### Exemplos de uso
```
# Busca simples em uma variável
rlm_search_index(var_name="scholten1", terms=["medo", "trabalho"])

# Busca com todos os termos (AND)
rlm_search_index(var_name="scholten1", terms=["medo", "fracasso"], require_all=True)

# Ver estatísticas de persistência
rlm_persistence_stats()
```

### Usando Coleções (Multi-assunto)
```
# Criar coleção de homeopatia
rlm_collection_create(name="homeopatia", description="Materiais de homeopatia unicista")

# Adicionar documentos à coleção
rlm_collection_add(collection="homeopatia", vars=["scholten1", "scholten2", "kent", "banerji"])

# Buscar em TODOS os documentos da coleção de uma vez
rlm_search_collection(collection="homeopatia", terms=["medo", "ansiedade"])

# Listar coleções disponíveis
rlm_collection_list()
```

Você pode ter múltiplas coleções no mesmo servidor:
- `homeopatia`: scholten, kent, banerji, matéria médica
- `nutrição`: protocolos, suplementos, dietas
- `fitoterapia`: plantas, formulações

## Rate Limiting

O servidor implementa rate limiting para proteger contra sobrecarga:

| Endpoint | Limite | Janela |
|----------|--------|--------|
| SSE/MCP requests | 100 requests | 1 minuto |
| `rlm_upload_url` | 10 uploads | 1 minuto |

Quando o limite é excedido, o servidor retorna HTTP 429 (Too Many Requests).

Configuração via variáveis de ambiente:
- `RLM_SSE_RATE_LIMIT`: requests por minuto (padrão: 100)
- `RLM_SSE_RATE_WINDOW`: janela em segundos (padrão: 60)
- `RLM_UPLOAD_RATE_LIMIT`: uploads por minuto (padrão: 10)
- `RLM_UPLOAD_RATE_WINDOW`: janela em segundos (padrão: 60)

## Observabilidade

### Endpoint `/metrics`

Retorna métricas do servidor em JSON:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "requests": {
    "total": 1500,
    "by_endpoint": {"/message": 1200, "/mcp": 300}
  },
  "errors": {
    "total": 5,
    "by_endpoint": {"/message": 3, "/mcp": 2}
  },
  "latency_ms": {
    "avg": 45.2,
    "p50": 30,
    "p95": 120,
    "p99": 250,
    "max": 500
  },
  "tools": {
    "calls_by_name": {"rlm_execute": 500, "rlm_load_s3": 200}
  },
  "rate_limiting": {
    "rejections": 3
  }
}
```

### Logging estruturado (JSON)

Ative logging JSON via variável de ambiente:
```bash
LOG_FORMAT=json  # ou LOG_FORMAT=text (padrão)
LOG_LEVEL=INFO   # DEBUG, INFO, WARNING, ERROR
```

### Request ID

Cada requisição inclui um `X-Request-Id` header para tracing. O ID também aparece nos logs e nas respostas de erro.

## Notas Importantes

- Volume `/data` é **read-only** por segurança
- Volume `/persist` guarda SQLite com variáveis e índices
- REPL Python roda em **sandbox** (imports limitados)
- PDFs machine readable usam **pdfplumber** (local, rápido)
- PDFs escaneados usam **Mistral OCR API** (requer API key)

## ⚠️ Dokploy + Traefik - Lições Aprendidas

### docker-compose.yml - Regras para Dokploy

1. **NÃO usar `container_name`** - Causa conflito no redeploy
   ```yaml
   # ERRADO - causa "container name already in use"
   container_name: rlm-mcp-server

   # CERTO - deixar Docker gerar nome automático
   # (não colocar container_name)
   ```

2. **Configurar domínios pela UI do Dokploy** (recomendado)
   - Aba "Domains" no painel do Dokploy
   - Dokploy adiciona labels Traefik automaticamente
   - Referência: https://docs.dokploy.com/docs/core/docker-compose/domains

3. **Se usar labels manuais**, manter simples:
   ```yaml
   labels:
     - "traefik.enable=true"
     - "traefik.docker.network=dokploy-network"
     - "traefik.http.services.NOME.loadbalancer.server.port=PORTA"
   ```

4. **NÃO duplicar routers** (HTTP + HTTPS separados causa conflito)
   - Dokploy/Traefik gerencia redirecionamento HTTP→HTTPS automaticamente

### Dockerfile - Cache busting

Para forçar rebuild quando código Python muda:
```dockerfile
# No builder stage
ARG CACHE_BUST=YYYYMMDDHH

# No runtime stage (também!)
ARG CACHE_BUST=YYYYMMDDHH
RUN echo "Version: ${CACHE_BUST}" && pip install ...
```

Incrementar `CACHE_BUST` força rebuild completo.

### Troubleshooting comum

| Erro | Causa | Solução |
|------|-------|---------|
| 502 Bad Gateway | Container não na dokploy-network | Adicionar `traefik.docker.network=dokploy-network` |
| Container name conflict | `container_name` no compose | Remover diretiva |
| Código antigo após deploy | Docker cache | Incrementar CACHE_BUST |
| MCP não aparece em /mcp | Servidor 502 | Verificar health endpoint |
| Multiple services conflict | Labels Traefik duplicadas | Usar UI do Dokploy para domínios |

### Referências

- [Dokploy Domains](https://docs.dokploy.com/docs/core/docker-compose/domains)
- [Issue #3435 - Traefik routing](https://github.com/Dokploy/dokploy/issues/3435)
- [Traefik Docker docs](https://doc.traefik.io/traefik/reference/routing-configuration/other-providers/docker/)
