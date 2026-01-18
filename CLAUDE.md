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

## Tools Disponíveis

### Carregamento de dados
- `rlm_load_data` - Dados diretos (string)
- `rlm_load_file` - Arquivo do volume /data (text, json, csv, lines, **pdf**, **pdf_ocr**)
- `rlm_load_s3` - Arquivo do Minio/S3 (text, json, csv, lines, **pdf**, **pdf_ocr**)

### Execução
- `rlm_execute` - Código Python no REPL

### Gerenciamento
- `rlm_list_vars` - Lista variáveis
- `rlm_var_info` - Info de uma variável
- `rlm_clear` - Limpa variáveis
- `rlm_memory` - Uso de memória

### S3/Minio
- `rlm_list_buckets` - Lista buckets
- `rlm_list_s3` - Lista objetos
- `rlm_upload_url` - URL assinada para upload

## Estrutura do Código

```
src/rlm_mcp/
├── server.py      # Servidor MCP stdio (NÃO usado pelo Dokploy)
├── http_server.py # Servidor HTTP/SSE (USADO pelo Dokploy!)
├── repl.py        # REPL Python sandboxed
├── pdf_parser.py  # Extração de PDF (pdfplumber + Mistral OCR)
├── s3_client.py   # Cliente Minio/S3
├── llm_client.py  # Cliente para sub-chamadas LLM
└── tcp_bridge.py  # Bridge TCP (alternativo)
```

**⚠️ ATENÇÃO**: Dokploy usa `http_server.py`, não `server.py`!

## Notas Importantes

- Volume `/data` é **read-only** por segurança
- REPL Python roda em **sandbox** (imports limitados)
- PDFs machine readable usam **pdfplumber** (local, rápido)
- PDFs escaneados usam **Mistral OCR API** (requer API key)

## ⚠️ Dokploy + Traefik - Lições Aprendidas

### Dois arquivos de servidor - CUIDADO!

O projeto tem **dois modos de transporte MCP**:

| Arquivo | Transporte | Quando usar |
|---------|------------|-------------|
| `server.py` | stdio | MCP local via `command` no claude.json |
| `http_server.py` | HTTP/SSE | **Dokploy usa este!** Via URL HTTPS |

**IMPORTANTE**: Ao modificar tools, **atualizar AMBOS os arquivos** ou remover `server.py` se não for usado.

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
