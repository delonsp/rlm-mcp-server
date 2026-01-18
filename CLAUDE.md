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

```bash
# Upload de PDF para o Minio
mc cp arquivo.pdf minio/bucket-name/pasta/

# Listar arquivos
mc ls minio/bucket-name/

# Download
mc cp minio/bucket-name/arquivo.pdf ./
```

Depois do upload, no Claude Code:
```
# Para arquivos de texto
rlm_load_s3(bucket="bucket-name", key="pasta/arquivo.json", name="data", data_type="json")

# Para PDFs (auto-detecta se precisa OCR)
rlm_load_s3(bucket="bucket-name", key="pasta/relatorio.pdf", name="doc", data_type="pdf")

# Para PDFs escaneados (força OCR)
rlm_load_s3(bucket="bucket-name", key="pasta/escaneado.pdf", name="doc", data_type="pdf_ocr")
```

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
├── server.py      # Servidor MCP principal (tools)
├── repl.py        # REPL Python sandboxed
├── pdf_parser.py  # Extração de PDF (pdfplumber + Mistral OCR)
├── s3_client.py   # Cliente Minio/S3
├── llm_client.py  # Cliente para sub-chamadas LLM
├── http_server.py # Servidor HTTP/SSE
└── tcp_bridge.py  # Bridge TCP (alternativo)
```

## Notas Importantes

- Volume `/data` é **read-only** por segurança
- REPL Python roda em **sandbox** (imports limitados)
- PDFs machine readable usam **pdfplumber** (local, rápido)
- PDFs escaneados usam **Mistral OCR API** (requer API key)
