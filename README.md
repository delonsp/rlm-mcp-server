# RLM MCP Server

Servidor MCP (Model Context Protocol) que implementa **Recursive Language Models** para análise de dados massivos sem poluir o contexto do Claude.

Baseado no paper ["Recursive Language Models"](https://arxiv.org/abs/2512.24601) do MIT CSAIL.

## Por Que Usar?

O Claude Code tradicional:
- Carrega todo output no contexto
- Precisa de `/compact` frequente
- Limita análise de arquivos grandes

Com RLM MCP:
- Dados ficam em variáveis **fora** do contexto
- Contexto permanece pequeno
- Analise arquivos de 100MB+ sem compact

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│  Sua máquina local                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Claude Code                                        │    │
│  │  Tools MCP: rlm_execute, rlm_load_data, etc.       │    │
│  └──────────────────────┬──────────────────────────────┘    │
└─────────────────────────┼───────────────────────────────────┘
                          │ SSH Tunnel (porta 8765)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Seu Servidor (Digital Ocean, AWS, etc.)                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Docker: rlm-mcp-server                             │    │
│  │  ┌───────────────────────────────────────────────┐  │    │
│  │  │  Python REPL Persistente                      │  │    │
│  │  │  variables = {"logs": <500MB>, ...}           │  │    │
│  │  │  Dados em memória, NÃO no contexto            │  │    │
│  │  └───────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Instalação

### 1. No Servidor (via Dokploy ou Docker)

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/rlm-mcp-server.git
cd rlm-mcp-server

# Copie o .env
cp .env.example .env
# Edite .env com suas configurações

# Crie o diretório de dados
mkdir -p data
# Coloque seus arquivos em ./data/

# Build e start
docker-compose up -d --build
```

#### Via Dokploy

1. No Dokploy, crie um novo Application
2. Source: Git → URL do repositório
3. Build: Docker Compose
4. Deploy

### 2. Na Máquina Local (Claude Code)

#### Opção A: SSH Tunnel (Recomendado)

```bash
# Crie um túnel SSH para a porta do servidor
ssh -L 8765:localhost:8765 root@seu-servidor.com -N &

# O túnel fica rodando em background
```

#### Opção B: Uso Local (sem servidor remoto)

Se quiser rodar localmente para testes:

```bash
cd rlm-mcp-server
pip install -e .

# Teste manualmente
python -c "from rlm_mcp.server import main; main()"
```

### 3. Configure o Claude Code

Adicione ao `~/.claude/claude.json`:

```json
{
  "mcpServers": {
    "rlm": {
      "type": "sse",
      "url": "https://rlm.seudominio.com/sse",
      "headers": {
        "Authorization": "Bearer SUA_API_KEY_AQUI"
      }
    }
  }
}
```

> **Nota**: Substitua `rlm.seudominio.com` pelo domínio configurado no Traefik/Dokploy.

**Alternativa** - via SSH tunnel (se não quiser expor publicamente):

```json
{
  "mcpServers": {
    "rlm": {
      "command": "socat",
      "args": ["TCP:localhost:8765", "STDIO"]
    }
  }
}
```

### 4. Reinicie o Claude Code

```bash
# Feche e abra novamente, ou:
claude --mcp-restart
```

## Uso

### Tools Disponíveis

| Tool | Descrição |
|------|-----------|
| `rlm_load_data` | Carrega dados diretamente em variável |
| `rlm_load_file` | Carrega arquivo do servidor (text, json, csv, lines, **pdf**, **pdf_ocr**) |
| `rlm_execute` | Executa código Python |
| `rlm_list_vars` | Lista variáveis disponíveis |
| `rlm_var_info` | Info detalhada de uma variável |
| `rlm_clear` | Limpa variáveis |
| `rlm_memory` | Estatísticas de memória |
| `rlm_load_s3` | Carrega arquivo do Minio/S3 (text, json, csv, lines, **pdf**, **pdf_ocr**) |
| `rlm_list_buckets` | Lista buckets do Minio |
| `rlm_list_s3` | Lista objetos em um bucket |
| `rlm_upload_url` | Gera URL assinada para upload

### Funções Disponíveis Dentro do Código (RLM)

Dentro do código executado via `rlm_execute`, estas funções estão disponíveis:

| Função | Descrição |
|--------|-----------|
| `llm_query(prompt, data=None, model=None)` | Faz sub-chamada a um LLM |
| `llm_stats()` | Retorna estatísticas de uso |
| `llm_reset_counter()` | Reseta contador de chamadas |

### Exemplos de Uso no Claude Code

**Analisar logs massivos:**

```
Você: "Analise o log /data/app.log e encontre todos os erros"

Claude: [usa rlm_load_file para carregar em variável 'logs']
Claude: [usa rlm_execute com código Python para filtrar erros]
Claude: "Encontrei 1,234 erros. Os mais comuns são..."
```

**Busca exata em código:**

```
Você: "Quantas vezes 'TODO' aparece perto de 'FIXME' no código?"

Claude: [usa rlm_load_file para carregar código]
Claude: [usa rlm_execute]:
    import re
    matches = re.findall(r'TODO.{0,50}FIXME|FIXME.{0,50}TODO', data)
    print(f"Encontrados: {len(matches)}")
```

**Agregação de dados:**

```
Você: "Agrupe os logs por hora e conte requests"

Claude: [usa rlm_execute]:
    from collections import Counter
    hours = [line.split()[0][:13] for line in logs if 'request' in line]
    counts = Counter(hours)
    for hour, count in counts.most_common(10):
        print(f"{hour}: {count}")
```

**Sub-chamadas LLM (Recursive Language Model):**

```
Você: "Analise 1GB de logs e encontre padrões de erro"

Claude: [usa rlm_execute com llm_query]:
    # Divide dados massivos em chunks
    chunk_size = 50000
    chunks = [logs[i:i+chunk_size] for i in range(0, len(logs), chunk_size)]

    # Processa cada chunk com sub-LLM (padrão map-reduce)
    summaries = []
    for i, chunk in enumerate(chunks):
        summary = llm_query(
            "Liste os erros críticos encontrados neste log:",
            data="\n".join(chunk)
        )
        summaries.append(summary)
        print(f"Chunk {i+1}/{len(chunks)} processado")

    # Sintetiza resultados
    final = llm_query(
        "Combine estes resumos em um relatório final:",
        data="\n---\n".join(summaries)
    )
    print(final)
```

Este padrão implementa o paper ["Recursive Language Models"](https://arxiv.org/abs/2512.24601) do MIT CSAIL, permitindo processar dados que excedem a janela de contexto do LLM.

### Processamento de PDFs

Use `data_type="pdf"` ou `data_type="pdf_ocr"` em `rlm_load_file` ou `rlm_load_s3`:

| data_type | Uso | Requer |
|-----------|-----|--------|
| `pdf` | Auto-detecta: tenta pdfplumber, fallback para OCR | `MISTRAL_API_KEY` para fallback |
| `pdf_ocr` | Força OCR (para escaneados/imagens) | `MISTRAL_API_KEY` |

**Exemplo com Minio (recomendado):**

```bash
# Upload via mc CLI
mc cp relatorio.pdf minio/docs/
```

```
# No Claude Code
rlm_load_s3(bucket="docs", key="relatorio.pdf", name="doc", data_type="pdf")
```

**Forçando OCR para documentos escaneados:**

```
rlm_load_s3(bucket="docs", key="escaneado.pdf", name="doc", data_type="pdf_ocr")
```

## Segurança

### Sandbox Python

O REPL executa em sandbox com:

- **Imports permitidos**: `re`, `json`, `math`, `collections`, `datetime`, `csv`, etc.
- **Imports bloqueados**: `os`, `subprocess`, `socket`, `requests`, etc.
- **Funções bloqueadas**: `exec`, `eval`, `open`, `__import__`, etc.

### Acesso a Arquivos

- Somente arquivos em `/data/` são acessíveis
- Volume montado como **read-only**
- Path traversal é bloqueado

### Rede

- Container em rede isolada (sem acesso à internet)
- Conexão apenas via localhost (SSH tunnel)

## Configuração

### Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `RLM_MAX_MEMORY_MB` | 1024 | Limite de memória para variáveis |
| `RLM_API_KEY` | (vazio) | API key para autenticação (opcional) |
| `OPENAI_API_KEY` | (obrigatório para llm_query) | API key do OpenAI |
| `RLM_SUB_MODEL` | gpt-4o-mini | Modelo para sub-chamadas LLM |
| `RLM_MAX_SUB_CALLS` | 100 | Limite de sub-chamadas por execução |
| `MISTRAL_API_KEY` | (opcional) | API key do Mistral para OCR de PDFs |
| `MINIO_ENDPOINT` | (opcional) | Endpoint do Minio/S3 |
| `MINIO_ACCESS_KEY` | (opcional) | Access key do Minio |
| `MINIO_SECRET_KEY` | (opcional) | Secret key do Minio |
| `MINIO_SECURE` | true | Usar HTTPS para Minio |

### Limites de Recursos (Docker)

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
```

## Troubleshooting

### Claude Code não encontra o MCP

1. Verifique se o túnel SSH está ativo: `ps aux | grep ssh`
2. Teste conexão: `nc -zv localhost 8765`
3. Verifique logs: `docker logs rlm-mcp-server`

### Erro "SecurityError: Import bloqueado"

O sandbox bloqueia imports perigosos. Use apenas imports permitidos.

### Memória insuficiente

1. Aumente `RLM_MAX_MEMORY_MB`
2. Use `rlm_clear` para limpar variáveis não usadas
3. Processe dados em chunks menores

## Desenvolvimento

```bash
# Clone
git clone https://github.com/seu-usuario/rlm-mcp-server.git
cd rlm-mcp-server

# Ambiente virtual
python -m venv venv
source venv/bin/activate

# Instale em modo desenvolvimento
pip install -e ".[dev]"

# Testes
pytest

# Rode localmente
rlm-mcp
```

## Licença

MIT

## Referências

- [Recursive Language Models (MIT CSAIL)](https://arxiv.org/abs/2512.24601)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Claude Code](https://claude.ai/claude-code)
