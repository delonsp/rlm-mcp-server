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
      "command": "socat",
      "args": ["TCP:localhost:8765", "STDIO"]
    }
  }
}
```

**Ou**, para uso local direto:

```json
{
  "mcpServers": {
    "rlm": {
      "command": "python",
      "args": ["-m", "rlm_mcp.server"],
      "cwd": "/caminho/para/rlm-mcp-server"
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
| `rlm_load_file` | Carrega arquivo do servidor |
| `rlm_execute` | Executa código Python |
| `rlm_list_vars` | Lista variáveis disponíveis |
| `rlm_var_info` | Info detalhada de uma variável |
| `rlm_clear` | Limpa variáveis |
| `rlm_memory` | Estatísticas de memória |

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
