# RLM MCP Server - Quick Start

## Setup em 5 Minutos

### 1. Deploy no Servidor

```bash
cd rlm-mcp-server

# Deploy (usa sua VPS Digital Ocean)
./scripts/deploy-dokploy.sh root@159.223.140.36
```

### 2. Coloque Seus Dados no Servidor

```bash
# Copie arquivos para análise
scp seu-arquivo-grande.log root@159.223.140.36:/opt/rlm-mcp-server/data/
```

### 3. Conecte do Seu Mac

```bash
# Crie o túnel SSH
./scripts/connect.sh root@159.223.140.36
```

### 4. Configure o Claude Code

Adicione ao seu `~/.claude/claude.json`:

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

> **Nota**: Instale socat se necessário: `brew install socat`

### 5. Use!

Reinicie o Claude Code e use os comandos:

```
Você: "Carregue o arquivo /data/app.log e encontre todos os erros de timeout"

Claude: [carrega arquivo em variável, executa análise, retorna apenas resultado]
```

## Comandos Úteis

| Comando | O que faz |
|---------|-----------|
| `rlm_load_file` | Carrega arquivo grande em variável |
| `rlm_execute` | Executa código Python |
| `rlm_list_vars` | Mostra variáveis carregadas |
| `rlm_memory` | Mostra uso de memória |
| `rlm_clear` | Limpa variáveis |

## Exemplo Completo

```
Você: "Preciso analisar 500MB de logs para encontrar padrões de erro"

Claude: Vou carregar os logs e analisar.

[rlm_load_file: name="logs", path="/data/app.log", data_type="lines"]
→ "Variável 'logs' carregada: 487.3 MB (list)"

[rlm_execute]:
errors = [l for l in logs if 'ERROR' in l]
print(f"Total de erros: {len(errors)}")

from collections import Counter
error_types = Counter(l.split('ERROR')[1].split(':')[0].strip() for l in errors)
for err, count in error_types.most_common(10):
    print(f"  {err}: {count}")
→
Total de erros: 12,456
  DatabaseTimeout: 3,421
  ConnectionRefused: 2,891
  ...

Claude: "Encontrei 12,456 erros. Os mais comuns são DatabaseTimeout (3,421) e..."
```

O contexto do Claude recebeu apenas o **resultado** (poucas linhas), não os 500MB de logs!
