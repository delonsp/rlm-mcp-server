# RLM MCP Server - Refactoring Fases 1-2

**Escopo**: Extrair helpers e schemas (baixo risco)
**Resultado esperado**: Redução de ~640 linhas em http_server.py

## Problema Atual
- `http_server.py`: 2,454 linhas
- `get_tools_list()`: 540 linhas com schemas inline
- Código duplicado: S3 check (5x), response wrapper (20x), persistência (3x)

## Nova Estrutura Proposta

```
src/rlm_mcp/
├── http_server.py          # Slim: ~1800 linhas (após fases 1-2)
├── services/
│   ├── __init__.py         # Pacote vazio
│   ├── s3_guard.py         # require_s3_configured()
│   └── persistence_service.py  # persist_and_index() helper
└── tools/
    ├── __init__.py         # Pacote vazio
    ├── base.py             # text_response(), error_response()
    └── schemas.py          # TOOL_SCHEMAS (extraído de get_tools_list)
```

---

## Tarefas

### Tarefa 1: Criar estrutura de diretórios
- [x] Criar `src/rlm_mcp/services/__init__.py` (arquivo vazio)
- [x] Criar `src/rlm_mcp/tools/__init__.py` (arquivo vazio)

**Validação**: `ls src/rlm_mcp/services/ src/rlm_mcp/tools/`

---

### Tarefa 2: Criar services/s3_guard.py
- [x] Criar arquivo com função `require_s3_configured()`
- [x] Função retorna `(s3_client, None)` se configurado
- [x] Função retorna `(None, error_dict)` se não configurado

```python
from ..s3_client import get_s3_client

def require_s3_configured():
    """Verifica se S3 está configurado. Retorna (client, error_response)."""
    s3 = get_s3_client()
    if not s3.is_configured():
        return None, {
            "content": [{"type": "text", "text": "Erro: Minio não configurado. Configure MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY."}],
            "isError": True
        }
    return s3, None
```

**Validação**: `python -c "from rlm_mcp.services.s3_guard import require_s3_configured; print('OK')"`

---

### Tarefa 3: Criar tools/base.py
- [x] Criar arquivo com funções `text_response()` e `error_response()`

```python
def text_response(text: str) -> dict:
    """Cria resposta MCP com texto."""
    return {"content": [{"type": "text", "text": text}]}

def error_response(message: str) -> dict:
    """Cria resposta MCP de erro."""
    return {"content": [{"type": "text", "text": message}], "isError": True}
```

**Validação**: `python -c "from rlm_mcp.tools.base import text_response, error_response; print('OK')"`

---

### Tarefa 4: Criar services/persistence_service.py
- [x] Criar arquivo com função `persist_and_index()`
- [x] Extrair padrão repetido 3x em http_server.py (linhas 1146-1173, 1368-1391, 1412-1435)

```python
import logging
from ..persistence import get_persistence
from ..indexer import auto_index_if_large, set_index

logger = logging.getLogger(__name__)

def persist_and_index(var_name: str, value, repl) -> tuple[str, str, str]:
    """Persiste variável e indexa se grande.

    Returns:
        tuple: (persist_msg, index_msg, error_msg)
    """
    persist_msg = ""
    index_msg = ""
    error_msg = ""

    try:
        persistence = get_persistence()
        if value is not None:
            persistence.save_variable(var_name, value)
            persist_msg = "💾 Persistido"

            if isinstance(value, str) and len(value) >= 100000:
                idx = auto_index_if_large(value, var_name)
                if idx:
                    set_index(var_name, idx)
                    persistence.save_index(var_name, idx.to_dict())
                    index_msg = f"📑 Indexado ({idx.get_stats()['indexed_terms']} termos)"
    except Exception as e:
        logger.warning(f"Erro ao persistir/indexar {var_name}: {e}")
        error_msg = f"\n⚠️ Erro de persistência: {e}"

    return persist_msg, index_msg, error_msg
```

**Validação**: `python -c "from rlm_mcp.services.persistence_service import persist_and_index; print('OK')"`

---

### Tarefa 5: Criar tools/schemas.py
- [x] Extrair todas as definições de tools de `get_tools_list()` (linhas 578-1115 de http_server.py)
- [x] Criar constante `TOOL_SCHEMAS` com lista de dicts
- [x] Manter exatamente o mesmo conteúdo dos schemas existentes

**Validação**: `python -c "from rlm_mcp.tools.schemas import TOOL_SCHEMAS; print(f'{len(TOOL_SCHEMAS)} schemas')"`

---

### Tarefa 6: Atualizar http_server.py - imports e get_tools_list
- [x] Adicionar import: `from .tools.schemas import TOOL_SCHEMAS`
- [x] Simplificar `get_tools_list()` para retornar `TOOL_SCHEMAS`

```python
from .tools.schemas import TOOL_SCHEMAS

def get_tools_list() -> list[dict]:
    """Retorna lista de tools disponíveis"""
    return TOOL_SCHEMAS
```

**Validação**: `pytest tests/test_http_server.py::TestMcpToolsList -v`

---

### Tarefa 7: Atualizar http_server.py - usar s3_guard
- [ ] Adicionar import: `from .services.s3_guard import require_s3_configured`
- [ ] Substituir 5 ocorrências de S3 check em `call_tool()`:
  - rlm_load_s3: substituir padrão `s3 = get_s3_client()` + `if not s3.is_configured()`
  - rlm_list_buckets: mesmo padrão
  - rlm_list_s3: mesmo padrão
  - rlm_upload_url: mesmo padrão
  - rlm_process_pdf: mesmo padrão

Padrão a substituir:
```python
# ANTES
s3 = get_s3_client()
if not s3.is_configured():
    return {
        "content": [{"type": "text", "text": "Erro: Minio não configurado..."}],
        "isError": True
    }

# DEPOIS
s3, error = require_s3_configured()
if error:
    return error
```

**Validação**: `pytest tests/test_http_server.py -v -k "s3 or minio or bucket"`

---

### Tarefa 8: Atualizar http_server.py - usar persistence_service
- [ ] Adicionar import: `from .services.persistence_service import persist_and_index`
- [ ] Substituir 3 blocos de persistência em `call_tool()`:
  - Bloco após rlm_load_data/rlm_load_file (linhas ~1146-1173)
  - Bloco após carregar PDF do S3 (linhas ~1368-1391)
  - Bloco após carregar texto do S3 (linhas ~1412-1435)

Padrão a substituir:
```python
# ANTES (20+ linhas)
persist_msg = ""
index_msg = ""
persist_error = ""
try:
    persistence = get_persistence()
    value = repl.variables.get(var_name)
    if value is not None:
        persistence.save_variable(var_name, value)
        persist_msg = "💾 Persistido"
        if isinstance(value, str) and len(value) >= 100000:
            idx = auto_index_if_large(value, var_name)
            ...

# DEPOIS (3 linhas)
value = repl.variables.get(var_name)
persist_msg, index_msg, persist_error = persist_and_index(var_name, value, repl)
```

**Validação**: `pytest tests/test_http_server.py -v -k "persist or load"`

---

### Tarefa 9: Teste final completo
- [ ] Rodar suite completa de testes
- [ ] Verificar que todos os 1514+ testes passam

**Validação**: `pytest tests/ -v --tb=short`

---

## Critério de Sucesso
- Todos os testes passando
- `http_server.py` reduzido de 2454 para ~1800 linhas
- Novos módulos criados e funcionais

## Arquivos Críticos - NÃO MODIFICAR
- `docker-compose.yml`
- `Dockerfile`

## Notas para Ralph

### Comando de teste
```bash
cd /Users/alain_dutra/massive-context-window/rlm-mcp-server && pytest tests/ -v --tb=short
```

### Padrões do projeto
- Usar logging (não print)
- Imports no topo do arquivo
- Type hints em funções públicas
- Docstrings em funções públicas
- Testes em tests/ com pytest

### Ordem de execução
1. Tarefas 1-4: Criar novos módulos (sem modificar http_server.py)
2. Tarefa 5: Extrair schemas
3. Tarefas 6-8: Atualizar http_server.py para usar novos módulos
4. Tarefa 9: Validação final
