# PRD: RLM MCP Server Test Suite

## Objetivo

Adicionar cobertura de testes ao RLM MCP Server para garantir confiabilidade e permitir refatorações seguras.

## Arquitetura do Projeto

```
src/rlm_mcp/
├── http_server.py   # Servidor HTTP/SSE MCP (FastAPI)
├── repl.py          # REPL Python sandboxed
├── persistence.py   # Persistência SQLite (variáveis + índices)
├── indexer.py       # Indexação semântica automática
├── s3_client.py     # Cliente MinIO/S3
├── pdf_parser.py    # Extração de PDF (pdfplumber + Mistral OCR)
└── llm_client.py    # Cliente para sub-chamadas LLM
```

## Critério de Sucesso

- Cada tarefa marcada [x] significa que `pytest` passa sem erros
- Cobertura mínima: funções críticas testadas
- Mocks para dependências externas (MinIO, Mistral API)

---

## Fase 1: Setup de Testes

- [x] Adicionar pytest e pytest-asyncio ao pyproject.toml (dev dependencies)
- [x] Criar diretório tests/ com __init__.py
- [x] Criar tests/conftest.py com fixtures: temp_db (SQLite em memória), sample_text (texto de 200k chars para indexação)

## Fase 2: Testes do Módulo persistence.py

- [x] Testar save_variable e load_variable (roundtrip de string, dict, list)
- [x] Testar delete_variable remove do banco
- [x] Testar list_variables retorna metadados corretos
- [x] Testar save_index e load_index (roundtrip de índice semântico)
- [x] Testar clear_all remove todas as variáveis
- [x] Testar get_stats retorna contagens corretas
- [x] Testar create_collection e list_collections
- [x] Testar add_to_collection e get_collection_vars
- [x] Testar delete_collection remove associações mas não variáveis

## Fase 3: Testes do Módulo indexer.py

- [x] Testar create_index gera índice com termos padrão
- [x] Testar create_index com additional_terms indexa termos customizados
- [x] Testar TextIndex.search retorna matches corretos
- [x] Testar TextIndex.search_multiple com require_all=False (OR)
- [x] Testar TextIndex.search_multiple com require_all=True (AND)
- [x] Testar auto_index_if_large indexa apenas textos >= 100k chars
- [x] Testar _detect_structure detecta headers markdown
- [x] Testar TextIndex.to_dict e from_dict (serialização)

## Fase 4: Testes do Módulo repl.py

- [x] Testar execute com código simples (print, atribuição)
- [x] Testar execute preserva variáveis entre execuções
- [x] Testar execute bloqueia imports perigosos (os, subprocess, socket)
- [x] Testar execute permite imports seguros (re, json, math, collections)
- [x] Testar load_data com data_type="text"
- [x] Testar load_data com data_type="json"
- [x] Testar load_data com data_type="csv"
- [x] Testar load_data com data_type="lines"
- [x] Testar get_memory_usage retorna valores razoáveis
- [x] Testar clear_namespace limpa variáveis

## Fase 5: Testes do Módulo s3_client.py (com mocks)

- [x] Criar mock do MinIO client em conftest.py
- [x] Testar is_configured retorna False sem credenciais
- [x] Testar list_buckets com mock retorna lista
- [x] Testar list_objects com mock retorna objetos
- [x] Testar get_object com mock retorna bytes
- [x] Testar get_object_info com mock retorna metadados
- [x] Testar object_exists com mock retorna True/False

## Fase 6: Testes do Módulo pdf_parser.py (com mocks)

- [x] Testar extract_with_pdfplumber com PDF machine readable (criar fixture)
- [x] Testar extract_with_pdfplumber retorna erro se arquivo não existe
- [x] Testar extract_pdf com method="auto" usa pdfplumber primeiro
- [x] Testar extract_pdf faz fallback para OCR se pdfplumber extrai pouco
- [x] Testar split_pdf_into_chunks divide corretamente
- [x] Mockar Mistral API para testar extract_with_mistral_ocr

## Fase 7: Testes de Integração HTTP (FastAPI TestClient)

- [x] Testar endpoint /health retorna 200
- [x] Testar MCP initialize retorna capabilities
- [x] Testar MCP tools/list retorna todas as tools
- [x] Testar tool rlm_execute com código simples
- [x] Testar tool rlm_load_data carrega variável
- [x] Testar tool rlm_list_vars lista variáveis carregadas
- [x] Testar tool rlm_var_info retorna info da variável
- [x] Testar tool rlm_clear limpa namespace
- [ ] Testar tool rlm_load_s3 com skip_if_exists=True pula se existe
- [ ] Testar tool rlm_load_s3 com skip_if_exists=False força reload
- [ ] Testar tool rlm_search_index busca termos
- [ ] Testar tool rlm_persistence_stats retorna estatísticas

## Fase 8: Testes de Edge Cases e Segurança

- [ ] Testar persistence.py com caracteres especiais em nomes de variáveis
- [ ] Testar indexer.py com texto vazio
- [ ] Testar indexer.py com texto None (deve tratar gracefully)
- [ ] Testar repl.py com código malicioso (eval, exec em string)
- [ ] Testar repl.py com loop infinito (timeout)
- [ ] Testar SQLite não vulnerável a injection (usar parâmetros)
- [ ] Testar http_server.py valida inputs obrigatórios

---

## Notas para Ralph

### Comando de teste
```bash
pytest tests/ -v
```

### Estrutura de um teste
```python
# tests/test_persistence.py
import pytest
from rlm_mcp.persistence import PersistenceManager

def test_save_and_load_variable(temp_db):
    pm = PersistenceManager(db_path=temp_db)
    pm.save_variable("test", "hello world")
    result = pm.load_variable("test")
    assert result == "hello world"
```

### Fixtures úteis (criar em conftest.py)
```python
import pytest
import tempfile
import os

@pytest.fixture
def temp_db():
    """SQLite em arquivo temporário"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)

@pytest.fixture
def sample_text():
    """Texto grande para testar indexação"""
    return "medo ansiedade trabalho família " * 25000  # ~200k chars
```

### Padrões do projeto
- Usar logging (não print)
- Imports no topo do arquivo
- Type hints em funções públicas
- Docstrings em funções públicas
