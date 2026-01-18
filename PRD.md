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
- [ ] Testar delete_variable remove do banco
- [ ] Testar list_variables retorna metadados corretos
- [ ] Testar save_index e load_index (roundtrip de índice semântico)
- [ ] Testar clear_all remove todas as variáveis
- [ ] Testar get_stats retorna contagens corretas
- [ ] Testar create_collection e list_collections
- [ ] Testar add_to_collection e get_collection_vars
- [ ] Testar delete_collection remove associações mas não variáveis

## Fase 3: Testes do Módulo indexer.py

- [ ] Testar create_index gera índice com termos padrão
- [ ] Testar create_index com additional_terms indexa termos customizados
- [ ] Testar TextIndex.search retorna matches corretos
- [ ] Testar TextIndex.search_multiple com require_all=False (OR)
- [ ] Testar TextIndex.search_multiple com require_all=True (AND)
- [ ] Testar auto_index_if_large indexa apenas textos >= 100k chars
- [ ] Testar _detect_structure detecta headers markdown
- [ ] Testar TextIndex.to_dict e from_dict (serialização)

## Fase 4: Testes do Módulo repl.py

- [ ] Testar execute com código simples (print, atribuição)
- [ ] Testar execute preserva variáveis entre execuções
- [ ] Testar execute bloqueia imports perigosos (os, subprocess, socket)
- [ ] Testar execute permite imports seguros (re, json, math, collections)
- [ ] Testar load_data com data_type="text"
- [ ] Testar load_data com data_type="json"
- [ ] Testar load_data com data_type="csv"
- [ ] Testar load_data com data_type="lines"
- [ ] Testar get_memory_usage retorna valores razoáveis
- [ ] Testar clear_namespace limpa variáveis

## Fase 5: Testes do Módulo s3_client.py (com mocks)

- [ ] Criar mock do MinIO client em conftest.py
- [ ] Testar is_configured retorna False sem credenciais
- [ ] Testar list_buckets com mock retorna lista
- [ ] Testar list_objects com mock retorna objetos
- [ ] Testar get_object com mock retorna bytes
- [ ] Testar get_object_info com mock retorna metadados
- [ ] Testar object_exists com mock retorna True/False

## Fase 6: Testes do Módulo pdf_parser.py (com mocks)

- [ ] Testar extract_with_pdfplumber com PDF machine readable (criar fixture)
- [ ] Testar extract_with_pdfplumber retorna erro se arquivo não existe
- [ ] Testar extract_pdf com method="auto" usa pdfplumber primeiro
- [ ] Testar extract_pdf faz fallback para OCR se pdfplumber extrai pouco
- [ ] Testar split_pdf_into_chunks divide corretamente
- [ ] Mockar Mistral API para testar extract_with_mistral_ocr

## Fase 7: Testes de Integração HTTP (FastAPI TestClient)

- [ ] Testar endpoint /health retorna 200
- [ ] Testar MCP initialize retorna capabilities
- [ ] Testar MCP tools/list retorna todas as tools
- [ ] Testar tool rlm_execute com código simples
- [ ] Testar tool rlm_load_data carrega variável
- [ ] Testar tool rlm_list_vars lista variáveis carregadas
- [ ] Testar tool rlm_var_info retorna info da variável
- [ ] Testar tool rlm_clear limpa namespace
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
