# AGENTS.md - Padrões do Projeto RLM MCP Server

Este arquivo contém padrões aprendidos durante o desenvolvimento. Ralph atualiza conforme descobre novos padrões.

## Arquitetura

- **Servidor HTTP MCP**: FastAPI em `http_server.py`, expõe tools via protocolo MCP
- **REPL Sandboxed**: `repl.py` executa Python com imports limitados por segurança
- **Persistência**: SQLite em `/persist/rlm_data.db` (pickle + zlib para compressão)
- **Indexação**: Automática para textos >= 100k caracteres

## Padrões de Código

- Usar `logger.info/warning/error` para logs, não `print()`
- Singleton pattern para `PersistenceManager` via `get_persistence()`
- Type hints em todas as funções públicas
- Docstrings descritivas em funções públicas

## Padrões de Teste

- Fixtures em `conftest.py` para recursos compartilhados
- Mock de dependências externas (MinIO, Mistral API)
- SQLite em memória ou tempfile para testes de persistência
- FastAPI TestClient para testes de integração HTTP

## Dependências Externas

- **MinIO**: Requer MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
- **Mistral OCR**: Requer MISTRAL_API_KEY (para PDFs escaneados)
- **OpenAI**: Requer OPENAI_API_KEY (para llm_client.py)

## Gotchas Conhecidos

- `repl.namespace` persiste entre execuções - limpar com `clear_namespace()`
- PDFs grandes (>40MB) precisam de chunking para Mistral OCR
- Índices são salvos separadamente das variáveis em `indices` table
