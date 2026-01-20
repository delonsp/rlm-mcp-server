# PRD: RLM MCP Server - Melhorias Best Practices

## Objetivo

Implementar melhorias baseadas nas best practices de MCP (Model Context Protocol) e SQLite para aumentar performance, segurança e compliance com a especificação oficial.

## Referências

- [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
- [Anthropic - Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [Going Fast with SQLite and Python](https://charlesleifer.com/blog/going-fast-with-sqlite-and-python/)

## Arquitetura Atual

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

- Cada tarefa marcada [x] significa que `pytest tests/ -v` passa sem erros
- Melhorias não quebram funcionalidade existente
- Performance de SQLite melhorada com WAL mode

---

## Fase 1: SQLite Performance (WAL Mode)

- [x] Adicionar PRAGMA journal_mode=WAL no _init_db() de persistence.py
- [x] Adicionar PRAGMA synchronous=NORMAL para melhor performance
- [x] Adicionar PRAGMA cache_size=-64000 (64MB cache)
- [x] Criar teste para verificar que WAL mode está ativo
- [x] Criar teste de performance comparando antes/depois (opcional)

## Fase 2: Erros Visíveis ao Usuário

- [x] Em http_server.py, modificar rlm_load_s3 para mostrar erros de persistência no output
- [x] Em http_server.py, modificar rlm_load_data para mostrar erros de persistência no output
- [x] Criar constante SHOW_PERSISTENCE_ERRORS=True para controlar comportamento
- [x] Criar teste que verifica que erros de persistência aparecem no output

## Fase 3: Pagination para Grandes Resultados

- [x] Adicionar parâmetros offset e limit em rlm_search_index
- [x] Adicionar parâmetros offset e limit em rlm_search_collection
- [x] Adicionar parâmetros offset e limit em rlm_list_vars
- [x] Adicionar parâmetros offset e limit em rlm_list_s3
- [x] Criar testes para pagination em cada endpoint modificado

## Fase 4: Helper Functions Pré-definidas no REPL

- [x] Criar função buscar(texto, termo) no namespace inicial do REPL
- [x] Criar função contar(texto, termo) no namespace inicial do REPL
- [x] Criar função extrair_secao(texto, inicio, fim) no namespace inicial do REPL
- [x] Criar função resumir_tamanho(bytes) que retorna string humanizada
- [x] Documentar helpers na description do rlm_execute
- [x] Criar testes para cada helper function

## Fase 5: MCP Resources (Spec Compliance)

- [x] Adicionar suporte a resources/list no handle_mcp_request
- [x] Criar resource "rlm://variables" que lista variáveis persistidas
- [x] Criar resource "rlm://memory" que mostra uso de memória
- [x] Criar resource "rlm://collections" que lista coleções
- [x] Adicionar resources nas capabilities do initialize
- [ ] Criar testes para cada resource

## Fase 6: Rate Limiting Básico

- [ ] Criar classe RateLimiter com sliding window algorithm
- [ ] Adicionar rate limit de 100 requests/minuto por sessão SSE
- [ ] Adicionar rate limit de 10 uploads/minuto para rlm_upload_url
- [ ] Retornar erro 429 Too Many Requests quando limite excedido
- [ ] Criar testes para rate limiting

## Fase 7: Melhorias de Logging e Observabilidade

- [ ] Adicionar logging estruturado (JSON) como opção
- [ ] Criar endpoint /metrics com estatísticas básicas (requests, erros, latência)
- [ ] Adicionar request_id em cada requisição para tracing
- [ ] Criar teste para endpoint /metrics

## Fase 8: Documentação e Cleanup

- [ ] Atualizar CLAUDE.md com novas features (pagination, resources, helpers)
- [ ] Adicionar docstrings em todas as funções públicas que faltam
- [ ] Criar arquivo CHANGELOG.md com versão 0.2.0
- [ ] Atualizar version em http_server.py para 0.2.0

---

## Notas para Ralph

### Comando de teste
```bash
pytest tests/ -v
```

### Estrutura de um teste
```python
# tests/test_persistence.py
def test_wal_mode_enabled(temp_db):
    pm = PersistenceManager(db_path=temp_db)
    with sqlite3.connect(temp_db) as conn:
        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"
```

### Padrões do projeto
- Usar logging (não print)
- Imports no topo do arquivo
- Type hints em funções públicas
- Docstrings em funções públicas
- Testes em tests/ com pytest

### Ordem de execução
1. Fase 1 e 2 são prioritárias (alta prioridade, baixo esforço)
2. Fase 3 e 4 melhoram UX
3. Fase 5, 6 e 7 são compliance e segurança
4. Fase 8 é documentação final
