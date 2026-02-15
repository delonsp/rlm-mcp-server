# RLM MCP Server - Manual Completo

## O que e o RLM?

O **RLM (Recursive Language Model)** e um servidor MCP que da ao Claude a capacidade de processar **milhoes de caracteres** de dados. Diferente do RAG tradicional, o RLM carrega documentos inteiros em memoria e permite que o Claude escreva codigo Python sofisticado para analisar, cruzar e transformar dados massivos.

**Em resumo**: o Claude ganha um REPL Python persistente com acesso a arquivos do S3, PDFs, CSVs, codigo-fonte, e pode fazer analises que nenhum sistema de RAG consegue — cruzamento, agregacao, logica condicional, tudo via codigo.

### Arquitetura

```
Claude Code (local) ───── HTTPS ────► VPS (Docker: rlm-mcp-server)
                                           │
                                           ├── REPL Python (variaveis em memoria)
                                           ├── SQLite (persistencia automatica)
                                           ├── Minio/S3 (storage de arquivos)
                                           └── Tree-sitter (parsing de codigo)
```

- **19 tools** disponiveis via protocolo MCP
- **Persistencia automatica**: variaveis sobrevivem a restarts do servidor
- **Indexacao automatica**: textos grandes (100k+ chars) sao indexados para busca rapida
- **Embeddings vetoriais**: busca semantica via OpenAI embeddings
- **Sandbox seguro**: imports perigosos (os, subprocess, socket) sao bloqueados

---

## Configuracao

### Arquivo `.mcp.json`

Para conectar o Claude Code ao RLM, crie um arquivo `.mcp.json` na raiz do projeto:

```json
{
  "mcpServers": {
    "rlm": {
      "type": "sse",
      "url": "https://SEU-DOMINIO/sse",
      "headers": {
        "Authorization": "Bearer SUA-API-KEY"
      }
    }
  }
}
```

### Variaveis de Ambiente (servidor)

| Variavel | Padrao | Descricao |
|----------|--------|-----------|
| `RLM_API_KEY` | — | Chave de autenticacao (recomendado) |
| `RLM_MAX_MEMORY_MB` | `1024` | Memoria maxima para variaveis |
| `RLM_PORT` | `8765` | Porta HTTP |
| `OPENAI_API_KEY` | — | Para sub-chamadas LLM e embeddings vetoriais |
| `MISTRAL_API_KEY` | — | Para OCR de PDFs escaneados |
| `MINIO_ENDPOINT` | — | Endpoint do Minio/S3 |
| `MINIO_ACCESS_KEY` | — | Access key do Minio |
| `MINIO_SECRET_KEY` | — | Secret key do Minio |
| `MINIO_SECURE` | `true` | Usar HTTPS para Minio |
| `RLM_RESPONSE_VERBOSITY` | `compact` | Verbosidade das respostas: compact, normal, verbose |
| `RLM_CLEANUP_STRATEGY` | `weighted` | Estrategia de GC: weighted, lru, lfu, size |
| `RLM_MAX_CONCURRENT_TASKS` | `3` | Workers para tasks assincronas |
| `RLM_BATCH_MAX_WORKERS` | `4` | Workers para operacoes batch S3 |
| `RLM_EMBEDDING_MODE` | `openai` | Modo de embeddings: openai, disabled |
| `RLM_PERSIST_DIR` | `/persist` | Diretorio do SQLite |

---

## Tools — Referencia Completa

### 1. Carregar Dados

#### `rlm_load_data` — Carrega dados diretos

Carrega uma string diretamente em uma variavel do REPL. A variavel e automaticamente persistida no SQLite.

```
rlm_load_data(name="meus_dados", data="conteudo aqui", data_type="text")
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `name` | string | Sim | Nome da variavel |
| `data` | string | Sim | Conteudo a carregar |
| `data_type` | string | Nao | `text` (padrao), `json`, `lines`, `csv`, `code` |

**Tipos de dados:**
- `text` — String simples
- `json` — Parse para dict/list
- `lines` — Split por `\n` para lista de strings
- `csv` — Parse para lista de dicts (cada linha vira um dict)
- `code` — Texto com parsing estrutural tree-sitter (detecta linguagem automaticamente)

---

#### `rlm_load_file` — Carrega arquivo do volume /data

```
rlm_load_file(name="doc", path="/data/relatorio.pdf", data_type="pdf")
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `name` | string | Sim | Nome da variavel |
| `path` | string | Sim | Caminho do arquivo (deve comecar com `/data/`) |
| `data_type` | string | Nao | `text`, `json`, `lines`, `csv`, `code`, `pdf`, `pdf_ocr` |

**Tipos adicionais:**
- `pdf` — Extrai texto de PDF (auto-detecta se precisa OCR)
- `pdf_ocr` — Forca OCR via Mistral API (para PDFs escaneados)
- `code` — Parse estrutural com tree-sitter (detecta linguagem pelo nome do arquivo)

---

#### `rlm_load_s3` — Carrega arquivo(s) do Minio/S3

O arquivo e baixado direto do Minio para o servidor, sem passar pelo contexto do Claude. Suporta modo unitario e batch.

**Modo unitario:**
```
rlm_load_s3(key="data/planilha.csv", name="dados", data_type="csv")
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `key` | string | Sim* | Caminho do objeto no bucket |
| `name` | string | Sim* | Nome da variavel |
| `bucket` | string | Nao | Nome do bucket (padrao: `claude-code`) |
| `data_type` | string | Nao | `text`, `json`, `lines`, `csv`, `code`, `pdf`, `pdf_ocr` |
| `skip_if_exists` | boolean | Nao | Se True (padrao), pula download se variavel ja existe |

**Modo batch** (carrega multiplos arquivos em paralelo):
```
rlm_load_s3(keys=[
    {"key": "data/doc1.txt", "name": "doc1"},
    {"key": "data/doc2.csv", "name": "doc2", "data_type": "csv"},
    {"key": "data/doc3.json", "name": "doc3", "data_type": "json"}
])
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `keys` | array | Sim | Lista de objetos com `key`, `name`, e opcionalmente `data_type` |
| `bucket` | string | Nao | Bucket (padrao: `claude-code`) |

Se o batch tiver mais de 5 arquivos ou mais de 50MB total, roda como task assincrona.

> *`key` e `name` sao obrigatorios no modo unitario; no modo batch, use `keys`.

**Exemplos praticos:**
```
# CSV grande
rlm_load_s3(key="data/vendas.csv", name="vendas", data_type="csv")

# PDF com texto selecionavel
rlm_load_s3(key="pdfs/contrato.pdf", name="contrato", data_type="pdf")

# PDF escaneado (OCR)
rlm_load_s3(key="pdfs/receita_medica.pdf", name="receita", data_type="pdf_ocr")

# Codigo-fonte
rlm_load_s3(key="code/app.py", name="app", data_type="code")

# Batch: carregar 3 documentos de uma vez
rlm_load_s3(keys=[
    {"key": "docs/cap1.txt", "name": "cap1"},
    {"key": "docs/cap2.txt", "name": "cap2"},
    {"key": "docs/cap3.txt", "name": "cap3"}
])
```

---

### 2. Executar Codigo

#### `rlm_execute` — Executa Python no REPL

Executa codigo Python em um REPL persistente. Variaveis criadas permanecem entre execucoes. Use `print()` para retornar dados ao Claude.

```
rlm_execute(code="print(len(meus_dados))")
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `code` | string | Sim | Codigo Python para executar |

**Sandbox de seguranca:**
- Imports permitidos: `re`, `json`, `math`, `collections`, `datetime`, `csv`, `statistics`, `itertools`, `functools`, `string`, `textwrap`, `difflib`, `hashlib`, `base64`, `html`, `urllib.parse`
- Imports bloqueados: `os`, `subprocess`, `socket`, `requests`, `http`, `sys`
- Funcoes bloqueadas: `open()`, `exec()`, `eval()`, `__import__()`, acesso a dunder attrs

**Funcoes auxiliares pre-definidas:**

| Funcao | Descricao | Retorno |
|--------|-----------|---------|
| `buscar(texto, termo)` | Busca termo (case-insensitive) | `[{posicao, linha, contexto}]` |
| `contar(texto, termo)` | Conta ocorrencias | `{total, por_linha}` |
| `extrair_secao(texto, inicio, fim)` | Extrai texto entre marcadores | `[{conteudo, linha_inicio, linha_fim}]` |
| `resumir_tamanho(bytes)` | Humaniza tamanho | `"1.5 MB"` |

**Exemplos avancados:**
```python
# Indice semantico manual
indice = {c: [] for c in ['medo', 'trabalho', 'familia']}
for i, linha in enumerate(texto.split('\n')):
    for c in indice:
        if c in linha.lower():
            indice[c].append({'linha': i, 'ctx': linha[:100]})
print(json.dumps({k: len(v) for k, v in indice.items()}))

# Analise cruzada entre variaveis
from collections import defaultdict
scores = defaultdict(int)
for sintoma in ['cefaleia', 'nausea', 'febre']:
    if sintoma in doc1.lower():
        scores['doc1'] += 1
    if sintoma in doc2.lower():
        scores['doc2'] += 1
print(sorted(scores.items(), key=lambda x: -x[1]))

# Sub-chamada LLM (se OPENAI_API_KEY configurada)
resposta = llm_query("Resuma este texto em 3 pontos:", contexto=texto[:5000])
print(resposta)
```

---

### 3. Gerenciamento de Variaveis

#### `rlm_list_vars` — Lista variaveis

```
rlm_list_vars()
rlm_list_vars(offset=10, limit=5)
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `offset` | integer | Nao | Pular N variaveis (padrao: 0) |
| `limit` | integer | Nao | Maximo a retornar (padrao: 50) |

---

#### `rlm_var_info` — Info detalhada de uma variavel

```
rlm_var_info(name="dados")
```

Retorna: tipo, tamanho, contagem de acessos, status de pin, preview do conteudo.

---

#### `rlm_clear` — Remove variaveis

```
rlm_clear(name="dados")        # Remove uma variavel
rlm_clear(all=true)             # Remove TODAS as variaveis
```

---

#### `rlm_memory` — Uso de memoria e persistencia

```
rlm_memory()
```

Retorna: memoria total usada, numero de variaveis, porcentagem do limite, estatisticas de persistencia (variaveis salvas, indices, embeddings).

---

#### `rlm_pin_var` — Proteger variavel do GC

Variaveis pinned nunca sao removidas pelo garbage collector automatico.

```
rlm_pin_var(name="dados_importantes", pin=true)   # Proteger
rlm_pin_var(name="dados_importantes", pin=false)   # Desproteger
```

---

### 4. S3 / Minio

#### `rlm_list_buckets` — Lista buckets

```
rlm_list_buckets()
```

---

#### `rlm_list_s3` — Lista objetos em um bucket

```
rlm_list_s3()
rlm_list_s3(bucket="claude-code", prefix="data/", limit=20, offset=0)
```

---

#### `rlm_upload_url` — Upload de URL para S3

O servidor baixa o arquivo da URL e envia direto para o S3, sem passar pelo Claude.

```
rlm_upload_url(url="https://example.com/dados.csv", key="data/dados.csv")
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `url` | string | Sim | URL do arquivo |
| `key` | string | Sim | Caminho destino no bucket |
| `bucket` | string | Nao | Bucket (padrao: `claude-code`) |

---

#### `rlm_save_to_s3` — Salva variavel(eis) no S3

Exporta variaveis do REPL para o S3 como arquivo(s). Suporta modo unitario e batch.

**Modo unitario:**
```
rlm_save_to_s3(var_name="resultado", key="output/analise.json", format="json")
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `var_name` | string | Sim* | Nome da variavel |
| `key` | string | Sim* | Caminho destino no bucket |
| `bucket` | string | Nao | Bucket (padrao: `claude-code`) |
| `format` | string | Nao | `auto` (padrao), `text`, `json` |

**Modo batch** (exporta multiplas variaveis em paralelo):
```
rlm_save_to_s3(vars=[
    {"var_name": "resultado1", "key": "output/res1.json", "format": "json"},
    {"var_name": "resultado2", "key": "output/res2.txt"}
])
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `vars` | array | Sim | Lista de objetos com `var_name`, `key`, e opcionalmente `format` |
| `bucket` | string | Nao | Bucket (padrao: `claude-code`) |

> *`var_name` e `key` sao obrigatorios no modo unitario; no modo batch, use `vars`.

---

### 5. Processamento de PDF

#### `rlm_process_pdf` — Extrai texto de PDF grande

Workflow em duas etapas para PDFs grandes (evita timeout):

```
# Etapa 1: Processar PDF e salvar .txt no bucket
rlm_process_pdf(key="pdfs/livro.pdf")

# Etapa 2: Carregar o texto extraido
rlm_load_s3(key="pdfs/livro.txt", name="livro", data_type="text")
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `key` | string | Sim | Caminho do PDF no bucket |
| `bucket` | string | Nao | Bucket (padrao: `claude-code`) |
| `output_key` | string | Nao | Caminho do .txt (padrao: mesmo path, extensao .txt) |
| `method` | string | Nao | `auto` (padrao), `pdfplumber`, `ocr` |

**Metodos:**
- `auto` — Usa pdfplumber primeiro; se pouco texto for extraido, fallback para OCR
- `pdfplumber` — Rapido, para PDFs com texto selecionavel
- `ocr` — Mistral OCR, para PDFs escaneados (requer `MISTRAL_API_KEY`)

PDFs grandes rodam como task assincrona automaticamente.

---

### 6. Busca e Indexacao

#### `rlm_search_index` — Busca no indice de uma variavel

Textos com 100k+ caracteres sao indexados automaticamente ao carregar.

```
rlm_search_index(var_name="texto", terms=["medo", "ansiedade"])
rlm_search_index(var_name="texto", terms=["medo"], mode="hybrid")
rlm_search_index(var_name="texto", terms=["medo", "fracasso"], require_all=true)
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `var_name` | string | Sim | Nome da variavel indexada |
| `terms` | array | Sim | Termos para buscar |
| `mode` | string | Nao | `keyword` (padrao), `semantic`, `hybrid` |
| `require_all` | boolean | Nao | Se true, exige todos os termos (so keyword) |
| `limit` | integer | Nao | Max resultados por termo (padrao: 20) |
| `offset` | integer | Nao | Paginacao (padrao: 0) |

**Modos de busca:**
- **keyword** — Busca exata por termos no texto. Rapido, sem dependencias.
- **semantic** — Busca por similaridade vetorial usando embeddings OpenAI. Encontra conteudo semanticamente relacionado mesmo sem match exato.
- **hybrid** — Combina keyword + semantic usando Reciprocal Rank Fusion (RRF). Os resultados que aparecem em ambos os modos recebem score mais alto.

> Busca semantica e hibrida requerem `OPENAI_API_KEY` e textos >= 100k chars para auto-embedding.

---

### 7. Busca Estrutural em Codigo

#### `rlm_search_code` — Busca funcoes, classes, metodos

Usa tree-sitter para parsear codigo-fonte e buscar simbolos estruturalmente.

```
rlm_search_code(var_name="meu_codigo")
rlm_search_code(var_name="meu_codigo", kind="function")
rlm_search_code(var_name="meu_codigo", query="parse", include_source=true)
```

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `var_name` | string | Sim | Variavel com o codigo |
| `query` | string | Nao | Busca substring no nome do simbolo |
| `kind` | string | Nao | `function`, `class`, `method`, `import`, `variable` |
| `include_source` | boolean | Nao | Incluir codigo-fonte (padrao: false) |
| `language` | string | Nao | Forcar linguagem (auto-detecta se omitido) |

**Linguagens suportadas:** Python, JavaScript, TypeScript, Go, Rust, Java, C, C++

**Deteccao automatica:**
- Por extensao do arquivo (`.py`, `.js`, `.ts`, etc.) quando carregado do S3 ou `/data`
- Por conteudo (heuristica) quando carregado via `rlm_load_data` com `data_type="code"`

---

### 8. Colecoes

Colecoes agrupam variaveis por assunto, permitindo busca unificada em todos os documentos de um tema. Todas as operacoes de colecao usam uma unica tool com o parametro `action`.

#### `rlm_collection` — Gerenciar colecoes

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `action` | string | Sim | `create`, `add`, `list`, `info`, `rebuild`, `search` |
| `name` | string | Depende | Nome da colecao (para create, add, info, rebuild, search) |
| `description` | string | Nao | Descricao (apenas para create) |
| `vars` | array | Depende | Lista de nomes de variaveis (para add) |
| `terms` | array | Depende | Termos de busca (para search) |
| `require_all` | boolean | Nao | Exigir todos os termos na busca (padrao: false) |
| `limit` | integer | Nao | Max resultados (padrao: 20, para search) |
| `offset` | integer | Nao | Paginacao (padrao: 0, para search) |

**Exemplos:**
```
# Criar colecao
rlm_collection(action="create", name="homeopatia", description="Materiais de homeopatia unicista")

# Adicionar variaveis
rlm_collection(action="add", name="homeopatia", vars=["scholten1", "scholten2", "kent"])

# Listar colecoes
rlm_collection(action="list")

# Info de uma colecao
rlm_collection(action="info", name="homeopatia")

# Reconstruir indice (apos atualizar variaveis)
rlm_collection(action="rebuild", name="homeopatia")

# Buscar em todos os documentos da colecao
rlm_collection(action="search", name="homeopatia", terms=["medo", "ansiedade"])
```

---

### 9. Tasks Assincronas

Operacoes longas (PDFs grandes, batch S3 volumoso) rodam automaticamente como tasks assincronas. Todas as operacoes de task usam uma unica tool com o parametro `action`.

#### `rlm_task` — Gerenciar tasks assincronas

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|-------------|-----------|
| `action` | string | Sim | `list`, `status`, `cancel` |
| `task_id` | string | Depende | ID da task (para status e cancel) |
| `status` | string | Nao | Filtrar por status na listagem (para list) |

**Exemplos:**
```
# Listar todas as tasks
rlm_task(action="list")

# Listar apenas tasks em execucao
rlm_task(action="list", status="running")

# Ver status de uma task
rlm_task(action="status", task_id="abc12345")

# Cancelar task
rlm_task(action="cancel", task_id="abc12345")
```

Status possiveis: `pending`, `running`, `completed`, `failed`, `cancelled`.

---

### 10. Ajuda

#### `rlm_help` — Guia de uso integrado

Retorna documentacao diretamente do servidor, sem necessidade de arquivos locais.

```
rlm_help()                          # Visao geral
rlm_help(topic="workflows")         # Workflows comuns
rlm_help(topic="collections")       # Como usar colecoes
rlm_help(topic="code")              # Busca em codigo
rlm_help(topic="pdf")               # Processamento de PDF
rlm_help(topic="search")            # Busca e indexacao
rlm_help(topic="s3")                # Operacoes S3
rlm_help(topic="security")          # Seguranca e sandbox
rlm_help(topic="config")            # Variaveis de ambiente
```

---

## MCP Resources

O servidor expoe 3 resources via protocolo MCP:

| URI | Descricao |
|-----|-----------|
| `rlm://variables` | Lista variaveis persistidas com metadados |
| `rlm://memory` | Uso de memoria do REPL |
| `rlm://collections` | Lista de colecoes com contagem de variaveis |

---

## Workflows Comuns

### Workflow 1: Analisar CSV grande

```
# 1. Carregar
rlm_load_s3(key="data/vendas.csv", name="vendas", data_type="csv")

# 2. Explorar estrutura
rlm_execute(code="print(len(vendas)); print(vendas[0].keys())")

# 3. Analisar
rlm_execute(code="""
from collections import Counter
cidades = Counter(r['cidade'] for r in vendas)
print(cidades.most_common(10))
""")

# 4. Salvar resultado
rlm_save_to_s3(var_name="resultado", key="output/top_cidades.json", format="json")
```

### Workflow 2: Processar PDF grande

```
# 1. Processar PDF (salva .txt no bucket)
rlm_process_pdf(key="pdfs/livro_500p.pdf")

# 2. Se demorar, verificar progresso
rlm_task(action="list")
rlm_task(action="status", task_id="...")

# 3. Carregar texto extraido
rlm_load_s3(key="pdfs/livro_500p.txt", name="livro", data_type="text")

# 4. Buscar no indice (auto-criado para textos 100k+)
rlm_search_index(var_name="livro", terms=["capitulo", "conclusao"])
```

### Workflow 3: Biblioteca de documentos com colecoes

```
# 1. Carregar varios documentos de uma vez (batch)
rlm_load_s3(keys=[
    {"key": "docs/manual1.txt", "name": "manual1"},
    {"key": "docs/manual2.txt", "name": "manual2"},
    {"key": "docs/manual3.txt", "name": "manual3"}
])

# 2. Criar colecao e adicionar
rlm_collection(action="create", name="manuais", description="Manuais tecnicos")
rlm_collection(action="add", name="manuais", vars=["manual1", "manual2", "manual3"])

# 3. Buscar em todos de uma vez
rlm_collection(action="search", name="manuais", terms=["instalacao", "configuracao"])
```

### Workflow 4: Analisar codigo-fonte

```
# 1. Carregar codigo do S3
rlm_load_s3(key="code/server.py", name="server", data_type="code")

# 2. Ver estrutura
rlm_search_code(var_name="server")

# 3. Buscar funcoes especificas
rlm_search_code(var_name="server", query="handle", kind="function")

# 4. Ver codigo-fonte de uma classe
rlm_search_code(var_name="server", query="Router", kind="class", include_source=true)
```

### Workflow 5: Busca hibrida em texto grande

```
# 1. Carregar texto grande (auto-indexa e auto-embed se >= 100k chars)
rlm_load_s3(key="docs/livro_completo.txt", name="livro", data_type="text")

# 2. Busca por keyword (exata)
rlm_search_index(var_name="livro", terms=["ansiedade"], mode="keyword")

# 3. Busca semantica (por significado)
rlm_search_index(var_name="livro", terms=["preocupacao excessiva"], mode="semantic")

# 4. Busca hibrida (combina ambos)
rlm_search_index(var_name="livro", terms=["medo do futuro"], mode="hybrid")
```

### Workflow 6: Upload e processamento de dados externos

```
# 1. Upload de URL diretamente para o S3
rlm_upload_url(url="https://dados.gov.br/dataset.csv", key="data/gov_dataset.csv")

# 2. Carregar para analise
rlm_load_s3(key="data/gov_dataset.csv", name="gov", data_type="csv")

# 3. Analisar
rlm_execute(code="print(f'{len(gov)} registros, colunas: {list(gov[0].keys())}')")
```

### Workflow 7: Exportar multiplos resultados para S3

```
# Salvar varias variaveis de uma vez (batch)
rlm_save_to_s3(vars=[
    {"var_name": "analise1", "key": "output/analise1.json", "format": "json"},
    {"var_name": "analise2", "key": "output/analise2.json", "format": "json"},
    {"var_name": "resumo", "key": "output/resumo.txt"}
])
```

---

## Organizacao de Arquivos no S3

Estrutura recomendada para o bucket `claude-code`:

```
claude-code/
├── data/          # Dados estruturados (.csv, .json, .txt, .xml)
├── pdfs/          # Documentos PDF (.pdf)
├── code/          # Codigo-fonte (.py, .js, .ts, etc.)
├── logs/          # Arquivos de log (.log)
├── output/        # Resultados de analises
└── test/          # Arquivos temporarios de teste
```

**Upload via Minio CLI (`mc`):**
```bash
mc cp planilha.csv minio/claude-code/data/
mc cp relatorio.pdf minio/claude-code/pdfs/
mc cp app.py minio/claude-code/code/
mc ls minio/claude-code/ --recursive
```

---

## Persistencia e Ciclo de Vida

### Persistencia automatica (SQLite)

Toda variavel carregada via `rlm_load_data`, `rlm_load_s3`, ou `rlm_load_file` e automaticamente persistida no SQLite. Ao reiniciar o servidor, todas as variaveis sao restauradas. Use `rlm_memory()` para ver estatisticas de persistencia.

### Indexacao automatica

Textos com 100k+ caracteres recebem automaticamente:
- **Indice de keywords** — termos relevantes mapeados para posicoes no texto
- **Embeddings vetoriais** — chunks de texto com vetores OpenAI (se `OPENAI_API_KEY` configurada)

### Garbage Collector

Quando a memoria atinge 80% do limite (`RLM_MAX_MEMORY_MB`), o GC remove automaticamente variaveis menos importantes. O scoring considera:

- **Recencia** — quando foi acessada pela ultima vez
- **Frequencia** — quantas vezes foi acessada
- **Tamanho** — variaveis maiores tem prioridade de remocao

Variaveis **pinned** (`rlm_pin_var`) nunca sao removidas.

Estrategias de cleanup (`RLM_CLEANUP_STRATEGY`):
- `weighted` (padrao) — Score ponderado: (recencia x frequencia) / (1 + tamanho_mb)
- `lru` — Remove a menos recentemente usada
- `lfu` — Remove a menos frequentemente usada
- `size` — Remove a maior primeiro

---

## Seguranca

### Sandbox do REPL

O REPL Python bloqueia:
- Imports perigosos: `os`, `subprocess`, `socket`, `requests`, `http`, `sys`
- Funcoes perigosas: `open()`, `exec()`, `eval()`, `__import__()`
- Acesso a dunder attrs (`__class__`, `__bases__`, etc.)

### Rate Limiting

| Endpoint | Limite | Janela |
|----------|--------|--------|
| `/message` (SSE) | 100 req | 60s (por session) |
| `/mcp` (direto) | 100 req | 60s (por IP) |
| `rlm_upload_url` | 10 uploads | 60s |

### Limites de Memoria

- Variavel individual: max `RLM_MAX_VAR_SIZE_MB` (padrao: 50MB)
- Total do REPL: max `RLM_MAX_MEMORY_MB` (padrao: 1024MB)
- Volume `/data`: read-only

---

## Observabilidade

| Endpoint | Descricao |
|----------|-----------|
| `/health` | Status, memoria, versao |
| `/metrics` | Request counts, latencia (p50/p95/p99), tool calls, rate limits |

Todas as respostas incluem header `X-Request-Id` para tracing.

---

## Respostas Compactas

O servidor usa respostas compactas por padrao (`RLM_RESPONSE_VERBOSITY=compact`) para economizar tokens:

| Tool | Formato |
|------|---------|
| Load | `[var:nome \| 1.5MB \| str \| persisted \| indexed:50t]` |
| Execute | `[exec:OK \| 15ms \| out:42c]` + stdout |
| List vars | `4 vars \| nome:str:1.5MB, dados:list:500KB` |
| Search | `"medo":15hits \| top results...` |
| Batch | `[batch:5/7 loaded \| ok:d1,d2,d3 \| err:d4(not found)]` |
| Pin | `[pin:nome \| pinned]` |
| Code | `[code:nome \| python \| 5 symbols \| function:foo,bar]` |
| Collection | `[collection:homeopatia \| 5 vars \| created]` |
| Task | `[task:abc123 \| running \| 45%]` |

Para respostas detalhadas, configure `RLM_RESPONSE_VERBOSITY=normal` ou `verbose`.

---

## Resumo das 19 Tools

| # | Tool | Descricao |
|---|------|-----------|
| 1 | `rlm_execute` | Executa Python no REPL |
| 2 | `rlm_load_data` | Carrega dados diretos |
| 3 | `rlm_load_file` | Carrega arquivo de /data |
| 4 | `rlm_load_s3` | Carrega arquivo(s) do S3 (unitario ou batch) |
| 5 | `rlm_list_vars` | Lista variaveis |
| 6 | `rlm_var_info` | Info de uma variavel |
| 7 | `rlm_clear` | Remove variaveis |
| 8 | `rlm_memory` | Memoria + persistencia |
| 9 | `rlm_pin_var` | Pin/unpin do GC |
| 10 | `rlm_list_buckets` | Lista buckets S3 |
| 11 | `rlm_list_s3` | Lista objetos S3 |
| 12 | `rlm_upload_url` | Upload URL para S3 |
| 13 | `rlm_save_to_s3` | Salva variavel(eis) no S3 (unitario ou batch) |
| 14 | `rlm_process_pdf` | Processa PDF grande |
| 15 | `rlm_search_index` | Busca keyword/semantic/hybrid |
| 16 | `rlm_search_code` | Busca estrutural em codigo |
| 17 | `rlm_collection` | Gerencia colecoes (create/add/list/info/rebuild/search) |
| 18 | `rlm_task` | Gerencia tasks (list/status/cancel) |
| 19 | `rlm_help` | Documentacao integrada |

---

## Troubleshooting

| Problema | Causa | Solucao |
|----------|-------|---------|
| "Variavel nao possui indice" | Texto < 100k chars | Normal — indexacao so ocorre em textos grandes. Use `rlm_execute` com `buscar()` |
| "Import bloqueado" | Import perigoso no sandbox | Use imports permitidos (re, json, math, collections, etc.) |
| "Objeto nao encontrado" no S3 | Key errada | Use `rlm_list_s3(prefix="...")` para verificar |
| Timeout em PDF grande | PDF muito grande para processamento sincrono | Use `rlm_process_pdf` (workflow 2 etapas) |
| Busca semantica sem resultados | Sem embeddings | Verifique se `OPENAI_API_KEY` esta configurada e texto >= 100k chars |
| Variavel desapareceu | GC removeu | Use `rlm_pin_var` para proteger variaveis importantes |
| 502 Bad Gateway | Container fora da rede | Verificar label `traefik.docker.network=dokploy-network` |
| Codigo antigo apos deploy | Docker cache | Incrementar `CACHE_BUST` no Dockerfile |
