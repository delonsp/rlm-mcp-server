"""
Textos de ajuda do rlm_help — movidos do http_server (refator do call_tool).

Conteúdo estático puro: sem dependências de runtime.
"""

HELP_SECTIONS = {
    "workflows": """## Workflows Essenciais

1. **PDF grande (2 etapas)** — Evita timeout:
   rlm_process_pdf(key="pdfs/livro.pdf")  → salva pdfs/livro.txt no bucket
   rlm_load_s3(key="pdfs/livro.txt", name="livro")  → carrega texto rápido

2. **Batch load** — Múltiplos arquivos de uma vez:
   rlm_load_s3(keys=[{"key":"data/a.csv","name":"a","data_type":"csv"}, {"key":"data/b.csv","name":"b","data_type":"csv"}])

3. **Coleção** — Agrupar + buscar em vários docs:
   rlm_collection(action="create", name="manuais", description="Manuais técnicos")
   rlm_collection(action="add", name="manuais", vars=["doc1","doc2","doc3"])
   rlm_collection(action="search", name="manuais", terms=["instalação"])

4. **Análise com código** — Carregar → executar Python → salvar resultado:
   rlm_load_s3(key="data/vendas.csv", name="v", data_type="csv")
   rlm_execute(code="from collections import Counter; print(Counter(r['cidade'] for r in v).most_common(5))")
   rlm_save_to_s3(var_name="resultado", key="output/analise.json")""",

    "s3": """## Convenções S3

Bucket padrão: claude-code

Estrutura recomendada:
  data/    → Dados estruturados (.csv, .json, .txt)
  pdfs/    → PDFs (.pdf) e textos extraídos (.txt)
  code/    → Código-fonte (.py, .js, .ts)
  logs/    → Logs (.log)
  output/  → Resultados de análises

Upload de URL externa direto para S3:
  rlm_upload_url(url="https://example.com/data.csv", key="data/externo.csv")

Listar conteúdo:
  rlm_list_s3(prefix="data/", limit=20)
  rlm_list_buckets()""",

    "search": """## Busca e Indexação

**Auto-indexação**: Textos >= 100k chars recebem índice de keywords e embeddings vetoriais automaticamente ao carregar.

**Modos de busca** (rlm_search_index):
- keyword: busca exata por termo (rápido, sem API)
- semantic: busca por significado via embeddings OpenAI
- hybrid: combina ambos com Reciprocal Rank Fusion

Exemplos:
  rlm_search_index(var_name="livro", terms=["ansiedade"], mode="keyword")
  rlm_search_index(var_name="livro", terms=["preocupação excessiva"], mode="semantic")
  rlm_search_index(var_name="livro", terms=["medo do futuro"], mode="hybrid")

**Textos < 100k chars**: Use rlm_execute com buscar(texto, termo) ou Python direto.

**Busca em coleção** (rlm_collection action="search"): varre TODOS os docs de uma vez,
porém é LEXICAL (token), sem perna semântica. Passe termos como array de palavras, não
frase (frase casa literal → quase sempre zero; há auto-tokenize de fallback com aviso).
Para recall semântico/cross-idioma, rode rlm_search_index(mode="hybrid") fonte por fonte.""",

    "code": """## Análise de Código-Fonte

Carregar com data_type="code" para parsing estrutural tree-sitter:
  rlm_load_s3(key="code/app.py", name="app", data_type="code")

Linguagens suportadas: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++
Auto-detecção por extensão do arquivo ou conteúdo.

Buscar símbolos:
  rlm_search_code(var_name="app")  → todos os símbolos
  rlm_search_code(var_name="app", kind="function")  → só funções
  rlm_search_code(var_name="app", query="parse", include_source=true)  → com código""",

    "pdf": """## Processamento de PDF

**PDFs pequenos** (< 5MB): Carregar direto
  rlm_load_s3(key="pdfs/doc.pdf", name="doc", data_type="pdf")

**PDFs grandes**: Workflow em 2 etapas (evita timeout)
  rlm_process_pdf(key="pdfs/livro.pdf")  → extrai texto → salva .txt no bucket
  rlm_load_s3(key="pdfs/livro.txt", name="livro")  → carrega texto rápido

**PDFs escaneados**: Usar OCR (requer MISTRAL_API_KEY)
  rlm_load_s3(key="pdfs/scan.pdf", name="scan", data_type="pdf_ocr")

PDFs grandes rodam como task assíncrona:
  rlm_task_list()  → ver progresso
  rlm_task_status(task_id="...")  → resultado""",

    "collections": """## Coleções

Agrupam variáveis por assunto para busca unificada.
Tool consolidada: rlm_collection(action, ...)

Criar e popular:
  rlm_collection(action="create", name="docs", description="Documentação técnica")
  rlm_collection(action="add", name="docs", vars=["manual1", "manual2", "manual3"])

Buscar em todos de uma vez (LEXICAL — casa palavras/tokens, não significado):
  rlm_collection(action="search", name="docs", terms=["configuração", "instalação"])
  • Termos = ARRAY de palavras. Frase ("a b c") casa literal numa linha → quase zero;
    se não casar, o servidor tokeniza e re-busca (AND→OR) avisando que é fallback.
  • Frase literal mesmo? Use aspas no termo: terms=['"erro fatal"'].
  • snippet_len ajusta o tamanho do trecho (default 150).
  • Recall por significado/sinônimo/cross-idioma NÃO existe aqui: use
    rlm_search_index(var=..., mode="hybrid") por fonte.

Listar e inspecionar:
  rlm_collection(action="list")
  rlm_collection(action="info", name="docs")

Se a busca parar de funcionar após atualização:
  rlm_collection(action="rebuild", name="docs")

Remover a coleção (as variáveis membras ficam):
  rlm_collection(action="delete", name="docs")""",

    "execute": """## REPL Python

Variáveis persistem entre execuções na sessão, mas SÓ dados (funções/objetos e
mutação in-place não voltam — o execute roda isolado em subprocesso). Vars
carregadas via load_* são persistidas no SQLite (sobrevivem a restart); vars
criadas no execute ficam só em memória.

**Helpers pré-definidos:**
  buscar(texto, termo) → [{posicao, linha, contexto}]
  contar(texto, termo) → {total, por_linha}
  extrair_secao(texto, inicio, fim) → [{conteudo, linha_inicio, linha_fim}]
  resumir_tamanho(bytes) → "1.5 MB"

**Sub-chamada LLM** (requer OPENAI_API_KEY):
  resposta = llm_query("Resuma:", contexto=texto[:5000])

**Imports permitidos**: re, json, math, statistics, collections, itertools, textwrap, unicodedata, datetime, time, calendar, dataclasses, typing, enum, csv, html, xml.etree.ElementTree, hashlib, base64
**Bloqueados**: os, subprocess, socket, requests, open(), exec(), eval(), functools/operator/string (vetores de escape), gzip/zipfile/tarfile (I/O de arquivo)

**GC**: Quando memória atinge 80%, variáveis menos usadas são removidas.
  rlm_pin_var(name="importante") protege do GC.
  rlm_memory() mostra uso atual.""",

    "security": """## Segurança

**Sandbox do REPL:**
  Bloqueados: os, subprocess, socket, requests, http, sys, open(), exec(), eval(), __import__()
  Permitidos: re, json, math, statistics, collections, itertools, functools, operator, string, textwrap, unicodedata, datetime, time, calendar, dataclasses, typing, enum, csv, html, xml.etree.ElementTree, hashlib, base64, gzip, zipfile, tarfile

**Rate Limiting:**
  /message (SSE): 100 req / 60s (por session)
  /mcp (direto): 100 req / 60s (por IP)
  rlm_upload_url: 10 uploads / 60s

**Limites de Memória:**
  Variável individual: RLM_MAX_VAR_SIZE_MB (padrão: 50MB)
  Total do REPL: RLM_MAX_MEMORY_MB (padrão: 1024MB)
  Volume /data: read-only""",

    "config": """## Variáveis de Ambiente

**Obrigatórias/Recomendadas:**
  RLM_API_KEY — Autenticação Bearer token
  OPENAI_API_KEY — Sub-chamadas LLM e embeddings vetoriais
  MISTRAL_API_KEY — OCR de PDFs escaneados

**S3/Minio:**
  MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE (padrão: true)

**Limites:**
  RLM_MAX_MEMORY_MB (padrão: 1024) — Memória total do REPL
  RLM_MAX_VAR_SIZE_MB (padrão: 50) — Limite por variável
  RLM_MAX_CONCURRENT_TASKS (padrão: 3) — Workers para tasks assíncronas
  RLM_BATCH_MAX_WORKERS (padrão: 4) — Workers para operações batch S3

**Comportamento:**
  RLM_RESPONSE_VERBOSITY (padrão: compact) — compact, normal, verbose
  RLM_CLEANUP_STRATEGY (padrão: weighted) — weighted, lru, lfu, size
  RLM_EMBEDDING_MODE (padrão: openai) — openai, disabled
  RLM_PERSIST_DIR (padrão: /persist) — Diretório do SQLite""",
}


def get_help_text(topic: str = "all") -> str:
    """Retorna texto de ajuda para o tópico especificado."""
    if topic != "all" and topic in HELP_SECTIONS:
        return HELP_SECTIONS[topic]

    # all: header + todas as seções
    parts = [
        "# RLM MCP Server — Guia Rápido",
        "",
        "Servidor MCP com REPL Python persistente para processar milhões de caracteres.",
        "19 tools | Persistência SQLite | S3/Minio | Tree-sitter | Embeddings vetoriais",
        "",
        "Tópicos: rlm_help(topic=\"workflows|s3|search|code|pdf|collections|execute\")",
        "",
    ]
    for section in HELP_SECTIONS.values():
        parts.append(section)
        parts.append("")
    return "\n".join(parts)
