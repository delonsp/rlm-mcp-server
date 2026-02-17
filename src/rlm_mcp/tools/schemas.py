"""Tool schemas for MCP server.

Defines the TOOL_SCHEMAS constant with all tool definitions extracted from get_tools_list().

Consolidated from 29 to 19 tools:
- Collections (6→1): rlm_collection with action parameter
- Tasks (3→1): rlm_task with action parameter
- Batch S3 load: absorbed into rlm_load_s3 via keys[] parameter
- Batch S3 upload: absorbed into rlm_save_to_s3 via vars[] parameter
- persistence_stats: merged into rlm_memory
"""

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "rlm_execute",
        "description": """Executa código Python no REPL persistente.

As variáveis criadas persistem entre execuções. Use print() para retornar dados.

IMPORTANTE: O código roda em sandbox seguro:
- Imports permitidos: re, json, math, collections, datetime, csv, etc.
- Imports bloqueados: os, subprocess, socket, requests, etc.

=== PADRÕES DE USO AVANÇADO (RLM) ===

O poder do RLM está em escrever código SOFISTICADO para analisar dados massivos.
NÃO use apenas regex simples. Use estas estratégias:

1. ÍNDICE SEMÂNTICO - mapear conceitos para localização:
   ```python
   indice = {conceito: [] for conceito in ['medo', 'trabalho', 'pai']}
   for i, linha in enumerate(texto.split('\\n')):
       for c in indice:
           if c in linha.lower():
               indice[c].append({'linha': i, 'ctx': linha[:100]})
   ```

2. ANÁLISE CRUZADA - buscar múltiplos critérios:
   ```python
   def analise_diferencial(sintomas, texto):
       scores = defaultdict(int)
       for sintoma in sintomas:
           if sintoma in secao.lower():
               scores[remedio] += 1
       return sorted(scores.items(), key=lambda x: -x[1])
   ```

3. ESTRUTURA DOCUMENTAL - mapear seções/capítulos:
   ```python
   secoes = re.findall(r'^#+ (.+)$', texto, re.MULTILINE)
   ```

4. FUNÇÕES REUTILIZÁVEIS - definir helpers que persistem entre chamadas

=== FUNÇÕES AUXILIARES PRÉ-DEFINIDAS ===

O REPL já inclui estas funções prontas para uso:

1. buscar(texto, termo) -> list[dict]
   Busca um termo no texto (case-insensitive).
   Retorna: [{'posicao': int, 'linha': int, 'contexto': str}]
   Exemplo: buscar(meu_texto, "erro")

2. contar(texto, termo) -> dict
   Conta ocorrências de um termo (case-insensitive).
   Retorna: {'total': int, 'por_linha': {linha: count}}
   Exemplo: contar(meu_texto, "warning")

3. extrair_secao(texto, inicio, fim) -> list[dict]
   Extrai seções entre marcadores (case-insensitive).
   Retorna: [{'conteudo': str, 'posicao_inicio': int, 'posicao_fim': int, 'linha_inicio': int, 'linha_fim': int}]
   Exemplo: extrair_secao(doc, "## Intro", "## Conclusão")

4. resumir_tamanho(bytes) -> str
   Converte bytes para formato humanizado.
   Retorna: string como "1.5 MB", "256 KB"
   Exemplo: resumir_tamanho(1048576) -> "1.0 MB"

LEMBRE-SE: Você tem acesso a MILHÕES de caracteres. Use Python para
fazer análises que RAG não consegue - cruzamento, agregação, lógica condicional.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Código Python para executar"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "rlm_load_data",
        "description": """Carrega dados diretamente em uma variável do REPL.

A variável é automaticamente persistida no banco local (SQLite)
e sobrevive a restarts do servidor.

Tipos suportados:
- "text": String simples
- "json": Parse JSON para dict/list
- "lines": Split por \\n para lista
- "csv": Parse CSV para lista de dicts
- "code": Parse estrutural com tree-sitter""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nome da variável"
                },
                "data": {
                    "type": "string",
                    "description": "Dados para carregar"
                },
                "data_type": {
                    "type": "string",
                    "enum": ["text", "json", "lines", "csv", "code"],
                    "default": "text"
                }
            },
            "required": ["name", "data"]
        }
    },
    {
        "name": "rlm_load_file",
        "description": """Carrega arquivo do volume /data do servidor em uma variável.

Tipos: text, json, lines, csv, code, pdf, pdf_ocr""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nome da variável"
                },
                "path": {
                    "type": "string",
                    "description": "Caminho do arquivo (deve começar com /data/)"
                },
                "data_type": {
                    "type": "string",
                    "enum": ["text", "json", "lines", "csv", "code", "pdf", "pdf_ocr"],
                    "default": "text"
                }
            },
            "required": ["name", "path"]
        }
    },
    {
        "name": "rlm_load_s3",
        "description": """Carrega arquivo(s) do Minio/S3 em variável(is) do REPL.

Modo único: rlm_load_s3(key="data/doc.csv", name="doc", data_type="csv")
Modo batch: rlm_load_s3(keys=[{"key":"data/a.csv","name":"a","data_type":"csv"}, {"key":"data/b.txt","name":"b"}])

No modo batch, todos os arquivos são baixados em paralelo.
Batch grande (>5 arquivos ou >50MB) roda como task assíncrona.

Tipos: text, json, lines, csv, code, pdf, pdf_ocr
Variáveis são auto-persistidas no SQLite.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Caminho/chave do objeto no bucket (modo único)"
                },
                "name": {
                    "type": "string",
                    "description": "Nome da variável no REPL (modo único)"
                },
                "keys": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Caminho no bucket"},
                            "name": {"type": "string", "description": "Nome da variável"},
                            "data_type": {"type": "string", "enum": ["text", "json", "lines", "csv", "code", "pdf", "pdf_ocr"], "default": "text"}
                        },
                        "required": ["key", "name"]
                    },
                    "description": "Lista de objetos para carregar (modo batch)"
                },
                "bucket": {
                    "type": "string",
                    "default": "claude-code",
                    "description": "Nome do bucket (padrão: claude-code)"
                },
                "data_type": {
                    "type": "string",
                    "enum": ["text", "json", "lines", "csv", "code", "pdf", "pdf_ocr"],
                    "default": "text",
                    "description": "Tipo de parsing (modo único)"
                },
                "skip_if_exists": {
                    "type": "boolean",
                    "default": True,
                    "description": "Se True, pula download se variável já existe (modo único)"
                }
            }
        }
    },
    {
        "name": "rlm_list_vars",
        "description": "Lista variáveis no REPL com metadados. Suporta paginação via offset/limit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0}
            }
        }
    },
    {
        "name": "rlm_var_info",
        "description": "Retorna informações detalhadas de uma variável (tipo, tamanho, acessos, pin status).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Nome da variável"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "rlm_clear",
        "description": "Remove variáveis do REPL. Use 'name' para uma específica ou 'all=true' para todas.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "all": {"type": "boolean", "default": False}
            }
        }
    },
    {
        "name": "rlm_memory",
        "description": "Retorna uso de memória do REPL e estatísticas de persistência SQLite (variáveis salvas, índices, embeddings).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "rlm_pin_var",
        "description": """Protege (pin) ou desprotege (unpin) uma variável do garbage collector.

Variáveis pinned NUNCA são removidas pelo auto-cleanup.

Exemplo: rlm_pin_var(name="dados_importantes", pin=True)""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Nome da variável"},
                "pin": {"type": "boolean", "default": True, "description": "True=proteger, False=desproteger"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "rlm_list_buckets",
        "description": "Lista buckets disponíveis no Minio/S3.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "rlm_list_s3",
        "description": """Lista objetos em um bucket do Minio. Suporta paginação.

Exemplo: rlm_list_s3(prefix="data/", limit=20)""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bucket": {"type": "string", "default": "claude-code"},
                "prefix": {"type": "string", "default": ""},
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0}
            }
        }
    },
    {
        "name": "rlm_upload_url",
        "description": """Faz upload de arquivo de uma URL para o Minio/S3.

O servidor baixa diretamente da URL e envia para o Minio.

Exemplo: rlm_upload_url(url="https://example.com/data.csv", key="data/file.csv")""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL do arquivo"},
                "key": {"type": "string", "description": "Caminho destino no bucket"},
                "bucket": {"type": "string", "default": "claude-code"}
            },
            "required": ["url", "key"]
        }
    },
    {
        "name": "rlm_save_to_s3",
        "description": """Salva variável(is) do REPL para o Minio/S3.

Modo único: rlm_save_to_s3(var_name="resultado", key="output/res.json", format="json")
Modo batch: rlm_save_to_s3(vars=[{"var_name":"r1","key":"output/r1.json","format":"json"}, {"var_name":"r2","key":"output/r2.txt"}])

Formatos: auto (detecta), text, json""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "var_name": {"type": "string", "description": "Nome da variável (modo único)"},
                "key": {"type": "string", "description": "Caminho destino no bucket (modo único)"},
                "vars": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "var_name": {"type": "string"},
                            "key": {"type": "string"},
                            "format": {"type": "string", "enum": ["auto", "text", "json"], "default": "auto"}
                        },
                        "required": ["var_name", "key"]
                    },
                    "description": "Lista de variáveis para exportar (modo batch)"
                },
                "bucket": {"type": "string", "default": "claude-code"},
                "format": {"type": "string", "enum": ["auto", "text", "json"], "default": "auto", "description": "Formato (modo único)"}
            }
        }
    },
    {
        "name": "rlm_process_pdf",
        "description": """Processa PDF do Minio e salva texto extraído de volta no bucket.

WORKFLOW 2 ETAPAS para PDFs grandes:
1. rlm_process_pdf(key="pdfs/livro.pdf") → salva pdfs/livro.txt
2. rlm_load_s3(key="pdfs/livro.txt", name="livro") → carrega texto

Métodos: auto (padrão), pdfplumber, ocr (Mistral, requer MISTRAL_API_KEY)""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Caminho do PDF no bucket"},
                "bucket": {"type": "string", "default": "claude-code"},
                "output_key": {"type": "string", "description": "Caminho do .txt (padrão: mesmo path .txt)"},
                "method": {"type": "string", "enum": ["auto", "pdfplumber", "ocr"], "default": "auto"}
            },
            "required": ["key"]
        }
    },
    {
        "name": "rlm_search_index",
        "description": """Busca termos no índice de uma variável.

Índice auto-criado para textos >= 100k chars.

Modos: keyword (exato, padrão), semantic (vetorial, requer OPENAI_API_KEY), hybrid (combina ambos com RRF)

Exemplo: rlm_search_index(var_name="livro", terms=["medo","ansiedade"], mode="hybrid")""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "var_name": {"type": "string", "description": "Nome da variável indexada"},
                "terms": {"type": "array", "items": {"type": "string"}, "description": "Termos para buscar"},
                "mode": {"type": "string", "enum": ["keyword", "semantic", "hybrid"], "default": "keyword"},
                "require_all": {"type": "boolean", "default": False, "description": "Exigir TODOS os termos (só keyword)"},
                "limit": {"type": "integer", "default": 20},
                "offset": {"type": "integer", "default": 0}
            },
            "required": ["var_name", "terms"]
        }
    },
    {
        "name": "rlm_search_code",
        "description": """Busca estrutural em código-fonte via tree-sitter.

Busca funções, classes, métodos por nome ou tipo.
Linguagens: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++ (auto-detecta).

Exemplo: rlm_search_code(var_name="app", query="parse", kind="function", include_source=true)""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "var_name": {"type": "string", "description": "Variável com o código"},
                "query": {"type": "string", "description": "Substring no nome do símbolo"},
                "kind": {"type": "string", "enum": ["function", "class", "method", "import", "variable"]},
                "include_source": {"type": "boolean", "default": False},
                "language": {"type": "string", "description": "Forçar linguagem (auto-detecta se omitido)"}
            },
            "required": ["var_name"]
        }
    },
    {
        "name": "rlm_collection",
        "description": """Gerencia coleções de variáveis (agrupamento por assunto para busca unificada).

Ações:
- create: Cria coleção. Params: name, description
- add: Adiciona variáveis. Params: name, vars[]
- list: Lista todas as coleções
- info: Info de uma coleção. Params: name
- rebuild: Reconstrói índice. Params: name
- search: Busca em toda a coleção. Params: name, terms[], limit, offset

Exemplos:
  rlm_collection(action="create", name="docs", description="Documentação")
  rlm_collection(action="add", name="docs", vars=["doc1","doc2"])
  rlm_collection(action="search", name="docs", terms=["instalação"])
  rlm_collection(action="list")""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "add", "list", "info", "rebuild", "search"],
                    "description": "Ação a executar"
                },
                "name": {"type": "string", "description": "Nome da coleção (para create/add/info/rebuild/search)"},
                "description": {"type": "string", "description": "Descrição (para create)"},
                "vars": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variáveis para adicionar (para add)"
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Termos para buscar (para search)"
                },
                "limit": {"type": "integer", "default": 20, "description": "Max resultados (para search)"},
                "offset": {"type": "integer", "default": 0, "description": "Paginação (para search)"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "rlm_task",
        "description": """Gerencia tasks assíncronas (operações longas como PDFs grandes ou batch S3).

Ações:
- list: Lista tasks. Param opcional: status (pending/running/completed/failed/cancelled)
- status: Status de uma task. Param: task_id
- cancel: Cancela task. Param: task_id

Exemplos:
  rlm_task(action="list")
  rlm_task(action="status", task_id="abc123")
  rlm_task(action="cancel", task_id="abc123")""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "status", "cancel"],
                    "description": "Ação a executar"
                },
                "task_id": {"type": "string", "description": "ID da task (para status/cancel)"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "running", "completed", "failed", "cancelled"],
                    "description": "Filtro de status (para list)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "rlm_help",
        "description": "Guia de uso do RLM MCP Server com workflows, convenções e dicas.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["all", "workflows", "s3", "search", "code", "pdf", "collections", "execute", "security", "config"],
                    "default": "all"
                }
            }
        }
    }
]
