"""Tool schemas for MCP server.

Defines the TOOL_SCHEMAS constant with all tool definitions extracted from get_tools_list().
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
- "csv": Parse CSV para lista de dicts""",
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
                    "enum": ["text", "json", "lines", "csv"],
                    "default": "text"
                }
            },
            "required": ["name", "data"]
        }
    },
    {
        "name": "rlm_load_file",
        "description": """Carrega arquivo do servidor em uma variável.

O arquivo deve estar no diretório /data do container.

Tipos suportados:
- text: String simples
- json: Parse JSON para dict/list
- lines: Split por \\n para lista
- csv: Parse CSV para lista de dicts
- pdf: Extrai texto de PDF (auto-detecta método)
- pdf_ocr: Força OCR para PDFs escaneados (requer MISTRAL_API_KEY)""",
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
                    "enum": ["text", "json", "lines", "csv", "pdf", "pdf_ocr"],
                    "default": "text"
                }
            },
            "required": ["name", "path"]
        }
    },
    {
        "name": "rlm_list_vars",
        "description": "Lista todas as variáveis no REPL com metadados (nome, tipo, tamanho, preview). Suporta paginação via offset/limit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Máximo de variáveis a retornar"
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "Número de variáveis a pular (para paginação)"
                }
            }
        }
    },
    {
        "name": "rlm_var_info",
        "description": "Retorna informações detalhadas de uma variável específica.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nome da variável"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "rlm_clear",
        "description": "Limpa variáveis do REPL. Use 'name' para uma específica ou 'all=true' para todas.",
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
        "description": "Retorna estatísticas de uso de memória do REPL.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "rlm_load_s3",
        "description": """Carrega arquivo do Minio/S3 diretamente em uma variável.

O arquivo é baixado direto do Minio para o servidor RLM,
sem passar pelo contexto do Claude Code. Ideal para arquivos grandes.

Ao carregar, a variável é automaticamente persistida no banco local (SQLite)
e sobrevive a restarts do servidor.

Tipos suportados:
- text: String simples
- json: Parse JSON para dict/list
- lines: Split por \\n para lista
- csv: Parse CSV para lista de dicts
- pdf: Extrai texto de PDF (auto-detecta método)
- pdf_ocr: Força OCR para PDFs escaneados (requer MISTRAL_API_KEY)

Opções:
- skip_if_exists: Se True (padrão), pula download se variável já existe no REPL

Exemplo: rlm_load_s3(key="pdfs/doc.pdf", name="doc", data_type="pdf")""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Caminho/chave do objeto no bucket"
                },
                "name": {
                    "type": "string",
                    "description": "Nome da variável no REPL"
                },
                "bucket": {
                    "type": "string",
                    "default": "claude-code",
                    "description": "Nome do bucket (padrão: claude-code)"
                },
                "data_type": {
                    "type": "string",
                    "enum": ["text", "json", "lines", "csv", "pdf", "pdf_ocr"],
                    "default": "text",
                    "description": "Tipo de parsing dos dados"
                },
                "skip_if_exists": {
                    "type": "boolean",
                    "default": True,
                    "description": "Se True, pula download se variável já existe (padrão: True)"
                }
            },
            "required": ["key", "name"]
        }
    },
    {
        "name": "rlm_list_buckets",
        "description": """Lista buckets disponíveis no Minio.

Use para descobrir quais buckets existem antes de carregar arquivos.""",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "rlm_list_s3",
        "description": """Lista objetos em um bucket do Minio.

Retorna nome, tamanho e data de modificação dos arquivos.
Suporta paginação via offset e limit.

Exemplo: rlm_list_s3() para listar bucket padrão, ou rlm_list_s3(prefix="logs/") para filtrar""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bucket": {
                    "type": "string",
                    "default": "claude-code",
                    "description": "Nome do bucket (padrão: claude-code)"
                },
                "prefix": {
                    "type": "string",
                    "default": "",
                    "description": "Prefixo para filtrar (opcional)"
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Máximo de objetos a retornar (padrão: 50)"
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "Número de objetos a pular para paginação (padrão: 0)"
                }
            }
        }
    },
    {
        "name": "rlm_upload_url",
        "description": """Faz upload de arquivo de uma URL para o Minio/S3.

O servidor RLM baixa o arquivo diretamente da URL e envia para o Minio,
sem passar pelo contexto do Claude Code. Ideal para arquivos grandes.

Exemplo: rlm_upload_url(url="https://example.com/data.csv", key="data/file.csv")""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL do arquivo para baixar"
                },
                "key": {
                    "type": "string",
                    "description": "Caminho/chave do objeto no bucket"
                },
                "bucket": {
                    "type": "string",
                    "default": "claude-code",
                    "description": "Nome do bucket (padrão: claude-code)"
                }
            },
            "required": ["url", "key"]
        }
    },
    {
        "name": "rlm_process_pdf",
        "description": """Processa PDF do Minio e salva texto extraído de volta no bucket.

WORKFLOW EM DUAS ETAPAS para PDFs grandes:
1. rlm_process_pdf() → Extrai texto e salva como .txt no bucket (esta ferramenta)
2. rlm_load_s3() com o .txt → Carrega texto rápido para análise

O PDF é processado no servidor e o texto é salvo no mesmo bucket.
NÃO carrega em variável (evita timeout em PDFs grandes).

Métodos de extração:
- auto: Usa pdfplumber primeiro, fallback para OCR se pouco texto
- pdfplumber: Força pdfplumber (rápido, para PDFs com texto selecionável)
- ocr: Força Mistral OCR (para PDFs escaneados, requer MISTRAL_API_KEY)

Exemplo: rlm_process_pdf(key="pdfs/livro.pdf") → salva pdfs/livro.txt""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Caminho do PDF no bucket (ex: pdfs/documento.pdf)"
                },
                "bucket": {
                    "type": "string",
                    "default": "claude-code",
                    "description": "Nome do bucket (padrão: claude-code)"
                },
                "output_key": {
                    "type": "string",
                    "description": "Caminho para salvar o .txt (padrão: mesmo path com extensão .txt)"
                },
                "method": {
                    "type": "string",
                    "enum": ["auto", "pdfplumber", "ocr"],
                    "default": "auto",
                    "description": "Método de extração (padrão: auto)"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "rlm_search_index",
        "description": """Busca termos no índice semântico de uma variável.

O índice é criado automaticamente ao carregar textos grandes (100k+ chars).
Permite busca rápida sem varrer o texto todo.

Modos de busca:
- termo único: retorna linhas onde o termo aparece
- múltiplos termos: retorna linhas com qualquer um dos termos
- require_all=true: retorna apenas linhas com TODOS os termos

Exemplo: rlm_search_index(var_name="scholten1", terms=["medo", "fracasso"])""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "var_name": {
                    "type": "string",
                    "description": "Nome da variável indexada"
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de termos para buscar"
                },
                "require_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "Se True, retorna apenas linhas com TODOS os termos"
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Máximo de resultados por termo"
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "Número de resultados a pular (para paginação)"
                }
            },
            "required": ["var_name", "terms"]
        }
    },
    {
        "name": "rlm_persistence_stats",
        "description": """Retorna estatísticas do banco SQLite local (variáveis salvas, índices, etc).

Mostra quais variáveis estão persistidas no SQLite e sobreviverão ao restart do servidor.""",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "rlm_collection_create",
        "description": """Cria uma nova coleção para agrupar variáveis por assunto.

Coleções permitem organizar variáveis relacionadas (ex: homeopatia, nutrição, fitoterapia)
e fazer buscas em todas de uma vez.

Exemplo: rlm_collection_create(name="homeopatia", description="Materiais de homeopatia unicista")""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nome da coleção (único)"
                },
                "description": {
                    "type": "string",
                    "description": "Descrição opcional da coleção"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "rlm_collection_add",
        "description": """Adiciona variáveis a uma coleção existente.

A coleção é criada automaticamente se não existir.

Exemplo: rlm_collection_add(collection="homeopatia", vars=["scholten1", "scholten2", "kent"])""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Nome da coleção"
                },
                "vars": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de nomes de variáveis para adicionar"
                }
            },
            "required": ["collection", "vars"]
        }
    },
    {
        "name": "rlm_collection_list",
        "description": """Lista todas as coleções existentes com contagem de variáveis.""",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "rlm_collection_info",
        "description": """Retorna informações detalhadas de uma coleção específica.

Mostra todas as variáveis na coleção com seus tamanhos.""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nome da coleção"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "rlm_collection_rebuild",
        "description": """Reconstrói o índice combinado de uma coleção.

Use após atualizar o servidor ou quando a busca na coleção não funcionar.
Concatena todas as variáveis e cria índice semântico unificado.

Exemplo: rlm_collection_rebuild(name="injetaveis")""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nome da coleção"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "rlm_search_collection",
        "description": """Busca termos em TODAS as variáveis de uma coleção.

Busca unificada que varre todos os documentos da coleção e retorna
resultados agrupados por documento.

Exemplo: rlm_search_collection(collection="homeopatia", terms=["medo", "ansiedade"])""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Nome da coleção"
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de termos para buscar"
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Máximo de resultados por documento/termo"
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "Número de resultados a pular (para paginação)"
                }
            },
            "required": ["collection", "terms"]
        }
    },
    {
        "name": "rlm_save_to_s3",
        "description": """Salva uma variável do REPL para o Minio/S3.

Exporta para o S3 como arquivo, útil para compartilhar externamente
ou manter o formato original no bucket.
A variável já está salva localmente no SQLite — use esta ferramenta
apenas quando precisar do arquivo no S3.

Formatos suportados:
- auto: Detecta automaticamente (str → text, dict/list → json)
- text: Força salvar como texto plano (.txt)
- json: Força salvar como JSON (.json)

Exemplo: rlm_save_to_s3(var_name="resultado", key="output/analise.json")""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "var_name": {
                    "type": "string",
                    "description": "Nome da variável no REPL para salvar"
                },
                "key": {
                    "type": "string",
                    "description": "Caminho/chave do objeto no bucket (ex: output/resultado.json)"
                },
                "bucket": {
                    "type": "string",
                    "default": "claude-code",
                    "description": "Nome do bucket (padrão: claude-code)"
                },
                "format": {
                    "type": "string",
                    "enum": ["auto", "text", "json"],
                    "default": "auto",
                    "description": "Formato de serialização (padrão: auto)"
                }
            },
            "required": ["var_name", "key"]
        }
    }
]
