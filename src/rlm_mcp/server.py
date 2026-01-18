"""
RLM MCP Server - Servidor Model Context Protocol para Recursive Language Model

Este servidor expõe tools MCP que permitem ao Claude Code:
1. Carregar dados massivos em variáveis (sem poluir contexto)
2. Executar código Python em REPL persistente
3. Consultar metadados das variáveis
4. Gerenciar memória

Baseado no paper "Recursive Language Models" (MIT CSAIL, 2025)
"""

import os
import json
import logging
import asyncio
from typing import Any
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)
from pydantic import BaseModel

from .repl import SafeREPL, ExecutionResult, VariableInfo
from .s3_client import get_s3_client
from .pdf_parser import extract_pdf

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rlm-mcp")

# Instância global do REPL
repl = SafeREPL(max_memory_mb=int(os.getenv("RLM_MAX_MEMORY_MB", "1024")))

# API Key para autenticação (se configurada)
API_KEY = os.getenv("RLM_API_KEY", "")


def create_server() -> Server:
    """Cria e configura o servidor MCP"""
    server = Server("rlm-mcp-server")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Lista todas as tools disponíveis"""
        return [
            Tool(
                name="rlm_execute",
                description="""Executa código Python no REPL persistente.

As variáveis criadas persistem entre execuções. Use print() para retornar dados.

IMPORTANTE: O código roda em sandbox seguro:
- Imports permitidos: re, json, math, collections, datetime, csv, etc.
- Imports bloqueados: os, subprocess, socket, requests, etc.
- Funções bloqueadas: exec, eval, open, etc.

Exemplo:
```python
# Variável 'data' já foi carregada com rlm_load_data
lines = data.split('\\n')
errors = [l for l in lines if 'ERROR' in l]
print(f"Encontrados {len(errors)} erros")
print(errors[:5])  # Primeiros 5
```""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Código Python para executar"
                        }
                    },
                    "required": ["code"]
                }
            ),
            Tool(
                name="rlm_load_data",
                description="""Carrega dados diretamente em uma variável do REPL.

Use para carregar dados massivos que ficarão em memória, sem poluir o contexto.

Tipos suportados:
- "text": String simples
- "json": Parse JSON para dict/list
- "lines": Split por \\n para lista
- "csv": Parse CSV para lista de dicts

Exemplo: Carregar um log de 100MB em 'logs' e depois usar rlm_execute para analisar.""",
                inputSchema={
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
                            "default": "text",
                            "description": "Tipo de parsing dos dados"
                        }
                    },
                    "required": ["name", "data"]
                }
            ),
            Tool(
                name="rlm_load_file",
                description="""Carrega arquivo do servidor em uma variável.

O arquivo deve estar no diretório /data do container.

Tipos suportados:
- "text": String simples
- "json": Parse JSON para dict/list
- "lines": Split por \\n para lista
- "csv": Parse CSV para lista de dicts
- "pdf": Extrai texto de PDF (auto-detecta se precisa OCR)
- "pdf_ocr": Força OCR via Mistral API (para PDFs escaneados)

Exemplo: rlm_load_file(name="doc", path="/data/relatorio.pdf", data_type="pdf")""",
                inputSchema={
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
                            "default": "text",
                            "description": "Tipo de parsing"
                        }
                    },
                    "required": ["name", "path"]
                }
            ),
            Tool(
                name="rlm_list_vars",
                description="""Lista todas as variáveis no REPL com metadados.

Retorna: nome, tipo, tamanho, preview (sem o conteúdo completo).

Use para saber quais dados estão disponíveis antes de executar código.""",
                inputSchema={
                    "type": "object",
                    "properties": {},
                }
            ),
            Tool(
                name="rlm_var_info",
                description="""Retorna informações detalhadas de uma variável específica.

Inclui: tipo, tamanho, preview, timestamps de criação/acesso.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Nome da variável"
                        }
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="rlm_clear",
                description="""Limpa variáveis do REPL.

Se 'name' for fornecido, limpa apenas essa variável.
Se 'all' for True, limpa todas as variáveis.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Nome da variável para limpar (opcional)"
                        },
                        "all": {
                            "type": "boolean",
                            "default": False,
                            "description": "Se True, limpa todas as variáveis"
                        }
                    },
                }
            ),
            Tool(
                name="rlm_memory",
                description="""Retorna estatísticas de uso de memória do REPL.

Útil para monitorar se está próximo do limite.""",
                inputSchema={
                    "type": "object",
                    "properties": {},
                }
            ),
            Tool(
                name="rlm_load_s3",
                description="""Carrega arquivo do Minio/S3 diretamente em uma variável.

O arquivo é baixado direto do Minio para o servidor RLM,
sem passar pelo contexto do Claude Code. Ideal para arquivos grandes.

Tipos suportados:
- "text": String simples
- "json": Parse JSON para dict/list
- "lines": Split por \\n para lista
- "csv": Parse CSV para lista de dicts
- "pdf": Extrai texto de PDF (auto-detecta se precisa OCR)
- "pdf_ocr": Força OCR via Mistral API (para PDFs escaneados)

Exemplo: rlm_load_s3(bucket="docs", key="relatorio.pdf", name="doc", data_type="pdf")""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bucket": {
                            "type": "string",
                            "description": "Nome do bucket no Minio"
                        },
                        "key": {
                            "type": "string",
                            "description": "Caminho/chave do objeto no bucket"
                        },
                        "name": {
                            "type": "string",
                            "description": "Nome da variável no REPL"
                        },
                        "data_type": {
                            "type": "string",
                            "enum": ["text", "json", "lines", "csv", "pdf", "pdf_ocr"],
                            "default": "text",
                            "description": "Tipo de parsing dos dados"
                        }
                    },
                    "required": ["bucket", "key", "name"]
                }
            ),
            Tool(
                name="rlm_list_buckets",
                description="""Lista buckets disponíveis no Minio.

Use para descobrir quais buckets existem antes de carregar arquivos.""",
                inputSchema={
                    "type": "object",
                    "properties": {},
                }
            ),
            Tool(
                name="rlm_list_s3",
                description="""Lista objetos em um bucket do Minio.

Retorna nome, tamanho e data de modificação dos arquivos.

Exemplo: rlm_list_s3(bucket="logs", prefix="app/") para listar só arquivos em app/""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bucket": {
                            "type": "string",
                            "description": "Nome do bucket"
                        },
                        "prefix": {
                            "type": "string",
                            "default": "",
                            "description": "Prefixo para filtrar (opcional)"
                        }
                    },
                    "required": ["bucket"]
                }
            ),
            Tool(
                name="rlm_upload_url",
                description="""Gera URL assinada para upload de arquivo para o Minio.

Use esta URL para fazer upload via HTTP PUT de arquivos grandes
diretamente para o Minio, sem passar pelo servidor RLM.

Exemplo: rlm_upload_url(bucket="data", key="docs/report.pdf")""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bucket": {
                            "type": "string",
                            "description": "Nome do bucket"
                        },
                        "key": {
                            "type": "string",
                            "description": "Caminho/chave do objeto no bucket"
                        },
                        "expires": {
                            "type": "integer",
                            "default": 3600,
                            "description": "Tempo de expiração em segundos (padrão: 1 hora)"
                        }
                    },
                    "required": ["bucket", "key"]
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Executa uma tool"""
        try:
            if name == "rlm_execute":
                result = repl.execute(arguments["code"])
                output = _format_execution_result(result)

            elif name == "rlm_load_data":
                result = repl.load_data(
                    name=arguments["name"],
                    data=arguments["data"],
                    data_type=arguments.get("data_type", "text")
                )
                output = _format_execution_result(result)

            elif name == "rlm_load_file":
                path = arguments["path"]
                data_type = arguments.get("data_type", "text")

                # Validação de segurança: só permite /data/
                if not path.startswith("/data/"):
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Erro: Caminho deve começar com /data/ por segurança"
                        )],
                        isError=True
                    )

                # Previne path traversal
                import os.path
                real_path = os.path.realpath(path)
                if not real_path.startswith("/data/"):
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Erro: Path traversal detectado"
                        )],
                        isError=True
                    )

                try:
                    # Tratamento especial para PDF
                    if data_type in ("pdf", "pdf_ocr"):
                        method = "ocr" if data_type == "pdf_ocr" else "auto"
                        pdf_result = extract_pdf(path, method=method)

                        if not pdf_result.success:
                            return CallToolResult(
                                content=[TextContent(
                                    type="text",
                                    text=f"Erro ao extrair PDF: {pdf_result.error}"
                                )],
                                isError=True
                            )

                        result = repl.load_data(
                            name=arguments["name"],
                            data=pdf_result.text,
                            data_type="text"
                        )

                        output = f"""✅ PDF carregado:
Arquivo: {path}
Páginas: {pdf_result.pages}
Método: {pdf_result.method}
Caracteres: {len(pdf_result.text)}

{_format_execution_result(result)}"""
                    else:
                        # Arquivos de texto normais
                        with open(path, 'r', encoding='utf-8', errors='replace') as f:
                            data = f.read()

                        result = repl.load_data(
                            name=arguments["name"],
                            data=data,
                            data_type=data_type
                        )
                        output = _format_execution_result(result)

                except FileNotFoundError:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Erro: Arquivo não encontrado: {path}"
                        )],
                        isError=True
                    )
                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Erro ao ler arquivo: {e}"
                        )],
                        isError=True
                    )

            elif name == "rlm_list_vars":
                vars_list = repl.list_variables()
                if not vars_list:
                    output = "Nenhuma variável no REPL."
                else:
                    lines = ["Variáveis no REPL:", ""]
                    for v in vars_list:
                        lines.append(f"  {v.name}: {v.type_name} ({v.size_human})")
                        lines.append(f"    Preview: {v.preview[:100]}...")
                        lines.append("")
                    output = "\n".join(lines)

            elif name == "rlm_var_info":
                info = repl.get_variable_info(arguments["name"])
                if not info:
                    output = f"Variável '{arguments['name']}' não encontrada."
                else:
                    output = f"""Variável: {info.name}
Tipo: {info.type_name}
Tamanho: {info.size_human} ({info.size_bytes} bytes)
Criada em: {info.created_at.isoformat()}
Último acesso: {info.last_accessed.isoformat()}

Preview:
{info.preview}"""

            elif name == "rlm_clear":
                if arguments.get("all"):
                    count = repl.clear_all()
                    output = f"Todas as {count} variáveis foram removidas."
                elif "name" in arguments:
                    if repl.clear_variable(arguments["name"]):
                        output = f"Variável '{arguments['name']}' removida."
                    else:
                        output = f"Variável '{arguments['name']}' não encontrada."
                else:
                    output = "Especifique 'name' ou 'all=true'."

            elif name == "rlm_memory":
                mem = repl.get_memory_usage()
                output = f"""Uso de Memória do REPL:
Total: {mem['total_human']}
Variáveis: {mem['variable_count']}
Limite: {mem['max_allowed_mb']} MB
Uso: {mem['usage_percent']:.1f}%"""

            elif name == "rlm_load_s3":
                import tempfile

                s3 = get_s3_client()
                if not s3.is_configured():
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Erro: Minio não configurado. Configure MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY."
                        )],
                        isError=True
                    )

                bucket = arguments["bucket"]
                key = arguments["key"]
                var_name = arguments["name"]
                data_type = arguments.get("data_type", "text")

                try:
                    # Obter info primeiro para mostrar tamanho
                    info = s3.get_object_info(bucket, key)
                    if not info:
                        return CallToolResult(
                            content=[TextContent(
                                type="text",
                                text=f"Erro: Objeto não encontrado: {bucket}/{key}"
                            )],
                            isError=True
                        )

                    # Tratamento especial para PDF
                    if data_type in ("pdf", "pdf_ocr"):
                        # Baixar PDF para arquivo temporário
                        pdf_bytes = s3.get_object(bucket, key)

                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            tmp.write(pdf_bytes)
                            tmp_path = tmp.name

                        # Extrair texto do PDF
                        method = "ocr" if data_type == "pdf_ocr" else "auto"
                        pdf_result = extract_pdf(tmp_path, method=method)

                        # Limpar arquivo temporário
                        import os as _os
                        _os.unlink(tmp_path)

                        if not pdf_result.success:
                            return CallToolResult(
                                content=[TextContent(
                                    type="text",
                                    text=f"Erro ao extrair PDF: {pdf_result.error}"
                                )],
                                isError=True
                            )

                        result = repl.load_data(
                            name=var_name,
                            data=pdf_result.text,
                            data_type="text"
                        )

                        output = f"""✅ PDF carregado do Minio:
Bucket: {bucket}
Objeto: {key}
Tamanho: {info['size_human']}
Páginas: {pdf_result.pages}
Método: {pdf_result.method}
Caracteres: {len(pdf_result.text)}

{_format_execution_result(result)}"""
                    else:
                        # Arquivos de texto normais
                        data = s3.get_object_text(bucket, key)

                        result = repl.load_data(
                            name=var_name,
                            data=data,
                            data_type=data_type
                        )

                        output = f"""✅ Carregado do Minio:
Bucket: {bucket}
Objeto: {key}
Tamanho: {info['size_human']}
Variável: {var_name} (tipo: {data_type})

{_format_execution_result(result)}"""

                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Erro ao carregar do Minio: {e}"
                        )],
                        isError=True
                    )

            elif name == "rlm_list_buckets":
                s3 = get_s3_client()
                if not s3.is_configured():
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Erro: Minio não configurado."
                        )],
                        isError=True
                    )

                try:
                    buckets = s3.list_buckets()
                    if not buckets:
                        output = "Nenhum bucket encontrado."
                    else:
                        output = "Buckets disponíveis:\n" + "\n".join(f"  - {b}" for b in buckets)
                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Erro ao listar buckets: {e}"
                        )],
                        isError=True
                    )

            elif name == "rlm_list_s3":
                s3 = get_s3_client()
                if not s3.is_configured():
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Erro: Minio não configurado."
                        )],
                        isError=True
                    )

                bucket = arguments["bucket"]
                prefix = arguments.get("prefix", "")

                try:
                    objects = s3.list_objects(bucket, prefix)
                    if not objects:
                        output = f"Nenhum objeto encontrado em {bucket}/{prefix}"
                    else:
                        lines = [f"Objetos em {bucket}/{prefix}:", ""]
                        for obj in objects[:50]:  # Limitar a 50
                            lines.append(f"  {obj['name']} ({obj['size_human']})")
                        if len(objects) > 50:
                            lines.append(f"  ... e mais {len(objects) - 50} objetos")
                        output = "\n".join(lines)
                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Erro ao listar objetos: {e}"
                        )],
                        isError=True
                    )

            elif name == "rlm_upload_url":
                s3 = get_s3_client()
                if not s3.is_configured():
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Erro: Minio não configurado."
                        )],
                        isError=True
                    )

                bucket = arguments["bucket"]
                key = arguments["key"]
                expires = arguments.get("expires", 3600)

                try:
                    url = s3.get_presigned_put_url(bucket, key, expires)
                    output = f"""URL para upload gerada:

URL: {url}

Use HTTP PUT para enviar o arquivo:
curl -X PUT -T seu_arquivo.pdf "{url}"

Expira em: {expires} segundos"""
                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=f"Erro ao gerar URL: {e}"
                        )],
                        isError=True
                    )

            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Tool desconhecida: {name}")],
                    isError=True
                )

            return CallToolResult(
                content=[TextContent(type="text", text=output)]
            )

        except Exception as e:
            logger.exception(f"Erro ao executar tool {name}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Erro interno: {e}")],
                isError=True
            )


def _format_execution_result(result: ExecutionResult) -> str:
    """Formata resultado de execução para output"""
    parts = []

    if result.stdout:
        parts.append(f"=== OUTPUT ===\n{result.stdout}")

    if result.stderr:
        parts.append(f"=== ERRORS ===\n{result.stderr}")

    if result.variables_changed:
        parts.append(f"=== VARIÁVEIS ALTERADAS ===\n{', '.join(result.variables_changed)}")

    parts.append(f"\n[Execução: {result.execution_time_ms:.1f}ms | Status: {'OK' if result.success else 'ERRO'}]")

    return "\n".join(parts) if parts else "Execução concluída sem output."


async def run_server():
    """Executa o servidor MCP via stdio"""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point"""
    logger.info("Iniciando RLM MCP Server...")
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
