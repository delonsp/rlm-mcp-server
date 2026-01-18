"""
RLM MCP Server - HTTP/SSE Transport

Expõe o MCP server via HTTP com Server-Sent Events (SSE).
Permite conexão direta do Claude Code via URL, sem SSH tunnel.

Endpoints:
- GET  /health     → Health check
- GET  /sse        → SSE stream para MCP
- POST /message    → Envia mensagem para MCP
"""

import os
import json
import asyncio
import logging
from typing import Any
from contextlib import asynccontextmanager
from datetime import datetime
import uuid
import hashlib
import hmac

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .repl import SafeREPL, ExecutionResult, VariableInfo
from .s3_client import get_s3_client

# Configuração
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rlm-http")

# API Key para autenticação
API_KEY = os.getenv("RLM_API_KEY", "")
MAX_MEMORY_MB = int(os.getenv("RLM_MAX_MEMORY_MB", "1024"))

# Instância global do REPL
repl = SafeREPL(max_memory_mb=MAX_MEMORY_MB)

# Sessões SSE ativas
sse_sessions: dict[str, asyncio.Queue] = {}


# =============================================================================
# Autenticação
# =============================================================================

async def verify_api_key(request: Request):
    """Verifica API key se configurada"""
    if not API_KEY:
        return True

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if hmac.compare_digest(token, API_KEY):
            return True

    # Também aceita como query param para SSE (browsers não enviam headers custom em EventSource)
    token = request.query_params.get("token", "")
    if token and hmac.compare_digest(token, API_KEY):
        return True

    raise HTTPException(status_code=401, detail="Invalid or missing API key")


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hooks"""
    logger.info(f"RLM MCP Server iniciando (max_memory={MAX_MEMORY_MB}MB)")
    yield
    logger.info("RLM MCP Server encerrando")


app = FastAPI(
    title="RLM MCP Server",
    description="Recursive Language Model via MCP over HTTP/SSE",
    version="0.1.0",
    lifespan=lifespan
)

# CORS para permitir conexões do Claude Code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja isso
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Models
# =============================================================================

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: dict | None = None


# =============================================================================
# MCP Protocol Implementation
# =============================================================================

def handle_mcp_request(request: MCPRequest) -> MCPResponse:
    """Processa uma requisição MCP"""
    try:
        method = request.method
        params = request.params or {}

        if method == "initialize":
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "rlm-mcp-server",
                        "version": "0.1.0"
                    }
                }
            )

        elif method == "notifications/initialized":
            # Notificação, não precisa de resposta
            return None

        elif method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={
                    "tools": get_tools_list()
                }
            )

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            result = call_tool(tool_name, tool_args)
            return MCPResponse(
                id=request.id,
                result=result
            )

        else:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            )

    except Exception as e:
        logger.exception(f"Erro ao processar request MCP: {e}")
        return MCPResponse(
            id=request.id,
            error={
                "code": -32603,
                "message": str(e)
            }
        )


def get_tools_list() -> list[dict]:
    """Retorna lista de tools disponíveis"""
    return [
        {
            "name": "rlm_execute",
            "description": """Executa código Python no REPL persistente.

As variáveis criadas persistem entre execuções. Use print() para retornar dados.

IMPORTANTE: O código roda em sandbox seguro:
- Imports permitidos: re, json, math, collections, datetime, csv, etc.
- Imports bloqueados: os, subprocess, socket, requests, etc.

Exemplo:
```python
lines = data.split('\\n')
errors = [l for l in lines if 'ERROR' in l]
print(f"Encontrados {len(errors)} erros")
```""",
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

O arquivo deve estar no diretório /data do container.""",
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
                        "enum": ["text", "json", "lines", "csv"],
                        "default": "text"
                    }
                },
                "required": ["name", "path"]
            }
        },
        {
            "name": "rlm_list_vars",
            "description": "Lista todas as variáveis no REPL com metadados (nome, tipo, tamanho, preview).",
            "inputSchema": {
                "type": "object",
                "properties": {}
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

Exemplo: rlm_load_s3(key="logs/app.log", name="logs")""",
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
                        "enum": ["text", "json", "lines", "csv"],
                        "default": "text",
                        "description": "Tipo de parsing dos dados"
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
                    }
                }
            }
        },
        {
            "name": "rlm_upload_s3",
            "description": """Faz upload de dados para o Minio/S3.

Permite enviar texto/dados diretamente para um arquivo no Minio.
Útil para salvar resultados de análises ou criar arquivos de teste.

Exemplo: rlm_upload_s3(key="logs/test.log", data="conteudo do arquivo")""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Caminho/chave do objeto no bucket"
                    },
                    "data": {
                        "type": "string",
                        "description": "Conteúdo do arquivo (texto)"
                    },
                    "bucket": {
                        "type": "string",
                        "default": "claude-code",
                        "description": "Nome do bucket (padrão: claude-code)"
                    }
                },
                "required": ["key", "data"]
            }
        }
    ]


def call_tool(name: str, arguments: dict) -> dict:
    """Executa uma tool e retorna resultado"""
    try:
        if name == "rlm_execute":
            result = repl.execute(arguments["code"])
            return {
                "content": [
                    {"type": "text", "text": format_execution_result(result)}
                ]
            }

        elif name == "rlm_load_data":
            result = repl.load_data(
                name=arguments["name"],
                data=arguments["data"],
                data_type=arguments.get("data_type", "text")
            )
            return {
                "content": [
                    {"type": "text", "text": format_execution_result(result)}
                ]
            }

        elif name == "rlm_load_file":
            path = arguments["path"]

            # Validação de segurança
            if not path.startswith("/data/"):
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Caminho deve começar com /data/"}
                    ],
                    "isError": True
                }

            import os.path
            real_path = os.path.realpath(path)
            if not real_path.startswith("/data"):
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Path traversal detectado"}
                    ],
                    "isError": True
                }

            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    data = f.read()

                result = repl.load_data(
                    name=arguments["name"],
                    data=data,
                    data_type=arguments.get("data_type", "text")
                )
                return {
                    "content": [
                        {"type": "text", "text": format_execution_result(result)}
                    ]
                }
            except FileNotFoundError:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro: Arquivo não encontrado: {path}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_list_vars":
            vars_list = repl.list_variables()
            if not vars_list:
                text = "Nenhuma variável no REPL."
            else:
                lines = ["Variáveis no REPL:", ""]
                for v in vars_list:
                    lines.append(f"  {v.name}: {v.type_name} ({v.size_human})")
                    lines.append(f"    Preview: {v.preview[:100]}...")
                    lines.append("")
                text = "\n".join(lines)
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_var_info":
            info = repl.get_variable_info(arguments["name"])
            if not info:
                text = f"Variável '{arguments['name']}' não encontrada."
            else:
                text = f"""Variável: {info.name}
Tipo: {info.type_name}
Tamanho: {info.size_human} ({info.size_bytes} bytes)
Criada em: {info.created_at.isoformat()}
Último acesso: {info.last_accessed.isoformat()}

Preview:
{info.preview}"""
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_clear":
            if arguments.get("all"):
                count = repl.clear_all()
                text = f"Todas as {count} variáveis foram removidas."
            elif "name" in arguments:
                if repl.clear_variable(arguments["name"]):
                    text = f"Variável '{arguments['name']}' removida."
                else:
                    text = f"Variável '{arguments['name']}' não encontrada."
            else:
                text = "Especifique 'name' ou 'all=true'."
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_memory":
            mem = repl.get_memory_usage()
            text = f"""Uso de Memória do REPL:
Total: {mem['total_human']}
Variáveis: {mem['variable_count']}
Limite: {mem['max_allowed_mb']} MB
Uso: {mem['usage_percent']:.1f}%"""
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_load_s3":
            s3 = get_s3_client()
            if not s3.is_configured():
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Minio não configurado. Configure MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY."}
                    ],
                    "isError": True
                }

            bucket = arguments.get("bucket", "claude-code")
            key = arguments["key"]
            var_name = arguments["name"]
            data_type = arguments.get("data_type", "text")

            try:
                info = s3.get_object_info(bucket, key)
                if not info:
                    return {
                        "content": [
                            {"type": "text", "text": f"Erro: Objeto não encontrado: {bucket}/{key}"}
                        ],
                        "isError": True
                    }

                data = s3.get_object_text(bucket, key)
                result = repl.load_data(name=var_name, data=data, data_type=data_type)

                text = f"""✅ Carregado do Minio:
Bucket: {bucket}
Objeto: {key}
Tamanho: {info['size_human']}
Variável: {var_name} (tipo: {data_type})

{format_execution_result(result)}"""
                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao carregar do Minio: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_list_buckets":
            s3 = get_s3_client()
            if not s3.is_configured():
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Minio não configurado."}
                    ],
                    "isError": True
                }

            try:
                buckets = s3.list_buckets()
                if not buckets:
                    text = "Nenhum bucket encontrado."
                else:
                    text = "Buckets disponíveis:\n" + "\n".join(f"  - {b}" for b in buckets)
                return {"content": [{"type": "text", "text": text}]}
            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao listar buckets: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_list_s3":
            s3 = get_s3_client()
            if not s3.is_configured():
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Minio não configurado."}
                    ],
                    "isError": True
                }

            bucket = arguments.get("bucket", "claude-code")
            prefix = arguments.get("prefix", "")

            try:
                objects = s3.list_objects(bucket, prefix)
                if not objects:
                    text = f"Nenhum objeto encontrado em {bucket}/{prefix}"
                else:
                    lines = [f"Objetos em {bucket}/{prefix}:", ""]
                    for obj in objects[:50]:
                        lines.append(f"  {obj['name']} ({obj['size_human']})")
                    if len(objects) > 50:
                        lines.append(f"  ... e mais {len(objects) - 50} objetos")
                    text = "\n".join(lines)
                return {"content": [{"type": "text", "text": text}]}
            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao listar objetos: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_upload_s3":
            s3 = get_s3_client()
            if not s3.is_configured():
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Minio não configurado."}
                    ],
                    "isError": True
                }

            bucket = arguments.get("bucket", "claude-code")
            key = arguments["key"]
            data = arguments["data"]

            try:
                result = s3.put_object_text(bucket, key, data)
                text = f"""✅ Upload concluído:
Bucket: {result['bucket']}
Objeto: {result['key']}
Tamanho: {result['size_human']}"""
                return {"content": [{"type": "text", "text": text}]}
            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao fazer upload: {e}"}
                    ],
                    "isError": True
                }

        else:
            return {
                "content": [
                    {"type": "text", "text": f"Tool desconhecida: {name}"}
                ],
                "isError": True
            }

    except Exception as e:
        logger.exception(f"Erro ao executar tool {name}")
        return {
            "content": [
                {"type": "text", "text": f"Erro: {e}"}
            ],
            "isError": True
        }


def format_execution_result(result: ExecutionResult) -> str:
    """Formata resultado de execução"""
    parts = []

    if result.stdout:
        parts.append(f"=== OUTPUT ===\n{result.stdout}")

    if result.stderr:
        parts.append(f"=== ERRORS ===\n{result.stderr}")

    if result.variables_changed:
        parts.append(f"=== VARIÁVEIS ALTERADAS ===\n{', '.join(result.variables_changed)}")

    parts.append(f"\n[Execução: {result.execution_time_ms:.1f}ms | Status: {'OK' if result.success else 'ERRO'}]")

    return "\n".join(parts) if parts else "Execução concluída sem output."


# =============================================================================
# HTTP Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    mem = repl.get_memory_usage()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory": mem,
        "version": "0.1.0"
    }


@app.get("/sse")
async def sse_endpoint(request: Request, _: bool = Depends(verify_api_key)):
    """
    SSE endpoint para MCP.
    O cliente se conecta aqui para receber eventos do servidor.
    """
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    sse_sessions[session_id] = queue

    logger.info(f"Nova sessão SSE: {session_id}")

    async def event_generator():
        try:
            # Envia o session_id para o cliente usar no POST
            yield f"event: endpoint\ndata: /message?session_id={session_id}\n\n"

            while True:
                try:
                    # Aguarda mensagens na fila (com timeout para manter conexão viva)
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Envia ping para manter conexão
                    yield f": ping\n\n"
                except asyncio.CancelledError:
                    break
        finally:
            sse_sessions.pop(session_id, None)
            logger.info(f"Sessão SSE encerrada: {session_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id
        }
    )


@app.post("/message")
async def message_endpoint(
    request: Request,
    session_id: str = None,
    _: bool = Depends(verify_api_key)
):
    """
    Endpoint para enviar mensagens MCP.
    Se session_id for fornecido, resposta vai via SSE.
    Caso contrário, resposta direta no POST.
    """
    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)

        response = handle_mcp_request(mcp_request)

        if response is None:
            # Notificação, não precisa responder
            return Response(status_code=202)

        response_dict = response.model_dump(exclude_none=True)

        # Se tem sessão SSE, envia por lá
        if session_id and session_id in sse_sessions:
            await sse_sessions[session_id].put(response_dict)
            return Response(status_code=202)

        # Senão, responde diretamente
        return JSONResponse(response_dict)

    except Exception as e:
        logger.exception("Erro ao processar mensagem")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.post("/mcp")
async def mcp_direct_endpoint(
    request: Request,
    _: bool = Depends(verify_api_key)
):
    """
    Endpoint direto para MCP (sem SSE).
    Útil para clientes que preferem request/response simples.
    """
    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)
        response = handle_mcp_request(mcp_request)

        if response is None:
            return Response(status_code=202)

        return JSONResponse(response.model_dump(exclude_none=True))

    except Exception as e:
        logger.exception("Erro ao processar MCP request")
        return JSONResponse(
            {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}},
            status_code=500
        )


# =============================================================================
# Main
# =============================================================================

def main():
    """Entry point"""
    host = os.getenv("RLM_HOST", "0.0.0.0")
    port = int(os.getenv("RLM_PORT", "8765"))

    logger.info(f"Iniciando RLM MCP HTTP Server em {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
