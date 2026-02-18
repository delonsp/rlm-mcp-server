"""
Response formatter for MCP tool responses.

Supports three verbosity levels:
- compact: Minimal handle-style responses (saves ~60-80% tokens)
- normal: Current behavior (default compatibility)
- verbose: Extra detail for debugging
"""

import os
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .task_manager import TaskInfo


class Verbosity(str, Enum):
    COMPACT = "compact"
    NORMAL = "normal"
    VERBOSE = "verbose"


def get_verbosity() -> Verbosity:
    """Get current verbosity from environment."""
    val = os.getenv("RLM_RESPONSE_VERBOSITY", "compact").lower()
    try:
        return Verbosity(val)
    except ValueError:
        return Verbosity.COMPACT


def _count_lines(text: str) -> int:
    """Count lines in text."""
    return text.count('\n') + 1 if text else 0


def _type_summary(value) -> str:
    """Short type summary for a value."""
    if isinstance(value, str):
        lines = _count_lines(value)
        return f"str | {lines} lines"
    elif isinstance(value, list):
        return f"list | {len(value)} items"
    elif isinstance(value, dict):
        return f"dict | {len(value)} keys"
    else:
        return type(value).__name__


def _truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text with overflow indicator."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + f" [+{len(text) - max_chars}c]"


# =============================================================================
# Execution result formatting
# =============================================================================

def format_execution_result(result, verbosity: Optional[Verbosity] = None) -> str:
    """Format ExecutionResult.

    Args:
        result: ExecutionResult from repl.execute() or repl.load_data()
        verbosity: Override verbosity level
    """
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return _format_exec_compact(result)
    elif v == Verbosity.VERBOSE:
        return _format_exec_verbose(result)
    return _format_exec_normal(result)


def _format_exec_compact(result) -> str:
    """Compact execution format."""
    status = "OK" if result.success else "ERR"
    parts = []

    stdout = result.stdout.strip() if result.stdout else ""
    stderr = result.stderr.strip() if result.stderr else ""

    if stdout:
        parts.append(_truncate(stdout, 500))

    if stderr:
        parts.append(f"ERR: {_truncate(stderr, 200)}")

    vars_info = ""
    if result.variables_changed:
        vars_info = f" | vars:{','.join(result.variables_changed)}"

    time_str = f"{result.execution_time_ms:.0f}ms" if result.execution_time_ms > 0 else ""

    meta_parts = [f"exec:{status}"]
    if time_str:
        meta_parts.append(time_str)
    if stdout:
        meta_parts.append(f"out:{len(stdout)}c")

    meta = " | ".join(meta_parts) + vars_info
    output = "\n".join(parts) if parts else ""

    if output:
        return f"{output}\n[{meta}]"
    return f"[{meta}]"


def _format_exec_normal(result) -> str:
    """Normal execution format (original behavior)."""
    parts = []
    if result.stdout:
        parts.append(f"=== OUTPUT ===\n{result.stdout}")
    if result.stderr:
        parts.append(f"=== ERRORS ===\n{result.stderr}")
    if result.variables_changed:
        parts.append(f"=== VARIÁVEIS ALTERADAS ===\n{', '.join(result.variables_changed)}")
    parts.append(f"\n[Execução: {result.execution_time_ms:.1f}ms | Status: {'OK' if result.success else 'ERRO'}]")
    return "\n".join(parts) if parts else "Execução concluída sem output."


def _format_exec_verbose(result) -> str:
    """Verbose execution format."""
    text = _format_exec_normal(result)
    text += f"\n\n[Debug: stdout_len={len(result.stdout or '')}, stderr_len={len(result.stderr or '')}, "
    text += f"vars_changed={result.variables_changed}]"
    return text


# =============================================================================
# Load response formatting (rlm_load_s3, rlm_load_data, rlm_load_file)
# =============================================================================

def format_load_response(
    source: str,
    var_name: str,
    size_human: str,
    data_type: str,
    exec_result,
    persist_msg: str = "",
    index_msg: str = "",
    persist_error: str = "",
    extra_info: Optional[dict] = None,
    verbosity: Optional[Verbosity] = None,
) -> str:
    """Format a load tool response.

    Args:
        source: Source description (e.g., "s3:bucket/key", "file:/data/x.txt")
        var_name: Variable name
        size_human: Human-readable size
        data_type: Type of data loaded
        exec_result: ExecutionResult from load_data()
        persist_msg: Persistence message
        index_msg: Index message
        persist_error: Persistence error if any
        extra_info: Additional info dict (e.g., pdf method, pages)
        verbosity: Override verbosity level
    """
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return _format_load_compact(source, var_name, size_human, data_type,
                                     exec_result, persist_msg, index_msg, persist_error, extra_info)

    return _format_load_normal(source, var_name, size_human, data_type,
                                exec_result, persist_msg, index_msg, persist_error, extra_info)


def _format_load_compact(source, var_name, size_human, data_type,
                          exec_result, persist_msg, index_msg, persist_error, extra_info) -> str:
    """Compact load format: [var:name | size | type | details | persisted | indexed]"""
    parts = [f"var:{var_name}", size_human, data_type]

    if extra_info:
        if "pages" in extra_info:
            parts.append(f"pdf:{extra_info['pages']}p")
        if "method" in extra_info:
            parts.append(extra_info["method"])

    value = exec_result  # May use for line count later
    if persist_msg:
        parts.append("persisted")
    if index_msg:
        # Extract term count from "Indexado (50 termos)"
        parts.append(index_msg.replace("📑 ", "").replace("Indexado ", "indexed:").replace("(", "").replace(")", "").replace(" termos", "t").strip())

    handle = "[" + " | ".join(parts) + "]"

    if persist_error:
        handle += persist_error

    return handle


def _format_load_normal(source, var_name, size_human, data_type,
                         exec_result, persist_msg, index_msg, persist_error, extra_info) -> str:
    """Normal load format (original verbose behavior)."""
    output = format_execution_result(exec_result, Verbosity.NORMAL)
    extras = f"\n\n{persist_msg} {index_msg}".strip() if (persist_msg or index_msg) else ""
    if persist_error:
        extras += persist_error
    if extras:
        output += extras
    return output


# =============================================================================
# S3 load with full info
# =============================================================================

def format_s3_load_response(
    bucket: str,
    key: str,
    var_name: str,
    size_human: str,
    data_type: str,
    exec_result,
    persist_msg: str = "",
    index_msg: str = "",
    persist_error: str = "",
    pdf_info: Optional[dict] = None,
    verbosity: Optional[Verbosity] = None,
) -> str:
    """Format rlm_load_s3 response."""
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return format_load_response(
            f"s3:{bucket}/{key}", var_name, size_human, data_type,
            exec_result, persist_msg, index_msg, persist_error, pdf_info, v
        )

    # Normal/verbose: original format
    extras = f"\n{persist_msg} {index_msg}".strip() if (persist_msg or index_msg) else ""
    show_errors = os.getenv("RLM_SHOW_PERSISTENCE_ERRORS", "true").lower() in ("true", "1", "yes")
    if show_errors and persist_error:
        extras += persist_error

    if pdf_info:
        text = f"""✅ PDF extraído do Minio:
Bucket: {bucket}
Objeto: {key}
Tamanho original: {size_human}
Método: {pdf_info.get('method', 'auto')}
Páginas: {pdf_info.get('pages', '?')}
Caracteres extraídos: {pdf_info.get('chars', 0):,}
Variável: {var_name}{extras}

{format_execution_result(exec_result, Verbosity.NORMAL)}"""
    else:
        text = f"""✅ Carregado do Minio:
Bucket: {bucket}
Objeto: {key}
Tamanho: {size_human}
Variável: {var_name} (tipo: {data_type}){extras}

{format_execution_result(exec_result, Verbosity.NORMAL)}"""
    return text


# =============================================================================
# List vars formatting
# =============================================================================

def format_list_vars(vars_list, total: int, offset: int, limit: int,
                     verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_list_vars response."""
    v = verbosity or get_verbosity()

    if not vars_list:
        return "Nenhuma variável no REPL."

    paginated = vars_list[offset:offset + limit]
    start_idx = offset + 1 if paginated else 0
    end_idx = offset + len(paginated)

    if v == Verbosity.COMPACT:
        items = []
        for vi in paginated:
            pin = "📌" if getattr(vi, 'pinned', False) else ""
            items.append(f"{pin}{vi.name}:{vi.type_name}:{vi.size_human}")
        return f"{total} vars | {', '.join(items)}"

    # Normal/verbose
    lines = [f"Variáveis no REPL ({total} total, mostrando {start_idx}-{end_idx}):", ""]
    for vi in paginated:
        pin_str = " 📌" if getattr(vi, 'pinned', False) else ""
        lines.append(f"  {vi.name}: {vi.type_name} ({vi.size_human}){pin_str}")
        lines.append(f"    Preview: {vi.preview[:100]}...")
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# Variable info formatting
# =============================================================================

def format_var_info(info, verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_var_info response."""
    v = verbosity or get_verbosity()

    if not info:
        return "Variável não encontrada."

    if v == Verbosity.COMPACT:
        parts = [f"var:{info.name}", info.type_name, info.size_human]
        if getattr(info, 'pinned', False):
            parts.append("📌pinned")
        if getattr(info, 'access_count', 0) > 0:
            parts.append(f"access:{info.access_count}")
        return "[" + " | ".join(parts) + "]"

    # Normal/verbose
    text = f"""Variável: {info.name}
Tipo: {info.type_name}
Tamanho: {info.size_human} ({info.size_bytes} bytes)
Criada em: {info.created_at.isoformat()}
Último acesso: {info.last_accessed.isoformat()}"""

    if getattr(info, 'pinned', False):
        text += "\nPin: 📌 Pinned (protegida do GC)"
    if getattr(info, 'access_count', 0) > 0:
        text += f"\nAcessos: {info.access_count}"
    if getattr(info, 'source', None) and info.source != "unknown":
        text += f"\nOrigem: {info.source}"

    text += f"\n\nPreview:\n{info.preview}"
    return text


# =============================================================================
# Search formatting
# =============================================================================

def format_search_response(results, terms, require_all: bool, total_results: int,
                            offset: int, limit: int, index_stats: Optional[dict] = None,
                            verbosity: Optional[Verbosity] = None,
                            max_results: int = 30, total_available: int = 0) -> str:
    """Format rlm_search_index response."""
    v = verbosity or get_verbosity()

    if not results:
        if require_all:
            return f"Nenhuma linha com TODOS os termos: {', '.join(terms)}"
        return f"Nenhum resultado para: {', '.join(terms)}"

    if v == Verbosity.COMPACT:
        return _format_search_compact(results, terms, require_all, index_stats)

    return _format_search_normal(results, terms, require_all, total_results, offset, limit, index_stats,
                                  max_results=max_results, total_available=total_available)


def _format_search_compact(results, terms, require_all, index_stats) -> str:
    """Compact search results."""
    if require_all:
        count = len(results)
        return f"AND({','.join(terms)}):{count} lines"

    parts = []
    for term, matches in results.items():
        parts.append(f'"{term}":{len(matches)}hits')

    text = " | ".join(parts)
    if index_stats:
        text += f" | idx:{index_stats.get('indexed_terms', 0)}t"
    return text


def _format_search_normal(results, terms, require_all, total_results, offset, limit, index_stats,
                           max_results: int = 30, total_available: int = 0) -> str:
    """Normal search format."""
    # Count shown results
    if require_all:
        shown_count = len(results)
    else:
        shown_count = sum(len(v) for v in results.values())

    # Summary header
    terms_str = ",".join(f'"{t}"' for t in terms)
    avail = total_available if total_available else shown_count
    header = f'[search: {terms_str} | {shown_count} shown / {avail} total | next: max_results={max_results},offset={offset}]'

    if require_all:
        paginated = sorted(results.items())[offset:offset + limit]
        lines = [header, ""]
        for linha, found_terms in paginated:
            lines.append(f"  Linha {linha}: {found_terms}")
        text = "\n".join(lines)
    else:
        lines = [header, ""]
        for term, matches in results.items():
            total_matches = len(matches)
            lines.append(f"'{term}' ({total_matches}):")
            for m in matches:
                lines.append(f"  L{m['linha']}: {m['contexto'][:60]}")
            lines.append("")
        text = "\n".join(lines)

    return text


# =============================================================================
# Memory formatting
# =============================================================================

def format_memory(mem: dict, verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_memory response."""
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return f"[mem: {mem['total_human']} | {mem['variable_count']} vars | {mem['usage_percent']:.0f}% of {mem['max_allowed_mb']}MB]"

    return f"""Uso de Memória do REPL:
Total: {mem['total_human']}
Variáveis: {mem['variable_count']}
Limite: {mem['max_allowed_mb']} MB
Uso: {mem['usage_percent']:.1f}%"""


# =============================================================================
# Pin var formatting
# =============================================================================

def format_pin_response(name: str, pinned: bool, verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_pin_var response."""
    v = verbosity or get_verbosity()
    status = "pinned" if pinned else "unpinned"

    if v == Verbosity.COMPACT:
        return f"[pin:{name} | {status}]"

    if pinned:
        return f"📌 Variável '{name}' pinned (protegida do garbage collector)"
    return f"🔓 Variável '{name}' unpinned (sujeita ao garbage collector)"


# =============================================================================
# PDF process formatting
# =============================================================================

def format_process_pdf(bucket, key, output_key, info, pdf_result, upload_result,
                        verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_process_pdf response."""
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return (f"[pdf:{key} | {pdf_result.method} | {pdf_result.pages}p | "
                f"{len(pdf_result.text):,}c | saved:{output_key}]")

    return f"""✅ PDF processado com sucesso!

📄 Origem:
  Bucket: {bucket}
  Arquivo: {key}
  Tamanho: {info['size_human']}

📝 Extração:
  Método: {pdf_result.method}
  Páginas: {pdf_result.pages}
  Caracteres: {len(pdf_result.text):,}

💾 Texto salvo:
  Bucket: {bucket}
  Arquivo: {output_key}
  Tamanho: {upload_result['size_human']}

Próximo passo: rlm_load_s3(key="{output_key}", name="texto", data_type="text")"""


# =============================================================================
# Upload URL formatting
# =============================================================================

def format_upload_url(url, result, verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_upload_url response."""
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return f"[upload:OK | {result['key']} | {result['size_human']}]"

    return f"""✅ Upload concluído:
URL: {url}
Bucket: {result['bucket']}
Objeto: {result['key']}
Tamanho: {result['size_human']}"""


# =============================================================================
# Save to S3 formatting
# =============================================================================

def format_save_to_s3(var_name, value_type, fmt, result, key,
                       verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_save_to_s3 response."""
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return f"[saved:{var_name} | {result['key']} | {result['size_human']} | {fmt}]"

    return f"""✅ Variável salva no S3:
Variável: {var_name}
Tipo original: {value_type}
Formato: {fmt}

Destino:
  Bucket: {result['bucket']}
  Key: {result['key']}
  Tamanho: {result['size_human']}

Para carregar novamente: rlm_load_s3(key="{key}", name="{var_name}", data_type="{'json' if fmt == 'json' else 'text'}")"""


# =============================================================================
# Persistence stats formatting
# =============================================================================

def format_persistence_stats(stats, saved_vars, verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_persistence_stats response."""
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        parts = [
            f"db:{stats.get('variables_count', 0)} vars",
            f"{stats.get('indices_count', 0)} idx",
            f"{stats.get('total_indexed_terms', 0)} terms",
        ]
        emb_vars = stats.get('embedding_vars', 0)
        if emb_vars:
            parts.append(f"emb:{emb_vars}vars/{stats.get('embedding_chunks', 0)}chunks")
        return "[persist: " + " | ".join(parts) + "]"

    lines = ["📦 Estatísticas de Persistência", ""]
    lines.append(f"Variáveis salvas: {stats.get('variables_count', 0)}")
    lines.append(f"Tamanho total: {stats.get('variables_total_size', 0):,} bytes")
    lines.append(f"Índices salvos: {stats.get('indices_count', 0)}")
    lines.append(f"Termos indexados: {stats.get('total_indexed_terms', 0):,}")
    emb_vars = stats.get('embedding_vars', 0)
    emb_chunks = stats.get('embedding_chunks', 0)
    if emb_vars:
        lines.append(f"Embeddings: {emb_vars} variáveis, {emb_chunks} chunks")
    lines.append(f"Arquivo DB: {stats.get('db_path', 'N/A')}")
    lines.append(f"Tamanho DB: {stats.get('db_file_size', 0):,} bytes")

    if saved_vars:
        lines.append("")
        lines.append("Variáveis persistidas:")
        for sv in saved_vars:
            lines.append(f"  - {sv['name']} ({sv['type']}, {sv['size_bytes']:,} bytes)")
            lines.append(f"    Atualizado: {sv['updated_at']}")

    return "\n".join(lines)


# =============================================================================
# Collection formatting
# =============================================================================

def format_collection_list(collections, verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_collection_list response."""
    v = verbosity or get_verbosity()

    if not collections:
        return "Nenhuma coleção criada ainda."

    if v == Verbosity.COMPACT:
        items = [f"{c['name']}({c['var_count']})" for c in collections]
        return f"{len(collections)} collections: {', '.join(items)}"

    lines = ["📚 Coleções disponíveis:", ""]
    for c in collections:
        lines.append(f"  📁 {c['name']} ({c['var_count']} variáveis)")
        if c['description']:
            lines.append(f"     {c['description']}")
    return "\n".join(lines)


def format_collection_info(info, verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_collection_info response."""
    v = verbosity or get_verbosity()

    if not info:
        return "Coleção não encontrada."

    if v == Verbosity.COMPACT:
        var_items = [f"{vi['name']}:{vi['type']}" for vi in info.get('variables', [])]
        return f"[coll:{info['name']} | {info['var_count']} vars | {', '.join(var_items)}]"

    lines = [f"📁 Coleção: {info['name']}", ""]
    if info['description']:
        lines.append(f"Descrição: {info['description']}")
    lines.append(f"Criada em: {info['created_at']}")
    lines.append(f"Total: {info['var_count']} variáveis, {info['total_size']:,} bytes")
    lines.append("")
    lines.append("Variáveis:")
    for vi in info['variables']:
        lines.append(f"  - {vi['name']} ({vi['type']}, {vi['size_bytes']:,} bytes)")
    return "\n".join(lines)


# =============================================================================
# S3 list formatting
# =============================================================================

def format_list_s3(objects, bucket, prefix, total, offset, limit,
                    verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_list_s3 response."""
    v = verbosity or get_verbosity()

    if not objects:
        return f"Nenhum objeto encontrado em {bucket}/{prefix}"

    paginated = objects[offset:offset + limit]
    start_idx = offset + 1 if paginated else 0
    end_idx = offset + len(paginated)

    if v == Verbosity.COMPACT:
        items = [f"{o['name']}({o['size_human']})" for o in paginated]
        return f"{total} objects in {bucket}/{prefix}: {', '.join(items)}"

    lines = [f"Objetos em {bucket}/{prefix} ({total} total, mostrando {start_idx}-{end_idx}):", ""]
    for obj in paginated:
        lines.append(f"  {obj['name']} ({obj['size_human']})")
    return "\n".join(lines)


def format_list_buckets(buckets, verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_list_buckets response."""
    v = verbosity or get_verbosity()

    if not buckets:
        return "Nenhum bucket encontrado."

    if v == Verbosity.COMPACT:
        return f"buckets: {', '.join(buckets)}"

    return "Buckets disponíveis:\n" + "\n".join(f"  - {b}" for b in buckets)


# =============================================================================
# File load formatting
# =============================================================================

def format_file_load_pdf(path, pdf_result, exec_result, var_name,
                          verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_load_file (PDF) response."""
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return format_load_response(
            f"file:{path}", var_name,
            f"{len(pdf_result.text):,}c", "pdf",
            exec_result,
            extra_info={"pages": pdf_result.pages, "method": pdf_result.method},
            verbosity=v
        )

    return f"""✅ PDF extraído com sucesso:
Arquivo: {path}
Método: {pdf_result.method}
Páginas: {pdf_result.pages}
Caracteres: {len(pdf_result.text):,}
Variável: {var_name}

{format_execution_result(exec_result, Verbosity.NORMAL)}"""


# =============================================================================
# Task formatting (async tasks)
# =============================================================================

def format_task_submitted(task_id: str, tool_name: str, description: str,
                           verbosity: Optional[Verbosity] = None) -> str:
    """Format response when an async task is submitted."""
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return f"[task:{task_id} | {description} | use rlm_task_status(\"{task_id}\")]"

    return f"""⏳ Task assíncrona criada:
ID: {task_id}
Tool: {tool_name}
Descrição: {description}

Use rlm_task_status(task_id="{task_id}") para verificar o progresso."""


def format_task_status(task: "TaskInfo", verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_task_status response."""
    v = verbosity or get_verbosity()

    status_icons = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫",
    }
    icon = status_icons.get(task.status, "❓")

    if v == Verbosity.COMPACT:
        parts = [f"task:{task.task_id}", task.status]
        if task.status == "running":
            parts.append(f"{task.progress:.0%}")
            if task.progress_message:
                parts.append(task.progress_message)
        elif task.status == "completed" and task.result:
            # Show truncated result
            result_text = task.result.get("content", [{}])[0].get("text", "") if task.result.get("content") else ""
            if result_text:
                parts.append(_truncate(result_text, 200))
        elif task.status == "failed" and task.error:
            parts.append(task.error[:100])

        if task.completed_at and task.started_at:
            elapsed = (task.completed_at - task.started_at).total_seconds()
            parts.append(f"{elapsed:.1f}s")

        return f"[{' | '.join(parts)}]"

    # Normal/verbose
    lines = [f"{icon} Task: {task.task_id}"]
    lines.append(f"Tool: {task.tool_name}")
    lines.append(f"Descrição: {task.description}")
    lines.append(f"Status: {task.status}")

    if task.status == "running":
        pct = f"{task.progress:.0%}"
        bar_len = 20
        filled = int(task.progress * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        lines.append(f"Progresso: [{bar}] {pct}")
        if task.progress_message:
            lines.append(f"Etapa: {task.progress_message}")

    if task.started_at:
        lines.append(f"Início: {task.started_at.strftime('%H:%M:%S')}")
    if task.completed_at:
        lines.append(f"Fim: {task.completed_at.strftime('%H:%M:%S')}")
        if task.started_at:
            elapsed = (task.completed_at - task.started_at).total_seconds()
            lines.append(f"Duração: {elapsed:.1f}s")

    if task.status == "completed" and task.result:
        result_text = task.result.get("content", [{}])[0].get("text", "") if task.result.get("content") else ""
        if result_text:
            lines.append(f"\nResultado:\n{result_text}")

    if task.status == "failed" and task.error:
        lines.append(f"\nErro: {task.error}")

    return "\n".join(lines)


def format_task_list(tasks: list, verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_task_list response."""
    v = verbosity or get_verbosity()

    if not tasks:
        return "Nenhuma task encontrada."

    status_icons = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫",
    }

    if v == Verbosity.COMPACT:
        items = []
        for t in tasks:
            icon = status_icons.get(t.status, "?")
            progress = f" {t.progress:.0%}" if t.status == "running" else ""
            items.append(f"{icon}{t.task_id}:{t.tool_name}{progress}")
        return f"{len(tasks)} tasks | {', '.join(items)}"

    lines = [f"📋 Tasks ({len(tasks)}):", ""]
    for t in tasks:
        icon = status_icons.get(t.status, "?")
        line = f"  {icon} {t.task_id} | {t.tool_name} | {t.status}"
        if t.status == "running":
            line += f" ({t.progress:.0%})"
        if t.description:
            line += f" - {t.description}"
        lines.append(line)
    return "\n".join(lines)


# =============================================================================
# Batch S3 formatting
# =============================================================================

def format_batch_load_s3(
    results: list[dict],
    verbosity: Optional[Verbosity] = None,
) -> str:
    """Format rlm_batch_load_s3 response.

    Args:
        results: List of dicts with {name, key, size_human, data_type, success, error}
    """
    v = verbosity or get_verbosity()

    ok = [r for r in results if r.get("success")]
    err = [r for r in results if not r.get("success")]

    if v == Verbosity.COMPACT:
        parts = [f"batch:{len(ok)}/{len(results)} loaded"]
        if ok:
            ok_names = ",".join(r["name"] for r in ok)
            parts.append(f"ok:{ok_names}")
        if err:
            err_items = ",".join(f"{r['name']}({r.get('error', '?')[:20]})" for r in err)
            parts.append(f"err:{err_items}")
        return "[" + " | ".join(parts) + "]"

    lines = [f"Batch Load S3 ({len(ok)}/{len(results)} carregados):", ""]
    for r in ok:
        lines.append(f"  ✅ {r['name']} ← {r['key']} ({r['size_human']}, {r['data_type']})")
    for r in err:
        lines.append(f"  ❌ {r['name']} ← {r['key']}: {r.get('error', 'erro desconhecido')}")
    return "\n".join(lines)


def format_batch_upload_s3(
    results: list[dict],
    verbosity: Optional[Verbosity] = None,
) -> str:
    """Format rlm_batch_upload_s3 response.

    Args:
        results: List of dicts with {var_name, key, size_human, format, success, error}
    """
    v = verbosity or get_verbosity()

    ok = [r for r in results if r.get("success")]
    err = [r for r in results if not r.get("success")]

    if v == Verbosity.COMPACT:
        parts = [f"batch:{len(ok)}/{len(results)} uploaded"]
        if ok:
            ok_names = ",".join(r["var_name"] for r in ok)
            parts.append(f"ok:{ok_names}")
        if err:
            err_items = ",".join(f"{r['var_name']}({r.get('error', '?')[:20]})" for r in err)
            parts.append(f"err:{err_items}")
        return "[" + " | ".join(parts) + "]"

    lines = [f"Batch Upload S3 ({len(ok)}/{len(results)} enviados):", ""]
    for r in ok:
        lines.append(f"  ✅ {r['var_name']} → {r['key']} ({r['size_human']}, {r['format']})")
    for r in err:
        lines.append(f"  ❌ {r['var_name']} → {r['key']}: {r.get('error', 'erro desconhecido')}")
    return "\n".join(lines)


# =============================================================================
# Code search formatting
# =============================================================================

def format_search_code(
    results: list[dict],
    var_name: str,
    language: str,
    query: Optional[str] = None,
    kind: Optional[str] = None,
    total_symbols: int = 0,
    verbosity: Optional[Verbosity] = None,
    limit: int = 20,
    offset: int = 0,
    max_source_lines: int = 5,
    total_matched: int = 0,
) -> str:
    """Format rlm_search_code response.

    Args:
        results: List of symbol dicts from CodeStructure.search() (already sliced)
        var_name: Variable name searched
        language: Detected language
        query: Search query used
        kind: Kind filter used
        total_symbols: Total symbols in the code structure
        verbosity: Override verbosity level
        limit: Page size used
        offset: Current offset
        max_source_lines: Max source lines per symbol
        total_matched: Total matched before pagination
    """
    v = verbosity or get_verbosity()

    matched = total_matched if total_matched else len(results)

    if not results:
        filter_desc = []
        if query:
            filter_desc.append(f'query="{query}"')
        if kind:
            filter_desc.append(f"kind={kind}")
        filt = ", ".join(filter_desc) if filter_desc else "no filter"
        return f"No symbols found in {var_name} ({filt})"

    # Summary header
    query_str = query or "*"
    header = f'[code: {var_name} | {language} | {len(results)} shown / {matched} matched | next: offset={offset + limit}]'

    if v == Verbosity.COMPACT:
        parts = [f"code:{var_name}", language, f"{len(results)}/{matched} symbols"]
        # Group by kind
        kinds: dict[str, list[str]] = {}
        for r in results:
            k = r["kind"]
            if k not in kinds:
                kinds[k] = []
            kinds[k].append(r["name"])
        for k, names in kinds.items():
            parts.append(f"{k}:{','.join(names[:10])}" + (f"+{len(names)-10}" if len(names) > 10 else ""))
        return "[" + " | ".join(parts) + "]"

    # Normal/verbose
    lines = [header, ""]

    for r in results:
        icon = {"function": "fn", "class": "cls", "method": "mtd", "import": "imp", "variable": "var"}.get(r["kind"], r["kind"])
        parent = f" ({r['parent']})" if r.get("parent") else ""
        lines.append(f"  [{icon}] {r['name']}{parent}  L{r['line_start']}-{r['line_end']}")
        lines.append(f"        {r['signature'][:80]}")
        if r.get("docstring"):
            doc = r["docstring"][:60]
            lines.append(f"        \"{doc}\"")
        if r.get("source"):
            src_lines = r["source"].split("\n")
            for sl in src_lines[:max_source_lines]:
                lines.append(f"        | {sl}")
            if len(src_lines) > max_source_lines:
                lines.append(f"        | ... (+{len(src_lines)-max_source_lines} lines)")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Hybrid/Semantic search formatting
# =============================================================================

def format_hybrid_search(
    search_result: dict,
    terms: list[str],
    var_name: str,
    offset: int = 0,
    limit: int = 20,
    verbosity: Optional[Verbosity] = None,
    max_results: int = 30,
) -> str:
    """Format hybrid/semantic search results.

    Args:
        search_result: dict from indexer.hybrid_search()
        terms: Original search terms
        var_name: Variable searched
        offset: Pagination offset
        limit: Max results
        verbosity: Override verbosity
        max_results: Global cap for results
    """
    v = verbosity or get_verbosity()
    mode = search_result.get("mode", "keyword")
    terms_str = ",".join(f'"{t}"' for t in terms)

    # Semantic mode
    if "semantic" in mode and search_result.get("semantic_results"):
        results = search_result["semantic_results"]
        stats = search_result.get("stats", {}).get("vector", {})
        header = f'[search: {terms_str} | semantic | {len(results)} shown | {var_name}]'

        if v == Verbosity.COMPACT:
            parts = [f"sem:{var_name}", f"mode:{mode}", f"{len(results)} hits"]
            for r in results[:5]:
                parts.append(f"L{r['line_start']}({r['score']:.2f})")
            return "[" + " | ".join(parts) + "]"

        lines = [header, ""]
        for r in results:
            lines.append(f"  L{r['line_start']}-{r['line_end']} (score: {r['score']:.3f})")
            lines.append(f"    {r['chunk_text'][:120]}...")
            lines.append("")
        return "\n".join(lines)

    # Hybrid mode
    if search_result.get("hybrid_results"):
        results = search_result["hybrid_results"]
        header = f'[search: {terms_str} | hybrid | {len(results)} shown | {var_name}]'

        if v == Verbosity.COMPACT:
            parts = [f"hybrid:{var_name}", f"{len(results)} hits"]
            for r in results[:5]:
                src = "+".join(r.get("sources", []))
                parts.append(f"L{r['line']}({src},{r['rrf_score']:.3f})")
            return "[" + " | ".join(parts) + "]"

        lines = [header, ""]
        for r in results:
            src = ", ".join(r.get("sources", []))
            lines.append(f"  L{r['line']} (RRF: {r['rrf_score']:.4f}, sources: {src})")
            lines.append(f"    {r['text'][:120]}...")
            lines.append("")
        return "\n".join(lines)

    # Fallback to keyword formatting
    keyword_results = search_result.get("keyword_results")
    if keyword_results:
        stats = search_result.get("stats", {}).get("keyword")
        total_results = len(keyword_results)
        return format_search_response(
            keyword_results, terms, False, total_results,
            offset, limit, index_stats=stats, verbosity=v,
            max_results=max_results,
        )

    return f"No results for: {', '.join(terms)} (mode: {mode})"


def format_task_cancel(task_id: str, success: bool,
                        verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_task_cancel response."""
    v = verbosity or get_verbosity()

    if v == Verbosity.COMPACT:
        return f"[task:{task_id} | {'cancelled' if success else 'cancel failed'}]"

    if success:
        return f"🚫 Task '{task_id}' cancelada com sucesso."
    return f"Não foi possível cancelar task '{task_id}' (não encontrada ou já finalizada)."
