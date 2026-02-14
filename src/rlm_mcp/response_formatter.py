"""
Response formatter for MCP tool responses.

Supports three verbosity levels:
- compact: Minimal handle-style responses (saves ~60-80% tokens)
- normal: Current behavior (default compatibility)
- verbose: Extra detail for debugging
"""

import os
from enum import Enum
from typing import Optional


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
                            verbosity: Optional[Verbosity] = None) -> str:
    """Format rlm_search_index response."""
    v = verbosity or get_verbosity()

    if not results:
        if require_all:
            return f"Nenhuma linha com TODOS os termos: {', '.join(terms)}"
        return f"Nenhum resultado para: {', '.join(terms)}"

    if v == Verbosity.COMPACT:
        return _format_search_compact(results, terms, require_all, index_stats)

    return _format_search_normal(results, terms, require_all, total_results, offset, limit, index_stats)


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


def _format_search_normal(results, terms, require_all, total_results, offset, limit, index_stats) -> str:
    """Normal search format."""
    if require_all:
        paginated = sorted(results.items())[offset:offset + limit]
        lines = [f"Linhas com todos os termos ({total_results} encontradas, mostrando {offset + 1}-{offset + len(paginated)}):", ""]
        for linha, found_terms in paginated:
            lines.append(f"  Linha {linha}: {found_terms}")
        text = "\n".join(lines)
    else:
        lines = ["Resultados da busca:", ""]
        for term, matches in results.items():
            total_matches = len(matches)
            paginated_matches = matches[offset:offset + limit]
            showing = f"{offset + 1}-{offset + len(paginated_matches)}" if paginated_matches else "0"
            lines.append(f"📌 '{term}' ({total_matches} ocorrências, mostrando {showing}):")
            for m in paginated_matches:
                lines.append(f"    Linha {m['linha']}: {m['contexto'][:80]}...")
            lines.append("")
        text = "\n".join(lines)

    if index_stats:
        text += f"\n\n📊 Índice: {index_stats['indexed_terms']} termos, {index_stats['total_occurrences']} ocorrências totais"

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
        return "[persist: " + " | ".join(parts) + "]"

    lines = ["📦 Estatísticas de Persistência", ""]
    lines.append(f"Variáveis salvas: {stats.get('variables_count', 0)}")
    lines.append(f"Tamanho total: {stats.get('variables_total_size', 0):,} bytes")
    lines.append(f"Índices salvos: {stats.get('indices_count', 0)}")
    lines.append(f"Termos indexados: {stats.get('total_indexed_terms', 0):,}")
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
