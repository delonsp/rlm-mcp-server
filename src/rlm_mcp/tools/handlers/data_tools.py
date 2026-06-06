"""
Handlers de dados/REPL: rlm_execute, rlm_load_data, rlm_load_file,
rlm_list_vars, rlm_var_info, rlm_clear, rlm_memory, rlm_pin_var.

Corpos movidos verbatim do call_tool monolítico (http_server) — só as
referências a globals viraram acessos via ctx (ToolContext).
"""

import logging

from ... import response_formatter as fmt
from ... import code_parser
from ...pdf_parser import extract_pdf
from ...persistence import get_persistence
from ...services.persistence_service import persist_and_index
from ..context import ToolContext

logger = logging.getLogger("rlm-http")


def rlm_execute(arguments: dict, ctx: ToolContext) -> dict:
    result = ctx.repl.execute(arguments["code"])
    return {
        "content": [
            {"type": "text", "text": fmt.format_execution_result(result)}
        ]
    }


def rlm_load_data(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    var_name = arguments["name"]
    data = arguments["data"]
    data_type = arguments.get("data_type", "text")

    # "code" loads as text but also auto-parses the code structure
    actual_type = "text" if data_type == "code" else data_type

    # Set source on metadata
    result = repl.load_data(name=var_name, data=data, data_type=actual_type)
    if not result.success:
        # Sem isto, JSON inválido etc. retornava resposta de SUCESSO
        # ([var | ? | json]) com a var inexistente — erro mascarado
        # (achado do harness QA 2026-06-06).
        return {
            "content": [{"type": "text", "text": result.stderr or "Erro ao carregar dados"}],
            "isError": True
        }
    if var_name in repl.variable_metadata:
        repl.variable_metadata[var_name].source = "load_data"

    # Auto-parse code structure if data_type="code"
    if data_type == "code" and result.success:
        lang = code_parser.detect_language(var_name, data)
        if lang and code_parser.is_available():
            structure = code_parser.parse(data, lang)
            if structure:
                repl.variables[f"_code_structure_{var_name}"] = structure

    # Auto-persistência e indexação
    value = repl.variables.get(var_name)
    persist_msg, index_msg, persist_error = persist_and_index(var_name, value, repl)
    if ctx.show_persistence_errors:
        pass  # persist_error already contains the error
    else:
        persist_error = ""

    size_human = repl.variable_metadata[var_name].size_human if var_name in repl.variable_metadata else "?"
    output = fmt.format_load_response(
        source="direct", var_name=var_name, size_human=size_human,
        data_type=data_type, exec_result=result,
        persist_msg=persist_msg, index_msg=index_msg, persist_error=persist_error,
    )

    return {"content": [{"type": "text", "text": output}]}


def rlm_load_file(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    path = arguments["path"]
    data_type = arguments.get("data_type", "text")

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
        # PDF handling
        if data_type in ("pdf", "pdf_ocr"):
            method = "ocr" if data_type == "pdf_ocr" else "auto"
            pdf_result = extract_pdf(path, method=method)

            if not pdf_result.success:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao extrair PDF: {pdf_result.error}"}
                    ],
                    "isError": True
                }

            data = pdf_result.text
            var_name = arguments["name"]
            result = repl.load_data(
                name=var_name,
                data=data,
                data_type="text"
            )
            if not result.success:
                # PDF extraiu mas a var não coube/carregou — não mascarar
                return {
                    "content": [{"type": "text", "text": result.stderr or "Erro ao carregar dados do PDF"}],
                    "isError": True
                }
            if var_name in repl.variable_metadata:
                repl.variable_metadata[var_name].source = "file"

            text = fmt.format_file_load_pdf(path, pdf_result, result, var_name)
            return {"content": [{"type": "text", "text": text}]}

        # Regular file handling
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            data = f.read()

        var_name = arguments["name"]
        actual_type = "text" if data_type == "code" else data_type
        result = repl.load_data(
            name=var_name,
            data=data,
            data_type=actual_type
        )
        if var_name in repl.variable_metadata:
            repl.variable_metadata[var_name].source = f"file:{path}"

        # Auto-parse code structure if data_type="code"
        if data_type == "code" and result.success:
            lang = code_parser.detect_language(path, data)
            if lang and code_parser.is_available():
                structure = code_parser.parse(data, lang)
                if structure:
                    repl.variables[f"_code_structure_{var_name}"] = structure

        return {
            "content": [
                {"type": "text", "text": fmt.format_execution_result(result)}
            ]
        }
    except FileNotFoundError:
        return {
            "content": [
                {"type": "text", "text": f"Erro: Arquivo não encontrado: {path}"}
            ],
            "isError": True
        }


def rlm_list_vars(arguments: dict, ctx: ToolContext) -> dict:
    limit = arguments.get("limit", 50)
    offset = arguments.get("offset", 0)
    vars_list = ctx.repl.list_variables()
    text = fmt.format_list_vars(vars_list, len(vars_list), offset, limit)
    return {"content": [{"type": "text", "text": text}]}


def rlm_var_info(arguments: dict, ctx: ToolContext) -> dict:
    info = ctx.repl.get_variable_info(arguments["name"])
    if not info:
        text = f"Variável '{arguments['name']}' não encontrada."
    else:
        text = fmt.format_var_info(info)
    return {"content": [{"type": "text", "text": text}]}


def rlm_clear(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
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


def rlm_memory(arguments: dict, ctx: ToolContext) -> dict:
    mem = ctx.repl.get_memory_usage()
    text = fmt.format_memory(mem)
    # Include persistence stats
    try:
        persistence = get_persistence()
        stats = persistence.get_stats()
        saved_vars = persistence.list_variables()
        persist_text = fmt.format_persistence_stats(stats, saved_vars)
        text += "\n" + persist_text
    except Exception:
        pass
    return {"content": [{"type": "text", "text": text}]}


def rlm_pin_var(arguments: dict, ctx: ToolContext) -> dict:
    var_name = arguments["name"]
    pin = arguments.get("pin", True)

    if ctx.repl.pin_variable(var_name, pin):
        text = fmt.format_pin_response(var_name, pin)
    else:
        text = f"Variável '{var_name}' não encontrada."
        return {"content": [{"type": "text", "text": text}], "isError": True}
    return {"content": [{"type": "text", "text": text}]}
