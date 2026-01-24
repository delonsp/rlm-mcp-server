"""Base helper functions for MCP tool responses."""


def text_response(text: str) -> dict:
    """Cria resposta MCP com texto.

    Args:
        text: Texto a ser retornado na resposta.

    Returns:
        dict: Resposta MCP formatada com conteúdo de texto.
    """
    return {"content": [{"type": "text", "text": text}]}


def error_response(message: str) -> dict:
    """Cria resposta MCP de erro.

    Args:
        message: Mensagem de erro a ser retornada.

    Returns:
        dict: Resposta MCP formatada com flag isError.
    """
    return {"content": [{"type": "text", "text": message}], "isError": True}
