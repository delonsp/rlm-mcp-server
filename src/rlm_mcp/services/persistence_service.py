"""
Service layer for persistence and indexing operations.

Extracts the repeated pattern of auto-persisting and auto-indexing
variables after loading data.
"""

import logging
from typing import TYPE_CHECKING

from ..persistence import get_persistence
from ..indexer import auto_index_if_large, set_index

if TYPE_CHECKING:
    from ..repl import PythonREPL

logger = logging.getLogger(__name__)


def persist_and_index(var_name: str, value, repl: "PythonREPL") -> tuple[str, str, str]:
    """Persiste variável e indexa se grande.

    Args:
        var_name: Nome da variável a persistir
        value: Valor a persistir (obtido de repl.variables.get(var_name))
        repl: Instância do REPL (não usado diretamente, mas mantido para compatibilidade)

    Returns:
        tuple: (persist_msg, index_msg, error_msg)
            - persist_msg: Mensagem de sucesso de persistência (ex: "💾 Persistido")
            - index_msg: Mensagem de sucesso de indexação (ex: "📑 Indexado (50 termos)")
            - error_msg: Mensagem de erro se houver (ex: "\n⚠️ Erro de persistência: ...")
    """
    persist_msg = ""
    index_msg = ""
    error_msg = ""

    try:
        persistence = get_persistence()
        if value is not None:
            persistence.save_variable(var_name, value)
            persist_msg = "💾 Persistido"

            # Indexar se for texto grande (>= 100k caracteres)
            if isinstance(value, str) and len(value) >= 100000:
                idx = auto_index_if_large(value, var_name)
                if idx:
                    set_index(var_name, idx)
                    persistence.save_index(var_name, idx.to_dict())
                    index_msg = f"📑 Indexado ({idx.get_stats()['indexed_terms']} termos)"
    except Exception as e:
        logger.warning(f"Erro ao persistir/indexar {var_name}: {e}")
        error_msg = f"\n⚠️ Erro de persistência: {e}"

    return persist_msg, index_msg, error_msg
