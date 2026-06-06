"""
Builder único do texto combinado de coleção + mapping linha→(var, linha_original).

Extraído de 3 cópias inline em http_server.py (lifespan auto-rebuild,
rlm_collection_add, rlm_collection_rebuild) que divergiam só em mensagens — e
todas carregavam o mesmo off-by-1 POR HEADER no mapping (P0 do line-mapping,
medido live: +16 na 8ª var da coleção; o texto citado era certo, a linha não).

Contrato do mapping:
- chave   = linha 1-indexed no combined_text FINAL (pós `"\n".join`);
- valor   = (var_name, linha 1-indexed dentro da var original);
- headers/separadores NÃO têm entrada (consumers fazem .get() e pulam).
"""
import logging

logger = logging.getLogger("rlm-collection-builder")

SEPARATOR = "=" * 60


def collection_header(var_name: str) -> str:
    """Header real entre vars no combinado. NÃO mudar o formato sem revisar o
    sentinel BM25 (`_COLLECTION_SENTINEL_RE` no indexer) que casa estas linhas
    para impedir segmento de cruzar fronteira de var."""
    return f"\n{SEPARATOR}\n=== VARIÁVEL: {var_name} ===\n{SEPARATOR}\n"


def build_collection_combined(
    var_names: list[str],
    variables: dict,
) -> tuple[str, dict[int, tuple[str, int]], int]:
    """Monta o texto combinado da coleção e o mapping de linhas.

    Args:
        var_names: nomes na ordem da coleção (persistence.get_collection_vars) —
            ordem preservada.
        variables: dict de variáveis do REPL (repl.variables). Vars ausentes ou
            não-str são puladas (comportamento histórico dos 3 builders).

    Returns:
        (combined_text, var_mapping, vars_included). Coleção vazia/sem texto →
        ("", {}, 0). vars_included = nº de vars str efetivamente incluídas
        (usado pelo rebuild p/ estatística "incluídas/total").

    A linha de início de cada parte é DERIVADA das próprias partes: `"\n".join`
    insere um '\n' entre partes adjacentes (separa, não funde), logo cada parte
    ocupa exatamente `parte.count('\n') + 1` linhas no combinado. (O bug
    histórico: somar `header.count('\n')` = 4 em vez das 5 linhas reais.)
    """
    parts: list[str] = []
    var_mapping: dict[int, tuple[str, int]] = {}
    first_lines: list[tuple[str, int]] = []  # (var, linha de início) p/ sanidade
    vars_included = 0
    next_line = 1  # linha 1-indexed onde a PRÓXIMA parte começa no combinado

    for var_name in var_names:
        value = variables.get(var_name)
        if not isinstance(value, str):
            continue

        header = collection_header(var_name)
        parts.append(header)
        next_line += header.count("\n") + 1

        first_lines.append((var_name, next_line))
        n_lines = value.count("\n") + 1  # == len(value.split('\n'))
        for i in range(n_lines):
            var_mapping[next_line + i] = (var_name, i + 1)
        parts.append(value)
        next_line += n_lines
        vars_included += 1

    combined_text = "\n".join(parts)

    # Cheque de sanidade fail-loud (barato: 1 split + O(vars) comparações):
    # a 1ª linha mapeada de cada var TEM que coincidir com a 1ª linha da var
    # no combinado. Citação de linha errada em repertório clínico é o pior
    # modo de falha silenciosa deste módulo — preferimos quebrar o rebuild.
    if first_lines:
        combined_lines = combined_text.split("\n")
        for var_name, start in first_lines:
            expected = variables[var_name].split("\n", 1)[0]
            actual = combined_lines[start - 1] if start - 1 < len(combined_lines) else None
            if actual != expected:
                raise AssertionError(
                    f"line-mapping inconsistente p/ '{var_name}': combined "
                    f"L{start} = {actual!r}, esperado 1ª linha da var = {expected!r}"
                )

    return combined_text, var_mapping, vars_included
