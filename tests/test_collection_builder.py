"""
Builder único do texto combinado de coleção (collection_builder.py).

Regressão do P0 de line-mapping (medido live: +16 na 8ª var): o mapping
linha-combinada → (var, linha_original) tem que bater com o texto REAL.
"""

from rlm_mcp.collection_builder import (
    build_collection_combined,
    collection_header,
    SEPARATOR,
)

VARS = {
    "var_a": "alfa um\nalfa dois\nalfa três",        # 3 linhas
    "var_b": "beta um\nbeta dois",                   # 2 linhas
    "var_c": "gama única",                           # 1 linha
    "var_d": "delta um\n\ndelta três\n",             # vazia interna + \n final (4 linhas)
}
ORDER = ["var_a", "var_b", "var_c", "var_d"]


def test_invariante_forte_toda_linha_mapeada_bate_com_o_texto_real():
    """combined[L-1] == var[orig-1] para TODA entrada do mapping (header real)."""
    combined, mapping, included = build_collection_combined(ORDER, VARS)
    lines = combined.split("\n")
    assert included == 4
    assert mapping, "mapping vazio"
    for combined_line, (var, orig) in mapping.items():
        assert lines[combined_line - 1] == VARS[var].split("\n")[orig - 1], (
            f"L{combined_line} → {var}:{orig} não bate: "
            f"{lines[combined_line - 1]!r} != {VARS[var].split(chr(10))[orig - 1]!r}"
        )


def test_cobertura_inversa_toda_linha_de_var_esta_no_mapping():
    _, mapping, _ = build_collection_combined(ORDER, VARS)
    by_var: dict = {}
    for _, (var, orig) in mapping.items():
        by_var.setdefault(var, set()).add(orig)
    for var, text in VARS.items():
        esperado = set(range(1, text.count("\n") + 2))
        assert by_var[var] == esperado, f"{var}: {by_var[var]} != {esperado}"


def test_offsets_conhecidos_canario_do_bug_historico():
    """Aritmética explícita: header ocupa 5 linhas no combinado (4 '\\n' + a
    linha extra do join). O código antigo somava 4 → começos em 5/12/18/23.
    Os corretos são 6/14/21/27."""
    _, mapping, _ = build_collection_combined(ORDER, VARS)
    assert mapping[6] == ("var_a", 1)
    assert mapping[14] == ("var_b", 1)   # 6 + 3 (var_a) + 5 (header)
    assert mapping[21] == ("var_c", 1)   # 14 + 2 + 5
    assert mapping[27] == ("var_d", 1)   # 21 + 1 + 5
    # E os começos do código antigo NÃO podem mapear linha 1 das vars erradas
    assert mapping.get(5) is None or mapping[5][1] != 1
    assert mapping.get(12, ("", 0))[0] != "var_b"


def test_preserva_ordem_das_vars():
    _, mapping, _ = build_collection_combined(ORDER, VARS)
    starts = {var: min(cl for cl, (v, _) in mapping.items() if v == var)
              for var in ORDER}
    assert starts["var_a"] < starts["var_b"] < starts["var_c"] < starts["var_d"]


def test_pula_ausentes_e_nao_string():
    variables = {"ok": "linha", "num": 123, "lista": ["x"]}
    combined, mapping, included = build_collection_combined(
        ["ok", "ausente", "num", "lista"], variables
    )
    assert included == 1
    assert {v for _, (v, _) in mapping.items()} == {"ok"}
    assert "=== VARIÁVEL: ok ===" in combined
    assert "ausente" not in combined


def test_colecao_vazia():
    assert build_collection_combined([], {}) == ("", {}, 0)
    assert build_collection_combined(["x"], {"x": 42}) == ("", {}, 0)


def test_headers_nao_tem_mapping():
    combined, mapping, _ = build_collection_combined(ORDER, VARS)
    lines = combined.split("\n")
    for i, line in enumerate(lines, start=1):
        if line == SEPARATOR or line.startswith("=== VARIÁVEL:"):
            assert i not in mapping, f"linha de header L{i} mapeada: {line!r}"


def test_header_real_tem_5_linhas_no_combinado():
    """Documenta a premissa central do fix: parte = count('\\n')+1 linhas."""
    h = collection_header("x")
    assert h.count("\n") == 4  # 4 quebras → 5 linhas internas no join
