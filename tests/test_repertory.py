"""
Testes do modo-repertório (rlm_mcp.repertory + handler rlm_repertorio).

A fixture reproduz VERBATIM os fenômenos medidos live no kent_repertorio
(2026-06-06): headings `#` como ruído em linhas arbitrárias, continuações de
lista (wrap), quebra de página com `(cont.)`, cross-references, glossário no
front matter, CAPS = grau 3, duplo-colon, ruído OCR (`1`→`i`, pontos internos).
"""

import re

from rlm_mcp import http_server as hs
from rlm_mcp import repertory as rep
from rlm_mcp import response_formatter as fmt
from rlm_mcp.response_formatter import Verbosity

# Fenômenos por linha (1-indexed) — manter os comentários sincronizados:
#  L4-5   glossário do front matter
#  L7     capítulo (heading sem ':')
#  L10    header de rubrica (### + ':')
#  L11    entry simples
#  L13    header ABANDONO
#  L14    crossref (pulada)
#  L16    entry com CAPS (AUR. grau 3)
#  L18    entry "mañana"
#  L20    continuação com ruído de heading (#) — anexa em L18
#  L22    página (pulada, não fecha a entry)
#  L24    header (cont.) — mantém a entry aberta
#  L26    continuação pós-página — TAMBÉM anexa em L18
#  L28    header via '## X:' (nível inconsistente)
#  L29    entry com duplo-colon (split no ÚLTIMO ':')
#  L30    entry com rótulo ALL-CAPS → vira rubrica corrente
#  L31    entry com OCR: CARD-1 → card-i; carb.-a. → carb-a
SAMPLE = """\
--- Página 1 ---
Dr. FRANCISCO XAVIER EIZAYAGA

KALI.BR: kali bromatum
POP: populus tremuloides

## PSIQUISMO
(Mentales)

### ABANDONA:
a sus propios hijos: lyc.

### ABANDONO:
(ver Desvalido).

sentimiento de: aliea., alum., arg-n., AUR., bar-c.

mañana: carb-an., carb-v., lach.

# ambr., anac, ARN, bar-c

--- Página 2 ---

## ABANDONO (cont.)

calc, sulph., ZINC.

## SOLEDAD:
comiendo: mej.: graph.
ABATIMIENTO: nat-m., SEP.
sacudida: CARD-1., carb.-a.
"""

LINES = SAMPLE.split("\n")


def _parse(text=SAMPLE, **kw):
    return rep.parse_kent_repertory(text, "kent_test", **kw)


# ---------------------------------------------------------------------------
# Parser / classificação
# ---------------------------------------------------------------------------

def test_estrutura_basica_e_hierarquia():
    idx = _parse()
    by_line = idx.by_line
    e = by_line[11]
    assert (e.chapter, e.rubric, e.text) == ("PSIQUISMO", "ABANDONA", "a sus propios hijos")
    assert e.remedies == (("lyc", 1),)
    e = by_line[16]
    assert e.rubric == "ABANDONO"
    assert ("aur", 3) in e.remedies          # CAPS → grau 3
    assert ("alum", 1) in e.remedies


def test_continuacao_anexa_inclusive_atravessando_pagina_e_cont():
    """L20 (# rem, rem) e L26 (pós '--- Página ---' + '(cont.)') anexam em L18."""
    idx = _parse()
    e = idx.by_line[18]
    assert e.extra_lines == 2
    rems = dict(e.remedies)
    assert rems["arn"] == 3 and rems["zinc"] == 3      # CAPS nas continuações
    assert "calc" in rems and "ambr" in rems
    assert idx.stats.continuations_merged == 2
    assert idx.stats.orphan_continuations == 0


def test_crossref_e_pagina_pulados_sem_alterar_line_mapping():
    idx = _parse()
    assert idx.stats.crossrefs == 1
    # '--- Página 1 ---' está ANTES do parse_start (front matter) — só a 2 conta
    assert idx.stats.pages == 1
    assert 14 not in idx.by_line               # crossref não vira rubrica


def test_duplo_colon_split_no_ultimo():
    idx = _parse()
    e = idx.by_line[29]
    assert e.text == "comiendo: mej."
    assert e.remedies == (("graph", 1),)


def test_entry_all_caps_vira_rubrica_corrente():
    idx = _parse()
    assert idx.by_line[30].rubric == "ABATIMIENTO"     # a própria linha
    assert idx.by_line[31].rubric == "ABATIMIENTO"     # e a seguinte


def test_ocr_micro_normalizacao_1_para_i_e_pontos_internos():
    idx = _parse()
    rems = dict(idx.by_line[31].remedies)
    assert "card-i" in rems and rems["card-i"] == 3    # CARD-1. CAPS
    assert "carb-a" in rems                            # carb.-a.


def test_lista_de_remedios_logo_apos_header_vira_entry_da_rubrica():
    """Padrão real (1.080 casos no corpus): '### RUBRICA:' seguido DIRETO da
    lista de remédios — são os remédios da rubrica principal, não órfãos."""
    text = """## PSIQUISMO

### ABSTRACCION MENTAL:
acon, agn., alum., AM-C.

mañana: calc.
"""
    idx = _parse(text)
    hdr = [e for e in idx.entries if e.text == ""]
    assert len(hdr) == 1
    e = hdr[0]
    assert e.rubric == "ABSTRACCION MENTAL"
    assert e.line_no == 4                       # cita a linha da LISTA
    rems = dict(e.remedies)
    assert rems["am-c"] == 3 and "acon" in rems
    assert idx.stats.orphan_continuations == 0
    # a entry seguinte continua na mesma rubrica
    assert idx.by_line[6].rubric == "ABSTRACCION MENTAL"


def test_running_head_caps_no_meio_da_lista_nao_quebra_continuacao():
    """P2: '# MIEDO' (running-head CAPS) entre uma entry e a continuação dela
    NÃO pode virar capítulo nem orfanar os remédios da quebra."""
    text = """## PSIQUISMO
### TEMOR:
a la muerte: acon., ars., bell., calc.
# MIEDO
cic., con., dig., lyc.
"""
    idx = _parse(text)
    e = idx.by_line[3]                       # a entry 'a la muerte'
    rems = dict(e.remedies)
    assert {"cic", "con", "dig", "lyc"} <= set(rems)   # continuação anexada
    assert e.extra_lines == 1
    assert idx.stats.orphan_continuations == 0
    # capítulo NÃO foi corrompido para 'MIEDO'
    assert e.chapter == "PSIQUISMO"


def test_cont_header_que_repete_capitulo_mantem_rubrica():
    """nit: '## PSIQUISMO (cont.)' (running-header do capítulo) não pode
    sobrescrever a rubrica corrente."""
    text = """## PSIQUISMO
### ANSIEDAD:
de noche: acon.
## PSIQUISMO (cont.)
matinal: nat-m.
"""
    idx = _parse(text)
    # a entry após o (cont.) continua sob ANSIEDAD, não sob 'PSIQUISMO'
    assert idx.by_line[5].rubric == "ANSIEDAD"


def test_glossario_do_front_matter():
    idx = _parse()
    assert idx.glossary["kalibr"] == "kali bromatum"
    assert idx.glossary["pop"] == "populus tremuloides"
    assert idx.stats.glossary_size == 2


def test_line_mapping_exato_em_todas_as_entries():
    """Invariante do P0: o rótulo da entry está na linha citada da var REAL."""
    idx = _parse()
    assert idx.entries, "parser não produziu entries"
    for e in idx.entries:
        src = LINES[e.line_no - 1]
        assert e.text in src, f"L{e.line_no}: '{e.text}' não está em '{src}'"


def test_front_matter_nao_vira_rubrica():
    idx = _parse()
    for e in idx.entries:
        assert e.line_no >= 7, f"front matter parseado como rubrica: L{e.line_no}"


# ---------------------------------------------------------------------------
# Canonicalização (conservadora)
# ---------------------------------------------------------------------------

def _counter(pairs):
    from collections import Counter
    c = Counter()
    for tok, n in pairs:
        c[tok] = n
    return c


def test_canonicaliza_raro_para_candidato_unico():
    c = _counter([("sulph", 50), ("sulp", 1)])
    cmap, _, corrected = rep._build_canonical_map(c, stable_min_freq=10,
                                                  min_stable_vocab=1)
    assert cmap["sulp"] == "sulph"
    assert corrected == 1


def test_nao_corrige_token_curto_a_distancia_2():
    """calc↔carb têm distância 2 — corrigir seria TROCAR de remédio."""
    c = _counter([("carb", 50), ("calk", 1)])         # calk: dist 1 de calc... mas calc não é estável aqui
    cmap, _, _ = rep._build_canonical_map(c, stable_min_freq=10, min_stable_vocab=1)
    # 'calk' len 4 → cap 1; dist(calk, carb) = 2 → NÃO corrige → descarta
    assert cmap["calk"] is None


def test_empate_na_menor_distancia_descarta():
    c = _counter([("nat-m", 50), ("nat-c", 50), ("nat-x", 1)])
    cmap, _, _ = rep._build_canonical_map(c, stable_min_freq=10, min_stable_vocab=1)
    assert cmap["nat-x"] is None                       # ambíguo → fora


def test_zona_cinza_mantida_como_esta():
    c = _counter([("sulph", 50), ("raro", 5)])
    cmap, _, _ = rep._build_canonical_map(c, stable_min_freq=10, min_stable_vocab=1)
    assert cmap["raro"] == "raro"


def test_corpus_pequeno_nao_descarta_nada():
    """Fixture/fonte parcial sem vocab estável: manter tudo (guarda)."""
    idx = _parse()
    assert idx.stats.tokens_discarded == 0


def test_glossario_protege_remedio_de_remapeamento():
    """P0: form-ac (no glossário) NUNCA pode virar ferr-ac por distância 2."""
    from collections import Counter
    c = Counter({"ferr-ac": 18, "form-ac": 3, "sulph": 50})
    # sem proteção: form-ac (raro, mesmo gênero? não — form≠ferr) já é barrado
    # pelo guard de gênero; testa a proteção do glossário com um caso de MESMO
    # gênero: agar-ph (estável) vs agar-pr (raro, glossário) — espécie diferente
    c2 = Counter({"agar-ph": 12, "agar-pr": 3, "sulph": 50})
    cmap, _, _ = rep._build_canonical_map(c2, stable_min_freq=10, min_stable_vocab=1,
                                          protected=frozenset({"agar-pr"}))
    assert cmap["agar-pr"] == "agar-pr"     # protegido: não vira agar-ph


def test_token_composto_nao_cruza_genero():
    """form-ac (ácido fórmico) ↛ ferr-ac (acetato de ferro): gêneros diferentes."""
    from collections import Counter
    c = Counter({"ferr-ac": 18, "form-ac": 3, "sulph": 50})
    cmap, _, _ = rep._build_canonical_map(c, stable_min_freq=10, min_stable_vocab=1)
    assert cmap["form-ac"] != "ferr-ac"     # gênero form ≠ ferr → não corrige


def test_hifen_perdido_reconciliado():
    """P1: phac (OCR sem hífen) funde em ph-ac quando este domina por ≥8×."""
    from collections import Counter
    c = Counter({"ph-ac": 2866, "phac": 50, "sulph": 50})
    cmap, _, _ = rep._build_canonical_map(c, stable_min_freq=10, min_stable_vocab=1)
    assert cmap["phac"] == "ph-ac"


def test_ambr_nao_funde_em_am_br():
    """ambr (ambra) e am-br (ammonium brom.) são remédios DIFERENTES — não fundir."""
    from collections import Counter
    c = Counter({"ambr": 200, "am-br": 30, "sulph": 50})
    cmap, _, _ = rep._build_canonical_map(c, stable_min_freq=10, min_stable_vocab=1)
    assert cmap["ambr"] == "ambr"           # ambr domina → não vira am-br


def test_modalidade_curta_nao_vira_remedio():
    """P1: 'peor: de noche, frio' (1/3 válido = 33%) NÃO é entry de remédio."""
    cls, _ = rep._classify("peor: de noche, al aire libre, frio")
    assert cls == "colon_prose"             # rejeitada (era 'entry' com o piso antigo)
    cls2, _ = rep._classify("agravacion: estando sentado, calor")
    assert cls2 == "colon_prose"            # 1/2 = 50% < 60%


def test_stopword_espanhol_nao_e_remedio():
    """Palavra de modalidade/anatomia (peor, mano, calor) nunca conta como remédio,
    mesmo numa cauda majoritariamente válida."""
    assert not rep._is_remedy_token("peor")
    assert not rep._is_remedy_token("mano")
    assert not rep._is_remedy_token("calor")
    assert rep._is_remedy_token("sola")     # Solanum (remédio REAL) fica de fora
    assert rep._is_remedy_token("lyc")
    # entry com remédios reais + um stopword no fim: stopword some, remédios ficam
    idx = _parse("### TEMOR:\nde la muerte: acon., ars., bell., peor\n")
    e = idx.by_line[2]
    rems = dict(e.remedies)
    assert "peor" not in rems and {"acon", "ars", "bell"} <= set(rems)


# ---------------------------------------------------------------------------
# Busca
# ---------------------------------------------------------------------------

def test_busca_and_por_caminho_completo():
    idx = _parse()
    m, total, fz = rep.search_rubrics(idx, "abandono sentimiento")
    assert total == 1 and m[0].entry.line_no == 16
    assert fz is None


def test_busca_pt_para_es():
    idx = _parse()
    m, total, _ = rep.search_rubrics(idx, "solidao")    # → soledad
    assert total >= 1
    assert any(x.entry.rubric == "SOLEDAD" for x in m)


def test_busca_ignora_acentos():
    idx = _parse()
    m, total, _ = rep.search_rubrics(idx, "mañana")
    m2, total2, _ = rep.search_rubrics(idx, "manana")
    assert total == total2 >= 1
    assert m[0].entry.line_no == m2[0].entry.line_no


def test_busca_fuzzy_fallback_anota():
    idx = _parse()
    m, total, fz = rep.search_rubrics(idx, "sentimeento")   # typo
    assert total >= 1
    assert fz and "sentimeento" in fz


def test_busca_match_no_rotulo_supera_match_no_caminho():
    """P1: rótulo com vírgula ('agua, de la') casa palavra exata (não perde p/
    quem só menciona o termo no capítulo/rubrica)."""
    text = """## AGUA
### MIEDO:
agua, al cruzar: ars., bell.
de mar, deseo de agua: calc., sulph.
"""
    idx = _parse(text)
    m, total, _ = rep.search_rubrics(idx, "agua")
    assert total == 2
    # a rubrica cujo RÓTULO começa com 'agua' (palavra exata) vem na frente
    assert m[0].entry.text.startswith("agua")


def test_busca_capitulo_nao_inunda_rubrica_especifica():
    """P1: match só no capítulo (0.5) perde p/ match no rótulo (>=2.0)."""
    text = """## TEMOR
### generico:
de algo: acon.
### especifico:
temor de la muerte: ars., bell.
"""
    idx = _parse(text)
    m, total, _ = rep.search_rubrics(idx, "temor")
    # a entry cujo RÓTULO contém 'temor' supera as que só herdam do capítulo TEMOR
    assert m[0].entry.text == "temor de la muerte"


def test_busca_query_com_separador_de_caminho():
    """P2: colar 'CAP > RUBRICA > texto' não pode zerar a busca por causa do '>'."""
    idx = _parse()
    m, total, _ = rep.search_rubrics(idx, "PSIQUISMO > ABANDONO > sentimiento")
    assert total == 1 and m[0].entry.line_no == 16


def test_busca_paginacao():
    idx = _parse()
    all_m, total, _ = rep.search_rubrics(idx, "abandono", limit=50)
    assert total >= 2
    page, t2, _ = rep.search_rubrics(idx, "abandono", limit=1, offset=1)
    assert t2 == total and len(page) == 1
    assert page[0].entry.line_no == all_m[1].entry.line_no


# ---------------------------------------------------------------------------
# Repertorização
# ---------------------------------------------------------------------------

def test_repertorizar_cobertura_domina_depois_score():
    idx = _parse()
    ents, errs, _ = rep.resolve_rubric_refs(idx, ["kent_test:L16", "L18"])
    assert not errs and len(ents) == 2
    r = rep.repertorize(idx, ents)
    # bar-c aparece nas duas rubricas (cov 2) → 1º mesmo com score 2
    assert r.rows[0][0] == "bar-c" and r.rows[0][2] == 2
    # entre cov 1, score 3 (CAPS) vem antes de score 1
    cov1 = [row for row in r.rows if row[2] == 1]
    assert cov1[0][1] == 3


def test_repertorizar_sort_score():
    idx = _parse()
    ents, _, _ = rep.resolve_rubric_refs(idx, ["L16", "L18"])
    r = rep.repertorize(idx, ents, sort="score")
    scores = [row[1] for row in r.rows]
    assert scores == sorted(scores, reverse=True)


def test_resolve_refs_texto_unico_e_ambiguo():
    idx = _parse()
    ents, errs, _ = rep.resolve_rubric_refs(idx, ["sentimiento"])
    assert not errs and ents[0].line_no == 16
    _, errs2, _ = rep.resolve_rubric_refs(idx, ["abandono"])   # casa 2 entries
    assert errs2 and "ambíguo" in errs2[0]


def test_resolve_refs_linha_inexistente():
    idx = _parse()
    _, errs, _ = rep.resolve_rubric_refs(idx, ["L999"])
    assert errs and "999" in errs[0]


def test_resolve_refs_textual_fuzzy_avisa():
    """Ref textual que só casa via fuzzy NÃO pode substituir rubrica em silêncio."""
    idx = _parse()
    ents, errs, notes = rep.resolve_rubric_refs(idx, ["sentimeento"])  # typo
    assert not errs and ents and ents[0].line_no == 16
    assert notes and "sentimeento" in notes[0]


# ---------------------------------------------------------------------------
# Cache (fingerprint auto-invalidante)
# ---------------------------------------------------------------------------

def test_cache_reusa_e_invalida_por_fingerprint():
    rep.clear_repertory_cache()
    try:
        i1, c1 = rep.get_repertory_index("kent_test", SAMPLE)
        i2, c2 = rep.get_repertory_index("kent_test", SAMPLE)
        assert i1 is i2
        assert c1 is False and c2 is True          # 1º parseia, 2º vem do cache
        i3, c3 = rep.get_repertory_index("kent_test", SAMPLE + "\nextra: lyc.")
        assert i3 is not i1 and c3 is False         # fingerprint mudou → reparse
    finally:
        rep.clear_repertory_cache()


# ---------------------------------------------------------------------------
# Handler (via call_tool) + formatter
# ---------------------------------------------------------------------------

def _load_sample():
    rep.clear_repertory_cache()
    res = hs.call_tool("rlm_load_data", {"name": "kent_repertorio", "data": SAMPLE})
    assert not res.get("isError"), res


def test_handler_var_ausente_erra_limpo():
    rep.clear_repertory_cache()
    res = hs.call_tool("rlm_repertorio", {"action": "info"})
    assert res.get("isError")
    assert "kent_repertorio" in res["content"][0]["text"]


def test_handler_fluxo_completo_buscar_e_repertorizar():
    _load_sample()
    res = hs.call_tool("rlm_repertorio",
                       {"action": "buscar_rubrica", "query": "abandono sentimiento"})
    out = res["content"][0]["text"]
    assert not res.get("isError"), out
    m = re.search(r"kent_repertorio:L(\d+)", out)
    assert m, out
    assert "AUR" in out                                  # grau 3 visível
    line_no = int(m.group(1))
    src_line = SAMPLE.split("\n")[line_no - 1]
    assert "sentimiento de" in src_line                  # citação verificada na fonte

    res2 = hs.call_tool("rlm_repertorio", {
        "action": "repertorizar",
        "rubrics": [f"kent_repertorio:L{line_no}", "kent_repertorio:L18"],
    })
    out2 = res2["content"][0]["text"]
    assert not res2.get("isError"), out2
    assert "bar-c" in out2

    res3 = hs.call_tool("rlm_repertorio", {"action": "info"})
    assert not res3.get("isError")
    rep.clear_repertory_cache()


def test_handler_acao_desconhecida():
    _load_sample()
    res = hs.call_tool("rlm_repertorio", {"action": "xyz"})
    assert res.get("isError")
    rep.clear_repertory_cache()


def test_handler_repertorizar_sem_rubrics():
    _load_sample()
    res = hs.call_tool("rlm_repertorio", {"action": "repertorizar"})
    assert res.get("isError")
    rep.clear_repertory_cache()


def test_formatter_compact_e_normal():
    idx = _parse()
    m, total, _ = rep.search_rubrics(idx, "abandono")
    compact = fmt.format_repertory_search(m, total, "abandono", "kent_test",
                                          verbosity=Verbosity.COMPACT)
    assert compact.startswith("[repertorio:kent_test")
    normal = fmt.format_repertory_search(m, total, "abandono", "kent_test",
                                         verbosity=Verbosity.NORMAL)
    assert "Rubrica" in normal and "kent_test:L" in normal

    ents, _, _ = rep.resolve_rubric_refs(idx, ["L16", "L18"])
    r = rep.repertorize(idx, ents)
    tab_c = fmt.format_repertorization(r, idx, verbosity=Verbosity.COMPACT)
    assert "repertorizacao" in tab_c and "bar-c" in tab_c
    tab_n = fmt.format_repertorization(r, idx, verbosity=Verbosity.NORMAL)
    assert "Grau" in tab_n or "Graus" in tab_n

    info_c = fmt.format_repertory_info(idx_with_fp(idx), cached=False,
                                       verbosity=Verbosity.COMPACT)
    assert "entries:" in info_c
    info_n = fmt.format_repertory_info(idx_with_fp(idx), cached=True,
                                       verbosity=Verbosity.NORMAL)
    assert "Perdas conhecidas" in info_n


def idx_with_fp(idx):
    if not idx.fingerprint:
        idx.fingerprint = "deadbeef00000000"
    return idx
