"""Testes da classificação de chunks "boilerplate" (bibliografia/cabeçalho) e do
rebaixamento de score na busca semântica do vector_index.

Tudo aqui é Python puro (stdlib + regex) → roda em qualquer plataforma, inclusive
no Mac do autor (ao contrário do lockdown B2, que é Linux-only).

Como ``tests/*`` é gitignored (exceto os testes de segurança do sandbox), este
arquivo NÃO é versionado — segue o padrão de test_bm25.py / test_search_output.py.
Ver plans/20260601-filter-boilerplate-chunks.md.

A heurística foi endurecida (v2) após um red-team adversarial (2026-06-01) que
expôs 23 falsos-positivos no corpus real: número-de-linha sozinho e "et al"
sozinho deixaram de ser sinal; referência exige co-sinal forte (doi / ano;vol:pág);
a regra caps-título foi removida. Os casos POSITIVE_* abaixo travam esses ganhos.
"""

import os

import pytest

from rlm_mcp import vector_index as vi
from rlm_mcp.vector_index import (
    ChunkInfo,
    VectorIndex,
    _boilerplate_penalty,
    _classify_boilerplate,
)

# ---------------------------------------------------------------------------
# BOILERPLATE real (deve dar True) — co-sinal forte de citação ou cabeçalho.
# ---------------------------------------------------------------------------

REFERENCE_BLOCK = """\
1. Warburg O. On the origin of cancer cells. Science. 1956;123:309-314.
2. Seyfried TN, Shelton LM. Cancer as a metabolic disease. Nutr Metab. 2010;7:7.
3. Hanahan D, Weinberg RA. Hallmarks of cancer: the next generation. Cell. 2011;144:646-674.
4. Pedersen PL. Tumor mitochondria and the bioenergetics. Prog Exp Tumor Res. 1978;22:190-274.
"""

REFERENCE_BLOCK_ET_AL = """\
Chinnaiyan P, et al. The metabolic effects of radiation. Cancer Res. 2012;72:5130.
Maher F, et al. Glucose transporter proteins in brain. FASEB J. 1994;8:1003-1011.
Klement RJ, et al. Calories, carbohydrates, and cancer therapy. doi:10.1016/j.crad.2019.
"""

HEADER_BLOCK = """\
--- Página 88 ---

Chapter 5

THE METABOLIC ORIGINS OF CANCER

--- Página 89 ---
"""

# Lista Vancouver com entradas QUEBRADAS em linhas (como em L10626/L11566 do
# Seyfried): o token "ano;vol:página" cai em ~metade das linhas, então o ratio
# por-linha fica < 0.6 e a v2 PERDIA. A v3 pega via contagem absoluta (>= 3 tokens).
REFERENCE_BLOCK_WRAPPED = """\
160. Smith AB, Jones CD, Garcia EF. Mitochondrial dysfunction drives
tumorigenesis through altered bioenergetics. BMC Cancer. 2005;5:102.
161. Lu J, Sharma LK, Bai Y. Implications of mitochondrial DNA mutations
in tumor progression and metastasis. Cell Res. 2009;19:802-815.
162. Carew JS, Huang P. Mitochondrial defects in cancer and their
therapeutic exploitation. Mol Cancer. 2002;1:9.
"""

# ---------------------------------------------------------------------------
# PROSA (deve dar False). Os blocos PROSE_FP_* são falsos-positivos confirmados
# da heurística v1, agora travados como regressão.
# ---------------------------------------------------------------------------

PROSE_BLOCK = """\
The mitochondria are the primary site of energy production in the cell. When their
function is impaired, the cell shifts toward fermentation even in the presence of
oxygen, a phenomenon Warburg first described in the 1920s. This metabolic
reprogramming is now considered a central feature of tumor tissue, and it has
practical consequences for how we think about therapy.
"""

# Prosa com UM marcador de página solto no meio (extração de PDF) — guarda de densidade.
PROSE_WITH_STRAY_MARKER = """\
The cell shifts toward fermentation even in the presence of oxygen, a phenomenon
that Warburg first described in his seminal experiments. This shift has profound
--- Página 42 ---
implications for how tumor tissue sustains its rapid proliferation under hypoxic
conditions, and it reframes the entire therapeutic question around metabolism.
"""

# FP v1 #1: prosa que cita autores inline ("et al" SEM ano) — EN.
PROSE_FP_ET_AL_EN = """\
Smith et al. demonstrated a clear dose-response relationship.
Jones et al. replicated the finding in a larger cohort the following year.
Critically, Garcia et al. extended these observations to elderly patients.
"""

# FP v1 #2: idem em PORTUGUÊS (corpus primário do autor: Bredesen/Seyfried).
PROSE_FP_ET_AL_PT = """\
Conforme Bredesen et al. propôs, o declínio cognitivo tem múltiplos subtipos.
Já Seyfried et al. defende a origem metabólica do processo neurodegenerativo.
Ambas as hipóteses convergem para a importância da função mitocondrial.
"""

# FP v1 #3: prosa que menciona periódico+ano+volume inline, SEM ":página".
PROSE_FP_INLINE_JOURNAL = """\
O estudo de coorte foi publicado na Lancet 2017; 389 e mudou a prática.
A meta-análise posterior, na BMJ 2020; 368, confirmou o achado principal.
Desde então, as diretrizes incorporaram a recomendação de forma unânime.
"""

# FP v1 #4: protocolo clínico numerado (estilo ReCODE) — conteúdo BUSCÁVEL.
PROSE_FP_NUMBERED_PROTOCOL = """\
1. Discontinue all anticholinergic medications at least two weeks before testing.
2. Establish a fasting insulin baseline drawn after a twelve-hour overnight fast.
3. Begin the ketogenic phase only once fasting glucose stabilizes below 90 mg/dL.
4. Reassess homocysteine and B12 at the eight-week mark before adjusting dose.
"""

# FP v1 #5: receita numerada.
PROSE_FP_RECIPE = """\
1. Preheat the oven to 200 degrees and line a tray with parchment paper.
2. Toss the cauliflower florets in olive oil, turmeric, and a pinch of sea salt.
3. Roast for 25 minutes until the edges turn golden and crisp.
4. Remove from the oven and finish with a squeeze of fresh lemon.
"""

# FP v1 #6: prosa enfática em caixa-alta (best-seller de saúde).
PROSE_FP_CAPS_EMPHATIC = """\
ISSO MUDA TUDO.
NÃO É EXAGERO.
MAS A CONCLUSÃO É INEVITÁVEL.
"""

# FP v1 #7: laudo laboratorial reproduzido em caixa-alta.
PROSE_FP_LAB_PANEL = """\
GLICOSE        99 MG/DL
INSULINA       12 UIU/ML
CORTISOL       18 UG/DL
TSH            2.1 MUI/L
"""

# FP v1 #8: glossário/estadiamento em caixa-alta.
PROSE_FP_STAGING = """\
ESTÁGIO I
ESTÁGIO II
ESTÁGIO III
ESTÁGIO IV
"""

# Chunk curto/ambíguo: 1 linha só → conservador, NÃO marca (guarda de min-linhas).
SHORT_AMBIGUOUS = "1. Warburg O. Science. 1956;123:309.\n"


# ---------------------------------------------------------------------------
# _classify_boilerplate — verdadeiros-positivos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,text", [
    ("vancouver_numbered", REFERENCE_BLOCK),
    ("vancouver_wrapped_lines", REFERENCE_BLOCK_WRAPPED),  # v3: contagem absoluta
    ("apa_et_al_with_year_vol", REFERENCE_BLOCK_ET_AL),
    ("page_chapter_headers", HEADER_BLOCK),
])
def test_true_boilerplate(name, text):
    assert _classify_boilerplate(text) is True, name


# ---------------------------------------------------------------------------
# _classify_boilerplate — prosa (inclui os 23-FP do red-team, travados)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,text", [
    ("prose", PROSE_BLOCK),
    ("prose_stray_page_marker", PROSE_WITH_STRAY_MARKER),
    ("et_al_inline_en", PROSE_FP_ET_AL_EN),
    ("et_al_inline_pt", PROSE_FP_ET_AL_PT),
    ("inline_journal_no_page", PROSE_FP_INLINE_JOURNAL),
    ("numbered_clinical_protocol", PROSE_FP_NUMBERED_PROTOCOL),
    ("numbered_recipe", PROSE_FP_RECIPE),
    ("caps_emphatic_prose", PROSE_FP_CAPS_EMPHATIC),
    ("caps_lab_panel", PROSE_FP_LAB_PANEL),
    ("caps_staging_glossary", PROSE_FP_STAGING),
    ("short_ambiguous", SHORT_AMBIGUOUS),
])
def test_prose_is_not_boilerplate(name, text):
    assert _classify_boilerplate(text) is False, name


def test_empty_text_is_not_boilerplate():
    assert _classify_boilerplate("") is False
    assert _classify_boilerplate("   \n  \n") is False


# ---------------------------------------------------------------------------
# Round-trip de serialização: flag computada, NÃO persistida, recomputada no load.
# ---------------------------------------------------------------------------

def _cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def test_serialization_does_not_persist_flag_and_recomputes_on_load():
    idx = VectorIndex("livro_teste")
    idx.total_chars = 100
    idx.total_lines = 6
    idx.chunks = [
        ChunkInfo(
            chunk_index=0,
            text=REFERENCE_BLOCK,
            line_start=0,
            line_end=4,
            is_boilerplate=True,
        ),
        ChunkInfo(
            chunk_index=1,
            text=PROSE_BLOCK,
            line_start=5,
            line_end=10,
            is_boilerplate=False,
        ),
    ]
    # Embeddings vivem na matriz do índice, não por-chunk; ingeridos alinhados 1:1.
    idx._ingest_embeddings([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    data = idx.to_serializable()

    # is_boilerplate NÃO vai pro JSON persistido (sem migração de schema).
    for chunk_dict in data["chunks"]:
        assert "is_boilerplate" not in chunk_dict

    # No load, recomputa do texto: refs → True, prosa → False.
    restored = VectorIndex.from_serializable(data)
    assert restored.chunks[0].is_boilerplate is True
    assert restored.chunks[1].is_boilerplate is False
    # Embeddings preservados (sem re-embed). A matriz normaliza (cosseno é
    # invariante a escala), então a direção é o que tem de bater: cos ≈ 1.0.
    assert _cos(restored._vector_at(0), [0.1, 0.2, 0.3]) == pytest.approx(1.0, abs=1e-4)
    assert _cos(restored._vector_at(1), [0.4, 0.5, 0.6]) == pytest.approx(1.0, abs=1e-4)


def test_chunk_text_sets_flag_on_reference_text():
    # _chunk_text computa o flag ao criar cada ChunkInfo (caminho do build()).
    chunks = vi._chunk_text(REFERENCE_BLOCK, chunk_size=512, overlap=50)
    assert chunks, "esperava ao menos um chunk"
    assert all(c.is_boilerplate for c in chunks)


# ---------------------------------------------------------------------------
# _boilerplate_penalty: parsing do env (default desligado, validação de range).
# ---------------------------------------------------------------------------

def test_penalty_default_is_one(monkeypatch):
    monkeypatch.delenv("RLM_BOILERPLATE_PENALTY", raising=False)
    assert _boilerplate_penalty() == 1.0


def test_penalty_valid_value(monkeypatch):
    monkeypatch.setenv("RLM_BOILERPLATE_PENALTY", "0.6")
    assert _boilerplate_penalty() == pytest.approx(0.6)


def test_penalty_invalid_string_falls_back_to_one(monkeypatch):
    monkeypatch.setenv("RLM_BOILERPLATE_PENALTY", "abc")
    assert _boilerplate_penalty() == 1.0


@pytest.mark.parametrize("bad", ["-0.5", "1.5", "2", "10"])
def test_penalty_out_of_range_falls_back_to_one(monkeypatch, bad):
    monkeypatch.setenv("RLM_BOILERPLATE_PENALTY", bad)
    assert _boilerplate_penalty() == 1.0


def test_penalty_zero_is_allowed(monkeypatch):
    # 0.0 está em [0,1]; é permitido (mesmo que o plano recomende 0.5-0.7).
    monkeypatch.setenv("RLM_BOILERPLATE_PENALTY", "0.0")
    assert _boilerplate_penalty() == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
