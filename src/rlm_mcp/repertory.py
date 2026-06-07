"""
Repertorização homeopática — parser, índice, busca e ranking.

Fonte MVP: var `kent_repertorio` (Eizayaga, "El Moderno Repertorio de Kent",
espanhol, extração OCR). A estrutura sobrevivente foi medida live (2026-06-06,
138.738 linhas classificadas): os marcadores `#` de heading são RUÍDO espalhado
em linhas arbitrárias — a classificação é por CONTEÚDO:

  - entry:        `texto: rem1, rem2, REM3.` (split no ÚLTIMO `:`; tail com
                  ≥60% de tokens de remédio válidos)         → 61.408 linhas
  - continuation: lista pura de remédios (wrap de linha)      →  5.467 linhas
  - header:       termina com `:` sem remédios → rubrica nova →  4.160 linhas
  - crossref:     `(ver X).` → pulada                         →    486 linhas
  - page:         `--- Página N ---` → pulada                 →    387 linhas
  - resto: prosa/front matter → pulada e CONTADA em stats

GRAUS: grau 3 = token em CAPS (sobreviveu à extração); grau 2 (itálico) foi
PERDIDO na fonte — tudo que não é CAPS vira grau 1 (decisão do usuário:
ranking binário; o campo `grade` permite upgrade futuro via re-extração).

OCR: canonicalização CONSERVADORA (anti-corrupção clínica): só tokens raros
(freq≤3) são corrigidos, exige candidato ÚNICO no vocabulário estável
(freq≥10), distância Levenshtein ≤2 (≤1 para tokens curtos — `calc`↔`carb`
têm distância 2!), empate descarta. O irreparável é DESCARTADO e contado.

Módulo puro (sem dependência do http_server) — testável isolado. Cache do
índice é lazy, thread-safe e auto-invalidante por fingerprint (sha256) da var.
"""

import hashlib
import logging
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modelo de dados
# ---------------------------------------------------------------------------

# Peso do score por grau (binário na fonte atual: 3 ou 1; o 2 fica reservado
# para quando uma re-extração preservar itálico).
GRADE_WEIGHTS = {3: 3, 2: 2, 1: 1}


@dataclass(frozen=True, slots=True)
class RubricEntry:
    """Uma linha de rubrica com remédios. `line_no` é 1-indexed na var REAL.

    `original_line` NÃO é armazenada (memória): derive de
    `text.split('\\n')[line_no - 1]` na exibição.
    """
    line_no: int
    chapter: str          # capítulo corrente (ex: "PSIQUISMO")
    rubric: str           # rubrica corrente (ex: "ABANDONO")
    text: str             # rótulo da sub-rubrica (ex: "sentimiento de")
    remedies: tuple       # tuple[(canonical: str, grade: int), ...]
    extra_lines: int = 0  # linhas de continuação anexadas a esta entry


@dataclass(slots=True)
class RepertoryStats:
    total_lines: int = 0
    parse_start_line: int = 1
    entries: int = 0
    headers: int = 0
    continuations_merged: int = 0
    orphan_continuations: int = 0
    crossrefs: int = 0
    pages: int = 0
    prose_skipped: int = 0
    colon_prose_skipped: int = 0
    entries_all_discarded: int = 0
    tokens_total: int = 0
    tokens_discarded: int = 0
    vocab_stable: int = 0
    vocab_corrected: int = 0
    glossary_size: int = 0
    build_ms: int = 0


@dataclass(slots=True)
class RepertoryIndex:
    source_var: str
    fingerprint: str
    entries: list = field(default_factory=list)        # list[RubricEntry]
    by_line: dict = field(default_factory=dict)        # line_no -> RubricEntry
    # norm_parts[i] = (chapter_fold, rubric_fold, text_fold) alinhado com entries;
    # o peso da busca depende de ONDE casa (rótulo > rubrica > capítulo)
    norm_parts: list = field(default_factory=list)
    glossary: dict = field(default_factory=dict)       # abrev -> nome completo
    canonical_map: dict = field(default_factory=dict)  # raw_norm -> canonical
    stats: RepertoryStats = field(default_factory=RepertoryStats)
    _vocab: object = None   # cache lazy do vocabulário p/ o fuzzy fallback


@dataclass(frozen=True, slots=True)
class RubricMatch:
    entry: "RubricEntry"
    score: float


@dataclass(slots=True)
class RepertorizationResult:
    rubric_lines: list                 # [line_no, ...] na ordem pedida
    rows: list                         # [(canonical, score, coverage, {line_no: grade})]
    sort: str = "coverage"


# ---------------------------------------------------------------------------
# Normalização / tokens
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"^[a-z][a-z-]{1,11}$")
_PAGE_RE = re.compile(r"^---\s*P[áa]gina\s+\d+\s*---\s*$", re.IGNORECASE)
_CROSSREF_RE = re.compile(r"^\(\s*ver\b[^)]*\)?\.?\s*$", re.IGNORECASE)
_CONT_RE = re.compile(r"\(\s*cont\.?\s*\)\s*:?\s*$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#+\s*")
_GLOSSARY_RE = re.compile(r"^([A-Z][A-Z.\-0-9]{1,11})\s*:\s+([a-zà-ÿ][a-zà-ÿ ()=,.\-']{2,60})$")

# Mini-dicionário PT→ES para termos clínicos frequentes (a fonte é em
# espanhol; o usuário consulta em português). Best-effort — 0 hits sugere
# rlm_search_index híbrido na resposta do handler.
PT_ES = {
    "medo": ["temor"],
    "medos": ["temor", "temores"],
    "sonho": ["sueños", "sueño"],   # capítulo de sonhos é SUEÑOS (plural)
    "sonhos": ["sueños"],
    "manha": ["mañana"],
    "noite": ["noche"],
    "tarde": ["tarde"],
    "choro": ["llanto"],
    "ansiedade": ["ansiedad"],
    "ciume": ["celos"],
    "ciumes": ["celos"],
    "raiva": ["ira", "colera"],
    "crianca": ["niños", "niño"],
    "criancas": ["niños"],
    "dor": ["dolor"],
    "dores": ["dolores"],
    "cabeca": ["cabeza"],
    "vertigem": ["vertigo"],
    "suor": ["sudor"],
    "sede": ["sed"],
    "desmaio": ["desmayo"],
    "morte": ["muerte"],
    "solidao": ["soledad"],
    "consolo": ["consuelo"],
    "susto": ["susto"],
    "mulher": ["mujer"],
    "homem": ["hombre"],
    "gravidez": ["embarazo"],
    "parto": ["parto"],
    "olho": ["ojo"],
    "olhos": ["ojos"],
    "ouvido": ["oido"],
    "garganta": ["garganta"],
    "estomago": ["estomago"],
    "barriga": ["vientre", "abdomen"],
    "pele": ["piel"],
    "febre": ["fiebre"],
    "tosse": ["tos"],
    "sono": ["sueño"],
}


def fold(s: str) -> str:
    """lower + remove acentos (PT/ES compartilham radicais após o fold)."""
    nfkd = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _norm_token(raw: str) -> str:
    """Normaliza um token de remédio: pontuação fora, pontos internos fora,
    OCR `1`→`i` (ex: `CARD-1` → `card-i`), lower."""
    t = raw.strip().strip(".,;:()[]\"'")
    t = t.replace(".", "")
    t = t.replace("1", "i")
    return t.lower()


def _split_tail(tail: str) -> list:
    """Quebra o tail de remédios em pedaços crus (preserva caixa p/ grau)."""
    return [p.strip() for p in tail.split(",") if p.strip()]


# Palavras espanholas de MODALIDADE/anatomia/tempo que aparecem em rótulos de
# rubrica mas NUNCA são remédio. Sem o filtro elas entram como remédio-fantasma
# (medido live: peor=79, mano=34, hora=30, pie=20 entries...) e distorcem o
# ranking. Validado contra o glossário do livro — nenhuma é remédio real
# (`sola`/`solo` = Solanum, REAIS, ficaram FORA da lista de propósito).
REMEDY_STOPWORDS = frozenset({
    # modalidades
    "peor", "mejor", "mejoria", "agravacion", "agrava", "mejora", "agravamiento",
    # tempo
    "noche", "dia", "manana", "tarde", "madrugada", "mediodia", "medianoche",
    "amanecer", "anochecer", "invierno", "verano", "otono", "primavera",
    "hora", "horas",
    # temperatura / clima
    "frio", "calor", "humedo", "seco", "tiempo", "caliente", "fresco",
    # posição / movimento
    "sentado", "sentada", "acostado", "acostada", "parado", "echado", "agachado",
    "moverse", "movimiento", "reposo", "caminar", "caminando", "subir", "bajar",
    "subiendo", "bajando", "estando", "levantarse",
    # corpo
    "mano", "manos", "pie", "pies", "cabeza", "brazo", "pierna", "dedo", "dedos",
    "ojo", "ojos", "boca", "nariz", "oreja", "cuello", "pecho", "espalda", "vientre",
    # lugar / direção
    "aire", "libre", "casa", "cama", "fuera", "dentro", "arriba", "abajo",
    "derecha", "izquierda", "lado", "afuera", "adentro",
    # comuns / relacionais
    "agua", "comida", "comiendo", "bebiendo", "antes", "despues", "durante",
    "mismo", "consuelo", "nada", "todo", "mucho", "poco", "otros", "propios",
})


def _is_remedy_token(norm: str) -> bool:
    return bool(_TOKEN_RE.match(norm)) and norm not in REMEDY_STOPWORDS


def _grade_of(raw: str) -> int:
    """Grau 3 sse as letras do token original são CAPS (sobrevivente da
    tipografia); itálico (grau 2) foi perdido na fonte → resto é grau 1."""
    alpha = [c for c in raw if c.isalpha()]
    return 3 if alpha and all(c.isupper() for c in alpha) else 1


def _levenshtein_capped(a: str, b: str, cap: int) -> int:
    """Distância de edição com teto (early-exit); retorna cap+1 se estourar."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
            row_min = min(row_min, cur[-1])
        if row_min > cap:
            return cap + 1
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Classificação de linha (por CONTEÚDO; o prefixo `#` é ruído de OCR)
# ---------------------------------------------------------------------------

def _classify(line: str) -> tuple:
    """Retorna (classe, payload). Classes: blank, page, crossref, cont_header,
    header, entry, continuation, prose."""
    s = _HEADING_RE.sub("", line).strip()
    if not s:
        return ("blank", None)
    if _PAGE_RE.match(s):
        return ("page", None)
    if _CROSSREF_RE.match(s):
        return ("crossref", None)
    if _CONT_RE.search(s):
        # "SOBRESALTO (cont.)" — continuação de rubrica após quebra de página
        name = _CONT_RE.sub("", s).strip().rstrip(":").strip()
        return ("cont_header", name)
    if s.rstrip().endswith(":"):
        return ("header", s.rstrip().rstrip(":").strip())
    if ":" in s:
        text_part, tail = s.rsplit(":", 1)
        pieces = _split_tail(tail)
        if pieces:
            valid = sum(1 for p in pieces if _is_remedy_token(_norm_token(p)))
            # >=60% REAL (sem truncar/piso): caudas curtas com modalidades
            # soltas ('peor: de noche, frio') seriam aceitas com o piso antigo
            # max(1,int(0.6n)) → remédio-fantasma. valid*5>=3n é exato; n=1
            # remédio único passa (5>=3).
            if valid * 5 >= 3 * len(pieces):
                return ("entry", (text_part.strip(), pieces))
        return ("colon_prose", None)
    pieces = _split_tail(s)
    if len(pieces) >= 2:
        valid = sum(1 for p in pieces if _is_remedy_token(_norm_token(p)))
        if valid * 10 >= 7 * len(pieces):          # >=70% real
            return ("continuation", pieces)
    return ("prose", None)


def _find_parse_start(lines: list) -> int:
    """Índice (0-based) do primeiro capítulo real — pula capa/prólogo/glossário.

    No Kent-Eizayaga o corpo começa em `## PSIQUISMO`. Se não achar (outra
    fonte futura), parseia desde o início.
    """
    for i, line in enumerate(lines):
        if _HEADING_RE.match(line) and fold(_HEADING_RE.sub("", line).strip()) == "psiquismo":
            return i
    return 0


def _harvest_glossary(lines: list, end: int) -> dict:
    """Extrai o glossário do próprio livro (front matter): `ABREV: nome completo`."""
    glossary = {}
    for line in lines[:end]:
        m = _GLOSSARY_RE.match(line.strip())
        if m:
            abbrev = _norm_token(m.group(1))
            if _is_remedy_token(abbrev):
                glossary[abbrev] = m.group(2).strip()
    return glossary


# ---------------------------------------------------------------------------
# Canonicalização (conservadora — feature clínica)
# ---------------------------------------------------------------------------

STABLE_MIN_FREQ = 10    # vocabulário estável (medido live: 1.210 abrevs = 92,4% da massa)
CORRECT_MAX_FREQ = 3    # só corrige tokens raros (cauda OCR)
MIN_STABLE_VOCAB = 30   # sem base estatística → não corrige nem descarta nada
ORPHAN_MERGE_RATIO = 8  # phac→ph-ac só se a forma com hífen domina por ≥8×


def _stem(tok: str) -> str:
    """Gênero da abreviação composta (parte antes do 1º hífen)."""
    return tok.split("-", 1)[0]


def _build_canonical_map(counter: Counter, stable_min_freq: int = STABLE_MIN_FREQ,
                         correct_max_freq: int = CORRECT_MAX_FREQ,
                         min_stable_vocab: int = MIN_STABLE_VOCAB,
                         protected: frozenset = frozenset()) -> tuple:
    """raw_norm -> canonical, ou None (descartado). Regras anti-corrupção:
    - estável (freq≥10): canônico é ele mesmo
    - protegido (no glossário do próprio livro): canônico é ele mesmo — NUNCA
      remapear um remédio que o livro define (form-ac, apoc-a, agar-pr... eram
      trocados por outro remédio real só por distância de edição pequena)
    - zona cinza (4≤freq≤9): mantido como está (remédio raro legítimo possível)
    - raro (freq≤3): corrige para candidato ESTÁVEL ÚNICO na menor distância
      (≤1 se len≤4 — `calc`↔`carb` dist 2!; senão ≤2); 1ª letra deve bater;
      para token COMPOSTO (com hífen), o gênero (antes do 1º hífen) deve bater —
      form-ac↛ferr-ac; empate na menor distância → descarta.
    - corpus pequeno (vocab estável < min_stable_vocab): sem base estatística
      para julgar — mantém TUDO como está (fonte parcial/teste não vira deserto).
    Pós-passe: reconcilia órfãos sem-hífen do OCR (phac→ph-ac) quando a forma
    com hífen é única p/ a chave sem-separador e domina por ≥ORPHAN_MERGE_RATIO×.
    """
    stable = sorted(t for t, n in counter.items() if n >= stable_min_freq)
    if len(stable) < min_stable_vocab:
        return {t: t for t in counter}, len(stable), 0
    # buckets (1ª letra, comprimento): só candidatos com |Δlen| <= cap entram
    # no DP — corta ~10x as chamadas de Levenshtein (corpus real: 2,2M → ~200k)
    by_first_len = {}
    for t in stable:
        by_first_len.setdefault((t[0], len(t)), []).append(t)
    cmap = {}
    corrected = 0
    for tok, n in counter.items():
        if n >= correct_max_freq + 1 or tok in protected:
            cmap[tok] = tok
            continue
        cap = 1 if len(tok) <= 4 else 2
        tok_stem = _stem(tok)
        best_d, best = cap + 1, []
        for ln in range(max(2, len(tok) - cap), len(tok) + cap + 1):
            for cand in by_first_len.get((tok[0], ln), ()):
                # token composto: não cruza gênero (form-ac↛ferr-ac)
                if "-" in tok and _stem(cand) != tok_stem:
                    continue
                d = _levenshtein_capped(tok, cand, cap)
                if d < best_d:
                    best_d, best = d, [cand]
                elif d == best_d:
                    best.append(cand)
        if best_d == 0:
            cmap[tok] = tok          # raro mas idêntico a estável (não acontece, defensivo)
        elif len(best) == 1 and best_d <= cap:
            cmap[tok] = best[0]
            corrected += 1
        else:
            cmap[tok] = None         # irreparável ou ambíguo → descartado

    # Pós-passe: hífen perdido pelo OCR ('ph.ac'/'phac' vs 'ph-ac') parte o mesmo
    # remédio em duas linhas-fantasma quando a forma degradada também é frequente.
    # O ALVO da fusão é a forma DOMINANTE do grupo (mesma chave sem separador) e
    # só se ela for COM hífen — o que protege ambr(ambra, dominante, sem hífen)
    # de virar am-br(ammonium brom.): ali o dominante não tem hífen → não funde.
    by_stripped = {}
    for t in counter:
        by_stripped.setdefault(t.replace("-", ""), []).append(t)
    for key, toks in by_stripped.items():
        if len(toks) < 2:
            continue
        canon = max(toks, key=lambda t: counter[t])
        if "-" not in canon or counter[canon] < stable_min_freq:
            continue                 # dominante sem hífen (ambr) → não reconcilia
        for orphan in toks:
            # funde a forma degradada (sem o hífen, ou ruído de hífen raro) na
            # dominante, só se hoje aponta p/ si mesma e a dominante a supera por
            # ≥ORPHAN_MERGE_RATIO×. NÃO aplica o guard de glossário aqui: a forma
            # degradada (phac) às vezes vazou para o glossário, e fundi-la na
            # canônica (ph-ac) é unir o MESMO remédio — a proteção contra unir
            # remédios distintos (ambr≠am-br) é a regra dominante-com-hífen acima.
            if (orphan != canon and cmap.get(orphan) == orphan
                    and counter[canon] >= ORPHAN_MERGE_RATIO * counter[orphan]):
                cmap[orphan] = canon
                corrected += 1
    return cmap, len(stable), corrected


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_kent_repertory(text: str, source_var: str = "kent_repertorio", *,
                         stable_min_freq: int = STABLE_MIN_FREQ,
                         correct_max_freq: int = CORRECT_MAX_FREQ,
                         min_stable_vocab: int = MIN_STABLE_VOCAB) -> RepertoryIndex:
    """Parseia o texto inteiro → RepertoryIndex. Puro (sem cache, sem I/O).

    Os thresholds de canonicalização são parametrizáveis p/ testes com
    fixtures pequenas (defaults calibrados no corpus real)."""
    t0 = time.time()
    lines = text.split("\n")
    start = _find_parse_start(lines)
    stats = RepertoryStats(total_lines=len(lines), parse_start_line=start + 1)

    glossary = _harvest_glossary(lines, start)
    stats.glossary_size = len(glossary)

    # ---- passada 1: inventário de tokens (p/ vocabulário estável) ----------
    counter = Counter()
    classified = []   # (line_no_1idx, classe, payload) — evita reclassificar
    for i, line in enumerate(lines[start:], start + 1):
        cls, payload = _classify(line)
        classified.append((i, cls, payload))
        if cls == "entry":
            for p in payload[1]:
                norm = _norm_token(p)
                if _is_remedy_token(norm):
                    counter[norm] += 1
        elif cls == "continuation":
            for p in payload:
                norm = _norm_token(p)
                if _is_remedy_token(norm):
                    counter[norm] += 1

    cmap, n_stable, n_corrected = _build_canonical_map(
        counter, stable_min_freq, correct_max_freq, min_stable_vocab,
        protected=frozenset(glossary))
    stats.vocab_stable = n_stable
    stats.vocab_corrected = n_corrected

    def _resolve(pieces):
        """pieces crus → tuple[(canonical, grade)], dedup intra-entry (grau max)."""
        out = {}
        discarded = 0
        for p in pieces:
            norm = _norm_token(p)
            if not _is_remedy_token(norm):
                discarded += 1
                continue
            canonical = cmap.get(norm)
            if canonical is None:
                discarded += 1
                continue
            g = _grade_of(p)
            if g > out.get(canonical, 0):
                out[canonical] = g
        return tuple(sorted(out.items())), len(pieces), discarded

    # ---- passada 2: montagem das entries ------------------------------------
    index = RepertoryIndex(source_var=source_var, fingerprint="", glossary=glossary,
                           canonical_map=cmap, stats=stats)
    chapter, rubric = "", ""
    # entries são imutáveis (frozen); continuations acumulam ANTES de fechar
    pending = None   # [line_no, chapter, rubric, text, {canonical: grade}, extra]
    # header recém-aberto: lista de remédios que vem LOGO APÓS um header de
    # rubrica são os remédios da rubrica principal (text="" → exibe a rubrica)
    header_open = False

    def _flush():
        nonlocal pending
        if pending is None:
            return
        line_no, chap, rub, txt, remedies, extra = pending
        pending = None
        if not remedies:
            stats.entries_all_discarded += 1
            return
        e = RubricEntry(line_no=line_no, chapter=chap, rubric=rub, text=txt,
                        remedies=tuple(sorted(remedies.items())), extra_lines=extra)
        index.entries.append(e)
        index.by_line[line_no] = e

    def _next_is_continuation(start: int) -> bool:
        """A próxima linha significativa (pulando blank/page) é uma lista de
        remédios? Distingue running-head de ruído (segue lista) de capítulo real."""
        for j in range(start + 1, len(classified)):
            nxt = classified[j][1]
            if nxt in ("blank", "page"):
                continue
            return nxt == "continuation"
        return False

    for _ci, (line_no, cls, payload) in enumerate(classified):
        if cls == "blank":
            continue
        if cls == "page":
            stats.pages += 1
            continue            # NÃO fecha pending: lista pode atravessar página
        if cls == "crossref":
            stats.crossrefs += 1
            continue
        if cls == "cont_header":
            stats.headers += 1
            # só sobrescreve a rubrica se o (cont.) NÃO repete o capítulo corrente
            # (running-header de página repete o capítulo → manter rubrica)
            if payload and fold(payload) != fold(chapter):
                rubric = payload  # mantém pending: continuação após quebra
            continue
        if cls == "header":
            _flush()
            stats.headers += 1
            rubric = payload
            header_open = True
            continue
        if cls == "entry":
            _flush()
            header_open = False
            stats.entries += 1
            text_part, pieces = payload
            resolved, n_tok, n_disc = _resolve(pieces)
            stats.tokens_total += n_tok
            stats.tokens_discarded += n_disc
            # entry com rótulo ALL-CAPS é rubrica principal com remédios inline
            alpha = re.sub(r"[^A-Za-zÀ-ÿ]", "", text_part)
            if len(alpha) >= 3 and alpha.isupper():
                rubric = text_part
            pending = [line_no, sys.intern(chapter), sys.intern(rubric),
                       text_part, dict(resolved), 0]
            continue
        if cls == "continuation":
            if pending is None and header_open and rubric:
                # remédios da rubrica principal (vêm logo após o header);
                # text="" → exibição usa a rubrica; citação = linha da lista
                stats.entries += 1
                pending = [line_no, sys.intern(chapter), sys.intern(rubric),
                           "", {}, -1]
            if pending is not None:
                resolved, n_tok, n_disc = _resolve(payload)
                stats.tokens_total += n_tok
                stats.tokens_discarded += n_disc
                stats.continuations_merged += 1
                for canonical, g in resolved:
                    if g > pending[4].get(canonical, 0):
                        pending[4][canonical] = g
                pending[5] += 1
            else:
                stats.orphan_continuations += 1
            continue
        if cls == "colon_prose":
            stats.colon_prose_skipped += 1
            header_open = False
            _flush()
            continue
        # prose: pode ser capítulo (heading CAPS curto sem ':') OU running-head
        # de ruído no meio de uma lista de remédios
        raw = _HEADING_RE.sub("", lines[line_no - 1]).strip()
        alpha = re.sub(r"[^A-Za-zÀ-ÿ]", "", raw)
        is_caps_heading = (_HEADING_RE.match(lines[line_no - 1]) and len(raw) < 60
                           and len(alpha) >= 3 and alpha.isupper())
        if is_caps_heading and _next_is_continuation(_ci):
            # '#'-CAPS seguido DIRETO de mais remédios = running-head partindo
            # uma lista; é ruído → não fecha pending nem troca capítulo (senão a
            # continuação vira órfã e o capítulo seguinte fica corrompido)
            continue
        _flush()
        header_open = False
        if is_caps_heading:
            chapter = raw
            rubric = ""
        else:
            stats.prose_skipped += 1

    _flush()

    # ---- busca: partes normalizadas alinhadas com entries -------------------
    index.norm_parts = [
        (fold(e.chapter), fold(e.rubric), fold(e.text)) for e in index.entries
    ]
    stats.build_ms = int((time.time() - t0) * 1000)
    logger.info(
        f"Repertory index built: {len(index.entries)} entries "
        f"({stats.continuations_merged} continuations merged, "
        f"{stats.tokens_discarded}/{stats.tokens_total} tokens discarded) "
        f"in {stats.build_ms}ms"
    )
    return index


# ---------------------------------------------------------------------------
# Cache lazy thread-safe auto-invalidante (padrão indexer._get_bm25_lock)
# ---------------------------------------------------------------------------

_repertory_cache: dict = {}          # var_name -> RepertoryIndex (fingerprint dentro)
_repertory_locks: dict = {}
_repertory_locks_guard = threading.Lock()


def _get_lock(var_name: str) -> threading.Lock:
    lock = _repertory_locks.get(var_name)
    if not lock:
        with _repertory_locks_guard:
            lock = _repertory_locks.get(var_name)
            if not lock:
                lock = threading.Lock()
                _repertory_locks[var_name] = lock
    return lock


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def get_repertory_index(source_var: str, text: str) -> tuple:
    """Índice cacheado; rebuild automático se a var mudou (fingerprint).
    Retorna (index, was_cached) — was_cached=False quando acabou de (re)parsear."""
    fp = _fingerprint(text)
    cached = _repertory_cache.get(source_var)
    if cached is not None and cached.fingerprint == fp:
        return cached, True
    with _get_lock(source_var):
        cached = _repertory_cache.get(source_var)
        if cached is not None and cached.fingerprint == fp:
            return cached, True
        index = parse_kent_repertory(text, source_var)
        index.fingerprint = fp
        _repertory_cache[source_var] = index
        return index, False


def clear_repertory_cache():
    """P/ testes (espelha clear_all_indices/clear_all_vector_indices)."""
    _repertory_cache.clear()


# ---------------------------------------------------------------------------
# Busca de rubricas
# ---------------------------------------------------------------------------

def _expand_query_tokens(query: str) -> list:
    """fold + split + expansão PT→ES. Retorna lista de conjuntos de alternativas.

    Descarta tokens sem letra (ex: '>' do caminho 'CAP > RUBRICA > texto' que o
    formatter exibe, '?', '¿', '!') — senão viram termo AND impossível e zeram
    a busca de quem cola o resultado de volta."""
    out = []
    for tok in fold(query).split():
        tok = tok.strip(".,;:()")
        if not tok or not any(c.isalpha() for c in tok):
            continue
        alts = {tok}
        for alt in PT_ES.get(tok, []):
            alts.add(fold(alt))
        out.append(alts)
    return out


_WORD_RE = re.compile(r"[a-z0-9-]+")


def search_rubrics(index: RepertoryIndex, query: str, limit: int = 10,
                   offset: int = 0) -> tuple:
    """Busca AND sobre `capítulo rubrica texto` normalizados.

    O peso depende de ONDE o token casa (especificidade): palavra no rótulo da
    sub-rubrica vale mais que na rubrica, que vale mais que no capítulo. Assim a
    rubrica canônica do termo não é soterrada por um capítulo inteiro empatado,
    nem perde por causa de vírgula no rótulo. Qualquer alternativa PT→ES vale.
    Fallback fuzzy (difflib) só se 0 resultados. Retorna (página, total, note).
    """
    token_sets = _expand_query_tokens(query)
    if not token_sets:
        return [], 0, None

    def _weight(alt, chap_f, rub_f, txt_f, rub_words, txt_words):
        if alt in txt_words:
            return 3.0          # palavra exata no rótulo da sub-rubrica
        if alt in txt_f:
            return 2.0          # substring no rótulo
        if alt in rub_words:
            return 1.5          # palavra exata na rubrica
        if alt in rub_f:
            return 1.0          # substring na rubrica
        if alt in chap_f:
            return 0.5          # só no capítulo (compartilhado por milhares)
        return 0.0

    def _run(tsets):
        matches = []
        for i, (chap_f, rub_f, txt_f) in enumerate(index.norm_parts):
            rub_words = txt_words = None
            score = 0.0
            ok = True
            for alts in tsets:
                best = 0.0
                for alt in alts:
                    if alt not in chap_f and alt not in rub_f and alt not in txt_f:
                        continue
                    if rub_words is None:
                        rub_words = set(_WORD_RE.findall(rub_f))
                        txt_words = set(_WORD_RE.findall(txt_f))
                    w = _weight(alt, chap_f, rub_f, txt_f, rub_words, txt_words)
                    if w > best:
                        best = w
                if best == 0.0:
                    ok = False
                    break
                score += best
            if ok:
                matches.append(RubricMatch(entry=index.entries[i], score=score))
        matches.sort(key=lambda m: (-m.score, m.entry.line_no))
        return matches

    fuzzy_note = None
    matches = _run(token_sets)
    if not matches:
        # fuzzy fallback: corrige cada token contra o vocabulário das rubricas.
        # Vocab memoizado no índice (imutável, fingerprint-keyed) — não refaz a
        # cada query 0-hit.
        import difflib
        vocab = index._vocab
        if vocab is None:
            vocab = set()
            for chap_f, rub_f, txt_f in index.norm_parts:
                vocab.update(_WORD_RE.findall(chap_f))
                vocab.update(_WORD_RE.findall(rub_f))
                vocab.update(_WORD_RE.findall(txt_f))
            index._vocab = vocab
        fixed, changed = [], []
        for alts in token_sets:
            tok = next(iter(alts))
            close = difflib.get_close_matches(tok, vocab, n=1, cutoff=0.8)
            if close and close[0] not in alts:
                fixed.append(alts | {close[0]})
                changed.append(f"{tok}→{close[0]}")
            else:
                fixed.append(alts)
        if changed:
            matches = _run(fixed)
            if matches:
                fuzzy_note = ", ".join(changed)
    total = len(matches)
    return matches[offset:offset + limit], total, fuzzy_note


# ---------------------------------------------------------------------------
# Repertorização
# ---------------------------------------------------------------------------

_REF_RE = re.compile(r"^(?:(?P<var>[A-Za-z_][\w-]*):)?L?(?P<line>\d+)$")


def resolve_rubric_refs(index: RepertoryIndex, refs: list) -> tuple:
    """refs (`var:L123`, `L123`, `123` ou texto) → entries. Texto só com match
    ÚNICO (ambíguo → erro com candidatos). Retorna (entries, erros, fuzzy_notes).

    fuzzy_notes avisa quando uma ref TEXTUAL só casou via correção fuzzy — numa
    ferramenta clínica, substituir silenciosamente a rubrica pedida é justamente
    o que o desenho evita."""
    entries, errors, fuzzy_notes = [], [], []
    for ref in refs:
        ref_s = str(ref).strip()
        m = _REF_RE.match(ref_s)
        if m:
            if m.group("var") and m.group("var") != index.source_var:
                errors.append(f"'{ref_s}': var '{m.group('var')}' não é a fonte ({index.source_var})")
                continue
            line_no = int(m.group("line"))
            e = index.by_line.get(line_no)
            if e is None:
                errors.append(f"'{ref_s}': linha {line_no} não é uma rubrica do índice")
            else:
                entries.append(e)
            continue
        # referência textual: exige match único
        matches, total, fnote = search_rubrics(index, ref_s, limit=4, offset=0)
        if total == 1:
            entries.append(matches[0].entry)
            if fnote:
                fuzzy_notes.append(f"'{ref_s}' → {fnote}")
        elif total == 0:
            errors.append(f"'{ref_s}': nenhuma rubrica casa")
        else:
            opts = "; ".join(
                f"{index.source_var}:L{m.entry.line_no} {m.entry.text[:40]}"
                for m in matches[:3]
            )
            errors.append(f"'{ref_s}': ambíguo ({total} rubricas — use o ID. Ex: {opts})")
    return entries, errors, fuzzy_notes


def repertorize(index: RepertoryIndex, entries: list, sort: str = "coverage") -> RepertorizationResult:
    """Cruza rubricas → ranking de remédios.

    score = soma dos pesos de grau; coverage = nº de rubricas onde aparece.
    sort="coverage": coverage desc → score desc → nome. sort="score": score
    desc → coverage desc → nome.
    """
    table: dict = {}
    for e in entries:
        for canonical, grade in e.remedies:
            row = table.setdefault(canonical, {})
            if grade > row.get(e.line_no, 0):
                row[e.line_no] = grade
    rows = []
    for canonical, per_line in table.items():
        score = sum(GRADE_WEIGHTS.get(g, 1) for g in per_line.values())
        rows.append((canonical, score, len(per_line), per_line))
    if sort == "score":
        rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    else:
        rows.sort(key=lambda r: (-r[2], -r[1], r[0]))
    return RepertorizationResult(
        rubric_lines=[e.line_no for e in entries], rows=rows, sort=sort,
    )
