"""
Canários do batching de embeddings (bug 2026-06-06): a estimativa antiga de
4 chars/token deixava lotes de 380-409k tokens REAIS passarem pro cap de 300k
da OpenAI → 400 em cascata → vars sem embedding ou com cobertura parcial
silenciosa. Fix: estimativa conservadora (2 chars/token) + split-retry no 400.
"""


from rlm_mcp.embeddings import EmbeddingService, _pack_batches


# Razão real medida live no corpus ReCODE: ~2,44 chars/token (pior caso).
WORST_CHARS_PER_TOKEN = 2.44
API_TOKEN_CAP = 300_000


def real_tokens(batch):
    return sum(len(t) / WORST_CHARS_PER_TOKEN for t in batch)


def test_pack_respeita_cap_real_no_pior_caso_medido():
    """Lotes da estimativa //2 devem caber no cap REAL mesmo a 2,44 chars/token."""
    chunks = ["x" * 512 for _ in range(40_000)]  # ~20M chars (recode_diagnostico)
    batches = _pack_batches(chunks)
    assert sum(len(b) for b in batches) == len(chunks)  # nada perdido
    for b in batches:
        assert real_tokens(b) < API_TOKEN_CAP, (
            f"lote de {len(b)} chunks = {real_tokens(b):,.0f} tokens reais "
            f"(>{API_TOKEN_CAP:,}) — regressão do bug do //4"
        )


def test_pack_preserva_ordem_e_alinhamento():
    chunks = [f"chunk-{i:05d}" for i in range(5000)]
    flat = [t for b in _pack_batches(chunks) for t in b]
    assert flat == chunks


def test_pack_respeita_max_batch():
    batches = _pack_batches(["a" for _ in range(5000)], max_batch=2048)
    assert all(len(b) <= 2048 for b in batches)


class _Item:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _Resp:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddingsApi:
    """Simula o endpoint: rejeita lotes acima do cap REAL de tokens."""

    def __init__(self, chars_per_token=WORST_CHARS_PER_TOKEN):
        self.chars_per_token = chars_per_token
        self.calls = 0

    def create(self, input, model):
        self.calls += 1
        tokens = sum(len(t) / self.chars_per_token for t in input)
        if tokens > API_TOKEN_CAP:
            raise RuntimeError(
                f"Error code: 400 - max_tokens_per_request: Requested "
                f"{tokens:.0f} tokens, max {API_TOKEN_CAP} tokens per request"
            )
        # devolve em ordem embaralhada de propósito (a API não garante ordem)
        data = [_Item(i, [float(i)]) for i in range(len(input))]
        return _Resp(list(reversed(data)))


class _FakeClient:
    def __init__(self, api):
        self.embeddings = api


def _service_with(api):
    svc = EmbeddingService.__new__(EmbeddingService)
    svc.mode = "openai"
    svc._client = _FakeClient(api)
    svc._model = "text-embedding-3-small"
    return svc


def test_split_retry_converge_em_lote_patologico():
    """Mesmo se a estimativa falhar, o 400 de token-cap divide e converge —
    nenhum chunk fica sem embedding."""
    api = _FakeEmbeddingsApi(chars_per_token=1.0)  # patológico: 1 char = 1 token
    svc = _service_with(api)
    # 1 lote de ~976 chunks × 512 chars = 500k tokens reais a 1c/t → estoura
    batch = ["y" * 512 for _ in range(976)]
    out = svc._embed_call(batch)
    assert len(out) == len(batch)
    assert all(e != [] for e in out), "split-retry não pode deixar buracos"
    assert api.calls > 1, "deveria ter dividido o lote"


def test_erro_nao_token_cap_preenche_vazios_sem_dividir():
    class _Boom:
        calls = 0

        def create(self, input, model):
            self.calls += 1
            raise RuntimeError("Error code: 429 - rate limited")

    api = _Boom()
    svc = _service_with(api)
    out = svc._embed_call(["a", "b", "c"])
    assert out == [[], [], []]
    assert api.calls == 1, "erro que não é token-cap NÃO deve disparar split"


def test_embed_texts_fim_a_fim_ordena_por_index():
    api = _FakeEmbeddingsApi()
    svc = _service_with(api)
    out = svc.embed_texts([f"t{i}" for i in range(10)])
    assert [e[0] for e in out] == [float(i) for i in range(10)], (
        "resultado deve ser re-ordenado por .index (API embaralha)"
    )
