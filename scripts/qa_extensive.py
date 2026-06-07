#!/usr/bin/env python3
"""
QA extensivo live do RLM MCP Server — caça-bugs por invariantes, não por status code.

Roda contra o servidor REAL via POST /mcp (Streamable HTTP), exercitando as 19
tools com checagem de comportamento: protocolo MCP, CRUD de variáveis, sandbox
do execute, busca (BM25/phrase/require_all/fallback), ciclo de coleção com
VERIFICAÇÃO DE RASTREABILIDADE (toda citação var:Lnnn é conferida contra o
conteúdo real da var — canário dos P0s de 2026-06-06), concorrência (event
loop livre durante execute lento) e edge cases (unicode, regex metachars,
payloads, erros limpos).

Uso:
    python3 scripts/qa_extensive.py                       # produção (default)
    python3 scripts/qa_extensive.py --base http://localhost:8765
    RLM_API_KEY=xxx python3 scripts/qa_extensive.py
    python3 scripts/qa_extensive.py --skip-s3 --quick

Key: --key > env RLM_API_KEY > ~/.claude.json (entry mcpServers.rlm).
Side effects: vars _qa_* e coleção 'qa_harness' — ambas removidas no cleanup
(delete de coleção exposto na API desde 2026-06-06).
Exit code = nº de FAILs.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

DEFAULT_BASE = "https://rlm.drsolution.online"
EXPECTED_TOOLS = {
    "rlm_execute", "rlm_load_data", "rlm_load_file", "rlm_load_s3",
    "rlm_list_vars", "rlm_var_info", "rlm_clear", "rlm_memory", "rlm_pin_var",
    "rlm_list_buckets", "rlm_list_s3", "rlm_upload_url", "rlm_save_to_s3",
    "rlm_process_pdf", "rlm_search_index", "rlm_search_code", "rlm_collection",
    "rlm_repertorio", "rlm_task", "rlm_help",
}
QA_PREFIX = "_qa_"
QA_COLLECTION = "qa_harness"

PASS, FAIL, SKIP, KNOWN = "PASS", "FAIL", "SKIP", "KNOWN"


class Reporter:
    def __init__(self):
        self.results = []  # (section, name, status, detail)
        self._lock = threading.Lock()

    def add(self, section, name, status, detail=""):
        with self._lock:
            self.results.append((section, name, status, str(detail)[:300]))
        mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "KNOWN": "⚠️"}[status]
        print(f"  {mark} [{section}] {name}" + (f" — {str(detail)[:160]}" if status != PASS and detail else ""))

    def check(self, section, name, cond, detail=""):
        self.add(section, name, PASS if cond else FAIL, "" if cond else detail)
        return bool(cond)

    def summary(self):
        counts = {s: sum(1 for r in self.results if r[2] == s) for s in (PASS, FAIL, KNOWN, SKIP)}
        print("\n" + "=" * 78)
        print(f"RESULTADO: {counts[PASS]} pass | {counts[FAIL]} FAIL | "
              f"{counts[KNOWN]} known-issue | {counts[SKIP]} skip")
        if counts[FAIL]:
            print("\nFALHAS:")
            for sec, name, st, det in self.results:
                if st == FAIL:
                    print(f"  ❌ [{sec}] {name}\n     {det}")
        if counts[KNOWN]:
            print("\nKNOWN ISSUES (não contam como falha):")
            for sec, name, st, det in self.results:
                if st == KNOWN:
                    print(f"  ⚠️ [{sec}] {name} — {det}")
        return counts[FAIL]


class RlmClient:
    """Cliente JSON-RPC mínimo (stdlib) com retry educado em 429."""

    def __init__(self, base, key, timeout=60):
        self.base = base.rstrip("/")
        self.key = key
        self.timeout = timeout
        self._id = 0
        self._lock = threading.Lock()
        self.calls = 0

    def _post_raw(self, path, body: bytes, headers=None, timeout=None):
        req = urllib.request.Request(self.base + path, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    def rpc(self, method, params=None, _retries=2, timeout=None):
        """POST /mcp. Retorna (http_status, parsed_json_or_None, raw_text)."""
        with self._lock:
            self._id += 1
            rid = self._id
            self.calls += 1
        body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                           "params": params or {}}).encode()
        status, text = self._post_raw("/mcp", body, timeout=timeout)
        if status == 429 and _retries > 0:
            try:
                wait = float(json.loads(text).get("retry_after") or 2)
            except Exception:
                wait = 2.0
            time.sleep(min(wait + 0.5, 15))
            return self.rpc(method, params, _retries - 1, timeout)
        try:
            return status, json.loads(text), text
        except Exception:
            return status, None, text

    def tool(self, name, args=None, timeout=None):
        """tools/call. Retorna (ok, text, raw_response_dict)."""
        status, parsed, raw = self.rpc(
            "tools/call", {"name": name, "arguments": args or {}}, timeout=timeout)
        if status != 200 or not parsed or "result" not in parsed:
            return False, f"HTTP {status}: {raw[:200]}", parsed
        result = parsed["result"]
        text = ""
        try:
            text = result["content"][0]["text"]
        except Exception:
            pass
        return not result.get("isError", False), text, parsed

    def execute(self, code, timeout=90):
        return self.tool("rlm_execute", {"code": code}, timeout=timeout)


def resolve_key(args):
    if args.key:
        return args.key
    if os.environ.get("RLM_API_KEY"):
        return os.environ["RLM_API_KEY"]
    try:
        cfg = json.load(open(os.path.expanduser("~/.claude.json")))
        auth = cfg["mcpServers"]["rlm"]["headers"]["Authorization"]
        return auth.split()[-1]
    except Exception:
        return ""


# =============================================================================
# Seções de teste
# =============================================================================

def sec_protocol(c: RlmClient, r: Reporter):
    print("\n[protocol] Protocolo MCP / transporte")
    st, p, _ = c.rpc("initialize", {"protocolVersion": "2025-03-26",
                                    "capabilities": {},
                                    "clientInfo": {"name": "qa", "version": "1"}})
    r.check("protocol", "initialize ecoa versão suportada",
            st == 200 and p and p.get("result", {}).get("protocolVersion") == "2025-03-26",
            f"st={st} resp={str(p)[:150]}")

    st, p, _ = c.rpc("tools/list")
    tools = {t["name"] for t in (p or {}).get("result", {}).get("tools", [])}
    r.check("protocol", f"tools/list devolve as {len(EXPECTED_TOOLS)} tools",
            tools == EXPECTED_TOOLS,
            f"faltando={EXPECTED_TOOLS - tools} extras={tools - EXPECTED_TOOLS}")

    body = json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled",
                       "params": {"requestId": 1, "reason": "qa"}}).encode()
    st, text = c._post_raw("/mcp", body)
    r.check("protocol", "notification → 202 sem body", st == 202 and not text,
            f"st={st} body={text[:100]}")

    st, p, _ = c.rpc("metodo/inexistente")
    r.check("protocol", "método desconhecido → erro -32601",
            p and p.get("error", {}).get("code") == -32601, str(p)[:150])

    # auth: key errada → 401 (se o servidor tem key configurada)
    bad = RlmClient(c.base, "chave-invalida-qa")
    st, _, raw = bad.rpc("tools/list", _retries=0)
    if st == 200:
        r.add("protocol", "auth rejeita Bearer inválido", KNOWN,
              "servidor em open-auth (RLM_API_KEY vazio?) — footgun conhecido")
    else:
        r.check("protocol", "auth rejeita Bearer inválido", st == 401, f"st={st}")

    st, text = c._post_raw("/mcp", json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
                           headers={"Origin": "https://evil.example"})
    r.check("protocol", "Origin desconhecida → 403", st == 403, f"st={st}")

    st, text = c._post_raw("/message?session_id=qa-stale-" + uuid.uuid4().hex[:8],
                           json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode())
    r.check("protocol", "session SSE stale → 404 fail-fast", st == 404, f"st={st} {text[:100]}")

    st, text = c._post_raw("/mcp", b"{json quebrado", timeout=15)
    r.check("protocol", "JSON malformado → resposta controlada (4xx/5xx, não hang)",
            st in (400, 422, 500), f"st={st}")

    # GET /mcp → 405 (spec permite; FastAPI dá de graça)
    req = urllib.request.Request(c.base + "/mcp", method="GET")
    req.add_header("Authorization", f"Bearer {c.key}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            st = resp.status
    except urllib.error.HTTPError as e:
        st = e.code
    r.check("protocol", "GET /mcp → 405", st == 405, f"st={st}")


def sec_data_tools(c: RlmClient, r: Reporter):
    print("\n[data] CRUD de variáveis")
    ok, t, _ = c.tool("rlm_load_data", {"name": f"{QA_PREFIX}text",
                                        "data": "linha um\nlinha dois\nlinha três"})
    r.check("data", "load_data text", ok, t)

    ok, t, _ = c.tool("rlm_load_data", {"name": f"{QA_PREFIX}json",
                                        "data": '{"a": 1, "b": [2, 3]}', "data_type": "json"})
    r.check("data", "load_data json", ok, t)

    ok, t, _ = c.tool("rlm_load_data", {"name": f"{QA_PREFIX}csv",
                                        "data": "col1,col2\nx,1\ny,2", "data_type": "csv"})
    r.check("data", "load_data csv", ok, t)

    ok, t, _ = c.tool("rlm_load_data", {"name": f"{QA_PREFIX}badjson",
                                        "data": "{json inválido", "data_type": "json"})
    r.check("data", "json inválido → erro limpo (não exceção crua)",
            (not ok) or "Erro" in t, t[:150])

    ok, t, _ = c.tool("rlm_var_info", {"name": f"{QA_PREFIX}text"})
    r.check("data", "var_info reporta tipo str e tamanho",
            ok and "str" in t and ("33" in t or "B" in t), t[:150])

    ok, t, _ = c.tool("rlm_var_info", {"name": f"{QA_PREFIX}inexistente_xyz"})
    r.check("data", "var_info de var inexistente → erro limpo", not ok or "não encontrada" in t.lower() or "nao encontrada" in t.lower(), t[:120])

    # list_vars é paginado e pode ter centenas de vars de produção — basta
    # responder com contagem; a presença das _qa_* é provada pelo var_info acima
    ok, t, _ = c.tool("rlm_list_vars", {"limit": 50})
    r.check("data", "list_vars responde paginado", ok and "vars" in t, t[:150])

    ok, t, _ = c.tool("rlm_pin_var", {"name": f"{QA_PREFIX}text", "pin": True})
    r.check("data", "pin_var", ok, t[:120])
    ok, t, _ = c.tool("rlm_pin_var", {"name": f"{QA_PREFIX}text", "pin": False})
    r.check("data", "unpin_var", ok, t[:120])

    ok, t, _ = c.tool("rlm_memory")
    r.check("data", "rlm_memory responde com uso", ok and ("MB" in t or "%" in t or "B" in t), t[:120])

    # unicode/emoji round-trip (lossless até o sandbox e de volta)
    payload = "ação coração 🧠 ñ ü\nsegunda linha çedilha"
    ok, t, _ = c.tool("rlm_load_data", {"name": f"{QA_PREFIX}uni", "data": payload})
    ok2, t2, _ = c.execute(f"print(repr({QA_PREFIX}uni.split(chr(10))[0]))")
    r.check("data", "unicode/emoji round-trip via execute",
            ok and ok2 and "🧠" in t2 and "ação" in t2, t2[:150])


def sec_execute(c: RlmClient, r: Reporter):
    print("\n[execute] Sandbox Python")
    ok, t, _ = c.execute("x = 21 * 2\nprint('resultado:', x)")
    r.check("execute", "execução básica + stdout", ok and "resultado: 42" in t, t[:150])

    ok, t, _ = c.execute("print(y_que_nao_existe)")
    r.check("execute", "NameError → erro limpo com traceback", "NameError" in t, t[:150])

    # persistência de DADOS entre execuções. NOTA: vars com prefixo "_" são
    # filtradas do merge-back do sandbox POR DESIGN (sandbox_worker:350,
    # protege _coll_*/_code_structure_* de sobrescrita) — usar nome sem "_".
    c.execute("qa_estado_exec = 'persistiu'")
    ok, t, _ = c.execute("print(qa_estado_exec)")
    r.check("execute", "dados persistem entre execuções", ok and "persistiu" in t, t[:150])
    c.tool("rlm_clear", {"name": "qa_estado_exec"})

    ok, t, _ = c.execute(f"{QA_PREFIX}privada = 1")
    ok2, t2, _ = c.execute(f"print({QA_PREFIX}privada)")
    r.check("execute", "vars com '_' NÃO persistem (design: merge-back filtra privadas)",
            "NameError" in t2, t2[:120])

    ok, t, _ = c.execute(
        f"res = buscar({QA_PREFIX}text, 'dois')\nprint(res[0]['linha'], res[0]['contexto'][:30])")
    r.check("execute", "helper buscar() disponível e 1-indexed",
            ok and t.strip().startswith("2 "), t[:150])

    ok, t, _ = c.execute(f"print(contar({QA_PREFIX}text, 'linha')['total'])")
    r.check("execute", "helper contar()", ok and "3" in t, t[:150])

    ok, t, _ = c.execute("import socket\ns = socket.socket()\ns.connect(('1.1.1.1', 80))")
    r.check("execute", "rede bloqueada no sandbox (B2 seccomp)",
            (not ok) or "EPERM" in t or "Erro" in t or "Error" in t or "denied" in t.lower()
            or "SecurityError" in t, t[:200])

    ok, t, _ = c.execute("print(eval('1+1'))")
    r.check("execute", "eval bloqueado (deny-list AST)",
            "SecurityError" in t or "Erro" in t or not ok, t[:150])

    ok, t, _ = c.execute("print(open('/persist/rlm_data.db','rb').read(10))")
    r.check("execute", "/persist inacessível do sandbox (B2 Landlock)",
            (not ok) or "Erro" in t or "Error" in t or "denied" in t.lower() or "SecurityError" in t,
            t[:200])


def sec_search_single(c: RlmClient, r: Reporter):
    print("\n[search] Busca single-var")
    doc = "\n".join(
        ["alvo logo na primeira frase deste documento de teste."]
        + [f"linha de enchimento numero {i} sem nada util." for i in range(2, 30)]
        + ["palavra raríssima zumthor aparece aqui na linha trinta."]
    )
    c.tool("rlm_load_data", {"name": f"{QA_PREFIX}doc", "data": doc})

    ok, t, _ = c.tool("rlm_search_index", {"var_name": f"{QA_PREFIX}doc", "terms": ["alvo"]})
    r.check("search", "BM25: termo na linha 1 exibe L1 (1-indexed, nunca L0)",
            ok and re.search(r"\bL1\b", t) and "L0" not in t, t[:200])

    # BM25 indexa todo o vocabulário e cita o INÍCIO do segmento (pode ser
    # antes da linha do termo) — checar hit + linha dentro do range do doc
    ok, t, _ = c.tool("rlm_search_index", {"var_name": f"{QA_PREFIX}doc", "terms": ["zumthor"]})
    mline = re.search(r"\bL(\d+)\b", t)
    r.check("search", "termo raro fora do vocabulário default é achado (BM25 full-vocab)",
            ok and mline and 1 <= int(mline.group(1)) <= 30, t[:200])

    ok, t, _ = c.tool("rlm_search_index",
                      {"var_name": f"{QA_PREFIX}doc", "terms": ["primeira frase deste"]})
    r.check("search", "frase literal (substring legacy)", ok and ("L1" in t or "1 hits" in t or "hits" in t),
            t[:200])

    ok, t, _ = c.tool("rlm_search_index",
                      {"var_name": f"{QA_PREFIX}doc", "terms": ["alvo", "documento"],
                       "require_all": True})
    r.check("search", "require_all (interseção na mesma linha)", ok and ("Linha 1" in t or "L1" in t or "lines" in t),
            t[:200])

    ok, t, _ = c.tool("rlm_search_index",
                      {"var_name": f"{QA_PREFIX}doc", "terms": ["linha"], "max_results": 5})
    shown = re.search(r"(\d+) (?:shown|hits)", t)  # normal usa "shown"; compact usa "hits"
    r.check("search", "max_results cap respeitado",
            ok and shown and int(shown.group(1)) <= 5, t[:200])

    ok, t, _ = c.tool("rlm_search_index", {"var_name": f"{QA_PREFIX}doc", "terms": [".*"]})
    r.check("search", "termo com regex metachars não explode",
            ok or "Erro" in t, t[:150])

    ok, t, _ = c.tool("rlm_search_index", {"var_name": f"{QA_PREFIX}doc", "terms": []})
    r.check("search", "terms vazio → resposta controlada", True if ok or t else False, t[:120])

    ok, t, _ = c.tool("rlm_search_index", {"var_name": "var_que_nao_existe_qa", "terms": ["x"]})
    r.check("search", "var inexistente → erro limpo", not ok and ("não encontrada" in t or "nao encontrada" in t.lower() or "Erro" in t), t[:150])

    ok, t, _ = c.tool("rlm_search_index",
                      {"var_name": f"{QA_PREFIX}doc", "terms": ["alvo"], "mode": "hybrid"})
    r.check("search", "hybrid responde (com ou sem embeddings, sem 500)",
            ok and ("hybrid" in t or "keyword" in t), t[:200])


def sec_collection(c: RlmClient, r: Reporter):
    print("\n[collection] Ciclo de coleção + rastreabilidade var:linha")
    v1 = "\n".join([f"v1 conteudo da linha {i} aqui." for i in range(1, 13)])
    v2_lines = [f"v2 filler {i} nada de especial." for i in range(1, 9)]
    v2_lines[4] = "v2 rubrica unica: jaspion melancolico agg ao anoitecer."  # linha 5
    v2 = "\n".join(v2_lines)
    v3 = "v3 primeira com jaspion tambem.\nv3 segunda linha final."
    for name, data in ((f"{QA_PREFIX}c1", v1), (f"{QA_PREFIX}c2", v2), (f"{QA_PREFIX}c3", v3)):
        c.tool("rlm_load_data", {"name": name, "data": data})

    ok, t, _ = c.tool("rlm_collection", {"action": "create", "name": QA_COLLECTION,
                                         "description": "harness QA — pode recriar"})
    r.add("collection", "create (ou já existe de run anterior)", PASS if ok or "exist" in t.lower() else FAIL, t[:120])

    ok, t, _ = c.tool("rlm_collection", {"action": "add", "name": QA_COLLECTION,
                                         "vars": [f"{QA_PREFIX}c1", f"{QA_PREFIX}c2", f"{QA_PREFIX}c3"]})
    r.check("collection", "add 3 vars + índice combinado", ok and "Índice combinado" in t, t[:200])

    ok, t, _ = c.tool("rlm_collection", {"action": "rebuild", "name": QA_COLLECTION})
    r.check("collection", "rebuild", ok and "reconstruído" in t, t[:150])

    ok, t, _ = c.tool("rlm_collection", {"action": "info", "name": QA_COLLECTION})
    r.check("collection", "info lista as vars", ok and f"{QA_PREFIX}c2" in t, t[:200])

    # === Invariante de rastreabilidade (canário do P0 do line-mapping) ===
    ok, t, _ = c.tool("rlm_collection", {"action": "search", "name": QA_COLLECTION,
                                         "terms": ["jaspion"]})
    cited = re.findall(r"L(\d+): (.+)", t)
    blocks = re.findall(r"📄 (\S+):", t)
    r.check("collection", "search acha o termo nas 2 vars",
            ok and f"{QA_PREFIX}c2" in t and f"{QA_PREFIX}c3" in t, t[:300])

    # Confere CADA citação contra o conteúdo real da var (via execute)
    data_map = {f"{QA_PREFIX}c1": v1, f"{QA_PREFIX}c2": v2, f"{QA_PREFIX}c3": v3}
    trace_ok, trace_detail = True, []
    cur_var = None
    for raw_line in t.split("\n"):
        mvar = re.match(r"📄 (\S+):", raw_line.strip())
        if mvar:
            cur_var = mvar.group(1)
            continue
        mcit = re.search(r"L(\d+): (.+)", raw_line)
        if mcit and cur_var in data_map:
            ln, ctx = int(mcit.group(1)), mcit.group(2).strip().rstrip(".")
            real_lines = data_map[cur_var].split("\n")
            if ln < 1 or ln > len(real_lines):
                trace_ok = False
                trace_detail.append(f"{cur_var}:L{ln} fora do range")
            else:
                real = real_lines[ln - 1].strip()
                probe = ctx[:25]
                if probe and probe not in real:
                    trace_ok = False
                    trace_detail.append(f"{cur_var}:L{ln} citou {probe!r} mas linha real é {real[:40]!r}")
    r.check("collection", "RASTREABILIDADE: toda citação L#### bate com a linha real",
            trace_ok and bool(cited), "; ".join(trace_detail) or f"nenhuma citação parseada: {t[:200]}")

    # NOTA: BM25 cita o INÍCIO do segmento (pode agrupar várias linhas-frase),
    # então não se assume linha exata do termo aqui — a RASTREABILIDADE acima
    # (citação ↔ conteúdo real) é o invariante forte. Só exige c3:L1 (termo na
    # 1ª linha → segmento começa nela).
    r.check("collection", "c3 citado em L1 (termo na primeira linha)",
            any(f"{QA_PREFIX}c3" in b for b in blocks) and "L1:" in t, t[:300])

    ok, t, _ = c.tool("rlm_collection", {"action": "search", "name": QA_COLLECTION,
                                         "terms": ["jaspion melancolico anoitecer"], "snippet_len": 60})
    r.check("collection", "phrase-trap: fallback tokenizado com banner",
            ok and ("fallback" in t.lower() or "tokeniz" in t.lower() or "L5" in t), t[:250])

    # P1 2026-06-06: paginação GLOBAL por relevância — limit=1 mostra 1 hit
    # NO TOTAL (antes: 1 por bucket var→termo = 2 com jaspion em 2 vars)
    ok, t, _ = c.tool("rlm_collection", {"action": "search", "name": QA_COLLECTION,
                                         "terms": ["jaspion"], "limit": 1})
    n_citations = len(re.findall(r"L\d+:", t))
    r.check("collection", "ranking global: limit=1 → exatamente 1 citação no total",
            ok and n_citations == 1 and "relevância global" in t,
            f"citações={n_citations}: {t[:250]}")

    # P1 2026-06-06: termo quoted inexistente é filtro OBRIGATÓRIO no fallback
    # tokenizado. CUIDADO na construção: o 2º termo NÃO pode existir como
    # substring literal (senão o caminho exato casa e o fallback nem dispara —
    # 'jaspion anoitecer' não é substring de '...jaspion melancolico agg ao
    # anoitecer', mas os TOKENS casam → fallback ativa → quoted filtra → zero).
    ok, t, _ = c.tool("rlm_collection", {"action": "search", "name": QA_COLLECTION,
                                         "terms": ['"frase inexistente xyzqa"', "jaspion anoitecer"]})
    r.check("collection", "mixed-quoted: literal inexistente zera o fallback",
            ok and "Nenhum resultado" in t, t[:250])

    ok, t, _ = c.tool("rlm_collection", {"action": "search", "name": "colecao_inexistente_qa",
                                         "terms": ["x"]})
    r.check("collection", "coleção inexistente → erro limpo",
            (not ok) or "vazia" in t or "não existe" in t or "nao existe" in t.lower(), t[:150])

    ok, t, _ = c.tool("rlm_collection", {"action": "add", "name": QA_COLLECTION,
                                         "vars": ["var_fantasma_qa"]})
    r.check("collection", "add de var inexistente → erro limpo", not ok or "não" in t, t[:150])

    ok, t, _ = c.tool("rlm_collection", {"action": "delete", "name": "coll_fantasma_qa"})
    r.check("collection", "delete de coleção inexistente → erro limpo",
            (not ok) and ("não existe" in t or "nao existe" in t.lower()), t[:120])


def sec_concurrency(c: RlmClient, r: Reporter):
    print("\n[concurrency] Event loop livre durante execute lento")
    results = {}

    def slow():
        results["slow"] = c.execute("import time\ntime.sleep(6)\nprint('done-slow')", timeout=90)

    def health():
        time.sleep(1.5)
        oks = 0
        for _ in range(3):
            try:
                req = urllib.request.Request(c.base + "/health")
                with urllib.request.urlopen(req, timeout=4) as resp:
                    oks += 1 if resp.status == 200 else 0
            except Exception:
                pass
            time.sleep(1)
        results["health"] = oks

    def queued_list():
        time.sleep(1.5)
        st, p, _ = c.rpc("tools/list", timeout=60)
        results["list"] = st == 200 and p and "result" in p

    threads = [threading.Thread(target=f) for f in (slow, health, queued_list)]
    t0 = time.time()
    [t.start() for t in threads]
    [t.join(timeout=120) for t in threads]
    elapsed = time.time() - t0

    r.check("concurrency", "/health respondeu 3/3 DURANTE execute de 6s",
            results.get("health") == 3, f"health_ok={results.get('health')}")
    slow_ok, slow_t, _ = results.get("slow", (False, "", None))
    r.check("concurrency", "execute lento completou", slow_ok and "done-slow" in slow_t, slow_t[:120])
    r.check("concurrency", "tools/list serializado atrás do execute mas completa",
            results.get("list") is True, str(results.get("list")))
    r.check("concurrency", "sem deadlock (tudo < 60s)", elapsed < 60, f"{elapsed:.1f}s")


def sec_s3_tasks_help(c: RlmClient, r: Reporter, skip_s3: bool):
    print("\n[aux] S3 / tasks / help")
    if skip_s3:
        r.add("aux", "S3 (pulado por flag)", SKIP, "--skip-s3")
    else:
        ok, t, _ = c.tool("rlm_list_buckets")
        if ok:
            r.add("aux", "list_buckets", PASS, "")
        else:
            r.add("aux", "list_buckets falha", KNOWN,
                  "SA bucket-scoped não tem ListAllMyBuckets (incidente 03/05) — esperado")
        ok, t, _ = c.tool("rlm_list_s3", {"bucket": "claude-code", "limit": 5})
        r.check("aux", "list_s3 no bucket claude-code", ok, t[:150])

    ok, t, _ = c.tool("rlm_task", {"action": "list"})
    r.check("aux", "task list", ok, t[:120])

    ok, t, _ = c.tool("rlm_help")
    r.check("aux", "help geral", ok and len(t) > 200, f"len={len(t)}")

    ok, t, _ = c.tool("rlm_help", {"topic": "topico_inexistente_qa"})
    r.check("aux", "help de tópico inexistente → resposta controlada", ok or "Erro" in t, t[:120])

    ok, t, _ = c.tool("rlm_search_code", {"var_name": f"{QA_PREFIX}pycode", "query": "soma"})
    # var ainda não existe → erro limpo; depois carrega e busca de verdade
    c.tool("rlm_load_data", {"name": f"{QA_PREFIX}pycode",
                             "data": "def soma(a, b):\n    return a + b\n\nclass Calc:\n    def dobro(self, x):\n        return x * 2\n",
                             "data_type": "code"})
    ok, t, _ = c.tool("rlm_search_code", {"var_name": f"{QA_PREFIX}pycode", "query": "soma"})
    r.check("aux", "search_code acha a função (compact mostra só o símbolo)",
            ok and "soma" in t and "function" in t, t[:200])


def sec_embeddings(c: RlmClient, r: Reporter):
    """Invariante: TODO índice vetorial deve ter cobertura total (emb:X/Y, X==Y).

    Canário do bug do batching (2026-06-06, commit 705f249): lotes estouravam
    o cap de 300k tokens da OpenAI → vars ficavam com cobertura PARCIAL e
    SILENCIOSA (recode_protocolos_geral tinha 9/17586 — busca semântica
    enxergava 0,05% do texto sem nenhum aviso). O var_info agora expõe emb:X/Y
    e este harness exige X==Y.
    """
    print("\n[embeddings] Invariante de cobertura total do índice vetorial")

    # (a) Genérico: var própria >=100k chars → auto-embed no load → emb:N/N.
    # Frases variadas (não repetição pura) p/ chunks não-vazios e embeds reais.
    palavras = ["memoria", "protocolo", "paciente", "cognicao", "dieta",
                "toxina", "exame", "suplemento", "sono", "exercicio"]
    frases = [f"Registro {i}: o tema {palavras[i % 10]} aparece no contexto "
              f"clinico numero {i} com variacao {i * 7 % 101}."
              for i in range(2200)]
    big = "\n".join(frases)  # ~190k chars
    ok, t, _ = c.tool("rlm_load_data", {"name": f"{QA_PREFIX}embed", "data": big},
                      timeout=240)
    if not ok:
        r.add("embeddings", "load da var de embed", FAIL, t[:200])
        return
    m = re.search(r"Embedded \(?(\d+)", t)
    if not m:
        r.add("embeddings", "auto-embed no load (var >=100k)", SKIP,
              "sem 'Embedded' na resposta — serviço de embeddings desligado?")
    else:
        n_load = int(m.group(1))
        ok, t, _ = c.tool("rlm_var_info", {"name": f"{QA_PREFIX}embed"})
        mi = re.search(r"(?:emb:|Embeddings: )(\d+)/(\d+)", t) if ok else None
        r.check("embeddings", "var_info expõe emb:X/Y",
                mi is not None, t[:150])
        if mi:
            x, y = int(mi.group(1)), int(mi.group(2))
            r.check("embeddings", f"cobertura total na var do harness ({x}/{y})",
                    x == y and x == n_load and y > 0,
                    f"emb:{x}/{y}, load reportou {n_load} — parcial = regressão do batching")

    # (b) Corpus conhecido (production-aware): as 6 recode_* re-embedadas em
    # 2026-06-06. Ausentes num server novo → SKIP, não FAIL.
    knowns = ["recode_suplementos", "recode_casos", "recode_toxico",
              "recode_nutricao", "recode_protocolos_geral", "recode_diagnostico"]
    found, partial = 0, []
    for v in knowns:
        ok, t, _ = c.tool("rlm_var_info", {"name": v})
        if not ok or "não encontrada" in t:
            continue
        mi = re.search(r"(?:emb:|Embeddings: )(\d+)/(\d+)", t)
        if not mi:
            partial.append(f"{v}: sem emb:X/Y (índice vetorial sumiu?)")
            continue
        found += 1
        if mi.group(1) != mi.group(2):
            partial.append(f"{v}: emb:{mi.group(1)}/{mi.group(2)}")
    if found == 0 and not partial:
        r.add("embeddings", "corpus recode_* (ausente neste server)", SKIP, "")
    else:
        r.check("embeddings", f"cobertura total no corpus recode_* ({found} vars)",
                found > 0 and not partial, "; ".join(partial))


def sec_repertorio(c: RlmClient, r: Reporter):
    """Modo-repertório (rlm_repertorio) contra o corpus real.

    Production-aware: kent_repertorio ausente neste server → SKIP, não FAIL.
    Canários: rubrica conhecida ABANDONO/'sentimiento de' (AUR. grau 3,
    validada live 2026-06-06), rastreabilidade var:linha, ranking estável.
    """
    print("\n[repertorio] Repertorização homeopática (kent_repertorio)")
    ok, t, _ = c.tool("rlm_repertorio", {"action": "info"})
    if not ok and "não encontrada" in t:
        r.add("repertorio", "kent_repertorio ausente neste server", SKIP, "")
        return
    r.check("repertorio", "info responde com stats do índice",
            ok and "entries:" in t, t[:150])

    ok, t, _ = c.tool("rlm_repertorio",
                      {"action": "buscar_rubrica", "query": "abandono sentimiento"})
    found = ok and re.search(r"kent_repertorio:L\d+", t or "") and "sentimiento" in t.lower()
    r.check("repertorio", "buscar_rubrica acha ABANDONO/sentimiento de",
            bool(found), (t or "")[:200])
    r.check("repertorio", "AUR aparece em CAPS (grau 3) no resultado",
            ok and "AUR" in t, (t or "")[:200])

    m = re.search(r"kent_repertorio:L(\d+)", t or "")
    if not m:
        r.add("repertorio", "repertorizar (sem ID da busca)", SKIP, "")
        return
    ln = int(m.group(1))
    # rastreabilidade: a linha citada existe na var REAL e contém a rubrica
    ok2, t2, _ = c.execute(
        f"print(kent_repertorio.split(chr(10))[{ln} - 1][:120])")
    r.check("repertorio", f"citação L{ln} bate com o texto real da var",
            ok2 and "sentimiento" in t2.lower(), (t2 or "")[:150])

    ok3, t3, _ = c.tool("rlm_repertorio",
                        {"action": "buscar_rubrica", "query": "temor"})
    m3 = re.search(r"kent_repertorio:L(\d+)", t3 or "")
    if not m3:
        r.add("repertorio", "repertorizar (rubrica 'temor' não achada)", SKIP, "")
        return
    ids = [f"kent_repertorio:L{ln}", f"kent_repertorio:L{m3.group(1)}"]
    ok4, t4, _ = c.tool("rlm_repertorio", {"action": "repertorizar", "rubrics": ids})
    r.check("repertorio", "repertorizar devolve tabela score/cov",
            ok4 and "score" in t4 and "cov" in t4, (t4 or "")[:200])
    ok5, t5, _ = c.tool("rlm_repertorio", {"action": "repertorizar", "rubrics": ids})
    head4 = [l for l in (t4 or "").split("\n") if l.strip()][:5]
    head5 = [l for l in (t5 or "").split("\n") if l.strip()][:5]
    r.check("repertorio", "ranking estável entre runs",
            ok5 and head4 == head5, f"{head4} != {head5}")


def sec_cleanup(c: RlmClient, r: Reporter):
    print("\n[cleanup] Remoção das vars _qa_* e da coleção do harness")
    # delete da coleção (gap fechado 2026-06-06): cada run nasce limpo
    ok, t, _ = c.tool("rlm_collection", {"action": "delete", "name": QA_COLLECTION})
    r.check("cleanup", f"coleção '{QA_COLLECTION}' removida via API",
            ok and "removida" in t, t[:150])

    ok, t, _ = c.tool("rlm_list_vars", {"limit": 500})
    qa_vars = re.findall(rf"({re.escape(QA_PREFIX)}\w+)", t) if ok else []
    removed, failed = 0, []
    for v in sorted(set(qa_vars)):
        ok2, t2, _ = c.tool("rlm_clear", {"name": v})
        if ok2:
            removed += 1
        else:
            failed.append(f"{v}: {t2[:60]}")
    ok3, t3, _ = c.tool("rlm_list_vars", {"limit": 500})
    leftover = re.findall(rf"({re.escape(QA_PREFIX)}\w+)", t3) if ok3 else ["?"]
    r.check("cleanup", f"vars _qa_* removidas ({removed})",
            not leftover and not failed, f"sobraram={leftover} falhas={failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--key", default=None)
    ap.add_argument("--skip-s3", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="pula concorrência (seção mais lenta, ~10s)")
    args = ap.parse_args()

    key = resolve_key(args)
    c = RlmClient(args.base, key)
    r = Reporter()

    print(f"QA extensivo RLM MCP — alvo: {args.base} (auth: {'Bearer' if key else 'NENHUMA'})")
    t0 = time.time()

    sec_protocol(c, r)
    sec_data_tools(c, r)
    sec_execute(c, r)
    sec_search_single(c, r)
    sec_collection(c, r)
    if not args.quick:
        sec_concurrency(c, r)
    sec_s3_tasks_help(c, r, args.skip_s3)
    sec_embeddings(c, r)
    sec_repertorio(c, r)
    sec_cleanup(c, r)

    fails = r.summary()
    print(f"\n{c.calls} chamadas em {time.time() - t0:.1f}s")
    sys.exit(min(fails, 125))


if __name__ == "__main__":
    main()
