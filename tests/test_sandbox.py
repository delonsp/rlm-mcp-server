"""Testes do isolamento do sandbox por subprocesso (sandbox_worker / B1).

Cobre:
  - migração dos 11 casos de regressão (7 exploits bloqueados + 4 legítimos);
  - isolamento: env-scrub, FD-close, RLIMIT, timeout fora da main thread, netos;
  - round-trip de dados (inline + shm), estado entre execs, concorrência;
  - proxy de llm_query (chave fica no pai);
  - PROTOCOLO HOSTIL (R3/R11): pickle malicioso, frame > cap, tag inválida,
    EOF no meio, 'done' sem os blobs prometidos.

O modo subprocess é forçado via env ANTES de qualquer SafeREPL ser criado.
Os "filhos hostis" de teste são spawnados com o contexto `fork` (herdam o
módulo de teste por memória, sem precisar reimportar), e rodados contra o
loop de serviço real do PAI (`sandbox_worker._service_loop`).
"""

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

os.environ["RLM_SANDBOX_MODE"] = "subprocess"
os.environ.setdefault("RLM_EXECUTE_TIMEOUT", "30")

from rlm_mcp.repl import SafeREPL
import rlm_mcp.sandbox_worker as sw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _forkserver():
    """Pré-aquece o forkserver uma vez por sessão."""
    sw.init_forkserver()


@pytest.fixture()
def repl():
    """SafeREPL fresca em modo subprocess."""
    return SafeREPL()


# ---------------------------------------------------------------------------
# Helpers para spawnar "filhos hostis" de teste (contexto fork)
# ---------------------------------------------------------------------------

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _run_evil_service(target, repl_obj, args=(), timeout=6):
    """Spawna `target` (filho hostil) e roda o loop de serviço real do pai."""
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=target, args=(child_conn,) + tuple(args), daemon=True)
    proc.start()
    child_conn.close()
    deadline = time.monotonic() + timeout
    try:
        return sw._service_loop(parent_conn, proc, repl_obj, deadline)
    finally:
        sw._kill_group(proc)
        try:
            proc.join(timeout=3)
        except Exception:
            pass
        try:
            parent_conn.close()
        except Exception:
            pass


# ---- alvos de filho hostil (module-level p/ serem herdados via fork) -------

def _evil_pickle_blob(conn, sentinel_path):
    """Envia 'done' + um pickle malicioso (__reduce__ -> os.system) como var."""
    import pickle
    import os as _os
    import time as _time

    class _Bomb:
        def __reduce__(self):
            return (_os.system, (f"touch {sentinel_path}",))

    blob = pickle.dumps(_Bomb())
    sw._send_json(conn, {"t": "done", "success": True, "stdout": "", "stderr": "",
                         "vars": [{"name": "pwned", "nbytes": len(blob)}], "rejected": []})
    conn.send_bytes(blob)
    _time.sleep(2)


def _evil_bad_tag(conn):
    import time as _time
    sw._send_json(conn, {"t": "TAG_INVALIDA", "x": 1})
    _time.sleep(3)


def _evil_eof(conn):
    conn.close()


def _evil_done_no_blob(conn):
    import time as _time
    sw._send_json(conn, {"t": "done", "success": True, "stdout": "", "stderr": "",
                         "vars": [{"name": "x", "nbytes": 500}], "rejected": []})
    _time.sleep(5)  # promete um blob e nunca envia


def _grandchild_target(conn):
    """setsid + fork de um neto que dorme; reporta o pid do neto ao teste."""
    import time as _time
    os.setsid()
    pid = os.fork()
    if pid == 0:
        _time.sleep(60)
        os._exit(0)
    conn.send(pid)  # filho de teste CONFIÁVEL → send() (pickle) é ok aqui
    _time.sleep(60)


def _fdtest_child(conn):
    """Abre um arquivo temp, fecha FDs herdados, reporta se o fd sumiu."""
    import tempfile
    f = tempfile.TemporaryFile()
    fd = f.fileno()
    sw._close_inherited_fds(keep={conn.fileno()})
    try:
        os.fstat(fd)
        closed = False
    except OSError:
        closed = True
    conn.send(("closed", closed))


# ---------------------------------------------------------------------------
# 1) Migração dos 11 casos de regressão
# ---------------------------------------------------------------------------

EXPLOITS = [
    ("attrgetter", 'import operator\nx=operator.attrgetter("__class__")("")'),
    ("str.format dunder", 'y="{0.__class__}".format("")'),
    ("getattr aliasing", 'g=getattr\nc=g("", "__class__")'),
    ("functools.partial(getattr)",
     'import functools\np=functools.partial(getattr,"","__class__")\np()'),
    ("string.Formatter", 'import string\nstring.Formatter().vformat("{0.__class__}", [""], {})'),
    ("help() introspect", 'help(str)'),
    ("dunder direto", 'x="".__class__'),
]

LEGIT = [
    ("f-string", 'x="hello"\ny=f"{x} world"\nprint(len(y))'),
    ("json", 'import json\nprint(json.dumps({"a":1}))'),
    ("comprehension", 'd={i:i*2 for i in range(5)}\nprint(sum(d.values()))'),
    ("ciclo sem RecursionError", 'a=[1,2,3]\na.append(a)\nprint("ok cycle")'),
]


@pytest.mark.parametrize("label,code", EXPLOITS, ids=[e[0] for e in EXPLOITS])
def test_exploit_blocked(repl, label, code):
    r = repl.execute(code)
    assert not r.success
    assert "SecurityError" in r.stderr


@pytest.mark.parametrize("label,code", LEGIT, ids=[c[0] for c in LEGIT])
def test_legit_passes(repl, label, code):
    r = repl.execute(code)
    assert r.success, f"{label}: stderr={r.stderr!r}"


@pytest.mark.parametrize("mod", ["gzip", "zipfile", "tarfile", "os", "subprocess", "socket", "pathlib"])
def test_blocked_imports(repl, mod):
    """Task #13: gzip/zipfile/tarfile saíram do ALLOWED_IMPORTS (I/O de arquivo
    burlando open()); os/subprocess/socket/pathlib permanecem bloqueados."""
    r = repl.execute(f"import {mod}")
    assert not r.success
    assert "SecurityError" in r.stderr


# ---------------------------------------------------------------------------
# 2) Round-trip, estado, concorrência, proxy
# ---------------------------------------------------------------------------

def test_roundtrip_inline(repl):
    repl.load_data("texto", "linha 1\nlinha 2\nerro\n", "text")
    r = repl.execute('n = contar(texto, "linha")["total"]\nnovo = {"n": n, "lista": [1,2,3]}')
    assert r.success, r.stderr
    assert repl.variables["novo"] == {"n": 2, "lista": [1, 2, 3]}


def test_roundtrip_large_shm(repl):
    repl.load_data("base", "x" * (5 * 1024 * 1024), "text")
    r = repl.execute("dup = base[:1000] * 5\nln = len(base)")
    assert r.success, r.stderr
    assert repl.variables["ln"] == 5 * 1024 * 1024
    assert len(repl.variables["dup"]) == 5000


def test_shm_failure_falls_back_to_inline(repl, monkeypatch):
    """Se /dev/shm estiver cheio (Docker 64MB), input var grande vai via pipe."""
    def _boom(_blob):
        raise OSError("No space left on device (/dev/shm)")
    monkeypatch.setattr(sw, "_put_input_shm", _boom)
    repl.load_data("big2", "y" * (2 * 1024 * 1024), "text")
    r = repl.execute("ln2 = len(big2)")
    assert r.success, r.stderr
    assert repl.variables["ln2"] == 2 * 1024 * 1024


def test_state_between_executions(repl):
    assert repl.execute("a = 21").success
    r = repl.execute("b = a * 2")
    assert r.success
    assert repl.variables["b"] == 42


def test_non_data_value_rejected(repl):
    r = repl.execute("lam = lambda v: v + 1\nd = {'a': 1}")
    assert "lam" not in repl.variables          # função não atravessa
    assert repl.variables.get("d") == {"a": 1}  # dado atravessa
    assert "descartadas" in r.stderr


def test_module_not_treated_as_var(repl):
    r = repl.execute("import statistics\nm = statistics.mean([2, 4, 6])")
    assert r.success, r.stderr
    assert "statistics" not in repl.variables
    assert repl.variables.get("m") == 4


def test_concurrency_no_corruption(repl):
    def run(i):
        return repl.execute(f"v{i} = {i} * 10")
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(run, range(4)))
    assert all(r.success for r in results)
    for i in range(4):
        assert repl.variables.get(f"v{i}") == i * 10


def test_llm_query_proxied_in_parent(repl):
    parent_pid = os.getpid()
    rec = {}

    def fake(prompt, data=None, model=None, max_tokens=4096, temperature=0.0):
        rec["pid"] = os.getpid()
        rec["prompt"] = prompt
        return "RESPOSTA-MOCK"

    repl.llm_client.query = fake
    r = repl.execute("out = llm_query('oi'); print('R:', out)")
    assert r.success, r.stderr
    assert "R: RESPOSTA-MOCK" in r.stdout
    assert rec.get("pid") == parent_pid  # rodou NO PAI → chave nunca no filho


# ---------------------------------------------------------------------------
# 3) Timeout fora da main thread + limite de memória
# ---------------------------------------------------------------------------

def test_timeout_in_worker_thread(repl):
    """SIGALRM não funcionaria aqui; killpg do pai funciona."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(repl.execute, "while True:\n    pass", 2)
        r = fut.result(timeout=25)
    assert not r.success
    assert ("deadline" in r.stderr) or ("Timeout" in r.stderr)


@pytest.mark.skipif(sys.platform == "darwin", reason="RLIMIT_AS não é confiável no macOS")
def test_memory_limit_kills_child_parent_survives(repl):
    repl.sandbox_mem_mb = 1024
    r = repl.execute("b = bytearray(1500 * 1024 * 1024)")
    assert not r.success
    # o PAI sobrevive e continua funcional:
    r2 = repl.execute("ok = 2 + 2")
    assert r2.success and repl.variables.get("ok") == 4


# ---------------------------------------------------------------------------
# 4) Isolamento — white-box dos mecanismos
# ---------------------------------------------------------------------------

def test_scrub_env_removes_secrets():
    saved = dict(os.environ)
    try:
        os.environ["OPENAI_API_KEY"] = "secret-openai"
        os.environ["RLM_API_KEY"] = "bearer-secret"
        os.environ["MINIO_SECRET_KEY"] = "secret-minio"
        os.environ["DEEPSEEK_API_KEY"] = "secret-ds"
        os.environ.setdefault("PATH", "/usr/bin")
        sw._scrub_env()
        for leaked in ("OPENAI_API_KEY", "RLM_API_KEY", "MINIO_SECRET_KEY", "DEEPSEEK_API_KEY"):
            assert leaked not in os.environ, f"{leaked} vazou no env do filho!"
        assert "PATH" in os.environ  # runtime essencial preservado
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_close_inherited_fds_in_child():
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=_fdtest_child, args=(child_conn,), daemon=True)
    proc.start()
    child_conn.close()
    try:
        assert parent_conn.poll(timeout=10), "filho não respondeu"
        tag, closed = parent_conn.recv()
        assert tag == "closed" and closed is True
    finally:
        sw._kill_group(proc)
        proc.join(timeout=3)
        parent_conn.close()


def test_killpg_reaps_grandchildren():
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_grandchild_target, args=(child_conn,), daemon=True)
    proc.start()
    child_conn.close()
    try:
        assert parent_conn.poll(timeout=10), "filho não reportou pid do neto"
        gc_pid = parent_conn.recv()
        assert _alive(proc.pid) and _alive(gc_pid)
        sw._kill_group(proc)
        proc.join(timeout=5)
        # dá um instante para o init reapar o neto
        for _ in range(20):
            if not _alive(proc.pid) and not _alive(gc_pid):
                break
            time.sleep(0.2)
        assert not _alive(proc.pid), "filho sobreviveu ao killpg"
        assert not _alive(gc_pid), "NETO sobreviveu ao killpg (vazamento de processo)"
    finally:
        sw._kill_group(proc)
        parent_conn.close()


# ---------------------------------------------------------------------------
# 5) PROTOCOLO HOSTIL (R3/R11) — o pai trata o filho como adversário
# ---------------------------------------------------------------------------

def test_safe_unpickler_blocks_os_system():
    """Unit: _SafeUnpickler recusa global perigoso; aceita tipos de dados."""
    import pickle
    import os as _os

    class _Bomb:
        def __reduce__(self):
            return (_os.system, ("echo PWNED",))

    with pytest.raises(pickle.UnpicklingError):
        sw._safe_loads(pickle.dumps(_Bomb()))

    assert sw._safe_loads(pickle.dumps({"a": [1, 2], "b": "ok"})) == {"a": [1, 2], "b": "ok"}


def test_recv_json_enforces_cap():
    a, b = mp.Pipe(duplex=True)
    b.send_bytes(b'{"t":"llm"}' + b" " * 1000)
    with pytest.raises(sw._ProtocolViolation):
        sw._recv_json(a, max_bytes=50)


def test_recv_json_rejects_unknown_tag():
    a, b = mp.Pipe(duplex=True)
    b.send_bytes(json.dumps({"t": "evil"}).encode("utf-8"))
    with pytest.raises(sw._ProtocolViolation):
        sw._recv_json(a, max_bytes=10000, allowed_tags={"llm", "done"})


def test_hostile_pickle_blob_not_executed(repl, tmp_path):
    """O P0: filho envia pickle malicioso como valor de var → pai NÃO executa."""
    sentinel = tmp_path / "pwned.flag"
    outcome = _run_evil_service(_evil_pickle_blob, repl, args=(str(sentinel),))
    assert "pwned" not in outcome["result_vars"]   # var recusada pelo _SafeUnpickler
    assert not sentinel.exists()                   # os.system JAMAIS rodou no pai


def test_hostile_bad_tag(repl):
    outcome = _run_evil_service(_evil_bad_tag, repl)
    assert outcome["violation"] == "protocol"
    assert not outcome["success"]


def test_hostile_eof(repl):
    outcome = _run_evil_service(_evil_eof, repl)
    assert outcome["violation"] in ("eof", "died")
    assert not outcome["success"]


def test_hostile_done_without_blob(repl):
    """Pai não pode pendurar esperando um blob prometido que nunca chega."""
    t0 = time.monotonic()
    outcome = _run_evil_service(_evil_done_no_blob, repl, timeout=3)
    assert time.monotonic() - t0 < 12          # não pendurou
    assert "x" not in outcome["result_vars"]   # var fantasma não entrou
