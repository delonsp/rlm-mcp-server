"""Testes do lockdown B2 (Landlock FS + seccomp rede) do sandbox do rlm_execute.

LINUX-ONLY na essência: Landlock/seccomp não existem no macOS → os testes de FS/
rede/shm são ``skip`` fora do Linux (e fora de um kernel com Landlock/seccomp
viável). Os testes de no-op/fallback rodam em qualquer plataforma (smoke local).

Os testes de FS/rede NÃO usam ``repl.execute`` de propósito: ``open``/``socket``
já são deny-listed na 1ª camada (AST), o que daria FALSO-POSITIVO. Em vez disso,
spawnam um processo FORKADO que chama ``apply_child_lockdown`` direto e então
tenta a operação real (``open``/``socket``), provando o isolamento no kernel.

Como ``tests/*`` é gitignored (exceto ``test_sandbox.py``), este arquivo não é
versionado — roda no container Linux durante a validação real.
"""

import os
import sys

import pytest

os.environ["RLM_SANDBOX_MODE"] = "subprocess"
os.environ.setdefault("RLM_EXECUTE_TIMEOUT", "30")

import rlm_mcp.sandbox_lockdown as L  # noqa: E402
import rlm_mcp.sandbox_worker as sw  # noqa: E402
from rlm_mcp.repl import SafeREPL  # noqa: E402

_IS_LINUX = sys.platform.startswith("linux")

# Capacidades reais do ambiente (no container: ambas True; no Mac: ambas False).
try:
    _LANDLOCK_OK = bool(_IS_LINUX and L.probe_landlock())
except Exception:
    _LANDLOCK_OK = False
try:
    _SECCOMP_OK = bool(_IS_LINUX and L.probe_seccomp_bpf())
except Exception:
    _SECCOMP_OK = False

_needs_landlock = pytest.mark.skipif(
    not _LANDLOCK_OK, reason="requer Landlock (kernel >= 5.13, ex.: container Linux)"
)
_needs_seccomp = pytest.mark.skipif(
    not _SECCOMP_OK, reason="requer seccomp-BPF instalável (ex.: container Linux)"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _forkserver():
    """Pré-aquece o forkserver uma vez (para os testes que usam run_sandboxed)."""
    try:
        sw.init_forkserver()
    except Exception:
        pass


@pytest.fixture()
def repl():
    return SafeREPL()


def _run_child(fn):
    """Fork; roda ``fn(write_fd)`` no filho. Retorna (exitcode, mensagem_do_pipe).

    O filho escreve uma mensagem curta no fd e retorna um exitcode (int). Exceções
    não tratadas viram exitcode 98 + mensagem ``EXC:...``.
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # filho
        os.close(r)
        ec = 99
        try:
            ec = fn(w)
        except BaseException as e:  # noqa: BLE001
            try:
                os.write(w, f"EXC:{type(e).__name__}:{e}".encode())
            except Exception:
                pass
            ec = 98
        finally:
            try:
                os.close(w)
            except OSError:
                pass
            os._exit(int(ec) if ec is not None else 0)
    # pai
    os.close(w)
    buf = b""
    while True:
        try:
            chunk = os.read(r, 65536)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
    os.close(r)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        ec = os.WEXITSTATUS(status)
    else:
        ec = -1
    return ec, buf.decode("utf-8", "replace")


# ===========================================================================
# No-op / fallback — rodam em QUALQUER plataforma (smoke local no Mac)
# ===========================================================================

def test_offmode_is_noop():
    st = L.apply_child_lockdown(mode="off")
    assert st.fs_applied is False and st.net_applied is False
    assert any("off" in r for r in st.reasons)


def test_status_as_dict_shape():
    st = L.apply_child_lockdown(mode="off")
    d = st.as_dict()
    assert set(d) == {"mode", "fs", "net", "abi", "reasons"}
    assert d["mode"] == "off"


def test_default_ro_paths_nonempty():
    paths = L._default_ro_paths()
    assert isinstance(paths, list) and paths
    assert len(paths) == len(set(paths))  # sem duplicatas


def test_seccomp_prog_structure_x86_64():
    """BPF x86_64: arch-mismatch e x32 saltam p/ KILL (NÃO ALLOW); rede → DENY(EPERM)."""
    prog = L._build_seccomp_prog(L._AUDIT_ARCH_X86_64, L._NET_SYSCALLS_X86_64)
    n = len(L._NET_SYSCALLS_X86_64)
    g = 1  # guard x32 ligado em x86_64
    assert len(prog) == n + g + 6
    allow_idx, deny_idx, kill_idx = len(prog) - 3, len(prog) - 2, len(prog) - 1
    # RETs terminais
    assert prog[allow_idx].code == L._BPF_RET_K and prog[allow_idx].k == L.SECCOMP_RET_ALLOW
    assert prog[deny_idx].k == (L.SECCOMP_RET_ERRNO | 1)
    assert prog[kill_idx].k == L.SECCOMP_RET_KILL_PROCESS
    # arch mismatch (instr 1): match → continua (jt=0); mismatch → KILL (NÃO ALLOW)
    assert prog[1].code == L._BPF_JEQ_K and prog[1].jt == 0
    assert 1 + 1 + prog[1].jf == kill_idx
    # guard x32 (instr 3): nr >= bit x32 → KILL
    assert prog[3].code == L._BPF_JGE_K and prog[3].k == L._X32_SYSCALL_BIT
    assert 3 + 1 + prog[3].jt == kill_idx
    # cada syscall de rede salta p/ DENY(EPERM)
    base = 3 + g
    for i in range(n):
        instr = prog[base + i]
        assert instr.code == L._BPF_JEQ_K
        assert (base + i) + 1 + instr.jt == deny_idx
    # fall-through (nenhum match) cai em ALLOW
    assert base + n == allow_idx


def test_seccomp_prog_structure_aarch64():
    """BPF arm64: sem guard x32; arch-mismatch → KILL; rede → DENY."""
    prog = L._build_seccomp_prog(L._AUDIT_ARCH_AARCH64, L._NET_SYSCALLS_AARCH64)
    n = len(L._NET_SYSCALLS_AARCH64)
    assert len(prog) == n + 6  # g=0
    kill_idx = len(prog) - 1
    assert all(instr.code != L._BPF_JGE_K for instr in prog)  # nenhum guard x32
    assert prog[3].code == L._BPF_JEQ_K  # instr 3 já é o 1º JEQ de syscall
    assert 1 + 1 + prog[1].jf == kill_idx
    assert prog[kill_idx].k == L.SECCOMP_RET_KILL_PROCESS


def test_invalid_mode_normalizes_to_warn():
    """Typo num modo de segurança NÃO pode virar fail-open: vira 'warn' c/ motivo, sem raise."""
    st = L.apply_child_lockdown(mode="requred")  # typo proposital de "required"
    assert st.mode == "warn"
    assert any("inválido" in r for r in st.reasons)


@pytest.mark.skipif(_IS_LINUX, reason="comportamento específico de não-Linux")
def test_required_raises_on_non_linux():
    with pytest.raises(L.LockdownError):
        L.apply_child_lockdown(mode="required")


@pytest.mark.skipif(_IS_LINUX, reason="comportamento específico de não-Linux")
def test_warn_degrades_on_non_linux():
    st = L.apply_child_lockdown(mode="warn")
    assert not st.active and st.reasons


# ===========================================================================
# Probes (sanity do ambiente) — container Linux
# ===========================================================================

@_needs_landlock
def test_probe_landlock_reports_abi():
    abi = L.probe_landlock()
    assert abi is not None and abi >= 1


@_needs_seccomp
def test_probe_seccomp_bpf_true():
    assert L.probe_seccomp_bpf() is True


# ===========================================================================
# FS — Landlock nega leitura fora da allowlist (forkado, open real)
# ===========================================================================

@_needs_landlock
def test_b2_denies_outside_read(tmp_path):
    # Pai (sem lockdown) cria um arquivo FORA da allowlist (tmp não está em /usr).
    probe = tmp_path / "secret.txt"
    probe.write_text("conteudo secreto")
    target = str(probe)

    def _child(w):
        L.apply_child_lockdown(mode="required", fs=True, net=False)
        try:
            with open(target, "rb") as f:
                f.read()
            os.write(w, b"OPENED")
            return 1
        except PermissionError:
            os.write(w, b"EPERM")
            return 0
        except OSError as e:
            os.write(w, f"OSERR:{e.errno}".encode())
            return 0 if e.errno in (1, 13) else 2

    ec, msg = _run_child(_child)
    assert ec == 0, f"esperava bloqueio, veio: ec={ec} msg={msg!r}"
    assert msg.startswith("EPERM") or msg.startswith("OSERR:13") or msg.startswith("OSERR:1")


@_needs_landlock
@pytest.mark.skipif(not os.path.isdir("/persist"), reason="/persist ausente (fora do container)")
def test_b2_denies_persist_read():
    probe = "/persist/.b2_probe"
    try:
        with open(probe, "w") as f:  # pai escreve (sem lockdown)
            f.write("sqlite-like secret")

        def _child(w):
            L.apply_child_lockdown(mode="required", fs=True, net=False)
            try:
                with open(probe, "rb") as f:
                    f.read()
                os.write(w, b"OPENED")
                return 1
            except (PermissionError, OSError) as e:
                os.write(w, f"DENIED:{getattr(e, 'errno', '?')}".encode())
                return 0

        ec, msg = _run_child(_child)
        assert ec == 0, f"/persist deveria ser negado; ec={ec} msg={msg!r}"
        assert msg.startswith("DENIED")
    finally:
        try:
            os.unlink(probe)  # pai limpa (sem lockdown)
        except OSError:
            pass


@_needs_landlock
@pytest.mark.skipif(not os.path.isdir("/data"), reason="/data ausente (fora do container)")
def test_b2_denies_data_listdir():
    def _child(w):
        L.apply_child_lockdown(mode="required", fs=True, net=False)
        try:
            os.listdir("/data")
            os.write(w, b"LISTED")
            return 1
        except (PermissionError, OSError) as e:
            os.write(w, f"DENIED:{getattr(e, 'errno', '?')}".encode())
            return 0

    ec, msg = _run_child(_child)
    assert ec == 0, f"/data listdir deveria ser negado; ec={ec} msg={msg!r}"
    assert msg.startswith("DENIED")


@_needs_landlock
def test_b2_stdlib_imports_still_work():
    mods = ("json", "re", "math", "collections", "statistics",
            "hashlib", "base64", "csv")

    def _child(w):
        # Força re-leitura do FS pós-lockdown p/ alguns leaf-modules.
        for m in ("csv", "statistics", "base64"):
            sys.modules.pop(m, None)
        L.apply_child_lockdown(mode="required", fs=True, net=False)
        failed = []
        for m in mods:
            try:
                __import__(m)
            except Exception as e:  # noqa: BLE001
                failed.append(f"{m}:{type(e).__name__}")
        os.write(w, ("OK" if not failed else "FAIL:" + ",".join(failed)).encode())
        return 0 if not failed else 1

    ec, msg = _run_child(_child)
    assert ec == 0, f"imports stdlib quebraram pós-lockdown: {msg!r}"
    assert msg == "OK"


@_needs_landlock
def test_b2_datetime_time_no_crash_under_lockdown():
    """tz lê /etc/localtime (fora da allowlist) → glibc cai p/ UTC SEM crashar."""
    def _child(w):
        import datetime
        import time
        L.apply_child_lockdown(mode="required", fs=True, net=False)
        try:
            datetime.datetime.now()        # usa localtime → /etc/localtime negado → UTC
            time.localtime()
            time.gmtime()
            os.write(w, b"OK")
            return 0
        except Exception as e:  # noqa: BLE001
            os.write(w, f"EXC:{type(e).__name__}:{e}".encode())
            return 1

    ec, msg = _run_child(_child)
    assert ec == 0, f"datetime/time crasharam sob lockdown: {msg!r}"
    assert msg == "OK"


# ===========================================================================
# Rede — seccomp nega socket novo (forkado, socket real)
# ===========================================================================

@_needs_seccomp
def test_b2_denies_new_socket():
    def _child(w):
        import socket
        L.apply_child_lockdown(mode="required", fs=False, net=True)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect(("1.1.1.1", 80))
            finally:
                s.close()
            os.write(w, b"CONNECTED")
            return 1
        except PermissionError as e:
            os.write(w, f"EPERM:{e.errno}".encode())
            return 0
        except OSError as e:
            os.write(w, f"OSERR:{e.errno}".encode())
            return 0 if e.errno == 1 else 2

    ec, msg = _run_child(_child)
    assert ec == 0, f"socket novo deveria ser negado; ec={ec} msg={msg!r}"
    assert "EPERM" in msg or msg.startswith("OSERR:1")


# ===========================================================================
# Fail-safe — required fecha; warn degrada (forkado, probe_landlock patchado)
# ===========================================================================

@pytest.mark.skipif(not _IS_LINUX, reason="lógica de fail-safe Linux-only")
def test_b2_required_fails_closed():
    def _child(w):
        L.probe_landlock = lambda: None  # simula kernel sem Landlock
        executed = False
        try:
            L.apply_child_lockdown(mode="required", fs=True, net=False)
            proceed = True
        except L.LockdownError:
            proceed = False
        if proceed:
            executed = True  # o worker rodaria o payload aqui
        os.write(w, b"EXEC" if executed else b"BLOCKED")
        return 0

    ec, msg = _run_child(_child)
    assert ec == 0 and msg == "BLOCKED", f"required deveria fechar; msg={msg!r}"


@pytest.mark.skipif(not _IS_LINUX, reason="lógica de fail-safe Linux-only")
def test_b2_warn_degrades_to_b1():
    def _child(w):
        L.probe_landlock = lambda: None  # simula kernel sem Landlock
        st = L.apply_child_lockdown(mode="warn", fs=True, net=False)
        ok = (not st.fs_applied) and bool(st.reasons)
        os.write(w, b"DEGRADED" if ok else b"UNEXPECTED")
        return 0 if ok else 1

    ec, msg = _run_child(_child)
    assert ec == 0 and msg == "DEGRADED", f"warn deveria degradar p/ B1; msg={msg!r}"


# ===========================================================================
# Integração — var grande via shm + llm_query no pai, com lockdown ativo
# ===========================================================================

@_needs_landlock
def test_b2_large_shm_still_works(repl):
    repl.sandbox_lockdown_mode = "required"  # falha-fechado se Landlock não engatar
    big = "x" * (8 * 1024 * 1024)  # 8 MB → vai via shared_memory (> threshold)
    with repl._execute_lock:
        repl.variables["big"] = big
    res = sw.run_sandboxed("n = len(big)", repl, 30)
    assert res.success, res.stderr
    assert repl.variables.get("n") == len(big)


@_needs_landlock
def test_b2_llm_query_still_parent(repl):
    repl.sandbox_lockdown_mode = "required"
    parent_pid = str(os.getpid())
    # Mocka o llm_client no PAI: retorna o PID do processo que executa a query.
    repl.llm_client.query = lambda *a, **k: parent_pid
    res = sw.run_sandboxed("pid = llm_query('quem executa isto?')", repl, 30)
    assert res.success, res.stderr
    # llm_query roda no PAI (pipe RPC, não socket) mesmo com a rede cortada no filho.
    assert repl.variables.get("pid") == parent_pid
