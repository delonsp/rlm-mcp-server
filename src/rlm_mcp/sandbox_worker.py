"""
Isolamento do sandbox do rlm_execute por subprocesso (B1) + lockdown FS/rede (B2).

O código do usuário (potencialmente comprometido por prompt-injection de um
documento ingerido) roda num processo-filho EFÊMERO criado via ``forkserver``,
com:
  - env scrubado (sem OPENAI/MINIO/RLM_API_KEY) — não há credencial para exfiltrar;
  - FDs herdados fechados — não há socket/conexão viva do pai para reusar;
  - ``setrlimit`` (memória/CPU) e ``os.setsid()`` + ``killpg`` no deadline;
  - ``llm_query`` proxied para o pai (que é quem tem a chave LLM);
  - (B2) **lockdown por-filho ANTES do ``exec``**: Landlock (allowlist de FS) +
    seccomp (corte de rede), via ``sandbox_lockdown.apply_child_lockdown``.

PONTO CRÍTICO DE SEGURANÇA — *trust assimétrico*:
  - pai → filho (params/input_vars/llm_reply): o pai é confiável → pickle normal OK.
  - filho → pai (controle/valores de var): o filho é HOSTIL → o pai NUNCA executa
    código vindo dele. Controle = JSON (``_recv_json``); valores de var =
    ``_SafeUnpickler`` (allowlist de tipos de dados). O pai JAMAIS chama
    ``conn.recv()`` / ``pickle.load`` cru em bytes do filho (isso reabriria um
    RCE-reverso por ``__reduce__``).

A fronteira de segurança é o limite de processo + a desserialização restrita.
A deny-list AST (``repl.validate_code``) é só 1ª camada barata, defense-in-depth.

LOCKDOWN B2 (FS/rede): quando ativo (``RLM_SANDBOX_LOCKDOWN`` != off e em Linux),
o filho aplica Landlock + seccomp logo após materializar os inputs e montar o
namespace, ANTES de validar/exec o código. A partir daí um escape que fure a
deny-list AST NÃO LÊ ``/persist``/``/data`` nem abre socket novo. Resíduo pós-B2:
os dados que o pai já enviou seguem em memória do filho (por design); sem cgroup
dedicado por-filho (CPU/mem via RLIMIT/deadline do B1); em ``warn``/``off`` ou
kernel sem Landlock, reverte ao resíduo B1.
"""

import ast
import io
import json
import logging
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
import traceback
import types
from datetime import datetime
from multiprocessing import shared_memory
from multiprocessing.connection import wait

from .repl import (
    ExecutionResult,
    INTERNAL_FUNCTION_NAMES,
    SecurityError,
    VariableInfo,
    _buscar,
    _contar,
    _extrair_secao,
    _resumir_tamanho,
    create_safe_builtins,
    estimate_size,
    get_preview,
    human_size,
    validate_code,
)
from .sandbox_lockdown import LockdownError, apply_child_lockdown

logger = logging.getLogger("rlm-mcp.sandbox")


# ============================================================================
# Constantes de protocolo / limites anti-hostil
# ============================================================================

# Caps de tamanho de frame. Frames de controle são pequenos (llm_query trunca
# data a 100k chars antes de enviar). Caps protegem o pai de um filho hostil
# que tente OOM com um frame gigante.
_MAX_CTRL_BYTES = int(os.getenv("RLM_SANDBOX_MAX_CTRL_BYTES", str(8 * 1024 * 1024)))

# Tags que o pai aceita do filho (qualquer outra = violação de protocolo).
_CHILD_CTRL_TAGS = frozenset({"llm", "llm_stats", "llm_reset", "done"})
# Tags que o filho aceita do pai (respostas de RPC).
_PARENT_REPLY_TAGS = frozenset({"llm_result", "llm_error"})

# Módulos pré-importados no namespace (não são "variáveis do usuário").
_PRE_IMPORT_MODULES = ("re", "json", "math", "collections", "datetime")

# Env preservado no filho (todo o resto é apagado, incl. segredos).
_ENV_KEEP = frozenset({
    "PATH", "LANG", "TZ", "HOME", "TMPDIR",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "PYTHONHASHSEED", "PYTHONIOENCODING",
})
_ENV_KEEP_PREFIXES = ("LC_",)

# Allowlist de tipos que o pai aceita desserializar do filho. Default-deny:
# qualquer global fora desta lista levanta UnpicklingError em find_class.
# Tipos primitivos (int/float/str/bool/None) usam opcodes dedicados e NÃO
# passam por find_class — sempre seguros.
_SAFE_GLOBALS = frozenset({
    ("builtins", "list"), ("builtins", "dict"), ("builtins", "set"),
    ("builtins", "frozenset"), ("builtins", "tuple"), ("builtins", "bytes"),
    ("builtins", "bytearray"), ("builtins", "complex"),
    ("builtins", "int"), ("builtins", "float"), ("builtins", "str"),
    ("builtins", "bool"),
    ("collections", "OrderedDict"), ("collections", "Counter"),
    ("collections", "defaultdict"), ("collections", "deque"),
    ("datetime", "datetime"), ("datetime", "date"), ("datetime", "time"),
    ("datetime", "timedelta"), ("datetime", "timezone"),
})


class _ProtocolViolation(Exception):
    """Filho violou o protocolo (frame inválido/gigante/tag inesperada)."""
    pass


# ============================================================================
# (#2) Desserialização restrita — _SafeUnpickler (lado PAI, dados do filho)
# ============================================================================

class _SafeUnpickler(pickle.Unpickler):
    """Unpickler default-deny: só reconstrói tipos de DADOS allowlistados.

    Isto é o que fecha o RCE-reverso: um filho hostil que envie um pickle com
    ``__reduce__ -> os.system`` gera um opcode GLOBAL ('os','system') →
    find_class recusa → o pai NÃO executa nada.
    """

    def find_class(self, module, name):
        if (module, name) in _SAFE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"global proibido no retorno do sandbox: {module}.{name}"
        )


def _safe_loads(blob: bytes):
    """Desserializa bytes vindos do filho HOSTIL (allowlist de tipos)."""
    return _SafeUnpickler(io.BytesIO(blob)).load()


# ============================================================================
# (#3) Framing JSON + transporte (send_bytes/recv_bytes — NUNCA recv()/send())
# ============================================================================

def _send_json(conn, obj) -> None:
    """Envia um dict como JSON em um único frame (send_bytes, sem pickle)."""
    conn.send_bytes(json.dumps(obj).encode("utf-8"))


def _recv_json(conn, max_bytes: int, allowed_tags=None) -> dict:
    """Recebe um frame JSON validado.

    NUNCA usa ``recv()`` (que auto-despickla). Usa ``recv_bytes(maxlength=...)``,
    que levanta OSError se o frame exceder o cap (e fecha a conexão) — tratamos
    como violação. EOFError (peer fechou) propaga para o chamador.
    """
    try:
        raw = conn.recv_bytes(maxlength=max_bytes)
    except OSError as e:
        raise _ProtocolViolation(f"frame excede cap de {max_bytes} bytes: {e}")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise _ProtocolViolation(f"frame não é JSON válido: {e}")
    if not isinstance(obj, dict) or "t" not in obj:
        raise _ProtocolViolation("frame sem campo de tag 't'")
    if allowed_tags is not None and obj["t"] not in allowed_tags:
        raise _ProtocolViolation(f"tag inesperada: {obj['t']!r}")
    return obj


def _recv_blob(conn, max_bytes: int) -> bytes:
    """Recebe um blob de bytes (valor de var) com cap. Raw, sem desserializar."""
    return conn.recv_bytes(maxlength=max_bytes)


# ---- shared_memory: SÓ na direção pai→filho (pai cria + unlink; limpo) ------

def _put_input_shm(blob: bytes) -> shared_memory.SharedMemory:
    """Pai cria um segmento shm com `blob`. Pai mantém a ref para unlink depois."""
    shm = shared_memory.SharedMemory(create=True, size=max(1, len(blob)))
    shm.buf[: len(blob)] = blob
    return shm


def _read_input_shm(name: str, nbytes: int) -> bytes:
    """Filho abre shm por nome (create=False → não registra no tracker), lê, fecha."""
    shm = shared_memory.SharedMemory(name=name, create=False)
    try:
        return bytes(shm.buf[:nbytes])
    finally:
        shm.close()


# ============================================================================
# (#4) Seleção de variáveis referenciadas (AST) — só envia o necessário
# ============================================================================

def _referenced_vars(code: str, variables: dict) -> dict:
    """Retorna o subconjunto de `variables` cujos nomes aparecem no código.

    forkserver não tem COW das vars atuais → enviamos só as referenciadas
    (custo de shipping mínimo). Var referenciada inexistente vira NameError
    claro no filho (isto é CORREÇÃO, não segurança).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}  # validate_code reporta o erro de sintaxe depois
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return {k: variables[k] for k in names if k in variables}


# ============================================================================
# (#5/#6) Lado FILHO — entry point + stubs de RPC
# ============================================================================

def _close_inherited_fds(keep) -> None:
    """Fecha FDs herdados (exceto controle + std + resource_tracker).

    forkserver NÃO faz exec → O_CLOEXEC não dispara → fechamos manualmente.
    Defense-in-depth: o template do forkserver é criado cedo (antes do SQLite/
    minio do pai), então a superfície herdada já é mínima; isto garante.
    """
    keep = set(keep) | {0, 1, 2}
    # Preserva o fd do resource_tracker para que shm/locks ainda funcionem.
    try:
        from multiprocessing import resource_tracker
        rt = getattr(resource_tracker, "_resource_tracker", None)
        rfd = getattr(rt, "_fd", None)
        if rfd is not None:
            keep.add(rfd)
    except Exception:
        pass

    try:
        listed = os.listdir("/proc/self/fd")  # Linux (produção)
        fds = [int(x) for x in listed]
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError):
        # macOS / sem procfs → closerange sobre o limite soft de NOFILE
        import resource as _res
        soft, _ = _res.getrlimit(_res.RLIMIT_NOFILE)
        if soft == _res.RLIM_INFINITY or soft > 65536:
            soft = 65536
        for fd in range(3, soft):
            if fd in keep:
                continue
            try:
                os.close(fd)
            except OSError:
                pass
        return

    for fd in fds:
        if fd in keep:
            continue
        try:
            os.close(fd)
        except OSError:
            pass


def _scrub_env() -> None:
    """Apaga todo o env e restaura só a allowlist runtime (sem segredos)."""
    preserved = {
        k: v for k, v in list(os.environ.items())
        if k in _ENV_KEEP or any(k.startswith(p) for p in _ENV_KEEP_PREFIXES)
    }
    os.environ.clear()
    os.environ.update(preserved)


def _apply_limits(mem_mb: int, cpu_s: int) -> None:
    """Aplica RLIMIT_AS (backstop de memória virtual) + RLIMIT_CPU (backstop)."""
    try:
        import resource
    except ImportError:
        return
    if mem_mb and mem_mb > 0:
        nbytes = mem_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        except (ValueError, OSError):
            pass  # macOS frequentemente ignora RLIMIT_AS; wall-clock+killpg é o real
    if cpu_s and cpu_s > 0:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError):
            pass


def _make_llm_stub(conn, tag: str):
    """Fábrica de stubs llm_* que fazem RPC JSON para o pai pelo canal de controle."""
    def _rpc(extra=None):
        msg = {"t": tag}
        if extra:
            msg.update(extra)
        _send_json(conn, msg)
        reply = _recv_json(conn, _MAX_CTRL_BYTES, allowed_tags=_PARENT_REPLY_TAGS)
        if reply["t"] == "llm_error":
            raise RuntimeError(reply.get("error", "erro na sub-chamada LLM"))
        return reply.get("value")

    if tag == "llm":
        def llm_query(prompt, data=None, model=None, max_tokens=4096, temperature=0.0):
            # Espelha o truncamento do llm_client → mantém o frame de controle pequeno.
            if data is not None and len(data) > 100_000:
                data = data[:100_000] + "\n... [TRUNCADO]"
            return _rpc({
                "prompt": prompt, "data": data, "model": model,
                "max_tokens": max_tokens, "temperature": temperature,
            })
        return llm_query
    if tag == "llm_stats":
        return lambda: _rpc()
    if tag == "llm_reset":
        return lambda: _rpc()
    raise ValueError(tag)


def _build_namespace(input_vars: dict, conn) -> dict:
    """Monta o namespace de execução (builtins seguros + vars + helpers + stubs)."""
    ns = {"__builtins__": create_safe_builtins(), **input_vars}
    for mod in _PRE_IMPORT_MODULES:
        try:
            ns[mod] = __import__(mod)
        except ImportError:
            pass
    ns["llm_query"] = _make_llm_stub(conn, "llm")
    ns["llm_stats"] = _make_llm_stub(conn, "llm_stats")
    ns["llm_reset_counter"] = _make_llm_stub(conn, "llm_reset")
    ns["buscar"] = _buscar
    ns["contar"] = _contar
    ns["extrair_secao"] = _extrair_secao
    ns["resumir_tamanho"] = _resumir_tamanho
    return ns


def _collect_result_vars(ns: dict, originals: dict, max_var_mb: int):
    """Coleta vars novas/reatribuídas para devolver ao pai (regra de repl.execute).

    Espelha repl.py:620-658: ignora `_`-prefixadas, módulos pré-importados,
    helpers/internos; usa is-check para new/reassigned (mutação in-place de var
    pré-existente NÃO é detectada — idêntico ao comportamento atual). Funções/
    objetos custom serão rejeitados na serialização (só DADOS atravessam).
    """
    out, rejected = {}, []
    max_bytes = max_var_mb * 1024 * 1024
    for name, value in ns.items():
        if name.startswith("_"):
            continue
        if name in _PRE_IMPORT_MODULES:
            continue
        if name in INTERNAL_FUNCTION_NAMES:
            continue
        if isinstance(value, types.ModuleType):
            continue  # `import statistics` não é variável de dados
        # new OR reassigned (mesma semântica is-check do repl atual)
        if name in originals and originals[name] is value:
            continue
        size = estimate_size(value)
        if size > max_bytes:
            rejected.append(name)
            continue
        out[name] = value
    return out, rejected


def _materialize_inputs(params: dict) -> dict:
    """Reconstrói as input_vars no filho (direção CONFIÁVEL → pickle normal)."""
    input_vars = dict(params.get("input_inline", {}))
    for spec in params.get("input_shm", []):
        try:
            blob = _read_input_shm(spec["shm_name"], spec["nbytes"])
            input_vars[spec["name"]] = pickle.loads(blob)  # pai é confiável
        except Exception:
            pass  # var não materializada → NameError claro no código
    return input_vars


def _sandbox_entry(conn, params: dict) -> None:
    """Alvo do processo-filho. Isola, executa o código e devolve o resultado."""
    # 1. Sessão/grupo próprio → o pai pode killpg a árvore inteira (netos incluídos).
    try:
        os.setsid()
    except OSError:
        pass
    # 2. Fecha FDs herdados (mantém só o canal de controle + std + tracker).
    try:
        _close_inherited_fds(keep={conn.fileno()})
    except Exception:
        pass
    # 3. Apaga segredos do env.
    _scrub_env()
    # 4. Limites de recurso (backstops).
    _apply_limits(params.get("mem_mb", 0), params.get("cpu_s", 0))
    # 5. Redireciona stdout/stderr.
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    success = True
    result_vars, rejected = {}, []
    lockdown_summary = None
    try:
        input_vars = _materialize_inputs(params)
        ns = _build_namespace(input_vars, conn)
        originals = dict(input_vars)
        code = params["code"]
        # 5b. (B2) Lockdown FS/rede POR-FILHO — inputs já materializados (shm já
        #     lido), namespace já montado; só então a porteira fecha, ANTES de exec.
        proceed = True
        try:
            status = apply_child_lockdown(
                mode=params.get("lockdown_mode", "warn"),
                fs=params.get("lockdown_fs", True),
                net=params.get("lockdown_net", True),
            )
            lockdown_summary = status.as_dict()
        except LockdownError as e:
            # Só ocorre em modo 'required': fail-closed (não executa o código).
            sys.stderr.write(f"LockdownError: {e}\n")
            lockdown_summary = {"mode": params.get("lockdown_mode", "warn"),
                                "fs": False, "net": False, "abi": None,
                                "reasons": [f"required falhou: {e}"]}
            success = False
            proceed = False
        if proceed:
            # 6. validate (defense-in-depth — o pai já validou antes de spawnar).
            try:
                validate_code(code)
            except SecurityError as e:
                sys.stderr.write(f"SecurityError: {e}\n")
                success = False
            else:
                # 7. Executa o código do usuário.
                try:
                    exec(code, ns)
                except Exception as e:
                    sys.stderr.write(f"{type(e).__name__}: {e}\n")
                    sys.stderr.write(traceback.format_exc())
                    success = False
        # 8. Coleta vars de saída.
        result_vars, rejected = _collect_result_vars(
            ns, originals, params.get("max_var_mb", 50)
        )
    except Exception as e:  # erro catastrófico do próprio worker
        success = False
        try:
            sys.stderr.write(f"sandbox worker error: {type(e).__name__}: {e}\n")
        except Exception:
            pass

    stdout = sys.stdout.getvalue() if isinstance(sys.stdout, io.StringIO) else ""
    stderr = sys.stderr.getvalue() if isinstance(sys.stderr, io.StringIO) else ""

    # 9. Serializa as vars (lado emissor: pickle normal). Não-serializável → rejected.
    specs, blobs = [], []
    for name, value in result_vars.items():
        try:
            blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            rejected.append(name)
            continue
        specs.append({"name": name, "nbytes": len(blob)})
        blobs.append(blob)

    # 10. Envia o envelope `done` e então os blobs (raw), em ordem.
    try:
        _send_json(conn, {
            "t": "done", "success": success, "stdout": stdout, "stderr": stderr,
            "vars": specs, "rejected": rejected, "lockdown": lockdown_summary,
        })
        for blob in blobs:
            conn.send_bytes(blob)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ============================================================================
# (#7) Lado PAI — orquestrador hostil-aware
# ============================================================================

def _build_params(code: str, input_vars: dict, repl):
    """Monta os params para o filho. Var grande → shm (pai cria+unlink)."""
    inline, shm_specs, shms = {}, [], []
    threshold = getattr(repl, "sandbox_shm_threshold", 256 * 1024)
    for name, value in input_vars.items():
        if estimate_size(value) > threshold:
            try:
                blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                continue  # não enviável → NameError no filho
            try:
                shm = _put_input_shm(blob)
                shm_specs.append({"name": name, "shm_name": shm.name, "nbytes": len(blob)})
                shms.append(shm)
            except Exception:
                # /dev/shm cheio (default 64MB no Docker) ou indisponível →
                # fallback p/ inline (mp pickla via pipe; funciona em qualquer
                # tamanho, só sem o ganho de cópia única do shm).
                inline[name] = value
        else:
            inline[name] = value
    params = {
        "code": code,
        "input_inline": inline,
        "input_shm": shm_specs,
        "mem_mb": getattr(repl, "sandbox_mem_mb", 2048),
        "cpu_s": getattr(repl, "sandbox_cpu_s", 60),
        "max_var_mb": getattr(repl, "max_var_size_mb", 50),
        # (B2) config do lockdown FS/rede aplicado no filho (ver sandbox_lockdown).
        "lockdown_mode": getattr(repl, "sandbox_lockdown_mode", "warn"),
        "lockdown_fs": getattr(repl, "sandbox_lockdown_fs", True),
        "lockdown_net": getattr(repl, "sandbox_lockdown_net", True),
    }
    return params, shms


def _serve_llm(conn, repl, msg: dict) -> None:
    """Atende um RPC llm_* DO FILHO no pai (que tem a chave + enforça budget)."""
    t = msg["t"]
    try:
        if t == "llm":
            value = repl.llm_client.query(
                msg.get("prompt", ""),
                msg.get("data"),
                msg.get("model"),
                int(msg.get("max_tokens", 4096)),
                float(msg.get("temperature", 0.0)),
            )
        elif t == "llm_stats":
            value = repl.llm_client.get_stats()
        else:  # llm_reset
            repl.llm_client.reset_counter()
            value = None
        _send_json(conn, {"t": "llm_result", "value": value})
    except Exception as e:
        try:
            _send_json(conn, {"t": "llm_error", "error": str(e)})
        except Exception:
            pass


def _explain_exit(ec) -> str:
    """Mensagem legível para morte do filho sem `done`."""
    if ec is None:
        return "sandbox terminou em estado desconhecido\n"
    if ec < 0:
        sig = -ec
        if sig == signal.SIGKILL:
            return "sandbox morto (SIGKILL) — provável estouro de memória/timeout\n"
        if sig == getattr(signal, "SIGXCPU", -1):
            return "sandbox excedeu o limite de CPU (SIGXCPU)\n"
        if sig == signal.SIGSEGV:
            return "sandbox crashou (SIGSEGV)\n"
        try:
            return f"sandbox terminado pelo sinal {signal.Signals(sig).name}\n"
        except ValueError:
            return f"sandbox terminado pelo sinal {sig}\n"
    return f"sandbox terminou com código de saída {ec}\n"


def _kill_group(proc) -> None:
    """SIGKILL no grupo de processos do filho (reapa netos). Hostil-aware."""
    pid = proc.pid
    if pid is None or proc.exitcode is not None:
        return
    # O filho fez setsid() → pgid == pid. Matamos o grupo inteiro.
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        # grupo ainda não existe (setsid não rodou) → mata só o processo
        try:
            proc.kill()
        except Exception:
            pass
    except OSError:
        pass


def _log_lockdown(summary) -> None:
    """Loga o status do lockdown B2 reportado pelo filho (logging confiável do pai).

    O filho roda sob forkserver (logging não-confiável lá); por isso o status do
    Landlock/seccomp volta no envelope ``done`` e é logado AQUI, no pai. Degradação
    (modo warn sem Landlock/seccomp) → WARNING; lockdown ativo → INFO.
    """
    if not summary:
        return
    reasons = summary.get("reasons") or []
    if reasons:
        logger.warning("sandbox B2 lockdown degradado (mode=%s): %s",
                       summary.get("mode"), "; ".join(str(r) for r in reasons))
    else:
        logger.info("sandbox B2 lockdown ativo: fs=%s net=%s abi=%s mode=%s",
                    summary.get("fs"), summary.get("net"),
                    summary.get("abi"), summary.get("mode"))


def _service_loop(parent_conn, proc, repl, deadline: float) -> dict:
    """Loop de serviço do pai: atende llm, lê `done`, vigia morte e deadline.

    Trata o filho como HOSTIL: tags allowlistadas, frames com cap, máquina de
    estados (llm só em RUNNING; nada após done), desserialização restrita.
    """
    state = "RUNNING"
    stdout, stderr = "", ""
    max_blob = (getattr(repl, "max_var_size_mb", 50) + 64) * 1024 * 1024

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"success": False, "stdout": stdout,
                    "stderr": "ExecutionTimeoutError: deadline de execução excedido\n",
                    "result_vars": {}, "violation": "timeout"}
        try:
            ready = wait([proc.sentinel, parent_conn], timeout=remaining)
        except Exception as e:
            return {"success": False, "stdout": stdout,
                    "stderr": f"erro aguardando o sandbox: {e}\n",
                    "result_vars": {}, "violation": "wait_error"}
        if not ready:
            return {"success": False, "stdout": stdout,
                    "stderr": "ExecutionTimeoutError: deadline de execução excedido\n",
                    "result_vars": {}, "violation": "timeout"}

        # Prioriza drenar o canal de controle sobre o sentinel de morte.
        if parent_conn in ready:
            try:
                msg = _recv_json(parent_conn, _MAX_CTRL_BYTES,
                                 allowed_tags=_CHILD_CTRL_TAGS)
            except EOFError:
                return {"success": False, "stdout": stdout,
                        "stderr": "sandbox encerrou o canal sem enviar resultado (EOF)\n",
                        "result_vars": {}, "violation": "eof"}
            except _ProtocolViolation as e:
                return {"success": False, "stdout": stdout,
                        "stderr": f"violação de protocolo do sandbox: {e}\n",
                        "result_vars": {}, "violation": "protocol"}

            t = msg["t"]
            if t in ("llm", "llm_stats", "llm_reset"):
                if state != "RUNNING":
                    return {"success": False, "stdout": stdout,
                            "stderr": f"protocolo: '{t}' após 'done'\n",
                            "result_vars": {}, "violation": "llm_after_done"}
                _serve_llm(parent_conn, repl, msg)
                continue

            if t == "done":
                success = bool(msg.get("success", False))
                stdout = msg.get("stdout", "")
                stderr = msg.get("stderr", "")
                rejected = list(msg.get("rejected", []))
                specs = msg.get("vars", [])
                _log_lockdown(msg.get("lockdown"))
                rv = {}
                for spec in specs:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        stderr += "\n[timeout aguardando blobs de variáveis]\n"
                        break
                    # Vigia: blob prometido pode nunca chegar (filho hostil/morto).
                    r2 = wait([parent_conn, proc.sentinel], timeout=rem)
                    if parent_conn not in r2:
                        stderr += f"\n[blob de '{spec.get('name')}' não chegou]\n"
                        break
                    try:
                        blob = _recv_blob(parent_conn, max_blob)
                    except (EOFError, OSError) as e:
                        stderr += f"\n[falha lendo blob de '{spec.get('name')}': {e}]\n"
                        break
                    try:
                        rv[spec["name"]] = _safe_loads(blob)  # _SafeUnpickler!
                    except Exception:
                        rejected.append(spec.get("name"))
                if rejected:
                    nomes = ", ".join(str(r) for r in rejected if r)
                    stderr += (f"\n[vars descartadas (não-serializáveis ou tipo "
                               f"não permitido no retorno): {nomes}]\n")
                return {"success": success, "stdout": stdout, "stderr": stderr,
                        "result_vars": rv, "violation": None}

        # Canal sem dados, mas o processo morreu sem 'done'.
        elif proc.sentinel in ready:
            return {"success": False, "stdout": stdout,
                    "stderr": _explain_exit(proc.exitcode),
                    "result_vars": {}, "violation": "died"}


def run_sandboxed(code: str, repl, timeout_s: float,
                  mem_mb: int = None, cpu_s: int = None) -> ExecutionResult:
    """Executa `code` num processo-filho isolado e funde o resultado no pai.

    Substitui o caminho in-process de SafeREPL.execute quando
    RLM_SANDBOX_MODE=subprocess. Assinatura/resposta idênticas (ExecutionResult).
    """
    start = time.perf_counter()

    # Validação no PAI: fast-fail dos vetores conhecidos sem nem spawnar
    # (preserva exatamente o comportamento atual para os exploits conhecidos).
    try:
        validate_code(code)
    except SecurityError as e:
        return ExecutionResult(success=False, stdout="",
                               stderr=f"SecurityError: {e}", execution_time_ms=0.0)

    # Snapshot das vars referenciadas sob lock (task workers são threads reais).
    with repl._execute_lock:
        input_vars = _referenced_vars(code, repl.variables)

    params, shms = _build_params(code, input_vars, repl)
    ctx = _get_context()
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=_sandbox_entry, args=(child_conn, params), daemon=True)

    outcome = None
    try:
        proc.start()
        child_conn.close()  # o pai não usa a ponta do filho
        deadline = time.monotonic() + timeout_s
        outcome = _service_loop(parent_conn, proc, repl, deadline)
    except Exception as e:
        outcome = {"success": False, "stdout": "",
                   "stderr": f"Erro ao iniciar o sandbox: {type(e).__name__}: {e}\n",
                   "result_vars": {}, "violation": "spawn_error"}
    finally:
        # Saída limpa → join breve. Violação/timeout → mata o grupo já.
        if outcome is not None and outcome.get("violation") is None:
            try:
                proc.join(timeout=2)
            except Exception:
                pass
        if proc.is_alive():
            _kill_group(proc)
        try:
            proc.join(timeout=3)
        except Exception:
            pass
        try:
            child_conn.close()
        except Exception:
            pass
        try:
            parent_conn.close()
        except Exception:
            pass
        for shm in shms:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass

    elapsed = (time.perf_counter() - start) * 1000.0
    success = outcome["success"]
    stdout = outcome["stdout"]
    stderr = outcome["stderr"]
    variables_changed = []

    # Merge sob lock: bump de acesso (cleanup score) + aplica per-var
    # max_var_size_mb (backstop final) + metadata. Sempre sob lock (task
    # workers são threads reais) e o _auto_cleanup roda como no in-process.
    result_vars = outcome.get("result_vars") or {}
    with repl._execute_lock:
        now = datetime.now()
        # Vars referenciadas (lidas) bumpam access_count/last_accessed —
        # espelha repl._execute_inprocess (score de cleanup = recency × freq).
        for name in input_vars:
            meta = repl.variable_metadata.get(name)
            if meta is not None and name in repl.variables:
                meta.access_count += 1
                meta.last_accessed = now
        for name, value in result_vars.items():
            size = estimate_size(value)
            if size > repl.max_var_size_mb * 1024 * 1024:
                stderr += (f"\nVariavel '{name}' rejeitada: {human_size(size)} "
                           f"excede limite de {repl.max_var_size_mb}MB\n")
                success = False
                continue
            repl.variables[name] = value
            variables_changed.append(name)
            existing = repl.variable_metadata.get(name)
            repl.variable_metadata[name] = VariableInfo(
                name=name,
                type_name=type(value).__name__,
                size_bytes=size,
                size_human=human_size(size),
                preview=get_preview(value),
                created_at=existing.created_at if existing else now,
                last_accessed=now,
                access_count=(existing.access_count if existing else 0) + 1,
                pinned=existing.pinned if existing else False,
                source=existing.source if existing else "execute",
            )
        cleanup_info = repl._auto_cleanup()
    if cleanup_info:
        stdout += (f"\n[Auto-cleanup: removidas {cleanup_info['removed_count']} "
                   f"variáveis antigas, liberados {cleanup_info['removed_bytes_human']}]")

    repl.execution_count += 1
    return ExecutionResult(success=success, stdout=stdout, stderr=stderr,
                           execution_time_ms=elapsed,
                           variables_changed=variables_changed)


# ============================================================================
# (#8) Contexto forkserver + pré-aquecimento
# ============================================================================

_CTX = None
_PRELOADED = False


def _get_context():
    """Context forkserver (single-threaded → sem hazard de fork multi-thread)."""
    global _CTX
    if _CTX is None:
        _CTX = mp.get_context("forkserver")
        try:
            # Preload enxuto: só este módulo (puxa repl+llm_client, SEM openai/minio).
            _CTX.set_forkserver_preload(["rlm_mcp.sandbox_worker"])
        except Exception:
            pass
    return _CTX


def _prewarm_noop() -> None:
    pass


def init_forkserver() -> None:
    """Inicializa e pré-aquece o forkserver (chamar cedo no lifespan)."""
    global _PRELOADED
    if _PRELOADED:
        return
    ctx = _get_context()
    try:
        p = ctx.Process(target=_prewarm_noop)
        p.start()
        p.join(timeout=15)
        _PRELOADED = True
        logger.info("forkserver do sandbox pré-aquecido (preload: rlm_mcp.sandbox_worker)")
    except Exception as e:
        logger.warning(f"pré-aquecimento do forkserver falhou (seguirá lazy): {e}")
