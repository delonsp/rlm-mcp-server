"""
Self-sandbox do processo-filho do rlm_execute (modo B2): Landlock FS + seccomp rede.

Este módulo fecha o RESÍDUO HONESTO do B1 (subprocesso ``forkserver``,
``sandbox_worker.py``): um escape que fura a deny-list AST ainda LÊ ``/persist``
(SQLite das vars) e ``/data`` (volume) pelo filesystem e pode ABRIR rede nova.
O B2 aplica, **dentro do processo-filho e ANTES do ``exec`` do código do usuário**:

  - **Landlock LSM** (FS): allowlist mínima — ``/usr...`` (stdlib/site-packages,
    RO) + ``/dev/shm`` (var grande, RW). Todo o resto (``/persist``, ``/data``,
    ``/app``, ``/etc``, ``$HOME``, ...) é negado por default. A fronteira de FS
    passa a ser o kernel, não a enumeração de primitivas Python.
  - **seccomp-BPF** (rede): filtro *default-allow* que retorna ``EPERM`` (não
    ``SIGSYS``-kill, p/ erro limpo) para a família de syscalls de rede
    (``socket``/``socketpair``/``connect``/``bind``/``listen``/``accept(4)``/
    ``sendto``/``recvfrom``/``sendmsg``/``recvmsg``). Sem ``socket()`` não há
    socket novo → nenhuma conexão de rede nova é possível.

Tudo via ``ctypes`` *unprivileged* (syscalls Landlock + ``prctl``/``seccomp``),
**sem mudar a postura do container** (sem ``cap_add``/``security_opt``/
``--privileged``). Zero deps novas.

ORDEM CRÍTICA: Landlock ANTES de seccomp (o seccomp não bloqueia os syscalls
``landlock_*``, mas mantemos a ordem do plano por robustez). Ambos exigem
``PR_SET_NO_NEW_PRIVS=1`` antes do ``restrict_self``/instalação do filtro.

O canal de controle pai↔filho (``Pipe(duplex=True)``) usa ``os.read``/``os.write``
no Unix (não ``send``/``recv``) → cortar a família de rede NÃO afeta ``llm_query``
(proxied pro pai) nem o retorno de blobs.

LINUX-ONLY: em ``sys.platform != "linux"`` (ex.: macOS de dev) ``apply_child_lockdown``
é no-op com status "indisponível". Validação real é só Docker/VPS.
"""

import ctypes
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("rlm-mcp.sandbox.lockdown")

_IS_LINUX = sys.platform.startswith("linux")
_DEBUG = os.getenv("RLM_SANDBOX_LOCKDOWN_DEBUG", "").strip().lower() in (
    "1", "true", "yes", "on",
)

# `ctypes` (layouts de struct) está disponível em qualquer plataforma; só as
# CHAMADAS a `libc` (syscalls Landlock/seccomp) são Linux-only. Assim a lógica de
# montagem do BPF (`_build_seccomp_prog`) é testável fora do Linux.
if _IS_LINUX:
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.syscall.restype = ctypes.c_long
    _libc.prctl.restype = ctypes.c_int
else:  # pragma: no cover - caminho de dev (macOS): structs ok, sem chamadas a libc
    _libc = None


# ============================================================================
# Constantes de syscall / flags (x86_64 e arm64)
# ============================================================================

# --- prctl / seccomp ---
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_KILL_PROCESS = 0x80000000  # mata o processo inteiro (kernel >= 4.14)
SECCOMP_DATA_NR_OFFSET = 0
SECCOMP_DATA_ARCH_OFFSET = 4
_EPERM = 1
# Bit do namespace de syscall x32 no x86_64 (nr = __X32_SYSCALL_BIT | nr_base).
_X32_SYSCALL_BIT = 0x40000000

# BPF clássico (struct sock_filter)
_BPF_LD = 0x00
_BPF_W = 0x00
_BPF_ABS = 0x20
_BPF_JMP = 0x05
_BPF_JEQ = 0x10
_BPF_JGE = 0x30
_BPF_RET = 0x06
_BPF_K = 0x00
_BPF_LD_W_ABS = _BPF_LD | _BPF_W | _BPF_ABS      # 0x20
_BPF_JEQ_K = _BPF_JMP | _BPF_JEQ | _BPF_K        # 0x15
_BPF_JGE_K = _BPF_JMP | _BPF_JGE | _BPF_K        # 0x35
_BPF_RET_K = _BPF_RET | _BPF_K                   # 0x06

# AUDIT_ARCH_* (linux/audit.h)
_AUDIT_ARCH_X86_64 = 0xC000003E
_AUDIT_ARCH_AARCH64 = 0xC00000B7

# Números de syscall de rede a negar — POR ARQUITETURA (diferem!).
# Lista do plano (R2): socket, socketpair, connect, bind, listen, accept,
# accept4, sendto, recvfrom, sendmsg, recvmsg.
_NET_SYSCALLS_X86_64 = (
    41,   # socket
    53,   # socketpair
    42,   # connect
    49,   # bind
    50,   # listen
    43,   # accept
    288,  # accept4
    44,   # sendto
    45,   # recvfrom
    46,   # sendmsg
    47,   # recvmsg
)
_NET_SYSCALLS_AARCH64 = (
    198,  # socket
    199,  # socketpair
    203,  # connect
    200,  # bind
    201,  # listen
    202,  # accept
    242,  # accept4
    206,  # sendto
    207,  # recvfrom
    211,  # sendmsg
    212,  # recvmsg
)

# --- Landlock (números iguais em x86_64 e arm64) ---
NR_LANDLOCK_CREATE_RULESET = 444
NR_LANDLOCK_ADD_RULE = 445
NR_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

# LANDLOCK_ACCESS_FS_* (linux/landlock.h)
LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13       # ABI v2
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14    # ABI v3
LANDLOCK_ACCESS_FS_IOCTL_DEV = 1 << 15   # ABI v5

# O_PATH p/ abrir diretórios como referência sem permissão de leitura de dados.
_O_PATH = getattr(os, "O_PATH", 0o10000000)

# Allowlist RW: só /dev/shm (var grande via SharedMemory abre O_RDWR).
_DEFAULT_RW_PATHS = ("/dev/shm",)


# ============================================================================
# Structs ctypes (só layouts — válidos em qualquer plataforma)
# ============================================================================

class _LandlockRulesetAttr(ctypes.Structure):
    """Só ``handled_access_fs``; passamos size=8 → kernel zera net/scoped."""
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    """struct landlock_path_beneath_attr (packed na UAPI)."""
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


class _SockFilter(ctypes.Structure):
    """struct sock_filter (instrução BPF clássica)."""
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    """struct sock_fprog (programa BPF passado ao seccomp)."""
    _fields_ = [
        ("len", ctypes.c_uint16),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


# ============================================================================
# Erros / status
# ============================================================================

class LockdownError(Exception):
    """Falha ao aplicar o lockdown (relevante em modo ``required``)."""
    pass


@dataclass
class LockdownStatus:
    """Resultado de ``apply_child_lockdown`` (vai no envelope ``done`` p/ o pai logar)."""
    mode: str
    fs_applied: bool = False
    net_applied: bool = False
    landlock_abi: Optional[int] = None
    reasons: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.fs_applied or self.net_applied

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "fs": self.fs_applied,
            "net": self.net_applied,
            "abi": self.landlock_abi,
            "reasons": list(self.reasons),
        }


# ============================================================================
# Helpers de allowlist
# ============================================================================

def _default_ro_paths() -> list:
    """Allowlist RO mínima: stdlib/site-packages + libs compartilhadas.

    Resolve o diretório real da stdlib via ``sysconfig`` (robusto a versão) e
    adiciona os paths fixos do ``python:3.12-slim`` + ``/usr/lib``/``/lib``/
    ``/lib64`` (``.so`` de extensões C e do linker dinâmico). Paths inexistentes
    são ignorados em ``add_rule``.
    """
    paths = []
    try:
        import sysconfig
        for key in ("stdlib", "platstdlib"):
            p = sysconfig.get_path(key)
            if p:
                paths.append(p)
                paths.append(os.path.join(p, "lib-dynload"))
    except Exception:
        pass
    paths += [
        "/usr/local/lib/python3.12",
        "/usr/local/lib/python3.12/lib-dynload",
        "/usr/lib",
        "/lib",
        "/lib64",
    ]
    seen, out = set(), []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ============================================================================
# Probes (queries sem efeito colateral — usados em gate/healthcheck/testes)
# ============================================================================

def probe_landlock() -> Optional[int]:
    """Retorna a ABI do Landlock (>=1) ou ``None`` se indisponível.

    ``landlock_create_ruleset(NULL, 0, VERSION)`` é uma query pura (não cria
    ruleset nem muda estado). errno=ENOSYS → kernel sem Landlock; EPERM → Docker
    bloqueando o syscall.
    """
    if not _IS_LINUX:
        return None
    ctypes.set_errno(0)
    abi = _libc.syscall(
        ctypes.c_long(NR_LANDLOCK_CREATE_RULESET),
        ctypes.c_void_p(0),
        ctypes.c_size_t(0),
        ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
    )
    if abi < 1:
        if _DEBUG:
            logger.warning("probe_landlock: abi=%s errno=%s", abi, ctypes.get_errno())
        return None
    return int(abi)


def probe_seccomp_bpf() -> bool:
    """Testa, num filho FORKADO, se conseguimos instalar o filtro seccomp.

    Instalar seccomp/``NO_NEW_PRIVS`` é irreversível → fazemos num fork descartável
    para não tocar o estado do processo-pai. Retorna True se o filho instalou e
    saiu 0.
    """
    if not _IS_LINUX:
        return False
    try:
        pid = os.fork()
    except OSError:
        return False
    if pid == 0:  # pragma: no cover - filho descartável
        try:
            apply_seccomp_no_network()
            os._exit(0)
        except BaseException:
            os._exit(1)
    try:
        _, status = os.waitpid(pid, 0)
    except OSError:
        return False
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


# ============================================================================
# seccomp — corte de rede (default-allow + deny-rede → EPERM)
# ============================================================================

def _network_syscalls():
    """(audit_arch, tupla de nrs de rede) para a arquitetura atual, ou (None, None)."""
    try:
        machine = os.uname().machine.lower()
    except Exception:
        return None, None
    if machine in ("x86_64", "amd64"):
        return _AUDIT_ARCH_X86_64, _NET_SYSCALLS_X86_64
    if machine in ("aarch64", "arm64"):
        return _AUDIT_ARCH_AARCH64, _NET_SYSCALLS_AARCH64
    return None, None


def _build_seccomp_prog(arch: int, syscalls) -> list:
    """Monta o BPF clássico: valida arch + bloqueia x32, nega rede com EPERM, senão ALLOW.

    Hardening (fechamento de bypass — fail-CLOSED no que o filtro não sabe interpretar):
      - arch != alvo → **KILL_PROCESS** (NÃO ALLOW): fecha evasão via ABI i386/compat,
        onde o mesmo nr tem semântica diferente sob outra arquitetura.
      - (x86_64) nr com bit x32 (>= 0x40000000) → **KILL_PROCESS**: fecha evasão x32
        (arch ainda é X86_64, mas o namespace de syscall difere → os nrs de rede
        "41/42/…" não baterão e escapariam).
      - syscall de rede → ERRNO(EPERM): erro limpo (→ PermissionError no Python), não kill.

    Layout (n = nº de syscalls; g = 1 se guard x32 [x86_64], senão 0; total = n+g+6):
      0:        LD  arch
      1:        JEQ arch, jt=0, jf=(n+g+3)          # arch != alvo → KILL
      2:        LD  nr
      [3]:      JGE x32bit, jt=(n+g+1), jf=0        # (x86_64) nr x32 → KILL
      base..:   JEQ scno_i, jt=(n-i), jf=0          # match rede → DENY(EPERM)
      base+n:   RET ALLOW
      base+n+1: RET ERRNO(EPERM)                    # DENY
      base+n+2: RET KILL_PROCESS                    # KILL
    """
    n = len(syscalls)
    block_x32 = arch == _AUDIT_ARCH_X86_64
    g = 1 if block_x32 else 0
    prog = [
        _SockFilter(_BPF_LD_W_ABS, 0, 0, SECCOMP_DATA_ARCH_OFFSET),
        # arch != alvo → KILL (fail-CLOSED, não ALLOW).
        _SockFilter(_BPF_JEQ_K, 0, n + g + 3, arch & 0xFFFFFFFF),
        _SockFilter(_BPF_LD_W_ABS, 0, 0, SECCOMP_DATA_NR_OFFSET),
    ]
    if block_x32:
        # nr >= 0x40000000 (bit x32) → KILL.
        prog.append(_SockFilter(_BPF_JGE_K, n + g + 1, 0, _X32_SYSCALL_BIT))
    for i, scno in enumerate(syscalls):
        prog.append(_SockFilter(_BPF_JEQ_K, n - i, 0, scno & 0xFFFFFFFF))
    prog.append(_SockFilter(_BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    prog.append(_SockFilter(_BPF_RET_K, 0, 0, SECCOMP_RET_ERRNO | (_EPERM & 0xFFFF)))
    prog.append(_SockFilter(_BPF_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS))
    return prog


def apply_seccomp_no_network() -> None:
    """Instala o filtro seccomp que nega criação/uso de sockets de rede (EPERM).

    Exige ``PR_SET_NO_NEW_PRIVS=1`` antes (instalação unprivileged). Levanta
    ``LockdownError`` em qualquer falha.
    """
    if not _IS_LINUX:
        raise LockdownError("seccomp indisponível fora do Linux")
    arch, syscalls = _network_syscalls()
    if arch is None:
        raise LockdownError(
            f"arquitetura não suportada p/ seccomp: {os.uname().machine!r}"
        )
    prog = _build_seccomp_prog(arch, syscalls)
    # Mantém filt/fprog vivos até depois do prctl (o kernel copia o filtro).
    filt = (_SockFilter * len(prog))(*prog)
    fprog = _SockFprog(len(prog), ctypes.cast(filt, ctypes.POINTER(_SockFilter)))

    ctypes.set_errno(0)
    if _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise LockdownError(f"PR_SET_NO_NEW_PRIVS falhou: errno={ctypes.get_errno()}")
    ctypes.set_errno(0)
    rc = _libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER,
                     ctypes.byref(fprog), 0, 0)
    if rc != 0:
        raise LockdownError(f"PR_SET_SECCOMP falhou: errno={ctypes.get_errno()}")


# ============================================================================
# Landlock — allowlist de FS
# ============================================================================

def _landlock_fs_mask(abi: int) -> int:
    """``handled_access_fs`` = todos os bits suportados pela ABI (resto → EINVAL)."""
    mask = (
        LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_WRITE_FILE
        | LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR
        | LANDLOCK_ACCESS_FS_REMOVE_DIR | LANDLOCK_ACCESS_FS_REMOVE_FILE
        | LANDLOCK_ACCESS_FS_MAKE_CHAR | LANDLOCK_ACCESS_FS_MAKE_DIR
        | LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_SOCK
        | LANDLOCK_ACCESS_FS_MAKE_FIFO | LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | LANDLOCK_ACCESS_FS_MAKE_SYM
    )
    if abi >= 2:
        mask |= LANDLOCK_ACCESS_FS_REFER
    if abi >= 3:
        mask |= LANDLOCK_ACCESS_FS_TRUNCATE
    if abi >= 5:
        mask |= LANDLOCK_ACCESS_FS_IOCTL_DEV
    return mask


def _landlock_add_path(ruleset_fd: int, path: str, access: int) -> int:
    """Adiciona uma regra path_beneath. Retorna 1 se adicionada, 0 se ignorada.

    Path inexistente (ex.: ``/lib64`` ausente) → ``open`` falha → ignora (não
    aborta o lockdown inteiro por um path opcional).
    """
    try:
        fd = os.open(path, _O_PATH | os.O_CLOEXEC)
    except OSError:
        return 0
    try:
        pb = _LandlockPathBeneathAttr(allowed_access=access, parent_fd=fd)
        ctypes.set_errno(0)
        rc = _libc.syscall(
            ctypes.c_long(NR_LANDLOCK_ADD_RULE),
            ctypes.c_long(ruleset_fd),
            ctypes.c_long(LANDLOCK_RULE_PATH_BENEATH),
            ctypes.c_void_p(ctypes.addressof(pb)),
            ctypes.c_uint(0),
        )
        if rc != 0:
            if _DEBUG:
                logger.warning("landlock add_rule %s falhou: errno=%s",
                               path, ctypes.get_errno())
            return 0
        return 1
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def apply_landlock_fs(allow_ro, allow_rw) -> int:
    """Aplica a allowlist de FS via Landlock e tranca o processo (``restrict_self``).

    ``allow_ro``: paths legíveis (READ_FILE|READ_DIR|EXECUTE).
    ``allow_rw``: paths leitura+escrita (+TRUNCATE se ABI>=3) — ``/dev/shm``.
    Retorna a ABI usada. Levanta ``LockdownError`` em falha.
    """
    if not _IS_LINUX:
        raise LockdownError("Landlock indisponível fora do Linux")
    abi = probe_landlock()
    if abi is None:
        raise LockdownError("Landlock indisponível (kernel < 5.13 ou bloqueado pelo Docker)")

    mask = _landlock_fs_mask(abi)
    attr = _LandlockRulesetAttr(handled_access_fs=mask)
    ctypes.set_errno(0)
    ruleset_fd = _libc.syscall(
        ctypes.c_long(NR_LANDLOCK_CREATE_RULESET),
        ctypes.c_void_p(ctypes.addressof(attr)),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint(0),
    )
    if ruleset_fd < 0:
        raise LockdownError(f"landlock_create_ruleset falhou: errno={ctypes.get_errno()}")

    try:
        ro_access = (
            LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR
            | LANDLOCK_ACCESS_FS_EXECUTE
        ) & mask
        rw_access = (
            LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE
            | LANDLOCK_ACCESS_FS_READ_DIR
        ) & mask
        if abi >= 3:
            rw_access |= LANDLOCK_ACCESS_FS_TRUNCATE

        granted = 0
        for path in allow_ro:
            granted += _landlock_add_path(ruleset_fd, path, ro_access)
        for path in allow_rw:
            granted += _landlock_add_path(ruleset_fd, path, rw_access)
        if granted == 0:
            raise LockdownError("nenhum path da allowlist pôde ser adicionado (Landlock abortado)")

        ctypes.set_errno(0)
        if _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise LockdownError(f"PR_SET_NO_NEW_PRIVS falhou: errno={ctypes.get_errno()}")
        ctypes.set_errno(0)
        rc = _libc.syscall(
            ctypes.c_long(NR_LANDLOCK_RESTRICT_SELF),
            ctypes.c_long(ruleset_fd),
            ctypes.c_uint(0),
        )
        if rc != 0:
            raise LockdownError(f"landlock_restrict_self falhou: errno={ctypes.get_errno()}")
    finally:
        try:
            os.close(ruleset_fd)
        except OSError:
            pass
    return abi


# ============================================================================
# Orquestrador — chamado pelo sandbox_worker no filho
# ============================================================================

def apply_child_lockdown(mode: str = "warn", fs: bool = True, net: bool = True,
                         ro_paths=None, rw_paths=None) -> LockdownStatus:
    """Aplica o lockdown B2 no processo-filho. Landlock (FS) ANTES de seccomp (rede).

    Modos:
      - ``required``: fail-closed — qualquer falha levanta ``LockdownError``
        (o worker não executa o código).
      - ``warn`` (default): degrada p/ B1 + acumula motivos em ``status.reasons``
        (o worker loga WARNING e segue).
      - ``off``: break-glass de dev — no-op.

    No-op (status "indisponível") em plataforma não-Linux; em ``required`` fora do
    Linux levanta ``LockdownError`` (não há como garantir o isolamento).
    """
    mode = (mode or "warn").strip().lower()
    # Defesa em profundidade: modo desconhecido (typo) NÃO pode virar fail-open
    # silencioso. Normaliza p/ 'warn' e registra o motivo (o repl.py já loga WARNING
    # alto na borda de config).
    if mode not in ("required", "warn", "off"):
        status = LockdownStatus(mode="warn")
        status.reasons.append(f"modo {mode!r} inválido → tratado como 'warn'")
        mode = "warn"
    else:
        status = LockdownStatus(mode=mode)

    if mode == "off":
        status.reasons.append("modo off (break-glass) — lockdown desativado")
        return status

    if not _IS_LINUX:
        msg = f"plataforma {sys.platform!r} sem Landlock/seccomp (no-op)"
        if mode == "required":
            raise LockdownError(f"modo required mas {msg}")
        status.reasons.append(msg)
        return status

    ro = list(ro_paths) if ro_paths is not None else _default_ro_paths()
    rw = list(rw_paths) if rw_paths is not None else list(_DEFAULT_RW_PATHS)

    # FS (Landlock) primeiro.
    if fs:
        try:
            status.landlock_abi = apply_landlock_fs(ro, rw)
            status.fs_applied = True
        except (LockdownError, OSError) as e:
            if mode == "required":
                raise LockdownError(f"lockdown FS falhou em modo required: {e}") from e
            status.reasons.append(f"FS não aplicado: {e}")

    # Rede (seccomp) depois.
    if net:
        try:
            apply_seccomp_no_network()
            status.net_applied = True
        except (LockdownError, OSError) as e:
            if mode == "required":
                raise LockdownError(f"lockdown de rede falhou em modo required: {e}") from e
            status.reasons.append(f"rede não cortada: {e}")

    return status
