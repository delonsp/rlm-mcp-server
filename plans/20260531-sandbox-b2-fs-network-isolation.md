# Plano: Isolamento de FS + rede do sandbox do `rlm_execute` (B2)
Data: 2026-05-31
Spec de origem: sessão `/plan-code-codex b2`; resíduo honesto do B1 (`plans/20260530-sandbox-subprocess-isolation.md`); memórias `project-sandbox-isolation-plan`, `project-bug-sweep-2026-05`
Geração: plano do **Codex** (self-sandbox Landlock+seccomp) + **crítica do Claude** com contexto da sessão (gate de kernel, default `warn`, testes forkados, ABI-mask). Apêndices A/B no fim.

## Visão geral
B1 (subprocesso `forkserver`, shippado em `c4c733f`) fechou a CLASSE de RCE: o filho não tem credenciais (env scrubado), nem FDs/conexões herdadas, e morre confiável (`killpg`). Mas deixou um **resíduo honesto**: um escape que fura a deny-list AST ainda **lê** `/persist` (SQLite das vars) e `/data` (volume read-only) via filesystem, e pode **abrir rede nova**. B2 fecha esse resíduo aplicando, **dentro do processo-filho e antes de `exec(user)`**, **Landlock LSM** (allowlist mínima de FS) + **seccomp-BPF** (nega syscalls de rede) — ambos *unprivileged*, **sem mudar a postura do container**.

## Modelo de ameaça e o que B2 garante (framing honesto)
- **Gatilho:** prompt-injection de documento ingerido escreve código malicioso no `rlm_execute` (single-user, Bearer).
- **Assumimos que o código FURA a deny-list AST** (por isso isolamos no kernel, não no parser).
- **B2 garante (quando `required` e Landlock+seccomp engatam):** o filho **não lê** `/persist` nem `/data`, e **não abre socket novo**. A fronteira passa a ser o kernel (LSM + filtro de syscall), não a enumeração de primitivas Python.
- **B2 NÃO garante (resíduo explícito):**
  - Dados que o **pai já enviou** ao filho (as vars referenciadas pelo código) seguem em memória do filho — por design.
  - CPU/memória continuam mitigados por `RLIMIT`/deadline (B1), **não** por cgroup dedicado por-filho.
  - Bug de kernel/LSM/seccomp ou syscall não coberta.
  - Em `warn`/`off`, **reverte ao resíduo B1**.
  - Allowlist de FS ampla demais enfraquece o isolamento → manter mínima e testada.
- **Não vender B2 como sandbox de uso geral.** É fechamento do resíduo FS/rede do filho efêmero do `rlm_execute`.

## ⛔ PASSO 0 — GATE de pré-requisito (BLOQUEANTE, antes de qualquer código)
Landlock exige **kernel ≥ 5.13** E o **seccomp default do Docker** precisa permitir `landlock_create_ruleset/add_rule/restrict_self`. O kernel da VPS (191.252.92.235) é desconhecido. **Rodar o probe DENTRO do container** (compartilha o kernel do host, mas sofre o seccomp do Docker):

```bash
# na VPS, achar o container e checar kernel + docker
docker ps --format '{{.ID}}  {{.Image}}  {{.Names}}' | grep -i rlm
uname -r                       # >= 5.13 ?  (host == container, mesmo kernel)
docker version --format '{{.Server.Version}}'

# probe Landlock + NO_NEW_PRIVS DENTRO do container (ajuste <CID>):
docker exec -u rlm <CID> python - <<'PY'
import ctypes, os
libc = ctypes.CDLL(None, use_errno=True)
NR_LANDLOCK_CREATE_RULESET = 444  # x86_64 e arm64
LANDLOCK_CREATE_RULESET_VERSION = 1
abi = libc.syscall(NR_LANDLOCK_CREATE_RULESET, None, 0, LANDLOCK_CREATE_RULESET_VERSION)
print("kernel:", os.uname().release)
print("landlock_abi:", abi, "errno:", ctypes.get_errno() if abi < 0 else 0)
# NO_NEW_PRIVS (pré-req do seccomp self-install, deve dar 0):
print("set_no_new_privs:", libc.prctl(38, 1, 0, 0, 0))
PY
```

**Interpretação:**
- `landlock_abi >= 1` → Landlock OK (anote a ABI: define a máscara de `handled_access_fs`). **Verde → implementar.**
- `landlock_abi == -1, errno=38 (ENOSYS)` → kernel sem Landlock / Docker bloqueando o syscall. **Vermelho.**
- `landlock_abi == -1, errno=1 (EPERM)` → seccomp do Docker bloqueando. **Vermelho.**

**Branch vermelho:** o lado **FS** de B2 por este caminho é inviável sem mudar postura (bubblewrap/userns/cap) → decisão separada com o usuário. O lado **rede** (seccomp) provavelmente ainda funciona isolado (`set_no_new_privs:0` confirma) → pode-se shippar só o corte de rede e deixar FS como resíduo documentado.

## Requisitos rastreados
| ID | Requisito | Como será atendido | Arquivo(s) |
|----|-----------|---------------------|------------|
| R1 | Filho não lê `/persist` nem `/data` | Landlock: allowlist só `/usr`,`/lib`,`/lib64` (RO) + `/dev/shm` (RW existente); resto negado por default | `sandbox_lockdown.py`, `sandbox_worker.py` |
| R2 | Filho não abre rede nova | seccomp-BPF default-allow negando `socket/socketpair/connect/bind/listen/accept(4)/sendto/recvfrom/sendmsg/recvmsg` com `EPERM` | `sandbox_lockdown.py` |
| R3 | Não quebrar `/dev/shm` (var grande) | Landlock allow `/dev/shm` `READ_FILE\|WRITE_FILE\|READ_DIR` (`SharedMemory(create=False)` abre `O_RDWR`) | `sandbox_lockdown.py` |
| R4 | Stdlib continua importável (lazy) | Landlock RO em `/usr/local/lib/python3.12` + `lib-dynload` + `/usr/lib`,`/lib` (`.so` de `_hashlib`,`_csv`) | `sandbox_lockdown.py` |
| R5 | `llm_query` intacto | Já é proxy-pro-pai (`_serve_llm` no pai); cortar rede do filho não afeta o pai | (nenhuma mudança) |
| R6 | Aplicar **por-filho**, não no forkserver | Lockdown em `_sandbox_entry`, após namespace, antes de `exec` | `sandbox_worker.py` |
| R7 | Fail-safe operacional | Modos `required` (fail-closed) / `warn` (degrada p/ B1 + WARNING) / `off` (dev) | `repl.py`, `sandbox_worker.py` |
| R8 | Não mudar postura do container | ctypes + syscalls unprivileged; sem `cap_add`/`security_opt`/`--privileged` | (nenhuma mudança Docker obrigatória) |
| R9 | Zero deps novas | `ctypes` (stdlib) p/ Landlock e seccomp BPF | `sandbox_lockdown.py` |
| R10 | Assinatura dos 19 tools inalterada | `execute()` idem; handler não muda | (nenhuma) |
| R11 | Provar isolamento (não só deny-list) | Testes **white-box forkados** chamando `apply_child_lockdown` direto + `open`/`socket` reais | `tests/test_sandbox_lockdown.py` |

## O que NÃO estamos fazendo (anti-scope creep)
- **Não** mudar postura do container (cap/security_opt/userns/privileged) no caminho recomendado.
- **Não** adicionar dep (PyPI ou apt). Tudo `ctypes`.
- **Não** isolar CPU/mem via cgroup por-filho (segue `RLIMIT`/deadline do B1).
- **Não** mexer em `indexer.py`/`vector_index.py`/`persistence.py`/coleções/BM25.
- **Não** mudar o protocolo IPC do B1 nem a serialização de vars.
- **Não** persistir vars do execute (segue "como está").
- **Não** allowlistar rede por host/porta — corte total no filho (`llm_query` é no pai).

## Arquitetura

### Mecanismo: self-sandbox no filho (Landlock FS + seccomp rede), via ctypes
- **Ordem (crítica):** o `_sandbox_entry` já faz `setsid` → `_close_inherited_fds` → `_scrub_env` → `_apply_limits` → redirect stdout/err (linhas 377-392), e dentro do `try`: `_materialize_inputs` (lê `/dev/shm`) → `_build_namespace` (linhas 397-398). **O lockdown entra logo após a linha 399** (`originals = dict(...)`) e **antes** de `validate_code`/`exec` (403/410). Assim: dados legítimos já materializados, shm já lido, e só então a porteira fecha.
- **Landlock ANTES de seccomp** (Landlock precisa dos syscalls `landlock_*`; depois do seccomp eles poderiam ser barrados).
- **Landlock ABI-aware:** `landlock_create_ruleset(NULL,0,VERSION)` retorna a ABI; mascarar `handled_access_fs` apenas com os bits suportados por essa ABI (v1 base; v2 `REFER`; v3 `TRUNCATE`; v4 net rules; v5 `IOCTL_DEV`) — `handled_access` com bit não-suportado → `EINVAL`.
- **seccomp:** `prctl(PR_SET_NO_NEW_PRIVS,1)` → instalar filtro BPF **default-allow** que retorna `EPERM` (não `SIGSYS`-kill, p/ erro limpo `PermissionError`) para a família de syscalls de rede.

### Arquivos novos
- `src/rlm_mcp/sandbox_lockdown.py` — primitives Linux isoladas (não polui o worker com ctypes/syscall):
  - `class LockdownError(Exception)`
  - `@dataclass LockdownStatus` (fs_applied, net_applied, landlock_abi, mode, reasons)
  - `apply_child_lockdown(mode, fs=True, net=True) -> LockdownStatus` — orquestra; em `required` levanta `LockdownError` se algo falhar; em `warn` retorna status degradado.
  - `apply_landlock_fs(allow_ro, allow_rw) -> int` (ABI-mask + create_ruleset + add_rule por path + NNP + restrict_self).
  - `apply_seccomp_no_network()` (BPF program p/ negar rede com EPERM).
  - `probe_landlock() -> int|None` e `probe_seccomp_bpf() -> bool` (usados no gate/healthcheck/testes).
  - constantes de syscall (444/445/446, generic em x86_64/arm64), flags `LANDLOCK_ACCESS_FS_*`, `PR_SET_NO_NEW_PRIVS=38`, `PR_SET_SECCOMP`, `SECCOMP_MODE_FILTER`; helpers `prctl()`/`syscall()`/structs `ctypes` (`landlock_ruleset_attr`, `landlock_path_beneath_attr`, `sock_filter`/`sock_fprog`).
  - **Linux-only:** em `sys.platform != "linux"`, `apply_child_lockdown` no-op com status "unavailable" (p/ dev macOS).
- `tests/test_sandbox_lockdown.py` (gitignored, como o resto de `tests/` exceto `test_sandbox.py`) — ver Critérios.

### Arquivos modificados
- `src/rlm_mcp/sandbox_worker.py` — `_sandbox_entry`: inserir `apply_child_lockdown(...)` entre linha 399 e `validate_code` (403), dentro do `try`, com `except LockdownError` → `success=False` + stderr claro (em `required`). `_build_params`: enviar `lockdown_mode/lockdown_fs/lockdown_network`. Docstring do módulo: remover "resíduo B1 FS/rede" como estado atual e documentar B2.
- `src/rlm_mcp/repl.py` — novos envs perto da linha 517: `RLM_SANDBOX_LOCKDOWN` (**default `warn` no ship inicial**; flipar p/ `required` após probe live), `RLM_SANDBOX_FS_LOCKDOWN` (default true), `RLM_SANDBOX_NET_LOCKDOWN` (default true), opcional `RLM_SANDBOX_LOCKDOWN_DEBUG`. Expor como attrs no `SafeREPL` p/ `_build_params` ler.
- `Dockerfile` — (opcional, cosmético) `ENV PYTHONDONTWRITEBYTECODE=1`. **Não** é segurança; só evita tentativas de `.pyc` write pós-Landlock (que dariam EPERM benigno).
- `docker-compose.yml` — (após validar VPS) expor `- RLM_SANDBOX_LOCKDOWN=${RLM_SANDBOX_LOCKDOWN:-warn}` p/ poder flipar via env sem rebuild.
- `CLAUDE.md` — atualizar a linha de Segurança/sandbox: B2 fecha FS/rede do filho quando ativo; resíduo passa a ser "dados já em memória + sem cgroup por-filho".

### Allowlist Landlock mínima (proposta — validar nos testes)
- **RO** (`READ_FILE|READ_DIR|EXECUTE` mascarado por ABI): `/usr/local/lib/python3.12`, `/usr/local/lib/python3.12/lib-dynload`, `/usr/lib`, `/lib`, `/lib64`. (cobre stdlib `.py` + `.so` de extensões.)
- **RW** (`READ_FILE|WRITE_FILE|READ_DIR`): `/dev/shm`. **Sem** `MAKE_REG`/`REMOVE` (o pai é dono de criar/unlink; o filho só abre existente `O_RDWR`).
- **Negado por default:** `/persist`, `/data`, `/app`, `/tmp`, `/etc`, `/proc`, `$HOME`, e tudo o mais.

## Edge cases
- **`/dev/shm`** — `SharedMemory(create=False)` abre `O_RDWR` → precisa `WRITE_FILE`. Resource-tracker no exit do filho **pode** tentar unlink → negado pelo Landlock → erro **cosmético** no shutdown (filho já morrendo). Testar: não deve afetar o retorno de var grande.
- **Stdlib lazy** — `import csv`/`hashlib` pós-lockdown lê `/usr` → coberto pelo RO. C-extensões (`_hashlib.so`) → `lib-dynload`/`/usr/lib` RO+EXEC.
- **`/proc` pós-lockdown** — nada deveria ler `/proc` depois (o `_close_inherited_fds` lê `/proc/self/fd` ANTES, na linha 383). Se algum stdlib ler, teste pega → adicionar regra RO estreita.
- **Kernel sem Landlock** — `required` falha fechado (execute não roda, erro claro); `warn` degrada p/ B1 + WARNING; `off` break-glass dev.
- **seccomp × forkserver** — filtro **por-filho** em `_sandbox_entry`, nunca no template do forkserver (senão o pai/forkserver herdariam). Netos (`fork` do escape) herdam o filtro → não abrem rede.
- **`llm_query`** — pipe já existe antes do seccomp; `_serve_llm` roda no pai. Cortar `socket/connect` no filho não toca o pai.
- **macOS dev** — Landlock/seccomp Linux-only; `apply_child_lockdown` no-op + status unavailable; testes `skipif(sys.platform != "linux")`.

## Armadilhas (red herrings)
- **Não** colocar o lockdown no `init_forkserver`/preload — tem que ser por-filho.
- **Não** tentar allowlist de syscalls no seccomp (quebra o Python) — só **deny-list de rede** sobre default-allow.
- **Não** usar `SIGSYS`-kill no seccomp — `EPERM` p/ erro limpo.
- **Não** assumir `ALL_FLAGS` em `handled_access_fs` — mascarar pela ABI ou `EINVAL`.
- **`PYTHONDONTWRITEBYTECODE`** não é medida de segurança.
- **Testar FS via `repl.execute("open(...)")` é FALSO-POSITIVO** — prova a deny-list (open bloqueado), não o Landlock. Usar processo forkado chamando `apply_child_lockdown` direto.
- **Não** tocar `persistence.py`/`indexer.py`/`vector_index.py`.

## Critérios de verificação (success criteria)
**Local (macOS) — limitado:** só dá pra testar no-op/fallback + lógica de probe.
- [ ] `uv run python -c "import rlm_mcp.sandbox_lockdown, rlm_mcp.sandbox_worker"` — ok.
- [ ] `uv run ruff check src/` — sem erros novos.
- [ ] `uv run pytest tests/test_sandbox_lockdown.py -v` — testes Linux marcados `skip` no Mac; os de fallback/no-op passam.

**Linux/Docker (validação real — obrigatória):**
- [ ] Build + probe no container: `probe_landlock()>=1` e `probe_seccomp_bpf()==True`.
- [ ] `tests/test_sandbox_lockdown.py` (forkados, dentro do container Linux):
  - `test_b2_denies_persist_read` — fork → `apply_child_lockdown` → `open("/persist/<algo>")` real → `PermissionError`.
  - `test_b2_denies_data_read` — idem `/data` (read + listdir negados).
  - `test_b2_denies_new_socket` — fork → lockdown → `socket.socket(); s.connect(("1.1.1.1",80))` → `PermissionError`/`OSError(EPERM)`.
  - `test_b2_large_shm_still_works` — var ~5–20 MB via shm volta correta com lockdown ativo.
  - `test_b2_stdlib_imports_still_work` — `json,re,math,collections,statistics,hashlib,base64,csv` importam pós-lockdown.
  - `test_b2_llm_query_still_parent` — mock no pai; PID confirma execução no pai.
  - `test_b2_required_fails_closed` — monkeypatch `probe_landlock`→falha; `required` não executa payload.
  - `test_b2_warn_degrades_to_b1` — `warn` + Landlock indisponível → executa como B1 + WARNING.
- [ ] Regressão B1: `uv run pytest tests/test_sandbox.py -v` segue verde (lockdown não quebra B1).

**Live pós-deploy (VPS, via `/mcp`):**
- [ ] `rlm_execute` normal + `llm_query` dentro do execute + var grande (`recode_*`) + `/health` durante/depois.
- [ ] Confirmar nos logs que o lockdown engatou (status "active"); só então flipar `RLM_SANDBOX_LOCKDOWN=required`.

## Estado final desejado
Com `required` + Landlock/seccomp ativos: o código do `rlm_execute` roda num filho que **não consegue abrir `/persist`/`/data`** nem **criar socket**, mantendo `/usr` (stdlib) RO e `/dev/shm` (var grande) RW. `llm_query`, estado entre execs, vars grandes e embedding lazy de vars novas seguem idênticos. Pior caso de um escape: ler em memória os dados que o pai já lhe enviou. A fronteira de FS/rede passa a ser o **kernel**, não a enumeração de primitivas.

## Checklist de implementação (ordem)
0. [ ] **GATE** (Passo 0): probe na VPS. Verde → segue. Vermelho → reavaliar com o usuário. — dep: nenhuma (BLOQUEANTE)
1. [ ] `sandbox_lockdown.py`: constantes/structs ctypes + `prctl()`/`syscall()` helpers. — dep: 0
2. [ ] `sandbox_lockdown.py`: `apply_seccomp_no_network()` (BPF deny-rede→EPERM) + `probe_seccomp_bpf()`. — dep: 1
3. [ ] `sandbox_lockdown.py`: `apply_landlock_fs()` (ABI-mask, create_ruleset, add_rule por path, NNP, restrict_self) + `probe_landlock()`. — dep: 1
4. [ ] `sandbox_lockdown.py`: `apply_child_lockdown(mode,fs,net)` + `LockdownError`/`LockdownStatus`; no-op em não-Linux. — dep: 2,3
5. [ ] `repl.py`: envs `RLM_SANDBOX_LOCKDOWN`(default `warn`)/`_FS_`/`_NET_`; attrs no `SafeREPL`. — dep: nenhuma
6. [ ] `sandbox_worker.py`: `_build_params` envia config; `_sandbox_entry` chama `apply_child_lockdown` entre namespace e `exec`, `except LockdownError`. — dep: 4,5
7. [ ] `tests/test_sandbox_lockdown.py`: testes forkados FS/rede/shm/stdlib/llm/required/warn + `skipif` não-Linux. — dep: 6
8. [ ] Smoke local: import + ruff + pytest (no-op no Mac). — dep: 6,7
9. [ ] `docker-compose.yml` expõe `RLM_SANDBOX_LOCKDOWN`; (opc) `PYTHONDONTWRITEBYTECODE` no Dockerfile; `CLAUDE.md` atualizado. — dep: 6
10. [ ] Commit + push `main` (deploy Dokploy mata a sessão MCP). — dep: 8,9
11. [ ] Validação live (`/mcp`): execute/llm_query/var grande/health + logs confirmam lockdown. — dep: 10
12. [ ] Flipar `RLM_SANDBOX_LOCKDOWN=required` só após #11 verde. — dep: 11

## Notas para o implementador
- **Style:** convenções do `sandbox_worker.py`/`repl.py` (docstrings PT-BR, `dataclass`, logging `rlm-mcp.*`). Tudo stdlib (`ctypes`, `struct`, `os`).
- **Segurança inegociável:** lockdown é **por-filho**, **Landlock antes de seccomp**, seccomp **default-allow + deny-rede→EPERM**, `handled_access_fs` **mascarado pela ABI**.
- **Honestidade:** se o probe der vermelho, NÃO shippar `required`; degradar/documentar. Não vender B2 como sandbox de uso geral.
- **forkserver preload enxuto:** `sandbox_lockdown` só importa `ctypes`/`struct`/`os` — seguro pro preload do `sandbox_worker`.
- Validação real é Docker/VPS — **não roda no Mac**.

---

## Apêndice A — Plano original do Codex (resumo fiel)
**Abordagem:** self-sandbox no filho = Landlock LSM (FS) + seccomp-BPF (rede) via `PR_SET_NO_NEW_PRIVS`, em `_sandbox_entry` após `_build_namespace` e antes de `validate_code`/`exec`. Preserva postura (non-root `rlm`, seccomp default Docker, sem cap/privileged/security_opt). `llm_query` já é proxy-pro-pai → cortar rede do filho 100%, sem allowlist de host.
**Novos:** `sandbox_lockdown.py` (LockdownError/Status, apply_child_lockdown, apply_landlock_fs, apply_seccomp_no_network, probe_landlock, probe_seccomp_bpf, constantes/ctypes), `tests/test_sandbox_lockdown.py`, este plano.
**Modificados:** `sandbox_worker.py` (hook + `_build_params` config), `repl.py` (envs `RLM_SANDBOX_LOCKDOWN`/`_FS_`/`_NET_`, modos required/warn/off), Dockerfile (`PYTHONDONTWRITEBYTECODE` opc), docker-compose (`RLM_SANDBOX_LOCKDOWN`).
**Edge:** `/dev/shm` RO+WR preservado (O_RDWR); stdlib RO em `/usr...`+lib-dynload; kernel-sem-Landlock → fallback; seccomp por-filho (não forkserver); `llm_query` intacto.
**Alternativas rejeitadas:** bubblewrap (userns/apt/risco shm), namespaces unshare (CAP_SYS_ADMIN/userns), gVisor/Kata (muda runtime do container, excessivo).
**Verificação:** 7 testes + smoke Docker. **Resíduo pós-B2:** bugs de kernel; dados já em memória; sem cgroup por-filho; warn/off reverte; allowlist ampla enfraquece.
**Recomendação do Codex:** modo `required`. **(Divergência minha — ver Apêndice B item ⚠️#1.)**

## Apêndice B — Crítica do Claude (contexto da sessão)
**SÓLIDO (mantido):** self-sandbox Landlock+seccomp sem mudar postura; hook entre namespace e exec; seccomp por-filho default-allow→EPERM; `/dev/shm` RW + `/usr` RO; cortar rede 100% (llm_query no pai); rejeição de bubblewrap/namespaces/gVisor.
**ADAPTADO:** (1) testes de FS/rede = processos **forkados** chamando `apply_child_lockdown` direto, NÃO via `repl.execute` (open é deny-listed → falso-positivo); (2) `/usr` RO é o mecanismo load-bearing, preload é só otimização; (3) Landlock **ABI-masked** explícito; (4) lockdown dentro do `try` com `except LockdownError`.
**QUESTIONÁVEL (corrigido no consolidado):** (⚠️#1) **default `required` é perigoso** no modelo Dokploy (auto-deploy mata a sessão; se kernel<5.13 ou Docker bloqueia landlock_*, todo execute falha fechado e rollback é lento) → **default `warn`, flipar p/ `required` após probe live**; (2) `PYTHONDONTWRITEBYTECODE` não é segurança.
**FALTANDO (adicionado):** (🔴) **gate de kernel/Docker como Passo 0 bloqueante** (probe na VPS); (🔴) **B2 untestable localmente (darwin)** → validação Docker/VPS-only, testes `skipif` não-Linux; resource-tracker do shm no exit do filho (cosmético, testar); ordem **Landlock antes de seccomp**.
