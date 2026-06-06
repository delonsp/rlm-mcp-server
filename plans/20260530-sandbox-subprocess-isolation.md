# Plano: Isolamento do sandbox do `rlm_execute` por subprocesso (B1) — v2
Data: 2026-05-30
Spec de origem: sessão plan-code; memória `project-bug-sweep-2026-05` (RCE 🔴); decisões B1 + "manter persistência como está"
Revisão: **v2 pós-crítica convergente Codex + Gemini** (corrige furo P0 de RCE-reverso por pickle + 8 refinamentos genuínos)

## Visão geral
Substituir a fronteira de segurança do `rlm_execute` — hoje deny-list via AST no MESMO processo (`SafeREPL._validate_code` + `exec`) — por **isolamento de processo com trust assimétrico**: o código do usuário roda num processo-filho efêmero (`forkserver`) com env scrubado (sem API keys), FDs herdados fechados, `setrlimit`, sessão de processo própria, e timeout por `killpg`. O pai continua dono do `repl.variables`. **Ponto crítico de segurança:** o pai trata o filho como **hostil** — o canal de volta filho→pai **nunca executa código** (JSON p/ controle + unpickler restrito com allowlist de tipos p/ dados). `llm_query` é proxied pro pai. Fecha a *classe* de escape (segredos e conexões somem do processo do usuário) e mata 2 bugs estruturais (timeout-em-thread, guard de memória furável).

## Modelo de ameaça e o que este plano garante (framing honesto)
- **Gatilho realista:** prompt-injection de documento ingerido faz o assistente escrever código malicioso no `rlm_execute` (single-user, Bearer-gated).
- **Assumimos que o código FURA a deny-list AST** (é por isso que isolamos — a deny-list vira só 1ª camada barata).
- **B1 garante:** o processo do usuário **não tem** as credenciais (env scrubado) **nem** conexões vivas (FDs fechados); não consegue exfiltrar segredo nem reusar socket/conn do pai; é limitado em CPU/memória e morto de forma confiável (inclusive netos).
- **B1 NÃO garante (resíduo explícito):** um escape ainda pode **ler** `/persist` (SQLite das vars do próprio user) e `/data` (volume read-only do próprio user) via filesystem, e abrir rede nova (sem credencial). Isolar FS/rede exige namespaces/seccomp = **B2** (fora deste plano). **Não vender B1 como isolamento forte de FS/rede** — é isolamento de **segredo/credencial + controle de recurso**.

## Requisitos rastreados
| ID | Requisito | Como será atendido | Arquivo(s) |
|----|-----------|---------------------|------------|
| R1 | Fechar a CLASSE de escape | Fronteira no limite de processo; deny-list AST = defense-in-depth | `sandbox_worker.py`, `repl.py` |
| R2 | Segredos fora do alcance | Env scrub (clear + allowlist) **e** fechar FDs herdados (sockets/SQLite/minio) | `sandbox_worker.py` |
| R3 | **Canal de volta não executa código** (corrige RCE-reverso) | Trust assimétrico: filho→pai = JSON (controle) + **unpickler restrito allowlist-de-tipos** (dados); pai nunca faz `recv()`/`pickle.loads` cru de bytes do filho | `sandbox_worker.py` |
| R4 | Não quebrar killer feature | Pai dono do estado + merge in-memory; AST-selective shipping; `ALLOWED_IMPORTS` preservado no filho | `repl.py`, `sandbox_worker.py` |
| R5 | `llm_query` continua funcionando | Proxy via canal de controle; budget no pai | `sandbox_worker.py`, `repl.py` |
| R6 | Timeout fora da main thread + netos | `killpg` no deadline (sem `signal`); filho em `setsid()` próprio | `sandbox_worker.py` |
| R7 | Limite de memória/CPU | `RLIMIT_AS` (backstop generoso) + `RLIMIT_CPU`; wall-clock real no deadline do pai | `sandbox_worker.py` |
| R8 | Var grande sem deadlock de pipe | `shared_memory` p/ payloads > threshold (evita backpressure do buffer ~64KB) | `sandbox_worker.py` |
| R9 | Concorrência segura | Lock no pai serializando snapshot+merge de `repl.variables` (task workers são threads reais) | `repl.py`, `sandbox_worker.py` |
| R10 | Assinatura dos 19 tools inalterada | `SafeREPL.execute(code, timeout)` idem; handler `http_server.py:886` não muda | `repl.py`, `http_server.py` |
| R11 | Suíte cobre protocolo HOSTIL | Testes: pickle malicioso, msg gigante, tag inválida, EOF mid-LLM, `llm` após `done`, neto sobrevivente | `tests/test_sandbox.py` |
| R12 | Sem mudar postura do container | Sem `cap_add`/`security_opt`; forkserver+setrlimit como non-root sob seccomp default | (nenhuma mudança Docker) |
| R13 | Fallback de emergência marcado INSEGURO | `RLM_SANDBOX_MODE=inprocess` loga WARNING no startup; documentado como break-glass | `repl.py`, `http_server.py` |

## O que NÃO estamos fazendo (anti-scope creep)
- **B2** (namespaces/seccomp/bubblewrap, gVisor, Wasm/Wasmtime): muda postura do container e/ou o modelo de execução (mataria `llm_query`/Python arbitrário). Hardening futuro — **fora deste plano**. (Gemini sugeriu Wasm/gVisor; descartado por escopo, registrado como futuro.)
- **Persistir vars do `execute`**: "manter como está" — só replica o merge in-memory.
- **Mudar contrato/resposta dos tools** ou adicionar `isError`.
- **Tocar** `indexer.py`/`vector_index.py`/`persistence.py`/coleções/BM25.
- **Offload do `execute` pra fora do event loop**: já bloqueia hoje; subprocesso não piora (otimização futura).

## Arquitetura

### Modelo: pai dono do estado + filho efêmero (`forkserver`) + trust assimétrico
- **Pai** (uvicorn) mantém `repl.variables`/`variable_metadata` — os outros 18 tools não mudam.
- Cada `execute` → filho via `multiprocessing.get_context("forkserver")`. forkserver é **single-threaded** → elimina hazard de fork em processo com event loop + ThreadPoolExecutor vivos. (forkserver **não tem COW** das vars atuais → o pai envia as referenciadas; mitigado por AST-selective + `shared_memory` p/ grandes.)
- **Trust assimétrico** (a correção central da v2):
  - **pai→filho** (params, input_vars, llm_reply): pai é confiável → pickle normal OK; vars grandes via `shared_memory`.
  - **filho→pai** (llm_request, done-envelope, var values): filho é **HOSTIL** → **nunca** `pickle.loads` cru. Controle = **JSON** framed; valores de var = **`_SafeUnpickler`** (allowlist de tipos).

### Arquivos novos
- `src/rlm_mcp/sandbox_worker.py`:
  - **`_SafeUnpickler(pickle.Unpickler)`** — `find_class(module, name)` levanta `UnpicklingError` salvo se `(module,name)` ∈ `SAFE_GLOBALS` (tipos de dados: `builtins.{list,dict,set,frozenset,tuple,bytes,bytearray,complex,...}` implícitos + ex. `collections.OrderedDict/Counter/defaultdict`, `datetime.*`). Default-deny. Usado **só** no lado pai p/ desserializar bytes vindos do filho. Var de tipo não-allowlistado → descartada com aviso (extensão do trade-off "não-picklável descartada" → agora "não-tipo-de-dado descartada").
  - **`_send_json(conn, obj)` / `_recv_json(conn, max_bytes)`** — framing length-prefixed manual sobre `recv_bytes()`/`send_bytes()` (NUNCA `recv()`/`send()` que auto-pickla). Cap de tamanho; `_recv_json` valida que é JSON e que `t` ∈ tags permitidas.
  - **`_sandbox_entry(ctrl_conn, params)`** — alvo do filho:
    1. `os.setsid()` (sessão/grupo próprio p/ killpg).
    2. **Fechar FDs herdados**: enumerar `/proc/self/fd` (fallback `os.closerange(3, NOFILE)`), mantendo só os FDs do `ctrl_conn` + shm necessários. (forkserver não faz `exec` → `O_CLOEXEC` não dispara; fechar manual é obrigatório.)
    3. **Env scrub**: `os.environ.clear()` e restaurar allowlist runtime (`PATH,LANG,LC_*,TZ,SSL_CERT_FILE,SSL_CERT_DIR,TMPDIR,HOME`). Segredos nunca entram.
    4. `setrlimit(RLIMIT_AS, generoso)` + `setrlimit(RLIMIT_CPU, cpu_s)`.
    5. Redireciona stdout/stderr p/ `StringIO`.
    6. Materializa `input_vars` (pickle normal / lê de shm — direção confiável).
    7. Namespace: `_safe_builtins` + input_vars + helpers + stubs `llm_query/llm_stats/llm_reset_counter` (RPC via `_send_json`/`_recv_json` no `ctrl_conn`).
    8. `_validate_code(code)` (defense-in-depth).
    9. `exec(code, namespace)`.
    10. Coleta vars criadas/mudadas (regra do `repl.py:620-658`). Serializa cada uma com `pickle` (lado emissor); grande → `shared_memory`, pequena → bytes inline. Não-serializável → `rejected`.
    11. `_send_json(ctrl, {"t":"done", success, stdout, stderr, "vars":[{name, transport:"inline"|"shm", shm_name?, nbytes}], rejected})` e então envia os blobs de bytes (raw, framed) p/ os inline.
  - **`run_sandboxed(code, repl, timeout, mem_mb, cpu_s) -> ExecutionResult`** — orquestrador no pai:
    1. **Sob `repl._execute_lock`** (R9): `_referenced_vars(code, repl.variables)` (AST `ast.Name` ∩ keys) → monta `input_vars` (grandes via shm).
    2. `ctx.Process(target=_sandbox_entry, args=(child_ctrl, params))`; `ctrl = ctx.Pipe()`.
    3. **Loop de serviço** (`connection.wait([proc.sentinel, parent_ctrl], timeout=remaining)`), tratando filho como hostil:
       - `parent_ctrl` pronto → `_recv_json(parent_ctrl, MAX)`:
         - `{"t":"llm",...}` **e estado==RUNNING** → `repl.llm_client.query(...)` (pai tem key, enforce budget) → `_send_json(reply)`. `llm` após `done` ou nested inesperado → **violação** → killpg + erro.
         - `{"t":"done",...}` → lê os blobs inline (framed, com cap) + abre shm; **`_SafeUnpickler`** em cada → vars válidas; estado=DONE; sai.
         - tag desconhecida / JSON inválido / frame > cap → **violação** → killpg + erro.
       - `proc.sentinel` pronto sem `done` → filho morreu (OOM/segfault/kill); mapear por `exitcode`.
       - deadline estourado → `os.killpg` (SIGTERM→SIGKILL) → `ExecutionTimeoutError`.
    4. **Sob `repl._execute_lock`**: merge das vars no `repl.variables` + `VariableInfo` (size via `_estimate_size`, `max_var_size_mb` ainda aplicado aqui) + `access_count` + `_auto_cleanup()`.
    5. `finally`: `join()`, fechar ambos os ends do pipe, `unlink` das shm, reap (anti-zumbi).
    6. Retorna `ExecutionResult`.
  - **`init_forkserver()`** — lifespan: `ctx = mp.get_context("forkserver")`; `ctx.set_forkserver_preload(["rlm_mcp.sandbox_worker"])` (módulo enxuto, **sem** importar `openai`/`llm_client`/`minio`/clientes — evita herança de superfície); pré-aquece.

### Arquivos modificados
- `repl.py`: `execute()` mantém assinatura; delega p/ `run_sandboxed` se `RLM_SANDBOX_MODE=="subprocess"`, senão caminho in-process (fallback, SIGALRM só aqui). Adiciona `self._execute_lock = threading.Lock()`. Expõe helpers p/ o worker. Novos envs: `RLM_SANDBOX_MODE` (default `subprocess`), `RLM_SANDBOX_MEM_MB`, `RLM_SANDBOX_CPU_S`, `RLM_EXECUTE_TIMEOUT` (default 60), `RLM_SANDBOX_SHM_THRESHOLD` (default ~256KB).
- `llm_client.py`: sem mudança de API; `LLMClient` fica no pai; budget atravessa as chamadas proxied.
- `http_server.py`: `call_tool:886` não muda; lifespan chama `init_forkserver()`; se `RLM_SANDBOX_MODE==inprocess`, loga `WARNING: sandbox INSEGURO (in-process)`.
- `task_manager.py`: sem mudança; ganha timeout real (killpg) de graça.

### Protocolo IPC v2 (resumo)
```
pai→filho  (confiável): params/input_vars (pickle/shm); {"t":"llm_result"|"llm_error", ...} (JSON)
filho→pai  (HOSTIL):    {"t":"llm", prompt,data,model,max_tokens,temperature} (JSON, validado)
filho→pai  (HOSTIL):    {"t":"done", success,stdout,stderr, vars:[{name,transport,shm_name?,nbytes}], rejected} (JSON)
                        + blobs de bytes p/ vars inline → desserializados SÓ com _SafeUnpickler
```
Regras anti-hostil: framing length-prefixed (nunca `recv()`); cap de tamanho por frame e por blob; máquina de estados (`llm` só em RUNNING; nada após `done`); allowlist de tags; allowlist de tipos no unpickle; qualquer violação → `killpg` + `ExecutionResult(success=False)`.

### Timeout × llm_query (semântica documentada)
- `RLM_EXECUTE_TIMEOUT` (60s) = wall-clock, checado em fronteiras de mensagem. Uma chamada LLM **em voo** (feita pelo pai, com timeout próprio de 60s no `llm_client`) não é interrompida no meio → **o wall-time de um execute pode exceder o deadline em até ~1 timeout de LLM**. Aceitável (single-user); documentar.
- Loop **CPU-bound** puro → morto no deadline (`wait` expira → `killpg`); `RLIMIT_CPU` backstop.
- Hang em **I/O** (não consome CPU) → pego pelo deadline wall-clock + `killpg` (não pelo `RLIMIT_CPU`).

## Armadilhas (red herrings)
- `persistence_service.py`/`indexer.py`/`vector_index.py` — não tocar.
- **NÃO usar `Connection.recv()`/`send()`** com o filho (auto-pickla → reabre o RCE-reverso). Só `recv_bytes`/`send_bytes` + JSON/_SafeUnpickler.
- **NÃO usar `multiprocessing` start method `fork`** (deadlock multi-thread) — só `forkserver` via `get_context`.
- `O_CLOEXEC` **não** protege aqui (forkserver não faz `exec`) — fechar FDs manualmente.
- AST-selective é **correção** (miss → `NameError` claro), não segurança; não confundir.
- deny-list AST e `ALLOWED_IMPORTS` = defense-in-depth, **não** a fronteira.

## Critérios de verificação (success criteria)
- [ ] `uv run python -c "import rlm_mcp.http_server, rlm_mcp.sandbox_worker"` — ok.
- [ ] `uv run pytest tests/test_sandbox.py -v` — TODOS passam:
  - 7 exploits bloqueados + 4 legítimos passam (migrados de `/tmp/sandbox_regression.py`).
  - **env-scrub**: filho não vê `OPENAI/RLM_API_KEY/MINIO_*` (white-box).
  - **FDs fechados**: filho não tem FD herdado além do canal (checar `/proc/self/fd`).
  - **memória**: aloca > `RLIMIT_AS` → erro/kill, pai sobrevive.
  - **timeout em thread não-main**: loop infinito → morto no deadline.
  - **netos**: filho que dá `os.fork`/spawn (via stub de teste) → `killpg` mata o grupo, sem órfão.
  - **proxy llm_query**: mock no pai; key nunca no filho.
  - **round-trip**: var data (str/dict/list, inclusive ~5 MB via shm) volta; **não-tipo-de-dado** (lambda/obj custom) descartada com aviso.
  - **estado entre execuções**; **concorrência**: 2 executes simultâneos (threads) não corrompem `repl.variables` (lock).
  - **PROTOCOLO HOSTIL** (R11/R3): (a) filho envia **pickle malicioso** (`__reduce__→os.system`) como valor de var → `_SafeUnpickler` recusa, **pai NÃO executa**, var descartada; (b) frame > cap → violação tratada; (c) tag inválida → violação; (d) `llm` após `done` → violação; (e) EOF/pipe fechado mid-protocolo → erro limpo; (f) `done` sem blobs prometidos → erro limpo.
- [ ] `uv run ruff check src/` — sem erros novos.
- [ ] Manual (deploy): execute real sobre var grande (`recode_*`); `llm_query` dentro do execute; exploit contido; `/health` responde durante/depois.

## Estado final desejado
Código no `rlm_execute` roda num processo sem credenciais no env, sem FDs/conexões vivas herdadas, limitado em CPU/mem, morto de forma confiável (grupo inteiro). **O pai nunca executa nada vindo do filho** — o canal de volta só carrega JSON e dados de tipos allowlistados. Pior caso de um escape: ler os próprios dados em `/persist`/`/data` (resíduo B2). Do ponto de vista do cliente MCP, `rlm_execute` é idêntico (assinatura, resposta, estado entre execs, `llm_query`, vars grandes, embedding de vars novas). Os 7 exploits e a classe de introspecção param de ser exploráveis por construção, e o RCE-reverso por pickle (achado P0 da revisão) está fechado.

## Checklist de implementação (ordem)
1. [ ] `repl.py`: expor helpers (`_validate_code`,`_create_safe_builtins`,`HELPER_*`,`_estimate_size`,`_get_preview`,`_human_size`,`INTERNAL_FUNCTION_NAMES`); add `_execute_lock`; envs novos. — dep: nenhuma
2. [ ] `sandbox_worker.py`: `_SafeUnpickler` + `SAFE_GLOBALS` (allowlist de tipos). — dep: nenhuma
3. [ ] `sandbox_worker.py`: `_send_json`/`_recv_json` (framing + caps + validação de tag) e transporte shm (`_put_shm`/`_get_shm`). — dep: nenhuma
4. [ ] `sandbox_worker.py`: `_referenced_vars` (AST). — dep: nenhuma
5. [ ] `sandbox_worker.py`: `_sandbox_entry` (setsid, fechar FDs, env scrub, setrlimit, namespace, validate, exec, coleta+serialização inline/shm com skip, envia `done`). — dep: #1,#2,#3,#4
6. [ ] `sandbox_worker.py`: stubs `llm_query`/`llm_stats`/`llm_reset_counter` (RPC JSON). — dep: #3,#5
7. [ ] `sandbox_worker.py`: `run_sandboxed` (Process+Pipe, loop hostil-aware servindo llm/done/sentinel/deadline-killpg, `_SafeUnpickler` no retorno, merge sob lock, `finally` reap/unlink). — dep: #5,#6
8. [ ] `sandbox_worker.py`: `init_forkserver` (context+preload enxuto+pré-aquece). — dep: #5
9. [ ] `repl.py`: `execute` delega p/ `run_sandboxed` (modo subprocess); fallback inprocess. — dep: #7
10. [ ] `http_server.py`: lifespan chama `init_forkserver()`; WARNING se inprocess. — dep: #8
11. [ ] `tests/test_sandbox.py`: 11 casos migrados + isolamento (env/FD/mem/timeout-thread/netos/proxy/round-trip/estado/concorrência) + **protocolo hostil** (a–f). — dep: #7,#9
12. [ ] Smoke: `pytest`, `ruff`, import; medir latência de execute com var ~10 MB (sanity shm). — dep: #9,#11
13. [ ] (Opcional) remover `gzip`/`zipfile`/`tarfile` do `ALLOWED_IMPORTS` se não quebrar uso. — dep: nenhuma
14. [ ] Commit + push `main` (deploy Dokploy mata a sessão MCP). — dep: #12
15. [ ] Validação live pós-deploy (`/mcp`): execute, llm_query, exploit contido, var grande, `/health`. — dep: #14

## Notas para o implementador
- **Style**: convenções do `repl.py` (docstrings PT-BR, `dataclass`, logging `rlm-mcp.*`). Tudo stdlib (`multiprocessing`, `multiprocessing.shared_memory`, `resource`, `ast`, `pickle`, `json`, `os`). Sem dep nova.
- **Segurança inegociável (R3)**: o lado pai **jamais** chama `conn.recv()`, `pickle.load(s)` ou `pickle.Unpickler` (padrão) em bytes do filho. Só `_recv_json` (controle) e `_SafeUnpickler` (dados). Code review deve grep por `recv(`/`pickle.load` no caminho filho→pai.
- **forkserver preload enxuto**: `sandbox_worker` não pode importar transitivamente `openai`/`minio`/`llm_client`/clientes com side-effect (senão a superfície vaza pros filhos). Manter os imports do worker mínimos; `_validate_code` e helpers vêm de `repl` — garantir que `repl` import-level não puxe clientes pesados (se puxar, extrair helpers p/ módulo leve).
- **Trade-offs assumidos (v2)**:
  1. Sem COW → custo de shipping; mitigado por AST-selective + shm. Var referenciada que não exista → `NameError` claro (correção, não segurança).
  2. Só **tipos de dados** voltam do filho (`_SafeUnpickler`); lambdas/objetos custom/handles descartados com aviso — breaking change real vs hoje; documentar no `rlm_help`.
  3. `execute` bloqueia o event loop como hoje (offload = futuro).
  4. `RLIMIT_AS` (virtual) é backstop generoso, não limite fino; o per-var `max_var_size_mb` continua no merge do pai.
  5. Wall-time do execute pode exceder `RLM_EXECUTE_TIMEOUT` em até ~1 timeout de LLM (chamada em voo não é cortada).
  6. `inprocess` = INSEGURO (reabre o RCE original) — break-glass, com WARNING no startup.
  7. Resíduo B1: `/persist`+`/data` legíveis por um escape; só B2 fecha. **Não** vender como isolamento de FS/rede.
- Outro dev deve conseguir implementar a partir daqui sem perguntas.
