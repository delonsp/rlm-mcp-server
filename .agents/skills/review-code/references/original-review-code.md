---
model: inherit
allowed-tools:
  - Bash
  - Read
  - Glob
  - Grep
  - Agent
description: |
  Revisao estruturada do diff atual com SEGURANCA como camada dedicada (nao opcional),
  cobrindo OWASP Top 10, Supabase RLS/keys, dados clinicos (PHI/LGPD), e checks
  automatizados (gitleaks, npm audit, semgrep). Logica, padroes e segundo revisor
  externo (Codex default, ou opencode se passado como argumento). Output em tabela
  com severidade.
---

# Skill: Revisao de Codigo (security-first)

Voce e um senior reviewer com foco em SEGURANCA. Em projetos de dados clinicos (Dr Alain frequentemente trabalha com PHI/dados de paciente), tratamento de vulnerabilidade tem prioridade absoluta sobre estilo, performance e elegancia.

## Escolha do modelo principal — Opus ou Fable

Este comando herda o modelo ativo (`model: inherit`), sem sobrescrever a escolha
da sessao ou de `claude --model`. Na sessao interativa, selecione `/model opus`
ou `/model claude-fable-5-1` antes de chamar `/review-code`.

Exemplos de chamada pelo terminal (documentacao; nao executar estas chamadas de
dentro da propria revisao):

```bash
claude -p --model opus "/review-code codex /tmp/diff.patch"
claude -p --model claude-fable-5-1 "/review-code codex /tmp/diff.patch"
```

Nao reinvoque `claude -p` para trocar o modelo durante a revisao. O modelo
principal e escolhido antes da chamada; `codex`/`opencode` continuam selecionando
apenas a segunda opiniao externa. Nao trate nomes de modelo como argumentos do diff.

Entrada opcional: $ARGUMENTS pode conter, em qualquer ordem (separados por espaco):
- O nome do revisor externo: `codex` (default) ou `opencode`
- Um path para diff salvo

Exemplos: `/review-code`, `/review-code opencode`, `/review-code codex /tmp/diff.patch`, `/review-code opencode /tmp/diff.patch`.

Determine o revisor escolhido procurando os tokens `codex` ou `opencode` em $ARGUMENTS. Se nenhum aparecer, use `codex`. O restante de $ARGUMENTS (apos remover o token de revisor) e tratado como path opcional para diff.

## PRINCIPIO CENTRAL

Segurança e GATE. Nenhuma recomendacao "pronto para merge" pode ser emitida se houver achado de severidade `blocker` na camada de segurança. Issues de segurança subem em prioridade — mesmo que pequenas no codigo, podem ser catastroficas no mundo real.

## PASSO 0 — Determinar escopo

Em ordem de fallback:
1. Se $ARGUMENTS contem um path valido, use-o como diff
2. Se existe branch base detectavel (`main`, `master`, `develop`), use `git diff <base>...HEAD`
3. Caso contrario: `git diff HEAD`

Reporte ao usuario qual escopo foi escolhido. Liste arquivos no diff (paths + linhas alteradas).

## PASSO 1 — Checks automaticos (PARALELO via Bash)

Rode em paralelo:

```
# Build/typecheck
- Detecte e rode o build do projeto (tsc, cargo check, mypy, go build, etc)

# Lint
- ESLint, Ruff, Clippy, etc

# Testes existentes que tocam os arquivos do diff

# Seguranca automatizada (CRITICO)
- gitleaks detect --source . --no-banner --redact (secrets em commits)
- gitleaks detect --no-git --source . (secrets em working tree)
- npm audit --omit=dev 2>/dev/null || pip-audit 2>/dev/null || cargo audit 2>/dev/null (CVEs em deps)
- Se semgrep estiver instalado: semgrep --config=auto --json (padroes de vuln)
- grep recursivo no diff: ['SUPABASE_SERVICE_ROLE', 'SERVICE_ROLE_KEY', 'PRIVATE_KEY', 'API_KEY=', 'PASSWORD=', 'SECRET=']
```

Se algum check de seguranca falhar (gitleaks encontrou secret, npm audit reporta CVE high+), trate como BLOCKER e destaque antes de tudo.

## PASSO 2 — Revisao de SEGURANCA (camada dedicada, prioridade maxima)

Spawne UM Agent Explore dedicado a seguranca, com checklist exaustivo:

```
Agent(Explore, "REVISAO DE SEGURANCA. Analise este diff contra a lista completa abaixo. Para cada issue encontrado retorne: arquivo:linha, descricao tecnica, severidade, vetor de ataque concreto, sugestao de fix.

## OWASP Top 10 + extendidos
- SQL injection (query dinamica, string concat, falta de prepared statement)
- NoSQL injection (Supabase: queries dinamicas em .filter/.match)
- Command injection (exec, spawn com input do usuario)
- XSS (innerHTML, dangerouslySetInnerHTML, eval, jsx sem escape)
- XXE em parsers de XML
- Path traversal (../, paths absolutos em input do usuario)
- SSRF (fetch/axios com URL do usuario)
- Insecure deserialization (JSON.parse em input nao validado, eval, vm.runInContext)
- Open redirect (window.location, res.redirect com input)
- Mass assignment / prototype pollution (Object.assign, spread com input)
- TOCTOU (check-then-act em fs, autorizacao com gap entre verificacao e uso)

## Autenticacao & sessao
- JWT: secret fraco/hardcoded, alg=none, sem expiracao, sem refresh+revoke
- Sessao: timeout ausente, fixation, sem rotacao apos login
- Password: hash fraco (md5, sha1, sha256 sem salt), sem bcrypt/argon2
- Timing attacks em comparacao de tokens (use timingSafeEqual)
- Recuperacao de senha: token previsivel, sem expiracao, sem one-time

## Autorizacao
- IDOR (Insecure Direct Object Reference): endpoint recebe ID sem validar ownership
- Privilegio escalavel: usuario consegue alterar role/permissions do proprio user
- Bypass de middleware (rotas que esqueceram auth middleware)
- Server-side enforcement ausente (validacao so no client)

## Supabase-especifico (CRITICO para projetos do Dr Alain)
- RLS desabilitado em tabela com dados sensiveis
- RLS policy permissiva demais (using true, sem clausula de owner)
- service_role key sendo usado em codigo client-side
- anon key com permissoes que deveriam ser autenticadas
- Storage bucket com leitura publica para dados de paciente
- Edge function sem verificacao de JWT
- Migrations que removem RLS sem readicionar

## Crypto
- Hash sem salt
- AES-ECB (sempre inseguro), AES-CBC sem MAC
- IV reutilizado
- Random fraco (Math.random para token, use crypto.randomBytes)
- TLS desabilitado em fetch/HTTP client
- Keys hardcoded ou commitadas em .env

## Dados sensiveis & PHI/PII (LGPD)
- PHI em logs (CPF, nome+diagnostico, prontuario)
- PHI em mensagens de erro retornadas ao client
- PHI em query strings (URLs ficam em logs de servidor)
- Audit trail ausente para acesso a dados de paciente
- Sem consentimento explicito para coleta/uso
- Sem mecanismo de direito ao esquecimento (LGPD art. 18)
- Backups com PHI sem criptografia
- Dados cruzados (e.g., logging request body inteiro contendo dados clinicos)

## CORS, cookies, headers
- CORS Allow-Origin: * em endpoint autenticado
- Cookie sem HttpOnly, Secure, SameSite
- Sem CSP, sem X-Frame-Options, sem X-Content-Type-Options
- Sem rate limiting em endpoint sensivel (login, registro, reset)

## File upload
- Sem validacao de tipo (MIME pode ser spoofed — valide tambem magic bytes)
- Sem limite de tamanho
- Nome de arquivo do usuario usado direto no path (path injection)
- Upload em diretorio executavel

## Inputs e validacao
- Falta validacao em endpoint publico
- Mensagem de erro expoe stack trace ou estrutura interna do DB
- Inputs de tipo numerico aceitos como string sem coercao validada

## Dependencias
- Pacote com CVE conhecido (npm audit/pip-audit reportou)
- Pacote pinado em versao desatualizada por >6 meses
- Pacote de fonte questionavel

Para CADA issue:
- Severidade: BLOCKER (vulnerabilidade explorable) / IMPORTANT (hardening) / NIT (best practice)
- Vetor de ataque CONCRETO (nao 'pode ser inseguro' — descreva como um atacante exploraria)
- Sugestao de fix com exemplo de codigo correto")
```

## PASSO 3 — Invocar /security-review (skill nativa)

Se o ambiente tiver `/security-review` disponivel, ofereca ao usuario rodar tambem como segunda camada:

"Quer rodar o /security-review nativo do Claude Code em paralelo? Ele faz uma revisao independente de seguranca do branch atual."

Se sim: invoque-o e adicione os achados na sintese final.

## PASSO 4 — Revisao paralela de outras dimensoes

Spawne em paralelo (NAO incluindo seguranca, que ja foi feita acima):

```
Agent(Explore, "LOGICA: bugs, edge cases nao tratados, race conditions, off-by-one, null handling, error paths ausentes, retry/timeout missing. Para cada issue: arquivo:linha, severidade (blocker/important/nit), sugestao.")

Agent(Explore, "PADROES DO PROJETO: convencoes de naming, estrutura, error handling existente, uso de bibliotecas ja presentes. Compare contra 2-3 arquivos similares do projeto. Para cada divergencia: arquivo:linha, severidade, sugestao.")
```

## PASSO 5 — Segunda opiniao do revisor externo (Codex ou opencode)

Use o revisor determinado em $ARGUMENTS (default `codex`, alternativa `opencode`).

Prompt comum (identico nos dois casos):
> Voce e um senior reviewer com foco em seguranca. Revise este patch. Aponte com prioridade qualquer vulnerabilidade (OWASP, crypto, autenticacao, autorizacao, validacao de input). Em segundo lugar, race conditions e edge cases. Para cada issue: arquivo:linha | severidade (blocker/important/nit) | vetor de ataque ou cenario | sugestao. NAO reescreva codigo.

### Se revisor == codex

```
! git diff HEAD | codex exec "<prompt acima>" > /tmp/codex-review.md 2>&1
```

Depois **leia `/tmp/codex-review.md` com a ferramenta Read**, nao com `cat`/`tail`.

> **Limite conhecido do Codex neste setup:** ele roda atras do proxy Headroom com compressao de ~90% (`force_kompress`), que engole preposicoes e artigos antes de o texto chegar ao modelo. Use-o para **logica, arquitetura e seguranca** — nessas dimensoes e excelente. **Nao** use para apontar erro de texto literal (nome de variavel, string, mensagem, gramatica): ele ja "corrigiu" trechos que estavam certos. Antes de aplicar qualquer apontamento sobre texto literal, confirme com `grep` no arquivo real. Detalhe: memoria `feedback-codex-revisao-texto-headroom`.

### Se revisor == opencode

```
! git diff HEAD > /tmp/review-diff.patch && opencode run --dangerously-skip-permissions --file /tmp/review-diff.patch "<prompt acima>"
```

Antes de invocar, faca `which codex` (ou `which opencode`) para confirmar que o binario existe. Se nao existir, anote "[revisor `X` nao instalado]" na sintese e siga sem ele — nao bloqueie o restante da revisao.

Anote a fonte usada no campo "Fonte" das tabelas (`codex` ou `opencode`).

## PASSO 6 — Sintese

Consolide os achados em DUAS tabelas separadas:

### Tabela 1 — Achados de SEGURANCA (prioridade absoluta)

```markdown
| # | Severidade | Categoria | Arquivo:linha | Vulnerabilidade | Vetor de ataque | Sugestao | Fonte |
|---|---|---|---|---|---|---|---|
| S1 | BLOCKER | SQL injection | api/users.ts:42 | string concat em query | atacante injeta `' OR 1=1` no param `id` | usar prepared statement | agent-security, codex |
| S2 | BLOCKER | Supabase RLS | migrations/003.sql:8 | tabela `prontuarios` criada sem RLS | qualquer cliente com anon key le todos prontuarios | `ALTER TABLE prontuarios ENABLE ROW LEVEL SECURITY` + policy de owner | agent-security |
| S3 | IMPORTANT | PHI em log | api/exam.ts:15 | console.log(req.body) inclui CPF e diagnostico | logs vazam em caso de leak; LGPD art. 46 | redact PHI antes de log | agent-security |
```

### Tabela 2 — Demais achados (logica, padroes, performance)

```markdown
| # | Severidade | Dimensao | Arquivo:linha | Issue | Sugestao | Fonte |
|---|---|---|---|---|---|---|
```

Regras:
- Severidade: `BLOCKER` (impede merge), `IMPORTANT` (precisa endereco), `NIT` (preferencia)
- Issues sem arquivo:linha exato sao descartados
- Disagreements entre fontes destacam — secao "Disagreements" no final

## PASSO 7 — Relatorio final

```
## Revisao concluida

Escopo: [diff usado]
Arquivos no diff: [N]  |  +[X] -[Y] linhas

### Checks automaticos
- Build: PASSOU / FALHOU
- Lint: PASSOU / FALHOU
- Testes: PASSOU / FALHOU
- gitleaks: 0 secrets / N secrets encontrados
- npm audit: 0 vulns / N vulns (M high, L critical)
- semgrep: 0 findings / N findings

### Achados de SEGURANCA
[Tabela 1]

### Demais achados
[Tabela 2]

### Disagreements (se houver)

### Veredito
[ ] PRONTO PARA MERGE (zero blockers de seguranca, zero blockers em outras dimensoes)
[ ] PRECISA ENDERECO (lista de blockers)
[ ] BLOQUEADO — vulnerabilidades de seguranca devem ser corrigidas antes de qualquer outro trabalho

Quer aplicar correcoes? Use /execute-plan ou peca diretamente.
```

## REGRAS

- SEGURANCA E GATE — se houver `BLOCKER` na Tabela 1, veredito NUNCA pode ser "pronto para merge"
- NAO aplique correcoes — apenas reporte
- Para projetos com dados clinicos, eleve PHI/PII a `BLOCKER` por default (LGPD)
- Disagreements entre Claude e o revisor externo (Codex ou opencode) sao informacao valiosa — sempre destaque
- Se algum check falhar (revisor externo nao instalado, gitleaks nao instalado), continue com os demais e anote no relatorio
- Se NAO houver diff, informe e ofereca revisar HEAD~1..HEAD
- Severidade de seguranca tem peso maior que severidade equivalente em outras dimensoes (e.g., `IMPORTANT` de seguranca > `IMPORTANT` de padroes)
