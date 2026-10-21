---
name: review-code
description: Revisão de código do RLM com Claude Opus 5.5 e segunda opinião do Codex, cobrindo segurança, lógica e testes. Use quando pedir review de código com Claude; respeite outro modelo explicitamente escolhido pelo usuário.
---

# Review local com Claude Opus 5.5

Cópia local do comando compartilhado `~/.claude/commands/review-code.md`, disponível
também no projeto `~/prontuario-alain`. O fluxo original está preservado em
[references/original-review-code.md](references/original-review-code.md); leia-o
para a revisão completa de segurança, lógica e padrões. Esta cópia funciona sem
depender do arquivo global. `.claude/skills/review-code` aponta para esta pasta.

## Modelo e execução neste projeto

- O revisor principal padrão é **Claude Opus 5.5** (`claude-opus-5-5`). Uma escolha
  explícita diferente do usuário prevalece. Confirme disponibilidade na instalação
  e o modelo realmente usado em `modelUsage` ou evento estruturado equivalente.
- Com Codex como coordenador, execute Claude Code CLI com `--model` explícito,
  escopo somente leitura e saída JSON. Nunca substitua Claude por um agente Codex.
- Se esta skill já estiver rodando no próprio Claude selecionado, faça a revisão
  diretamente; não invoque outra cópia de Claude ou desta skill recursivamente.
  Codex/OpenCode são apenas a segunda opinião prevista no fluxo original.
- A revisão deve abranger alterações locais tracked e arquivos novos. A raiz Git
  é `rlm-mcp-server/`, mesmo quando a sessão começa em `massive-context-window/`.
- Verificações locais: `uv run --frozen --extra dev pytest tests/ -q -rs` e
  `uv run --frozen --extra dev ruff check src/ tests/`. Reaproveite resultados
  recentes quando o código não mudou. Não rode `scripts/qa_extensive.py`: atua
  no servidor real e altera dados. Não use o acervo remoto como material de review.
- Não aplique correções durante um pedido apenas de revisão. Relate os achados
  verificáveis com arquivo, linha, cenário e gravidade; diferencie regressões de
  problemas preexistentes. Não faça commit, push ou deploy como parte do review.

## Adaptação para Codex

- O modelo solicitado pelo usuário é obrigatório: execute Claude Code CLI com `--model` explícito. Não substitua Opus por outro modelo silenciosamente e não chame um agente Codex de Claude. Confirme o modelo efetivamente usado no resultado estruturado (`modelUsage` ou evento equivalente). Se houver indisponibilidade, reporte antes de trocar.
- Descubra flags com `claude --help`. Prefira modo não interativo, ferramentas somente leitura e resultados JSON. O CLI pode usar autenticação já instalada; não copie nem exponha credenciais. `--bare` pode desabilitar OAuth, portanto não presuma que sirva para uma conta Claude Max.
- O escopo explícito do usuário prevalece. Para alterações locais, inclua staged + unstaged e arquivos novos. `git diff base...HEAD` não contém alterações ainda não commitadas. Preserve mudanças anteriores alheias ao pedido.
- Ao precisar limitar o acesso do revisor, crie uma cópia temporária de fontes e patch, excluindo `.env*`, credenciais, dados de pacientes, logs e arquivos de produção. Inclua as fontes novas numa lista e explique que elas não aparecem no diff.
- Execute a revisão externa com instruções para NÃO editar, commitar, publicar ou enviar mensagens. A resposta deve trazer achados concretos, arquivo/linha, cenário, severidade e correção sugerida. Verifique cada achado nas fontes reais antes de adotá-lo.
- Para a segunda opinião, use Codex/OpenCode conforme escolha do usuário; quando Codex já é o coordenador, uma revisão independente de Codex pode ser delegada aqui. Não inicie chamadas recursivas ao próprio fluxo. Escolha Claude/modelo antes da chamada, não de dentro do revisor.
- Preserve a camada dedicada de segurança do fluxo original: autorização entre clínicas, RLS, credenciais, PHI, XSS e dados interpolados. Separe defeitos introduzidos de problemas preexistentes e de limites acordados.
- Reaproveite checks de build/lint/testes válidos para o mesmo código. Use `pnpm` quando exigido pelo projeto. Rode ferramentas de auditoria disponíveis; não instale scanners ou ignore permissões para completar checklist. Resultados de scanners devem ser redigidos sem segredos. Registre indisponibilidades e diferencie vulnerabilidades preexistentes das introduzidas.
- A oferta opcional de `/security-review` do original não obriga interromper uma revisão já autorizada. Use recursos disponíveis e reporte limitações reais.
- Entregue relatório com modelo confirmado, escopo, achados ordenados, checks e limites. Só diga que passou quando o processo concluir e houver evidência. Revise apenas; correções dependem do escopo pedido, sem aprovação adicional inventada.

A skill suporta seleção de modelos; isso não implica executar todos os modelos a cada pedido. Não presuma identificadores ou disponibilidade atuais: confirme na instalação/provedor.
