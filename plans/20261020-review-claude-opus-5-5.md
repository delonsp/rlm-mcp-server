# Review — Claude Opus 5.5 + segunda opinião Codex

**Veredito original: corrigir os seis P2 abaixo antes de publicar.** Não foi identificado
P1 nem regressão concreta de autenticação, SQL ou desserialização. Isso não equivale
a uma auditoria completa de segurança ou validação clínica.

## Execução e escopo

- Revisor principal: **Claude Opus 5.5**, solicitado com
  `--model claude-opus-5-5`; confirmado em `modelUsage` e `canonicalModel`.
  Claude Code 2.1.303; processo terminou com código 0, `subtype=success` e
  `is_error=false`. [Identificador oficial do modelo](https://platform.claude.com/docs/en/models/opus-5-5/overview).
- Skill: cópia local em `.agents/skills/review-code/SKILL.md`, derivada do comando
  compartilhado `~/.claude/commands/review-code.md`, disponível também no
  `~/prontuario-alain`. Não havia uma cópia específica naquele projeto.
- Escopo: alterações locais contra `0cade55`, incluindo arquivos novos:
  **35 arquivos, +1.532/-200 linhas**. Inclui código, testes, documentação e a skill.
  A configuração pessoal `.codex` preexistente ficou fora do material enviado.
- Claude recebeu uma cópia temporária de fontes, testes, versões anteriores e
  patch, com ferramentas apenas `Read`, `Glob` e `Grep`; sem acesso ao acervo,
  credenciais, bancos, logs ou serviços de produção.
- Três revisões independentes Codex cobriram segurança, lógica e padrões.
  O coordenador conferiu os achados no código e executou reproduções sintéticas.
  Os arquivos do snapshot permaneceram idênticos durante a revisão.
- [Parecer original completo do Claude](20261020-review-claude-opus-5-5-original.txt).

## Segurança e uso de memória

| ID | Gravidade | Local | Problema e evidência | Correção sugerida | Fonte |
|---|---|---|---|---|---|
| S1 | **P2** | `src/rlm_mcp/collection_builder.py:116–118` | O índice global guarda as fontes, o texto combinado e o mapa. `rlm_clear(all=true)` zera o REPL, mas essas referências continuam vivas. Probe do coordenador: 0 variáveis após a limpeza, com 1.000.000 caracteres da fonte e 1.000.146 do combinado ainda retidos. A revisão independente confirmou crescimento entre ciclos de novas coleções. | Descartar índices e caches dependentes ao limpar/remover variáveis e no descarte automático; contabilizar memória retida fora do REPL. | Claude S1 + Codex segurança; gravidade elevada após reprodução |

S1 exige uso autenticado e pode esgotar a memória do processo; não é bypass de
autenticação nem evidência de vazamento entre usuários. O problema de limpeza
incompleta de índices já tinha antecedentes, mas as referências completas adicionadas
pelo patch ampliam a retenção. Os caches de versão e leitura também precisam
participar da limpeza, embora tenham limites próprios.

## Recuperação e referências

| ID | Gravidade | Local | Problema e evidência | Correção sugerida | Fonte |
|---|---|---|---|---|---|
| L1 | **P2** | `src/rlm_mcp/document_reader.py:65–66` | A origem não está vinculada à versão da extração. Após carregar de `s3:books/original.pdf` e substituir o texto via `rlm_execute`, `rlm_read` devolve o conteúdo novo atribuído ao PDF original. Na carga S3 assíncrona também existe intervalo entre publicar o texto e atualizar a origem (`s3_tools.py:402–405`). | Guardar origem e versão da carga juntas; marcar texto reatribuído como derivado e publicar texto/metadados sob o mesmo bloqueio. | Claude L4 + Codex lógica/padrões; reprodução confirmada |
| L2 | **P2** | `src/rlm_mcp/indexer.py:538,583–585` | A palavra portuguesa “no” passou a ser tratada como negação inglesa. Consultar `['dor','no','peito']` com `require_all=True` deixou de encontrar “Peito: dor ardente”. | Distinguir a contração portuguesa da negação inglesa na consulta; acrescentar casos em ambos os idiomas. Remover cegamente “no” também pode prejudicar consultas em inglês. | Claude L1; comparação com base confirmada |
| L3 | **P2** | `src/rlm_mcp/indexer.py:583–585` | Em “Suplemento com vitamina A e D.” apenas A é preservado. Consultar `['vitamina','D']` com `require_all=True` retorna zero; antes retornava a passagem. Em outro probe, inverter para `['D','vitamina']` remove o filtro D e retorna também A/E. | Preservar qualificadores em enumerações e tornar a interpretação da consulta independente da ordem do array. | Codex padrões; comparação com base confirmada |
| L4 | **P2** | `src/rlm_mcp/collection_builder.py:105–106`; `tools/handlers/collection_tools.py:480` | Fonte removida da memória pela limpeza automática desaparece da consulta sem aviso, embora continue persistida e pertencendo à coleção. Probe: `doc_b` removido pelo `_auto_cleanup`, ainda no SQLite; a busca pesquisou só `doc_a`, mas anunciou “2 vars na coleção”. | Informar nomes das fontes não pesquisadas e distinguir membros cadastrados de fontes efetivamente consultadas; definir recarga respeitando o limite de memória. | Claude L3; reprodução confirmada |
| L5 | **P2** | `src/rlm_mcp/pdf_parser.py:86–98,311–322`; `text_quality.py:17–18` | Um único NUL na saída do extrator reprova todo o PDF legível. No probe de página sintética, a base removia o NUL e carregava o texto; o patch rejeitou e devolveu apenas “MISTRAL_API_KEY não configurada”, ocultando o primeiro motivo. | Distinguir resíduo isolado de extração de conteúdo binário/corrupção relevante; preservar os dois erros quando extração e OCR falharem. Manter proteção da entrada direta contra binário. | Claude L2; comparação com base confirmada |

L1 expõe um problema antigo de metadados por meio do novo contrato de referência
do `rlm_read`; o patch não criou a preservação de `source` no execute. A remoção
explícita de fontes de uma coleção foi uma decisão da implementação; L4 trata da
omissão silenciosa após limpeza automática. Em L5, a rejeição é intencional na
regra atual, mas a regressão funcional foi confirmada: a intensidade do filtro
para extração PDF precisa ser ajustada ou assumida explicitamente.

## Comparações reproduzidas

As reproduções usaram memória e SQLite temporários. A extração PDF foi simulada,
sem documento real, OCR remoto ou chamadas pagas.

| Caso | Base `0cade55` | Código revisado |
|---|---|---|
| “dor no peito” com todos os termos exigidos | 1 resultado | 0 resultados |
| “vitamina D” numa enumeração “vitamina A e D” | 1 resultado | 0 resultados |
| Página legível com um NUL isolado | Extração aceita | Falha; mensagem final sobre chave Mistral |

## Menor prioridade e divergências

- **Logs ANSI (Claude L7, P3):** confirmado que logs com códigos de cor são
  rejeitados pelo contador de controles (`text_quality.py:8,19–21`). Diferenciar
  sequências ANSI válidas de corrupção, sem remover a proteção para binários.
- **Leitura de muitas linhas (Codex segurança):** o índice de linhas é construído
  inteiro antes de verificar o limite do cache (`document_reader.py:33–41`). Probe
  independente: 8 MiB de quebras de linha geraram pico adicional de 73,23 MiB para
  devolver um caractere; o resultado não entrou no cache. É melhoria de robustez,
  sem caracterizar um novo bloqueador, pois a indexação anterior já tinha operações
  caras semelhantes.
- **Índices legados após reinício (Claude L5, P3):** índices sem versão são
  descartados corretamente, mas a reconstrução sob demanda não os regrava. O
  custo volta nos reinícios seguintes. Confirmado no fluxo de código, sem ensaio
  de latência de produção; considerar migração ou persistência da reconstrução.
- **S1:** Claude classificou como P3. Codex confirmou acúmulo por coleções
  sucessivas após limpeza, portanto o relatório consolidado usa P2.
- **Cache de versões (Claude L6):** expulsão por limite de memória é comportamento
  esperado do cache limitado. Não foi adotado como defeito independente; o ganho
  medido vale para consultas que permanecem no cache.
- **Leitura do blob antigo (Claude L8):** hipótese de otimização, sem custo
  relevante demonstrado. Não foi elevada a problema que impeça publicação.
- A observação do Claude sobre publicar caminhos pessoais e exemplos do comando
  original não exige confirmação adicional: a cópia foi solicitada explicitamente
  e esta etapa não fez push nem deploy.

## Checks e limites

- Reaproveitados para o mesmo código Python: **283 testes aprovados, 13 skips**
  por condições de macOS/Linux/container; lint aprovado. Os novos probes acima
  mostram lacunas que a suíte existente ainda não cobre.
- Skill local aprovada por `quick_validate.py`; referência local e symlink do
  Claude conferidos. `git diff --check` aprovado.
- Busca limitada por padrões de segredos nas linhas adicionadas: nenhum candidato.
  Isso não substitui uma auditoria de segredos.
- `gitleaks`, `pip-audit` e `semgrep` não estão instalados, inclusive como módulos
  do ambiente do projeto. Não foram instalados nem simulados; CVEs não foram auditados.
- Nenhuma correção de aplicação foi feita durante o review. Apenas skill, links
  e relatórios foram criados. Sem commit, push, deploy ou alterações no acervo.

## Skill local criada

- Fonte: `.agents/skills/review-code/SKILL.md`.
- Referência copiada: `.agents/skills/review-code/references/original-review-code.md`.
- Claude: `.claude/skills/review-code -> ../../.agents/skills/review-code`.
- Na pasta externa: `.agents -> rlm-mcp-server/.agents`, para descoberta também
  quando o projeto é aberto por `massive-context-window/`.
- Padrão: Claude Opus 5.5 explícito, confirmação do modelo efetivo, revisão
  somente leitura e segunda opinião Codex; outra escolha explícita do usuário
  prevalece. A referência local elimina dependência do arquivo global.

## Correções posteriores autorizadas para publicação

Os seis P2 foram corrigidos após o pedido de commit, push e deploy:

- S1: limpeza e substituição descartam índices dependentes, visualizações de
  documentos e versões retidas; a limpeza automática usa o mesmo descarte.
- L1: carga publica texto e origem sob o mesmo lock; execute marca reatribuição
  como `derived:` nos dois modos de execução.
- L2: consultas usam `query_language="pt"` por padrão; `en` preserva “no”.
- L3: qualificadores são mantidos em enumerações e arrays reordenados; a conjunção
  “e” entre letras não é promovida a vitamina E.
- L4: a busca informa nomes de fontes não consultadas e a contagem carregada/total,
  inclusive sem resultados. Índice, texto e mapa são capturados como um conjunto
  antes da busca, preservando coerência durante cargas concorrentes.
- L5: extração PDF admite NUL esparso, mantém rejeição de corrupção densa e
  preserva os erros da extração e OCR. A entrada direta continua estrita.

Os testes reproduziram as falhas antes das correções. Verificação final local:
**305 testes aprovados, 13 skips de ambiente; Ruff e git diff --check aprovados.**
A segunda opinião Codex conferiu as correções e encontrou o caso adicional da
conjunção, também reproduzido e corrigido. Claude não foi executado novamente:
o parecer original acima e seu arquivo integral permanecem preservados.

A publicação atual é manual via `ssh magalu`; push para main executa somente CI.
Os P3 e os limites de validação clínica listados acima permanecem registrados.
