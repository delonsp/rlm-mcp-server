# Plano: Rebaixar chunks de bibliografia/cabeçalho na busca semântica
Data: 2026-06-01
Spec de origem: sessão `/plan-code` (investigação do "gargalo de chunking/embedding"). **A premissa original foi REFUTADA empiricamente** — ver Visão geral. Este plano é o resíduo real, pequeno e OPCIONAL.
Prioridade: **BAIXA** (a busca já está boa; isto é polimento de nicho).

## Visão geral
Suspeitava-se que a busca semântica do rlm-mcp estava fraca (chunks "grandes demais", scores 0.47–0.59 "baixos"). **14 buscas reais em 2 livros (2026-05-31) refutaram isso:**
- PT→PT (Bredesen, `recode_bredesen_guia_pratico`): **4/4 acertos limpos**, scores 0.64–0.79.
- PT→EN (Seyfried acadêmico, `livro_seyfried_metabolic_disease`): **10/10 no tema, 5/10 parágrafo limpo**.
- 0.47–0.59 é a **escala NORMAL** do `text-embedding-3-small` (melhor acerto limpo = 0.64). Score nunca foi o problema.

**Único padrão de fraqueza medido:** em ~metade das queries PT→EN do livro acadêmico, o top-1 caiu numa **lista de referências** (bibliografia densa em termos do tema) ou num **cabeçalho de capítulo/marcador de página**, em vez do parágrafo explicativo. Isso NÃO aparece no caso PT→PT. Causa: o chunking por caractere (512c, `vector_index.py:17`) transforma a bibliografia/títulos em chunks que competem com a prosa.

Este plano **NÃO** re-arquiteta nada. Adiciona uma classificação leve de "chunk boilerplate" (referências/cabeçalho) e um **rebaixamento de score** opcional na busca, validado pelas MESMAS 14 queries manuais (sem harness).

## Requisitos rastreados
| ID | Requisito | Como será atendido | Arquivo(s) |
|----|-----------|---------------------|------------|
| R1 | Detectar chunks que são predominantemente bibliografia | Heurística conservadora (densidade de linhas "NN. Autor. Título. Journal. Ano;vol:pp", `et al`, DOI) | `vector_index.py` |
| R2 | Detectar chunks de cabeçalho/marcador | Heurística (`--- Página N ---`, `Chapter/Capítulo N`, título curto/caps) | `vector_index.py` |
| R3 | Rebaixar (não apagar) no ranking, reversível | Flag `is_boilerplate` no `ChunkInfo` (computada, não persistida) + penalidade no `search()` via env | `vector_index.py` |
| R4 | Zero re-embed e backward-compat | Flag computada do texto do chunk no build E no `from_serializable` (índices existentes pegam o flag sem re-embeddar) | `vector_index.py` |
| R5 | Medir antes de ligar por default | Default `RLM_BOILERPLATE_PENALTY=1.0` (desligado) no ship; validar 14 queries; só então flipar o default | `vector_index.py`, validação manual |
| R6 | Não falso-positivar prosa real | Thresholds conservadores; na dúvida, NÃO marcar; teste de não-regressão de classificação | `vector_index.py`, `tests/` |

## O que NÃO estamos fazendo (anti-scope creep)
- **NÃO** re-chunkar (não mudar 512c/overlap nem virar sentence-aware).
- **NÃO** trocar modelo (3-small→3-large) nem mexer em embeddings/dimensão.
- **NÃO** apagar chunks do índice (só rebaixar score — reversível, sem perda de dados).
- **NÃO** mexer em RRF/BM25/keyword, persistência (schema), B2/segurança, nem nos 19 tools.
- **NÃO** construir harness de avaliação (descartado como over-engineering; a validação é o spot-check manual das 14 queries que já temos).

## Arquitetura

### Arquivos modificados (aditivo)
- `src/rlm_mcp/vector_index.py`:
  - `ChunkInfo`: adicionar campo `is_boilerplate: bool = False` (computado, **não** vai pra `to_serializable` → sem migração de schema).
  - `_classify_boilerplate(text: str) -> bool` (nova helper): True se o chunk é predominantemente referência OU cabeçalho/marcador. Conservadora.
  - `_chunk_text(...)`: ao criar cada `ChunkInfo`, setar `is_boilerplate=_classify_boilerplate(chunk_text)`.
  - `from_serializable(...)`: recomputar `is_boilerplate` do `text` (índices já persistidos ganham o flag sem re-embeddar).
  - `search(...)`: após `_cosine_similarity`, se `chunk.is_boilerplate`, multiplicar o score por `_BOILERPLATE_PENALTY` (env `RLM_BOILERPLATE_PENALTY`, default `1.0` = sem efeito). Reordenar normalmente.
- (NENHUMA mudança em `embeddings.py`/`indexer.py`/`persistence.py`/`http_server.py`.)

### Heurística de classificação (proposta — validar)
- **Referência** (alto sinal): das linhas não-vazias do chunk, fração ≥ ~0.6 casando QUALQUER de:
  - `^\s*\[?\d{1,4}[\.\)\]]\s+\S` (entrada numerada de bibliografia);
  - `\b\d{4};\s*\d+` ou `\bdoi:` ou `\bet al\b\.?` com vírgulas de autores.
- **Cabeçalho/marcador**: chunk com ≥ ~0.5 do conteúdo casando `---\s*P[áa]gina\s*\d+` / `^\s*(Chapter|Cap[íi]tulo)\s+\d+` / linha-título curta em CAPS isolada.
- **Conservador:** exigir um mínimo de linhas (ex: ≥3) antes de classificar como referência; chunk curto ambíguo → NÃO marca (default prosa).

### Decisão de design: rebaixar vs apagar
- **Escolhido: REBAIXAR no `search` (penalidade de score), não apagar do índice.** Por quê: (a) reversível por env; (b) sem re-embed/migração; (c) um chunk de fronteira (referências + um pouco de prosa) não é perdido — só desce; (d) se a heurística errar, o custo é um chunk mal-rankeado, não um buraco no índice.
- Rejeitado: drop no build (irreversível, perde prosa de fronteira, exige re-embed).

## Armadilhas (red herrings)
- **NÃO** persistir `is_boilerplate` em `to_serializable` (evita migração de schema na tabela `embeddings`); recomputar no load é barato e mantém compat.
- **NÃO** aplicar penalidade tão forte que zere o chunk (ex: usar 0.5–0.7, não 0.0) — uma referência ainda pode ser o único lugar que cita algo.
- **`indexer.py` (RRF/BM25)** não muda — a penalidade é só na perna semântica (`VectorIndex.search`). A fusão RRF usa rank, então rebaixar o chunk semântico já propaga.
- **Marcador `--- Página N ---`** aparece no MEIO de chunks de prosa (extração de PDF) — não marcar o chunk inteiro como boilerplate só por conter 1 marcador; exigir DENSIDADE.

## Critérios de verificação (success criteria)
- [ ] `uv run python -c "import rlm_mcp.vector_index"` — ok.
- [ ] `uv run ruff check src/rlm_mcp/vector_index.py` — sem erros novos.
- [ ] Teste unitário de `_classify_boilerplate`: bibliografia real (ex: trecho de L10626/L11566 do Seyfried) → True; prosa real (L4800, L16530) → False. (gitignored ou versionado conforme padrão.)
- [ ] `to_serializable`/`from_serializable` round-trip preserva tudo e recomputa o flag (sem campo novo no JSON persistido).
- [ ] **Validação live (mesmo método da investigação):** re-rodar as 5 queries PT→EN do Seyfried que deram "parcial" (Q1 mito, Q5 metab-vs-genético, Q7 cetônicos, Q8 transf. nuclear, Q9 hipóxia) com `RLM_BOILERPLATE_PENALTY=0.6` e confirmar que ≥3 sobem para "parágrafo limpo" SEM regredir os 5 acertos limpos (Q2,Q3,Q4,Q6,Q10) nem as 4 do Bredesen (4/4).

## Estado final desejado
Com `RLM_BOILERPLATE_PENALTY` < 1.0, chunks de bibliografia/cabeçalho param de "ganhar" do parágrafo explicativo na perna semântica, melhorando o caso PT→EN acadêmico, sem tocar produção quando a penalidade = 1.0 (default no ship). Nenhum re-embed, nenhuma migração, totalmente reversível por env. Se a validação não mostrar ganho claro, o default fica 1.0 e o código é dead-weight inofensivo (decisão: manter ou reverter).

## Checklist de implementação (ordem)
1. [ ] `_classify_boilerplate(text)` + constantes/regex. — dep: nenhuma
2. [ ] `ChunkInfo.is_boilerplate` + setar em `_chunk_text` e recomputar em `from_serializable`. — dep: 1
3. [ ] `_BOILERPLATE_PENALTY` (env, default 1.0) + aplicar em `search()`. — dep: 2
4. [ ] Teste unitário de classificação (casos reais do Seyfried/Bredesen). — dep: 1-3
5. [ ] Smoke: import + ruff + round-trip serialização. — dep: 1-4
6. [ ] Commit + push (deploy). — dep: 5
7. [ ] Validação live: 5 queries PT→EN com penalty=0.6, comparar com baseline desta sessão. — dep: 6
8. [ ] Decisão: se melhora → flipar default p/ ~0.6 via env Dokploy; se não → manter 1.0 (ou reverter). — dep: 7

## Notas para o implementador
- **Style:** convenções de `vector_index.py` (dataclass `ChunkInfo`, docstrings, logging `rlm-mcp.*`). Stdlib `re`.
- **Baseline desta sessão (pra comparar na validação):** Seyfried PT→EN top-1 — Q1 L10626=refs, Q5 L20374=conclusão ambígua, Q7 L5638=cabeçalho cap.5, Q8 L11690=cabeçalho cap.11, Q9 L11566=refs; limpos: Q2 L18183, Q3 L16197, Q4 L19800, Q6 L19651, Q10 L16530. Bredesen PT→PT: B1 L4285, B2 L8559, B3 L3042, B4 L24014 — todos limpos.
- **Onde validar:** container de produção (dados + key), via `rlm_search_index(mode="semantic")` + `rlm_execute` pra ler o texto das linhas (SSH `alana` se precisar dos logs).
- **Honestidade:** isto é OPCIONAL e de baixa prioridade. Se na implementação a heurística se mostrar frágil/falso-positiva, abortar é uma resposta legítima — a busca já é utilizável sem isto.
