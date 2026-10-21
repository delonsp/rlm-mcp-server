# Precisão e eficiência da consulta ao acervo

## Objetivo

Impedir a entrada de conteúdo evidentemente inválido, eliminar citações de versões
antigas, permitir leitura verificável e reduzir trabalho repetido. Preservar a
compatibilidade das ferramentas existentes e medir melhorias com dados sintéticos.

## Fontes de inspiração consultadas

- [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx/blob/dev/docs/faq.md):
  reconhecer conteúdo do arquivo e identificar duplicatas por conteúdo.
- [LlamaIndex CitationQueryEngine](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/citation_query_engine.py):
  referências ligadas a trechos identificáveis da fonte.
- [RAGFlow FAQ](https://github.com/infiniflow/ragflow/blob/main/docs/faq.mdx):
  testar extração e recuperação separadamente da geração da resposta.
- [SQLite FTS5](https://sqlite.org/fts5.html): alternativa de indexação textual;
  comparar com o BM25 existente antes de migrar o armazenamento.

São referências de desenho; esses frameworks não foram adicionados ao servidor.

## Etapas e critérios de aceitação

- [x] 1. Validar importação: rejeitar PDF/binário passado como texto e corrupção
  evidente antes de alterar a variável; OCR deve ser tentado para extração ilegível,
  sem publicar texto inválido quando todas as tentativas falharem.
- [x] 2. Atualizar coleções: carga, reatribuição e remoção de uma fonte devem refletir
  na próxima busca; consultas repetidas sem mudança não devem reconstruir o índice.
- [x] 3. Leitura verificável: ferramenta paginada que devolva texto completo dentro
  do limite solicitado, fonte, linhas, páginas conhecidas e versão; impedir mistura
  silenciosa de versões ao continuar uma leitura.
- [x] 4. Eficiência e busca: medir custos, reduzir recomputação demonstrada e preservar
  a busca por relevância e frases. Não acrescentar sinônimos clínicos inventados nem
  presumir que vetores melhoram todos os tipos de consulta.
- [x] 5. Verificação: regressões dos erros reproduzidos, suíte completa e lint;
  registrar medidas e limites. Atualizar instruções, schemas e ajuda das ferramentas.

Dados clínicos reais e qualidade médica das respostas não serão simulados como
validados pelos testes de software. A implantação remota é uma etapa separada.

## Resultado implementado — 20/10/2026

- Importação rejeita assinaturas binárias e corrupção evidente antes de substituir
  uma fonte válida. Extração PDF ilegível aciona OCR no modo automático; se ambas
  as tentativas falham, o texto não é publicado como extração bem-sucedida.
- Coleções usam a versão atual das fontes na próxima consulta. Texto, mapeamento
  de linhas e índice são construídos antes de substituir o conjunto anterior.
  Consultas sem mudanças reutilizam esse conjunto.
- `rlm_read` entrega passagens exatas, com origem, linhas, páginas conhecidas e
  SHA-256. A continuação exige a mesma versão. Testes recompõem integralmente
  texto Unicode com linhas longas a partir das partes retornadas.
- A versão considera todo o texto; uma alteração de mesmo tamanho fora das
  amostras antigas agora invalida o índice. Substituir dados invalida índices
  persistidos na mesma transação SQLite. No reinício, índices vetoriais só são
  reutilizados após conferir todos os trechos e linhas contra a fonte atual.
  Índices divergentes são reconstruídos sob demanda, sem interromper a restauração
  das demais fontes. A conferência usa a divisão padrão de trechos do servidor.
- A busca preserva negações e qualificadores como vitamina A/D/E. A seleção dos
  melhores resultados evita ordenar todos os candidatos e desempata de forma
  estável. A consulta semântica reutiliza o vetor da pergunta entre fontes, em
  cache limitado, sem guardar falhas de API.
- `/metrics` distingue tempo por ferramenta e espera para executar. O tempo de
  rede até o cliente continua exigindo medição externa; routers incluem o tempo
  de seus handlers, portanto os valores não devem ser somados.

## Medidas locais

Ensaio sintético determinístico: 5.446.222 caracteres, 20.000 segmentos, 31
repetições por consulta. Compara a mediana das consultas após a construção do
índice; não usa APIs, rede nem documentos clínicos. Comando:

```bash
PYTHONPATH=src uv run --frozen --extra dev python scripts/benchmark_retrieval.py
```

| Medida | Antes | Depois |
|---|---:|---:|
| Busca dos 10 melhores resultados, índice já construído | 5,7201 ms | 3,9965 ms |
| Primeira construção e consulta BM25 | 543,826 ms | 556,727 ms |
| Verificação de versão, texto já consultado | 0,0657 ms | 0,0003 ms |

A consulta repetida levou aproximadamente **30% menos tempo**, com os mesmos
10 resultados e a mesma ordem. A primeira construção ficou cerca de 2% mais
lenta nesta medição; não há ganho demonstrado nessa etapa. Os valores variam
com a máquina e a carga. Não explicam nem reproduzem os 6,3–21,2 segundos relatados
para o servidor remoto. A verificação de versão agora custa uma leitura completa
na primeira consulta; as seguintes reutilizam o resultado em cache limitado.

## Verificação final

```bash
uv run --frozen --extra dev pytest tests/ -q -rs --tb=short
uv run --frozen --extra dev ruff check src/ tests/ scripts/benchmark_retrieval.py
git diff --check
```

**305 testes passaram; 13 foram ignorados pelas condições do ambiente**
(isolamento de memória específico, Landlock/seccomp, recursos Linux e caminhos
de container ausentes no macOS). Lint e verificação do diff passaram.
O `uv.lock` foi sincronizado com o `ruff` já declarado nas dependências de
desenvolvimento; nenhuma dependência de produção foi acrescentada.

## Adaptação ao Codex

- `AGENTS.md -> CLAUDE.md` no repositório: instruções únicas para os dois agentes.
- Na pasta externa, `AGENTS.md -> CLAUDE.md` e
  `.codex -> rlm-mcp-server/.codex` permitem abrir o projeto por qualquer raiz.
- As instruções comuns distinguem configurações e memórias exclusivas do Claude.
  README inclui conexão do Codex, leitura de passagens e comandos de verificação.
- A pasta externa foi marcada como confiável na configuração pessoal do Codex
  para permitir os ajustes locais. Links, TOML e cadastro MCP foram conferidos
  nas duas raízes; RLM ativo, integrações opcionais desativadas conforme o arquivo
  local preexistente. Arquivos fora do repositório não acompanham um clone Git.

## Limites restantes

O usuário autorizou commit, push e implantação manual no Magalu após as correções
do review. Push para main não publica; o servidor novo não usa Dokploy.
O acervo remoto e suas possíveis duplicatas ainda não foram auditados. Os filtros
detectam corrupção evidente, sem garantir OCR completo ou interpretação clínica.
As páginas retornadas vêm dos marcadores de extração; quando ausentes, são
explicitamente desconhecidas.

A coleção continua lexical; busca semântica/híbrida continua por fonte. Antes de
unificá-las ou acrescentar sinônimos, é necessário comparar resultados em perguntas
representativas com passagens corretas previamente identificadas. A decisão de
manter esse comportamento também respeita a experiência registrada do projeto
com recuperação semântica fraca no corpus homeopático.

Não há atualização automática quando apenas o arquivo S3 muda, nem transação
distribuída envolvendo S3 e SQLite. Recarregar a fonte é necessário. Não foi
demonstrada necessidade de trocar SQLite ou substituir o BM25 por FTS5.

## Medida após correções do review

Mesmo ensaio sintético: construção inicial **643,991 ms**, consulta repetida
**3,9642 ms**, verificação de versão **0,0003 ms**, mesmos dez resultados e ordem.
A preservação mais precisa de qualificadores acrescenta custo na primeira
construção. A medida anterior permanece acima como histórico; não se atribui
ganho à etapa inicial nem à latência remota.
