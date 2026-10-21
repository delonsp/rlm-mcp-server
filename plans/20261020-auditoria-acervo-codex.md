# Verificação da avaliação do acervo — 20/10/2026

> Diagnóstico do estado **anterior às correções**. As melhorias implementadas nesta
> sessão e sua verificação estão no [plano de precisão e eficiência](20261020-acuracia-eficiencia.md).

## Escopo e conclusão

Revisão do código local na base `0cade55`, testes existentes e quatro verificações
com dados fictícios e SQLite temporário. Nenhum documento do acervo remoto foi
inspecionado, alterado ou reenviado. A versão implantada na VPS não foi comparada.

A avaliação é válida em sua direção geral, com ressalvas sobre funcionalidades
já existentes e sobre observações históricas que o código não comprova. Esta é
uma avaliação da recuperação de documentos, não uma validação clínica das respostas.

## Resultado por ponto

| Ponto | Veredito | Evidência e correção |
|---|---|---|
| Importação | Lacuna confirmada | O tipo `text` aceita conteúdo com assinatura PDF. A leitura local substitui bytes inválidos; S3 tenta Latin-1 quando UTF-8 falha. O modo PDF automático considera o tamanho do texto, sem validar sua legibilidade. A existência desses problemas no acervo real continua sendo um relato a conferir. |
| Busca | Confirmado, com ressalva | Coleções já têm busca por palavras com BM25, que ordena por relevância, e tratamento de frases. Busca semântica/híbrida existe por variável, não como modo da busca de coleção. Sinônimos clínicos e recuperação entre idiomas precisam ser avaliados com exemplos representativos; acrescentar busca semântica não garante melhora. |
| Organização | Parcialmente confirmado | O servidor cria agregados de coleção deliberadamente e mapeia os resultados para variável e linha original. A extração PDF também inclui marcadores de página. Faltam identidade documental e referência estruturada de página/versão de ponta a ponta. Não foi verificada duplicidade indevida no acervo real. |
| Leitura | Confirmado apenas no modo compacto | `rlm_execute` corta stdout em 500 caracteres no modo padrão `compact`; `normal` e `verbose` preservam a saída nesse formatador. Não há parâmetro por chamada nem ferramenta dedicada para ler uma passagem completa por documento/página. Os trechos das buscas têm seus próprios limites. |
| Atualização | Lacuna confirmada e reproduzida | Carga, persistência e indexação já são automatizadas, e a busca individual invalida alguns índices antigos. Porém, substituir uma fonte deixou a coleção retornando o texto anterior até `rebuild`. Não há publicação de uma versão documental única envolvendo arquivo, variável e todos os índices. |
| Desempenho | Números históricos não confirmados | Não foram reproduzidos os 6,3–21,2 s citados. Já existem métricas de duração das requisições, contadores por ferramenta e logs por requisição. Falta separar tempos internos de busca, espera, vetorização e rede para atribuir a causa. |

## Evidências no código

- **Entrada de texto e PDF:** [data_tools.py](../src/rlm_mcp/tools/handlers/data_tools.py),
  `rlm_load_file` (linhas 79–175); [s3_client.py](../src/rlm_mcp/s3_client.py),
  `get_object_text` (101–114); [pdf_parser.py](../src/rlm_mcp/pdf_parser.py),
  `extract_pdf` (242–303). O fallback PDF usa 100 caracteres como limiar e pode
  aceitar extração curta quando OCR falha. Não existe etapa de quarentena/aprovação.
- **Busca e referências:** [collection_tools.py](../src/rlm_mcp/tools/handlers/collection_tools.py),
  `rlm_search_collection` (293 em diante); [search_tools.py](../src/rlm_mcp/tools/handlers/search_tools.py),
  `rlm_search_index` (19 em diante); [collection_builder.py](../src/rlm_mcp/collection_builder.py),
  contrato de mapeamento (9–12). Linha do texto não deve ser apresentada como página.
- **Saída cortada:** [response_formatter.py](../src/rlm_mcp/response_formatter.py),
  `get_verbosity`, `_format_exec_compact` e `_format_exec_normal` (23–122).
  [schemas.py](../src/rlm_mcp/tools/schemas.py) anuncia apenas `code` para `rlm_execute`.
- **Atualização parcial:** [persistence_service.py](../src/rlm_mcp/services/persistence_service.py),
  `invalidate_stale_indices` e `persist_and_index` (48–135);
  [s3_tools.py](../src/rlm_mcp/tools/handlers/s3_tools.py), `skip_if_exists` (80–91);
  [collection_tools.py](../src/rlm_mcp/tools/handlers/collection_tools.py), `rlm_collection_rebuild`.
  A verificação de índice individual não atualiza o agregado de sua coleção.
- **Métricas:** [http_server.py](../src/rlm_mcp/http_server.py), `MetricsCollector`
  (62–144), `/metrics` (955 em diante) e logs do endpoint `/mcp` (1251 em diante).
  Criação de vetores sob demanda e espera no bloqueio global são hipóteses de
  atraso presentes no código, não causas medidas dos tempos relatados.
- **Armazenamento:** [persistence.py](../src/rlm_mcp/persistence.py) guarda variáveis,
  índices, coleções e vetores em SQLite; [indexer.py](../src/rlm_mcp/indexer.py)
  implementa BM25 em Python. Não foi encontrado uso de FTS5 no código atual.

## Verificações executadas

**124 testes passaram**, cobrindo ranking BM25, construção e busca de coleções,
frases, mapeamento de linhas, formatação, invalidação de índices, vetores,
registro de ferramentas e transportes MCP/SSE. Comando, na raiz do repositório:

```bash
.venv/bin/python -m pytest \
  tests/test_bm25.py tests/test_collection_builder.py \
  tests/test_collection_phrase_fallback.py tests/test_collection_search_line_mapping.py \
  tests/test_search_output.py tests/test_review_fixes.py \
  tests/test_embedding_batching.py tests/test_var_info_embeddings.py \
  tests/test_tool_registry.py tests/test_mcp_notifications.py \
  tests/test_sse_transport_contract.py tests/test_sse_session_registry.py -q
```

Quatro verificações adicionais locais, sem serviços externos:

1. Simular bytes iniciados por `%PDF-` no download S3: `get_object_text` decodificou
   e `rlm_load_data(data_type="text")` aceitou e persistiu esse conteúdo.
2. Simular extração de 120 caracteres `�`: `extract_pdf(method="auto")` aceitou
   como sucesso sem acionar OCR. Isso comprova ausência de validação de legibilidade,
   não a frequência do problema em PDFs reais.
3. Formatar stdout com 800 caracteres: o modo compacto exibiu 500 e o aviso
   `[+300c]`; o modo normal preservou os 800.
4. Criar uma coleção, substituir a variável-fonte e consultar: a busca individual
   encontrou o conteúdo novo, enquanto a coleção ainda devolveu o antigo.
   `rlm_collection(action="rebuild")` atualizou o agregado.

Testes de código não medem precisão clínica nem comprovam o estado do acervo remoto.

## Prioridade recomendada

1. Validar extração e preservar identidade, origem, página e versão do documento.
2. Corrigir atualização das coleções e oferecer leitura completa com referência.
   A versão desatualizada foi reproduzida; esse item deve acompanhar as primeiras melhorias.
3. Avaliar uma busca unificada contra um conjunto de perguntas e passagens esperadas,
   comparando o BM25 existente com sinônimos e busca semântica, inclusive entre idiomas.
4. Medir os tempos por etapa antes de escolher otimizações ou mudar o armazenamento.

Não há evidência nesta revisão que justifique trocar o banco. O SQLite oferece
[FTS5 e ordenação por relevância](https://sqlite.org/fts5.html), mas isso seria uma
opção a comparar com o BM25 atual. FTS5 sozinho não resolve texto corrompido,
identidade documental, referências de página ou sincronização de versões.
