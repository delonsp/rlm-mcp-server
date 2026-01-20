# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-20

### Added

#### SQLite Performance (WAL Mode)
- PRAGMA journal_mode=WAL for better concurrent read/write performance
- PRAGMA synchronous=NORMAL for improved write performance
- PRAGMA cache_size=-64000 (64MB cache) for better query performance

#### Persistence Error Visibility
- `SHOW_PERSISTENCE_ERRORS` constant to control error visibility in tool output
- Persistence errors now visible in `rlm_load_s3` and `rlm_load_data` output when enabled

#### Pagination Support
- Added `offset` and `limit` parameters to `rlm_search_index`
- Added `offset` and `limit` parameters to `rlm_search_collection`
- Added `offset` and `limit` parameters to `rlm_list_vars`
- Added `offset` and `limit` parameters to `rlm_list_s3`

#### REPL Helper Functions
- `buscar(texto, termo)` - Search for term in text (case-insensitive), returns list of matches with position, line, and context
- `contar(texto, termo)` - Count occurrences of term (case-insensitive), returns total and per-line counts
- `extrair_secao(texto, inicio, fim)` - Extract sections between markers (case-insensitive)
- `resumir_tamanho(bytes)` - Convert bytes to human-readable format (e.g., "1.5 MB")

#### MCP Resources (Spec Compliance)
- Support for `resources/list` method in MCP protocol
- `rlm://variables` resource - lists persisted variables with metadata
- `rlm://memory` resource - shows REPL memory usage (total, used, free)
- `rlm://collections` resource - lists collections with variable counts
- Resources capability advertised in MCP initialize response

#### Rate Limiting
- `RateLimiter` class with sliding window algorithm
- 100 requests/minute limit for SSE/MCP sessions (configurable via `RLM_SSE_RATE_LIMIT`)
- 10 uploads/minute limit for `rlm_upload_url` (configurable via `RLM_UPLOAD_RATE_LIMIT`)
- HTTP 429 Too Many Requests response when limits exceeded

#### Observability
- Structured JSON logging (enable with `LOG_FORMAT=json` environment variable)
- `/metrics` endpoint with request counts, error counts, and latency statistics (avg, p50, p95, p99, max)
- `X-Request-Id` header on all HTTP responses for distributed tracing
- Request ID included in log messages and error responses

### Changed

- Updated documentation in CLAUDE.md with all new features
- Added docstrings to all public functions

## [0.1.0] - 2025-01-01

### Added

- Initial release
- HTTP/SSE MCP server with FastAPI
- Sandboxed Python REPL for code execution
- SQLite persistence for variables and semantic indices
- Automatic semantic indexing for large texts (100k+ characters)
- MinIO/S3 integration for file storage
- PDF parsing with pdfplumber and Mistral OCR
- Collection management for grouping variables
- LLM client for sub-calls

### Tools

- `rlm_execute` - Execute Python code in sandboxed REPL
- `rlm_load_data` - Load data directly (string)
- `rlm_load_file` - Load file from /data volume
- `rlm_load_s3` - Load file from MinIO/S3
- `rlm_list_vars` - List variables in REPL
- `rlm_var_info` - Get variable info
- `rlm_clear` - Clear variables
- `rlm_memory` - Memory usage statistics
- `rlm_list_buckets` - List S3 buckets
- `rlm_list_s3` - List S3 objects
- `rlm_upload_url` - Generate presigned upload URL
- `rlm_process_pdf` - Extract text from PDF
- `rlm_search_index` - Search semantic index
- `rlm_persistence_stats` - Persistence statistics
- `rlm_collection_create` - Create collection
- `rlm_collection_add` - Add variables to collection
- `rlm_collection_list` - List collections
- `rlm_collection_info` - Collection details
- `rlm_search_collection` - Search across collection
