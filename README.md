# Cortex Console

Local-first retrieval-augmented generation (RAG) with a TUI/CLI shell. Everything runs on your machine: embedding, vector storage, and generation—no SaaS LLMs.

## Why it’s useful
- **Private by default**: all embedding and generation stay local.
- **Batteries included**: Candle handles Granite embeddings and OLMo generation; SurrealDB stores vectors; Ratatui/Crossterm power the CLI.
- **Chunk + retrieve**: ingest txt/pdf, chunk content, store vectors, and query with cosine similarity.
- **Concurrent inference**: Tokio worker pool feeds multiple OLMo workers without blocking the REPL.

## What’s inside
- Generation: OLMo via Candle with shard-aware safetensors loading (`src/ai/inference.rs`).
- Embeddings: Granite sparse embedder via Candle (`src/ai/embedding.rs`).
- Vector store: SurrealDB RocksDB on disk (`ragme.db`) (`src/data/database.rs`).
- TUI/CLI: commands like `ask`, `upload`, `list`, `forget` (`src/cli/runner.rs`).
- HTTP scaffold: Axum starter in `src/http` (not wired yet).

## Quick start
Prereqs: Rust 2021 toolchain, disk space for model weights (fetched via `hf-hub`), optional GPU/MPS for Candle acceleration.

```bash
git clone https://github.com/your-org/cortex-console.git
cd cortex-console
cargo run
```

First run downloads:
- Generator: `allenai/Olmo-3-7B-Think`
- Embedder: `ibm-granite/granite-embedding-30m-sparse`

## Using the CLI
After `cargo run`, the REPL shows available commands. Examples:

```text
ask "how do I configure SurrealDB?"
upload ./docs/notes.pdf        # ingest pdf
upload ./notes.txt             # ingest txt
list --start 0 --limit 10      # list stored content
forget --content-id <id>       # remove one item
forget --all                   # wipe everything
```

Responses use retrieval-backed prompts: chunks are fetched from SurrealDB and passed to the OLMo workers via the inference pool.

## How it’s wired
```
TUI / (future HTTP)
   |
AI service (embedder + inference pool)  <-- src/ai
   |
SurrealDB (RocksDB) + corpus files      <-- src/data
```

## Development
- Build: `cargo check`
- Run: `cargo run`
- Key files: `src/ai/inference.rs`, `src/ai/embedding.rs`, `src/ai/worker_pool.rs`, `src/data/database.rs`, `src/cli/runner.rs`, `src/main.rs`.
