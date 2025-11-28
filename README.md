# Cortex Console

Local-first RAG in a TUI/CLI wrapper. Everything runs on your machine: embedding, retrieval, and generation. Candle handles the models, SurrealDB handles vectors, Axum wiring is stubbed for future HTTP.

## What’s here (today)
- **Generation**: OLMo via Candle (`src/ai/inference.rs`), with shard-aware safetensors loading and tokenizer merge-pair fallback so newer tokenizers still parse.
- **Embeddings**: BERT-based embedder (currently `ibm-granite/granite-embedding-30m-sparse`) via Candle (`src/ai/embedding.rs`).
- **Vector store**: SurrealDB RocksDB on disk (`ragme.db`) storing chunks + vectors (`src/data/database.rs`).
- **TUI/CLI**: REPL commands (`ask`, `upload`, etc.) in `src/cli/runner.rs`.
- **HTTP skeleton**: Axum scaffold in `main.rs` / `src/http` for later UI/API.

## How it’s wired
```
CLI / (future HTTP)
   |
AI service (embedder + inference pool)  <-- see src/ai
   |
SurrealDB vector store + corpus files   <-- see src/data
```

## Setup
- Rust 2021 toolchain.
- Disk for model weights (downloads via `hf-hub`; defaults: OLMo for gen, Granite embedder).
- Local RocksDB file is created automatically (`ragme.db`).

Build:
```bash
cargo check
```

Run:
```bash
cargo run
```

## Next steps (roadmap)
- Finish router/session plumbing and retrieval-backed prompting.
- Broaden ingestion (repo-aware chunking) and CLI ergonomics.
- Allow engine swaps (e.g., Ollama/vLLM) once routing is in place.
