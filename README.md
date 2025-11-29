# Cortex Console

Local-first retrieval-augmented generation (RAG) in a TUI/CLI shell. Embedding, vector storage, and generation all run on your machine—no SaaS dependencies.

## What it does
- Ingest txt/pdf files, chunk them, embed locally, and store vectors.
- Retrieve similar chunks with cosine similarity and answer queries using OLMo generation.
- Run entirely offline once weights are cached.

## Interesting techniques
- Shard-aware safetensors loading for large models to keep startup lean ([`src/ai/inference.rs`](src/ai/inference.rs)).
- Merge-pair tokenizer fallback to handle newer tokenizer JSON formats without upgrading the tokenizer crate ([`src/ai/inference.rs`](src/ai/inference.rs)).
- Tokio worker pool with `mpsc` + `oneshot` channels and `spawn_blocking` to drive concurrent generation without blocking the CLI ([`src/ai/worker_pool.rs`](src/ai/worker_pool.rs)).
- Retrieval prepends adjacent chunks to widen context before answering ([`src/qa/mod.rs`](src/qa/mod.rs)).
- Simple cosine-similarity ranking directly inside SurrealDB ([`src/data/database.rs`](src/data/database.rs)).

## Notable libraries
- [Candle](https://github.com/huggingface/candle) for inference and embeddings.
- [OLMo](https://huggingface.co/allenai/Olmo-3-7B-Think) for generation.
- [Granite sparse embedder](https://huggingface.co/ibm-granite/granite-embedding-30m-sparse) for embeddings.
- [SurrealDB](https://surrealdb.com/) (RocksDB backend) for local vector storage.
- [Ratatui](https://github.com/tui-rs-revival/ratatui) + [Crossterm](https://github.com/crossterm-rs/crossterm) for the TUI/CLI.
- [hf-hub](https://github.com/huggingface/hf-hub) for model artifact fetching.
- [pdf-extract](https://crates.io/crates/pdf-extract) for PDF ingestion.

## Project structure
```
Cargo.toml
README.md
context/
ragme.db
src/
  ai/
  cli/
  data/
  http/
  qa/
  router/
  utils.rs
target/
```
- `src/ai`: embedding, inference (OLMo), and worker pool.
- `src/cli`: Ratatui/Crossterm REPL commands (`ask`, `upload`, `list`, `forget`).
- `src/data`: SurrealDB access and ingestion (txt/pdf).
- `src/qa`: retrieval + context assembly for answers.
- `src/http`: Axum scaffold (future API).
- `context`: local artifacts; `ragme.db`: RocksDB file.

## TODOs (near-term)
- Run blocking actions off the UI task: send commands over a channel to a background service and receive replies via oneshot; log results in the UI pane.
- Start the TUI immediately, load AI + VDB in a background task, and surface readiness/status updates without blocking initial render.
