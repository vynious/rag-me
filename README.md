# rag-me

Local-first RAG sandbox aimed at answering code/docs questions from your own repos without shipping data to SaaS LLMs. Today it wires Candle for embeddings + inference, SurrealDB for vector storage, and a CLI/HTTP shell around it. Routing and engine-swaps are in progress.

## What’s here
- **Inference**: phi-2 (MixFormer quantized) via Candle, see `src/ai/inference.rs`.
- **Embeddings**: BERT-based embedder via Candle, see `src/ai/embedding.rs`.
- **Vector store**: SurrealDB (RocksDB local) storing content + vectors, see `src/data/database.rs`.
- **CLI REPL**: `run_repl` handles basic commands (`ask`, `upload`, etc.), see `src/cli/runner.rs`.
- **HTTP skeleton**: Axum router started in `main.rs` / `src/http`.

## Architecture (target)
```
CLI / HTTP
   |
rag-me core (commands + sessions)
   |
Router (session -> engine affinity, KV budgeting)   [planned]
   |
Engine trait -> CandleEngine (today)                [future: Ollama / vLLM drop-in]
   |
Vector DB (Surreal) + Corpus chunking (code/docs)
```

## Project status
- Core pieces compile in isolation; wiring still WIP (several modules need re-exports and routing between embedder, VDB, and inference).
- Router/KV modeling and engine-swapping are not implemented yet.
- CLI commands are basic; chat REPL shape exists but will evolve with Router/Session logic.

## Setup
- Rust 2021 toolchain.
- Disk space for phi-2 quantized weights via `hf-hub`.
- Local SurrealDB RocksDB file (`ragme.db`) is created on first use.
- Corpora live on disk; ingestion currently supports simple txt/pdf via CLI.

Build:
```bash
cargo check
```

Run (current shape):
```bash
cargo run
```

## Checklist
- [ ] Fix wiring: clean exports, inject embedder/inference into VDB and CLI, and ensure REPL uses the vector store + inference path.
- [ ] Add `Corpus`/`Session` plumbing and a thin Router with session→engine affinity.
- [ ] Implement repo indexing (chunk ~200–300 lines) and retrieval-backed prompts.
- [ ] Generalize the Engine trait for swapping Candle ↔ Ollama/vLLM once routing is stable.
