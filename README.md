# rag-me

Business goal: a self-hosted RAG copilot for engineering teams—answer codebase and docs questions quickly, keep sensitive material on-device, and shorten ramp-up/maintenance cycles without SaaS dependencies.

Local-first RAG base: embed sources, store in a vector DB, retrieve top-k, and generate answers with a local model. Candle provides embeddings + inference today; the design keeps engines swappable (Candle ↔ Ollama ↔ vLLM) once the routing layer lands.

## Architecture
```
User (CLI / API)
    |
rag-me core (commands, sessions)
    |
KV-aware Router (session reuse, KV budget, eviction)
    |
Engine trait  ──>  CandleEngine (phi-2 quantized)   [future: Ollama, vLLM]
    |
Vector DB (embeddings + metadata)  +  Corpora (code/docs)
```

### Component Roles
- **CLI**: chat REPL (`rag-me chat <corpus>`) plus in-REPL commands (`/session`).
- **Core**: parses commands, tracks sessions, hands chat to the router.
- **Router**: allocates session handles, tracks estimated KV bytes, evicts when over budget, logs events.
- **Engine trait**: unified interface for chat + session lifecycle; upper layers stay agnostic to Candle specifics.
- **CandleEngine**: current implementation; phi-2 quantized via Candle for embedding + generation.
- **Vector store**: chunked corpus docs with top-k retrieval for prompts.

### Status
- Candle/phi-2 quantized inference experiment in `src/ai/inference.rs`.
- Vector embedding + retrieval plumbing exists; CLI + HTTP entrypoints are being shaped with the chat REPL as first-class.
- Engine trait + Router are the next major pieces to land.

### Roadmap
1) **Phase 1 – CLI chat (Candle-backed)**  
   Add `Corpus`, `Session`, `Engine` + `CandleEngine`, and REPL commands: `rag-me chat <corpus>`, `/session new <name>`, `/session list`.
2) **Phase 2 – Repo RAG**  
   Index code/docs (chunk ~200–300 lines), embed, store in vector DB; retrieve top-k per question and inject as `[context]`.
3) **Phase 3 – Router**  
   Track `SessionKey`/`SessionState`, enforce KV budgets, log create/reuse/eviction even without real KV exposure in Candle.
4) **Phase 4 – Engine swaps**  
   Keep the trait stable so Ollama/vLLM can drop in with minimal surface-area changes.

## Getting Started
### Prereqs
- Rust toolchain (2021 edition; `cargo check` should pass).
- Disk space for model download via `hf-hub` (phi-2 quantized).
- Local corpus available on disk; local vector DB file (`ragme.db`).

### Build / Run
```bash
cargo check
cargo run -- chat <corpus>   # REPL entrypoint (in progress)
```

## Usage (target)
- `rag-me chat <corpus>` → session bound to the chosen corpus.
- In-REPL commands: `/session new <name>`, `/session list`.

## Implementation Notes
- Keep the Engine trait thin so engine swaps are mechanical.
- Chunk code around 200–300 lines and include path metadata in prompts for traceability.
- Router KV math is estimated but mirrors vLLM-style handle semantics for future engines.
