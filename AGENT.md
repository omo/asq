# ASQ (Ask Squpid Quesiton) - CLI-based AI search tool

ASQ is a CLI-based API frontend against LLM and search services.

## Features

ASQ can search the web and answer a question using following APIs:

- Gemini LLM API with Google Search grounding (streaming SSE).

## Architecture

- ASQ is written in Rust.
- Rust doesn't use SDKs from API providers, but hits the HTTP API endpoints directly.

## Pipeline

```
User question
       │
       ▼
┌──────────────────┐
│  Gemini API      │  ← POST /v1beta/models/gemini-3-flash-preview:streamGenerateContent?alt=sse
│  (grounded)      │     request: { contents, tools: [{ googleSearch: {} }] }
└────────┬─────────┘
         │ SSE stream (data: {...} events)
         ▼
┌──────────────────┐
│  Render output   │  ← Print text fragments as they arrive, then show citations
└──────────────────┘
```

## Dependencies

| Crate | Version | Purpose |
|---|---|---|
| [`tokio`](https://crates.io/crates/tokio) | 1 | Async runtime (features: `macros`, `rt-multi-thread`, `io-util`) |
| [`reqwest`](https://crates.io/crates/reqwest) | 0.12 | HTTP client (features: `json`, `rustls-tls`, `stream`) |
| [`serde`](https://crates.io/crates/serde) | 1 | Serialization/deserialization (feature: `derive`) |
| [`serde_json`](https://crates.io/crates/serde_json) | 1 | JSON encoding/decoding |
| [`clap`](https://crates.io/crates/clap) | 4 | CLI argument parsing (features: `derive`, `env`) |
| [`anyhow`](https://crates.io/crates/anyhow) | 1 | Ergonomic error handling |
| [`dotenvy`](https://crates.io/crates/dotenvy) | 0.15 | Load `.env` and `~/.env` for API keys |
| [`colored`](https://crates.io/crates/colored) | 2 | Terminal text coloring (reserved) |
| [`termimad`](https://crates.io/crates/termimad) | 0.31 | Render Markdown in terminal (reserved) |
| [`tracing`](https://crates.io/crates/tracing) | 0.1 | Debug/logging |
| [`tracing-subscriber`](https://crates.io/crates/tracing-subscriber) | 0.3 | Tracing output formatting |
| [`futures-util`](https://crates.io/crates/futures-util) | 0.3 | Stream processing for SSE (features: `std`, `async-await-macro`) |
| [`tokio-util`](https://crates.io/crates/tokio-util) | 0.7 | Stream-to-AsyncRead adapter for SSE line reading (feature: `io`) |

### Feature flags

| Crate | Features |
|---|---|
| `reqwest` | `json` (serde integration), `rustls-tls` (HTTPS), `stream` (byte streaming for SSE) |
| `serde` | `derive` for `#[derive(Serialize, Deserialize)]` |
| `clap` | `derive` for `#[derive(Parser)]`, `env` for `#[arg(env = "...")]` |
| `tokio` | `macros` for `#[tokio::main]`, `rt-multi-thread` for multi-threaded runtime, `io-util` for `BufReader` |

## API configuration

Both API keys can be set via environment variables or loaded from `.env` (project-local) or `~/.env` (home directory):

- `GEMINI_API_KEY` — **required**
- `BRAVE_API_KEY` — optional (not yet implemented)

## Module structure

```
src/
├── main.rs     # Entry point: parse CLI, init clients, stream answer
├── cli.rs      # Clap CLI definitions
├── config.rs   # Config & API key loading
├── brave.rs    # Brave Search API client (stub)
└── gemini.rs   # Gemini API client with streaming SSE + Google Search grounding
```
