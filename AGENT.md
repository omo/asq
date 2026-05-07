# ASQ

A CLI tool that asks questions to AI models with web search grounding. Runs
locally, speaks directly to provider HTTP APIs — no SDKs, no frameworks.

## Architecture

All providers conform to a single trait:

```
trait StreamClient {
    fn new(api_key: String) -> Self;
    async fn ask_stream(&self, query: &str) -> Result<UnboundedReceiver<Result<StreamEvent>>>;
}
```

**`StreamEvent`** is the universal output type — every provider emits it,
the renderer consumes it:

```rust
enum StreamEvent {
    Text(String),                    // print as it arrives
    Done(Option<GroundingData>),     // stream finished, optionally with search metadata
}
```

**`GroundingData`** carries web search queries and cited source URLs when the
provider supports them. Providers that don't support search grounding simply
emit `Done(None)`.

Each provider is a unit struct (e.g. `GeminiClient`) holding an API key and
a `reqwest::Client`. `ask_stream()` POSTs to the provider's endpoint, spawns a
task that parses the SSE response line-by-line, and returns the rx half of an
`mpsc::unbounded_channel()`. The spawned task sends `Text` / `Done` events as
they come in; the channel decouples parsing from rendering.

## Providers

Four backends, all implementing `StreamClient`:

| Provider | Module | API | Search grounding |
|---|---|---|---|
| Gemini | `gemini` | `:streamGenerateContent?alt=sse` | Yes — Google Search grounding, redirect resolution |
| Brave | `brave` | OpenAI-compatible chat completions | No |
| Claude | `claude` | Anthropic Messages API | Yes — web_search tool, collects citations |
| GPT | `gpt` | OpenAI Responses API | Yes — web_search_preview, collects queries + annotations |

Each provider has a companion `*_types.rs` module with serde types for its
request/response wire format.

### Gemini specifics

Google Search grounding returns URLs that are Google redirects. Gemini
resolves them via parallel HEAD requests so the user sees final destination
URLs.

### Claude specifics

Citations arrive in two forms: pre-populated in `content_block_start` text
blocks, and incrementally via `citations_delta` events. Both are collected and
emitted with the final `Done`.

### GPT specifics

Search queries arrive in `response.output_item.done` events (before text
generation); source annotations arrive in `response.completed`. Queries are
de-duplicated across multiple output items.

## Shared SSE infrastructure (`stream.rs`)

Both the `StreamEvent`/`GroundingData` types and two pure functions live here:

- **`line_reader(response) -> AsyncBufRead`** — converts a `reqwest` byte
  stream into a buffered async line reader using `tokio-util::StreamReader`.
  Every provider uses this to get a line iterator.

- **`parse_data_line(line) -> Option<&str>`** — extracts the JSON payload from
  an SSE `data:` line. Providers call this in their inner loop.

## Control flow

```
main()
  ├─ load .env  (HOME/.env first, then ./.env — clap's #[arg(env)] picks them up)
  ├─ parse CLI  (question, --engine, --{provider}-api-key)
  └─ match engine:
       ├─ resolve API key
       └─ run::<ProviderClient>(question, key)
            ├─ client.ask_stream() → spawn SSE task, return rx channel
            └─ print_until_done(rx)
                 ├─ print Text chunks as they arrive
                 └─ when Done: print search queries + source URLs
```

## API keys

Set via environment variables or `--flag` args. Loaded from `$HOME/.env`
(low priority) and `.env` (high priority). Required key depends on the engine:

| Engine | Env var |
|---|---|
| Gemini | `GEMINI_API_KEY` |
| Brave | `BRAVE_API_KEY` |
| Claude | `CLAUDE_API_KEY` |
| GPT | `OPENAI_API_KEY` |

## Design notes

- **No pipeline.** There is no chaining, no multi-step workflow. The
  "pipelining" is just: one HTTP call → SSE stream → print. That's it.

- **Channel, not iterator.** SSE parsing runs in a spawned task and pushes
  events through an unbounded mpsc channel. This keeps the rendering loop
  simple (just recv/print) and the parsing task independently cancellable
  (dropping the receiver closes the channel, the spawned task notices the
  send error and exits).

- **Trait over enum.** A single `run::<C: StreamClient>()` function drives all
  providers. Adding a new backend means: implement `StreamClient`, add a CLI
  variant, add one match arm in `main()`.

- **No SDKs.** Every provider is hit via raw HTTP with serde-typed
  request/response structs. This avoids dependency bloat and opaque
  abstractions.

- **Pure parsing functions are testable.** `parse_events()` (Gemini) and
  `read_sse_stream()` (others) take an `AsyncBufRead`, so tests feed them
  `BufReader<Cursor<&[u8]>>` — no network, no mocking.

- **Output is plain `print!` + `flush`.** No TUI framework. Text appears
  character-by-character as the model generates it.


