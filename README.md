# asq - ask stupid question

Ask a question, get an AI answer with web search — in your terminal.

```
$ asq "how tall is Mount Fuji"
Mount Fuji is 3,776 meters (12,389 feet)...

Search queries: Mount Fuji height meters
  https://en.wikipedia.org/wiki/Mount_Fuji
  https://www.britannica.com/place/Mount-Fuji
```

## Install

Requires Rust toolchain.

```
cargo install --path .
```

## Setup

Set your API keys as environment variables, or put them in `$HOME/.env` or a
project-local `.env`:

```
GEMINI_API_KEY=...
CLAUDE_API_KEY=...
OPENAI_API_KEY=...
BRAVE_API_KEY=...
```

## Usage

```
asq [OPTIONS] <QUESTION>
```

### Options

| Flag | Description |
|---|---|
| `-e`, `--engine <ENGINE>` | Backend to use. One of: `gemini` (default), `claude`, `gpt`, `brave` |
| `--gemini-api-key <KEY>` | Gemini API key (or set `GEMINI_API_KEY`) |
| `--claude-api-key <KEY>` | Claude API key (or set `CLAUDE_API_KEY`) |
| `--gpt-api-key <KEY>` | OpenAI API key (or set `OPENAI_API_KEY`) |
| `--brave-api-key <KEY>` | Brave API key (or set `BRAVE_API_KEY`) |

### Engines

- **gemini** — Google Gemini with automatic Google Search grounding. Returns
  search queries and source URLs.
- **claude** — Anthropic Claude with web search tool. Returns source URLs.
- **gpt** — OpenAI GPT with web search. Returns search queries and source URLs.
- **brave** — Brave's chat model. No search metadata in output.

### Examples

```
asq "when is the next total solar eclipse"

asq -e claude "rust async trait Send bound workaround"

asq -e gpt --gpt-api-key $OPENAI_KEY "What's new in PostgreSQL 18"
```

## How it works

Each engine call is a single HTTP request with streaming response. Text
prints as the model generates it. When the stream finishes, search queries
and cited source URLs are printed below the answer.
