use anyhow::Result;
use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::io::{AsyncBufRead, BufReader};
use tokio::sync::mpsc;
use tokio_util::io::StreamReader;

/// Create a buffered line reader from a `reqwest::Response` body stream.
///
/// Call `.lines()` on the result to get an async line iterator, or
/// `.read_line()` for line-by-line reading.
pub fn line_reader(response: reqwest::Response) -> impl AsyncBufRead + Send {
    let stream = response
        .bytes_stream()
        .map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
    let reader = StreamReader::new(stream);
    BufReader::new(reader)
}

/// Extract the JSON payload from an SSE `data:` line.
///
/// Returns `None` if the line doesn't start with `data: ` or is empty.
pub fn parse_data_line<'a>(line: &'a str) -> Option<&'a str> {
    let json_str = line.strip_prefix("data: ")?;
    if json_str.is_empty() {
        return None;
    }
    Some(json_str)
}

// ─── Event types ───────────────────────────────────────────────────────────

/// A source URL referenced by the model.
#[derive(Debug, Clone)]
pub struct Source {
    pub uri: String,
}

/// Search-grounding data attached to a `Done` event.
///
/// Emitted when the provider supports grounded search (e.g. Gemini with
/// Google Search). Providers without grounding emit `Done(None)`.
#[derive(Debug, Clone)]
pub struct GroundingData {
    pub web_search_queries: Vec<String>,
    pub sources: Vec<Source>,
}

/// An event yielded during streaming from any provider.
///
/// - `Text` — a text fragment to print immediately.
/// - `Done` — the response is complete, with optional grounding data.
#[derive(Debug)]
pub enum StreamEvent {
    Text(String),
    Done(Option<GroundingData>),
}

// ─── Client trait ──────────────────────────────────────────────────────────

/// Common interface for all LLM API clients.
///
/// Every provider returns the same [`StreamEvent`] type, so callers can
/// handle responses uniformly regardless of the backend.
#[async_trait]
pub trait StreamClient: Send + Sync {
    /// Create a new client with the given API key.
    fn new(api_key: String) -> Self
    where
        Self: Sized;

    /// Send a question and return a stream of response events.
    async fn ask_stream(
        &self,
        query: &str,
    ) -> Result<mpsc::UnboundedReceiver<Result<StreamEvent>>>;
}
