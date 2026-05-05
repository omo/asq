use anyhow::Result;
use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::io::{AsyncBufRead, BufReader};
use tokio::sync::mpsc;
use tokio_util::io::StreamReader;

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_data_line tests ────────────────────────────────────────────────

    #[test]
    fn parse_data_line_returns_json_payload() {
        assert_eq!(parse_data_line("data: {\"foo\":1}"), Some("{\"foo\":1}"));
    }

    #[test]
    fn parse_data_line_empty_payload_is_none() {
        assert_eq!(parse_data_line("data: "), None);
    }

    #[test]
    fn parse_data_line_no_prefix_is_none() {
        assert_eq!(parse_data_line(""), None);
    }

    #[test]
    fn parse_data_line_event_line_is_none() {
        assert_eq!(parse_data_line("event: ping"), None);
    }

    #[test]
    fn parse_data_line_done_marker_is_none() {
        assert_eq!(parse_data_line("[DONE]"), None);
    }

    // ── StreamEvent / GroundingData tests ────────────────────────────────────

    #[test]
    fn stream_event_text_holds_string() {
        match StreamEvent::Text("hello".into()) {
            StreamEvent::Text(s) => assert_eq!(s, "hello"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn stream_event_done_none() {
        match StreamEvent::Done(None) {
            StreamEvent::Done(None) => {}
            _ => panic!("expected Done(None)"),
        }
    }

    #[test]
    fn stream_event_done_with_grounding() {
        let data = GroundingData {
            web_search_queries: vec!["test query".into()],
            sources: vec![Source { uri: "https://example.com".into() }],
        };
        match StreamEvent::Done(Some(data)) {
            StreamEvent::Done(Some(g)) => {
                assert_eq!(g.web_search_queries, &["test query"]);
                assert_eq!(g.sources[0].uri, "https://example.com");
            }
            _ => panic!("expected Done(Some(…))"),
        }
    }

    #[test]
    fn grounding_data_can_be_cloned() {
        let a = GroundingData {
            web_search_queries: vec!["q".into()],
            sources: vec![Source { uri: "https://x.com".into() }],
        };
        let b = a.clone();
        assert_eq!(b.web_search_queries, &["q"]);
        assert_eq!(b.sources[0].uri, "https://x.com");
    }
}

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
