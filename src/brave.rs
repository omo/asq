use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::brave_types::{ChatRequest, Message, SseChunk};
use crate::stream::{StreamClient, StreamEvent};

/// Brave Chat Completions API client.
///
/// Uses the OpenAI-compatible `/res/v1/chat/completions` endpoint with
/// Brave's own `brave-pro` model which includes search grounding.
pub struct BraveClient {
    api_key: String,
    client: reqwest::Client,
}

#[async_trait]
impl StreamClient for BraveClient {
    fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    /// Ask a question using Brave's chat model, returning a stream of events.
    ///
    /// Text events arrive as the model generates, followed by a final
    /// `StreamEvent::Done`.
    async fn ask_stream(
        &self,
        query: &str,
    ) -> Result<mpsc::UnboundedReceiver<Result<StreamEvent>>> {
        let url = "https://api.search.brave.com/res/v1/chat/completions";

        let request = ChatRequest {
            stream: true,
            model: "brave-pro".to_string(),
            max_completion_tokens: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: query.to_string(),
            }],
        };

        let response = self
            .client
            .post(url)
            .header("x-subscription-token", &self.api_key)
            .header("Accept", "application/json")
            .header("Accept-Encoding", "gzip")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Brave Chat API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Brave Chat API returned {status}: {body}");
        }

        let (tx, rx) = mpsc::unbounded_channel();
        let reader = crate::stream::line_reader(response);

        // Spawn a task to read the SSE stream
        tokio::spawn(async move {
            if let Err(e) = BraveClient::read_sse_stream(reader, tx).await {
                // Channel might already be closed if receiver was dropped
                let _ = e;
            }
        });

        Ok(rx)
    }
}

// ─── SSE parsing (used by ask_stream, testable with in-memory readers) ─────

impl BraveClient {
    /// Read SSE events from the response body and send them through the channel.
    ///
    /// Uses typed deserialization (see [`SseChunk`]) instead of raw
    /// `serde_json::Value` access.
    async fn read_sse_stream<R: tokio::io::AsyncBufRead + Send + Unpin>(
        reader: R,
        tx: mpsc::UnboundedSender<Result<StreamEvent>>,
    ) -> Result<()> {
        use crate::stream::parse_data_line;
        use tokio::io::AsyncBufReadExt;

        let mut lines = reader.lines();

        while let Some(line) = lines.next_line().await? {
            let line = line.trim().to_string();

            let json_str = match parse_data_line(&line) {
                Some(s) => s,
                None => continue,
            };

            // SSE end marker (OpenAI / Brave format: `data: [DONE]`)
            if json_str == "[DONE]" {
                let _ = tx.send(Ok(StreamEvent::Done(None)));
                return Ok(());
            }

            let chunk: SseChunk = match serde_json::from_str(json_str) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!(
                        "Failed to parse Brave SSE JSON: {e}"
                    )));
                    return Ok(());
                }
            };

            for choice in chunk.choices {
                // Extract content delta first — the final chunk often
                // carries both text and finish_reason.
                if let Some(content) = &choice.delta.content {
                    if !content.is_empty() {
                        // Skip usage/cost metadata embedded as <usage>...</usage>
                        if content.starts_with("<usage>") {
                            continue;
                        }
                        if tx.send(Ok(StreamEvent::Text(content.clone()))).is_err() {
                            return Ok(()); // receiver dropped
                        }
                    }
                }

                // Check finish_reason after extracting text
                if choice.finish_reason.as_deref() == Some("stop") {
                    let _ = tx.send(Ok(StreamEvent::Done(None)));
                    return Ok(());
                }
            }
        }

        // Stream ended without a [DONE] or finish_reason
        let _ = tx.send(Ok(StreamEvent::Done(None)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    /// Helper: feed bytes through the SSE parser and collect all events.
    async fn parse_bytes(data: &[u8]) -> Vec<StreamEvent> {
        let reader = tokio::io::BufReader::new(std::io::Cursor::new(data.to_vec()));
        let (tx, mut rx) = mpsc::unbounded_channel();
        BraveClient::read_sse_stream(reader, tx).await.unwrap();

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event.unwrap());
        }
        events
    }

    // ── Basic text emission ─────────────────────────────────────────────────

    #[tokio::test]
    async fn text_delta_emitted() {
        let data = br#"data: {"choices":[{"delta":{"content":"Hello "}}]}
data: {"choices":[{"delta":{"content":"world"}}]}
data: [DONE]
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "Hello "));
        assert!(matches!(&events[1], StreamEvent::Text(s) if s == "world"));
        assert!(matches!(&events[2], StreamEvent::Done(None)));
    }

    // ── finish_reason="stop" terminates early ──────────────────────────────

    #[tokio::test]
    async fn finish_reason_stop_emits_done_without_done_marker() {
        let data = br#"data: {"choices":[{"delta":{"content":"hello"}}]}
data: {"choices":[{"finish_reason":"stop"}]}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hello"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }

    #[tokio::test]
    async fn text_and_finish_reason_in_same_chunk_emits_both() {
        // The final SSE chunk carries both a text delta and
        // finish_reason: "stop" inside the same choice object.
        let data = br#"data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hello"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }

    // ── <usage> chunks are filtered ────────────────────────────────────────

    #[tokio::test]
    async fn usage_content_is_filtered() {
        let data = br#"data: {"choices":[{"delta":{"content":"<usage>token_count:42</usage>"}}]}
data: {"choices":[{"delta":{"content":"real text"}}]}
data: [DONE]
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "real text"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }

    // ── Multiple choices in one chunk ───────────────────────────────────────

    #[tokio::test]
    async fn multiple_choices_each_produce_text() {
        let data = br#"data: {"choices":[{"delta":{"content":"a"}},{"delta":{"content":"b"}}]}
data: [DONE]
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "a"));
        assert!(matches!(&events[1], StreamEvent::Text(s) if s == "b"));
        assert!(matches!(&events[2], StreamEvent::Done(None)));
    }

    // ── Stream ends without [DONE] or finish_reason ─────────────────────────

    #[tokio::test]
    async fn stream_ends_without_done_emits_done() {
        let data = br#"data: {"choices":[{"delta":{"content":"hello"}}]}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hello"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }

    // ── Malformed JSON sends an error ──────────────────────────────────────

    #[tokio::test]
    async fn malformed_json_sends_error() {
        let reader = tokio::io::BufReader::new(std::io::Cursor::new(
            br#"data: {not json}
"#.to_vec(),
        ));
        let (tx, mut rx) = mpsc::unbounded_channel();
        BraveClient::read_sse_stream(reader, tx).await.unwrap();

        let event = rx.recv().await;
        assert!(event.is_some());
        assert!(event.unwrap().is_err());
    }

    // ── Empty lines and non-data lines are ignored ─────────────────────────

    #[tokio::test]
    async fn empty_lines_and_event_lines_are_skipped() {
        let data = br#"

event: ping
data: {"choices":[{"delta":{"content":"hi"}}]}

event: done
data: [DONE]
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hi"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }
}
