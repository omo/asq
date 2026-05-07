use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::claude_types::{ContentBlock, Delta, Message, MessagesRequest, SseEvent, Tool};
use crate::stream::{GroundingData, Source, StreamClient, StreamEvent};

/// Claude API client using Anthropic's Messages API with streaming.
pub struct ClaudeClient {
    api_key: String,
    client: reqwest::Client,
    system_prompt: Option<String>,
}

#[async_trait]
impl StreamClient for ClaudeClient {
    fn new(api_key: String, system_prompt: Option<String>) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            system_prompt,
        }
    }

    /// Ask a question using Claude Sonnet, returning a stream of events.
    ///
    /// Text events arrive as the model generates, followed by a final
    /// `StreamEvent::Done`.
    async fn ask_stream(
        &self,
        query: &str,
    ) -> Result<mpsc::UnboundedReceiver<Result<StreamEvent>>> {
        let url = "https://api.anthropic.com/v1/messages";

        let request = MessagesRequest {
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 4096,
            stream: true,
            system: self.system_prompt.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: query.to_string(),
            }],
            tools: Some(vec![Tool {
                tool_type: "web_search_20250305".to_string(),
                name: "web_search".to_string(),
            }]),
        };

        let response = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "web-search-2025-03-05")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Claude API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Claude API returned {status}: {body}");
        }

        let (tx, rx) = mpsc::unbounded_channel();
        let reader = crate::stream::line_reader(response);

        // Spawn a task to read the SSE stream
        let question = query.to_string();
        tokio::spawn(async move {
            if let Err(e) = ClaudeClient::read_sse_stream(reader, tx, &question).await {
                let _ = e;
            }
        });

        Ok(rx)
    }
}

// ─── SSE parsing (used by ask_stream, testable with in-memory readers) ─────

impl ClaudeClient {
    /// Read SSE events from the response body and send them through the channel.
    ///
    /// Anthropic's SSE format:
    ///   event: content_block_delta
    ///   data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"..."}}
    ///
    /// Uses typed deserialization (see [`SseEvent`]) instead of raw
    /// `serde_json::Value` access.
    async fn read_sse_stream<R: tokio::io::AsyncBufRead + Send + Unpin>(
        reader: R,
        tx: mpsc::UnboundedSender<Result<StreamEvent>>,
        query: &str,
    ) -> Result<()> {
        use crate::stream::parse_data_line;
        use tokio::io::AsyncBufReadExt;

        let mut lines = reader.lines();

        // Collect citations for the final Done event
        let mut sources: Vec<Source> = Vec::new();

        while let Some(line) = lines.next_line().await? {
            let json_str = match parse_data_line(line.trim()) {
                Some(s) => s,
                None => continue,
            };

            let event: SseEvent = match serde_json::from_str(json_str) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!(
                        "Failed to parse Claude SSE JSON: {e}"
                    )));
                    return Ok(());
                }
            };

            match event {
                SseEvent::ContentBlockStart {
                    content_block:
                        ContentBlock::Text(text_block),
                    ..
                } => {
                    // Collect pre-existing citations from the text block
                    for citation in text_block.citations {
                        if !citation.url.is_empty() {
                            sources.push(Source {
                                uri: citation.url,
                            });
                        }
                    }
                }
                SseEvent::ContentBlockDelta { delta, .. } => match delta {
                    Delta::Text(td) => {
                        if !td.text.is_empty() {
                            if tx
                                .send(Ok(StreamEvent::Text(td.text)))
                                .is_err()
                            {
                                return Ok(()); // receiver dropped
                            }
                        }
                    }
                    Delta::Citations(cd) => {
                        if !cd.citation.url.is_empty() {
                            sources.push(Source {
                                uri: cd.citation.url,
                            });
                        }
                    }
                    // Ignore thinking / signature / input_json deltas
                    Delta::Thinking | Delta::Signature | Delta::InputJson => {}
                },
                SseEvent::MessageStop => {
                    let grounding = Some(GroundingData {
                        web_search_queries: vec![query.to_string()],
                        sources,
                    });
                    let _ = tx.send(Ok(StreamEvent::Done(grounding)));
                    return Ok(());
                }
                // Ignore: content_block_stop, message_start, message_delta, ping
                _ => {}
            }
        }

        // Stream ended without message_stop
        let _ = tx.send(Ok(StreamEvent::Done(None)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    /// Helper: feed bytes through the Claude SSE parser and collect all events.
    async fn parse_bytes(data: &[u8]) -> Vec<StreamEvent> {
        let reader = tokio::io::BufReader::new(std::io::Cursor::new(data.to_vec()));
        let (tx, mut rx) = mpsc::unbounded_channel();
        ClaudeClient::read_sse_stream(reader, tx, "test query")
            .await
            .unwrap();

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event.unwrap());
        }
        events
    }

    // ── Basic text emission ─────────────────────────────────────────────────

    #[tokio::test]
    async fn text_delta_emitted() {
        let data = br#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello "}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}
event: message_stop
data: {"type":"message_stop"}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "Hello "));
        assert!(matches!(&events[1], StreamEvent::Text(s) if s == "world"));
        assert!(matches!(&events[2], StreamEvent::Done(Some(g)) if g.web_search_queries == ["test query"]));
    }

    // ── message_stop emits Done with grounding data ────────────────────────

    #[tokio::test]
    async fn message_stop_emits_done_with_sources() {
        let data = br#"event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"Hi","citations":[{"url":"https://example.com"}]}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}
event: message_stop
data: {"type":"message_stop"}
"#;
        let events = parse_bytes(data).await;
        // content_block_start only collects citations (no Text event),
        // content_block_delta emits Text, message_stop emits Done with sources.
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == " there"));
        match &events[1] {
            StreamEvent::Done(Some(g)) => {
                assert_eq!(g.web_search_queries, &["test query"]);
                assert_eq!(g.sources.len(), 1);
                assert_eq!(g.sources[0].uri, "https://example.com");
            }
            _ => panic!("expected Done(Some(…))"),
        }
    }

    // ── Citations collected from content_block_start ───────────────────────

    #[tokio::test]
    async fn citations_from_content_block_start() {
        let data = br#"event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"A","citations":[{"url":"https://a.com"},{"url":"https://b.com"}]}}
event: message_stop
data: {"type":"message_stop"}
"#;
        let events = parse_bytes(data).await;
        // content_block_start only collects citations, no Text event.
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::Done(Some(g)) => {
                assert_eq!(g.sources.len(), 2);
                assert_eq!(g.sources[0].uri, "https://a.com");
                assert_eq!(g.sources[1].uri, "https://b.com");
            }
            _ => panic!("expected Done(Some(…))"),
        }
    }

    // ── Citations collected from citations_delta ───────────────────────────

    #[tokio::test]
    async fn citations_from_delta() {
        let data = br#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"url":"https://c.com"}}}
event: message_stop
data: {"type":"message_stop"}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::Done(Some(g)) => {
                assert_eq!(g.sources.len(), 1);
                assert_eq!(g.sources[0].uri, "https://c.com");
            }
            _ => panic!("expected Done(Some(…))"),
        }
    }

    // ── thinking_delta / signature_delta / input_json_delta are ignored ────

    #[tokio::test]
    async fn thinking_and_input_json_deltas_ignored() {
        let data = br#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"..."}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"..."}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}
event: message_stop
data: {"type":"message_stop"}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 1); // only message_stop
        assert!(matches!(&events[0], StreamEvent::Done(Some(_))));
    }

    // ── Stream ends without message_stop ───────────────────────────────────

    #[tokio::test]
    async fn stream_ends_without_message_stop() {
        let data = br#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}
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
            br#"event: content_block_delta
data: {not json}
"#.to_vec(),
        ));
        let (tx, mut rx) = mpsc::unbounded_channel();
        ClaudeClient::read_sse_stream(reader, tx, "q")
            .await
            .unwrap();

        let event = rx.recv().await;
        assert!(event.is_some());
        assert!(event.unwrap().is_err());
    }

    // ── Empty lines, event: lines without data, etc. are ignored ───────────

    #[tokio::test]
    async fn empty_and_event_lines_ignored() {
        let data = br#"

event: ping
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}
event: message_start
data: {"type":"message_start","message":{"id":"msg_1"}}
event: message_stop
data: {"type":"message_stop"}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hi"));
        assert!(matches!(&events[1], StreamEvent::Done(Some(_))));
    }
}
