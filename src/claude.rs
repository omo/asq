use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::claude_types::{Message, MessagesRequest, Tool};
use crate::stream::{GroundingData, Source, StreamClient, StreamEvent};

/// Claude API client using Anthropic's Messages API with streaming.
pub struct ClaudeClient {
    api_key: String,
    client: reqwest::Client,
}

#[async_trait]
impl StreamClient for ClaudeClient {
    fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
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

        // Spawn a task to read the SSE stream
        let question = query.to_string();
        tokio::spawn(async move {
            if let Err(e) = read_sse_stream(response, tx, &question).await {
                let _ = e;
            }
        });

        Ok(rx)
    }
}

/// Read SSE events from the response body and send them through the channel.
///
/// Anthropic's SSE format:
///   event: content_block_delta
///   data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"..."}}
///
/// With web_search tool we also get:
///   - content_block_start with type: "server_tool_use" or "web_search_tool_result"
///   - content_block_delta with type: "input_json_delta" (tool query)
///   - content_block_delta with type: "citations_delta"
///   - content_block_delta with type: "thinking_delta"
async fn read_sse_stream(
    response: reqwest::Response,
    tx: mpsc::UnboundedSender<Result<StreamEvent>>,
    query: &str,
) -> Result<()> {
    use crate::stream::parse_data_line;
    use tokio::io::AsyncBufReadExt;

    let mut lines = crate::stream::line_reader(response).lines();

    // Track the current event type (from `event:` lines)
    let mut current_event = String::new();
    // Collect citations for the final Done event
    let mut sources: Vec<Source> = Vec::new();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();

        if line.is_empty() {
            continue;
        }

        if let Some(event_type) = line.strip_prefix("event: ") {
            current_event = event_type.to_string();
            continue;
        }

        let json_str = match parse_data_line(&line) {
            Some(s) => s,
            None => continue,
        };

        let payload: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(e) => {
                let _ = tx.send(Err(anyhow::anyhow!(
                    "Failed to parse Claude SSE JSON: {e}"
                )));
                return Ok(());
            }
        };

        match current_event.as_str() {
            "content_block_start" => {
                // If a text block starts with pre-existing citations, collect them
                let block_type = payload
                    .get("content_block")
                    .and_then(|b| b.get("type"))
                    .and_then(|t| t.as_str());
                if block_type == Some("text") {
                    if let Some(citations) = payload
                        .get("content_block")
                        .and_then(|b| b.get("citations"))
                        .and_then(|c| c.as_array())
                    {
                        for citation in citations {
                            if let Some(url) = citation.get("url").and_then(|u| u.as_str()) {
                                if !url.is_empty() {
                                    sources.push(Source { uri: url.to_string() });
                                }
                            }
                        }
                    }
                }
            }
            "content_block_delta" => {
                let delta_type = payload
                    .get("delta")
                    .and_then(|d| d.get("type"))
                    .and_then(|t| t.as_str());

                match delta_type {
                    Some("text_delta") => {
                        // Extract text from delta
                        if let Some(text) = payload
                            .get("delta")
                            .and_then(|d| d.get("text"))
                            .and_then(|t| t.as_str())
                        {
                            if !text.is_empty() {
                                if tx.send(Ok(StreamEvent::Text(text.to_string()))).is_err() {
                                    return Ok(()); // receiver dropped
                                }
                            }
                        }
                    }
                    Some("citations_delta") => {
                        // Collect citation (URL only, drop title for consistency)
                        if let Some(citation) = payload.get("delta").and_then(|d| d.get("citation"))
                        {
                            if let Some(url) = citation.get("url").and_then(|u| u.as_str()) {
                                if !url.is_empty() {
                                    sources.push(Source { uri: url.to_string() });
                                }
                            }
                        }
                    }
                    Some("thinking_delta") | Some("signature_delta") => {
                        // Ignore thinking/signature blocks
                    }
                    Some("input_json_delta") => {
                        // Search query being constructed — ignore
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                // No-op, block ended
            }
            "message_delta" => {
                // Could check stop_reason here if needed
            }
            "message_stop" => {
                let grounding = Some(GroundingData {
                    web_search_queries: vec![query.to_string()],
                    sources,
                });
                let _ = tx.send(Ok(StreamEvent::Done(grounding)));
                return Ok(());
            }
            _ => {
                // Ignore: message_start, ping
            }
        }
    }

    // Stream ended without message_stop
    let _ = tx.send(Ok(StreamEvent::Done(None)));
    Ok(())
}
