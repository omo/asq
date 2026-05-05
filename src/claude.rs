use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::claude_types::{Message, MessagesRequest};
use crate::stream::{StreamClient, StreamEvent};

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
        };

        let response = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
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
        tokio::spawn(async move {
            if let Err(e) = read_sse_stream(response, tx).await {
                let _ = e;
            }
        });

        Ok(rx)
    }
}

/// Read SSE events from the response body and send them through the channel.
///
/// Anthropic's SSE format uses lines like:
///   event: content_block_delta
///   data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"..."}}
async fn read_sse_stream(
    response: reqwest::Response,
    tx: mpsc::UnboundedSender<Result<StreamEvent>>,
) -> Result<()> {
    use crate::stream::parse_data_line;
    use tokio::io::AsyncBufReadExt;

    let mut lines = crate::stream::line_reader(response).lines();

    // Track the current event type (from `event:` lines)
    let mut current_event = String::new();

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
            "content_block_delta" => {
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
            "message_stop" => {
                let _ = tx.send(Ok(StreamEvent::Done(None)));
                return Ok(());
            }
            _ => {
                // Ignore other events: message_start, content_block_start,
                // content_block_stop, message_delta, ping
            }
        }
    }

    // Stream ended without message_stop
    let _ = tx.send(Ok(StreamEvent::Done(None)));
    Ok(())
}
