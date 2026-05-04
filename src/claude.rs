use anyhow::{Context, Result};
use serde::Serialize;
use tokio::sync::mpsc;

/// Claude API client using Anthropic's Messages API with streaming.
pub struct ClaudeClient {
    api_key: String,
    client: reqwest::Client,
}

// ─── Request structs ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    stream: bool,
    messages: Vec<Message>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

// ─── Public types ─────────────────────────────────────────────────────────

/// An event yielded during streaming from Claude.
#[derive(Debug)]
pub enum ClaudeStreamEvent {
    /// A text fragment to print immediately.
    Text(String),
    /// The response is complete.
    Done,
}

// ─── Client implementation ─────────────────────────────────────────────────

impl ClaudeClient {
    /// Create a new Claude client.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    /// Ask a question using Claude Sonnet, returning a stream of events.
    ///
    /// Text events arrive as the model generates, followed by a final
    /// `ClaudeStreamEvent::Done`.
    pub async fn ask_stream(
        &self,
        query: &str,
    ) -> Result<mpsc::UnboundedReceiver<Result<ClaudeStreamEvent>>> {
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
    tx: mpsc::UnboundedSender<Result<ClaudeStreamEvent>>,
) -> Result<()> {
    use futures_util::StreamExt;
    use tokio::io::AsyncBufReadExt;
    use tokio_util::io::StreamReader;

    let stream = response.bytes_stream();
    let reader = StreamReader::new(
        stream.map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))),
    );
    let mut lines = tokio::io::BufReader::new(reader).lines();

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

        // SSE data lines
        let json_str = match line.strip_prefix("data: ") {
            Some(s) => s,
            None => continue,
        };

        if json_str.is_empty() {
            continue;
        }

        // Parse the JSON payload
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
                        if tx
                            .send(Ok(ClaudeStreamEvent::Text(text.to_string())))
                            .is_err()
                        {
                            return Ok(()); // receiver dropped
                        }
                    }
                }
            }
            "message_stop" => {
                let _ = tx.send(Ok(ClaudeStreamEvent::Done));
                return Ok(());
            }
            _ => {
                // Ignore other events: message_start, content_block_start,
                // content_block_stop, message_delta, ping
            }
        }
    }

    // Stream ended without message_stop
    let _ = tx.send(Ok(ClaudeStreamEvent::Done));
    Ok(())
}
