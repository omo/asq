use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::brave_types::{ChatRequest, Message};
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

        // Spawn a task to read the SSE stream
        tokio::spawn(async move {
            if let Err(e) = read_sse_stream(response, tx).await {
                // Channel might already be closed if receiver was dropped
                let _ = e;
            }
        });

        Ok(rx)
    }
}

/// Read SSE events from the response body and send them through the channel.
async fn read_sse_stream(
    response: reqwest::Response,
    tx: mpsc::UnboundedSender<Result<StreamEvent>>,
) -> Result<()> {
    use crate::stream::parse_data_line;
    use tokio::io::AsyncBufReadExt;

    let mut lines = crate::stream::line_reader(response).lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();

        if line.is_empty() {
            continue;
        }

        // SSE end marker
        if line == "[DONE]" {
            let _ = tx.send(Ok(StreamEvent::Done(None)));
            return Ok(());
        }

        let json_str = match parse_data_line(&line) {
            Some(s) => s,
            None => continue,
        };

        // Parse the JSON chunk
        let chunk: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(e) => {
                let _ = tx.send(Err(anyhow::anyhow!(
                    "Failed to parse Brave SSE JSON: {e}"
                )));
                return Ok(());
            }
        };

        // Extract text delta from choices[0].delta.content
        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
            for choice in choices {
                // Check finish_reason first
                if let Some(finish) = choice.get("finish_reason") {
                    if finish.is_string() && finish.as_str().unwrap_or("") == "stop" {
                        let _ = tx.send(Ok(StreamEvent::Done(None)));
                        return Ok(());
                    }
                }

                // Extract content delta
                if let Some(content) = choice
                    .get("delta")
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    if !content.is_empty() {
                        // Skip usage/cost metadata embedded as <usage>...</usage>
                        if content.starts_with("<usage>") {
                            continue;
                        }
                        if tx.send(Ok(StreamEvent::Text(content.to_string()))).is_err() {
                            return Ok(()); // receiver dropped
                        }
                    }
                }
            }
        }
    }

    // Stream ended without a [DONE] or finish_reason
    let _ = tx.send(Ok(StreamEvent::Done(None)));
    Ok(())
}
