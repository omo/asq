use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Gemini LLM API client with Google Search grounding.
pub struct GeminiClient {
    api_key: String,
    client: reqwest::Client,
}

// ─── Request structs ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    tools: Vec<Tool>,
}

#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
struct Part {
    text: String,
}

#[derive(Debug, Serialize)]
struct Tool {
    google_search: GoogleSearchTool,
}

#[derive(Debug, Serialize)]
struct GoogleSearchTool {} // empty object

// ─── Response event (one SSE chunk) ──────────────────────────────────────

/// A single SSE event from the streaming response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct StreamEvent {
    candidates: Option<Vec<StreamCandidate>>,
    usage_metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct StreamCandidate {
    content: Option<StreamContent>,
    finish_reason: Option<String>,
    grounding_metadata: Option<GroundingMetadata>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct StreamContent {
    parts: Option<Vec<StreamPart>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamPart {
    text: Option<String>,
}

// ─── Public types ─────────────────────────────────────────────────────────

/// An event yielded during streaming.
#[derive(Debug)]
pub enum GeminiStreamEvent {
    /// A text fragment to print immediately.
    Text(String),
    /// The response is complete, with optional grounding metadata.
    Done(Option<GroundingMetadata>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    pub grounding_chunks: Option<Vec<GroundingChunk>>,
    #[allow(dead_code)]
    pub grounding_supports: Option<Vec<GroundingSupport>>,
    pub web_search_queries: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GroundingChunk {
    pub web: GroundingWeb,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GroundingWeb {
    pub uri: String,
    pub title: String,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
pub struct GroundingSupport {
    pub segment: TextSegment,
    pub grounding_chunk_indices: Vec<usize>,
    pub confidence_scores: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct TextSegment {
    pub text: String,
    #[serde(default)]
    pub start_index: Option<usize>,
    #[serde(default)]
    pub end_index: Option<usize>,
}

// ─── Client implementation ─────────────────────────────────────────────────

impl GeminiClient {
    /// Create a new Gemini client.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    /// Ask a question with Google Search grounding, returning a stream of events.
    ///
    /// Text events arrive as the model generates, followed by a final
    /// `GeminiStreamEvent::Done` with any grounding metadata.
    pub async fn ask_stream(
        &self,
        query: &str,
    ) -> Result<mpsc::UnboundedReceiver<Result<GeminiStreamEvent>>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:streamGenerateContent?alt=sse&key={}",
            self.api_key
        );

        let request = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: query.to_string(),
                }],
            }],
            tools: vec![Tool {
                google_search: GoogleSearchTool {},
            }],
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Gemini API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Gemini API returned {status}: {body}");
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
    tx: mpsc::UnboundedSender<Result<GeminiStreamEvent>>,
) -> Result<()> {
    use futures_util::StreamExt;
    use tokio::io::AsyncBufReadExt;
    use tokio_util::io::StreamReader;

    let stream = response.bytes_stream();
    let reader = StreamReader::new(
        stream.map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))),
    );
    let mut lines = tokio::io::BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();

        if line.is_empty() {
            continue;
        }

        // SSE data lines start with "data: "
        let json_str = match line.strip_prefix("data: ") {
            Some(s) => s,
            None => continue,
        };

        if json_str.is_empty() {
            continue;
        }

        // Parse the JSON event
        let event: StreamEvent = match serde_json::from_str(json_str) {
            Ok(e) => e,
            Err(e) => {
                let _ = tx.send(Err(anyhow::anyhow!("Failed to parse SSE JSON: {e}")));
                return Ok(());
            }
        };

        let candidates = match event.candidates {
            Some(c) => c,
            None => continue,
        };

        for candidate in candidates {
            // Emit text parts
            if let Some(content) = candidate.content {
                if let Some(parts) = content.parts {
                    for part in parts {
                        if let Some(text) = part.text {
                            if !text.is_empty() {
                                if tx.send(Ok(GeminiStreamEvent::Text(text))).is_err() {
                                    return Ok(()); // receiver dropped
                                }
                            }
                        }
                    }
                }
            }

            // Emit done event when finished
            if candidate.finish_reason.is_some() {
                let _ = tx.send(Ok(GeminiStreamEvent::Done(candidate.grounding_metadata)));
                return Ok(());
            }
        }
    }

    // Stream ended without a finish_reason (shouldn't happen, but be safe)
    let _ = tx.send(Ok(GeminiStreamEvent::Done(None)));
    Ok(())
}
