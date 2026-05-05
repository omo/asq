use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::future::join_all;
use tokio::sync::mpsc;

use crate::gemini_types::{
    Content, GenerateContentRequest, GoogleSearchTool, Part, StreamEvent as RawEvent, Tool,
};
use crate::stream::{GroundingData, Source, StreamEvent, StreamClient};

/// Gemini LLM API client with Google Search grounding.
pub struct GeminiClient {
    api_key: String,
    client: reqwest::Client,
}

#[async_trait]
impl StreamClient for GeminiClient {
    fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    async fn ask_stream(
        &self,
        query: &str,
    ) -> Result<mpsc::UnboundedReceiver<Result<StreamEvent>>> {
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
        let http_client = self.client.clone();

        tokio::spawn(async move {
            if let Err(e) = read_sse_stream(response, http_client, tx).await {
                let _ = e;
            }
        });

        Ok(rx)
    }
}

/// Convert raw Gemini grounding metadata into shared GroundingData.
fn convert_grounding(meta: crate::gemini_types::GroundingMetadata) -> GroundingData {
    GroundingData {
        web_search_queries: meta.web_search_queries.unwrap_or_default(),
        sources: meta
            .grounding_chunks
            .unwrap_or_default()
            .into_iter()
            .map(|chunk| Source {
                uri: chunk.web.uri,
            })
            .collect(),
    }
}

/// Follow Google redirect URLs for a batch of sources, updating them in place.
async fn resolve_sources(client: &reqwest::Client, sources: &mut [Source]) {
    let futures: Vec<_> = sources
        .iter()
        .map(|s| {
            let uri = s.uri.clone();
            async move {
                match client.head(&uri).send().await {
                    Ok(resp) => resp.url().as_str().to_string(),
                    Err(_) => uri,
                }
            }
        })
        .collect();
    let urls = join_all(futures).await;
    for (source, url) in sources.iter_mut().zip(urls.iter()) {
        source.uri.clone_from(url);
    }
}

/// Read SSE events, resolve grounding redirects, and send them through the channel.
async fn read_sse_stream(
    response: reqwest::Response,
    http_client: reqwest::Client,
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

        let json_str = match parse_data_line(&line) {
            Some(s) => s,
            None => continue,
        };

        let event: RawEvent = match serde_json::from_str(json_str) {
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
            if let Some(content) = candidate.content {
                if let Some(parts) = content.parts {
                    for part in parts {
                        if let Some(text) = part.text {
                            if !text.is_empty() {
                                if tx.send(Ok(StreamEvent::Text(text))).is_err() {
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
            }

            if candidate.finish_reason.is_some() {
                let mut grounding = candidate.grounding_metadata.map(convert_grounding);
                if let Some(ref mut g) = grounding {
                    resolve_sources(&http_client, &mut g.sources).await;
                }
                let _ = tx.send(Ok(StreamEvent::Done(grounding)));
                return Ok(());
            }
        }
    }

    let _ = tx.send(Ok(StreamEvent::Done(None)));
    Ok(())
}
