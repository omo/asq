use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::future::join_all;
use tokio::sync::mpsc;

use crate::gemini_types::{
    Content, GenerateContentRequest, GoogleSearchTool, Part, StreamEvent as RawEvent,
    SystemInstruction, SystemPart, Tool,
};
use crate::stream::{GroundingData, Source, StreamEvent, StreamClient};

/// Gemini LLM API client with Google Search grounding.
pub struct GeminiClient {
    api_key: String,
    client: reqwest::Client,
    system_prompt: Option<String>,
}

#[async_trait]
impl StreamClient for GeminiClient {
    fn new(api_key: String, system_prompt: Option<String>) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            system_prompt,
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

        let system_instruction = self.system_prompt.as_ref().map(|text| {
            SystemInstruction {
                parts: vec![SystemPart {
                    text: text.clone(),
                }],
            }
        });

        let request = GenerateContentRequest {
            system_instruction,
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
        let reader = crate::stream::line_reader(response);
        let http_client = self.client.clone();

        tokio::spawn(async move {
            if let Err(e) = GeminiClient::read_sse_stream(reader, http_client, tx).await {
                let _ = e;
            }
        });

        Ok(rx)
    }
}

/// Convert raw Gemini grounding metadata into shared GroundingData.
pub(crate) fn convert_grounding(meta: crate::gemini_types::GroundingMetadata) -> GroundingData {
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

// ─── SSE parsing (used by ask_stream, testable with in-memory readers) ─────

impl GeminiClient {
    /// Pure SSE parser: emit Text events, return raw GroundingMetadata on finish.
    async fn parse_events<R: tokio::io::AsyncBufRead + Send + Unpin>(
        reader: R,
        tx: &mpsc::UnboundedSender<Result<StreamEvent>>,
    ) -> Result<Option<crate::gemini_types::GroundingMetadata>> {
        use crate::stream::parse_data_line;
        use tokio::io::AsyncBufReadExt;

        let mut lines = reader.lines();

        while let Some(line) = lines.next_line().await? {
            let json_str = match parse_data_line(line.trim()) {
                Some(s) => s,
                None => continue,
            };

            let event: RawEvent = match serde_json::from_str(json_str) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("Failed to parse SSE JSON: {e}")));
                    return Ok(None);
                }
            };

            let candidates = match event.candidates {
                Some(c) => c,
                None => continue,
            };

            for candidate in candidates {
                for text in Self::text_strings(&candidate) {
                    if tx.send(Ok(StreamEvent::Text(text.to_string()))).is_err() {
                        return Ok(None);
                    }
                }

                if candidate.finish_reason.is_some() {
                    return Ok(candidate.grounding_metadata);
                }
            }
        }

        Ok(None)
    }

    /// Return an iterator over non-empty text strings from a candidate.
    fn text_strings(candidate: &crate::gemini_types::StreamCandidate) -> impl Iterator<Item = &str> {
        candidate
            .content
            .as_ref()
            .and_then(|c| c.parts.as_ref())
            .into_iter()
            .flatten()
            .filter_map(|p| p.text.as_deref())
            .filter(|t| !t.is_empty())
    }

    /// Read SSE events, resolve grounding redirects, and send them through the channel.
    async fn read_sse_stream<R: tokio::io::AsyncBufRead + Send + Unpin>(
        reader: R,
        http_client: reqwest::Client,
        tx: mpsc::UnboundedSender<Result<StreamEvent>>,
    ) -> Result<()> {
        let raw_meta = Self::parse_events(reader, &tx).await?;
        let mut grounding = raw_meta.map(convert_grounding);
        if let Some(ref mut g) = grounding {
            resolve_sources(&http_client, &mut g.sources).await;
        }
        let _ = tx.send(Ok(StreamEvent::Done(grounding)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    /// Helper: feed bytes through the pure SSE parser and collect all events.
    async fn parse_bytes(data: &[u8]) -> Vec<StreamEvent> {
        let reader = tokio::io::BufReader::new(std::io::Cursor::new(data.to_vec()));
        let (tx, mut rx) = mpsc::unbounded_channel();

        let raw_meta = GeminiClient::parse_events(reader, &tx).await.unwrap();
        drop(tx); // close channel so rx.recv() returns None when drained

        let mut events: Vec<StreamEvent> = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event.unwrap());
        }

        let grounding = raw_meta.map(convert_grounding);
        events.push(StreamEvent::Done(grounding));
        events
    }

    // ── Basic text emission ─────────────────────────────────────────────────

    #[tokio::test]
    async fn text_delta_emitted() {
        let data = br#"data: {"candidates":[{"content":{"parts":[{"text":"Hello "}]}}]}
data: {"candidates":[{"content":{"parts":[{"text":"world"}]}}]}
data: {"candidates":[{"content":{},"finishReason":"STOP"}]}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "Hello "));
        assert!(matches!(&events[1], StreamEvent::Text(s) if s == "world"));
        assert!(matches!(&events[2], StreamEvent::Done(None)));
    }

    // ── finish_reason without grounding ─────────────────────────────────────

    #[tokio::test]
    async fn finish_reason_emits_done_without_grounding() {
        let data = br#"data: {"candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}]}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hello"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }

    // ── finish_reason with grounding metadata ───────────────────────────────

    #[tokio::test]
    async fn finish_reason_with_grounding() {
        // parse_bytes uses parse_events (pure) + convert_grounding, so the raw
        // URI is preserved as-is (no redirect resolution in test helpers).
        let data = br#"data: {"candidates":[{"content":{"parts":[{"text":"result"}]},"finishReason":"STOP","groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://nonexistent.test/page","title":"Ex"}}],"webSearchQueries":["test query"]}}]}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "result"));
        match &events[1] {
            StreamEvent::Done(Some(g)) => {
                assert_eq!(g.web_search_queries, &["test query"]);
                assert_eq!(g.sources.len(), 1);
                assert_eq!(g.sources[0].uri, "https://nonexistent.test/page");
            }
            _ => panic!("expected Done(Some(…))"),
        }
    }

    // ── Empty text is not emitted ───────────────────────────────────────────

    #[tokio::test]
    async fn empty_text_skipped() {
        let data = br#"data: {"candidates":[{"content":{"parts":[{"text":""}]}}]}
data: {"candidates":[{"content":{"parts":[{"text":"real"}]},"finishReason":"STOP"}]}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "real"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }

    // ── Line with no candidates is skipped ──────────────────────────────────

    #[tokio::test]
    async fn no_candidates_skipped() {
        let data = br#"data: {"usageMetadata":{"promptTokenCount":10}}
data: {"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hi"));
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
        GeminiClient::read_sse_stream(reader, reqwest::Client::new(), tx)
            .await
            .unwrap();

        let event = rx.recv().await;
        assert!(event.is_some());
        assert!(event.unwrap().is_err());
    }

    // ── Stream ends without finish_reason ───────────────────────────────────

    #[tokio::test]
    async fn stream_ends_without_finish_reason() {
        let data = br#"data: {"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}
"#;
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hello"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }
}
