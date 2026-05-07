use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::gpt_types::{
    ContentPart, GptRequest, InputMessage, OutputItem, SseEvent, Tool,
};
use crate::stream::{GroundingData, Source, StreamClient, StreamEvent};

/// GPT API client using the OpenAI Responses API with web search.
pub struct GptClient {
    api_key: String,
    client: reqwest::Client,
}

#[async_trait]
impl StreamClient for GptClient {
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
        let url = "https://api.openai.com/v1/responses";

        let request = GptRequest {
            model: "gpt-5.4-mini".to_string(),
            max_output_tokens: Some(4096),
            stream: true,
            input: vec![InputMessage {
                role: "user".to_string(),
                content: query.to_string(),
            }],
            tools: Some(vec![Tool {
                tool_type: "web_search_preview".to_string(),
            }]),
        };

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to GPT API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("GPT API returned {status}: {body}");
        }

        let (tx, rx) = mpsc::unbounded_channel();

        // Spawn a task to read the SSE stream
        let reader = crate::stream::line_reader(response);
        tokio::spawn(async move {
            if let Err(e) = GptClient::read_sse_stream(reader, tx).await {
                let _ = e;
            }
        });

        Ok(rx)
    }
}

// ─── SSE parsing ───────────────────────────────────────────────────────────

impl GptClient {
    /// Read SSE events from the response body and send them through the channel.
    ///
    /// Responses API streaming format:
    ///   event: response.output_text.delta
    ///   data: {"type":"response.output_text.delta","delta":"text chunk",...}
    ///
    ///   event: response.completed
    ///   data: {"type":"response.completed","response":{...full response with web search queries and annotations...}}
    async fn read_sse_stream<R: tokio::io::AsyncBufRead + Send + Unpin>(
        reader: R,
        tx: mpsc::UnboundedSender<Result<StreamEvent>>,
    ) -> Result<()> {
        use crate::stream::parse_data_line;
        use tokio::io::AsyncBufReadExt;

        let mut lines = reader.lines();
        // Collect search queries and sources from the final response
        let mut search_queries: Vec<String> = Vec::new();
        let mut sources: Vec<Source> = Vec::new();

        while let Some(line) = lines.next_line().await? {
            let line = line.trim().to_string();

            if line.is_empty() {
                continue;
            }

            let json_str = match parse_data_line(&line) {
                Some(s) => s,
                None => continue,
            };

            let event: SseEvent = match serde_json::from_str(json_str) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!(
                        "Failed to parse GPT SSE JSON: {e}"
                    )));
                    return Ok(());
                }
            };

            match event.event_type.as_str() {
                "response.output_text.delta" => {
                    if let Some(text) = event.delta {
                        if !text.is_empty() {
                            if tx.send(Ok(StreamEvent::Text(text))).is_err() {
                                return Ok(()); // receiver dropped
                            }
                        }
                    }
                }
                "response.output_item.done" => {
                    // Extract search queries from web_search_call items
                    if let Some(item) = event.item {
                        if let OutputItem::WebSearchCall(w) = item {
                            if let Some(action) = w.action {
                                // Collect individual queries
                                if let Some(q) = action.query {
                                    if !q.is_empty() && !search_queries.contains(&q) {
                                        search_queries.push(q);
                                    }
                                }
                                // Also collect from queries array
                                if let Some(queries) = action.queries {
                                    for q in queries {
                                        if !q.is_empty() && !search_queries.contains(&q) {
                                            search_queries.push(q);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                "response.completed" => {
                    // Extract sources from annotations in the completed response
                    if let Some(resp) = event.response {
                        for item in resp.output {
                            if let OutputItem::Message(msg) = item {
                                for content in msg.content {
                                    let ContentPart::OutputText(text) = content;
                                    for annotation in text.annotations {
                                        if let Some(url) = annotation.url {
                                            if !url.is_empty() {
                                                sources.push(Source { uri: url });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let grounding = if search_queries.is_empty() && sources.is_empty() {
                        None
                    } else {
                        Some(GroundingData {
                            web_search_queries: search_queries,
                            sources,
                        })
                    };
                    let _ = tx.send(Ok(StreamEvent::Done(grounding)));
                    return Ok(());
                }
                _ => {
                    // Ignore: response.created, response.in_progress,
                    // response.web_search_call.*, response.content_part.*,
                    // response.output_item.added
                }
            }
        }

        // Stream ended without response.completed
        let _ = tx.send(Ok(StreamEvent::Done(None)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    /// Helper: feed bytes through the GPT SSE parser and collect all events.
    async fn parse_bytes(data: &[u8]) -> Vec<StreamEvent> {
        let reader = tokio::io::BufReader::new(std::io::Cursor::new(data.to_vec()));
        let (tx, mut rx) = mpsc::unbounded_channel();
        GptClient::read_sse_stream(reader, tx).await.unwrap();

        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event.unwrap());
        }
        events
    }

    // ── Basic text emission ─────────────────────────────────────────────────

    #[tokio::test]
    async fn text_delta_emitted() {
        let data = b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello \",\"content_index\":0,\"item_id\":\"msg_1\",\"output_index\":1}\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"world\",\"content_index\":0,\"item_id\":\"msg_1\",\"output_index\":1}\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n";
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "Hello "));
        assert!(matches!(&events[1], StreamEvent::Text(s) if s == "world"));
        assert!(matches!(&events[2], StreamEvent::Done(None)));
    }

    // ── response.completed with web search queries and annotations ─────────

    #[tokio::test]
    async fn completed_with_web_search_and_annotations() {
        let data = b"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"weather Tokyo\",\"queries\":[\"weather Tokyo\"]}},\"output_index\":0}\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"It is 75F.\",\"content_index\":0,\"item_id\":\"msg_1\",\"output_index\":1}\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"queries\":[\"weather Tokyo\"],\"query\":\"weather Tokyo\"}},{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"It is 75F.\",\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://weather.example.com\",\"title\":\"Tokyo Weather\",\"start_index\":0,\"end_index\":10}]}]}]}}\n";
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "It is 75F."));
        match &events[1] {
            StreamEvent::Done(Some(g)) => {
                assert_eq!(g.web_search_queries, &["weather Tokyo"]);
                assert_eq!(g.sources.len(), 1);
                assert_eq!(g.sources[0].uri, "https://weather.example.com");
            }
            _ => panic!("expected Done(Some(...))"),
        }
    }

    // ── Web search without annotations ─────────────────────────────────────

    #[tokio::test]
    async fn web_search_without_annotations() {
        let data = b"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"test\",\"queries\":[\"test\"]}},\"output_index\":0}\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"queries\":[\"test\"],\"query\":\"test\"}},{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"result\",\"annotations\":[]}]}]}}\n";
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::Done(Some(g)) => {
                assert_eq!(g.web_search_queries, &["test"]);
                assert!(g.sources.is_empty());
            }
            _ => panic!("expected Done(Some(...))"),
        }
    }

    // ── No text, just empty result ─────────────────────────────────────────

    #[tokio::test]
    async fn no_text_only_completed() {
        let data = b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n";
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::Done(None)));
    }

    // ── Malformed JSON sends an error ──────────────────────────────────────

    #[tokio::test]
    async fn malformed_json_sends_error() {
        let reader = tokio::io::BufReader::new(std::io::Cursor::new(
            b"data: {not json}\n".to_vec(),
        ));
        let (tx, mut rx) = mpsc::unbounded_channel();
        GptClient::read_sse_stream(reader, tx).await.unwrap();

        let event = rx.recv().await;
        assert!(event.is_some());
        assert!(event.unwrap().is_err());
    }

    // ── Empty lines are skipped ─────────────────────────────────────────────

    #[tokio::test]
    async fn empty_lines_skipped() {
        let data = b"\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n";
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hi"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }

    // ── Ignores other events ───────────────────────────────────────────────

    #[tokio::test]
    async fn other_events_ignored() {
        let data = b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}\nevent: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}\nevent: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"status\":\"in_progress\",\"content\":[]},\"output_index\":0}\nevent: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]},\"content_index\":0,\"item_id\":\"msg_1\",\"output_index\":0}\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n";
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hello"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }

    // ── Stream ends without response.completed ─────────────────────────────

    #[tokio::test]
    async fn stream_ends_without_completed() {
        let data = b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n";
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Text(s) if s == "hello"));
        assert!(matches!(&events[1], StreamEvent::Done(None)));
    }

    // ── Multiple search queries de-duplicated ──────────────────────────────

    #[tokio::test]
    async fn de_duplicates_search_queries() {
        let data = b"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"weather Tokyo\",\"queries\":[\"weather Tokyo\",\"weather Tokyo\"]}},\"output_index\":0}\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"queries\":[\"weather Tokyo\"],\"query\":\"weather Tokyo\"}}]}}\n";
        let events = parse_bytes(data).await;
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::Done(Some(g)) => {
                assert_eq!(g.web_search_queries, &["weather Tokyo"]);
            }
            _ => panic!("expected Done(Some(...))"),
        }
    }
}
