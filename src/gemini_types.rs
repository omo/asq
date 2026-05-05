use serde::{Deserialize, Serialize};

// ─── Request structs ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(crate) struct GenerateContentRequest {
    pub(crate) contents: Vec<Content>,
    pub(crate) tools: Vec<Tool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Content {
    pub(crate) parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Part {
    pub(crate) text: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct Tool {
    pub(crate) google_search: GoogleSearchTool,
}

#[derive(Debug, Serialize)]
pub(crate) struct GoogleSearchTool {} // empty object

// ─── Response event (one SSE chunk) ──────────────────────────────────────

/// A single SSE event from the streaming response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StreamEvent {
    pub(crate) candidates: Option<Vec<StreamCandidate>>,
    pub(crate) usage_metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StreamCandidate {
    pub(crate) content: Option<StreamContent>,
    pub(crate) finish_reason: Option<String>,
    pub(crate) grounding_metadata: Option<GroundingMetadata>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StreamContent {
    pub(crate) parts: Option<Vec<StreamPart>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct StreamPart {
    pub(crate) text: Option<String>,
}

// ─── Raw Gemini API types ─────────────────────────────────────────────────

/// Raw grounding metadata from the Gemini API.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GroundingMetadata {
    pub(crate) grounding_chunks: Option<Vec<GroundingChunk>>,
    #[allow(dead_code)]
    pub(crate) grounding_supports: Option<Vec<GroundingSupport>>,
    pub(crate) web_search_queries: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GroundingChunk {
    pub(crate) web: GroundingWeb,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GroundingWeb {
    pub(crate) uri: String,
    #[allow(dead_code)]
    pub(crate) title: String,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GroundingSupport {
    pub(crate) segment: TextSegment,
    pub(crate) grounding_chunk_indices: Vec<usize>,
    pub(crate) confidence_scores: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) struct TextSegment {
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) start_index: Option<usize>,
    #[serde(default)]
    pub(crate) end_index: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_content_request_serializes() {
        let req = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: "hello".into(),
                }],
            }],
            tools: vec![Tool {
                google_search: GoogleSearchTool {},
            }],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""text":"hello""#));
        assert!(json.contains(r#""google_search":{}"#));
    }

    #[test]
    fn stream_event_deserializes() {
        let json = r#"{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10}}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        let candidates = event.candidates.unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].finish_reason.as_deref(), Some("STOP"));
        let parts = candidates[0].content.as_ref().unwrap().parts.as_ref().unwrap();
        assert_eq!(parts[0].text.as_deref(), Some("hi"));
    }

    #[test]
    fn grounding_metadata_deserializes() {
        let json = r#"{"groundingChunks":[{"web":{"uri":"https://x.com","title":"X"}}],"webSearchQueries":["test"]}"#;
        let meta: GroundingMetadata = serde_json::from_str(json).unwrap();
        let chunks = meta.grounding_chunks.unwrap();
        assert_eq!(chunks[0].web.uri, "https://x.com");
        assert_eq!(chunks[0].web.title, "X");
        assert_eq!(meta.web_search_queries.unwrap(), vec!["test"]);
    }

    #[test]
    fn convert_grounding_tests() {
        let meta = crate::gemini_types::GroundingMetadata {
            grounding_chunks: Some(vec![
                crate::gemini_types::GroundingChunk {
                    web: crate::gemini_types::GroundingWeb {
                        uri: "https://a.com".into(),
                        title: "A".into(),
                    },
                },
            ]),
            grounding_supports: None,
            web_search_queries: Some(vec!["q1".into()]),
        };
        // Note: convert_grounding is a free fn in the parent module.
        // We import it from super to test the conversion logic.
        let g = crate::gemini::convert_grounding(meta);
        assert_eq!(g.web_search_queries, &["q1"]);
        assert_eq!(g.sources.len(), 1);
        assert_eq!(g.sources[0].uri, "https://a.com");
    }
}
