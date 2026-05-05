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
